import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast, override

import torch
import wandb
from jaxtyping import Bool, Float, Shaped, UInt
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RWSConfig

if TYPE_CHECKING:
    from formal.model import FormalModel


# TODO: Add comments to track float precisions


def _mask_and_pad(
    sequences: Shaped[Tensor, "B S"],
    masks_iterable: Iterable[Bool[Tensor, "B S"]],
    *,
    padding_value: float = 0,
) -> Shaped[Tensor, "C*B S_pad"]:
    return pad_sequence(
        [sequence[mask] for masks in masks_iterable for sequence, mask in zip(sequences, masks)],
        batch_first=True,
        padding_value=padding_value,
    )


def _cross_entropy(
    logits: Float[Tensor, "B S V"],
    input_ids: UInt[Tensor, "B S"],
    *,
    ignore_index: int = -100,
) -> Float[Tensor, "B"]:
    token_cross_entropies = cross_entropy(
        input=logits.permute(1, 2, 0)[:-1],  # (S-1, V, B)
        target=input_ids.transpose(0, 1)[1:],  # (S-1, B)
        ignore_index=ignore_index,
        reduction="none",
    )  # (S-1, B)

    sequence_cross_entropies = token_cross_entropies.sum(dim=0)  # (B,)

    return sequence_cross_entropies


class _TrainerBase(ABC):
    model: "FormalModel"
    optimizer: Optimizer
    dataloader: DataLoader[Tensor]

    max_grad_norm: float
    gradient_accumulation_steps: int

    def __init__(
        self,
        model: "FormalModel",
        optimizer: Optimizer,
        dataloader: DataLoader[Tensor],
        *,
        max_grad_norm: float,
        gradient_accumulation_steps: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader

        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

    @abstractmethod
    def _get_empty_train_losses(self) -> dict[str, list[Float[Tensor, ""]]]: ...

    @abstractmethod
    def _step(
        self,
        *,
        input_ids: UInt[Tensor, "B S"],
        text_mask: Bool[Tensor, "B S"],
        formal_mask: Bool[Tensor, "B S"],
        num_text_tokens_in_batch: int,
    ) -> dict[str, Float[Tensor, ""]]: ...

    def train(self) -> None:
        self.model.train()
        self.optimizer.zero_grad()

        num_batches = math.ceil(len(self.dataloader) / self.gradient_accumulation_steps)
        batch_iterator = itertools.batched(self.dataloader, self.gradient_accumulation_steps)
        batch_iterator = tqdm(enumerate(batch_iterator, start=1), total=num_batches, desc="Training")

        train_losses = self._get_empty_train_losses()

        for step_idx, micro_batches in batch_iterator:
            micro_batches = cast(list[dict[str, Any]], micro_batches)

            num_text_tokens_in_batch = sum(cast(int, inputs.pop("num_text_tokens")) for inputs in micro_batches)

            for inputs in micro_batches:
                # UInt[Tensor, "B S"],  Bool[Tensor, "B S"],  Bool[Tensor, "B S"]
                input_ids = cast(Tensor, inputs.pop("input_ids")).to(self.model.device, non_blocking=True)
                text_mask = cast(Tensor, inputs.pop("text_mask")).to(self.model.device, non_blocking=True)
                formal_mask = cast(Tensor, inputs.pop("formal_mask")).to(self.model.device, non_blocking=True)

                train_loss = self._step(
                    input_ids=input_ids,
                    text_mask=text_mask,
                    formal_mask=formal_mask,
                    num_text_tokens_in_batch=num_text_tokens_in_batch,
                )

                for key, lst in train_losses.items():
                    lst.append(train_loss[key])

            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            grad_norm = grad_norm.item()

            self.optimizer.step()  # type: ignore
            self.optimizer.zero_grad()

            logging_data = {
                **{f"train/{key}_loss": torch.stack(lst).sum().item() for key, lst in train_losses.items()},
                "train/grad_norm": grad_norm,
            }

            wandb.log(logging_data, step=step_idx)
            batch_iterator.set_postfix(logging_data)  # type: ignore

            for lst in train_losses.values():
                lst.clear()

    def _compute_generative_loss(
        self,
        *,
        input_ids: UInt[Tensor, "B S_pad"],
        text_mask: Bool[Tensor, "B S_pad"],
        formal_mask: Bool[Tensor, "B S_pad"],
        num_text_tokens_in_batch: int,
    ) -> Float[Tensor, ""]:
        eos_token_id = cast(int, self.model.text_model.config.eos_token_id)  # type: ignore

        generative_logits = self.model.forward(input_ids, text_mask, formal_mask)  # (B, S_pad, V)

        with torch.autocast(device_type=self.model.device.type, dtype=self.model.dtype):
            generative_loss = _cross_entropy(generative_logits, input_ids, ignore_index=eos_token_id)  # (B,)
            generative_loss = generative_loss.sum() / num_text_tokens_in_batch  # scalar

        return generative_loss


class RWSTrainer(_TrainerBase):
    rws_config: RWSConfig

    def __init__(
        self,
        model: "FormalModel",
        optimizer: Optimizer,
        dataloader: DataLoader[Tensor],
        *,
        rws_config: RWSConfig,
        max_grad_norm: float,
        gradient_accumulation_steps: int,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        self.rws_config = rws_config

    @override
    def _get_empty_train_losses(self) -> dict[str, list[Float[Tensor, ""]]]:
        return {"rws": [], "iwae": []}

    @override
    def _step(
        self,
        *,
        input_ids: UInt[Tensor, "B S"],
        text_mask: Bool[Tensor, "B S"],
        formal_mask: Bool[Tensor, "B S"],
        num_text_tokens_in_batch: int,
    ) -> dict[str, Float[Tensor, ""]]:
        eos_token_id = cast(int, self.model.text_model.config.eos_token_id)  # type: ignore

        # 1. Proposal phase
        submask_probs = self.model.forward_submask(input_ids, text_mask, formal_mask)  # (B, S)

        # bool (K, B, S), float (K, B, S)
        particle_masks, proposal_log_probs = self._sample_particles(submask_probs)

        proposal_log_probs[:, text_mask] = 0.0
        proposal_log_probs = proposal_log_probs.sum(dim=-1)  # (K, B)

        self.model.eval()

        with torch.no_grad():
            # 2. Model evaluation on particles
            generative_log_probs = self._compute_generative_log_probs(
                input_ids=input_ids,
                text_mask=text_mask,
                formal_mask=formal_mask,
                particle_masks=particle_masks,
            )

            # 3. IW-ELBO
            iw_elbo, normalized_weights = self._compute_iw_elbo(
                proposal_log_probs=proposal_log_probs,
                generative_log_probs=generative_log_probs,
                num_text_tokens_in_batch=num_text_tokens_in_batch,
            )

        self.model.train()

        # 4. Proposal update (wake-phi)
        proposal_loss = self._compute_proposal_loss(
            proposal_log_probs=proposal_log_probs,
            normalized_weights=normalized_weights,
            num_text_tokens_in_batch=num_text_tokens_in_batch,
        )

        # 5. Importance resampling
        particle_masks = self._subsample_particles(particle_masks, normalized_weights)

        particle_input_ids = _mask_and_pad(input_ids, (particle_masks,), padding_value=eos_token_id)
        particle_text_mask = _mask_and_pad(text_mask, (particle_masks,), padding_value=False)
        particle_formal_mask = _mask_and_pad(formal_mask, (particle_masks,), padding_value=False)

        # 6. Model update (wake-theta)
        generative_loss = self._compute_generative_loss(
            input_ids=particle_input_ids,
            text_mask=particle_text_mask,
            formal_mask=particle_formal_mask,
            num_text_tokens_in_batch=num_text_tokens_in_batch,
        )

        rws_loss = proposal_loss + generative_loss
        iwae_loss = -iw_elbo

        rws_loss.backward()  # type: ignore
        rws_loss = rws_loss.detach()

        return {"rws": rws_loss, "iwae": iwae_loss}

    def _sample_particles(
        self,
        submask_probs: Float[Tensor, "B S"],
    ) -> tuple[Bool[Tensor, "K B S"], Float[Tensor, "K B S"]]:
        submask_distribution = Bernoulli(probs=submask_probs)

        particle_masks = submask_distribution.sample((self.rws_config.num_particles,))  # float (K, B, S)
        proposal_log_probs = submask_distribution.log_prob(particle_masks)  # (K, B, S)

        return particle_masks.bool(), proposal_log_probs

    def _compute_generative_log_probs(
        self,
        *,
        input_ids: UInt[Tensor, "B S"],
        text_mask: Bool[Tensor, "B S"],
        formal_mask: Bool[Tensor, "B S"],
        particle_masks: Bool[Tensor, "K B S"],
    ) -> Float[Tensor, "K B"]:
        batch_size = input_ids.shape[0]
        eos_token_id = cast(int, self.model.text_model.config.eos_token_id)  # type: ignore

        generative_log_probs_list: list[Tensor] = []

        for chunk_masks in itertools.batched(particle_masks, self.rws_config.particle_chunk_size):
            particle_input_ids = _mask_and_pad(input_ids, chunk_masks, padding_value=eos_token_id)  # (C*B, S_pad)
            particle_text_mask = _mask_and_pad(text_mask, chunk_masks, padding_value=False)  # (C*B, S_pad)
            particle_formal_mask = _mask_and_pad(formal_mask, chunk_masks, padding_value=False)  # (C*B, S_pad)

            # (C*B, S_pad, V)
            particle_logits = self.model.forward(particle_input_ids, particle_text_mask, particle_formal_mask)

            with torch.autocast(device_type=self.model.device.type, dtype=self.model.dtype):
                # (C*B,)
                sequence_cross_entropies = _cross_entropy(
                    logits=particle_logits,
                    input_ids=particle_input_ids,
                    ignore_index=eos_token_id,
                )

                # (C, B)
                sequence_cross_entropies = sequence_cross_entropies.view(len(chunk_masks), batch_size)

            generative_log_probs_list.append(-sequence_cross_entropies)

        generative_log_probs = torch.cat(generative_log_probs_list)  # (K, B)

        return generative_log_probs

    def _compute_iw_elbo(
        self,
        *,
        proposal_log_probs: Float[Tensor, "K B"],
        generative_log_probs: Float[Tensor, "K B"],
        num_text_tokens_in_batch: int,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, "K B"]]:
        log_weights = generative_log_probs - proposal_log_probs  # (K, B)
        normalized_weights = torch.softmax(log_weights, dim=0)  # (K, B)

        iw_elbo = log_weights.logsumexp(dim=0) - math.log(self.rws_config.num_particles)  # (B,)
        iw_elbo = iw_elbo.sum() / num_text_tokens_in_batch  # scalar

        return iw_elbo, normalized_weights

    def _compute_proposal_loss(
        self,
        *,
        proposal_log_probs: Float[Tensor, "K B"],
        normalized_weights: Float[Tensor, "K B"],
        num_text_tokens_in_batch: int,
    ) -> Float[Tensor, ""]:
        proposal_loss = -(normalized_weights * proposal_log_probs).sum(dim=0)  # (B,)
        proposal_loss = proposal_loss.sum() / num_text_tokens_in_batch  # scalar

        return proposal_loss

    def _subsample_particles(
        self,
        particle_masks: Bool[Tensor, "K B S"],
        normalized_weights: Float[Tensor, "K B"],
    ) -> Bool[Tensor, "B S"]:
        batch_size = particle_masks.shape[1]
        normalized_weights = normalized_weights.transpose(0, 1)  # (B, K)
        subsample = Categorical(probs=normalized_weights).sample()  # (B,)

        batch_indices = torch.arange(batch_size, device=self.model.device)  # (B,)
        subsampled_particle_masks = particle_masks[subsample, batch_indices]  # (B, S)

        return subsampled_particle_masks


class SFTTrainer(_TrainerBase):
    def __init__(
        self,
        model: "FormalModel",
        optimizer: Optimizer,
        dataloader: DataLoader[Tensor],
        *,
        max_grad_norm: float,
        gradient_accumulation_steps: int,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    @override
    def _get_empty_train_losses(self) -> dict[str, list[Float[Tensor, ""]]]:
        return {"sft": []}

    @override
    def _step(
        self,
        *,
        input_ids: UInt[Tensor, "B S"],
        text_mask: Bool[Tensor, "B S"],
        formal_mask: Bool[Tensor, "B S"],
        num_text_tokens_in_batch: int,
    ) -> dict[str, Float[Tensor, ""]]:
        generative_loss = self._compute_generative_loss(
            input_ids=input_ids,
            text_mask=text_mask,
            formal_mask=formal_mask,
            num_text_tokens_in_batch=num_text_tokens_in_batch,
        )

        generative_loss.backward()  # type: ignore
        generative_loss = generative_loss.detach()

        return {"sft": generative_loss}
