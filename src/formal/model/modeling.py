from typing import cast

import torch
from jaxtyping import Bool, Float, UInt
from peft import PeftModel
from torch import Tensor, nn

from formal.config import FormalizationConfig
from formal.tokenization.vocab import count_formal_tokens

from .pretrained import PreTrainedFormalModel

# TODO: Add comments to track float precisions


class FormalModel(PreTrainedFormalModel):
    def __init__(self, text_model: PeftModel, formalization_config: FormalizationConfig) -> None:
        hidden_size = cast(int, text_model.config.hidden_size)  # type: ignore
        formal_vocab_size = count_formal_tokens(formalization_config)

        formal_mask_embedding = nn.Linear(
            in_features=hidden_size,
            out_features=1,
            bias=False,
            dtype=torch.float32,
            device=text_model.device,
        )

        formal_input_embedding = nn.Embedding(
            num_embeddings=formal_vocab_size,
            embedding_dim=hidden_size,
            dtype=torch.float32,
            device=text_model.device,
        )

        formal_output_embedding = nn.Linear(
            in_features=hidden_size,
            out_features=formal_vocab_size,
            bias=False,
            dtype=torch.float32,
            device=text_model.device,
        )

        super().__init__(text_model, formalization_config)

        self.formal_mask_embedding = formal_mask_embedding
        self.formal_input_embedding = formal_input_embedding
        self.formal_output_embedding = formal_output_embedding

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        text_input_weight = self.text_input_embedding.weight.float()
        text_output_weight = self.text_output_embedding.weight.float()

        input_std = text_input_weight.std().item()
        input_mean = text_input_weight.mean(dim=0)

        output_std = text_output_weight.std().item()
        output_mean = text_output_weight.mean(dim=0)

        nn.init.normal_(self.formal_input_embedding.weight, std=input_std)
        nn.init.normal_(self.formal_output_embedding.weight, std=output_std)

        self.formal_input_embedding.weight.add_(input_mean)
        self.formal_output_embedding.weight.add_(output_mean)

        nn.init.zeros_(self.formal_mask_embedding.weight)

    def forward(
        self,
        input_ids: UInt[Tensor, "B S"],
        text_mask: Bool[Tensor, "B S"],
        formal_mask: Bool[Tensor, "B S"],
    ) -> Float[Tensor, "B S V"]:
        hidden_state = self._forward(input_ids, text_mask, formal_mask)  # (B, S, H)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            text_logits = self.text_output_embedding(hidden_state)  # (B, S, V_text)
            formal_logits = self.formal_output_embedding(hidden_state)  # (B, S, V_formal)

        logits = torch.cat((text_logits, formal_logits), dim=-1)  # (B, S, V)

        return logits

    def forward_submask(
        self,
        input_ids: UInt[Tensor, "B S"],
        text_mask: Bool[Tensor, "B S"],
        formal_mask: Bool[Tensor, "B S"],
    ) -> Float[Tensor, "B S"]:
        hidden_state = self._forward(input_ids, text_mask, formal_mask)  # (B, S, H)
        formal_hidden_state = hidden_state[formal_mask]  # (N_formal, H)

        formal_submask_probs = torch.full_like(input_ids, fill_value=1.0, dtype=torch.float32)  # (B, S)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            formal_submask_logits = self.formal_mask_embedding(formal_hidden_state)  # (N_formal, 1)
            formal_submask_logits = cast(Tensor, formal_submask_logits).squeeze(-1)  # (N_formal,)

        formal_submask_probs[formal_mask] = formal_submask_logits.float().sigmoid()

        return formal_submask_probs

    def _forward(
        self,
        input_ids: UInt[Tensor, "B S"],
        text_mask: Bool[Tensor, "B S"],
        formal_mask: Bool[Tensor, "B S"],
    ) -> Float[Tensor, "B S H"]:
        base_model = cast(nn.Module, self.text_model.base_model.base_model)
        hidden_size = cast(int, self.text_model.config.hidden_size)  # type: ignore
        text_vocab_size = cast(int, self.text_model.config.vocab_size)  # type: ignore

        text_input_ids = input_ids[text_mask]  # (N_text,)
        formal_input_ids = input_ids[formal_mask] - text_vocab_size  # (N_formal,)

        text_inputs_embeds = self.text_input_embedding(text_input_ids).to(self.dtype)  # (N_text, H)
        formal_inputs_embeds = self.formal_input_embedding(formal_input_ids).to(self.dtype)  # (N_formal, H)

        # (B, S, H)
        inputs_embeds = torch.zeros(*input_ids.shape, hidden_size, device=self.device, dtype=self.dtype)
        inputs_embeds[text_mask] = text_inputs_embeds
        inputs_embeds[formal_mask] = formal_inputs_embeds

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            base_outputs = base_model(inputs_embeds=inputs_embeds)

        return base_outputs.last_hidden_state  # (B, S, H)
