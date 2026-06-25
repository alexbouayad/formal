from contextlib import nullcontext
from logging import getLogger
from typing import Any, cast

import jaxtyping
import torch
from hydra.utils import instantiate
from peft import PeftConfig, PeftModel, get_peft_model
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, default_collate  # type: ignore
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM

from formal import artifacts
from formal.utils.wandb import log_input_on_wandb, log_output_on_wandb, with_wandb

from .config import DType, TrainingConfig

logger = getLogger(__name__)


def _collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    batch = default_collate(features)
    batch = cast(dict[str, Any], batch)

    token_type_ids = batch.pop("token_type_ids")

    text_mask = token_type_ids == 0
    formal_mask = token_type_ids == 1
    num_text_tokens = text_mask.sum().item()

    batch["text_mask"] = text_mask
    batch["formal_mask"] = formal_mask
    batch["num_text_tokens"] = num_text_tokens

    return batch


@with_wandb
def train(training_config: TrainingConfig) -> None:
    with (
        jaxtyping.install_import_hook(["formal.model", ".trainer"], typechecker="beartype.beartype")
        if training_config.jaxtyping
        else nullcontext()
    ):
        from formal.model import FormalModel

        from .trainer import RWSTrainer, SFTTrainer

    device = torch.device(training_config.model.device)

    match training_config.model.dtype:
        case DType.BF16:
            dtype = torch.bfloat16
        case DType.FP16:
            dtype = torch.float16
        case DType.FP32:
            dtype = torch.float32

    dataset = artifacts.datasets.load(
        reference=training_config.input,
        split="train",
        runtime_config=training_config.runtime,
    )

    log_input_on_wandb(training_config.input, runtime_config=training_config.runtime)

    dataset = dataset.rename_column("token_ids", "input_ids")
    dataset = dataset.select_columns(["input_ids", "token_type_ids"])

    dataset.set_format("torch")  # type: ignore
    dataset = cast(TorchDataset[Tensor], dataset)

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        num_workers=1,
        pin_memory=device.type == "cuda",
        collate_fn=_collate_fn,
    )

    base_model = AutoModelForCausalLM.from_pretrained(  # type: ignore
        training_config.model.id,
        dtype=dtype,
        device_map=device,
        attn_implementation=training_config.model.attn_implementation,
        use_cache=False,
    )

    if training_config.model.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()  # type: ignore

    peft_config = instantiate(training_config.peft)
    peft_config = cast(PeftConfig, peft_config)

    peft_model = get_peft_model(base_model, peft_config=peft_config)
    peft_model = cast(PeftModel, peft_model)

    model = FormalModel(peft_model, training_config.formalization)

    if training_config.wandb and (watch_config := training_config.wandb.model_watch):
        import wandb

        wandb.watch(  # type: ignore
            model,
            log=watch_config.log.value if watch_config.log else None,
            log_freq=watch_config.log_freq,
            log_graph=watch_config.log_graph,
        )

    optimizer = instantiate(
        training_config.optimizer,
        params=[parameter for parameter in model.parameters() if parameter.requires_grad],
    )

    optimizer = cast(Optimizer, optimizer)

    if training_config.trainer.rws:
        trainer = RWSTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            rws_config=training_config.trainer.rws,
            max_grad_norm=training_config.trainer.max_grad_norm,
            gradient_accumulation_steps=training_config.trainer.gradient_accumulation_steps,
        )

    else:
        trainer = SFTTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            max_grad_norm=training_config.trainer.max_grad_norm,
            gradient_accumulation_steps=training_config.trainer.gradient_accumulation_steps,
        )

    try:
        trainer.train()

    except KeyboardInterrupt:
        logger.info("Training interrupted, breaking out of the training loop...")

    if training_config.wandb:
        import wandb

        wandb.unwatch(model)

    artifacts.models.save(
        model=model,
        reference=training_config.output,
        runtime_config=training_config.runtime,
    )

    log_output_on_wandb(training_config.output, runtime_config=training_config.runtime)
