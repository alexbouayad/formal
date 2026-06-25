from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from formal.artifacts import DatasetReference, ModelReference
from formal.config import FormalizationConfig, StageConfig
from formal.utils import hydra


class DType(StrEnum):
    BF16 = "bfloat16"
    FP16 = "float16"
    FP32 = "float32"


@dataclass(kw_only=True)
class ModelArguments:
    id: str
    dtype: DType = DType.BF16
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True


@dataclass(kw_only=True)
class RWSConfig:
    num_particles: int = 8
    particle_chunk_size: int = 8


@dataclass(kw_only=True)
class TrainerArguments:
    rws: RWSConfig | None = field(default_factory=RWSConfig)
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 10


@dataclass(kw_only=True)
class TrainingConfig(StageConfig):
    model: ModelArguments
    trainer: TrainerArguments
    formalization: FormalizationConfig

    peft: dict[str, Any] = hydra.instantiable_field("peft.PeftConfig")
    optimizer: dict[str, Any] = hydra.instantiable_field("torch.optim.Optimizer")

    jaxtyping: bool = False

    input: DatasetReference
    output: ModelReference
