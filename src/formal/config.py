from dataclasses import dataclass, field
from enum import StrEnum

from formal.language import Language


class FormalTokenPlacement(StrEnum):
    PREFIX = "prefix"
    POSTFIX = "postfix"
    ENCLOSING = "enclosing"
    OMIT = "omit"


@dataclass(kw_only=True)
class FormalizationConfig:
    language: Language = field(kw_only=False)
    include_fields: bool = False
    token_placement: FormalTokenPlacement = FormalTokenPlacement.PREFIX


class WandBWatchTarget(StrEnum):
    ALL = "all"
    GRADIENTS = "gradients"
    PARAMETERS = "parameters"


@dataclass(kw_only=True)
class WandBWatchConfig:
    log: WandBWatchTarget | None = None
    log_freq: int = 1000
    log_graph: bool = True


@dataclass(kw_only=True)
class WandBConfig:
    project: str | None = None
    run_name: str | None = None
    model_watch: WandBWatchConfig | None = None


@dataclass(kw_only=True)
class RuntimeConfig:
    num_proc: int | None = None
    max_threads: int | None = None
    registry_root: str = "./data"
    local_store_root: str | None = None


@dataclass(kw_only=True)
class RunConfig:
    wandb: WandBConfig | None = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
