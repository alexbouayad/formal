from dataclasses import dataclass

from formal.artifacts.references import DatasetReference
from formal.config import FormalizationConfig, StageConfig


@dataclass(kw_only=True)
class TokenizationConfig(StageConfig):
    tokenizer_id: str
    formalization: FormalizationConfig

    min_seq_length: int | None = None
    max_seq_length: int | None = None

    input: DatasetReference
    output: DatasetReference
