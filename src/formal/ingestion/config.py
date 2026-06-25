from dataclasses import dataclass

from formal.artifacts.references import DatasetReference
from formal.config import StageConfig
from formal.language import Language


@dataclass(kw_only=True)
class IngestionConfig(StageConfig):
    language: Language

    num_samples: int | None = None
    min_byte_size: int | None = None
    max_byte_size: int | None = None

    output: DatasetReference
