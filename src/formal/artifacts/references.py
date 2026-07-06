from dataclasses import dataclass
from typing import ClassVar


@dataclass(kw_only=True, slots=True)
class ArtifactReference:
    type: ClassVar[str]

    name: str
    config: str = "default"
    revision: str = "main"

    @property
    def identifier(self) -> str:
        return f"{self.name}-{self.config}"


class DatasetReference(ArtifactReference):
    artifact_type = "dataset"


@dataclass(kw_only=True, slots=True)
class ModelReference(ArtifactReference):
    artifact_type = "model"
