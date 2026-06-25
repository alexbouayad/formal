from dataclasses import dataclass
from typing import ClassVar


@dataclass(kw_only=True, slots=True)
class ArtifactReference:
    artifact_type: ClassVar[str]

    artifact_name: str
    config_name: str = "default"
    revision: str = "main"

    @property
    def identifier(self) -> str:
        return f"{self.artifact_name}-{self.config_name}"


class DatasetReference(ArtifactReference):
    artifact_type = "dataset"


@dataclass(kw_only=True, slots=True)
class ModelReference(ArtifactReference):
    artifact_type = "model"
