from dataclasses import dataclass
from pathlib import Path
from typing import Self

from .references import ArtifactReference

type ArtifactPath = HFHubPath | LocalPath


@dataclass(frozen=True, kw_only=True, slots=True)
class HFHubPath:
    namespace: str
    repo_name: str
    repo_type: str

    def __post_init__(self) -> None:
        namespace = self.namespace.removeprefix("hf://").removesuffix("/")
        object.__setattr__(self, "namespace", namespace)

    @property
    def repo_id(self) -> str:
        return f"{self.namespace}/{self.repo_name}"

    def __str__(self) -> str:
        return f"https://huggingface.co/{self.repo_type}s/{self.repo_id}"


class LocalPath(Path):
    def alias(self, target: Self) -> None:
        if self == target:
            return

        self.unlink(missing_ok=True)
        self.parent.mkdir(parents=True, exist_ok=True)
        self.symlink_to(target, target_is_directory=True)


def resolve_local_path(reference: ArtifactReference, root: str) -> LocalPath:
    return (
        LocalPath(root)
        / f"{reference.artifact_type}s"
        / reference.artifact_name
        / reference.config_name
        / reference.revision
    )


def resolve_hf_hub_path(reference: ArtifactReference, root: str) -> HFHubPath:
    return HFHubPath(
        namespace=root,
        repo_name=reference.artifact_name,
        repo_type=reference.artifact_type,
    )


def resolve_path(reference: ArtifactReference, root: str) -> ArtifactPath:
    if root.startswith("hf://"):
        return resolve_hf_hub_path(reference, root)

    else:
        return resolve_local_path(reference, root)
