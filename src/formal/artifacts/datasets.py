from typing import TYPE_CHECKING, Literal, cast

from formal.config import RuntimeConfig

from .path import HFHubPath, LocalPath, resolve_local_path, resolve_path
from .references import DatasetReference

if TYPE_CHECKING:
    from datasets import Dataset


def load(
    reference: DatasetReference,
    split: Literal["train", "test", "validation"],
    *,
    runtime_config: RuntimeConfig,
) -> "Dataset":
    import datasets

    match registry_path := resolve_path(reference, runtime_config.registry_root):
        case HFHubPath():
            dataset = datasets.load_dataset(  # type: ignore
                path=f"hf://datasets/{registry_path.repo_id}",
                name=reference.config,
                revision=reference.revision,
                num_proc=runtime_config.num_proc,
                split=split,
            )

        case LocalPath():
            dataset = datasets.load_from_disk(registry_path)  # type: ignore
            dataset = cast("Dataset", dataset)

    return dataset


def save(dataset: "Dataset", reference: DatasetReference, *, runtime_config: RuntimeConfig) -> None:
    if runtime_config.local_store_root is None:
        local_store_path = None
    else:
        local_store_path = resolve_local_path(reference, runtime_config.local_store_root)

    match registry_path := resolve_path(reference, runtime_config.registry_root):
        case HFHubPath():
            dataset.push_to_hub(
                registry_path.repo_id,
                config_name=reference.config,
                revision=reference.revision,
                num_proc=runtime_config.num_proc,
                private=True,
            )

        case LocalPath():
            if local_store_path:
                registry_path.alias(local_store_path)

            else:
                dataset.save_to_disk(registry_path, num_proc=runtime_config.num_proc)  # type: ignore

    if local_store_path:
        dataset.save_to_disk(local_store_path, num_proc=runtime_config.num_proc)  # type: ignore
