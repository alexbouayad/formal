from typing import TYPE_CHECKING

from formal.config import RuntimeConfig

from .path import HFHubPath, LocalPath, resolve_local_path, resolve_path
from .references import ModelReference

if TYPE_CHECKING:
    from formal.model import FormalModel


def load(reference: ModelReference, *, runtime_config: RuntimeConfig) -> "FormalModel":
    from formal.model import FormalModel

    match registry_path := resolve_path(reference, runtime_config.registry_root):
        case HFHubPath():
            model = FormalModel.from_pretrained(
                path=registry_path.repo_id,
                config_name=reference.config_name,
                revision=reference.revision,
            )

        case LocalPath():
            model = FormalModel.from_pretrained(str(registry_path), config_name=reference.config_name)

    return model


def save(model: "FormalModel", reference: ModelReference, *, runtime_config: RuntimeConfig) -> None:
    if runtime_config.local_store_root is None:
        local_store_path = None
    else:
        local_store_path = resolve_local_path(reference, runtime_config.local_store_root)

    match registry_path := resolve_path(reference, runtime_config.registry_root):
        case HFHubPath():
            model.push_to_hub(
                repo_id=registry_path.repo_id,
                config_name=reference.config_name,
                revision=reference.revision,
                private=True,
            )

        case LocalPath():
            if local_store_path:
                registry_path.alias(local_store_path)

            else:
                model.save_pretrained(str(registry_path), config_name=reference.config_name)

    if local_store_path:
        model.save_pretrained(str(local_store_path), config_name=reference.config_name)
