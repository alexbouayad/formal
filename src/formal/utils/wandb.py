from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict
from functools import wraps
from typing import Callable

import wandb

from formal.artifacts import ArtifactReference, HFHubPath, LocalPath, resolve_local_path, resolve_path
from formal.config import RunConfig, RuntimeConfig


def log_input_on_wandb(reference: ArtifactReference, *, runtime_config: RuntimeConfig) -> None:
    if not (wandb_run := wandb.run):
        return

    rev_prefix = "hf-rev" if runtime_config.registry_root.startswith("hf://") else "local-rev"
    name = f"{reference.identifier}:{rev_prefix}-{reference.revision}"

    wandb_run.use_artifact(name, type=reference.artifact_type)


def log_output_on_wandb(reference: ArtifactReference, *, runtime_config: RuntimeConfig) -> None:
    if not (wandb_run := wandb.run):
        return

    if runtime_config.local_store_root is None:
        local_store_path = None
    else:
        local_store_path = resolve_local_path(reference, runtime_config.local_store_root)

    artifact = wandb.Artifact(name=reference.identifier, type=reference.artifact_type)
    aliases: list[str] = []

    def add_local_reference(path: LocalPath):
        local_uri = path.resolve().as_uri()
        artifact.add_reference(local_uri, name="local", checksum=True)

        aliases.append(f"local-oid-{artifact.digest}")

    match registry_path := resolve_path(reference, runtime_config.registry_root):
        case HFHubPath():
            from huggingface_hub import repo_info

            repo_info = repo_info(
                repo_id=registry_path.repo_id,
                repo_type=registry_path.repo_type,
                revision=reference.revision,
            )

            if repo_info.sha is not None:
                hf_uri = f"{registry_path}/tree/{repo_info.sha}/{reference.config_name}"
                artifact.add_reference(hf_uri, name="hf_hub", checksum=False)

                aliases.append(f"hf-oid-{repo_info.sha}")
                aliases.append(f"hf-rev-{reference.revision}")

        case LocalPath():
            if not local_store_path and registry_path.exists():
                add_local_reference(registry_path)

            aliases.append(f"local-rev-{reference.revision}")

    if local_store_path and local_store_path.exists():
        add_local_reference(local_store_path)

    wandb_run.log_artifact(artifact, aliases=aliases)


def with_wandb[ConfigT: RunConfig](func: Callable[[ConfigT], None]) -> Callable[[ConfigT], None]:
    @wraps(func)
    def wrapper(config: ConfigT) -> None:
        if config.wandb is None:
            wandb_context = nullcontext()

        else:
            from hydra.core.hydra_config import HydraConfig

            hydra_output_dir = HydraConfig.get().runtime.output_dir if HydraConfig.initialized() else None

            wandb_context = wandb.init(
                project=config.wandb.project,
                dir=hydra_output_dir,
                name=config.wandb.run_name,
                config=asdict(config),
                reinit="finish_previous",
                save_code=True,
            )

        with wandb_context:
            func(config)

    return wrapper
