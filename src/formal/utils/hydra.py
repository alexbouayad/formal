# TODO
# import logging
# import sys

import os
from dataclasses import field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Protocol, cast

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from formal.config import RunConfig
from formal.language import Language
from formal.utils.hf_hub import get_hf_namespace


class HydraDecorator(Protocol):
    def __call__[ConfigT: RunConfig](self, func: Callable[[ConfigT], None]) -> Callable[[], None]: ...


def instantiable_field(target: str) -> dict[str, Any]:
    return field(default_factory=lambda: {"_target_": target})


def main(stage_name: str) -> HydraDecorator:
    def decorator[ConfigT: RunConfig](func: Callable[[ConfigT], None]) -> Callable[[], None]:
        @wraps(func)
        def wrapper(dict_config: DictConfig) -> None:
            # TODO
            # logging.basicConfig(
            #     level=logging.INFO,
            #     stream=sys.stdout,
            #     format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            #     datefmt="%Y-%m-%d %H:%M:%S",
            # )

            config = OmegaConf.to_object(dict_config)
            config = cast(ConfigT, config)

            return func(config)

        config_path = Path.cwd() / "conf"

        if (config_path / f"{stage_name}.yaml").exists():
            return hydra.main(config_path=str(config_path), config_name=stage_name, version_base="1.3")(wrapper)

        return hydra.main(config_path=None, version_base="1.3")(wrapper)

    return decorator


def setup(stage_name: str, config_cls: type[RunConfig]) -> ConfigStore:
    def get_hydra_output_dir() -> str | None:
        runtime_config = HydraConfig.get().runtime
        runtime_config = cast(DictConfig, runtime_config)

        return OmegaConf.select(runtime_config, "output_dir", default="<hydra_output_dir>")

    def get_default_registry_root() -> str:
        if hf_namespace := get_hf_namespace():
            return f"hf://{hf_namespace}"

        else:
            return "./data"

    def get_name(s: str) -> str:
        return s.split("/")[-1]

    def get_language_id(language: Language) -> str:
        return language.value

    def sweep_all_languages() -> str:
        return ",".join([language.name for language in Language])

    config_store = ConfigStore.instance()
    config_store.store(name=f"base_{stage_name}", node=config_cls)

    OmegaConf.register_new_resolver("cpu_count", os.process_cpu_count, use_cache=True)
    OmegaConf.register_new_resolver("hydra_output_dir", get_hydra_output_dir, use_cache=True)
    OmegaConf.register_new_resolver("default_registry_root", get_default_registry_root, use_cache=True)

    OmegaConf.register_new_resolver("get_name", get_name, use_cache=True)
    OmegaConf.register_new_resolver("get_language_id", get_language_id, use_cache=True)
    OmegaConf.register_new_resolver("sweep_all_languages", sweep_all_languages, use_cache=True)

    return config_store
