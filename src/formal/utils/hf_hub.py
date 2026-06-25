from typing import cast

from huggingface_hub import get_token, whoami  # type: ignore


def get_hf_namespace(cache: bool = False) -> str | None:
    if get_token() is None:
        return None

    try:
        hf_namespace = whoami(cache=cache).get("name")  # type: ignore
        hf_namespace = cast(str | None, hf_namespace)

    except Exception:
        return None

    return hf_namespace
