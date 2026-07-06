from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import IO, Any, cast

import boto3
import botocore.config
import smart_open

_s3_client: Any | None = None


def _fetch(blob_id: str, src_encoding: str) -> str:
    s3_url = f"s3://softwareheritage/content/{blob_id}"

    try:
        with smart_open.open(s3_url, "rb", compression=".gz", transport_params={"client": _s3_client}) as f:  # type: ignore
            f = cast(IO[bytes], f)
            content = f.read().decode(src_encoding)

            return content

    except OSError:
        return ""


def fetch(batch: Mapping[str, list[Any]], max_workers: int | None) -> dict[str, list[str]]:
    global _s3_client

    blob_ids = batch["blob_id"]
    src_encodings = batch["src_encoding"]

    if _s3_client is None:
        config = botocore.config.Config(max_pool_connections=max_workers)
        _s3_client = boto3.Session().client("s3", config=config)  # type: ignore

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        contents = [content for content in executor.map(_fetch, blob_ids, src_encodings)]

    return {"content": contents}
