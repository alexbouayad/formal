import logging
from collections.abc import Generator
from typing import Any

from datasets import Dataset, load_dataset  # type: ignore

from formal import artifacts
from formal.language import get_language_info
from formal.utils.wandb import log_output_on_wandb, with_wandb

from ._fetch import fetch
from .config import IngestionConfig

logger = logging.getLogger(__name__)


def _is_within_bounds(size: int, *, min_size: int | None, max_size: int | None) -> bool:
    match min_size, max_size:
        case None, None:
            return True

        case None, _:
            return size <= max_size

        case _, None:
            return min_size <= size

        case _, _:
            return min_size <= size <= max_size


@with_wandb
def ingest(ingestion_config: IngestionConfig) -> None:
    language_info = get_language_info(ingestion_config.language)

    iterable_dataset = load_dataset(
        path="bigcode/the-stack-v2-dedup",
        name=language_info.hf_name,
        streaming=True,
        split="train",
    )

    iterable_dataset = iterable_dataset.select_columns(["blob_id", "src_encoding", "length_bytes"])

    iterable_dataset = iterable_dataset.filter(  # type: ignore
        _is_within_bounds,
        input_columns="length_bytes",
        fn_kwargs={
            "min_size": ingestion_config.min_byte_size,
            "max_size": ingestion_config.max_byte_size,
        },
    )

    if ingestion_config.num_samples is not None:
        iterable_dataset = iterable_dataset.take(ingestion_config.num_samples)

    logger.info("Downloading SWHID data from Hugging Face...")

    def data_generator() -> Generator[dict[str, Any]]:
        yield from iterable_dataset

    dataset = Dataset.from_generator(data_generator)  # type: ignore

    logger.info("Loading content data...")

    dataset = dataset.map(  # type: ignore
        fetch,
        batched=True,
        batch_size=8 * (ingestion_config.runtime.max_threads or 1),
        remove_columns="src_encoding",
        fn_kwargs={"max_workers": ingestion_config.runtime.max_threads},
    )

    dataset = dataset.rename_column("content", "source")

    artifacts.datasets.save(
        dataset=dataset,
        reference=ingestion_config.output,
        runtime_config=ingestion_config.runtime,
    )

    log_output_on_wandb(ingestion_config.output, runtime_config=ingestion_config.runtime)
