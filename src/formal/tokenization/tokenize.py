import logging
from collections.abc import Sized
from typing import cast

from transformers import AutoTokenizer, TokenizersBackend

from formal import artifacts
from formal.tokenization.tokenizer import FormalTokenizer
from formal.utils.wandb import log_input_on_wandb, log_output_on_wandb, with_wandb

from .config import TokenizationConfig

logger = logging.getLogger(__name__)


# TODO: remove
# class _CharacterSet(set[str]):
#     def __init__(self, strings: Iterable[str], /) -> None:
#         super().__init__()

#         for string in strings:
#             self.update(string)

#     def __reduce__(self) -> tuple[type[Self], tuple[tuple[str, ...]]]:
#         return (self.__class__, (tuple(self),))


# TODO: remove
# def _contains_only_characters_from(source: str, character_set: Set[str]) -> bool:
#     return set(source).issubset(character_set)


# TODO: remove
# def _is_valid_unicode(decoded_tokens: list[str]) -> bool:
#     # Filter out samples that contain the Unicode replacement character (U+FFFD)
#     return all(b"\xef\xbf\xbd" not in token.encode() for token in decoded_tokens)


def _is_within_bounds(sequence: Sized, *, min_length: int | None, max_length: int | None) -> bool:
    match min_length, max_length:
        case None, None:
            return True

        case None, _:
            return len(sequence) <= max_length

        case _, None:
            return min_length <= len(sequence)

        case _, _:
            return min_length <= len(sequence) <= max_length


def _remove_byte_order_mark(source: str) -> dict[str, str]:
    return {"source": source.lstrip("\ufeff")}


@with_wandb
def tokenize(tokenization_config: TokenizationConfig) -> None:
    dataset = artifacts.datasets.load(
        reference=tokenization_config.input,
        split="train",
        runtime_config=tokenization_config.runtime,
    )

    log_input_on_wandb(tokenization_config.input, runtime_config=tokenization_config.runtime)

    # TODO: suppress the warnings about the eos and bos tokens
    text_backend = AutoTokenizer.from_pretrained(tokenization_config.tokenizer_id)  # type: ignore
    text_backend = cast(TokenizersBackend, text_backend)

    # TODO: remove
    # character_set = _CharacterSet(
    #     text_backend.decode(token_id, clean_up_tokenization_spaces=False)  # type: ignore
    #     for token_id in range(text_backend.vocab_size)
    # )

    # TODO: remove
    # dataset = dataset.filter(  # type: ignore
    #     function=_contains_only_characters_from,
    #     input_columns="source",
    #     fn_kwargs={"character_set": character_set},
    #     num_proc=tokenization_config.runtime.num_proc,
    # )

    dataset = dataset.map(  # type: ignore
        function=_remove_byte_order_mark,
        input_columns="source",
        num_proc=tokenization_config.runtime.num_proc,
    )

    tokenizer = FormalTokenizer(text_backend, tokenization_config.formalization)

    dataset = dataset.map(  # type: ignore
        function=tokenizer,
        input_columns=["source"],
        fn_kwargs={"return_as_dict": True},
        num_proc=tokenization_config.runtime.num_proc,
    )

    dataset = dataset.filter(  # type: ignore
        _is_within_bounds,
        input_columns="token_ids",
        fn_kwargs={
            "min_length": tokenization_config.min_seq_length,
            "max_length": tokenization_config.max_seq_length,
        },
        num_proc=tokenization_config.runtime.num_proc,
    )

    # TODO: remove
    # dataset = dataset.filter(  # type: ignore
    #     function=_is_valid_unicode,
    #     input_columns="decoded_tokens",
    #     num_proc=tokenization_config.runtime.num_proc,
    # )

    artifacts.datasets.save(
        dataset=dataset,
        reference=tokenization_config.output,
        runtime_config=tokenization_config.runtime,
    )

    log_output_on_wandb(tokenization_config.output, runtime_config=tokenization_config.runtime)
