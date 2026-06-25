from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum


class TokenTypeId(IntEnum):
    TEXT = 0
    FORMAL_RULE = 1
    FORMAL_FIELD = 2


@dataclass(kw_only=True, repr=False, slots=True)
class FormalEncoding:
    token_ids: list[int] = field(default_factory=list[int])
    token_type_ids: list[TokenTypeId] = field(default_factory=list[TokenTypeId])
    token_ast_depths: list[int] = field(default_factory=list[int])

    def __bool__(self) -> bool:
        return bool(self.token_ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[tuple[int, TokenTypeId, int]]:
        yield from zip(self.token_ids, self.token_type_ids, self.token_ast_depths)
