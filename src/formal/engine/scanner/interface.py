from typing import TYPE_CHECKING, NamedTuple, Protocol

from formal.engine.buffer import TextPosition
from formal.grammar.types import ExternalState, Symbol

if TYPE_CHECKING:
    from .scanlet import Scanlet


type ScanState = bytes


class ScanResult(NamedTuple):
    symbol: Symbol
    position: TextPosition


class Scanner(Protocol):
    __slots__ = ()

    state: "Scanlet | ScanState | None"
    external_state: ExternalState | None

    def __bool__(self) -> bool: ...
    def reset(self) -> None: ...
    def scan(self) -> ScanResult | None: ...
