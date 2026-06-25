from typing import NamedTuple, Protocol

from formal.engine.buffer import TextPosition
from formal.grammar.types import ExternalState, Symbol

from .scanlet import Scanlet


class ScanResult(NamedTuple):
    symbol: Symbol
    position: TextPosition


class ScanState(bytes):
    __slots__ = ()

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return self.hex()


class Scanner(Protocol):
    __slots__ = ()

    state: Scanlet | ScanState | None
    external_state: ExternalState | None

    def __bool__(self) -> bool: ...
    def reset(self) -> None: ...
    def scan(self) -> ScanResult | None: ...
