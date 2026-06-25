from collections.abc import Callable
from contextvars import ContextVar
from copy import replace
from dataclasses import dataclass, field
from logging import DEBUG, getLogger
from typing import Self

from greenlet import greenlet

from formal.engine.buffer import TextPosition

from .scanner import ScanResult, ScanState

logger = getLogger(__name__)


# TODO: remove debug pool
_debug_pool: set["Scanlet"] = set()
_current_scanlet: ContextVar["Scanlet | None"] = ContextVar("current_scanlet", default=None)


@dataclass(eq=False, frozen=True, kw_only=True, slots=True)
class Scanlet:
    run: Callable[[ScanState | None], ScanResult | None] = field(repr=False)
    start_state: ScanState | None
    start_position: TextPosition

    _glet: greenlet = field(default_factory=greenlet, init=False)

    def __post_init__(self) -> None:
        def run(start_state: ScanState | None) -> ScanResult | None:
            _current_scanlet.set(self)

            return self.run(start_state)

        self._glet.run = run

        # TODO: remove debug pool
        if logger.isEnabledFor(DEBUG):
            _debug_pool.add(self)

    def __bool__(self) -> bool:
        return bool(self._glet)

    def clone(self) -> Self:
        return replace(self)

    def switch(self) -> ScanResult | None:
        return self._glet.switch()

    def throw(self) -> None:
        # TODO: remove debug pool
        if logger.isEnabledFor(DEBUG):
            logger.debug(f"Throwing {self}")

            _debug_pool.discard(self)

        self._glet.throw()

    @staticmethod
    def get_current() -> "Scanlet | None":
        return _current_scanlet.get()
