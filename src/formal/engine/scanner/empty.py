from typing import override

from .scanner import ExternalState, Scanlet, Scanner, ScanState


class EmptyScanner(Scanner):
    __slots__ = ("state", "external_state")

    state: Scanlet | ScanState | None
    external_state: ExternalState | None

    def __init__(self) -> None:
        self.state = None
        self.external_state = None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @override
    def __bool__(self) -> bool:
        return False

    @override
    def reset(self) -> None:
        pass

    @override
    def scan(self) -> None:
        return None


EMPTY_SCANNER = EmptyScanner()
