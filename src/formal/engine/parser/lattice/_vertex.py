from dataclasses import dataclass, field
from logging import DEBUG, getLogger
from typing import Self

from ts_generate import START_PARSE_STATE_ID

from formal.engine.buffer import TextPosition
from formal.engine.parser.stack import GSSNode
from formal.engine.scanner import Scanlet, ScanState
from formal.grammar.types import LexState, ParseState

logger = getLogger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class ParseSignature:
    position: TextPosition
    lex_state: LexState | None
    scan_state: Scanlet | ScanState | None
    parse_state: ParseState


@dataclass(eq=False, kw_only=True, repr=False, slots=True)
class ParseVertex:
    stack: GSSNode = field(default_factory=lambda: GSSNode(START_PARSE_STATE_ID))
    position: TextPosition = field(default_factory=TextPosition)
    lex_state: LexState | None = field(default=None)
    scan_state: Scanlet | ScanState | None = field(default=None)

    parent: Self | None = field(default=None, init=False)
    children: dict[ParseSignature, Self] = field(default_factory=dict[ParseSignature, Self], init=False)

    @property
    def signature(self) -> ParseSignature:
        return ParseSignature(
            position=self.position,
            lex_state=self.lex_state,
            scan_state=self.scan_state,
            parse_state=self.stack.parse_state,
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}: "
            f"stack={self.stack}, "
            f"position={self.position}, "
            f"lex_state={self.lex_state}, "
            f"scan_state={self.scan_state}, "
            f"level={self.level()}, "
            f"children={len(self.children)}>"
        )

    def level(self) -> int:
        level = 0
        commit = self

        while commit := commit.parent:
            level += 1

        return level

    def release(self) -> None:
        if logger.isEnabledFor(DEBUG):
            logger.debug(f"Releasing {self}")

        match self.scan_state:
            case Scanlet() as scanlet:
                scanlet.throw()

            case _:
                pass
