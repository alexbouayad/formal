from collections import deque
from collections.abc import Iterator
from logging import DEBUG, getLogger
from typing import Final

from ts_generate import START_PARSE_STATE_ID

from formal.engine.buffer import ReadOnlyBuffer
from formal.engine.lexer import Lexer
from formal.engine.scanner import Scanlet, Scanner
from formal.grammar.types import (
    AcceptAction,
    LexState,
    ParseState,
    ParseTables,
    RecoverAction,
    ReduceAction,
    ShiftAction,
    Symbol,
)

from .lattice import ParseHead, ParseLattice
from .stack import StackQueue

logger = getLogger(__name__)


class Parser:
    __slots__ = (
        "parse_tables",
        "_buffer",
        "_lexer",
        "_scanner",
        "_queue",
        "_lattice",
        "_checkpoint",
    )

    parse_tables: Final[ParseTables]

    _buffer: Final[ReadOnlyBuffer]
    _lexer: Final[Lexer]
    _scanner: Final[Scanner]

    _queue: deque[ParseHead]
    _lattice: ParseLattice
    _checkpoint: ParseLattice | None

    @property
    def is_live(self) -> bool:
        return bool(self._lattice)

    def __init__(
        self,
        *,
        parse_tables: ParseTables,
        buffer: ReadOnlyBuffer,
        lexer: Lexer,
        scanner: Scanner,
        max_heads: int = 16,
    ) -> None:
        self.parse_tables = parse_tables

        self._buffer = buffer
        self._lexer = lexer
        self._scanner = scanner

        self._queue = deque(maxlen=max_heads)
        self._lattice = ParseLattice()
        self._checkpoint = None

    def __iter__(self) -> Iterator[ParseState]:
        for signature in self._lattice.surface:
            yield signature.parse_state

    def checkpoint(self) -> None:
        self._checkpoint = ParseLattice(self._lattice)

    def parse(self, *, is_eos: bool = False) -> bool:
        lattice = self._lattice

        self._queue += lattice

        if logger.isEnabledFor(DEBUG):
            logger.debug("===== PARSE CYCLE =====")

        while self._queue:
            if logger.isEnabledFor(DEBUG):
                logger.debug("")
                logger.debug(f"PARSE QUEUE: heads={len(self._queue)}")

            head = self._queue.popleft()

            if head not in lattice:
                if logger.isEnabledFor(DEBUG):
                    logger.debug(f"  STALE HEAD: {head}")

                continue

            lattice.focus(head)

            if logger.isEnabledFor(DEBUG):
                logger.debug(f"  PARSE LATTICE: {lattice}")

            self._lexer.reset()
            self._scanner.reset()

            lookahead = self._lex()

            if lookahead or self._lexer.state.is_negative:
                lattice.collapse()

                if logger.isEnabledFor(DEBUG):
                    logger.debug(f"  COLLAPSE LATTICE: {lattice}")

            if (lattice.lex_state and lattice.lex_state.is_live) or isinstance(lattice.scan_state, Scanlet):
                lattice.commit()

                continue

            self._advance(lookahead)

        if logger.isEnabledFor(DEBUG):
            logger.debug("")
            logger.debug("===== PARSE CYCLE COMPLETE =====")
            logger.debug("")
            logger.debug("")

        if not is_eos:
            return bool(lattice)

        for head in lattice:
            lattice.focus(head)

            if lattice.stack.parse_state == START_PARSE_STATE_ID:
                return True

        return False

    def reset(self) -> None:
        self._lexer.reset()
        self._scanner.reset()

        self._queue.clear()
        self._lattice.clear()
        self._checkpoint = None

    def revert(self) -> None:
        snapshot = self._checkpoint

        if snapshot is None:
            return

        self._lattice = ParseLattice(snapshot)

    def _advance(self, lookahead: Symbol | None) -> None:
        lattice = self._lattice

        lattice.annihilate()

        if logger.isEnabledFor(DEBUG):
            logger.debug(f"  ANNIHILATE PARSE VERTEX: {lattice}")

        if not lookahead:
            return

        stack_queue = StackQueue((lattice.stack,))

        while stack_queue:
            parse_state, stack = stack_queue.pop()
            lattice.stack = stack

            if logger.isEnabledFor(DEBUG):
                logger.debug(f"    STACK: {stack}")

            parse_actions = self.parse_tables.action_table.get((parse_state, lookahead), ())

            if logger.isEnabledFor(DEBUG) and not parse_actions:
                logger.debug("      NO ACTIONS")

            for parse_action in parse_actions:
                if logger.isEnabledFor(DEBUG):
                    logger.debug(f"      ACTION: {parse_action}")

                match parse_action:
                    case AcceptAction():
                        self._accept()

                    case RecoverAction():
                        self._recover()

                    case ReduceAction(reduce_symbol, child_count):
                        self._reduce(reduce_symbol, child_count, stack_queue=stack_queue)

                    case ShiftAction(None):
                        self._shift()

                    case ShiftAction(next_parse_state):
                        self._shift(next_parse_state)

    def _lex(self) -> Symbol | None:
        lattice = self._lattice

        self._buffer.position = lattice.position

        match (lattice.lex_state, lattice.scan_state):
            case (_, Scanlet()):
                if scanned_symbol := self._scan():
                    return scanned_symbol

                lattice.position = self._buffer.position
                lattice.scan_state = self._scanner.state

                return None

            case (None, _):
                if self._scanner:
                    if scanned_symbol := self._scan():
                        return scanned_symbol

                    if isinstance(self._scanner.state, Scanlet):
                        new_head = lattice.superpose(position=self._buffer.position, scan_state=self._scanner.state)

                        if logger.isEnabledFor(DEBUG):
                            with lattice.peek(new_head):
                                logger.debug(f"  SUPERPOSE PARSE VERTEX: {lattice}")

                    self._buffer.position = lattice.position

                self._lexer.state = self.parse_tables.lex_modes[lattice.stack.parse_state].lex_state

            case (LexState() as lex_state, _):
                self._lexer.state = lex_state

        if logger.isEnabledFor(DEBUG):
            logger.debug(f"  READER BEFORE LEX: {self._buffer}")
            logger.debug(f"  LEXER BEFORE LEX: {self._lexer}")

        lex_outcome = self._lexer.lex()

        if logger.isEnabledFor(DEBUG):
            logger.debug(f"  LEX OUTCOME: {lex_outcome}")
            logger.debug(f"  READER AFTER LEX: {self._buffer}")
            logger.debug(f"  LEXER AFTER LEX: {self._lexer}")

        if lex_outcome is None:
            lattice.position = self._buffer.position
            lattice.lex_state = self._lexer.state

            return None

        if self._lexer.state.is_live:
            new_head = lattice.superpose(position=lex_outcome.position, lex_state=self._lexer.state)

            if logger.isEnabledFor(DEBUG):
                with lattice.peek(new_head):
                    logger.debug(f"  SUPERPOSE PARSE VERTEX: {lattice}")

        lattice.position = lex_outcome.position
        lattice.lex_state = None

        return lex_outcome.symbol

    def _scan(self) -> Symbol | None:
        lattice = self._lattice

        self._scanner.state = lattice.scan_state
        self._scanner.external_state = self.parse_tables.lex_modes[lattice.stack.parse_state].external_state

        if logger.isEnabledFor(DEBUG):
            logger.debug(f"  READER BEFORE SCAN: {self._buffer}")
            logger.debug(f"  SCANNER BEFORE SCAN: {self._scanner}")

        scan_outcome = self._scanner.scan()

        if logger.isEnabledFor(DEBUG):
            logger.debug(f"  SCAN OUTCOME: {scan_outcome}")
            logger.debug(f"  READER AFTER SCAN: {self._buffer}")
            logger.debug(f"  SCANNER AFTER SCAN: {self._scanner}")

        if scan_outcome is None:
            return None

        lattice.position = scan_outcome.position
        lattice.scan_state = self._scanner.state

        return scan_outcome.symbol

    def _accept(self) -> None:
        lattice = self._lattice
        stack = lattice.stack

        for new_stack in stack.pop():
            lattice.stack = new_stack
            new_head = lattice.create()

            if logger.isEnabledFor(DEBUG):
                with lattice.peek(new_head):
                    logger.debug(f"        ACCEPT: {lattice}")

    def _recover(self) -> None:
        if logger.isEnabledFor(DEBUG):
            logger.debug("        RECOVER")

    def _reduce(self, symbol: Symbol, count: int, *, stack_queue: StackQueue) -> None:
        goto_table = self.parse_tables.goto_table
        lattice = self._lattice

        for popped_stack in lattice.stack.pop(count):
            parse_state = popped_stack.parse_state

            if logger.isEnabledFor(DEBUG):
                logger.debug(f"        POP: parse_state={parse_state}")

            next_parse_state = goto_table[parse_state, symbol]

            if next_parse_state is None:
                lattice.stack = popped_stack
                self._shift()

            else:
                new_stack = popped_stack.push(next_parse_state)
                stack_queue.add(new_stack)

                if logger.isEnabledFor(DEBUG):
                    logger.debug(f"          GOTO: parse_state={next_parse_state}")

    def _shift(self, next_parse_state: ParseState | None = None) -> None:
        buffer = self._buffer
        lattice = self._lattice

        stack = lattice.stack
        new_stack = stack if next_parse_state is None else stack.push(next_parse_state)

        lattice.stack = new_stack
        new_head = lattice.create()

        if lattice.position.offset < len(buffer) or buffer.at_eof:
            self._queue.append(new_head)

        if logger.isEnabledFor(DEBUG):
            action_text = "SHIFT_EXTRA" if next_parse_state is None else "SHIFT"

            if lattice.position.offset >= len(buffer) and not buffer.at_eof:
                action_text += " (PAUSED)"

            with lattice.peek(new_head):
                logger.debug((f"        {action_text}: {lattice}"))

        lattice.stack = stack
