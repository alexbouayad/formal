from collections import deque
from collections.abc import Iterator
from logging import DEBUG, getLogger
from typing import Final

from tree_sitter_generate import START_PARSE_STATE_ID

from formal.engine.buffer import ReadOnlyBuffer
from formal.engine.lexer import Lexer, LexResult
from formal.engine.scanner import Scanlet, Scanner, ScanResult
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
    ) -> None:
        self.parse_tables = parse_tables

        self._buffer = buffer
        self._lexer = lexer
        self._scanner = scanner

        self._queue = deque()
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
                logger.debug(f"  PARSE CONFIGURATION: {lattice}")

            self._lexer.reset()
            self._scanner.reset()

            lex_outcome = self._lex()

            if logger.isEnabledFor(DEBUG):
                logger.debug(f"  LEX OUTCOME: {lex_outcome}")

            end_lex_state = self._lexer.state
            end_scan_state = self._scanner.state

            if end_lex_state.is_live:
                new_head = lattice.superpose(
                    position=self._buffer.position,
                    lex_state=end_lex_state,
                    scan_state=lattice.scan_state,
                )

                if logger.isEnabledFor(DEBUG):
                    with lattice.peek(new_head):
                        logger.debug(f"  CREATE PARSE VERTEX: {lattice}")

            if isinstance(end_scan_state, Scanlet):
                new_head = lattice.superpose(
                    position=self._buffer.position,
                    lex_state=None,
                    scan_state=end_scan_state,
                )

                if logger.isEnabledFor(DEBUG):
                    with lattice.peek(new_head):
                        logger.debug(f"  CREATE PARSE VERTEX: {lattice}")

            match lex_outcome:
                case ScanResult(lookahead, end_position):
                    lattice.position = end_position
                    lattice.scan_state = self._scanner.state

                case LexResult(lookahead, end_position):
                    lattice.position = end_position
                    lattice.lex_state = None

                case _:
                    lookahead = None

            if lookahead or end_lex_state.is_negative:
                lattice.collapse()

                if logger.isEnabledFor(DEBUG):
                    logger.debug(f"  COLLAPSE CONFIGURATION: {lattice}")

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
        action_table = self.parse_tables.action_table
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

            parse_actions = action_table.get((parse_state, lookahead), ())

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

    def _lex(self) -> ScanResult | LexResult | None:
        lex_modes = self.parse_tables.lex_modes
        lattice = self._lattice

        self._buffer.position = lattice.position

        match (lattice.lex_state, lattice.scan_state):
            case (_, Scanlet()):
                return self._scan()

            case (None, _):
                if self._scanner:
                    scan_outcome = self._scan()

                    if scan_outcome:
                        return scan_outcome

                    self._buffer.position = lattice.position

                self._lexer.state = lex_modes[lattice.stack.parse_state].lex_state

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

        return lex_outcome

    def _scan(self) -> ScanResult | None:
        lex_modes = self.parse_tables.lex_modes
        lattice = self._lattice

        if logger.isEnabledFor(DEBUG):
            logger.debug(f"  READER BEFORE SCAN: {self._buffer}")
            logger.debug(f"  SCANNER BEFORE SCAN: {self._scanner}")

        self._scanner.state = lattice.scan_state
        self._scanner.external_state = lex_modes[lattice.stack.parse_state].external_state

        scan_outcome = self._scanner.scan()

        if logger.isEnabledFor(DEBUG):
            logger.debug(f"  SCAN OUTCOME: {scan_outcome}")
            logger.debug(f"  READER AFTER SCAN: {self._buffer}")
            logger.debug(f"  SCANNER AFTER SCAN: {self._scanner}")

        return scan_outcome

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
