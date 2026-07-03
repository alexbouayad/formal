from collections.abc import Mapping, Sequence
from ctypes import Array, c_bool
from dataclasses import dataclass
from typing import NamedTuple

from rich.pretty import pretty_repr

type DFAState = int
type ParseState = int
type Symbol = str


@dataclass(frozen=True, slots=True)
class LexState:
    plus: DFAState | None = None
    minus: DFAState | None = None

    def __bool__(self) -> bool:
        return self.plus is not None or self.minus is not None

    @property
    def is_live(self) -> bool:
        return self.plus is not None

    @property
    def is_positive(self) -> bool:
        return self.plus is not None and self.minus is None

    @property
    def is_negative(self) -> bool:
        return self.plus is None and self.minus is not None


type ExternalState = Array[c_bool]


class LexMode(NamedTuple):
    lex_state: LexState
    external_state: ExternalState


class AcceptAction(NamedTuple):
    pass


class RecoverAction(NamedTuple):
    pass


class ReduceAction(NamedTuple):
    symbol: Symbol
    child_count: int


class ShiftAction(NamedTuple):
    state: ParseState | None


type ParseAction = AcceptAction | RecoverAction | ReduceAction | ShiftAction


@dataclass(eq=False, frozen=True, slots=True)
class LexTables:
    advance_table: Mapping[tuple[DFAState, str], DFAState]
    accept_table: Mapping[DFAState, Symbol | None]
    eof_table: Mapping[DFAState, Symbol | None]

    def __repr__(self) -> str:
        return pretty_repr(self, max_length=1)

    def __rich_repr__(self):
        yield "advance_table", self.advance_table
        yield "accept_table", self.accept_table
        yield "eof_table", self.eof_table


@dataclass(eq=False, frozen=True, slots=True)
class ParseTables:
    action_table: Mapping[tuple[ParseState, Symbol], Sequence[ParseAction]]
    goto_table: Mapping[tuple[ParseState, Symbol], ParseState | None]
    lex_modes: Mapping[ParseState, LexMode]

    def __repr__(self) -> str:
        return pretty_repr(self, max_length=1)

    def __rich_repr__(self):
        yield "action_table", self.action_table
        yield "goto_table", self.goto_table
        yield "lex_modes", self.lex_modes


@dataclass(eq=False, frozen=True, kw_only=True, slots=True)
class Grammar:
    lex_tables: LexTables
    parse_tables: ParseTables

    terminal_symbols: Sequence[Symbol]
    nonterminal_symbols: Sequence[Symbol]
    external_symbols: Sequence[Symbol]

    end_symbol: Symbol
    end_of_nonterminal_extra_symbol: Symbol | None

    @property
    def formal_vocab_size(self) -> int:
        return len(self.terminal_symbols) + len(self.nonterminal_symbols) + len(self.external_symbols)

    def __repr__(self) -> str:
        return pretty_repr(self, max_length=1)

    def __rich_repr__(self):
        yield "lex_tables", self.lex_tables
        yield "parse_tables", self.parse_tables
        yield "terminal_symbols", self.terminal_symbols
        yield "nonterminal_symbols", self.nonterminal_symbols
        yield "external_symbols", self.external_symbols
        yield "end_symbol", self.end_symbol
        yield "end_of_nonterminal_extra_symbol", self.end_of_nonterminal_extra_symbol
