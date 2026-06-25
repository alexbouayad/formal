from collections.abc import Mapping, Sequence
from ctypes import Array, c_bool
from dataclasses import dataclass
from typing import NamedTuple

from formal.utils.formatting import ReprMixin

type DFAState = int
type ParseState = int


class Symbol(str):
    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"


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


@dataclass(eq=False, frozen=True, repr=False, slots=True)
class LexTables(ReprMixin):
    advance_table: Mapping[tuple[DFAState, str], DFAState]
    accept_table: Mapping[DFAState, Symbol | None]
    eof_table: Mapping[DFAState, Symbol | None]

    def __format__(self, format_spec: str) -> str:
        return (
            f"advance_table={{{len(self.advance_table)} entries}}, "
            f"accept_table={{{len(self.accept_table)} entries}}, "
            f"eof_table={{{len(self.eof_table)} entries}}"
        )


@dataclass(eq=False, frozen=True, repr=False, slots=True)
class ParseTables(ReprMixin):
    action_table: Mapping[tuple[ParseState, Symbol], Sequence[ParseAction]]
    goto_table: Mapping[tuple[ParseState, Symbol], ParseState | None]
    lex_modes: Mapping[ParseState, LexMode]

    def __format__(self, format_spec: str) -> str:
        return (
            f"action_table={{{len(self.action_table)} entries}}, "
            f"goto_table={{{len(self.goto_table)} entries}}, "
            f"lex_modes={{{len(self.lex_modes)} entries}}"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class EndSymbols:
    end_symbol: Symbol
    end_of_nonterminal_extra_symbol: Symbol | None


@dataclass(eq=False, frozen=True, kw_only=True, repr=False, slots=True)
class Grammar(ReprMixin):
    lex_tables: LexTables
    parse_tables: ParseTables

    terminal_symbols: Sequence[Symbol]
    nonterminal_symbols: Sequence[Symbol]
    external_symbols: Sequence[Symbol]

    end_symbols: EndSymbols

    @property
    def formal_vocab_size(self) -> int:
        return len(self.terminal_symbols) + len(self.nonterminal_symbols) + len(self.external_symbols)

    def __format__(self, format_spec: str) -> str:
        return (
            f"lex_tables={self.lex_tables}, "
            f"parse_tables={self.parse_tables}, "
            f"terminal_symbols={len(self.terminal_symbols)} symbols, "
            f"nonterminal_symbols={len(self.nonterminal_symbols)} symbols, "
            f"special_symbols={self.end_symbols}, "
        )
