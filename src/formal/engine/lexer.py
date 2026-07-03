from copy import replace
from typing import Final, NamedTuple

from formal.engine.buffer import ReadOnlyBuffer, TextPosition
from formal.grammar.types import LexState, LexTables, Symbol


class LexResult(NamedTuple):
    symbol: Symbol
    position: TextPosition


class Lexer:
    __slots__ = (
        "buffer",
        "lex_tables",
        "end_symbol",
        "end_of_nonterminal_extra_symbol",
        "state",
    )

    buffer: Final[ReadOnlyBuffer]
    lex_tables: Final[LexTables]
    end_symbol: Final[Symbol]
    end_of_nonterminal_extra_symbol: Final[Symbol | None]

    state: LexState

    def __init__(
        self,
        *,
        buffer: ReadOnlyBuffer,
        lex_tables: LexTables,
        end_symbol: Symbol,
        end_of_nonterminal_extra_symbol: Symbol | None,
    ) -> None:
        self.buffer = buffer
        self.lex_tables = lex_tables
        self.end_symbol = end_symbol
        self.end_of_nonterminal_extra_symbol = end_of_nonterminal_extra_symbol

        self.state = LexState()

    def __repr__(self) -> str:
        return f"<Lexer: state={self.state}>"

    def reset(self) -> None:
        self.state = LexState()

    def lex(self) -> LexResult | None:
        advance_table = self.lex_tables.advance_table
        accept_table = self.lex_tables.accept_table
        eof_table = self.lex_tables.eof_table

        end_symbol = self.end_symbol
        end_of_nonterminal_extra_symbol = self.end_of_nonterminal_extra_symbol

        plus_state = self.state.plus
        minus_state = self.state.minus

        lex_result = None

        while character := self.buffer.read():
            if plus_state is not None:
                plus_state = advance_table.get((plus_state, character))

            if minus_state is not None:
                minus_state = advance_table.get((minus_state, character))

            if plus_state is None and minus_state is None:
                break

            plus_symbol = accept_table.get(plus_state) if plus_state is not None else None
            minus_symbol = accept_table.get(minus_state) if minus_state is not None else None

            lex_symbol = plus_symbol or minus_symbol

            if lex_symbol is not None:
                lex_result = LexResult(lex_symbol, self.buffer.position)

        if self.buffer.at_eof:
            if not lex_result:
                eof_symbol = eof_table.get(plus_state) if plus_state is not None else None
                lex_result = LexResult(eof_symbol, self.buffer.position) if eof_symbol is not None else None

        elif lex_result and lex_result.symbol == end_symbol:
            lex_result = replace(lex_result, symbol=end_of_nonterminal_extra_symbol)

        self.state = LexState(plus_state, minus_state)

        return lex_result
