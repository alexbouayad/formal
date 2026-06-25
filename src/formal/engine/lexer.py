from copy import replace
from typing import Final, NamedTuple

from formal.engine.buffer import ReadOnlyBuffer, TextPosition
from formal.grammar.types import EndSymbols, LexState, LexTables, Symbol


class LexResult(NamedTuple):
    symbol: Symbol
    end_position: TextPosition


class Lexer:
    __slots__ = ("lex_tables", "special_symbols", "_buffer", "state")

    lex_tables: Final[LexTables]
    special_symbols: Final[EndSymbols]
    _buffer: Final[ReadOnlyBuffer]

    state: LexState

    def __init__(
        self,
        *,
        lex_tables: LexTables,
        end_symbols: EndSymbols,
        buffer: ReadOnlyBuffer,
    ) -> None:
        self.lex_tables = lex_tables
        self.special_symbols = end_symbols
        self._buffer = buffer

        self.state = LexState()

    def reset(self) -> None:
        self.state = LexState()

    def lex(self) -> LexResult | None:
        advance_table = self.lex_tables.advance_table
        accept_table = self.lex_tables.accept_table
        eof_table = self.lex_tables.eof_table

        end_symbol = self.special_symbols.end_symbol
        end_of_nonterminal_extra_symbol = self.special_symbols.end_of_nonterminal_extra_symbol

        plus_state = self.state.plus
        minus_state = self.state.minus

        lex_result = None

        while character := self._buffer.read():
            if plus_state is not None:
                plus_state = advance_table.get((plus_state, character))

            if minus_state is not None:
                minus_state = advance_table.get((minus_state, character))

            if plus_state is None and minus_state is None:
                break

            plus_symbol = accept_table.get(plus_state) if plus_state is not None else None
            minus_symbol = accept_table.get(minus_state) if minus_state is not None else None

            lex_symbol = plus_symbol or minus_symbol
            lex_result = LexResult(lex_symbol, self._buffer.position) if lex_symbol is not None else None

        if self._buffer.at_eof:
            if not lex_result:
                eof_symbol = eof_table.get(plus_state) if plus_state is not None else None
                lex_result = LexResult(eof_symbol, self._buffer.position) if eof_symbol is not None else None

        elif lex_result and lex_result.symbol == end_symbol:
            lex_result = replace(lex_result, symbol=end_of_nonterminal_extra_symbol)

        self.state = LexState(plus_state, minus_state)

        return lex_result
