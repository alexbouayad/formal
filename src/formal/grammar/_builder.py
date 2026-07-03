from collections import defaultdict
from collections.abc import Iterable
from ctypes import c_bool
from functools import cache

import ts_generate as tsg

from .types import (
    AcceptAction,
    DFAState,
    Grammar,
    LexMode,
    LexState,
    LexTables,
    ParseAction,
    ParseState,
    ParseTables,
    RecoverAction,
    ReduceAction,
    ShiftAction,
    Symbol,
)


class GrammarBuilder:
    __slots__ = ("ts_grammar", "text_tokens")

    ts_grammar: tsg.Grammar
    text_tokens: Iterable[str]

    def __init__(self, tsg_grammar: tsg.Grammar, text_tokens: Iterable[str]) -> None:
        self.ts_grammar = tsg_grammar
        self.text_tokens = text_tokens

    def build(self) -> Grammar:
        lex_tables = self._build_lex_tables()
        parse_tables = self._build_parse_tables()

        terminal_symbols = [variable.name for variable in self.ts_grammar.lexical_grammar.variables]
        nonterminal_symbols = [variable.name for variable in self.ts_grammar.syntax_grammar.variables]
        external_symbols = [variable.name for variable in self.ts_grammar.syntax_grammar.external_tokens]

        # TODO: define end symbols in ts_generate and convert using _convert
        return Grammar(
            lex_tables=lex_tables,
            parse_tables=parse_tables,
            terminal_symbols=terminal_symbols,
            nonterminal_symbols=nonterminal_symbols,
            external_symbols=external_symbols,
            end_symbol="<END>",
            end_of_nonterminal_extra_symbol="<END_OF_NONTERMINAL_EXTRA>",
        )

    def _build_lex_tables(self) -> LexTables:
        advance_table: dict[tuple[DFAState, str], DFAState] = {}
        accept_table: dict[DFAState, Symbol | None] = {}
        eof_table: dict[DFAState, Symbol | None] = {}

        ts_lex_states = self.ts_grammar.tables.main_lex_table.states
        vocab_characters = {character for token in self.text_tokens for character in token}

        for ts_lex_state_id, ts_lex_state in enumerate(ts_lex_states):
            for character_set, advance_action in ts_lex_state.advance_actions:
                for char_first, char_last in character_set.ranges:
                    for i in range(ord(char_first), ord(char_last) + 1):
                        character = chr(i)

                        if character in vocab_characters:
                            advance_table[ts_lex_state_id, character] = advance_action.state

            if ts_lex_state.accept_action is not None:
                accept_table[ts_lex_state_id] = self._convert(ts_lex_state.accept_action)

            if ts_lex_state.eof_action is not None:
                eof_ts_lex_state = ts_lex_states[ts_lex_state.eof_action.state]

                if eof_ts_lex_state.accept_action is not None:
                    eof_table[ts_lex_state_id] = self._convert(eof_ts_lex_state.accept_action)

        return LexTables(advance_table, accept_table, eof_table)

    def _build_parse_action_table(self) -> dict[tuple[ParseState, Symbol], list[ParseAction]]:
        action_table: dict[tuple[ParseState, Symbol], list[ParseAction]] = defaultdict(list)

        ts_parse_states = self.ts_grammar.tables.parse_table.states

        for parse_state, ts_parse_state in enumerate(ts_parse_states):
            for ts_terminal_symbol, ts_table_entry in ts_parse_state.terminal_entries.items():
                terminal_symbol = self._convert(ts_terminal_symbol)

                for ts_parse_action in ts_table_entry.actions:
                    match ts_parse_action:
                        case tsg.ParseAction.Accept():
                            parse_action = AcceptAction()

                        case tsg.ParseAction.Recover():
                            parse_action = RecoverAction()

                        # TODO: remove precedence and production fields from ts.ParseAction.Reduce
                        case tsg.ParseAction.Reduce(ts_nonterminal, child_count, _):
                            ts_nonterminal_symbol = self._convert(ts_nonterminal)
                            parse_action = ReduceAction(symbol=ts_nonterminal_symbol, child_count=child_count)

                        case tsg.ParseAction.Shift(goto_state, _):
                            if ts_parse_action.is_repetition:
                                continue

                            parse_action = ShiftAction(goto_state)

                        case tsg.ParseAction.ShiftExtra():
                            parse_action = ShiftAction(None)

                        case _:
                            continue

                    action_table[parse_state, terminal_symbol].append(parse_action)

        return dict(action_table)

    def _build_goto_table(self) -> dict[tuple[ParseState, Symbol], ParseState | None]:
        goto_table: dict[tuple[ParseState, Symbol], ParseState | None] = {}

        ts_parse_states = self.ts_grammar.tables.parse_table.states

        for parse_state, ts_parse_state in enumerate(ts_parse_states):
            for ts_nonterminal_symbol, ts_goto_action in ts_parse_state.nonterminal_entries.items():
                nonterminal_symbol = self._convert(ts_nonterminal_symbol)

                match ts_goto_action:
                    case tsg.GotoAction.Goto(goto_state):
                        pass

                    case tsg.GotoAction.ShiftExtra():
                        goto_state = None

                    case _:
                        continue

                goto_table[parse_state, nonterminal_symbol] = goto_state

        return goto_table

    def _build_lex_modes(self) -> dict[ParseState, LexMode]:
        lex_modes: dict[ParseState, LexMode] = {}

        ts_parse_states = self.ts_grammar.tables.parse_table.states
        ts_external_states = self.ts_grammar.tables.parse_table.external_lex_states
        ts_external_symbols = self.ts_grammar.syntax_grammar.external_tokens

        array_type = c_bool * len(ts_external_symbols)

        for parse_state, ts_parse_state in enumerate(ts_parse_states):
            lex_state = LexState(ts_parse_state.lex_state_id, ts_parse_state.anti_lex_state_id)
            external_state = array_type()

            ts_external_lex_state_id = ts_parse_state.external_lex_state_id
            ts_valid_external_symbols = ts_external_states[ts_external_lex_state_id].tokens

            for ts_external_symbol in ts_valid_external_symbols:
                external_state[ts_external_symbol.index] = True

            lex_modes[parse_state] = LexMode(lex_state, external_state)

        return lex_modes

    def _build_parse_tables(self) -> ParseTables:
        action_table = self._build_parse_action_table()
        goto_table = self._build_goto_table()
        lex_modes = self._build_lex_modes()

        return ParseTables(action_table, goto_table, lex_modes)

    @cache
    def _convert(self, tsg_symbol: tsg.Symbol) -> Symbol:
        match tsg_symbol.kind:
            case tsg.SymbolType.End:
                symbol = "<END>"

            case tsg.SymbolType.EndOfNonTerminalExtra:
                symbol = "<END_OF_NONTERMINAL_EXTRA>"

            case tsg.SymbolType.External:
                symbol = self.ts_grammar.syntax_grammar.external_tokens[tsg_symbol.index].name

            case tsg.SymbolType.NonTerminal:
                symbol = self.ts_grammar.syntax_grammar.variables[tsg_symbol.index].name

            case tsg.SymbolType.Terminal:
                lexical_variable = self.ts_grammar.lexical_grammar.variables[tsg_symbol.index]

                match lexical_variable.kind:
                    case tsg.VariableType.Anonymous:
                        symbol = rf"{lexical_variable.name!r}"

                    case _:
                        symbol = lexical_variable.name

            case _:
                symbol = "<UNKNOWN>"

        return symbol
