from collections import defaultdict
from collections.abc import Mapping
from ctypes import c_bool
from functools import cache

import tree_sitter_generate as tsg

from .types import (
    AcceptAction,
    DFAState,
    EndSymbols,
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
    __slots__ = ("tsg_grammar", "model_vocab")

    tsg_grammar: tsg.Grammar
    model_vocab: Mapping[str, int]

    def __init__(self, tsg_grammar: tsg.Grammar, model_vocab: Mapping[str, int]) -> None:
        self.tsg_grammar = tsg_grammar
        self.model_vocab = model_vocab

    def build(self) -> Grammar:
        lex_tables = self._build_lex_tables()
        parse_tables = self._build_parse_tables()

        terminal_symbols = [self._convert(ts_symbol) for ts_symbol in self.tsg_grammar.lexical_grammar.variables]
        nonterminal_symbols = [self._convert(ts_symbol) for ts_symbol in self.tsg_grammar.syntax_grammar.variables]
        external_symbols = [self._convert(ts_symbol) for ts_symbol in self.tsg_grammar.syntax_grammar.external_tokens]

        # TODO: define end symbols in tree_sitter_generate and convert using _convert
        end_symbols = EndSymbols(
            end_symbol=Symbol("<END>"),
            end_of_nonterminal_extra_symbol=Symbol("<END_OF_NONTERMINAL_EXTRA>"),
        )

        return Grammar(
            lex_tables=lex_tables,
            parse_tables=parse_tables,
            terminal_symbols=terminal_symbols,
            nonterminal_symbols=nonterminal_symbols,
            external_symbols=external_symbols,
            end_symbols=end_symbols,
        )

    def _build_lex_tables(self) -> LexTables:
        advance_table: dict[tuple[DFAState, str], DFAState] = {}
        accept_table: dict[DFAState, Symbol | None] = {}
        eof_table: dict[DFAState, Symbol | None] = {}

        ts_lex_states = self.tsg_grammar.tables.main_lex_table.states
        vocab_characters = {character for token in self.model_vocab for character in token}

        for dfa_state, ts_lex_state in enumerate(ts_lex_states):
            for character_set, advance_action in ts_lex_state.advance_actions:
                for char_first, char_last in character_set.ranges:
                    for i in range(ord(char_first), ord(char_last) + 1):
                        character = chr(i)

                        if character in vocab_characters:
                            advance_table[dfa_state, character] = advance_action.state

            if ts_lex_state.accept_action is not None:
                accept_table[dfa_state] = self._convert(ts_lex_state.accept_action)

            if ts_lex_state.eof_action is not None:
                eof_table[dfa_state] = self._convert(ts_lex_state.eof_action)

        return LexTables(advance_table, accept_table, eof_table)

    def _build_parse_action_table(self) -> dict[tuple[ParseState, Symbol], list[ParseAction]]:
        action_table: dict[tuple[ParseState, Symbol], list[ParseAction]] = defaultdict(list)

        ts_parse_states = self.tsg_grammar.tables.parse_table.states

        for parse_state, ts_parse_state in enumerate(ts_parse_states):
            for ts_terminal_symbol, ts_table_entry in ts_parse_state.terminal_entries.items():
                terminal_symbol = self._convert(ts_terminal_symbol)

                for ts_parse_action in ts_table_entry.actions:
                    match ts_parse_action:
                        case ts.ParseAction.Accept():
                            parse_action = AcceptAction()

                        case ts.ParseAction.Recover():
                            parse_action = RecoverAction()

                        # TODO: remove precedence and production fields from ts.ParseAction.Reduce
                        case ts.ParseAction.Reduce(ts_nonterminal, child_count, _):
                            ts_nonterminal_symbol = self._convert(ts_nonterminal)
                            parse_action = ReduceAction(symbol=ts_nonterminal_symbol, child_count=child_count)

                        case ts.ParseAction.Shift(goto_state, _):
                            if ts_parse_action.is_repetition:
                                continue

                            parse_action = ShiftAction(goto_state)

                        case ts.ParseAction.ShiftExtra():
                            parse_action = ShiftAction(None)

                        case _:
                            continue

                    action_table[parse_state, terminal_symbol].append(parse_action)

        return action_table

    def _build_goto_table(self) -> dict[tuple[ParseState, Symbol], ParseState | None]:
        goto_table: dict[tuple[ParseState, Symbol], ParseState | None] = {}

        ts_parse_states = self.tsg_grammar.tables.parse_table.states

        for parse_state, ts_parse_state in enumerate(ts_parse_states):
            for ts_nonterminal_symbol, ts_goto_action in ts_parse_state.nonterminal_entries.items():
                nonterminal_symbol = self._convert(ts_nonterminal_symbol)

                match ts_goto_action:
                    case ts.GotoAction.Goto(goto_state):
                        pass

                    case ts.GotoAction.ShiftExtra():
                        goto_state = None

                    case _:
                        continue

                goto_table[parse_state, nonterminal_symbol] = goto_state

        return goto_table

    def _build_lex_modes(self) -> dict[ParseState, LexMode]:
        lex_modes: dict[ParseState, LexMode] = {}

        ts_parse_states = self.tsg_grammar.tables.parse_table.states
        ts_external_states = self.tsg_grammar.tables.parse_table.external_lex_states
        ts_external_symbols = self.tsg_grammar.syntax_grammar.external_tokens

        array_type = c_bool * len(ts_external_symbols)

        for parse_state, ts_parse_state in enumerate(ts_parse_states):
            lex_state = LexState(ts_parse_state.plus_dfa_state, ts_parse_state.minus_dfa_state)
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
    def _convert(self, symbol: tsg.Symbol) -> Symbol:
        match symbol.kind:
            case ts.SymbolType.End:
                symbol_name = "<END>"

            case ts.SymbolType.EndOfNonTerminalExtra:
                symbol_name = "<END_OF_NONTERMINAL_EXTRA>"

            case ts.SymbolType.External:
                symbol_name = self.tsg_grammar.syntax_grammar.external_tokens[symbol.index].name

            case ts.SymbolType.NonTerminal:
                symbol_name = self.tsg_grammar.syntax_grammar.variables[symbol.index].name

            case ts.SymbolType.Terminal:
                lexical_variable = self.tsg_grammar.lexical_grammar.variables[symbol.index]

                match lexical_variable.kind:
                    case ts.VariableType.Anonymous:
                        symbol_name = rf"{lexical_variable.name!r}"

                    case _:
                        symbol_name = lexical_variable.name

            case _:
                symbol_name = "<UNKNOWN>"

        return Symbol(symbol_name)
