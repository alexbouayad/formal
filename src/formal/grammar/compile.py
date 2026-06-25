from collections.abc import Mapping

from formal.language import LanguageInfo

from ._builder import GrammarBuilder
from .types import Grammar


def compile_grammar(
    model_vocab: Mapping[str, int],
    language_info: LanguageInfo,
    *,
    extend_grammar: bool = False,
    minimize_lexer: bool = True,
    minimize_parser: bool = False,
) -> Grammar:
    import tree_sitter_generate as tsg

    tsg_grammar = tsg.Grammar(
        language_info.grammar_path,
        extend_grammar=extend_grammar,
        minimize_lexer=minimize_lexer,
        minimize_parser=minimize_parser,
    )

    builder = GrammarBuilder(tsg_grammar, model_vocab)

    return builder.build()
