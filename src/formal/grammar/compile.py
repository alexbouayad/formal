from collections.abc import Iterable

from ._builder import GrammarBuilder
from .types import Grammar


def compile_grammar(
    path: str,
    text_tokens: Iterable[str],
    *,
    extend_grammar: bool = False,
    minimize_parser: bool = False,
) -> Grammar:
    import ts_generate as tsg

    tsg_grammar = tsg.Grammar(path, extend_grammar=extend_grammar, minimize_parser=minimize_parser)
    builder = GrammarBuilder(tsg_grammar, text_tokens)

    return builder.build()
