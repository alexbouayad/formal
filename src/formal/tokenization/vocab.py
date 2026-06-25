from typing import Literal

from formal.config import FormalTokenPlacement, FormalizationConfig
from formal.language import get_ts_language


def format_formal_token(name: str, *, token_type: Literal["rule", "field"], tag_type: Literal["open", "close"]):
    match token_type:
        case "rule":
            left_bracket = "<"
            tag_right_bracket = ">"

        case "field":
            left_bracket = "["
            tag_right_bracket = "]"

    match tag_type:
        case "open":
            return f"{left_bracket}{name}{tag_right_bracket}"

        case "close":
            return f"{left_bracket}/{name}{tag_right_bracket}"


def count_formal_tokens(formalization_config: FormalizationConfig) -> int:

    if formalization_config.token_placement == FormalTokenPlacement.OMIT:
        return 0

    ts_language = get_ts_language(formalization_config.language)
    count = sum(ts_language.node_kind_is_named(node_id) for node_id in range(ts_language.node_kind_count))

    if formalization_config.include_fields:
        count += ts_language.field_count

    if formalization_config.token_placement == FormalTokenPlacement.ENCLOSING:
        count *= 2

    return count


def compile_formal_tokens(formalization_config: FormalizationConfig) -> list[str]:
    if formalization_config.token_placement == FormalTokenPlacement.OMIT:
        return []

    formal_tokens: list[str] = []
    ts_language = get_ts_language(formalization_config.language)

    def add_token(name: str, *, token_type: Literal["rule", "field"]) -> None:
        if formalization_config.token_placement in (FormalTokenPlacement.PREFIX, FormalTokenPlacement.ENCLOSING):
            open_token = format_formal_token(name, token_type=token_type, tag_type="open")
            formal_tokens.append(open_token)

        if formalization_config.token_placement in (FormalTokenPlacement.POSTFIX, FormalTokenPlacement.ENCLOSING):
            close_token = format_formal_token(name, token_type=token_type, tag_type="close")
            formal_tokens.append(close_token)

    node_id = 0

    while (rule_name := ts_language.node_kind_for_id(node_id)) is not None:
        if ts_language.node_kind_is_named(node_id):
            add_token(rule_name, token_type="rule")

        node_id += 1

    if formalization_config.include_fields:
        field_id = 1

        while (field_name := ts_language.field_name_for_id(field_id)) is not None:
            add_token(field_name, token_type="field")

            field_id += 1

    return sorted(formal_tokens)
