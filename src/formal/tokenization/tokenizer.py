from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Final, Literal, Self, cast, overload

from formal.config import FormalTokenPlacement, FormalizationConfig
from formal.language import get_ts_language, get_ts_parser

from .encoding import FormalEncoding, TokenTypeId
from .vocab import compile_formal_tokens, format_formal_token

if TYPE_CHECKING:
    import tree_sitter as ts
    from transformers import TokenizersBackend


class FormalTokenizer:
    text_backend: Final["TokenizersBackend"]
    formal_vocab: Final[Mapping[str, int]]
    formalization_config: Final[FormalizationConfig]

    _ts_parser: Final["ts.Parser"]
    _ts_language: Final["ts.Language"]
    _formal_decoder: Final[Mapping[int, str]]

    _encoding: FormalEncoding
    _source_bytes: bytes

    _ast_path: list[int]
    _byte_offset: int

    @property
    def text_vocab(self) -> Mapping[str, int]:
        return self.text_backend.get_vocab()

    def __init__(self, text_backend: "TokenizersBackend", formalization_config: FormalizationConfig) -> None:
        formal_tokens = compile_formal_tokens(formalization_config)

        formal_vocab = {
            formal_token: formal_token_id + text_backend.vocab_size
            for formal_token_id, formal_token in enumerate(formal_tokens)
        }

        formal_decoder = {formal_token_id: formal_token for formal_token, formal_token_id in formal_vocab.items()}

        self.text_backend = text_backend
        self.formal_vocab = formal_vocab
        self.formalization_config = formalization_config

        self._ts_parser = get_ts_parser(formalization_config.language)
        self._ts_language = get_ts_language(formalization_config.language)
        self._formal_decoder = formal_decoder

        self._setup()

    def __reduce__(self) -> "tuple[type[Self], tuple[TokenizersBackend, FormalizationConfig]]":
        return (self.__class__, (self.text_backend, self.formalization_config))

    @overload
    def __call__(self, source: str, *, return_as_dict: Literal[True]) -> dict[str, Any]: ...
    @overload
    def __call__(self, source: str, *, return_as_dict: Literal[False] = False) -> FormalEncoding: ...

    def __call__(self, source: str, *, return_as_dict: bool = False) -> FormalEncoding | dict[str, Any]:
        source_bytes = source.encode()

        try:
            encoding = self._encode(source_bytes)

            if encoding is None:
                encoding = FormalEncoding()

            if return_as_dict:
                return asdict(encoding)

            return encoding

        finally:
            self._setup()

    def decode(self, encoding: FormalEncoding) -> str:
        source = ""

        for token_id, token_type_id in zip(encoding.token_ids, encoding.token_type_ids):
            if token_type_id == TokenTypeId.TEXT:
                token = self.text_backend.decode(token_id, clean_up_tokenization_spaces=False)  # type: ignore
                token = cast(str, token)

            else:
                token = self._formal_decoder[token_id]

            source += token

        return source

    def print(self, encoding: FormalEncoding, *, start: int = 0, stop: int | None = None) -> None:
        for token_position, (token_id, token_type_id, token_ast_depth) in enumerate(encoding):
            if token_position < start:
                continue

            if stop is not None and token_position >= stop:
                break

            indent = " " * int(2 * token_ast_depth)
            decoded_token = self._decode(token_id, token_type_id)

            if token_type_id == TokenTypeId.TEXT:
                decoded_token = repr(decoded_token)

            print(indent + decoded_token)

    def _decode(self, token_id: int, token_type_id: TokenTypeId) -> str:
        if token_type_id == TokenTypeId.TEXT:
            decoded_token = self.text_backend.decode(token_id, clean_up_tokenization_spaces=False)  # type: ignore
            return cast(str, decoded_token)

        else:
            return self._formal_decoder[token_id]

    def _encode(self, source_bytes: bytes) -> FormalEncoding | None:
        ts_tree = self._ts_parser.parse(source_bytes)

        if ts_tree.root_node.has_error:
            return None

        self._source_bytes = source_bytes

        ts_cursor = ts_tree.walk()
        traversing_down = True

        while True:
            if ts_cursor.node is None:
                return None

            if traversing_down:
                self._pre_visit(ts_cursor.node)

                if ts_cursor.goto_first_child():
                    self._ast_path.append(0)

                else:
                    traversing_down = False

            else:
                self._post_visit(ts_cursor.node)

                if ts_cursor.goto_next_sibling():
                    self._ast_path[-1] += 1
                    traversing_down = True

                elif ts_cursor.goto_parent():
                    self._ast_path.pop()

                else:
                    break

        return self._encoding

    def _setup(self):
        self._encoding = FormalEncoding()
        self._source_bytes = b""

        self._ast_path = [0]
        self._byte_offset = 0

    def _post_visit(self, ts_node: "ts.Node") -> None:

        if ts_node.is_named:
            self._ast_path.append(0)

        node_postfix = self._source_bytes[self._byte_offset : ts_node.end_byte].decode()

        text_tokens = self.text_backend.tokenize(node_postfix, split_special_tokens=True)  # type: ignore
        token_type_ids = [TokenTypeId.TEXT] * len(text_tokens)

        self._append(text_tokens, token_type_ids)

        if ts_node.is_named:
            self._ast_path.pop()

        if self.formalization_config.token_placement in (FormalTokenPlacement.POSTFIX, FormalTokenPlacement.ENCLOSING):
            formal_tokens, token_type_ids = self._tokenize(ts_node, tag_type="close")
            self._append(formal_tokens, token_type_ids)

        self._byte_offset = ts_node.end_byte

    def _pre_visit(self, ts_node: "ts.Node") -> None:
        node_prefix = self._source_bytes[self._byte_offset : ts_node.start_byte].decode()

        text_tokens = self.text_backend.tokenize(node_prefix, split_special_tokens=True)  # type: ignore
        token_type_ids = [TokenTypeId.TEXT] * len(text_tokens)

        self._append(text_tokens, token_type_ids)

        if self.formalization_config.token_placement in (FormalTokenPlacement.PREFIX, FormalTokenPlacement.ENCLOSING):
            formal_tokens, token_type_ids = self._tokenize(ts_node, tag_type="open")
            self._append(formal_tokens, token_type_ids)

        self._byte_offset = ts_node.start_byte

    def _append(self, tokens: Sequence[str], token_type_ids: Sequence[TokenTypeId]) -> None:
        if not tokens:
            return

        encoding = self._encoding
        ast_depth = len(self._ast_path)

        token_ids = (
            cast(int, self.text_backend.convert_tokens_to_ids(token))  # type: ignore
            if token_type_id == TokenTypeId.TEXT
            else self.formal_vocab[token]
            for token, token_type_id in zip(tokens, token_type_ids)
        )

        encoding.token_ids += token_ids
        encoding.token_type_ids += token_type_ids
        encoding.token_ast_depths += (ast_depth,) * len(tokens)

    def _tokenize(
        self,
        ts_node: "ts.Node",
        *,
        tag_type: Literal["open", "close"],
    ) -> tuple[Sequence[str], Sequence[TokenTypeId]]:
        formal_tokens: list[str] = []
        token_type_ids: list[TokenTypeId] = []

        if self.formalization_config.include_fields and ts_node.parent is not None:
            child_index = self._ast_path[-1]
            field_name = ts_node.parent.field_name_for_child(child_index)

            if field_name is not None:
                field_token = format_formal_token(field_name, token_type="field", tag_type=tag_type)

                formal_tokens.append(field_token)
                token_type_ids.append(TokenTypeId.FORMAL_FIELD)

        if ts_node.is_named:
            rule_name = ts_node.type
            rule_token = format_formal_token(rule_name, token_type="rule", tag_type=tag_type)

            formal_tokens.append(rule_token)
            token_type_ids.append(TokenTypeId.FORMAL_RULE)

        if tag_type == "close":
            formal_tokens.reverse()
            token_type_ids.reverse()

        return formal_tokens, token_type_ids
