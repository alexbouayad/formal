from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tree_sitter as ts


class Language(StrEnum):
    BASH = "bash"
    C = "c"
    CPP = "cpp"
    C_SHARP = "c_sharp"
    GO = "go"
    HASKELL = "haskell"
    HTML = "html"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    JSON = "json"
    JULIA = "julia"
    OCAML = "ocaml"
    PHP = "php"
    PYTHON = "python"
    REGEX = "regex"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    TYPESCRIPT = "typescript"
    VERILOG = "verilog"


@dataclass(frozen=True, kw_only=True, slots=True)
class LanguageInfo:
    hf_name: str
    ts_name: str

    @property
    def ts_module_name(self) -> str:
        return f"tree_sitter_{self.ts_name}"

    @property
    def grammar_path(self) -> str:
        return f"./ts/{self.ts_name}/src/grammar.json"

    @property
    def scanner_path(self) -> str:
        return f"./ts/{self.ts_name}/src/scanner.so"


def get_language_info(language: Language) -> LanguageInfo:
    match language:
        case Language.BASH:
            return LanguageInfo(hf_name="Shell", ts_name="bash")
        case Language.C:
            return LanguageInfo(hf_name="C", ts_name="c")
        case Language.CPP:
            return LanguageInfo(hf_name="C++", ts_name="cpp")
        case Language.C_SHARP:
            return LanguageInfo(hf_name="C-Sharp", ts_name="c_sharp")
        case Language.GO:
            return LanguageInfo(hf_name="Go", ts_name="go")
        case Language.HASKELL:
            return LanguageInfo(hf_name="Haskell", ts_name="haskell")
        case Language.HTML:
            return LanguageInfo(hf_name="HTML", ts_name="html")
        case Language.JAVA:
            return LanguageInfo(hf_name="Java", ts_name="java")
        case Language.JAVASCRIPT:
            return LanguageInfo(hf_name="JavaScript", ts_name="javascript")
        case Language.JSON:
            return LanguageInfo(hf_name="JSON", ts_name="json")
        case Language.JULIA:
            return LanguageInfo(hf_name="Julia", ts_name="julia")
        case Language.OCAML:
            return LanguageInfo(hf_name="OCaml", ts_name="ocaml")
        case Language.PHP:
            return LanguageInfo(hf_name="PHP", ts_name="php")
        case Language.PYTHON:
            return LanguageInfo(hf_name="Python", ts_name="python")
        case Language.REGEX:
            return LanguageInfo(hf_name="Regular_Expression", ts_name="regex")
        case Language.RUBY:
            return LanguageInfo(hf_name="Ruby", ts_name="ruby")
        case Language.RUST:
            return LanguageInfo(hf_name="Rust", ts_name="rust")
        case Language.SCALA:
            return LanguageInfo(hf_name="Scala", ts_name="scala")
        case Language.TYPESCRIPT:
            return LanguageInfo(hf_name="TypeScript", ts_name="typescript")
        case Language.VERILOG:
            return LanguageInfo(hf_name="Verilog", ts_name="verilog")


@cache
def get_ts_language(language: Language) -> "ts.Language":
    import tree_sitter as ts

    language_info = get_language_info(language)
    ts_module = import_module(language_info.ts_module_name)

    ts_language_function = getattr(ts_module, "language", None)

    if ts_language_function is None:
        ts_language_function = getattr(ts_module, f"language_{language_info.ts_name}")

    ts_language_object: object = ts_language_function()
    ts_language = ts.Language(ts_language_object)

    return ts_language


@cache
def get_ts_parser(language: Language) -> "ts.Parser":
    import tree_sitter as ts

    ts_language = get_ts_language(language)
    ts_parser = ts.Parser(ts_language)

    return ts_parser
