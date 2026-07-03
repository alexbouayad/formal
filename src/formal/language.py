from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from importlib import import_module, resources
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
    ts_module_name: str
    ts_grammar_dir: str

    @property
    def ts_grammar_path(self) -> str:
        return f"{self.ts_grammar_dir}/grammar.json"

    @property
    def ts_scanner_path(self) -> str:
        return f"{self.ts_grammar_dir}/scanner.so"


def get_language_info(language: Language) -> LanguageInfo:
    resource_root = resources.files("formal.grammar.tree_sitter")

    match language:
        case Language.BASH:
            return LanguageInfo(
                hf_name="Shell",
                ts_name="bash",
                ts_module_name="tree_sitter_bash",
                ts_grammar_dir=f"{resource_root}/tree-sitter-bash/src",
            )
        case Language.C:
            return LanguageInfo(
                hf_name="C",
                ts_name="c",
                ts_module_name="tree_sitter_c",
                ts_grammar_dir=f"{resource_root}/tree-sitter-c/src",
            )
        case Language.CPP:
            return LanguageInfo(
                hf_name="C++",
                ts_name="cpp",
                ts_module_name="tree_sitter_cpp",
                ts_grammar_dir=f"{resource_root}/tree-sitter-cpp/src",
            )
        case Language.C_SHARP:
            return LanguageInfo(
                hf_name="C-Sharp",
                ts_name="c_sharp",
                ts_module_name="tree_sitter_c_sharp",
                ts_grammar_dir=f"{resource_root}/tree-sitter-c-sharp/src",
            )
        case Language.GO:
            return LanguageInfo(
                hf_name="Go",
                ts_name="go",
                ts_module_name="tree_sitter_go",
                ts_grammar_dir=f"{resource_root}/tree-sitter-go/src",
            )
        case Language.HASKELL:
            return LanguageInfo(
                hf_name="Haskell",
                ts_name="haskell",
                ts_module_name="tree_sitter_haskell",
                ts_grammar_dir=f"{resource_root}/tree-sitter-haskell/src",
            )
        case Language.HTML:
            return LanguageInfo(
                hf_name="HTML",
                ts_name="html",
                ts_module_name="tree_sitter_html",
                ts_grammar_dir=f"{resource_root}/tree-sitter-html/src",
            )
        case Language.JAVA:
            return LanguageInfo(
                hf_name="Java",
                ts_name="java",
                ts_module_name="tree_sitter_java",
                ts_grammar_dir=f"{resource_root}/tree-sitter-java/src",
            )
        case Language.JAVASCRIPT:
            return LanguageInfo(
                hf_name="JavaScript",
                ts_name="javascript",
                ts_module_name="tree_sitter_javascript",
                ts_grammar_dir=f"{resource_root}/tree-sitter-javascript/src",
            )
        case Language.JSON:
            return LanguageInfo(
                hf_name="JSON",
                ts_name="json",
                ts_module_name="tree_sitter_json",
                ts_grammar_dir=f"{resource_root}/tree-sitter-json/src",
            )
        case Language.JULIA:
            return LanguageInfo(
                hf_name="Julia",
                ts_name="julia",
                ts_module_name="tree_sitter_julia",
                ts_grammar_dir=f"{resource_root}/tree-sitter-julia/src",
            )
        case Language.OCAML:
            return LanguageInfo(
                hf_name="OCaml",
                ts_name="ocaml",
                ts_module_name="tree_sitter_ocaml",
                ts_grammar_dir=f"{resource_root}/tree-sitter-ocaml/grammars/ocaml/src",
            )
        case Language.PHP:
            return LanguageInfo(
                hf_name="PHP",
                ts_name="php",
                ts_module_name="tree_sitter_php",
                ts_grammar_dir=f"{resource_root}/tree-sitter-php/php/src",
            )
        case Language.PYTHON:
            return LanguageInfo(
                hf_name="Python",
                ts_name="python",
                ts_module_name="tree_sitter_python",
                ts_grammar_dir=f"{resource_root}/tree-sitter-python/src",
            )
        case Language.REGEX:
            return LanguageInfo(
                hf_name="Regular_Expression",
                ts_name="regex",
                ts_module_name="tree_sitter_regex",
                ts_grammar_dir=f"{resource_root}/tree-sitter-regex/src",
            )
        case Language.RUBY:
            return LanguageInfo(
                hf_name="Ruby",
                ts_name="ruby",
                ts_module_name="tree_sitter_ruby",
                ts_grammar_dir=f"{resource_root}/tree-sitter-ruby/src",
            )
        case Language.RUST:
            return LanguageInfo(
                hf_name="Rust",
                ts_name="rust",
                ts_module_name="tree_sitter_rust",
                ts_grammar_dir=f"{resource_root}/tree-sitter-rust/src",
            )
        case Language.SCALA:
            return LanguageInfo(
                hf_name="Scala",
                ts_name="scala",
                ts_module_name="tree_sitter_scala",
                ts_grammar_dir=f"{resource_root}/tree-sitter-scala/src",
            )
        case Language.TYPESCRIPT:
            return LanguageInfo(
                hf_name="TypeScript",
                ts_name="typescript",
                ts_module_name="tree_sitter_typescript",
                ts_grammar_dir=f"{resource_root}/tree-sitter-typescript/typescript/src",
            )
        case Language.VERILOG:
            return LanguageInfo(
                hf_name="Verilog",
                ts_name="verilog",
                ts_module_name="tree_sitter_verilog",
                ts_grammar_dir=f"{resource_root}/tree-sitter-verilog/src",
            )


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
