from collections.abc import Callable, Sequence
from contextvars import ContextVar
from ctypes import CDLL, POINTER, Array, byref, c_bool, c_char, c_uint, c_void_p, create_string_buffer
from typing import Final, override

from greenlet import GreenletExit, greenlet

from formal.engine.buffer import ReadOnlyBuffer, TextPosition
from formal.grammar.types import ExternalState, Symbol
from formal.language import LanguageInfo

from ._ts_lexer import (
    ADVANCE_FUNC,
    EOF_FUNC,
    GET_COLUMN_FUNC,
    IS_AT_INCLUDED_RANGE_START_FUNC,
    MARK_END_FUNC,
    TS_SERIALIZATION_BUFFER_SIZE,
    TSLexer,
)
from .interface import Scanner, ScanResult, ScanState
from .scanlet import Scanlet

_mainlet: Final = greenlet.getcurrent()


# TODO: try using a resource pool for ts_scanner
class ExternalScanner(Scanner):
    __slots__ = (
        "language_info",
        "external_symbols",
        "_buffer",
        "_c_buffer",
        "_ts_lexer",
        "_ts_methods",
        "_scanner_create",
        "_scanner_destroy",
        "_scanner_serialize",
        "_scanner_deserialize",
        "_scanner_scan",
        "_exited",
        "_end_position",
        "state",
        "external_state",
    )

    language_info: Final[LanguageInfo]
    external_symbols: Final[Sequence[Symbol]]
    _buffer: Final[ReadOnlyBuffer]

    _c_buffer: Final[Array[c_char]]
    _ts_lexer: Final[TSLexer]
    _ts_methods: Final[tuple[object, ...]]

    _scanner_create: Final[Callable[[], c_void_p]]
    _scanner_destroy: Final[Callable[[c_void_p], None]]
    _scanner_serialize: Final[Callable[[c_void_p, Array[c_char]], int]]
    _scanner_deserialize: Final[Callable[[c_void_p, bytes, int], None]]
    _scanner_scan: Final[Callable[[c_void_p, object, Array[c_bool]], bool]]

    _exited: Final[ContextVar[bool]]
    _end_position: Final[ContextVar[TextPosition]]

    state: Scanlet | ScanState | None
    external_state: ExternalState | None

    def __init__(
        self,
        *,
        buffer: ReadOnlyBuffer,
        language_info: LanguageInfo,
        external_symbols: Sequence[Symbol],
    ) -> None:
        cdll = CDLL(language_info.ts_scanner_path)

        ts_advance = ADVANCE_FUNC(self._ts_advance)
        ts_mark_end = MARK_END_FUNC(self._ts_mark_end)
        ts_get_column = GET_COLUMN_FUNC(self._ts_get_column)
        ts_is_at_start = IS_AT_INCLUDED_RANGE_START_FUNC(self._ts_is_at_included_range_start)
        ts_eof = EOF_FUNC(self._ts_eof)

        ts_lexer = TSLexer()
        ts_lexer.advance = ts_advance
        ts_lexer.mark_end = ts_mark_end
        ts_lexer.get_column = ts_get_column
        ts_lexer.is_at_included_range_start = ts_is_at_start
        ts_lexer.eof = ts_eof

        scanner_create = cdll[f"{language_info.ts_module_name}_external_scanner_create"]
        scanner_destroy = cdll[f"{language_info.ts_module_name}_external_scanner_destroy"]
        scanner_serialize = cdll[f"{language_info.ts_module_name}_external_scanner_serialize"]
        scanner_deserialize = cdll[f"{language_info.ts_module_name}_external_scanner_deserialize"]
        scanner_scan = cdll[f"{language_info.ts_module_name}_external_scanner_scan"]

        scanner_create.restype = c_void_p
        scanner_create.argtypes = []

        scanner_destroy.restype = None
        scanner_destroy.argtypes = [c_void_p]

        scanner_serialize.restype = c_uint
        scanner_serialize.argtypes = [c_void_p, POINTER(c_char)]

        scanner_deserialize.restype = None
        scanner_deserialize.argtypes = [c_void_p, POINTER(c_char), c_uint]

        scanner_scan.restype = c_bool
        scanner_scan.argtypes = [c_void_p, POINTER(TSLexer), POINTER(c_bool)]

        self._buffer = buffer
        self.language_info = language_info
        self.external_symbols = external_symbols

        self.state = None
        self.external_state = None

        self._c_buffer = create_string_buffer(TS_SERIALIZATION_BUFFER_SIZE)
        self._ts_lexer = ts_lexer
        self._ts_methods = ts_advance, ts_mark_end, ts_get_column, ts_is_at_start, ts_eof

        self._scanner_create = scanner_create
        self._scanner_destroy = scanner_destroy
        self._scanner_serialize = scanner_serialize
        self._scanner_deserialize = scanner_deserialize
        self._scanner_scan = scanner_scan

        self._exited = ContextVar("exited", default=False)
        self._end_position = ContextVar("end_position")

    def __repr__(self) -> str:
        if self.external_state is None:
            symbols = ()

        else:
            symbols = tuple(symbol for symbol, is_valid in zip(self.external_symbols, self.external_state) if is_valid)

        return f"<{self.__class__.__name__}: state={self.state}, symbols={symbols}>"

    @override
    def __bool__(self) -> bool:
        return True

    @override
    def reset(self) -> None:
        self.state = None
        self.external_state = None

    @override
    def scan(self) -> ScanResult | None:
        if isinstance(self.state, Scanlet):
            scanlet = self.state

        else:
            scanlet = Scanlet(start_state=self.state, start_position=self._buffer.position, run=self._run)

        return scanlet.switch()

    def _deserialize(self, ts_scanner_ptr: c_void_p, scan_state: ScanState, /) -> None:
        self._scanner_deserialize(ts_scanner_ptr, scan_state, len(scan_state))

    def _run(self, start_state: ScanState | None) -> ScanResult | None:
        self._update_ts_lookahead()

        self.state = Scanlet.get_current()

        if self.external_state is None:
            self.external_state = (c_bool * len(self.external_symbols))()

        ts_scanner_ptr = self._scanner_create()

        try:
            if start_state is not None:
                self._deserialize(ts_scanner_ptr, start_state)

            match = self._scanner_scan(ts_scanner_ptr, byref(self._ts_lexer), self.external_state)
            end_state = self._serialize(ts_scanner_ptr)

        finally:
            self._scanner_destroy(ts_scanner_ptr)

        self.state = end_state

        if match:
            symbol = self.external_symbols[self._ts_lexer.result_symbol]
            scan_result = ScanResult(symbol, self._end_position.get())

        else:
            scan_result = None

        return scan_result

    def _serialize(self, ts_scanner_ptr: c_void_p) -> ScanState:
        num_bytes = self._scanner_serialize(ts_scanner_ptr, self._c_buffer)
        raw_bytes = self._c_buffer.raw[:num_bytes]

        return raw_bytes

    def _update_ts_lookahead(self) -> None:
        if self._buffer.at_eof or self._exited.get():
            ts_lookahead = 0

        else:
            lookahead = self._buffer.lookahead
            ts_lookahead = ord(lookahead) if lookahead else 0

        self._ts_lexer.lookahead = ts_lookahead

    def _ts_advance(self, ts_lexer_ptr: object, skip: bool) -> None:
        buffer = self._buffer
        buffer.read()

        while not buffer.lookahead and not buffer.at_eof:
            external_state = self.external_state
            ts_lexer_result_symbol = self._ts_lexer.result_symbol

            try:
                _mainlet.switch()

            except GreenletExit:
                self._exited.set(True)
                break

            self.state = Scanlet.get_current()
            self.external_state = external_state
            self._ts_lexer.result_symbol = ts_lexer_result_symbol

        self._update_ts_lookahead()

    def _ts_mark_end(self, ts_lexer_ptr: object) -> None:
        self._end_position.set(self._buffer.position)

    def _ts_get_column(self, ts_lexer_ptr: object) -> int:
        return self._buffer.column

    def _ts_is_at_included_range_start(self, ts_lexer_ptr: object) -> bool:
        return False

    def _ts_eof(self, ts_lexer_ptr: object) -> bool:
        return self._exited.get() or self._buffer.at_eof
