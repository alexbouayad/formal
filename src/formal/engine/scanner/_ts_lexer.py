from ctypes import CFUNCTYPE, POINTER, Structure, c_bool, c_int32, c_uint16, c_uint32


class TSLexer(Structure):
    lookahead: int
    result_symbol: int
    advance: object
    mark_end: object
    get_column: object
    is_at_included_range_start: object
    eof: object
    log: object


ADVANCE_FUNC = CFUNCTYPE(None, POINTER(TSLexer), c_bool)
MARK_END_FUNC = CFUNCTYPE(None, POINTER(TSLexer))
GET_COLUMN_FUNC = CFUNCTYPE(c_uint32, POINTER(TSLexer))
IS_AT_INCLUDED_RANGE_START_FUNC = CFUNCTYPE(c_bool, POINTER(TSLexer))
EOF_FUNC = CFUNCTYPE(c_bool, POINTER(TSLexer))
LOG_FUNC = CFUNCTYPE(None, POINTER(TSLexer))


TSLexer._fields_ = [
    ("lookahead", c_int32),
    ("result_symbol", c_uint16),
    ("advance", ADVANCE_FUNC),
    ("mark_end", MARK_END_FUNC),
    ("get_column", GET_COLUMN_FUNC),
    ("is_at_included_range_start", IS_AT_INCLUDED_RANGE_START_FUNC),
    ("eof", EOF_FUNC),
    ("log", LOG_FUNC),
]


TS_SERIALIZATION_BUFFER_SIZE = 1024
