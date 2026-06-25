from io import SEEK_END, StringIO
from typing import Final, NamedTuple

EOS_CHAR = "\ue000"


class TextPosition(NamedTuple):
    offset: int = 0
    column: int = 0


class ReadOnlyBuffer:
    __slots__ = ("_string", "_column", "_lookahead", "_checkpoint")

    _string: Final[StringIO]

    _column: int
    _lookahead: str
    _checkpoint: int

    def __init__(self, text: str = "", /) -> None:
        self._string = StringIO(text)

        self._column = 0
        self._lookahead = ""
        self._checkpoint = 0

    @property
    def offset(self) -> int:
        return self._string.tell()

    @property
    def column(self) -> int:
        return self._column

    @property
    def position(self) -> TextPosition:
        return TextPosition(self._string.tell(), self._column)

    @position.setter
    def position(self, value: TextPosition, /) -> None:
        if value == self.position.offset:
            return

        self._sync(value.offset)
        self._column = value.column

    @property
    def lookahead(self) -> str:
        raw_lookahead = self._lookahead
        return "" if raw_lookahead == EOS_CHAR else raw_lookahead

    @property
    def at_eof(self) -> bool:
        return self._lookahead == EOS_CHAR

    def __len__(self) -> int:
        string = self._string
        offset = string.tell()

        string.seek(0, SEEK_END)
        length = string.tell()

        string.seek(offset)
        return length

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}: "
            f"length={len(self)}, "
            f"offset={self.offset}, "
            f"column={self.column}, "
            f"lookahead={self.lookahead!r}, "
            f"checkpoint={self._checkpoint}, "
            f"at_eof={self.at_eof}>"
        )

    def read(self) -> str:
        lookahead = self._lookahead

        if lookahead == "" or lookahead == EOS_CHAR:
            return ""

        self._string.read(1)
        self._sync()

        if lookahead == "\n":
            self._column = 0
        else:
            self._column += 1

        return lookahead

    def _sync(self, offset: int | None = None) -> None:
        string = self._string

        if offset is None:
            offset = string.tell()
        else:
            string.seek(offset)

        self._lookahead = string.read(1)
        string.seek(offset)


class Buffer(ReadOnlyBuffer):
    def clear(self) -> None:
        string = self._string

        string.seek(0)
        string.truncate()

        self._column = 0
        self._lookahead = ""
        self._checkpoint = 0

    def feed(self, text: str, *, is_eos: bool = False) -> None:
        string = self._string

        string.seek(0, SEEK_END)
        offset = string.tell()

        if text:
            string.write(text)

        if is_eos:
            string.write(EOS_CHAR)

        self._sync(offset)

    def revert(self) -> None:
        string = self._string
        checkpoint = self._checkpoint

        string.truncate(checkpoint)

        if string.tell() >= checkpoint:
            self._lookahead = ""

    def validate(self) -> None:
        self._checkpoint = len(self)
