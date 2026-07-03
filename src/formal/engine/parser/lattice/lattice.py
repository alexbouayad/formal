from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from copy import copy, replace
from dataclasses import replace
from enum import Enum
from typing import Self

from formal.engine.buffer import TextPosition
from formal.engine.parser.stack import GSSNode
from formal.engine.scanner import Scanlet, ScanState
from formal.grammar.types import LexState, ParseState

from ._vertex import ParseSignature, ParseVertex


class _Missing(Enum):
    MISSING = 0


_MISSING = _Missing.MISSING


class ParseSurface(Iterable[ParseSignature]):
    __slots__ = ("_vertices",)

    _vertices: Iterable[ParseVertex]

    def __init__(self, parse_vertices: Iterable[ParseVertex], /) -> None:
        self._vertices = parse_vertices

    def __iter__(self) -> Iterator[ParseSignature]:
        for vertex in self._vertices:
            yield vertex.signature


class ParseHead:
    __slots__ = ()

    @property
    def address(self) -> str:
        return f"{id(self):#x}"

    def __repr__(self) -> str:
        return f"<ParseHead at {self.address}>"


class ParseLattice:
    __slots__ = (
        "stack",
        "position",
        "lex_state",
        "scan_state",
        "_head",
        "_vertex",
        "_index",
        "_inverse",
        "_roots",
    )

    stack: GSSNode
    position: TextPosition
    lex_state: LexState | None
    scan_state: Scanlet | ScanState | None

    _head: ParseHead
    _vertex: ParseVertex

    _index: dict[ParseHead, ParseVertex]
    _inverse: dict[ParseVertex, set[ParseHead]]
    _roots: dict[ParseSignature, ParseVertex]

    @property
    def head(self) -> ParseHead:
        return self._head

    @property
    def parse_state(self) -> ParseState:
        return self.stack.parse_state

    @property
    def surface(self) -> ParseSurface:
        return ParseSurface(self._index.values())

    def __init__(self, lattice: Self | None = None, /) -> None:
        self._index = {}
        self._inverse = {}
        self._roots = {}

        if lattice is None:
            self._setup()

        else:
            self._clone(lattice)

    def __bool__(self) -> bool:
        return bool(self._index)

    def __contains__(self, head: ParseHead) -> bool:
        return head in self._index

    def __iter__(self) -> Iterator[ParseHead]:
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}: "
            f"stack={self.stack}, "
            f"position={self.position}, "
            f"lex_state={self.lex_state}, "
            f"scan_state={self.scan_state} | "
            f"head={self.head}, "
            f"level={self._vertex.level()}, "
            f"children={len(self._vertex.children)} | "
            f"heads={len(self._index)}, "
            f"vertices={len(self._inverse)}, "
            f"roots={len(self._roots)}>"
        )

    def clear(self) -> None:
        for vertex in self._index.values():
            vertex.release()

        self._index.clear()
        self._inverse.clear()
        self._roots.clear()

        self._setup()

    def focus(self, head: ParseHead) -> bool:
        if self._head is head:
            return True

        if vertex := self._index.get(head):
            self._head = head
            self._checkout(vertex)
            return True

        return False

    @contextmanager
    def peek(self, head: ParseHead) -> Generator[None]:
        previous_head = self._head
        previous_vertex = self._vertex

        stashed_stack = self.stack
        stashed_position = self.position
        stashed_lex_state = self.lex_state
        stashed_scan_state = self.scan_state

        if not self.focus(head):
            raise KeyError

        try:
            yield

        finally:
            self._head = previous_head
            self._vertex = previous_vertex

            self.stack = stashed_stack
            self.position = stashed_position
            self.lex_state = stashed_lex_state
            self.scan_state = stashed_scan_state

    def commit(self) -> None:
        self._unlink(self._vertex)

        self._vertex.stack = self.stack
        self._vertex.position = self.position
        self._vertex.lex_state = self.lex_state
        self._vertex.scan_state = self.scan_state

        self._link(self._vertex, parent=self._vertex.parent)

    def create(self) -> ParseHead:
        new_vertex = ParseVertex(
            stack=self.stack,
            position=self.position,
            lex_state=self.lex_state,
            scan_state=self.scan_state,
        )

        new_head = self._track(new_vertex)
        self._link(new_vertex, parent=self._vertex.parent)

        return new_head

    def annihilate(self) -> None:
        self._unlink(self._vertex)

        for child in self._vertex.children.values():
            self._link(child, parent=self._vertex.parent)

        self._untrack(self._vertex)

    def superpose(
        self,
        *,
        stack: GSSNode | _Missing = _MISSING,
        position: TextPosition | _Missing = _MISSING,
        lex_state: LexState | None | _Missing = _MISSING,
        scan_state: Scanlet | ScanState | None | _Missing = _MISSING,
    ) -> ParseHead:
        parent = self._vertex.parent

        new_vertex = ParseVertex(
            stack=self._vertex.stack if stack is _MISSING else stack,
            position=self._vertex.position if position is _MISSING else position,
            lex_state=self._vertex.lex_state if lex_state is _MISSING else lex_state,
            scan_state=self._vertex.scan_state if scan_state is _MISSING else scan_state,
        )

        new_head = self._track(new_vertex)

        self._unlink(self._vertex)
        self._link(self._vertex, parent=new_vertex)
        self._link(new_vertex, parent=parent)

        return new_head

    def collapse(self) -> None:
        worklist = [self._vertex]

        while worklist:
            vertex = worklist.pop()

            for child in vertex.children.values():
                self._untrack(child)
                worklist.append(child)

        self._vertex.children.clear()

    def _checkout(self, vertex: ParseVertex) -> None:
        self._vertex = vertex

        self.stack = vertex.stack
        self.position = vertex.position
        self.lex_state = vertex.lex_state
        self.scan_state = vertex.scan_state

    def _clone(self, lattice: Self) -> None:
        worklist: list[tuple[ParseVertex, ParseVertex | None]]

        root_vertices = lattice._roots.values()
        worklist = [(vertex, None) for vertex in root_vertices]

        while worklist:
            vertex, new_parent = worklist.pop()

            match vertex.scan_state:
                case Scanlet() as scanlet:
                    new_vertex = replace(vertex, scan_state=scanlet.clone(), position=scanlet.start_position)

                case _:
                    new_vertex = copy(vertex)

            self._track(new_vertex)
            self._link(new_vertex, parent=new_parent)

            for child in vertex.children.values():
                worklist.append((child, new_vertex))

        head, vertex = next(iter(self._index.items()))

        self._head = head
        self._checkout(vertex)

    def _setup(self) -> None:
        vertex = ParseVertex()

        head = self._track(vertex)
        self._link(vertex, parent=None)

        self._head = head
        self._checkout(vertex)

    def _track(self, vertex: ParseVertex) -> ParseHead:
        head = ParseHead()

        self._index[head] = vertex
        self._inverse[vertex] = {head}

        return head

    def _untrack(self, vertex: ParseVertex) -> None:
        heads = self._inverse.pop(vertex)

        for head in heads:
            del self._index[head]

        vertex.release()

    def _link(self, vertex: ParseVertex, *, parent: ParseVertex | None) -> None:
        worklist = [(vertex, parent)]

        while worklist:
            current_vertex, current_base = worklist.pop()

            current_siblings = self._roots if current_base is None else current_base.children
            signature = current_vertex.signature

            existing_vertex = current_siblings.get(signature)

            if existing_vertex is current_vertex:
                continue

            if existing_vertex is None:
                current_vertex.parent = current_base
                current_siblings[signature] = current_vertex
                continue

            existing_vertex.stack |= current_vertex.stack

            source_heads = self._inverse.pop(current_vertex)
            target_heads = self._inverse[existing_vertex]

            for head in source_heads:
                self._index[head] = existing_vertex
                target_heads.add(head)

            worklist.extend((current_child, existing_vertex) for current_child in current_vertex.children.values())

    def _unlink(self, vertex: ParseVertex) -> None:
        siblings = self._roots if vertex.parent is None else vertex.parent.children
        del siblings[vertex.signature]
