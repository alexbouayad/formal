from collections import deque
from collections.abc import Iterable

from formal.grammar.types import ParseState

from .node import GSSNode


class StackQueue:
    __slots__ = ("_queue", "_index")

    _queue: deque[ParseState]
    _index: dict[ParseState, GSSNode]

    def __init__(self, nodes: Iterable[GSSNode]) -> None:
        self._queue: deque[ParseState] = deque()
        self._index: dict[ParseState, GSSNode] = {}

        for node in nodes:
            self.add(node)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __len__(self) -> int:
        return len(self._queue)

    def add(self, node: GSSNode, /) -> None:
        stack_top = node.parse_state
        existing_node = self._index.get(stack_top)

        if existing_node is node:
            return

        if existing_node is None:
            self._queue.append(stack_top)
            self._index[stack_top] = node

        else:
            self._index[stack_top] = existing_node | node

    def pop(self) -> tuple[ParseState, GSSNode]:
        stack_top = self._queue.popleft()
        return stack_top, self._index.pop(stack_top)

    def clear(self) -> None:
        self._queue.clear()
        self._index.clear()
