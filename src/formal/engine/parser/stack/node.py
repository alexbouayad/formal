from collections.abc import Set
from dataclasses import dataclass, field
from typing import Self

from formal.grammar.types import ParseState


@dataclass(eq=False, frozen=True, repr=False, slots=True)
class GSSNode:
    parse_state: ParseState
    links: Set[Self] = field(default_factory=set[Self])

    @property
    def degree(self) -> int:
        return len(self.links)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: parse_state={self.parse_state}, degree={self.degree}>"

    def __or__(self, other: Self, /) -> Self:
        if self is other:
            return self

        if self.parse_state != other.parse_state:
            raise ValueError

        cls = type(self)

        state = self.parse_state
        links = self.links | other.links

        return cls(state, links)

    def pop(self, count: int = 1, /) -> set[Self]:
        if count < 0:
            return set()

        if count == 0:
            return {self}

        popped_stacks = set[Self]()
        start_state = self, count

        worklist = [start_state]
        visited = {start_state}

        while worklist:
            stack, countdown = worklist.pop()

            if countdown == 0:
                popped_stacks.add(stack)
                continue

            for next_stack in stack.links:
                next_state = next_stack, countdown - 1

                if next_state not in visited:
                    visited.add(next_state)
                    worklist.append(next_state)

        return popped_stacks

    def push(self, parse_state: ParseState, /) -> Self:
        links = {self}
        return type(self)(parse_state, links)
