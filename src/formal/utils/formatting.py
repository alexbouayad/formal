from abc import ABC, abstractmethod


class ReprMixin(ABC):
    __slots__ = ()

    @abstractmethod
    def __format__(self, format_spec: str) -> str: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"
