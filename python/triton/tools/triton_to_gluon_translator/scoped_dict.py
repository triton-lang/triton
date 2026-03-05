from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class scoped_dict(Generic[K, V]):
    stack: list[dict[K, V]] = field(default_factory=list)

    def __init__(self, d: dict[K, V] | None = None) -> None:
        self.stack = [d or {}]

    def __getitem__(self, key: K) -> V:
        for d in reversed(self.stack):
            if key in d:
                return d[key]
        raise KeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        self.stack[-1][key] = value

    def __contains__(self, key: K) -> bool:
        return any(key in d for d in reversed(self.stack))

    def setdefault(self, key: K, value: V) -> V:
        return self.stack[-1].setdefault(key, value)

    @contextmanager
    def scope(self, d: dict[K, V] | None = None) -> Generator[None, None, None]:
        self.stack.append(d or {})
        try:
            yield
        finally:
            self.stack.pop()
