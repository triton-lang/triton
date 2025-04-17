from __future__ import annotations

from functools import reduce
from typing import Any, Callable, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .language import core
    IterableType = Union[list[Any], tuple[Any, ...], core.tuple, core.tuple_type]
    ObjPath = tuple[int, ...]


def get_iterable_path(iterable: IterableType, path: ObjPath) -> Any:
    return reduce(lambda a, idx: a[idx], path, iterable)  # type: ignore[index]


def set_iterable_path(iterable: IterableType, path: tuple[int, ...], val: Any):
    assert len(path) != 0
    prev = iterable if len(path) == 1 else get_iterable_path(iterable, path[:-1])
    prev[path[-1]] = val  # type: ignore[index]


def find_paths_if(iterable: Union[IterableType, Any], pred: Callable[[ObjPath, Any], bool]) -> list[ObjPath]:
    from .language import core
    is_iterable: Callable[[Any], bool] = lambda x: isinstance(x, (list, tuple, core.tuple, core.tuple_type))
    # We need to use dict so that ordering is maintained, while set doesn't guarantee order
    ret: dict[ObjPath, None] = {}

    def _impl(path: tuple[int, ...], current: Any):
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl((*path, idx), item)
        elif pred(path, current):
            ret[path] = None

    _impl((), iterable)

    return list(ret.keys())
