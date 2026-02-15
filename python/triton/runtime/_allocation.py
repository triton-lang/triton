from typing import Optional, Protocol
from contextvars import ContextVar


class Buffer(Protocol):

    def data_ptr(self) -> int:
        ...


class Allocator(Protocol):

    def __call__(self, size: int, alignment: int, stream: Optional[int]) -> Buffer:
        ...


class NullAllocator:

    def __call__(self, size: int, alignment: int, stream: Optional[int]) -> Buffer:
        raise RuntimeError("Kernel requires a runtime memory allocation, but no allocator was set. " +
                           "Use triton.set_allocator to specify an allocator.")


_NULL_ALLOCATOR = NullAllocator()

_allocator: ContextVar[Allocator] = ContextVar("_allocator", default=_NULL_ALLOCATOR)


def set_allocator(allocator: Allocator) -> None:
    """
    The allocator function is called during kernel launch for kernels that
    require additional global memory workspace.
    """
    _allocator.set(allocator)


class _AllocatorWrapper:
    """
    Wrapper to provide ContextVar-like .get()/.set() methods. profile_allocator is
    used in same way as allocator so it is useful to maintain the interface.
    """

    def __init__(self, allocator: Allocator) -> None:
        self._allocator = allocator

    def get(self) -> Allocator:
        return self._allocator

    def set(self, allocator: Allocator) -> None:
        self._allocator = allocator

    def __call__(self, size: int, alignment: int, stream: Optional[int]) -> Buffer:
        return self._allocator(size, alignment, stream)


_profile_allocator = _AllocatorWrapper(_NULL_ALLOCATOR)


def set_profile_allocator(allocator: Optional[Allocator]) -> None:
    """
    The profile allocator function is called before kernel launch for kernels
    that require additional global memory workspace.
    """
    _profile_allocator.set(allocator if allocator is not None else _NULL_ALLOCATOR)
