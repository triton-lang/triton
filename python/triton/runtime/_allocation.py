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


_allocator: ContextVar[Allocator] = ContextVar("_allocator", default=NullAllocator())


def set_allocator(allocator: Allocator):
    """
    The allocator function is called during kernel launch for kernels that
    require additional global memory workspace.
    """
    _allocator.set(allocator)


_profile_allocator: Allocator = ContextVar("_allocator", default=NullAllocator())


def set_profile_allocator(allocator: Optional[Allocator]):
    """
    The profile allocator function is called before kernel launch for kernels
    that require additional global memory workspace.
    """
    global _profile_allocator
    _profile_allocator.set(allocator)
