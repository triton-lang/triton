from typing import Optional, Protocol


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


_allocator: Allocator = NullAllocator()


def set_allocator(allocator: Allocator):
    """
    The allocator function is called during kernel launch for kernels that
    require additional global memory workspace.
    """
    global _allocator
    _allocator = allocator
