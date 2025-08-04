from ..._core import builtin, int32, uint32
from ..._semantic import _check
from triton._C.libtriton import ir

__all__ = ["buffer_load_to_shared"]


@builtin
def buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    AMD Buffer load to shared operation. Buffer load is similar to normal load
    but it accesses global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers. This operation will load data
    directly into shared memory instead of registers.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (pointer to scalar): Global memory scalar base pointer to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor, optional): Tensor providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    builder = _semantic.builder

    _check(offsets.dtype in (int32, uint32),
           lambda: f"expected offsets dtype to be int32 or uint32 but got {offsets.dtype}")

    mask = mask.handle if mask is not None else ir.value()
    other = other.handle if other is not None else ir.value()
    stride = ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    builder.create_buffer_load_to_local(dest.handle, ptr.handle, offsets.handle, mask, other, stride, cache_modifier)
