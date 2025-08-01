from ..._core import builtin
from triton._C.libtriton import ir

__all__ = ["create_buffer_load_to_local"]


@builtin
def create_buffer_load_to_local(dest, ptr, offsets, mask=None, other=None, stride=None,
                                cache_modifier="", _semantic=None):
    """
    AMD Buffer load operation. Similar to amdgpu.buffer_load op but directly writes to shared memory instead of into registers.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (tensor): Pointer tensor to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor, optional): Additional tensor for the load operation. Defaults to None.
        stride (tensor, optional): Stride tensor for the load operation. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    builder = _semantic.builder

    mask = mask.handle if mask is not None else ir.value()
    other = other.handle if other is not None else ir.value()
    stride = stride.handle if stride is not None else ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    builder.create_buffer_load_to_local(dest.handle, ptr.handle, offsets.handle,
                                        mask, other, stride, cache_modifier)
