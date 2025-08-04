from __future__ import annotations
from typing import TYPE_CHECKING

from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import builtin
from triton._C.libtriton import ir

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = ["buffer_load", "buffer_store"]


def _verify_buffer_load_store(ptr, offsets, mask, other=None):
    assert ptr.type.is_ptr(), "ptr must be a scalar pointer type"

    assert isinstance(offsets.type, ttgl.distributed_type), "expected offsets type to be a distributed_type"
    assert offsets.dtype.is_int32() or offsets.dtype.is_uint32(), "offsets element type must be int32 or uint32"

    shape = offsets.shape
    element_type = ptr.type.scalar.element_ty

    if mask is not None:
        assert mask.shape == shape, "offsets must have the same shape as offsets"

    if other is not None:
        assert mask is not None, "when other is not None, mask should not be None"
        assert other.shape == shape, "other must have the same shape as offsets"
        assert other.dtype == element_type, "other must have the same data type as ptr scalar type"


@builtin
def buffer_load(ptr, offsets, mask=None, other=None, cache=None, _semantic=None):
    """
    AMD buffer load from global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers. This operation will load data
    directly into registers.
    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (pointer to scalar): Global memory scalar base pointer to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor, optional): Tensor providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
      This op is lowered into BufferLoadOp of TritonAMDGPU directly.
      It reads from global memory and return a tensor of data.
    """
    _verify_buffer_load_store(ptr, offsets, mask, other)

    mask = mask.handle if mask is not None else ir.value()
    other = other.handle if other is not None else ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

    shape = ttgl._unwrap_if_constexpr(offsets.shape)
    layout = ttgl._unwrap_if_constexpr(offsets.type.layout)

    ret_ty = ttgl.distributed_type(ptr.type.scalar.element_ty, shape, layout)
    builder = _semantic.builder
    handle = builder.create_buffer_load(ret_ty.to_ir(builder), ptr.handle, offsets.handle, mask, other, cache_modifier)
    return ttgl.tensor(handle, ret_ty)


@builtin
def buffer_store(stored_value, ptr, offsets, mask, cache=None, _semantic: GluonSemantic = None):
    """
    AMD buffer store a tensor directly to global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers.
    Args:
        stored_value (shared_memory_descriptor): The tensor to be stored to global memory.
        ptr (pointer to scalar): Global memory scalar base pointer to store to.
        offsets (tensor): Offsets tensor for the store operation.
        mask (tensor, optional): Mask tensor for predicated store. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    _verify_buffer_load_store(ptr, offsets, mask)

    mask = mask.handle if mask is not None else ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

    _semantic.builder.create_buffer_store(stored_value.handle, ptr.handle, offsets.handle, mask, cache_modifier)
