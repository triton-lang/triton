from __future__ import annotations
from typing import TYPE_CHECKING

from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import builtin
from triton._C.libtriton import ir

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = ["buffer_load", "buffer_store"]


@builtin
def buffer_load(ptr, offsets, mask=None, other=None, stride=None, cache=None, _semantic=None):
    """
      This op is lowered into BufferLoadOp of TritonAMDGPU directly.
      It reads from global memory and return a tensor of data.

      Params:
      ptr: A scalar base pointer, NOT a block of pointers.
      offsets: A tensor of offsets in the global memory.
      stride: (Optional) address difference between the first elements of each row in bytes.
      other: (Optional) Similiar with mask in a normal load.
      mask: (Optional) A tensor of boolean that determines if a given element should be read from memory, and `other` is the element that should be returned on lane `i` when `mask[i] == 0`.
    """
    verify_buffer_ldstr(ptr, offsets, mask, other)

    mask = mask.handle if mask is not None else ir.value()
    other = other.handle if other is not None else ir.value()
    stride = stride.handle if stride is not None else ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

    shape = ttgl._unwrap_if_constexpr(offsets.shape)
    layout = ttgl._unwrap_if_constexpr(offsets.type.layout)

    ret_ty = ttgl.distributed_type(ptr.type.scalar.element_ty, shape, layout)
    builder = _semantic.builder
    handle = builder.create_buffer_load(ret_ty.to_ir(builder), ptr.handle, offsets.handle, stride, mask, other,
                                        cache_modifier)
    return ttgl.tensor(handle, ret_ty)


@builtin
def buffer_store(stored_value, ptr, offsets, mask, stride=None, cache=None, _semantic: GluonSemantic = None):
    """
      This op is lowered into BufferStoreOp of TritonAMDGPU directly.
      It writes a tensor of da data to global memory.

      Params share the semantic as in buffer_load.
    """
    verify_buffer_ldstr(ptr, offsets, mask)

    mask = mask.handle if mask is not None else ir.value()
    stride = stride.handle if stride is not None else ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

    _semantic.builder.create_buffer_store(stored_value.handle, ptr.handle, offsets.handle, stride, mask, cache_modifier)


def verify_buffer_ldstr(ptr, offsets, mask, other=None):
    assert ptr.type.is_ptr(), "ptr must be a scalar pointer type"

    assert isinstance(offsets.type, ttgl.distributed_type), "expected offsets type to be a distributed_type"
    assert offsets.dtype.is_int32(), "offsets element type must be int32"

    shape = offsets.shape
    element_type = ptr.type.scalar.element_ty

    if mask is not None:
        assert mask.shape == shape, "offsets must have the same shape as offsets"
    if other is not None:
        assert mask is not None, "when other is not None, mask should be not None"
        assert other.shape == shape, "other must have the same shape as offsets"
        assert other.dtype == element_type, "other must have the same data type as ptr scalar type"
