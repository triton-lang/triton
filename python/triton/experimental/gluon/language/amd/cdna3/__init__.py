from __future__ import annotations
from typing import TYPE_CHECKING

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton._C.libtriton import ir
from ..._core import builtin, _unwrap_if_constexpr
from ..._semantic import _check

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = ["async_copy_global_to_shared", "async_wait", "buffer_load_to_shared", "buffer_load", "buffer_store", "mfma"]


@builtin
def async_copy_global_to_shared(dest, pointer, mask=None, cache_modifier="", eviction_policy="", volatile=False,
                                _semantic=None):
    """
    Asynchronously copy elements from global memory to shared memory.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        pointer (tensor): Source pointer tensor.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
        eviction_policy (str): Eviction policy specifier. Defaults to "".
        volatile (bool): Whether the load is volatile. Defaults to False.
    """
    mask = _unwrap_if_constexpr(mask)
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction_policy = _semantic._str_to_eviction_policy(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    _check(
        dest.shape == pointer.shape, lambda:
        f"expected dest shape to match pointer shape but got dest.shape = {dest.shape}, pointer.shape = {pointer.shape}"
    )
    _check(len(dest.shape) == 2, lambda: f"expected dest shape to be 2d but got {len(dest.shape)}d")
    mask_handle = mask.handle if mask is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(dest.handle, pointer.handle, mask_handle, cache_modifier,
                                                        eviction_policy, volatile)


@builtin
def async_wait(num_outstanding=0, _semantic=None):
    """
    Wait for outstanding asynchronous copy operations.

    Args:
        num_outstanding (int): Wait until `num_outstanding` or less async copy operations in-flight. Defaults to 0.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_wait_group(num_outstanding)


def _verify_buffer_load_store(ptr, offsets, mask, other=None):
    assert ptr.type.is_ptr(), "ptr must be a scalar pointer type"

    assert isinstance(offsets.type, ttgl.distributed_type), "expected offsets type to be a distributed_type"
    assert offsets.dtype.is_int32() or offsets.dtype.is_uint32(), "offsets element type must be int32 or uint32"

    element_type = ptr.type.scalar.element_ty

    if other is not None:
        assert mask is not None, "when other is not None, mask should not be None"
        assert other.shape == offsets.shape, "other shape must match the offsets shape"
        assert other.dtype == element_type, "other must have the same data type as ptr scalar type"


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
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)

    other = _unwrap_if_constexpr(other)
    if other is not None:
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

    _verify_buffer_load_store(ptr, offsets, mask, other)

    mask = mask.handle if mask is not None else ir.value()
    other = other.handle if other is not None else ir.value()
    stride = ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    _semantic.builder.create_buffer_load_to_local(dest.handle, ptr.handle, offsets.handle, mask, other, stride,
                                                  cache_modifier)


@builtin
def buffer_load(ptr, offsets, mask=None, other=None, cache=None, _semantic=None):
    """
    AMD buffer load from global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers. This operation will load data
    directly into registers.

    Args:
        ptr (pointer to scalar): Global memory scalar base pointer to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor, optional): Tensor providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)

    other = _unwrap_if_constexpr(other)
    if other is not None:
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

    _verify_buffer_load_store(ptr, offsets, mask, other)

    other = other.handle if other is not None else ir.value()
    mask = mask.handle if mask is not None else ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

    ret_ty = offsets.type.with_element_ty(ptr.type.scalar.element_ty)
    builder = _semantic.builder
    handle = builder.create_buffer_load(ret_ty.to_ir(builder), ptr.handle, offsets.handle, mask, other, cache_modifier)
    return ttgl.tensor(handle, ret_ty)


@builtin
def buffer_store(stored_value, ptr, offsets, mask=None, cache=None, _semantic: GluonSemantic = None):
    """
    AMD buffer store a tensor directly to global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers.
    Args:
        stored_value (tensor to be stored): The tensor to be stored to global memory.
        ptr (pointer to scalar): Global memory scalar base pointer to store to.
        offsets (tensor): Offsets tensor for the store operation.
        mask (tensor, optional): Mask tensor for predicated store. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)

    _verify_buffer_load_store(ptr, offsets, mask)

    mask = mask.handle if mask is not None else ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

    _semantic.builder.create_buffer_store(stored_value.handle, ptr.handle, offsets.handle, mask, cache_modifier)


@builtin
def mfma(a, b, acc, _semantic: GluonSemantic = None):
    """
    Computes matrix-multiplication of a * b + acc using AMD native matrix core units.
    Args:
        a (tensor): The first operand of mfma.
        b (tensor): The second operand of mfma.
        acc (tensor): The accumulator tensor.
    """
    assert acc is not None, "acc is required"
    ret_type = acc.type
    acc = ttgl._unwrap_if_constexpr(acc)

    handle = _semantic.dot(a, b, acc, input_precision=knobs.language.fp32_default, max_num_imprecise_acc=None,
                           out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, ret_type)
