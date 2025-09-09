from __future__ import annotations
from typing import TYPE_CHECKING

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton._C.libtriton import ir
from ..._core import builtin, _unwrap_if_constexpr

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = ["buffer_load", "buffer_store", "mfma", "buffer_atomic_rmw"]


def _verify_buffer_ops(ptr, offsets, mask=None, other=None):
    assert ptr.type.is_ptr(), "ptr must be a scalar pointer type"

    assert isinstance(offsets.type, ttgl.distributed_type), "expected offsets type to be a distributed_type"
    assert offsets.dtype.is_int32() or offsets.dtype.is_uint32(), "offsets element type must be int32 or uint32"

    if other is not None:
        assert mask is not None, "when other is not None, mask should not be None"


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
        other (tensor or scalar, optional): Tensor or scalar providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    _verify_buffer_ops(ptr, offsets, mask, other)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)

    other = _unwrap_if_constexpr(other)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, ptr.dtype.element_ty)
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

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
    _verify_buffer_ops(ptr, offsets, mask)

    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)

    mask = mask.handle if mask is not None else ir.value()
    cache_modifier = _semantic._str_to_store_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

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


op_str_to_op = {
    "max": ir.ATOMIC_OP.MAX, "min": ir.ATOMIC_OP.MIN, "fadd": ir.ATOMIC_OP.FADD, "add": ir.ATOMIC_OP.ADD, "and":
    ir.ATOMIC_OP.AND, "or": ir.ATOMIC_OP.OR, "xor": ir.ATOMIC_OP.XOR, "xchg": ir.ATOMIC_OP.XCHG
}


@builtin
def buffer_atomic_rmw(op, ptr, offsets, value, mask=None, sem=None, scope=None, stride=None, _semantic=None):
    """
    AMD Buffer atomic RMW operation. Buffer atomics are similar to normal atomics, but access global memory via a
    scalar base pointer and a tensor of offsets instead of a tensor of pointers.
    Similar to other buffer ops, the `mask` is a boolean vector that determines if a given element should be processed with
    the atomic RMW op. Elements with `mask[i] == 0` are dropped (i.e., the atomic is not executed).
    Similar to TT_AtomicRMWOp: Buffer atomic RMW ops load data at $ptr, do $rmw_op with $val, and store result to $ptr with
    the specified memory semantics and scope.

    Stride is the distance between the beginning of contiguous memory chunks. When performing a RMW, the `stride` is
    the address difference between the first elements of each row in bytes. Compiler tries to obtain the `stride`
    when it converts to the buffer ops because it is important for optimizing the cache memory access.

    Atomic RMW ops return the pre-op value in the global memory.

    Args:
        op (str) : The operator to be executed atomically.
        ptr (pointer to scalar): Global memory scalar base pointer to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        sem (str, optional): Memory Semantic Descriptor. Default is None.
        scope (str, optional): Memory Sync Scope. Default is None.
        stride (int32, optional): The address difference between the first elements of each row in bytes. Default is 1.
    """
    _verify_buffer_ops(ptr, offsets, mask)

    mask = _unwrap_if_constexpr(mask)
    mask = mask.handle if mask is not None else ir.value()

    sem = _semantic._str_to_sem(sem)
    scope = _semantic._str_to_scope(scope)

    op = op_str_to_op[_unwrap_if_constexpr(op)]
    ret_type = value.type
    if stride is None:
        stride = 1
    stride = _semantic.builder.get_int32(stride)

    return _semantic.tensor(
        _semantic.builder.create_buffer_atomic_rmw(op, ptr.handle, offsets.handle, mask, value.handle, sem, scope,
                                                   stride), ret_type)
