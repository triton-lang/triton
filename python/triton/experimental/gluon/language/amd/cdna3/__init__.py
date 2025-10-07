from __future__ import annotations
from typing import TYPE_CHECKING

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton._C.libtriton import ir
from ..._core import builtin, _unwrap_if_constexpr

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = [
    "buffer_atomic_add", "buffer_atomic_and", "buffer_atomic_min", "buffer_atomic_max", "buffer_atomic_or",
    "buffer_atomic_xor", "buffer_atomic_xor", "buffer_load", "buffer_store", "mfma"
]

_atomic_op_str_to_op = {
    "smax": ir.ATOMIC_OP.MAX, "smin": ir.ATOMIC_OP.MIN, "umax": ir.ATOMIC_OP.UMAX, "umin": ir.ATOMIC_OP.UMIN, "fadd":
    ir.ATOMIC_OP.FADD, "iadd": ir.ATOMIC_OP.ADD, "and": ir.ATOMIC_OP.AND, "or": ir.ATOMIC_OP.OR, "xor":
    ir.ATOMIC_OP.XOR, "xchg": ir.ATOMIC_OP.XCHG
}


def _verify_buffer_ops(ptr, offsets, mask=None, other=None):
    assert ptr.type.is_ptr(), "ptr must be a scalar pointer type"

    assert isinstance(offsets.type, ttgl.distributed_type), "expected offsets type to be a distributed_type"
    assert offsets.dtype.is_int32() or offsets.dtype.is_uint32(), "offsets element type must be int32 or uint32"

    if other is not None:
        assert mask is not None, "when other is not None, mask should not be None"


def _verify_element_type_and_dispatch_op(op, elem_type, arch):
    supported_types = [
        ttgl.float16, ttgl.float32, ttgl.bfloat16, ttgl.float64, ttgl.int32, ttgl.int64, ttgl.uint32, ttgl.uint64
    ]
    assert elem_type in supported_types, f"{elem_type} is not supported in buffer atomic on {arch}."

    if op in ['and', 'or', 'xor', 'xchg']:
        assert elem_type in [ttgl.int32, ttgl.int64], f"{op} with {elem_type} is not supported on CDNA3 or CDNA4"
        return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]

    if op in ['max', 'min']:
        if elem_type in [ttgl.int32, ttgl.int64, ttgl.float64]:
            op = 's' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type in [ttgl.uint32, ttgl.uint64]:
            op = 'u' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        else:
            raise ValueError(f"{op} with {elem_type} is not supported on CDNA3 and CDNA4")

    if op == 'add':
        if elem_type in [ttgl.uint32, ttgl.uint64]:
            op = 'i' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type in [ttgl.float16, ttgl.float32, ttgl.float64]:
            op = 'f' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type is ttgl.bfloat16:
            assert arch == "cdna4", "Buffer atomic fadd with bf16 is only supported on CDNA4 for now."
            op = 'f' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        else:
            raise ValueError(f"{op} with {elem_type} is not supported on CDNA3 and CDNA4")

    raise ValueError(f"Unknown {op} on CDNA3 or CDNA4")


def _buffer_atomic_rmw_impl(op, ptr, offsets, value, arch, mask, sem, scope, _semantic):
    _verify_buffer_ops(ptr, offsets, mask)

    op = _verify_element_type_and_dispatch_op(op, ptr.type.scalar.element_ty, arch)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
        mask = _semantic.cast(mask, ttgl.int1)
        _, mask = _semantic.broadcast_impl_value(offsets, mask)
    mask = mask.handle if mask is not None else ir.value()

    value = _unwrap_if_constexpr(value)
    value = _semantic.to_tensor(value)
    _, value = _semantic.broadcast_impl_value(offsets, value)

    sem = _semantic._str_to_sem(sem)
    scope = _semantic._str_to_scope(scope)
    return _semantic.tensor(
        _semantic.builder.create_buffer_atomic_rmw(op, ptr.handle, offsets.handle, value.handle, sem, scope, mask),
        value.type)


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


"""
AMD Buffer Atomic RMW operations.
The supported operatios are max, min, add, and, or, xor, xchg.
Similar to normal atomic ops: it loads data at ptr plus offsets, do `op` with `value`, and store result to `ptr` plus `offsets` with
the specified memory semantics and scope.

Buffer atomics access global memory via a scalar base pointer and a tensor of offsets instead of a tensor of pointers.
Similar to other buffer ops, the `mask` is a boolean vector that determines if a given element should be processed with
the atomic RMW op. Elements with `mask[i] == 0` are dropped (i.e., the atomic is not executed).

Buffer Atomic RMW ops return the pre-op value in the global memory.

Args:
    ptr (pointer to scalar): Global memory scalar base pointer to load from.
    offsets (tensor): Offsets tensor for the load operation.
    value (tensor): Another operand of `op`.
    mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
    sem (str, optional): Memory Semantic Descriptor. Default is None which means acq_rel memory semantic.
    scope (str, optional): Memory Sync Scope for atomic accesses. Default is None and it will be mapped to `gpu`, which is called `agent` for AMDGPU. Please ref https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942 for details.
"""


@builtin
def buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):
    return _buffer_atomic_rmw_impl('max', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('min', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('add', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('and', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('or', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xor', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xchg(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xchg', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)
