from typing import Callable, TypeVar, Tuple

import triton
import triton.language as tl
from .. import impl

T = TypeVar("T")


def _add_atomic_docstr(name: str) -> Callable[[T], T]:
    def _decorator(func: T) -> T:
        docstr = """
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to compare-and-swap.
    :type pointer: Block of dtype=triton.PointerDType
    :param cmp: The values expected to be found in the atomic object
    :type cmp: Block of dtype=`pointer.dtype.element_ty`
    :param val: The values to copy in case the expected value matches the contained value.
    :type val: Block of dtype=`pointer.dtype.element_ty`
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _atom_red_typechecking_impl(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    op: str,
    builder: tl.ir.builder,
) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    if not ptr.type.scalar.is_ptr():
        raise ValueError(
            "Pointer argument of store instruction is " + ptr.type.__repr__()
        )

    element_ty = ptr.type.scalar.element_ty
    if element_ty is tl.float16 and op != "add":
        raise ValueError("atomic_" + op + " does not support fp16")
    if element_ty in [tl.int1, tl.int8, tl.int16, tl.bfloat16]:
        raise ValueError("atomic_" + op + " does not support " + element_ty)
    if ptr.type.is_block():
        if mask:
            mask = impl._broadcast_impl_shape(
                mask,
                ptr.type.get_block_shapes(),
                builder,
            )
        if val:
            val = impl._broadcast_impl_shape(
                val,
                ptr.type.get_block_shapes(),
                builder,
            )
    val = impl._i_cast(val, ptr.type.scalar.element_ty, builder)
    if not mask:
        mask_ir = builder.get_int1(True)
        mask_ty = tl.int1
        if ptr.type.is_block():
            mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
            mask_ty = tl.block_type(tl.int1, ptr.type.get_block_shapes())
        mask = tl.tensor(mask_ir, mask_ty)
    return ptr, val, mask


def _i_atomic_cas(
    ptr: tl.tensor,
    cmp: tl.tensor,
    val: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    element_ty = ptr.type.scalar.element_ty
    if element_ty.primitive_bitwidth not in [16, 32, 64]:
        raise ValueError("atomic_cas only supports elements with width {16, 32, 64}")
    return tl.tensor(
        builder.create_atomic_cas(
            ptr.handle,
            cmp.handle,
            val.handle,
        ),
        val.type,
    )


@triton.builtin
@_add_atomic_docstr("compare-and-swap")
def atomic_cas(pointer, cmp, val, _builder=None):
    cmp = tl._to_tensor(cmp, _builder)
    val = tl._to_tensor(val, _builder)
    return _i_atomic_cas(pointer, cmp, val, _builder)


def _i_atomic_xchg(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(
        ptr,
        val,
        mask,
        "xchg",
        builder,
    )
    return tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.XCHG,
            ptr.handle,
            val.handle,
            mask.handle,
        ),
        val.type,
    )


@triton.builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, _builder=None):
    val = tl._to_tensor(val, _builder)
    return _i_atomic_xchg(pointer, val, mask, _builder)


def _i_atomic_add(
    ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, builder: tl.ir.builder
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "add", builder)
    sca_ty = val.type.scalar
    op = tl.ir.ATOMIC_OP.FADD if sca_ty.is_floating() else tl.ir.ATOMIC_OP.ADD
    return tl.tensor(
        builder.create_atomic_rmw(
            op,
            ptr.handle,
            val.handle,
            mask.handle,
        ),
        val.type,
    )


@triton.builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, _builder=None):
    val = tl._to_tensor(val, _builder)
    return _i_atomic_add(pointer, val, mask, _builder)


def _i_atomic_max(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "max", builder)
    sca_ty = val.type.scalar
    # direct call to atomic_max for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(
                builder.create_atomic_rmw(
                    tl.ir.ATOMIC_OP.MAX,
                    ptr.handle,
                    val.handle,
                    mask.handle,
                ),
                val.type,
            )
        else:
            return tl.tensor(
                builder.create_atomic_rmw(
                    tl.ir.ATOMIC_OP.UMAX,
                    ptr.handle,
                    val.handle,
                    mask.handle,
                ),
                val.type,
            )
    # for float
    # return atomic_smax(i_ptr, i_val) if val >= 0
    # return atomic_umin(i_ptr, i_val) if val < 0
    i_val = impl._i_bitcast(
        val,
        tl.int32,
        builder,
    )
    i_ptr = impl._i_bitcast(
        ptr,
        tl.pointer_type(tl.int32, 1),
        builder,
    )
    pos = impl._i_greater_equal(
        val,
        tl.tensor(builder.get_float32(0), sca_ty),
        builder,
    )
    neg = impl._i_less_than(
        val,
        tl.tensor(
            builder.get_float32(0),
            sca_ty,
        ),
        builder,
    )
    pos_ret = tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.MAX,
            i_ptr.handle,
            i_val.handle,
            impl._i_and_(mask, pos, builder).handle,
        ),
        i_val.type,
    )
    neg_ret = tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.UMIN,
            i_ptr.handle,
            i_val.handle,
            impl._i_and_(mask, neg, builder).handle,
        ),
        i_val.type,
    )
    return impl._i_where(pos, pos_ret, neg_ret, builder)


@triton.builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, _builder=None):
    val = tl._to_tensor(val, _builder)
    return _i_atomic_max(pointer, val, mask, _builder)


def _i_atomic_min(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "min", builder)
    sca_ty = val.type.scalar
    # direct call to atomic_min for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(
                builder.create_atomic_rmw(
                    tl.ir.ATOMIC_OP.MIN,
                    ptr.handle,
                    val.handle,
                    mask.handle,
                ),
                val.type,
            )
        else:
            return tl.tensor(
                builder.create_atomic_rmw(
                    tl.ir.ATOMIC_OP.UMIN,
                    ptr.handle,
                    val.handle,
                    mask.handle,
                ),
                val.type,
            )
    # for float
    # return atomic_smin(i_ptr, i_val) if val >= 0
    # return atomic_umax(i_ptr, i_val) if val < 0
    i_val = impl._i_bitcast(
        val,
        tl.int32,
        builder,
    )
    i_ptr = impl._i_bitcast(
        ptr,
        tl.pointer_type(tl.int32, 1),
        builder,
    )
    pos = impl._i_greater_equal(
        val,
        tl.tensor(builder.get_float32(0), sca_ty),
        builder,
    )
    neg = impl._i_less_than(
        val,
        tl.tensor(builder.get_float32(0), sca_ty),
        builder,
    )
    pos_ret = tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.MIN,
            i_ptr.handle,
            i_val.handle,
            impl._i_and_(mask, pos, builder).handle,
        ),
        i_val.type,
    )
    neg_ret = tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.UMAX,
            i_ptr.handle,
            i_val.handle,
            impl._i_and_(mask, neg, builder).handle,
        ),
        i_val.type,
    )
    return impl._i_where(pos, pos_ret, neg_ret, builder)


@triton.builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, _builder=None):
    val = tl._to_tensor(val, _builder)
    return _i_atomic_min(pointer, val, mask, _builder)


def _i_atomic_and(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "and", builder)
    return tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.AND,
            ptr.handle,
            val.handle,
            mask.handle,
        ),
        val.type,
    )


@triton.builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, _builder=None):
    val = tl._to_tensor(val, _builder)
    return _i_atomic_and(pointer, val, mask, _builder)


def _i_atomic_or(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "or", builder)
    return tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.OR,
            ptr.handle,
            val.handle,
            mask.handle,
        ),
        val.type,
    )


@triton.builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, _builder=None):
    val = tl._to_tensor(val, _builder)
    return _i_atomic_or(pointer, val, mask, _builder)


def _i_atomic_xor(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "xor", builder)
    return tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.XOR,
            ptr.handle,
            val.handle,
            mask.handle,
        ),
        val.type,
    )


@triton.builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, _builder=None):
    val = tl._to_tensor(val, _builder)
    return _i_atomic_xor(pointer, val, mask, _builder)
