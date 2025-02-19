from __future__ import annotations  # remove after python 3.11
import warnings

from typing import List, Optional, Sequence, Tuple, TypeVar
import numbers

from .._C.libtriton import ir
from . import core as tl

T = TypeVar('T')


class IncompatibleTypeErrorImpl(Exception):

    def __init__(self, type_a, type_b):
        self.type_a = type_a
        self.type_b = type_b
        self.message = "invalid operands of type " + self.type_a.__repr__() + " and " + self.type_b.__repr__()
        super(IncompatibleTypeErrorImpl, self).__init__(self.message)


# ===----------------------------------------------------------------------===##
# Programming Model
# ===----------------------------------------------------------------------===##


def program_id(axis: int, builder: ir.builder) -> tl.tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"program_id axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(builder.create_get_program_id(axis), tl.int32)


def num_programs(axis: int, builder: ir.builder) -> tl.tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"num_programs axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(builder.create_get_num_programs(axis), tl.int32)


# ===----------------------------------------------------------------------===//
#                               Implicit Casting Utilities
# ===----------------------------------------------------------------------===//


def integer_promote_impl(a_ty: tl.dtype, b_ty: tl.dtype) -> tl.dtype:
    a_rank = a_ty.int_bitwidth
    b_rank = b_ty.int_bitwidth
    a_sn = a_ty.int_signedness
    b_sn = b_ty.int_signedness
    # Rules for signedness taken from "Usual arithmetic conversions" on
    # https://en.cppreference.com/w/c/language/conversion.
    if a_sn == b_sn:
        return a_ty if a_rank > b_rank else b_ty
    elif a_sn == tl.dtype.SIGNEDNESS.UNSIGNED:
        return a_ty if a_rank >= b_rank else b_ty
    elif b_sn == tl.dtype.SIGNEDNESS.UNSIGNED:
        return b_ty if b_rank >= a_rank else a_ty
    raise TypeError(f"unexpected signedness {a_sn} and {b_sn}")


def computation_type_impl(a_ty: tl.dtype, a_is_scalar: bool, b_ty: tl.dtype, b_is_scalar: bool,
                          div_or_mod: bool) -> tl.dtype:
    # 0) For scalars we follow semantics similar to PyTorch, namely:
    # - If the scalar is of a lower or equal kind (bool < uint < int < fp),
    #   it doesn't participate in the promotion
    if a_is_scalar != b_is_scalar:
        scalar_ty, tensor_ty = (a_ty, b_ty) if a_is_scalar else (b_ty, a_ty)
        if scalar_ty.kind().value <= tensor_ty.kind().value:
            # Upcast because of 3) and 4) below!
            if div_or_mod and (tensor_ty in (tl.float16, tl.bfloat16)):
                return tl.float32
            return tensor_ty

    # 1) if one operand is double, the other is implicitly
    #    converted to double
    if a_ty.is_fp64() or b_ty.is_fp64():
        return tl.float64
    # 2) if one operand is float, the other is implicitly
    #    converted to float
    if a_ty.is_fp32() or b_ty.is_fp32():
        return tl.float32
    # 3 ) if one operand is half, the other is implicitly converted to half
    #     unless we're doing / or %, which do not exist natively in PTX for fp16.
    #     Supported PTX op: add, sub, mul, fma, neg, abs, min, max, tanh, ex2, setp
    if a_ty.is_fp16() or b_ty.is_fp16():
        if div_or_mod:
            return tl.float32
        else:
            return tl.float16
    # 4) return bf16 only if both operands are of bf16
    if a_ty.is_bf16() and b_ty.is_bf16():
        if div_or_mod:
            return tl.float32
        else:
            return tl.bfloat16
    if a_ty.is_bf16() or b_ty.is_bf16():
        return tl.float32
    # 5) return fp16 if operands are different fp8
    if a_ty.is_fp8() and b_ty.is_fp8():
        return a_ty if a_ty == b_ty else tl.float16
    if not a_ty.is_int() or not b_ty.is_int():
        raise TypeError(f"unexpected type {a_ty} and {b_ty}")
    # 6 ) both operands are integer and undergo
    #    integer promotion
    if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
        raise TypeError("Cannot use /, #, or % with " + a_ty.__repr__() + " and " + b_ty.__repr__() +
                        " because they have different signedness;"
                        "this is unlikely to result in a useful answer. Cast them to the same signedness.")
    return integer_promote_impl(a_ty, b_ty)


def to_tensor(x, builder, check_type: bool = True):
    if isinstance(x, bool):
        return tl.tensor(builder.get_int1(x), tl.int1)
    # Note: compile-time const integers are represented by unsigned values
    elif isinstance(x, int):
        if -2**31 <= x < 2**31:
            dtype = tl.int32
        elif 2**31 <= x < 2**32:
            dtype = tl.uint32
        elif -2**63 <= x < 2**63:
            dtype = tl.int64
        elif 2**63 <= x < 2**64:
            dtype = tl.uint64
        else:
            raise ValueError(f'Nonrepresentable integer {x}.')
        return full((), x, dtype=dtype, builder=builder)
    elif isinstance(x, float):
        min_float32 = 2**-126
        max_float32 = (2 - 2**-23) * 2**127
        abs_x = __builtins__['abs'](x)
        if abs_x == float("inf") or\
           abs_x == 0.0 or \
           x != x or \
           min_float32 <= abs_x <= max_float32:
            dtype = tl.float32
        else:
            dtype = tl.float64
        return full((), x, dtype=dtype, builder=builder)

    elif isinstance(x, tl.constexpr):
        return to_tensor(x.value, builder)
    elif isinstance(x, tl.tensor):
        return x
    if check_type:
        raise TypeError(f"cannot convert {x} of type {type(x)} to tensor")
    return x


# ===----------------------------------------------------------------------===//
#                               Binary Operators
# ===----------------------------------------------------------------------===//


def check_ptr_type_impl(type_a: tl.dtype, type_b: tl.dtype, allow_ptr_a: bool) -> None:
    if type_a.is_ptr():
        if not allow_ptr_a:
            raise IncompatibleTypeErrorImpl(type_a, type_b)
        # T* + U* with T != U
        if type_b.is_ptr() and (type_a != type_b):
            raise IncompatibleTypeErrorImpl(type_a, type_b)
        # T* + float
        if type_b.is_floating():
            raise IncompatibleTypeErrorImpl(type_a, type_b)


def binary_op_type_checking_impl(lhs: tl.tensor | numbers.Number, rhs: tl.tensor | numbers.Number, builder: ir.builder,
                                 allow_lhs_ptr=False, allow_rhs_ptr=False, arithmetic_check=True,
                                 div_or_mod=False) -> Tuple[tl.tensor, tl.tensor]:
    lhs_is_scalar = isinstance(lhs, numbers.Number)
    rhs_is_scalar = isinstance(rhs, numbers.Number)
    if lhs_is_scalar:
        lhs_scalar = lhs
        lhs = to_tensor(lhs, builder)
    if rhs_is_scalar:
        rhs_scalar = rhs
        rhs = to_tensor(rhs, builder)

    # implicit typecasting
    lhs_sca_ty = lhs.type.scalar
    rhs_sca_ty = rhs.type.scalar
    check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr)
    check_ptr_type_impl(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr)
    if arithmetic_check and not lhs_sca_ty.is_ptr() and not rhs_sca_ty.is_ptr():
        ret_sca_ty = computation_type_impl(lhs_sca_ty, lhs_is_scalar, rhs_sca_ty, rhs_is_scalar, div_or_mod)
        if (lhs_is_scalar and lhs_scalar < 0 and ret_sca_ty.is_int_unsigned()
                or rhs_is_scalar and rhs_scalar < 0 and ret_sca_ty.is_int_unsigned()):
            raise ValueError("Cannot perform a binary operation between an unsigned tensor and a negative scalar. "
                             "Perform a explicit cast on one of them.")
        if ret_sca_ty.is_int():
            if lhs_is_scalar and not (ret_sca_ty.get_int_min_value() <= lhs_scalar <= ret_sca_ty.get_int_max_value()):
                raise ValueError(f"Scalar {lhs_scalar} is out of range for type {ret_sca_ty}")
            if rhs_is_scalar and not (ret_sca_ty.get_int_min_value() <= rhs_scalar <= ret_sca_ty.get_int_max_value()):
                raise ValueError(f"Scalar {rhs_scalar} is out of range for type {ret_sca_ty}")
        lhs = full(
            (), lhs_scalar, dtype=ret_sca_ty, builder=builder) if lhs_is_scalar else cast(lhs, ret_sca_ty, builder)
        rhs = full(
            (), rhs_scalar, dtype=ret_sca_ty, builder=builder) if rhs_is_scalar else cast(rhs, ret_sca_ty, builder)

    # implicit broadcasting
    lhs, rhs = broadcast_impl_value(lhs, rhs, builder)
    return lhs, rhs


def binary_op_sanitize_overflow_impl(lhs: tl.tensor, rhs: tl.tensor, builder: ir.builder, binary_op: callable):
    if lhs.type.scalar.int_bitwidth >= 64 or not builder.options.sanitize_overflow:
        return
    lhs_sca_ty = lhs.type.scalar
    rhs_sca_ty = rhs.type.scalar
    assert lhs_sca_ty == rhs_sca_ty
    assert lhs_sca_ty.is_int()
    lhs = cast(lhs, tl.int64, builder)
    rhs = cast(rhs, tl.int64, builder)
    ret = binary_op(lhs, rhs, False, builder)
    max_value = lhs_sca_ty.get_int_max_value()
    max_value = tl.tensor(builder.get_int64(max_value), tl.int64)
    min_value = lhs_sca_ty.get_int_min_value()
    min_value = tl.tensor(builder.get_int64(min_value), tl.int64)
    cond = and_(less_equal(ret, max_value, builder), greater_equal(ret, min_value, builder), builder)
    msg = f"int{lhs_sca_ty.int_bitwidth} overflow detected for operation {binary_op.__name__}"
    device_assert(cond, msg, builder)


def add(input: tl.tensor | numbers.Number, other: tl.tensor | numbers.Number, sanitize_overflow: bool,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if input_scalar_ty.is_ptr() and other_scalar_ty.is_ptr():
        raise TypeError("cannot add pointers together")

    # offset + ptr
    # ptr + offset
    if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
        input, other = other, input
        input_scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
    if input_scalar_ty.is_ptr():
        other_handle = other.handle
        if other.dtype.is_int_unsigned() and other.dtype.int_bitwidth < 64:
            # addptr treats offset as signed. Zero-extend unsigned offsets to ensure they're positive
            if other.type.is_block():
                i64_ty = tl.block_type(tl.int64, other.type.get_block_shapes()).to_ir(builder)
            else:
                i64_ty = tl.int64.to_ir(builder)
            other_handle = builder.create_int_cast(other.handle, i64_ty, False)
        return tl.tensor(builder.create_addptr(input.handle, other_handle), input.type)
    # float + float
    elif input_scalar_ty.is_floating():
        return tl.tensor(builder.create_fadd(input.handle, other.handle), input.type)
    # int + int
    elif input_scalar_ty.is_int():
        if sanitize_overflow:
            binary_op_sanitize_overflow_impl(input, other, builder, add)
        return tl.tensor(builder.create_add(input.handle, other.handle), input.type)
    raise TypeError(f"unexpected type {input_scalar_ty}")


def sub(input: tl.tensor | numbers.Number, other: tl.tensor | numbers.Number, sanitize_overflow: bool,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, True, False)
    scalar_ty = input.type.scalar
    # ptr - offset
    if scalar_ty.is_ptr():
        return tl.tensor(builder.create_addptr(input.handle, minus(other, builder).handle), input.type)
    # float - float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fsub(input.handle, other.handle), input.type)
    # int - int
    elif scalar_ty.is_int():
        if sanitize_overflow:
            binary_op_sanitize_overflow_impl(input, other, builder, sub)
        return tl.tensor(builder.create_sub(input.handle, other.handle), input.type)
    raise TypeError(f"unexpected type {scalar_ty}")


def mul(input: tl.tensor | numbers.Number, other: tl.tensor | numbers.Number, sanitize_overflow: bool,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float * float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fmul(input.handle, other.handle), input.type)
    # int * int
    elif scalar_ty.is_int():
        if sanitize_overflow:
            binary_op_sanitize_overflow_impl(input, other, builder, mul)
        return tl.tensor(builder.create_mul(input.handle, other.handle), input.type)
    raise TypeError(f"unexpected type {scalar_ty}")


def truediv(input: tl.tensor | numbers.Number, other: tl.tensor | numbers.Number, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float / int
    if input_scalar_ty.is_floating() and other_scalar_ty.is_int():
        other = cast(other, input_scalar_ty, builder)
    # int / float
    elif input_scalar_ty.is_int() and other_scalar_ty.is_floating():
        input = cast(input, other_scalar_ty, builder)
    # int / int (cast to tl.float32)
    elif input_scalar_ty.is_int() and other_scalar_ty.is_int():
        input = cast(input, tl.float32, builder)
        other = cast(other, tl.float32, builder)
    # float / float (cast to the highest exponent type)
    elif input_scalar_ty.is_floating() and other_scalar_ty.is_floating():
        if input_scalar_ty.fp_mantissa_width > other_scalar_ty.fp_mantissa_width:
            other = cast(other, input_scalar_ty, builder)
        else:
            input = cast(input, other_scalar_ty, builder)
    # unreachable
    else:
        raise TypeError(f"unexpected type {input_scalar_ty}")
    return tl.tensor(builder.create_fdiv(input.handle, other.handle), input.type)


def floordiv(input: tl.tensor | numbers.Number, other: tl.tensor | numbers.Number, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if input_scalar_ty.is_int() and other_scalar_ty.is_int():
        ret_ty = integer_promote_impl(input_scalar_ty, other_scalar_ty)
        input = cast(input, ret_ty, builder)
        other = cast(other, ret_ty, builder)
        if ret_ty.is_int_signed():
            return tl.tensor(builder.create_sdiv(input.handle, other.handle), input.type)
        else:
            return tl.tensor(builder.create_udiv(input.handle, other.handle), input.type)
    raise TypeError(f"unexpected type {input_scalar_ty}")


def fdiv(input: tl.tensor | numbers.Number, other: tl.tensor | numbers.Number, ieee_rounding: bool,
         builder: ir.builder) -> tl.tensor:
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if not input_scalar_ty.is_floating() or not other_scalar_ty.is_floating():
        raise TypeError("both operands of fdiv must have floating scalar type")
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, False, True)
    ret = builder.create_fdiv(input.handle, other.handle)
    return tl.tensor(ret, input.type)


def mod(input: tl.tensor | numbers.Number, other: tl.tensor | numbers.Number, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float % float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_frem(input.handle, other.handle), input.type)
    # % int
    elif scalar_ty.is_int():
        if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
            raise TypeError("Cannot mod " + scalar_ty.__repr__() + " by " + other_scalar_ty.__repr__() + " "
                            "because they have different signedness;"
                            "this is unlikely to result in a useful answer. Cast them to the same signedness.")
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_srem(input.handle, other.handle), input.type)
        else:
            return tl.tensor(builder.create_urem(input.handle, other.handle), input.type)
    raise TypeError(f"unexpected type {scalar_ty}")


##############
# other arithmetic ops
##############


def minimum(x: tl.tensor, y: tl.tensor, propagate_nan: tl.PropagateNan, builder: ir.builder):
    x, y = binary_op_type_checking_impl(x, y, builder)
    dtype = x.dtype
    if dtype.is_floating():
        if propagate_nan == tl.PropagateNan.ALL:
            return tl.tensor(builder.create_minimumf(x.handle, y.handle), x.type)
        elif propagate_nan == tl.PropagateNan.NONE:
            return tl.tensor(builder.create_minnumf(x.handle, y.handle), x.type)
        else:
            raise ValueError(f"Unexpected propagate_nan {propagate_nan}")
    elif dtype.is_int_signed():
        return tl.tensor(builder.create_minsi(x.handle, y.handle), x.type)
    elif dtype.is_int_unsigned():
        return tl.tensor(builder.create_minui(x.handle, y.handle), x.type)
    else:
        raise TypeError(f"Unexpected dtype {dtype}")


def maximum(x: tl.tensor, y: tl.tensor, propagate_nan: tl.PropagateNan, builder: ir.builder):
    x, y = binary_op_type_checking_impl(x, y, builder)
    dtype = x.dtype
    if dtype.is_floating():
        if propagate_nan == tl.PropagateNan.ALL:
            return tl.tensor(builder.create_maximumf(x.handle, y.handle), x.type)
        elif propagate_nan == tl.PropagateNan.NONE:
            return tl.tensor(builder.create_maxnumf(x.handle, y.handle), x.type)
        else:
            raise ValueError(f"Unexpected propagate_nan {propagate_nan}")
    elif dtype.is_int_signed():
        return tl.tensor(builder.create_maxsi(x.handle, y.handle), x.type)
    elif dtype.is_int_unsigned():
        return tl.tensor(builder.create_maxui(x.handle, y.handle), x.type)
    else:
        raise TypeError(f"Unexpected dtype {dtype}")


def clamp(x: tl.tensor, min: tl.tensor, max: tl.tensor, propagate_nan: tl.PropagateNan, builder: ir.builder):
    min, max = binary_op_type_checking_impl(min, max, builder)
    x, min = binary_op_type_checking_impl(x, min, builder)
    x, max = binary_op_type_checking_impl(x, max, builder)

    dtype = x.dtype
    if dtype.is_floating():
        return tl.tensor(builder.create_clampf(x.handle, min.handle, max.handle, propagate_nan), x.type)
    else:
        raise TypeError(f"Unexpected dtype {dtype}. Only floating point clamp is supported")


##############
# bitwise ops
##############


def bitwise_op_type_checking_impl(input: tl.tensor, other: tl.tensor,
                                  builder: ir.builder) -> Tuple[tl.tensor, tl.tensor]:
    input, other = binary_op_type_checking_impl(input, other, builder)
    input_sca_ty = input.type.scalar
    other_sca_ty = other.type.scalar
    if not input_sca_ty.is_int() or not other_sca_ty.is_int():
        raise IncompatibleTypeErrorImpl(input_sca_ty, other_sca_ty)
    ret_sca_ty = integer_promote_impl(input_sca_ty, other_sca_ty)
    if ret_sca_ty != input_sca_ty:
        input = cast(input, ret_sca_ty, builder)
    if ret_sca_ty != other_sca_ty:
        other = cast(other, ret_sca_ty, builder)
    return input, other


def and_(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_and(input.handle, other.handle), input.type)


def or_(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_or(input.handle, other.handle), input.type)


def xor_(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_xor(input.handle, other.handle), input.type)


def logical_and(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    if not input.type.is_int1():
        input = bitcast(input, tl.dtype("int1"), builder)
    if not other.type.is_int1():
        other = bitcast(other, tl.dtype("int1"), builder)
    return and_(input, other, builder)


def logical_or(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    if not input.type.is_int1():
        input = bitcast(input, tl.dtype("int1"), builder)
    if not other.type.is_int1():
        other = bitcast(other, tl.dtype("int1"), builder)
    return or_(input, other, builder)


def not_(input: tl.tensor, builder: ir.builder):
    if not input.type.is_int1():
        input = bitcast(input, tl.dtype("int1"), builder)
    return invert(input, builder)


def lshr(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_lshr(input.handle, other.handle), input.type)


def ashr(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_ashr(input.handle, other.handle), input.type)


def shl(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_shl(input.handle, other.handle), input.type)


# ===----------------------------------------------------------------------===//
#                               Unary Operators
# ===----------------------------------------------------------------------===//


def plus(input: tl.tensor) -> tl.tensor:
    return input


def minus(input: tl.tensor, builder: ir.builder) -> tl.tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr():
        raise ValueError("wrong type argument to unary minus (" + input_sca_ty.__repr__() + ")")
    _0 = tl.tensor(builder.get_null_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return sub(_0, input, True, builder)


def invert(input: tl.tensor, builder: tl.tensor) -> tl.tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr() or input_sca_ty.is_floating():
        raise ValueError("wrong type argument to unary invert (" + input_sca_ty.__repr__() + ")")
    _1 = tl.tensor(builder.get_all_ones_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return xor_(input, _1, builder)


# ===----------------------------------------------------------------------===//
#                               Comparison Operators
# ===----------------------------------------------------------------------===//
def _bool_like(v: tl.tensor) -> tl.block_type:
    if not v.type.is_block():
        return tl.int1
    shape = v.type.shape
    return tl.block_type(tl.int1, shape)


def greater_than(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float > float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOGT(input.handle, other.handle), _bool_like(input))
    # > int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_icmpSGT(input.handle, other.handle), _bool_like(input))
        else:
            return tl.tensor(builder.create_icmpUGT(input.handle, other.handle), _bool_like(input))
    raise TypeError(f"unexpected type {scalar_ty}")


def greater_equal(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float >= float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOGE(input.handle, other.handle), _bool_like(input))
    # >= int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_icmpSGE(input.handle, other.handle), _bool_like(input))
        else:
            return tl.tensor(builder.create_icmpUGE(input.handle, other.handle), _bool_like(input))
    raise TypeError(f"unexpected type {scalar_ty}")


def less_than(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float < float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOLT(input.handle, other.handle), _bool_like(input))
    # < int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_icmpSLT(input.handle, other.handle), _bool_like(input))
        else:
            return tl.tensor(builder.create_icmpULT(input.handle, other.handle), _bool_like(input))
    raise TypeError(f"unexpected type {scalar_ty}")


def less_equal(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float < float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOLE(input.handle, other.handle), _bool_like(input))
    # < int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_icmpSLE(input.handle, other.handle), _bool_like(input))
        else:
            return tl.tensor(builder.create_icmpULE(input.handle, other.handle), _bool_like(input))
    raise TypeError(f"unexpected type {scalar_ty}")


def equal(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float == float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOEQ(input.handle, other.handle), _bool_like(input))
    # == int
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_icmpEQ(input.handle, other.handle), _bool_like(input))
    raise TypeError(f"unexpected type {scalar_ty}")


def not_equal(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float == float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpUNE(input.handle, other.handle), _bool_like(input))
    # == int
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_icmpNE(input.handle, other.handle), _bool_like(input))
    raise TypeError(f"unexpected type {scalar_ty}")


# ===----------------------------------------------------------------------===//
#                               Block Creation
# ===----------------------------------------------------------------------===//


def arange(start: int, end: int, builder: ir.builder) -> tl.tensor:
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("arange's arguments must be of type tl.constexpr")
    is_start_int64 = bool(start >> 32)
    is_end_int64 = bool(end >> 32)
    if is_start_int64 or is_end_int64:
        raise ValueError("arange must fit in int32")
    if end <= start:
        raise ValueError("arange's end argument must be greater than the start argument")
    range = end - start
    if (range & (range - 1)) != 0:
        raise ValueError("arange's range must be a power of 2")
    shape = [range]
    ret_ty = tl.block_type(tl.int32, shape)
    return tl.tensor(builder.create_make_range(start, end), ret_ty)


def full(shape: List[int], value, dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    if isinstance(value, tl.tensor):
        assert value.numel.value == 1, "only accepts size-1 tensor"
        value = cast(value, dtype, builder)
    else:
        # scalar
        if dtype is None:
            raise ValueError("dtype must be specified when value is not a tensor")
        if value == 0:
            value = builder.get_null_value(dtype.to_ir(builder))
        else:
            get_value_fn = getattr(builder, f"get_{dtype.name}")
            value = get_value_fn(value)
        value = tl.tensor(value, dtype)

    return splat(value, shape, builder)


# ===----------------------------------------------------------------------===//
#                               Shape Manipulation
# ===----------------------------------------------------------------------===//


def splat(value: tl.tensor, shape: List[int], builder: ir.builder) -> tl.tensor:
    assert not value.type.is_block(), "Cannot splat a block tensor"
    if len(shape) == 0:
        return value
    ret_ty = tl.block_type(value.dtype, shape)
    return tl.tensor(builder.create_splat(value.handle, shape), ret_ty)


def reshape(input: tl.tensor, dst_shape: List[int], can_reorder: bool, builder: ir.builder) -> tl.tensor:
    numel = 1
    for s in dst_shape:
        numel *= s
    if input.type.numel != numel:
        raise ValueError("reshape() cannot change total number of elements in tensor")
    ret_ty = tl.block_type(input.type.scalar, dst_shape)
    return tl.tensor(builder.create_reshape(input.handle, dst_shape, can_reorder), ret_ty)


def expand_dims(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    dst_shape = [tl._constexpr_to_value(x) for x in input.shape]
    dst_shape.insert(axis, 1)

    if not input.type.is_block():
        return splat(input, shape=dst_shape, builder=builder)

    ret_ty = tl.block_type(input.type.scalar, dst_shape)
    return tl.tensor(builder.create_expand_dims(input.handle, axis), ret_ty)


def cat(lhs: tl.tensor, rhs: tl.tensor, can_reorder: bool, builder: ir.builder) -> tl.tensor:
    assert can_reorder, "current implementation of `cat` always may reorder elements"
    assert len(lhs.shape) == 1
    ret_type = tl.block_type(lhs.type.scalar, [lhs.shape[0] + rhs.shape[0]])
    return tl.tensor(builder.create_cat(lhs.handle, rhs.handle), ret_type)


def join(a: tl.tensor, b: tl.tensor, builder: ir.builder) -> tl.tensor:
    a, b = broadcast_impl_value(a, b, builder)

    # The IR can't handle joining two scalars, so upcast them to 1D tensors,
    # then downcast the result.
    was_rank_1 = a.shape == []
    if was_rank_1:
        a = expand_dims(a, 0, builder)
        b = expand_dims(b, 0, builder)

    if isinstance(a.shape[-1], tl.constexpr):
        two = tl.constexpr(2)
    else:
        two = 2
    new_shape = a.shape + [two]

    ret_type = tl.block_type(a.type.scalar, new_shape)
    ret = tl.tensor(builder.create_join(a.handle, b.handle), ret_type)

    if was_rank_1:
        ret = reshape(ret, [2], can_reorder=False, builder=builder)

    return ret


def split(a: tl.tensor, builder: ir.builder) -> Tuple[tl.tensor, tl.tensor]:
    assert (len(a.shape) > 0)
    assert (tl._constexpr_to_value(a.shape[-1]) == 2)

    new_shape = a.shape[:-1]
    ret_type = tl.block_type(a.type.scalar, new_shape)
    outLHS, outRHS = builder.create_split(a.handle)
    return (
        tl.tensor(outLHS, ret_type),
        tl.tensor(outRHS, ret_type),
    )


def permute(input: tl.tensor, dims: Tuple[int], builder: ir.builder) -> tl.tensor:
    if len(input.shape) != len(dims):
        raise ValueError("permute dims must have the same length as input shape")
    if sorted(tl._constexpr_to_value(d) for d in dims) != list(range(len(dims))):
        raise ValueError(f"permute dims must be a permutation of 0, 1, ..., n-1, but were {dims}")

    ret_type = tl.block_type(input.type.scalar, [input.shape[d] for d in dims])
    return tl.tensor(builder.create_trans(input.handle, dims), ret_type)


def broadcast_impl_shape(input: tl.tensor, shape: List[int], builder: ir.builder) -> tl.tensor:
    if not input.type.is_block():
        ret_ty = tl.block_type(input.type, shape)
        return tl.tensor(builder.create_splat(input.handle, shape), ret_ty)
    src_shape = input.type.get_block_shapes()
    if len(src_shape) != len(shape):
        raise ValueError(f"Cannot broadcast, rank mismatch: {src_shape}, {shape}")
    if shape == src_shape:
        return input
    for i, item in enumerate(src_shape):
        if shape[i] != item and item != 1:
            raise ValueError(f"Cannot broadcast, the expanded size of the tensor ({shape[i]})"
                             f" must match the existing size ({item}) at non-singleton dimension"
                             f" {i}: {src_shape}, {shape}")
    ret_ty = tl.block_type(input.type.scalar, shape)
    return tl.tensor(builder.create_broadcast(input.handle, shape), ret_ty)


def broadcast_impl_value(lhs: tl.tensor, rhs: tl.tensor, builder: ir.builder) -> tl.tensor:
    lhs_ty = lhs.type
    rhs_ty = rhs.type

    # make_shape_compatible(block, scalar)
    if lhs_ty.is_block() and not rhs_ty.is_block():
        rhs_ty = tl.block_type(rhs_ty.scalar, lhs_ty.shape)
        rhs = tl.tensor(builder.create_splat(rhs.handle, lhs_ty.get_block_shapes()), rhs_ty)
    # make_shape_compatible(scalar, block)
    elif not lhs_ty.is_block() and rhs_ty.is_block():
        lhs_ty = tl.block_type(lhs_ty.scalar, rhs_ty.shape)
        lhs = tl.tensor(builder.create_splat(lhs.handle, rhs_ty.get_block_shapes()), lhs_ty)
    # make_shape_compatible(block, block)
    elif lhs_ty.is_block() and rhs_ty.is_block():
        lhs_shape = lhs_ty.get_block_shapes()
        rhs_shape = rhs_ty.get_block_shapes()

        if len(lhs_shape) < len(rhs_shape):
            # Add new axes to lhs
            for _ in range(len(lhs_shape), len(rhs_shape)):
                lhs = tl.tensor(builder.create_expand_dims(lhs.handle, 0),
                                tl.block_type(lhs_ty.scalar, [1] + lhs_shape.values))
                lhs_ty = lhs.type
                lhs_shape = lhs_ty.get_block_shapes()
        elif len(rhs_shape) < len(lhs_shape):
            # Add new axes to rhs
            for _ in range(len(rhs_shape), len(lhs_shape)):
                rhs = tl.tensor(builder.create_expand_dims(rhs.handle, 0),
                                tl.block_type(rhs_ty.scalar, [1] + rhs_shape.values))
                rhs_ty = rhs.type
                rhs_shape = rhs_ty.get_block_shapes()
        assert len(rhs_shape) == len(lhs_shape)

        ret_shape = []
        for i, left in enumerate(lhs_shape):
            right = rhs_shape[i]
            if left == 1:
                ret_shape.append(right)
            elif (right == 1) or (right == left):
                ret_shape.append(left)
            else:
                raise ValueError("Cannot make_shape_compatible: incompatible dimensions "
                                 "at index " + str(i) + ": " + str(left) + " and " + str(right))
        if lhs_shape != ret_shape:
            ret_ty = tl.block_type(lhs_ty.scalar, ret_shape)
            lhs = tl.tensor(builder.create_broadcast(lhs.handle, ret_shape), ret_ty)
        if rhs_shape != ret_shape:
            ret_ty = tl.block_type(rhs_ty.scalar, ret_shape)
            rhs = tl.tensor(builder.create_broadcast(rhs.handle, ret_shape), ret_ty)
    # (scalar, scalar) => returns original blocks
    return lhs, rhs


#######
# cast
#######


def _str_to_rounding_mode(rounding_mode: Optional[str]):
    if rounding_mode is None:
        return None
    if rounding_mode == 'rtne':
        return ir.ROUNDING_MODE.RTNE
    if rounding_mode == 'rtz':
        return ir.ROUNDING_MODE.RTZ
    raise ValueError(f"Invalid rounding mode: {rounding_mode}. Supported rounding modes are 'rtne' and 'rtz'.")


def bitcast(input: tl.tensor, dst_ty: tl.dtype, builder: ir.builder) -> tl.tensor:
    src_ty = input.type
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty.scalar, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input
    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    if src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
        return cast(input, dst_ty, builder)
    # Bitcast
    src_bits = src_sca_ty.primitive_bitwidth
    dst_bits = dst_sca_ty.primitive_bitwidth
    if src_bits != dst_bits:
        raise ValueError("Cannot bitcast data-type of size " + str(src_bits) + " to "
                         "data-type of size " + str(dst_bits))
    return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)


def cast(input: tl.tensor, dst_ty: tl.dtype, builder: ir.builder,
         fp_downcast_rounding: Optional[str] = None) -> tl.tensor:
    src_ty = input.type
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty.scalar, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input

    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar

    # For fp downcasting default rounding mode should be RTNE, for all other conversions it should
    # not be set
    fp_downcast_rounding = _str_to_rounding_mode(fp_downcast_rounding)
    use_custom_rounding = False
    if dst_sca_ty.is_floating() and src_sca_ty.is_floating(
    ) and dst_sca_ty.primitive_bitwidth < src_sca_ty.primitive_bitwidth:
        if fp_downcast_rounding is None: fp_downcast_rounding = ir.ROUNDING_MODE.RTNE
        elif fp_downcast_rounding != ir.ROUNDING_MODE.RTNE: use_custom_rounding = True
    else:
        if fp_downcast_rounding is not None:
            raise ValueError("fp_downcast_rounding should be set only for truncating fp conversions. "
                             "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))

    if (src_sca_ty.is_fp8e4b15() or dst_sca_ty.is_fp8e4b15()):
        assert builder.codegen_fns.get(
            "convert_custom_types") is not None, "target doesn't provide conversion for this type."
        return builder.codegen_fns["convert_custom_types"](input, dst_ty, fp_downcast_rounding, _builder=builder)
    # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
    # and non-default rounding modes for downcasting
    if (src_sca_ty.is_fp8() and dst_sca_ty.is_floating()) or \
       (src_sca_ty.is_floating() and dst_sca_ty.is_fp8()) or \
       use_custom_rounding:
        return tl.tensor(builder.create_fp_to_fp(input.handle, dst_ty.to_ir(builder), fp_downcast_rounding), dst_ty)

    # bf16 <=> (not fp32)
    if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
       (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
        return cast(cast(input, tl.float32, builder), dst_sca_ty, builder)

    # Standard floating types' casting: truncation
    #   fp64 => fp32, fp16, bf16
    #   fp32 => fp16, bf16
    truncate_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
    if truncate_fp:
        return tl.tensor(builder.create_fp_trunc(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Standard floating types' casting: extension
    #   fp32 => fp64
    #   fp16 => fp32, fp64
    #   bf16 => fp32, fp64
    ext_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
    if ext_fp:
        return tl.tensor(builder.create_fp_ext(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting between integer types
    if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
       (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        else:
            return tl.tensor(builder.create_int_cast(input.handle, dst_ty.to_ir(builder), sign_extend), dst_ty)

    # Casting standard floating types to integer types
    if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        elif dst_sca_ty.is_int_signed():
            return tl.tensor(builder.create_fp_to_si(input.handle, dst_ty.to_ir(builder)), dst_ty)
        else:
            return tl.tensor(builder.create_fp_to_ui(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting integer types to standard floating types
    if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tl.tensor(builder.create_ui_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty)
        else:
            return tl.tensor(builder.create_si_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting pointer types to integer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
        bitwidth = dst_sca_ty.int_bitwidth
        if bitwidth == 64:
            return tl.tensor(builder.create_ptr_to_int(input.handle, dst_ty.to_ir(builder)), dst_ty)
        if bitwidth == 1:
            return not_equal(cast(input, tl.int64, builder), tl.tensor(builder.get_int64(0), tl.int64), builder)

    # Casting integer types to pointer types
    if src_sca_ty.is_int() and dst_sca_ty.is_ptr():
        return tl.tensor(builder.create_int_to_ptr(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting pointer types to pointer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)

    assert False, f'cannot cast {input} to {dst_ty}'


# ===----------------------------------------------------------------------===//
#                               Memory Operators
# ===----------------------------------------------------------------------===//


def _str_to_load_cache_modifier(cache_modifier):
    cache = ir.CACHE_MODIFIER.NONE  # default
    if cache_modifier:
        if cache_modifier == ".ca":
            cache = ir.CACHE_MODIFIER.CA
        elif cache_modifier == ".cg":
            cache = ir.CACHE_MODIFIER.CG
        elif cache_modifier == ".cv":
            cache = ir.CACHE_MODIFIER.CV
        else:
            raise ValueError(f"Cache modifier {cache_modifier} not supported")
    return cache


def _str_to_store_cache_modifier(cache_modifier):
    cache = ir.CACHE_MODIFIER.NONE  # default
    if cache_modifier:
        if cache_modifier == ".wb":
            cache = ir.CACHE_MODIFIER.WB
        elif cache_modifier == ".cg":
            cache = ir.CACHE_MODIFIER.CG
        elif cache_modifier == ".cs":
            cache = ir.CACHE_MODIFIER.CS
        elif cache_modifier == ".wt":
            cache = ir.CACHE_MODIFIER.WT
        else:
            raise ValueError(f"Cache modifier {cache_modifier} not supported")
    return cache


def _str_to_eviction_policy(eviction_policy):
    eviction = ir.EVICTION_POLICY.NORMAL  # default
    if eviction_policy:
        if eviction_policy == "evict_last":
            eviction = ir.EVICTION_POLICY.EVICT_LAST
        elif eviction_policy == "evict_first":
            eviction = ir.EVICTION_POLICY.EVICT_FIRST
        else:
            raise ValueError(f"Eviction policy {eviction_policy} not supported")
    return eviction


def _str_to_padding_option(padding_option):
    padding = None  # default
    if padding_option:
        if padding_option == "zero":
            padding = ir.PADDING_OPTION.PAD_ZERO
        elif padding_option == "nan":
            padding = ir.PADDING_OPTION.PAD_NAN
        else:
            raise ValueError(f"Padding option {padding_option} not supported")
    return padding


def _str_to_sem(sem_option):
    sem = ir.MEM_SEMANTIC.ACQUIRE_RELEASE
    if sem_option:
        if sem_option == "acquire":
            sem = ir.MEM_SEMANTIC.ACQUIRE
        elif sem_option == "release":
            sem = ir.MEM_SEMANTIC.RELEASE
        elif sem_option == "acq_rel":
            sem = ir.MEM_SEMANTIC.ACQUIRE_RELEASE
        elif sem_option == "relaxed":
            sem = ir.MEM_SEMANTIC.RELAXED
        else:
            raise ValueError(f"Memory semantic {sem_option} not supported")
    return sem


def _str_to_scope(scope_option):
    scope = ir.MEM_SYNC_SCOPE.GPU
    if scope_option:
        if scope_option == "gpu":
            scope = ir.MEM_SYNC_SCOPE.GPU
        elif scope_option == "cta":
            scope = ir.MEM_SYNC_SCOPE.CTA
        elif scope_option == "sys":
            scope = ir.MEM_SYNC_SCOPE.SYSTEM
        else:
            raise ValueError(f"Memory semantic {scope_option} not supported")
    return scope


def _canonicalize_boundary_check(boundary_check, block_shape):
    if boundary_check:
        if not hasattr(boundary_check, "__iter__"):
            boundary_check = [boundary_check]
        boundary_check = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in boundary_check]
        for dim in boundary_check:
            assert isinstance(dim, int) and 0 <= dim < len(block_shape)
        assert len(boundary_check) > 0
        assert len(boundary_check) == len(set(boundary_check)), "Duplicate dimension in `boundary_check`"
        return sorted(boundary_check)
    return ()


def _load_block_pointer(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder):
    # Load by a block pointer: `pointer_type<block_type<>>`
    # Block pointer can not have `mask` and `other` arguments
    if mask is not None or other is not None:
        raise ValueError("`mask` and `other` arguments cannot be specified for loading block pointers")

    elt_ty = ptr.type.element_ty.element_ty
    assert elt_ty != tl.int1, "`tl.int1` should be rewritten in `tl.make_block_ptr`"
    if elt_ty.is_int() and padding == ir.PADDING_OPTION.PAD_NAN:
        raise ValueError("Padding option `nan` is not supported for integer block pointers")

    # `dst_ty` is de-referenced type of the pointer type
    dst_ty = ptr.type.element_ty

    # Check `boundary_check` argument
    boundary_check = _canonicalize_boundary_check(boundary_check, dst_ty.get_block_shapes())

    # Build IR
    return tl.tensor(
        builder.create_tensor_pointer_load(ptr.handle, boundary_check, padding, cache, eviction, is_volatile), dst_ty)


def _load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder):
    # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
    if not ptr.type.scalar.is_ptr():
        raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")

    # Check `mask`, `other`, `boundary_check`, and `padding` arguments
    if mask is None and other is not None:
        raise ValueError("`other` cannot be provided without `mask`")
    if padding or boundary_check:
        raise ValueError("`padding_option` or `boundary_check` argument is not supported for loading a tensor of"
                         "pointers or loading a scalar. Because the compiler does not know the boundary; please "
                         "use block pointers (defined by `make_block_ptr`) instead")

    # For a pointer of scalar, check the type of `mask` and `other`
    if not ptr.type.is_block():
        if mask and mask.type.is_block():
            raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
        if other and other.type.is_block():
            raise ValueError("Other argument cannot be block type if pointer argument is not a block")

    # Make `mask` and `other` into the same shape as `ptr`
    if ptr.type.is_block():
        if mask is not None:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if other is not None:
            other = broadcast_impl_shape(other, ptr.type.get_block_shapes(), builder)

    # Get `pointer_type<elt_ty>` and `elt_ty`
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty

    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    is_bool = elt_ty == tl.int1
    if is_bool:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)

    # Cast `other` into `elt_ty` type
    if other is not None:
        other = cast(other, elt_ty, builder)

    # Create loaded result type `dst_ty`
    if ptr.type.is_block():
        shape = ptr.type.get_block_shapes()
        dst_ty = tl.block_type(elt_ty, shape)
    else:
        # Load by de-referencing the pointer of scalar
        dst_ty = elt_ty

    # Build IR
    if mask is None:
        ret = tl.tensor(builder.create_load(ptr.handle, cache, eviction, is_volatile), dst_ty)
    else:
        ret = tl.tensor(
            builder.create_masked_load(ptr.handle, mask.handle, other.handle if other else None, cache, eviction,
                                       is_volatile), dst_ty)
    if is_bool:
        ret = cast(ret, tl.int1, builder)
    return ret


def load(ptr: tl.tensor, mask: Optional[tl.tensor], other: Optional[tl.tensor], boundary_check: Tuple,
         padding_option: str, cache_modifier: str, eviction_policy: str, is_volatile: bool,
         builder: ir.builder) -> tl.tensor:
    # Cache, eviction and padding options
    cache = _str_to_load_cache_modifier(cache_modifier)
    eviction = _str_to_eviction_policy(eviction_policy)
    padding = _str_to_padding_option(padding_option)

    if ptr.type.is_ptr() and ptr.type.element_ty.is_block():
        # Load by a block pointer: `pointer_type<block_type<>>`
        return _load_block_pointer(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder)
    else:
        # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        return _load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder)


def reinterpret_tensor_descriptor(desc_ptr: tl.tensor, block_ty: tl.block_type, builder: ir.builder):
    handle = builder.create_reinterpret_tensor_descriptor(desc_ptr.handle, block_ty.to_ir(builder))
    return tl._experimental_tensor_descriptor_base(handle, block_ty)


def validate_descriptor_block(shape, dtype):
    if len(shape) != 2:
        return
    # Due to limitations of the shared memory encoding, the TMA bounding box has
    # to be at least as big as the swizzle tile.
    assert shape[0] >= 8, f"tensor descriptor block shape must have at least 8 rows, but got {shape[0]}"
    min_cols = 32 // dtype.primitive_bitwidth * 8
    assert shape[
        1] >= min_cols, f"{dtype} tensor descriptor block shape must have at least {min_cols} columns, but got {shape[1]}"


def descriptor_load(desc: tl._experimental_tensor_desciptor_base, offsets, cache_modifier: str, eviction_policy: str,
                    builder: ir.builder) -> tl.tensor:
    assert isinstance(desc, tl._experimental_tensor_descriptor_base)
    validate_descriptor_block(desc.block_shape, desc.dtype)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"

    offsets = _convert_to_ir_values(builder, offsets, require_i64=False)
    x = builder.create_descriptor_load(desc.handle, offsets, _str_to_load_cache_modifier(cache_modifier),
                                       _str_to_eviction_policy(eviction_policy))
    return tl.tensor(x, desc.block_type)


def descriptor_store(desc: tl._experimental_tensor_descriptor_base, value: tl.tensor, offsets,
                     builder: ir.builder) -> tl.tensor:
    assert isinstance(desc, tl._experimental_tensor_descriptor_base)
    validate_descriptor_block(desc.block_shape, desc.dtype)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"
    assert value.shape == desc.block_shape

    offsets = _convert_to_ir_values(builder, offsets, require_i64=False)
    return tl.tensor(builder.create_descriptor_store(desc.handle, value.handle, offsets), tl.void)


def descriptor_gather(desc, x_offsets, y_offset, cache_modifier: str, eviction_policy: str,
                      builder: ir.builder) -> tl.tensor:
    assert isinstance(desc, tl._experimental_tensor_descriptor_base)
    assert cache_modifier == "", "cache modifier is not supported yet"
    assert eviction_policy == "", "eviction policy is not supported yet"

    # Validate descriptor.
    assert len(desc.block_shape) == 2, f"descriptor must be 2D, but got {desc.block_shape}"
    assert desc.block_shape[0] == 1, f"descriptor block must have 1 row, but got {desc.block_shape}"

    # Validate offsets.
    assert len(x_offsets.shape) == 1, f"x offsets must be 1D, but got {x_offsets.shape}"

    # Validate minimum block size.
    assert x_offsets.shape[0] >= 8, f"descriptor gather must have at least 8 rows, but got {x_offsets.shape}"
    dtype = desc.dtype
    min_cols = 32 // dtype.primitive_bitwidth * 8
    assert desc.block_shape[
        1] >= min_cols, f"descriptor gather of {dtype} must have at least {min_cols} columns, but got {desc.block_shape[1]}"

    type = tl.block_type(desc.dtype, [x_offsets.shape[0], desc.block_shape[1]])
    y_offset = _convert_to_ir_values(builder, (y_offset, ), require_i64=False)[0]
    x = builder.create_descriptor_gather(desc.handle, x_offsets.handle, y_offset, type.to_ir(builder))
    return tl.tensor(x, type)


def descriptor_scatter(desc, value: tl.tensor, x_offsets, y_offset, builder: ir.builder) -> tl.tensor:
    assert isinstance(desc, tl._experimental_tensor_descriptor_base)

    # Validate descriptor.
    assert len(desc.block_shape) == 2, f"descriptor must be 2D, but got {desc.block_shape}"
    assert desc.block_shape[0] == 1, f"descriptor block must have 1 row, but got {desc.block_shape}"

    # Validate offsets.
    assert len(x_offsets.shape) == 1, f"x offsets must be 1D, but got {x_offsets.shapae}"

    # Validate minimum block size.
    assert x_offsets.shape[0] >= 8, f"descriptor scatter must have at least 8 rows, but got {x_offsets.shape}"
    dtype = desc.dtype
    min_cols = 32 // dtype.primitive_bitwidth * 8
    assert desc.block_shape[
        1] >= min_cols, f"descriptor scatter of {dtype} must have at least {min_cols} columns, but got {desc.block_shape[1]}"

    y_offset = _convert_to_ir_values(builder, (y_offset, ), require_i64=False)[0]
    builder.create_descriptor_scatter(desc.handle, value.handle, x_offsets.handle, y_offset)
    return tl.tensor(None, tl.void)


def tensormap_create(
    desc_ptr: tl.tensor,
    global_address: tl.tensor,
    box_dim: List[tl.tensor],
    global_dim: List[tl.tensor],
    global_stride: List[tl.tensor],
    element_stride: List[tl.tensor],
    elem_type: int,
    interleave_layout: int,
    swizzle_mode: int,
    fill_mode: int,
    builder: ir.builder,
) -> tl.tensor:
    assert not global_stride or global_stride[0].dtype == tl.int64
    return tl.tensor(
        builder.create_tensormap_create(
            desc_ptr.handle,
            global_address.handle,
            [x.handle for x in box_dim],
            [x.handle for x in global_dim],
            [x.handle for x in global_stride],
            [x.handle for x in element_stride],
            elem_type,
            interleave_layout,
            swizzle_mode,
            fill_mode,
        ),
        tl.void,
    )


def tensormap_fenceproxy_acquire(desc_ptr: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_tensormap_fenceproxy_acquire(desc_ptr.handle), tl.void)


def _store_block_pointer(ptr, val, mask, boundary_check, cache, eviction, builder):
    # Store by a block pointer: `pointer_type<block_type<>>`
    # Block pointers can not have the `mask` argument
    if mask is not None:
        raise ValueError("`mask` and `other` arguments cannot be specified for loading block pointers")

    # Check same shape and element type
    block_shape = ptr.type.element_ty.get_block_shapes()
    if not val.type.is_block():
        val = broadcast_impl_shape(val, block_shape, builder)
    assert val.type.is_block(), "Value argument must be block type or a scalar"
    assert block_shape == val.type.get_block_shapes(
    ), f"Block shape({block_shape}) and value shape({val.type.get_block_shapes()}) mismatch"
    assert ptr.type.element_ty.element_ty == val.type.element_ty, f"Block element type({ptr.type.element_ty.element_ty}) and value element type({val.type.element_ty}) mismatch"

    elt_ty = ptr.type.element_ty.element_ty
    assert elt_ty != tl.int1, "`tl.int1` should be rewritten in `tl.make_block_ptr`"

    # Check `boundary_check` argument
    boundary_check = _canonicalize_boundary_check(boundary_check, block_shape)

    # Cast to target data type
    val = cast(val, elt_ty, builder)

    # Build IR
    return tl.tensor(builder.create_tensor_pointer_store(ptr.handle, val.handle, boundary_check, cache, eviction),
                     tl.void)


def _store_legacy(ptr, val, mask, boundary_check, cache, eviction, builder):
    # Store by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
    if not ptr.type.scalar.is_ptr():
        raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.store`")

    # Check `boundary_check` argument
    if boundary_check:
        raise ValueError("`boundary_check` argument is not supported for storing a tensor of pointers or storing a "
                         "scalar. Because the compiler does not know the boundary; please use block pointers "
                         "(defined by `make_block_ptr`) instead")

    # For a pointer of scalar, check the type of `val` and `mask`
    if not ptr.type.is_block():
        if val.type.is_block():
            raise ValueError("Value argument cannot be block type if pointer argument is not a block")
        if mask and mask.type.is_block():
            raise ValueError("Mask argument cannot be block type if pointer argument is not a block")

    # Make `mask` and `val` into the same shape as `ptr`
    if ptr.type.is_block():
        val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
        if mask is not None:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)

    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty

    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    if elt_ty == tl.int1:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)

    # Cast to target data type
    val = cast(val, elt_ty, builder)

    # Build IR
    if mask is None:
        return tl.tensor(builder.create_store(ptr.handle, val.handle, cache, eviction), tl.void)
    if not mask.type.scalar.is_bool():
        raise ValueError("Mask must have boolean scalar type")
    return tl.tensor(builder.create_masked_store(ptr.handle, val.handle, mask.handle, cache, eviction), tl.void)


def store(ptr: tl.tensor, val: tl.tensor, mask: Optional[tl.tensor], boundary_check, cache_modifier: str,
          eviction_policy: str, builder: ir.builder) -> tl.tensor:
    # Cache and eviction options
    cache = _str_to_store_cache_modifier(cache_modifier)
    eviction = _str_to_eviction_policy(eviction_policy)

    if ptr.type.is_const() or ptr.type.scalar.is_const():
        raise ValueError("Cannot store to a constant pointer")

    if ptr.type.is_ptr() and ptr.type.element_ty.is_block():
        # Store by a block pointer: `pointer_type<block_type<>>`
        return _store_block_pointer(ptr, val, mask, boundary_check, cache, eviction, builder)
    else:
        # Store by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        return _store_legacy(ptr, val, mask, boundary_check, cache, eviction, builder)


#########
# atomic
#########


def atomic_cas(ptr: tl.tensor, cmp: tl.tensor, val: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    element_ty = ptr.type.scalar.element_ty
    if element_ty.primitive_bitwidth not in [16, 32, 64]:
        raise ValueError("atomic_cas only supports elements with width {16, 32, 64}")
    return tl.tensor(builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle, sem, scope), val.type)


def atom_red_typechecking_impl(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, op: str,
                               builder: ir.builder) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    if not ptr.type.scalar.is_ptr():
        raise ValueError("Pointer argument of store instruction is " + ptr.type.__repr__())
    if ptr.type.is_const() or ptr.type.element_ty.is_const():
        raise ValueError("Cannot store to a constant pointer")
    element_ty = ptr.type.scalar.element_ty
    if element_ty is tl.float16 and op != 'add':
        raise ValueError("atomic_" + op + " does not support fp16")
    if element_ty in [tl.int1, tl.int8, tl.int16, tl.bfloat16]:
        raise ValueError("atomic_" + op + " does not support " + str(element_ty))
    if ptr.type.is_block():
        if mask is not None:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if val is not None:
            val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
    val = cast(val, ptr.type.scalar.element_ty, builder)
    if mask is None:
        mask_ir = builder.get_int1(True)
        mask_ty = tl.int1
        if ptr.type.is_block():
            mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
            mask_ty = tl.block_type(tl.int1, ptr.type.get_block_shapes())
        mask = tl.tensor(mask_ir, mask_ty)
    return ptr, val, mask


def atomic_max(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'max', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    sca_ty = val.type.scalar
    # direct call to atomic_max for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)
        else:
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)
    # for float
    # return atomic_smax(i_ptr, i_val) if val >= 0
    # return atomic_umin(i_ptr, i_val) if val < 0
    if sca_ty not in {tl.float32, tl.float64}:
        raise TypeError(f"atomic_max not supported for dtype {sca_ty}")

    zero = full([], 0.0, sca_ty, builder)

    i_type = tl.int32 if sca_ty == tl.float32 else tl.int64
    i_val = bitcast(val, i_type, builder)
    i_ptr = bitcast(ptr, tl.pointer_type(i_type, 1), builder)
    ui_type = tl.uint32 if sca_ty == tl.float32 else tl.uint64
    ui_val = bitcast(val, ui_type, builder)
    ui_ptr = bitcast(ptr, tl.pointer_type(ui_type, 1), builder)
    pos = greater_equal(val, zero, builder)
    neg = less_than(val, zero, builder)
    pos_ret = tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, i_ptr.handle, i_val.handle,
                                  and_(mask, pos, builder).handle, sem, scope), i_val.type)
    neg_ret = tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, ui_ptr.handle, ui_val.handle,
                                  and_(mask, neg, builder).handle, sem, scope), ui_val.type)
    ret = where(pos, pos_ret, neg_ret, builder)
    return bitcast(ret, sca_ty, builder)


def atomic_min(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'min', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    sca_ty = val.type.scalar
    # direct call to atomic_min for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)
        else:
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)
    # for float
    # return atomic_smin(i_ptr, i_val) if val >= 0
    # return atomic_umax(i_ptr, i_val) if val < 0
    if sca_ty not in {tl.float32, tl.float64}:
        raise TypeError(f"atomic_min not supported for dtype {sca_ty}")

    zero = full([], 0.0, sca_ty, builder)

    i_type = tl.int32 if sca_ty == tl.float32 else tl.int64
    i_val = bitcast(val, i_type, builder)
    i_ptr = bitcast(ptr, tl.pointer_type(i_type, 1), builder)
    ui_type = tl.uint32 if sca_ty == tl.float32 else tl.uint64
    ui_val = bitcast(val, ui_type, builder)
    ui_ptr = bitcast(ptr, tl.pointer_type(ui_type, 1), builder)
    pos = greater_equal(val, zero, builder)
    neg = less_than(val, zero, builder)
    pos_ret = tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, i_ptr.handle, i_val.handle,
                                  and_(mask, pos, builder).handle, sem, scope), i_val.type)
    neg_ret = tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX, ui_ptr.handle, ui_val.handle,
                                  and_(mask, neg, builder).handle, sem, scope), ui_ptr.type)
    ret = where(pos, pos_ret, neg_ret, builder)
    return bitcast(ret, sca_ty, builder)


def atomic_add(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'add', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    sca_ty = val.type.scalar
    op = ir.ATOMIC_OP.FADD if sca_ty.is_floating() else ir.ATOMIC_OP.ADD
    return tl.tensor(builder.create_atomic_rmw(op, ptr.handle, val.handle, mask.handle, sem, scope), val.type)


def atomic_and(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'and', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.AND, ptr.handle, val.handle, mask.handle, sem, scope),
                     val.type)


def atomic_or(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'or', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.OR, ptr.handle, val.handle, mask.handle, sem, scope),
                     val.type)


def atomic_xor(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'xor', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XOR, ptr.handle, val.handle, mask.handle, sem, scope),
                     val.type)


def atomic_xchg(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str,
                builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'xchg', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XCHG, ptr.handle, val.handle, mask.handle, sem, scope),
                     val.type)


# ===----------------------------------------------------------------------===//
#                               Linear Algebra
# ===----------------------------------------------------------------------===//


def _str_to_dot_input_precision(input_precision, builder):
    assert input_precision.lower() in builder.options.allowed_dot_input_precisions, \
        f"input_precision must be one of {builder.options.allowed_dot_input_precisions}. Got {input_precision}"
    input_precision = input_precision.upper()
    if input_precision == "TF32X3":
        input_precision = "TF32x3"
    return getattr(ir.INPUT_PRECISION, input_precision)


def dot(lhs: tl.tensor, rhs: tl.tensor, acc: tl.tensor, input_precision: Optional[str], max_num_imprecise_acc: int,
        out_dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()

    if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
        # All combinations of supported fp8 x fp8 are permitted
        pass
    else:
        assert lhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16,
                             tl.float32), f"Unsupported lhs dtype {lhs.dtype}"
        assert rhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16,
                             tl.float32), f"Unsupported rhs dtype {rhs.dtype}"
        assert lhs.dtype == rhs.dtype, f"Both operands must be same dtype. Got {lhs.dtype} and {rhs.dtype}"

    if lhs.dtype.is_fp8e4b15() or rhs.dtype.is_fp8e4b15():
        # We upcast because there's no fp8e4b15 type in MLIR
        lhs = cast(lhs, tl.float16, builder)
        rhs = cast(rhs, tl.float16, builder)

    if input_precision is None:
        input_precision = builder.options.default_dot_input_precision

    input_precision = _str_to_dot_input_precision(input_precision, builder)

    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
    assert lhs.shape[-1].value == rhs.shape[
        -2].value, f"First input shape ({lhs.shape}) and second input shape {rhs.shape} are not compatible for matmul (second index of first shape ({lhs.shape[-1].value}) must be equal to first index of second shape ({rhs.shape[-2].value})"
    assert builder.codegen_fns.get("min_dot_size") is not None, "target doesn't provide lower shape bounds for dot."
    min_dot_size = builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
    assert lhs.shape[-2].value >= min_dot_size[0] and lhs.shape[-1].value >= min_dot_size[2] \
        and rhs.shape[-1].value >= min_dot_size[1], \
            f"Input shapes should have M >= {min_dot_size[0]}, N >= {min_dot_size[1]} and K >= {min_dot_size[2]}"
    if lhs.type.scalar.is_int():
        assert lhs.type.scalar == tl.int8, "only int8 supported!"
        _0 = builder.get_int32(0)
        ret_scalar_ty = tl.int32
    elif out_dtype.is_bf16():
        raise ValueError(
            "out_dtype=bfloat16 is unsupported. Please use out_dtype=float32/float16 and cast with `.to(tl.bfloat16)`")
    elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
        _0 = builder.get_fp32(0)
        ret_scalar_ty = tl.float32
    else:
        _0 = builder.get_fp16(0) if out_dtype.is_fp16() else builder.get_fp32(0)
        ret_scalar_ty = out_dtype

    M = lhs.type.shape[-2]
    N = rhs.type.shape[-1]
    K = lhs.type.shape[-1]
    B = lhs.type.shape[0] if lhs_rank == 3 else None
    ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])
    if acc is None:
        acc_handle = builder.create_splat(_0, [B, M, N] if B else [M, N])
    else:
        acc_handle = acc.handle
        assert acc.type == ret_ty

    # max_num_imprecise_acc only applies to fp8 -> fp32 dot on sm_90
    if max_num_imprecise_acc is None:
        if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
            max_num_imprecise_acc = builder.options.max_num_imprecise_acc_default
        else:
            max_num_imprecise_acc = 0
    else:
        if lhs.dtype.is_fp8() and rhs.dtype.is_fp8() and max_num_imprecise_acc > K:
            raise ValueError(f"max_num_imprecise_acc ({max_num_imprecise_acc}) must be <= K ({K})")

    return tl.tensor(builder.create_dot(lhs.handle, rhs.handle, acc_handle, input_precision, max_num_imprecise_acc),
                     ret_ty)


def _str_to_fp_type(float_format: str):
    ty_enum = getattr(ir.ScaleDotElemTypeTY, float_format.upper(), None)
    if ty_enum is None:
        raise ValueError(f"Invalid float format: {float_format}.")
    return ty_enum


def _bitcast_to_fp_type(val: tl.tensor, float_format: str, builder: ir.builder):
    """
    If float_format is subbyte, make sure it's packed as uint8 and return it.
    Otherwise, return a tensor (perhaps bitcasting) of the specified float format.
    """
    triton_ty = {"e5m2": tl.float8e5, "e4m3": tl.float8e4nv, "bf16": tl.bfloat16, "fp16": tl.float16}.get(float_format)
    if triton_ty is None:
        assert float_format == "e2m1", f"Internal Error: Unexpected float format: {float_format}"
        assert val.dtype == tl.uint8, f"e2m1 format must be packed as uint8. Got {val.dtype}"
        return val
    if val.dtype == triton_ty:
        return val
    else:
        unsigned_ty = {"e5m2": tl.uint8, "e4m3": tl.uint8, "bf16": tl.uint16, "fp16": tl.uint16}[float_format]
        assert val.dtype == unsigned_ty, f"Unexpected dtype for {float_format}. Got {val.dtype}"
        return bitcast(val, triton_ty, builder)


def dot_scaled(lhs: tl.tensor, lhs_scale: tl.tensor, lhs_format: str, rhs: tl.tensor, rhs_scale: Optional[tl.tensor],
               rhs_format: str, acc: tl.tensor | None, fast_math: bool, out_dtype: tl.dtype,
               builder: ir.builder) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()
    #TODO: validate types.
    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
    lhs_format: str = lhs_format.value
    rhs_format: str = rhs_format.value
    lhs_format_enum = _str_to_fp_type(lhs_format)
    rhs_format_enum = _str_to_fp_type(rhs_format)
    allowed_formats = {"e2m1", "e4m3", "e5m2", "bf16", "fp16"}
    assert lhs_format in allowed_formats, f"NYI: lhs_format {lhs_format}"
    assert rhs_format in allowed_formats, f"NYI: rhs_format {rhs_format}"
    rhs_scale_is_none = rhs_scale is None or (isinstance(rhs_scale, tl.constexpr) and rhs_scale.value is None)
    lhs_scale_is_none = lhs_scale is None or (isinstance(lhs_scale, tl.constexpr) and lhs_scale.value is None)
    lhs = _bitcast_to_fp_type(lhs, lhs_format, builder)
    rhs = _bitcast_to_fp_type(rhs, rhs_format, builder)

    M = lhs.type.shape[-2]
    K, N = rhs.type.shape[-2:]
    PACKED_A = 2 if lhs_format == "e2m1" else 1
    PACKED_B = 2 if rhs_format == "e2m1" else 1
    assert K * PACKED_B == PACKED_A * lhs.type.shape[
        -1], f"Reduction dimension should pack the same number of elements; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
    #assert K * PACKED_B >= 64, f"scaled_dot NYI for K < 64. Got {K=}"
    B = lhs.type.shape[0] if lhs_rank == 3 else None

    ret_ty = tl.block_type(out_dtype, [B, M, N] if B else [M, N])
    _0 = builder.get_fp32(0)
    if acc is None:
        acc_handle = builder.create_splat(_0, [B, M, N] if B else [M, N])
    else:
        acc_handle = acc.handle
        assert acc.type == ret_ty
    rhs_scale_handle = None if rhs_scale_is_none else rhs_scale.handle
    lhs_scale_handle = None if lhs_scale_is_none else lhs_scale.handle
    return tl.tensor(
        builder.create_dot_scaled(lhs.handle, lhs_scale_handle, lhs_format_enum, rhs.handle, rhs_scale_handle,
                                  rhs_format_enum, fast_math, acc_handle), ret_ty)


# ===----------------------------------------------------------------------===//
#                               Indexing
# ===----------------------------------------------------------------------===//


def where(condition: tl.tensor, x: tl.tensor, y: tl.tensor, builder: ir.builder) -> tl.tensor:
    if condition.dtype != tl.int1:
        warnings.warn(
            f"tl.where with a non-boolean condition is deprecated and will error out in a future triton release. Got {condition.dtype}"
        )
    condition = cast(condition, tl.int1, builder)
    x, y = binary_op_type_checking_impl(x, y, builder, True, True)
    # x, y are broadcasted
    if condition.type.is_block():
        condition, x = broadcast_impl_value(condition, x, builder)
        x, y = broadcast_impl_value(x, y, builder)
    else:
        condition, _ = broadcast_impl_value(condition, x, builder)
    ret_ty = x.type
    return tl.tensor(builder.create_select(condition.handle, x.handle, y.handle), ret_ty)


# ===----------------------------------------------------------------------===//
#                               Reduction
# ===----------------------------------------------------------------------===


def wrap_tensor(x, scalar_ty, ret_shape):
    if ret_shape:
        res_ty = tl.block_type(scalar_ty, ret_shape)
    else:
        # 0d-tensor -> scalar
        res_ty = scalar_ty
    return tl.tensor(x, res_ty)


def reduction(inputs: Sequence[tl.tensor], axis: int, region_builder_fn, builder: ir.builder) -> Tuple[tl.tensor, ...]:
    if axis is None:
        inputs = tuple(reshape(t, [t.numel.value], can_reorder=True, builder=builder) for t in inputs)
        axis = 0
    # get result shape
    shape = inputs[0].type.shape
    rank = len(shape)
    assert axis < rank, f"reduction axis must be < inputs rank ({rank})"
    ret_shape = [s for i, s in enumerate(shape) if i != axis]
    assert all(t.type.shape == shape for t in inputs), "all reduction inputs must have the same shape"

    reduce_op = builder.create_reduce([t.handle for t in inputs], axis)
    region_builder_fn(reduce_op)
    reduce_op.verify()

    return tuple(wrap_tensor(reduce_op.get_result(i), inputs[i].type.scalar, ret_shape) for i in range(len(inputs)))


# ===----------------------------------------------------------------------===
#                               Associative Scan
# ===----------------------------------------------------------------------===


def associative_scan(inputs: Sequence[tl.tensor], axis: int, region_builder_fn, reverse: bool,
                     builder: ir.builder) -> Tuple[tl.tensor, ...]:
    shape = inputs[0].type.shape
    rank = len(shape)

    assert -rank <= axis < rank, f"scan axis {axis} must be < inputs rank ({rank})"

    if axis < 0:
        axis += rank

    for t in inputs:
        assert t.type.shape == shape, "all scan inputs must have the same shape"

    scan_op = builder.create_scan([t.handle for t in inputs], axis, reverse)
    region_builder_fn(scan_op)
    scan_op.verify()

    return tuple(wrap_tensor(scan_op.get_result(i), inputs[i].type.scalar, shape) for i in range(len(inputs)))


# ===----------------------------------------------------------------------===
#                               Gather
# ===----------------------------------------------------------------------===


def gather(src: tl.tensor, index: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    assert index.dtype.is_int(), "index must be an integer tensor"

    rank = len(src.type.shape)
    assert len(index.type.shape) == rank, "source and index tensors must have the same rank"

    assert -rank <= axis < rank, f"gather axis {axis} must be < source rank ({rank})"
    if axis < 0:
        axis += rank

    for d in range(rank):
        if d == axis:
            continue
        assert index.type.shape[d] == src.type.shape[d], f"index dim {axis} must match the corresponding source dim"

    gather = builder.create_gather(src.handle, index.handle, axis)
    return wrap_tensor(gather, src.type.scalar, index.type.shape)


# ===----------------------------------------------------------------------===
#                               Histogram
# ===----------------------------------------------------------------------===


def histogram(input: tl.tensor, num_bins: int, builder: ir.builder) -> tl.tensor:
    assert len(input.shape) == 1, "histogram only supports 1D input"
    assert input.dtype.is_int(), "histogram only supports integer input"
    return tl.tensor(builder.create_histogram(input.handle, num_bins), tl.block_type(tl.int32, [num_bins]))


def multiple_of(x: tl.tensor, values: List[int]) -> tl.tensor:
    if max(1, len(x.shape)) != len(values):
        raise ValueError("Shape of input to multiple_of does not match the length of values")
    x.handle.set_attr("tt.divisibility", ir.make_attr(values, x.handle.get_context()))
    return x


def max_contiguous(x: tl.tensor, values: List[int]) -> tl.tensor:
    if len(x.shape) != len(values):
        raise ValueError("Shape of input to max_contiguous does not match the length of values")
    x.handle.set_attr("tt.contiguity", ir.make_attr(values, x.handle.get_context()))
    return x


def max_constancy(x: tl.tensor, values: List[int]) -> tl.tensor:
    if len(x.shape) != len(values):
        raise ValueError("Shape of input to max_constancy does not match the length of values")
    x.handle.set_attr("tt.constancy", ir.make_attr(values, x.handle.get_context()))
    return x


def debug_barrier(builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_barrier(), tl.void)


def device_print(prefix: str, args: List[tl.tensor], hex: bool, builder: ir.builder) -> tl.tensor:
    # It makes sense visually for prefix to end in ": "; make it so.  Also,
    # non-empty prefixes should start with " ".
    if not prefix.endswith(" ") and args:
        prefix += " "
    if not prefix.endswith(": ") and args:
        prefix = prefix[:-1] + ": "
    if len(prefix) > 2 and not prefix.startswith(" "):
        prefix = " " + prefix

    new_args = [arg.handle for arg in args]
    is_signed = [arg.dtype in (tl.int1, tl.int8, tl.int16, tl.int32, tl.int64) for arg in args]
    return tl.tensor(builder.create_print(prefix, hex, new_args, is_signed), tl.void)


def device_assert(cond: tl.tensor, msg: str, builder: ir.builder) -> tl.tensor:
    if not builder.options.debug:
        return
    return tl.tensor(builder.create_assert(cond.handle, msg), tl.void)


def assume(cond, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_assume(cond.handle), tl.void)


def _convert_elem_to_ir_value(builder, elem, require_i64):
    if isinstance(elem, int):
        elem = tl.constexpr(elem)
    if isinstance(elem, tl.constexpr):
        if require_i64:
            assert -2**63 <= elem.value < 2**63, f"Block pointers only support 64 bit `shape/strides`, " \
                f"got a value {elem.value} which is out of the range"
            return builder.get_int64(elem.value)
        else:
            assert -2**31 <= elem.value < 2**31, f"Block pointers only support 32 bit `offsets/block_shape`, " \
                f"got a value {elem.value} which is out of the range"
            return builder.get_int32(elem.value)
    elif isinstance(elem, tl.tensor):
        assert elem.numel.value == 1, "Expected a scalar in shape/strides/offsets"
        assert elem.dtype.is_int(), "Expected an integer scalar type in shape/strides/offsets"
        if elem.dtype != tl.int64 and require_i64:
            return builder.create_int_cast(elem.handle, builder.get_int64_ty(), elem.dtype.is_int_signed())
        elif elem.dtype != tl.int32 and not require_i64:
            assert False, "Block pointers only support 32 bit `offsets/block_shape`, " \
                "add a `.to(tl.int32)` or use regular indexing for 64 bit support"
        return elem.handle
    assert False, f"Unsupported element type in shape/strides/offsets: {type(elem)}"


def _convert_to_ir_values(builder, list_like, require_i64=True):
    if hasattr(list_like, "__iter__"):
        return [_convert_elem_to_ir_value(builder, elem, require_i64) for elem in list_like]
    return [_convert_elem_to_ir_value(builder, list_like, require_i64)]


def make_block_ptr(base: tl.tensor, shape, strides, offsets, block_shape, order, builder: ir.builder) -> tl.tensor:
    # Convert dynamic arguments to IR values
    # NOTES(Chenggang): current `shape/strides` are `int64_t`, while `offsets/block_shape` are `int32_t`
    shape = _convert_to_ir_values(builder, shape)
    strides = _convert_to_ir_values(builder, strides)
    offsets = _convert_to_ir_values(builder, offsets, require_i64=False)

    # Check `base` type
    if not base.type.is_ptr() or base.type.element_ty.is_block():
        raise ValueError("Expected `base` to be a pointer type (but not a block pointer type or others)")

    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    if base.type.element_ty == tl.int1:
        base = cast(base, tl.pointer_type(tl.int8, base.type.address_space), builder)

    # Check whether `block_shape` is static
    if not hasattr(block_shape, "__iter__"):
        block_shape = [block_shape]
    block_shape = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in block_shape]
    assert all(isinstance(elem, int) and -2**31 <= elem < 2**31 for elem in block_shape), \
        "Expected a list of constant integers (`int32_t` range) in `block_shape`"

    # Check `order`
    if not hasattr(order, "__iter__"):
        order = [order]
    order = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in order]
    assert sorted(order) == list(range(len(order))), "Expected a permutation of (0, 1, ..., len(order)-1) in order"

    # Must have same length
    assert all(len(block_shape) == len(list_like) for list_like in [shape, strides, offsets, order]), \
        "Expected shape/strides/offsets/block_shape to have the same length"

    # Build value, the type is:
    #   `pointer_type<blocked<shape, element_type>>` in Python
    #   `tt.ptr<tensor<shape, element_type>>` in MLIR
    handle = builder.create_make_block_ptr(base.handle, shape, strides, offsets, block_shape, order)
    return tl.tensor(handle, tl.pointer_type(tl.block_type(base.type.element_ty, block_shape)))


def advance(base: tl.tensor, offsets, builder: ir.builder) -> tl.tensor:
    # Convert dynamic offsets to IR values
    offsets = _convert_to_ir_values(builder, offsets, require_i64=False)

    # Advanced block pointer type is the same as before
    return tl.tensor(builder.create_advance(base.handle, offsets), base.type)


def make_tensor_descriptor(
    base: tl.tensor,
    shape: List[tl.tensor],
    strides: List[tl.tensor],
    block_shape: List[tl.constexpr],
    builder: ir.builder,
) -> tl._experimental_tensor_descriptor:
    ndim = len(shape)
    if not (2 <= ndim <= 5):
        raise ValueError(f"Expected 2 <= ndim <= 5 but got {ndim} dimensions")
    if len(strides) != ndim:
        raise ValueError(f"Expected {ndim} strides but got {len(strides)}")
    if len(block_shape) != ndim:
        raise ValueError(f"Expected block_shape to have {ndim} dimensions but got {len(strides)}")

    strides[-1] = tl._constexpr_to_value(strides[-1])
    if strides[-1] != 1:
        raise ValueError(f"Tensor descriptor last dim must be 1 but got {strides[-1]}")

    shape = [to_tensor(x, builder) for x in shape]
    strides = [to_tensor(x, builder).to(tl.int64, _builder=builder) for x in strides]

    # Check whether `block_shape` is static
    block_shape = tl._unwrap_shape(block_shape)

    assert isinstance(base.type, tl.pointer_type)
    type = tl.block_type(base.type.element_ty, block_shape)
    handle = builder.create_make_tensor_descriptor(base.handle, [s.handle for s in shape], [s.handle for s in strides],
                                                   block_shape)
    return tl._experimental_tensor_descriptor(handle, shape, strides, type)
