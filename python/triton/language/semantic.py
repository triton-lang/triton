from __future__ import annotations  # remove after python 3.11

from typing import List, Optional, Tuple

from . import core as tl
from triton._C.libtriton.triton import ir


# Create custom exception that prints message "hello"
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
    return tl.tensor(builder.create_get_program_id(axis), tl.int32)


def num_programs(axis: int, builder: ir.builder) -> tl.tensor:
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
    assert False


def computation_type_impl(a_ty: tl.dtype, b_ty: tl.dtype, div_or_mod: bool) -> tl.dtype:
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
    if a_ty.is_bf16() or b_ty.is_bf16():
        if div_or_mod:
            return tl.float32
        if a_ty.is_bf16() and b_ty.is_bf16():
            return tl.bfloat16
        return tl.float32
    if not a_ty.is_int() or not b_ty.is_int():
        assert False
    # 5 ) both operands are integer and undergo
    #    integer promotion
    if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
        raise ValueError("Cannot use /, #, or % with " + a_ty.__repr__() + " and " + b_ty.__repr__() + " because they have different signedness;"
                         "this is unlikely to result in a useful answer. Cast them to the same signedness.")
    return integer_promote_impl(a_ty, b_ty)

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


def binary_op_type_checking_impl(lhs: tl.tensor,
                                 rhs: tl.tensor,
                                 builder: ir.builder,
                                 allow_lhs_ptr=False, allow_rhs_ptr=False,
                                 arithmetic_check=True, div_or_mod=False
                                 ) -> Tuple[tl.tensor, tl.tensor]:
    # implicit broadcasting
    lhs, rhs = broadcast_impl_value(lhs, rhs, builder)
    # implicit typecasting
    lhs_sca_ty = lhs.type.scalar
    rhs_sca_ty = rhs.type.scalar
    check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr)
    check_ptr_type_impl(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr)
    if arithmetic_check and not lhs_sca_ty.is_ptr() and not rhs_sca_ty.is_ptr():
        ret_sca_ty = computation_type_impl(lhs_sca_ty, rhs_sca_ty, div_or_mod)
        lhs = cast(lhs, ret_sca_ty, builder)
        rhs = cast(rhs, ret_sca_ty, builder)
    return lhs, rhs


def add(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar

    # offset + ptr
    # ptr + offset
    if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
        input, other = other, input
    if input_scalar_ty.is_ptr():
        return tl.tensor(builder.create_addptr(input.handle, other.handle), input.type)
    # float + float
    elif input_scalar_ty.is_floating():
        return tl.tensor(builder.create_fadd(input.handle, other.handle), input.type)
    # int + int
    elif input_scalar_ty.is_int():
        return tl.tensor(builder.create_add(input.handle, other.handle), input.type)
    assert False


def sub(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, True, False)
    scalar_ty = input.type.scalar
    # ptr - offset
    if scalar_ty.is_ptr():
        return tl.tensor(builder.create_addptr(input.handle, minus(other, builder).handle),
                         input.type)
    # float - float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fsub(input.handle, other.handle), input.type)
    # int - int
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_sub(input.handle, other.handle), input.type)
    assert False


def mul(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float * float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fmul(input.handle, other.handle), input.type)
    # * int
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_mul(input.handle, other.handle), input.type)
    assert False


def truediv(input: tl.tensor,
            other: tl.tensor,
            builder: ir.builder) -> tl.tensor:
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
    # float / float (cast to highest exponent type)
    elif input_scalar_ty.is_floating() and other_scalar_ty.is_floating():
        if input_scalar_ty.fp_mantissa_width > other_scalar_ty.fp_mantissa_width:
            other = cast(other, input_scalar_ty, builder)
        else:
            input = cast(input, other_scalar_ty, builder)
    # unreachable
    else:
        assert False
    return tl.tensor(builder.create_fdiv(input.handle, other.handle), input.type)


def floordiv(input: tl.tensor,
             other: tl.tensor,
             builder: ir.builder) -> tl.tensor:
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
    assert False


def fdiv(input: tl.tensor,
         other: tl.tensor,
         ieee_rounding: bool,
         builder: ir.builder) -> tl.tensor:
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if not input_scalar_ty.is_floating() or not other_scalar_ty.is_floating():
        raise ValueError("both operands of fdiv must have floating scalar type")
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, False, True)
    ret = builder.create_fdiv(input.handle, other.handle)
    return tl.tensor(ret, input.type)


def mod(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float % float
    if scalar_ty.is_floating():
        # input - input.div(other, rounding_mode="floor") * other
        ret = sub(input, mul(floor(fdiv(input, other, False, builder), builder),
                             other, builder),
                  builder)
        return ret
    # % int
    elif scalar_ty.is_int():
        if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
            raise ValueError("Cannot mod " + scalar_ty.__repr__() + " by " + other_scalar_ty.__repr__() + " "
                             "because they have different signedness;"
                             "this is unlikely to result in a useful answer. Cast them to the same signedness.")
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_srem(input.handle, other.handle), input.type)
        else:
            return tl.tensor(builder.create_urem(input.handle, other.handle), input.type)
    assert False

##############
# bitwise ops
##############


def bitwise_op_type_checking_impl(input: tl.tensor,
                                  other: tl.tensor,
                                  builder: ir.builder) -> Tuple[tl.tensor, tl.tensor]:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, False)
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


def and_(input: tl.tensor,
         other: tl.tensor,
         builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_and(input.handle, other.handle), input.type)


def or_(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_or(input.handle, other.handle), input.type)


def xor_(input: tl.tensor,
         other: tl.tensor,
         builder: ir.builder) -> tl.tensor:
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


def lshr(input: tl.tensor,
         other: tl.tensor,
         builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_lshr(input.handle, other.handle), input.type)


def ashr(input: tl.tensor,
         other: tl.tensor,
         builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_ashr(input.handle, other.handle), input.type)


def shl(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_shl(input.handle, other.handle), input.type)

# ===----------------------------------------------------------------------===//
#                               Unary Operators
# ===----------------------------------------------------------------------===//


def plus(input: tl.tensor) -> tl.tensor:
    return input


def minus(input: tl.tensor,
          builder: ir.builder) -> tl.tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr():
        raise ValueError("wrong type argument to unary minus (" + input_sca_ty.__repr__() + ")")
    _0 = tl.tensor(builder.get_null_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return sub(_0, input, builder)


def invert(input: tl.tensor,
           builder: tl.tensor) -> tl.tensor:
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


def greater_than(input: tl.tensor,
                 other: tl.tensor,
                 builder: ir.builder) -> tl.tensor:
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
    assert False


def greater_equal(input: tl.tensor,
                  other: tl.tensor,
                  builder: ir.builder) -> tl.tensor:
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
    assert False


def less_than(input: tl.tensor,
              other: tl.tensor,
              builder: ir.builder) -> tl.tensor:
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
    assert False


def less_equal(input: tl.tensor,
               other: tl.tensor,
               builder: ir.builder) -> tl.tensor:
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
    assert False


def equal(input: tl.tensor,
          other: tl.tensor,
          builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float == float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOEQ(input.handle, other.handle), _bool_like(input))
    # == int
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_icmpEQ(input.handle, other.handle), _bool_like(input))
    assert False


def not_equal(input: tl.tensor,
              other: tl.tensor,
              builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float == float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpUNE(input.handle, other.handle), _bool_like(input))
    # == int
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_icmpNE(input.handle, other.handle), _bool_like(input))
    assert False

# ===----------------------------------------------------------------------===//
#                               Block Creation
# ===----------------------------------------------------------------------===//


def arange(start: int, end: int, builder: ir.builder) -> tl.tensor:
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("arange's arguments must be of type tl.constexpr")

    shape = [end - start]
    ret_ty = tl.block_type(tl.int32, shape)
    return tl.tensor(builder.create_make_range(start, end), ret_ty)


def full(shape: List[int], value, dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    if value == 0:
        _value = builder.get_null_value(dtype.to_ir(builder))
    else:
        get_value_fn = getattr(builder, f"get_{dtype.name}")
        _value = get_value_fn(value)
    ret_ty = tl.block_type(dtype, shape)
    return tl.tensor(builder.create_splat(_value, shape), ret_ty)


# ===----------------------------------------------------------------------===//
#                               Shape Manipulation
# ===----------------------------------------------------------------------===//


def view(input: tl.tensor,
         dst_shape: List[int],
         builder: ir.builder) -> tl.tensor:
    # TODO: disable when TritonToTritonGPU handles views properly

    # assert len(input.shape) == len(dst_shape)
    numel = 1
    for s in dst_shape:
        numel *= s
    if input.type.numel != numel:
        raise ValueError("cannot view block of different shape")
    ret_ty = tl.block_type(input.type.scalar, dst_shape)
    return tl.tensor(builder.create_view(input.handle, dst_shape), ret_ty)


def expand_dims(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    dst_shape = list(input.type.shape)
    dst_shape.insert(axis, 1)
    ret_ty = tl.block_type(input.type.scalar, dst_shape)
    return tl.tensor(builder.create_expand_dims(input.handle, axis), ret_ty)


def cat(lhs: tl.tensor, rhs: tl.tensor, can_reorder: bool, builder: ir.builder) -> tl.tensor:
    assert can_reorder, "current implementation of `cat` always may reorder elements"
    assert len(lhs.shape) == 1
    ret_type = tl.block_type(lhs.type.scalar, [lhs.shape[0] + rhs.shape[0]])
    return tl.tensor(builder.create_cat(lhs.handle, rhs.handle), ret_type)


def trans(input: tl.tensor, builder: ir.builder) -> tl.tensor:
    if len(input.shape) != 2:
        raise ValueError("Only 2D tensors can be transposed")
    ret_type = tl.block_type(input.type.scalar, [input.shape[1], input.shape[0]])
    return tl.tensor(builder.create_trans(input.handle), ret_type)


def broadcast_impl_shape(input: tl.tensor,
                         shape: List[int],
                         builder: ir.builder) -> tl.tensor:
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


def broadcast_impl_value(lhs: tl.tensor,
                         rhs: tl.tensor,
                         builder: ir.builder) -> tl.tensor:
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
            for dim in range(len(lhs_shape), len(rhs_shape)):
                lhs = tl.tensor(builder.create_expand_dims(lhs.handle, dim), tl.block_type(lhs_ty.scalar, lhs_shape + [1]))
                lhs_ty = lhs.type
                lhs_shape = lhs_ty.get_block_shapes()
        elif len(rhs_shape) < len(lhs_shape):
            # Add new axes to rhs
            for dim in range(len(rhs_shape), len(lhs_shape)):
                rhs = tl.tensor(builder.create_expand_dims(rhs.handle, dim), tl.block_type(rhs_ty.scalar, rhs_shape + [1]))
                rhs_ty = rhs.type
                rhs_shape = rhs_ty.get_block_shapes()
        assert len(rhs_shape) == len(lhs_shape)

        ret_shape = []
        for i, left in enumerate(lhs_shape):
            right = rhs_shape[i]
            if left == 1:
                ret_shape.append(right)
            elif right == 1:
                ret_shape.append(left)
            elif left == right:
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


def bitcast(input: tl.tensor,
            dst_ty: tl.dtype,
            builder: ir.builder) -> tl.tensor:
    src_ty = input.type
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty, input.type.get_block_shapes())
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
        raise ValueError("Cannot bitcast data-type of size " + str(src_bits) + "to "
                         "data-type of size " + str(dst_bits))
    return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)),
                     dst_ty)


def cast(input: tl.tensor,
         dst_ty: tl.dtype,
         builder: ir.builder) -> tl.tensor:
    src_ty = input.type
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input

    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar

    # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
    if (src_sca_ty.is_customized_floating() and dst_sca_ty.is_floating()) or \
       (src_sca_ty.is_floating() and dst_sca_ty.is_customized_floating()):
        return tl.tensor(builder.create_fp_to_fp(input.handle, dst_ty.to_ir(builder)),
                         dst_ty)

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
        return tl.tensor(builder.create_fp_trunc(input.handle,
                                                 dst_ty.to_ir(builder)),
                         dst_ty)

    # Standard floating types' casting: extension
    #   fp32 => fp64
    #   fp16 => fp32, fp64
    #   bf16 => fp32, fp64
    ext_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
    if ext_fp:
        return tl.tensor(builder.create_fp_ext(input.handle,
                                               dst_ty.to_ir(builder)),
                         dst_ty)

    # Casting between integer types
    if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
       (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        else:
            return tl.tensor(builder.create_int_cast(input.handle,
                                                     dst_ty.to_ir(builder), sign_extend),
                             dst_ty)

    # Casting standard floating types to integer types
    if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        elif dst_sca_ty.is_int_signed():
            return tl.tensor(builder.create_fp_to_si(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)
        else:
            return tl.tensor(builder.create_fp_to_ui(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)

    # Casting integer types to standard floating types
    if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tl.tensor(builder.create_ui_to_fp(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)
        else:
            return tl.tensor(builder.create_si_to_fp(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)

    # Casting pointer types to integer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
        bitwidth = dst_sca_ty.int_bitwidth
        if bitwidth == 64:
            return tl.tensor(builder.create_ptr_to_int(input.handle, dst_ty.to_ir(builder)),
                             dst_ty)
        if bitwidth == 1:
            return not_equal(cast(input, tl.int64, builder),
                             tl.tensor(builder.get_int64(0), tl.int64),
                             builder)

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


def load(ptr: tl.tensor,
         mask: Optional[tl.tensor],
         other: Optional[tl.tensor],
         cache_modifier: str,
         eviction_policy: str,
         is_volatile: bool,
         builder: ir.builder) -> tl.tensor:
    if not ptr.type.scalar.is_ptr():
        raise ValueError("Pointer argument of load instruction is " + ptr.type.__repr__())
    if ptr.type.is_block():
        if mask:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if other:
            other = broadcast_impl_shape(other, ptr.type.get_block_shapes(), builder)

    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty

    # treat bool* as tl.int8*
    if elt_ty == tl.int1:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)

    if other:
        other = cast(other, elt_ty, builder)

    # cache modifier
    cache = ir.CACHE_MODIFIER.NONE  # default
    if cache_modifier:
        if cache_modifier == ".ca":
            cache = ir.CACHE_MODIFIER.CA
        elif cache_modifier == ".cg":
            cache = ir.CACHE_MODIFIER.CG
        else:
            raise ValueError(f"Cache modifier {cache_modifier} not supported")

    # eviction policy
    eviction = ir.EVICTION_POLICY.NORMAL  # default
    if eviction_policy:
        if eviction_policy == "evict_last":
            eviction = ir.EVICTION_POLICY.EVICT_LAST
        elif eviction_policy == "evict_first":
            eviction = ir.EVICTION_POLICY.EVICT_FIRST
        else:
            raise ValueError(f"Eviction policy {eviction_policy} not supported")

    if ptr.type.is_block():
        shape = ptr.type.get_block_shapes()
        dst_ty = tl.block_type(elt_ty, shape)
    else:
        dst_ty = elt_ty

    if not mask:
        if other:
            raise ValueError("`other` cannot be provided without `mask`")
        return tl.tensor(builder.create_load(ptr.handle, cache, eviction, is_volatile),
                         dst_ty)
    else:
        return tl.tensor(builder.create_masked_load(ptr.handle,
                                                    mask.handle,
                                                    other.handle if other else None,
                                                    cache, eviction, is_volatile),
                         dst_ty)


def store(ptr: tl.tensor,
          val: tl.tensor,
          mask: Optional[tl.tensor],
          builder: ir.builder) -> tl.tensor:
    if not ptr.type.scalar.is_ptr():
        raise ValueError("Pointer argument of store instruction is " + ptr.type.__repr__())
    if ptr.type.is_block():
        val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
    if mask and ptr.type.is_block():
        mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty
    # treat bool* as tl.int8*
    if elt_ty == tl.int1:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)

    # cast to target data-type
    val = cast(val, elt_ty, builder)
    if not mask:
        return tl.tensor(builder.create_store(ptr.handle, val.handle), tl.void)
    if not mask.type.scalar.is_bool():
        raise ValueError("Mask must have boolean scalar type")
    return tl.tensor(builder.create_masked_store(ptr.handle, val.handle, mask.handle), tl.void)

#########
# atomic
#########


def atomic_cas(ptr: tl.tensor,
               cmp: tl.tensor,
               val: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    element_ty = ptr.type.scalar.element_ty
    if element_ty.primitive_bitwidth not in [16, 32, 64]:
        raise ValueError("atomic_cas only supports elements with width {16, 32, 64}")
    return tl.tensor(builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle), val.type)


def atom_red_typechecking_impl(ptr: tl.tensor,
                               val: tl.tensor,
                               mask: tl.tensor,
                               op: str,
                               builder: ir.builder) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    if not ptr.type.scalar.is_ptr():
        raise ValueError("Pointer argument of store instruction is " + ptr.type.__repr__())

    element_ty = ptr.type.scalar.element_ty
    if element_ty is tl.float16 and op != 'add':
        raise ValueError("atomic_" + op + " does not support fp16")
    if element_ty in [tl.int1, tl.int8, tl.int16, tl.bfloat16]:
        raise ValueError("atomic_" + op + " does not support " + str(element_ty))
    if ptr.type.is_block():
        if mask:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if val:
            val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
    val = cast(val, ptr.type.scalar.element_ty, builder)
    if not mask:
        mask_ir = builder.get_int1(True)
        mask_ty = tl.int1
        if ptr.type.is_block():
            mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
            mask_ty = tl.block_type(tl.int1, ptr.type.get_block_shapes())
        mask = tl.tensor(mask_ir, mask_ty)
    return ptr, val, mask


def atomic_max(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'max', builder)
    sca_ty = val.type.scalar
    # direct call to atomic_max for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MAX,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle),
                             val.type)
        else:
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle),
                             val.type)
    # for float
    # return atomic_smax(i_ptr, i_val) if val >= 0
    # return atomic_umin(i_ptr, i_val) if val < 0
    i_val = bitcast(val, tl.int32, builder)
    i_ptr = bitcast(ptr, tl.pointer_type(tl.int32, 1), builder)
    pos = greater_equal(val, tl.tensor(builder.get_fp32(0), sca_ty), builder)
    neg = less_than(val, tl.tensor(builder.get_fp32(0), sca_ty), builder)
    pos_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, i_ptr.handle, i_val.handle, and_(mask, pos, builder).handle), i_val.type)
    neg_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, i_ptr.handle, i_val.handle, and_(mask, neg, builder).handle), i_val.type)
    return where(pos, pos_ret, neg_ret, builder)


def atomic_min(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'min', builder)
    sca_ty = val.type.scalar
    # direct call to atomic_min for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MIN,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle),
                             val.type)
        else:
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle),
                             val.type)
    # for float
    # return atomic_smin(i_ptr, i_val) if val >= 0
    # return atomic_umax(i_ptr, i_val) if val < 0
    i_val = bitcast(val, tl.int32, builder)
    i_ptr = bitcast(ptr, tl.pointer_type(tl.int32, 1), builder)
    pos = greater_equal(val, tl.tensor(builder.get_fp32(0), sca_ty), builder)
    neg = less_than(val, tl.tensor(builder.get_fp32(0), sca_ty), builder)
    pos_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MIN,
                                                  i_ptr.handle,
                                                  i_val.handle,
                                                  and_(mask, pos, builder).handle),
                        i_val.type)
    neg_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX,
                                                  i_ptr.handle,
                                                  i_val.handle,
                                                  and_(mask, neg, builder).handle),
                        i_val.type)
    return where(pos, pos_ret, neg_ret, builder)


def atomic_add(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'add', builder)
    sca_ty = val.type.scalar
    op = ir.ATOMIC_OP.FADD if sca_ty.is_floating() else ir.ATOMIC_OP.ADD
    return tl.tensor(builder.create_atomic_rmw(op, ptr.handle, val.handle, mask.handle), val.type)


def atomic_and(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'and', builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.AND, ptr.handle, val.handle, mask.handle), val.type)


def atomic_or(ptr: tl.tensor,
              val: tl.tensor,
              mask: tl.tensor,
              builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'or', builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.OR, ptr.handle, val.handle, mask.handle), val.type)


def atomic_xor(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'xor', builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XOR, ptr.handle, val.handle, mask.handle), val.type)


def atomic_xchg(ptr: tl.tensor,
                val: tl.tensor,
                mask: tl.tensor,
                builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'xchg', builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XCHG, ptr.handle, val.handle, mask.handle), val.type)

# ===----------------------------------------------------------------------===//
#                               Linear Algebra
# ===----------------------------------------------------------------------===//


def dot(lhs: tl.tensor,
        rhs: tl.tensor,
        allow_tf32: bool,
        builder: ir.builder) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()
    assert lhs.dtype == rhs.dtype, "lhs and rhs must have the same dtype!"
    assert len(lhs.shape) == 2 and len(rhs.shape) == 2
    assert lhs.shape[1].value == rhs.shape[0].value
    assert lhs.shape[0].value >= 16 and lhs.shape[1].value >= 16 \
        and rhs.shape[1].value >= 16,\
        "small blocks not supported!"
    if lhs.type.scalar.is_int():
        _0 = builder.get_int32(0)
        ret_scalar_ty = tl.int32
    else:
        _0 = builder.get_fp32(0)
        ret_scalar_ty = tl.float32
    M = lhs.type.shape[0]
    N = rhs.type.shape[1]
    _0 = builder.create_splat(_0, [M, N])
    ret_ty = tl.block_type(ret_scalar_ty, [M, N])
    return tl.tensor(builder.create_dot(lhs.handle, rhs.handle, _0, allow_tf32),
                     ret_ty)


# ===----------------------------------------------------------------------===//
#                               Indexing
# ===----------------------------------------------------------------------===//

def where(condition: tl.tensor,
          x: tl.tensor,
          y: tl.tensor,
          builder: ir.builder) -> tl.tensor:
    condition = cast(condition, tl.int1, builder)
    if condition.type.is_block():
        condition, x = broadcast_impl_value(condition, x, builder)
        x, y = broadcast_impl_value(x, y, builder)
        condition, x = broadcast_impl_value(condition, x, builder)

    x, y = binary_op_type_checking_impl(x, y, builder, True, True)
    if not condition.type.is_block():
        condition, _ = broadcast_impl_value(condition, x, builder)
    ret_ty = x.type
    return tl.tensor(builder.create_select(condition.handle, x.handle, y.handle), ret_ty)

# ===----------------------------------------------------------------------===//
#                               Reductions
# ===----------------------------------------------------------------------===


def reduce_impl(input: tl.tensor, axis: int, builder: ir.builder, name: str,
                FLOAT_OP: ir.REDUCE_OP, INT_OP: ir.REDUCE_OP) -> tl.tensor:
    scalar_ty = input.type.scalar
    out_scalar_ty = scalar_ty
    # input is extended to 32-bits if necessary
    # this increases numerical accuracy and can be done pretty much for free
    # on GPUs
    if scalar_ty.is_int() and scalar_ty.int_bitwidth <= 32:
        input = cast(input, tl.int32, builder)
        out_scalar_ty = tl.int32

    # hardware doesn't support FMAX, FMIN, CMP for bfloat16
    if scalar_ty is tl.bfloat16:
        input = cast(input, tl.float32, builder)
        out_scalar_ty = tl.float32

    # choose the right unsigned operation
    if scalar_ty.is_int_unsigned():
        int_op_to_unit = {
            ir.REDUCE_OP.MIN: ir.REDUCE_OP.UMIN,
            ir.REDUCE_OP.MAX: ir.REDUCE_OP.UMAX,
            ir.REDUCE_OP.ARGMIN: ir.REDUCE_OP.ARGUMIN,
            ir.REDUCE_OP.ARGMAX: ir.REDUCE_OP.ARGUMAX,
        }
        if INT_OP in int_op_to_unit:
            INT_OP = int_op_to_unit[INT_OP]

    # If we are doing an argmin or argmax we want to use an int32 output type
    if FLOAT_OP is ir.REDUCE_OP.ARGFMAX or INT_OP is ir.REDUCE_OP.ARGMAX:
        out_scalar_ty = tl.int32
    elif FLOAT_OP is ir.REDUCE_OP.ARGFMIN or INT_OP is ir.REDUCE_OP.ARGMIN:
        out_scalar_ty = tl.int32

    # get result type
    shape = input.type.shape
    ret_shape = []
    for i, s in enumerate(shape):
        if i != axis:
            ret_shape.append(s)
    if ret_shape:
        res_ty = tl.block_type(out_scalar_ty, ret_shape)
    else:
        # 0d-tensor -> scalar
        res_ty = out_scalar_ty

    if scalar_ty.is_floating():
        return tl.tensor(builder.create_reduce(input.handle, FLOAT_OP, axis), res_ty)
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_reduce(input.handle, INT_OP, axis), res_ty)
    assert False


def min(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    return reduce_impl(input, axis, builder, "min", ir.REDUCE_OP.FMIN, ir.REDUCE_OP.MIN)


def argmin(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    return reduce_impl(input, axis, builder, "argmin", ir.REDUCE_OP.ARGFMIN, ir.REDUCE_OP.ARGMIN)


def max(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    return reduce_impl(input, axis, builder, "max", ir.REDUCE_OP.FMAX, ir.REDUCE_OP.MAX)


def argmax(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    return reduce_impl(input, axis, builder, "argmax", ir.REDUCE_OP.ARGFMAX, ir.REDUCE_OP.ARGMAX)


def sum(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    return reduce_impl(input, axis, builder, "sum", ir.REDUCE_OP.FADD, ir.REDUCE_OP.ADD)


def xor_sum(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    scalar_ty = input.type.scalar
    if not scalar_ty.is_int():
        raise ValueError("xor_sum only supported for integers")
    return reduce_impl(input, axis, builder, "sum", ir.REDUCE_OP.XOR, ir.REDUCE_OP.XOR)


# ===----------------------------------------------------------------------===
#                               Math
# ===----------------------------------------------------------------------===

def umulhi(x: tl.tensor, y: tl.tensor, builder: ir.builder) -> tl.tensor:
    x, y = binary_op_type_checking_impl(x, y, builder)
    # FIXME(Keren): not portable, should be fixed
    from . import libdevice
    return libdevice.mulhi(x, y, _builder=builder)


def floor(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    # FIXME(Keren): not portable, should be fixed
    from . import libdevice
    return libdevice.floor(x, _builder=builder)


def exp(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_exp(x.handle), x.type)


def log(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_log(x.handle), x.type)


def cos(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_cos(x.handle), x.type)


def sin(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_sin(x.handle), x.type)


def sqrt(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_sqrt(x.handle), x.type)


##

def multiple_of(x: tl.tensor, values: List[int]) -> tl.tensor:
    if len(x.shape) != len(values):
        raise ValueError("Shape of input to multiple_of does not match the length of values")
    x.handle.set_attr("tt.divisibility", ir.make_attr(values, x.handle.get_context()))
    return x


def max_contiguous(x: tl.tensor, values: List[int]) -> tl.tensor:
    if len(x.shape) != len(values):
        raise ValueError("Shape of input to max_contiguous does not match the length of values")
    x.handle.set_attr("tt.contiguity", ir.make_attr(values, x.handle.get_context()))
    return x


def debug_barrier(builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_barrier(), tl.void)


def printf(prefix: str, args: List[tl.tensor], builder: ir.builder) -> tl.tensor:
    new_args = []
    for arg in args:
        new_args.append(arg.handle)
    return tl.tensor(builder.create_printf(prefix, new_args), tl.void)
