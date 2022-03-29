from __future__ import annotations  # remove after python 3.11

from typing import List, Optional, Tuple

from . import core as tl
from triton._C.libtriton.triton import ir


# Create custom exception that prints message "hello"
class IncompatibleTypeErrorimpl(Exception):
    def __init__(self, type_a, type_b):
        self.type_a = type_a
        self.type_b = type_b
        self.message = "invalid operands of type " + self.type_a.__repr__() + " and " + self.type_b.__repr__()
        super(IncompatibleTypeErrorimpl, self).__init__(self.message)


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
    if a_ty.is_fp16() or b_ty.is_fp16():
        if div_or_mod:
            return tl.float32
        else:
            return tl.float16
    if not a_ty.is_int() or not b_ty.is_int():
        assert False
    # 4 ) both operands are integer and undergo
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
            raise IncompatibleTypeErrorimpl(type_a, type_b)
        # T* + U* with T != U
        if type_b.is_ptr() and (type_a != type_b):
            raise IncompatibleTypeErrorimpl(type_a, type_b)
        # T* + float
        if type_b.is_floating():
            raise IncompatibleTypeErrorimpl(type_a, type_b)


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
        return tl.tensor(builder.create_gep(input.handle, [other.handle]), input.type)
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
        return tl.tensor(builder.create_gep(input.handle, [minus(other, builder).handle]),
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
        raise ValueError("both operands of fdiv must have floating poscalar type")
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, False, True)
    ret = builder.create_fdiv(input.handle, other.handle)
    ret.set_fdiv_ieee_rounding(ieee_rounding)
    return tl.tensor(ret, input.type)


def mod(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float % float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_frem(input.handle, other.handle), input.type)
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
        raise IncompatibleTypeErrorimpl(input_sca_ty, other_sca_ty)
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


def lshr(input: tl.tensor,
         other: tl.tensor,
         builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_lshr(input.handle, other.handle), input.type)


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
    _0 = tl.tensor(ir.constant.get_null_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return sub(_0, input, builder)


def invert(input: tl.tensor,
           builder: tl.tensor) -> tl.tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr() or input_sca_ty.is_floating():
        raise ValueError("wrong type argument to unary invert (" + input_sca_ty.__repr__() + ")")
    _1 = tl.tensor(ir.constant.get_all_ones_value(input_sca_ty.to_ir(builder)), input_sca_ty)
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
    shape = [end - start]
    ret_ty = tl.block_type(tl.int32, shape)
    return tl.tensor(builder.get_range(start, end), ret_ty)


def zeros(shape: List[int], dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    _0 = ir.constant.get_null_value(dtype.to_ir(builder))
    ret_ty = tl.block_type(dtype, shape)
    return tl.tensor(builder.create_splat(_0, shape), ret_ty)

# ===----------------------------------------------------------------------===//
#                               Shape Manipulation
# ===----------------------------------------------------------------------===//


def reshape(input: tl.tensor,
            dst_shape: List[int],
            builder: ir.builder) -> tl.tensor:
    numel = 1
    for s in dst_shape:
        numel *= s
    if input.type.numel != numel:
        raise ValueError("cannot reshape block of different shape")
    ret_ty = tl.block_type(input.type.scalar, dst_shape)
    return tl.tensor(builder.create_reshape(input.handle, dst_shape), ret_ty)


def cat(lhs: tl.tensor, rhs: tl.tensor, builder: ir.builder) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()
    assert lhs.type.shape[1:] == rhs.type.shape[1:]
    ret_shape = [lhs.type.shape[0] + rhs.type.shape[0]]
    ret_ty = tl.block_type(lhs.type.scalar, ret_shape)
    return tl.tensor(builder.create_cat(lhs.handle, rhs.handle), ret_ty)


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
        if len(lhs_shape) != len(rhs_shape):
            raise ValueError("Cannot make_shape_compatible: blocks must have the same rank")
        ret_shape = []
        for i in range(len(lhs_shape)):
            left = lhs_shape[i]
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

    # bf16 <=> (not fp32)
    if (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()) or \
       (dst_sca_ty.is_bf16() and not src_sca_ty.is_fp32()):
        return cast(cast(input, tl.float32, builder), dst_sca_ty, builder)

    # FP Truncation
    truncate_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.fp_mantissa_width > dst_sca_ty.fp_mantissa_width
    if truncate_fp:
        return tl.tensor(builder.create_fp_trunc(input.handle,
                                                 dst_ty.to_ir(builder)),
                         dst_ty)

    # FP Extension
    ext_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.fp_mantissa_width < dst_sca_ty.fp_mantissa_width
    if ext_fp:
        return tl.tensor(builder.create_fp_ext(input.handle,
                                               dst_ty.to_ir(builder)),
                         dst_ty)

    # Int cast
    if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
       (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        return tl.tensor(builder.create_int_cast(input.handle,
                                                 dst_ty.to_ir(builder), sign_extend),
                         dst_ty)

    # Float to Int
    if src_sca_ty.is_floating() and dst_sca_ty.is_int():
        # TODO: is this correct?
        if dst_sca_ty.is_bool():
            return tl.tensor(builder.create_fp_to_ui(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)
        else:
            return tl.tensor(builder.create_fp_to_si(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)

    # int => float
    if src_sca_ty.is_int() and dst_sca_ty.is_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tl.tensor(builder.create_ui_to_fp(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)
        else:
            return tl.tensor(builder.create_si_to_fp(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)

    # ptr => int
    if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
        bitwidth = dst_sca_ty.int_bitwidth
        if bitwidth == 64:
            return tl.tensor(builder.create_cast(ir.PtrToInt, input.handle, dst_ty.to_ir(builder)),
                             dst_ty)
        if bitwidth == 1:
            return not_equal(cast(input, tl.int64, builder),
                             tl.tensor(builder.get_int64(0), tl.int64),
                             builder)

    if not src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tl.tensor(builder.create_int_to_ptr(input.handle, dst_ty.to_ir(builder)), dst_ty)
    # Ptr . Ptr
    if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)
    # * . Bool
    if dst_sca_ty.is_bool():
        if src_sca_ty.is_ptr():
            input = cast(input, tl.int64, builder)
        other = builder.get_int64(0)
        if src_ty.is_bool():
            other = builder.create_splat(other, src_ty.get_block_shapes())
        return tl.tensor(builder.create_icmpNE(input.handle, other), dst_ty)
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

    if other:
        other = cast(other, ptr.type.scalar.element_ty, builder)
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty
    # treat bool* as tl.int8*
    if elt_ty == tl.int1:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)

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

    if not mask and not other:
        return tl.tensor(builder.create_load(ptr.handle, cache, eviction, is_volatile),
                         dst_ty)
    if not mask:
        raise ValueError("`other` cannot be provided without `mask`")

    if not other:
        other_ir = ir.undef.get(elt_ty.to_ir(builder))
        if ptr.type.is_block():
            other_ir = builder.create_splat(other_ir, ptr.type.get_block_shapes())
        other = tl.tensor(other_ir, dst_ty)

    return tl.tensor(builder.create_masked_load(ptr.handle,
                                                mask.handle,
                                                other.handle,
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
    if mask:
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
    # TODO: type checking
    return tl.tensor(builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle), val.type)


def atom_red_typechecking_impl(ptr: tl.tensor,
                               val: tl.tensor,
                               mask: tl.tensor,
                               builder: ir.builder) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    if not ptr.type.scalar.is_ptr():
        raise ValueError("Pointer argument of store instruction is " + ptr.type.__repr__())
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
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
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
    pos = greater_equal(val, tl.tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder)
    neg = less_than(val, tl.tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder)
    pos_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, i_ptr.handle, i_val.handle, and_(mask, pos, builder).handle), i_val.type)
    neg_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, i_ptr.handle, i_val.handle, and_(mask, neg, builder).handle), i_val.type)
    return where(pos, pos_ret, neg_ret, builder)


def atomic_min(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
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
    pos = greater_equal(val, tl.tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder)
    neg = less_than(val, tl.tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder)
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
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
    sca_ty = val.type.scalar
    op = ir.ATOMIC_OP.FADD if sca_ty.is_floating() else ir.ATOMIC_OP.ADD
    return tl.tensor(builder.create_atomic_rmw(op, ptr.handle, val.handle, mask.handle), val.type)


def atomic_and(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.AND, ptr.handle, val.handle, mask.handle), val.type)


def atomic_or(ptr: tl.tensor,
              val: tl.tensor,
              mask: tl.tensor,
              builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.OR, ptr.handle, val.handle, mask.handle), val.type)


def atomic_xor(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XOR, ptr.handle, val.handle, mask.handle), val.type)


def atomic_xchg(ptr: tl.tensor,
                val: tl.tensor,
                mask: tl.tensor,
                builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XCHG, ptr.handle, val.handle, mask.handle), val.type)

# ===----------------------------------------------------------------------===//
#                               Linear Algebra
# ===----------------------------------------------------------------------===//


def dot(lhs: tl.tensor,
        rhs: tl.tensor,
        allow_tf32: bool,
        builder: ir.builder) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()
    if lhs.type.scalar.is_int():
        _0 = builder.get_int32(0)
        ret_scalar_ty = tl.int32
    else:
        _0 = builder.get_float32(0)
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
        x = broadcast_impl_shape(x, condition.type.get_block_shapes(), builder)
        y = broadcast_impl_shape(y, condition.type.get_block_shapes(), builder)

    # TODO: we need to check x's and y's shape?
    x_ty = x.type.scalar
    y_ty = y.type.scalar
    ty = computation_type_impl(x_ty, y_ty, div_or_mod=False)
    x = cast(x, ty, builder)
    y = cast(y, ty, builder)
    if x.type.is_block():
        ret_ty = tl.block_type(ty, x.type.shape)
    else:
        ret_ty = ty
    return tl.tensor(builder.create_select(condition.handle, x.handle, y.handle), ret_ty)


# ===----------------------------------------------------------------------===//
#                               Reductions
# ===----------------------------------------------------------------------===

def reduce_impl(input: tl.tensor, axis: int, builder: ir.builder, name: str,
                FLOAT_OP: ir.REDUCE_OP, INT_OP: ir.REDUCE_OP) -> tl.tensor:
    scalar_ty = input.type.scalar
    # input is extended to 32-bits if necessary
    # this increases numerical accuracy and can be done pretty much for free
    # on GPUs
    if scalar_ty.is_int() and scalar_ty.int_bitwidth <= 32:
        input = cast(input, tl.int32, builder)

    # get result type
    shape = input.type.shape
    ret_shape = []
    for i, s in enumerate(shape):
        if i != axis:
            ret_shape.append(s)
    if len(ret_shape) == 0:
        res_ty = scalar_ty
    else:
        res_ty = tl.block_type(scalar_ty, ret_shape)

    if scalar_ty.is_floating():
        return tl.tensor(builder.create_reduce(input.handle, FLOAT_OP, axis), res_ty)
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_reduce(input.handle, INT_OP, axis), res_ty)
    assert False


def min(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    return reduce_impl(input, axis, builder, "min", ir.REDUCE_OP.FMIN, ir.REDUCE_OP.MIN)


def max(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    return reduce_impl(input, axis, builder, "max", ir.REDUCE_OP.FMAX, ir.REDUCE_OP.MAX)


def sum(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    return reduce_impl(input, axis, builder, "sum", ir.REDUCE_OP.FADD, ir.REDUCE_OP.ADD)


def xor_sum(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    scalar_ty = input.type.scalar
    if not scalar_ty.is_int():
        raise ValueError("xor_sum only supported for integers")
    return reduce_impl(input, axis, builder, "sum", ir.REDUCE_OP.XOR, ir.REDUCE_OP.XOR)


# -----------------------
# Utilities
# -----------------------

def clock(builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_clock(), tl.int64)


def globaltimer(builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_globaltimer, tl.int64)


# ===----------------------------------------------------------------------===
#                               Math
# ===----------------------------------------------------------------------===

def umulhi(x: tl.tensor, y: tl.tensor, builder: ir.builder) -> tl.tensor:
    x, y = binary_op_type_checking_impl(x, y, builder)
    return tl.tensor(builder.create_umulhi(x.handle, y.handle), x.type)


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

def multiple_of(x: tl.tensor, value: int) -> tl.tensor:
    x.handle.multiple_of(value)
    return x


def max_contiguous(x: tl.tensor, value: int) -> tl.tensor:
    x.handle.max_contiguous(value)
    return x


def debug_barrier(builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_barrier(''), tl.void)
