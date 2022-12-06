from typing import Callable, TypeVar

import triton

from ..impl import base
from .math import _add_math_1arg_docstr
import triton.language as tl

T = TypeVar("T")


def _reduce_impl(
    *,
    name: str,
    FLOAT_OP: tl.ir.REDUCE_OP,
    INT_OP: tl.ir.REDUCE_OP,
    input: tl.tensor,
    axis: int,
    builder: tl.ir.builder,
) -> tl.tensor:
    scalar_ty = input.type.scalar
    # input is extended to 32-bits if necessary
    # this increases numerical accuracy and can be done pretty much for free
    # on GPUs
    if scalar_ty.is_int() and scalar_ty.int_bitwidth <= 32:
        input = base._i_cast(input, tl.int32, builder)

    # hardware doesn't support FMAX, FMIN, CMP for bfloat16
    if scalar_ty is tl.bfloat16:
        input = base._i_cast(input, tl.float32, builder)

    # choose the right unsigned operation
    if scalar_ty.is_int_unsigned():
        int_op_to_unit = {
            tl.ir.REDUCE_OP.MIN: tl.ir.REDUCE_OP.UMIN,
            tl.ir.REDUCE_OP.MAX: tl.ir.REDUCE_OP.UMAX,
            tl.ir.REDUCE_OP.ARGMIN: tl.ir.REDUCE_OP.ARGUMIN,
            tl.ir.REDUCE_OP.ARGMAX: tl.ir.REDUCE_OP.ARGUMAX,
        }
        if INT_OP in int_op_to_unit:
            INT_OP = int_op_to_unit[INT_OP]

    # get result type
    shape = input.type.shape
    ret_shape = []
    for i, s in enumerate(shape):
        if i != axis:
            ret_shape.append(s)
    if ret_shape:
        res_ty = tl.block_type(scalar_ty, ret_shape)
    else:
        # 0d-tensor -> scalar
        res_ty = scalar_ty

    if scalar_ty.is_floating():
        return tl.tensor(
            builder.create_reduce(
                input.handle,
                FLOAT_OP,
                axis,
            ),
            res_ty,
        )
    elif scalar_ty.is_int():
        return tl.tensor(
            builder.create_reduce(
                input.handle,
                INT_OP,
                axis,
            ),
            res_ty,
        )
    assert False


def _add_reduction_docstr(name: str) -> Callable[[T], T]:
    def _decorator(func: T) -> T:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :param axis: the dimension along which the reduction should be done
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _i_max(
    input: tl.tensor,
    axis: int,
    builder: tl.ir.builder,
) -> tl.tensor:
    return _reduce_impl(
        name="max",
        FLOAT_OP=tl.ir.REDUCE_OP.FMAX,
        INT_OP=tl.ir.REDUCE_OP.MAX,
        input=input,
        axis=axis,
        builder=builder,
    )


@tl.builtin
@_add_reduction_docstr("maximum")
def max(input, axis, _builder=None):
    axis = tl._constexpr_to_value(axis)
    return _i_max(input, axis, _builder)


def _i_argmax(
    input: tl.tensor,
    axis: int,
    builder: tl.ir.builder,
) -> tl.tensor:
    return _reduce_impl(
        name="argmax",
        FLOAT_OP=tl.ir.REDUCE_OP.ARGFMAX,
        INT_OP=tl.ir.REDUCE_OP.ARGMAX,
        input=input,
        axis=axis,
        builder=builder,
    )


@tl.builtin
@_add_reduction_docstr("maximum index")
def argmax(input, axis, _builder=None):
    axis = tl._constexpr_to_value(axis)
    return _i_argmax(input, axis, _builder)


def _i_min(
    input: tl.tensor,
    axis: int,
    builder: tl.ir.builder,
) -> tl.tensor:
    return _reduce_impl(
        name="min",
        FLOAT_OP=tl.ir.REDUCE_OP.FMIN,
        INT_OP=tl.ir.REDUCE_OP.MIN,
        input=input,
        axis=axis,
        builder=builder,
    )


@tl.builtin
@_add_reduction_docstr("minimum")
def min(input, axis, _builder=None):
    axis = tl._constexpr_to_value(axis)
    return _i_min(input, axis, _builder)


def _i_argmin(
    input: tl.tensor,
    axis: int,
    builder: tl.ir.builder,
) -> tl.tensor:
    return _reduce_impl(
        name="argmin",
        FLOAT_OP=tl.ir.REDUCE_OP.ARGFMIN,
        INT_OP=tl.ir.REDUCE_OP.ARGMIN,
        input=input,
        axis=axis,
        builder=builder,
    )


@tl.builtin
@_add_reduction_docstr("minimum index")
def argmin(input, axis, _builder=None):
    axis = tl._constexpr_to_value(axis)
    return _i_argmin(input, axis, _builder)


def _i_sum(
    input: tl.tensor,
    axis: int,
    builder: tl.ir.builder,
) -> tl.tensor:
    return _reduce_impl(
        name="sum",
        FLOAT_OP=tl.ir.REDUCE_OP.FADD,
        INT_OP=tl.ir.REDUCE_OP.ADD,
        input=input,
        axis=axis,
        builder=builder,
    )


@tl.builtin
@_add_reduction_docstr("sum")
def sum(input, axis, _builder=None):
    axis = tl._constexpr_to_value(axis)
    return _i_sum(input, axis, _builder)


def _i_xor_sum(
    input: tl.tensor,
    axis: int,
    builder: tl.ir.builder,
) -> tl.tensor:
    scalar_ty = input.type.scalar
    if not scalar_ty.is_int():
        raise ValueError("xor_sum only supported for integers")
    return _reduce_impl(
        name="sum",
        FLOAT_OP=tl.ir.REDUCE_OP.XOR,
        INT_OP=tl.ir.REDUCE_OP.XOR,
        input=input,
        axis=axis,
        builder=builder,
    )


@tl.builtin
@_add_reduction_docstr("xor sum")
def xor_sum(input, axis, _builder=None):
    axis = tl._constexpr_to_value(axis)
    return _i_xor_sum(input, axis, _builder)


@triton.jit
@_add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding=False):
    z = x - tl.max(x, 0)
    num = tl.exp(z)
    den = tl.sum(num, 0)
    return tl.fdiv(num, den, ieee_rounding)
