from __future__ import division, annotations

from typing import TypeVar, Callable

import triton
import triton.language as tl
from triton.language.math import _add_math_1arg_docstr


CallableT = TypeVar("CallableT", bound=Callable)


def _add_reduction_docstr(name: str) -> Callable[[CallableT], CallableT]:
    def _decorator(func: CallableT) -> CallableT:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :param axis: the dimension along which the reduction should be done
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _reduce_impl(
    input: tl.tensor,
    axis: int,
    builder: tl.ir.builder,
    name: str,
    FLOAT_OP: tl.ir.REDUCE_OP,
    INT_OP: tl.ir.REDUCE_OP,
) -> tl.tensor:
    scalar_ty = input.type.scalar
    # input is extended to 32-bits if necessary
    # this increases numerical accuracy and can be done pretty much for free
    # on GPUs
    if scalar_ty.is_int() and scalar_ty.int_bitwidth <= 32:
        input = tl._cast(input, tl.int32, builder)

    # hardware doesn't support FMAX, FMIN, CMP for bfloat16
    if scalar_ty is tl.bfloat16:
        input = tl._cast(input, tl.float32, builder)

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
    if len(ret_shape) == 0:
        res_ty = scalar_ty
    else:
        res_ty = tl.block_type(scalar_ty, ret_shape)

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


def _max(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    return _reduce_impl(
        input,
        axis,
        builder,
        "max",
        tl.ir.REDUCE_OP.FMAX,
        tl.ir.REDUCE_OP.MAX,
    )


@tl.builtin
@_add_reduction_docstr("maximum")
def max(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _max(input, axis, _builder)


def _argmax(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    return _reduce_impl(
        input,
        axis,
        builder,
        "argmax",
        tl.ir.REDUCE_OP.ARGFMAX,
        tl.ir.REDUCE_OP.ARGMAX,
    )


@tl.builtin
@_add_reduction_docstr("maximum index")
def argmax(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _argmax(input, axis, _builder)


def _min(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    return _reduce_impl(
        input,
        axis,
        builder,
        "min",
        tl.ir.REDUCE_OP.FMIN,
        tl.ir.REDUCE_OP.MIN,
    )


@tl.builtin
@_add_reduction_docstr("minimum")
def min(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _min(input, axis, _builder)


def _argmin(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    return _reduce_impl(
        input,
        axis,
        builder,
        "argmin",
        tl.ir.REDUCE_OP.ARGFMIN,
        tl.ir.REDUCE_OP.ARGMIN,
    )


@tl.builtin
@_add_reduction_docstr("minimum index")
def argmin(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _argmin(input, axis, _builder)


def _sum(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    return _reduce_impl(
        input,
        axis,
        builder,
        "sum",
        tl.ir.REDUCE_OP.FADD,
        tl.ir.REDUCE_OP.ADD,
    )


@tl.builtin
@_add_reduction_docstr("sum")
def sum(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _sum(input, axis, _builder)


def _xor_sum(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    scalar_ty = input.type.scalar
    if not scalar_ty.is_int():
        raise ValueError("xor_sum only supported for integers")
    return _reduce_impl(
        input,
        axis,
        builder,
        "sum",
        tl.ir.REDUCE_OP.XOR,
        tl.ir.REDUCE_OP.XOR,
    )


@tl.builtin
@_add_reduction_docstr("xor sum")
def xor_sum(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _xor_sum(input, axis, _builder)


@triton.jit
def maximum(x, y) -> tl.tensor:
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return tl.where(x > y, x, y)


# minimum is in trition.core
minimum = tl.minimum


@triton.jit
@_add_math_1arg_docstr("sigmoid")
def sigmoid(x) -> tl.tensor:
    return 1 / (1 + tl.exp(-x))


@triton.jit
@_add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding: tl.constexpr = False) -> tl.tensor:
    z = x - max(x, 0)
    num = tl.exp(z)
    den = sum(num, 0)
    return tl.fdiv(num, den, ieee_rounding)
