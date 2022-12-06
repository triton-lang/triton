from typing import Callable, TypeVar

import triton
from ..impl import base
from . import libdevice
import triton.language as tl

T = TypeVar("T")


def _add_math_1arg_docstr(name: str) -> Callable[[T], T]:
    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x`

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _i_umulhi(
    x: tl.tensor,
    y: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    x, y = base._binary_op_type_checking_impl(lhs=x, rhs=y, builder=builder)
    return libdevice.mulhi(x, y, _builder=builder)


@tl.builtin
def umulhi(x, y, _builder=None):
    x = tl._to_tensor(x, _builder)
    y = tl._to_tensor(y, _builder)
    return _i_umulhi(x, y, _builder)


@tl.builtin
def fdiv(x, y, ieee_rounding=False, _builder=None):
    ieee_rounding = tl._constexpr_to_value(ieee_rounding)
    return base._i_fdiv(x, y, ieee_rounding, _builder)


def _i_exp(
    x: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    return tl.tensor(builder.create_exp(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("exponential")
def exp(x, _builder=None):
    return _i_exp(x, _builder)


def _i_log(
    x: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    return tl.tensor(builder.create_log(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("natural logarithm")
def log(x, _builder=None):
    return _i_log(x, _builder)


def _i_cos(
    x: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    return tl.tensor(builder.create_cos(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("cosine")
def cos(x, _builder=None):
    return _i_cos(x, _builder)


def _i_sin(
    x: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    return tl.tensor(builder.create_sin(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("sine")
def sin(x, _builder=None):
    return _i_sin(x, _builder)


def _i_sqrt(
    x: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    return tl.tensor(builder.create_sqrt(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("square root")
def sqrt(x, _builder=None):
    return _i_sqrt(x, _builder)


@triton.jit
def abs(x):
    return tl.where(x >= 0, x, -x)


@triton.jit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type input: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div


# minimum is sourced from triton.impl.core


@triton.jit
def maximum(x, y):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return tl.where(x > y, x, y)


@triton.jit
@_add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    return 1 / (1 + tl.exp(-x))
