from __future__ import division, annotations

from typing import Callable, TypeVar

import triton
import triton.language as tl


CallableT = TypeVar("CallableT", bound=Callable)


def _umulhi(x: tl.tensor, y: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    x, y = tl._binary_op_type_checking_impl(x, y, builder=builder)
    return tl.tensor(
        builder.create_umulhi(x.handle, y.handle),
        x.type,
    )


@tl.builtin
def umulhi(x, y, _builder: tl.ir.builder = None) -> tl.tensor:
    x = tl._to_tensor(x, _builder)
    y = tl._to_tensor(y, _builder)
    return _umulhi(x, y, _builder)


@tl.builtin
def fdiv(x, y, ieee_rounding=False, _builder: tl.ir.builder = None) -> tl.tensor:
    ieee_rounding = tl._constexpr_to_value(ieee_rounding)
    return tl._fdiv(x, y, ieee_rounding, _builder)


def _add_math_1arg_docstr(name: str) -> Callable[[CallableT], CallableT]:
    def _decorator(func: CallableT) -> CallableT:
        docstr = """
    Computes the element-wise {name} of :code:`x`

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _exp(x: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_exp(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("exponential")
def exp(x, _builder: tl.ir.builder = None) -> tl.tensor:
    return _exp(x, _builder)


def _log(x: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_log(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("natural logarithm")
def log(x, _builder: tl.ir.builder = None) -> tl.tensor:
    return _log(x, _builder)


def _cos(x: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_cos(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("cosine")
def cos(x, _builder: tl.ir.builder = None) -> tl.tensor:
    return _cos(x, _builder)


def _sin(x: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_sin(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("sine")
def sin(x, _builder: tl.ir.builder = None) -> tl.tensor:
    return _sin(x, _builder)


def _sqrt(x: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_sqrt(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("square root")
def sqrt(x, _builder: tl.ir.builder = None) -> tl.tensor:
    return _sqrt(x, _builder)


@triton.jit
def abs(x) -> tl.tensor:
    return tl.where(x >= 0, x, -x)


@triton.jit
def cdiv(x, div) -> tl.tensor:
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type input: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div
