"""
Builtins required to complete the @jit cycle.
"""

from . import base
from .jitlib import jit


def _i_where(
    condition: base.tensor,
    x: base.tensor,
    y: base.tensor,
    builder: base.ir.builder,
) -> base.tensor:
    condition = base._i_cast(condition, base.int1, builder)
    if condition.type.is_block():
        condition, x = base._broadcast_impl_value(condition, x, builder)
        x, y = base._broadcast_impl_value(x, y, builder)
        condition, x = base._broadcast_impl_value(condition, x, builder)

    x, y = base._binary_op_type_checking_impl(
        lhs=x,
        rhs=y,
        builder=builder,
        allow_lhs_ptr=True,
        allow_rhs_ptr=True,
    )
    if not condition.type.is_block():
        condition, _ = base._broadcast_impl_value(condition, x, builder)
    ret_ty = x.type
    return base.tensor(
        builder.create_select(
            condition.handle,
            x.handle,
            y.handle,
        ),
        ret_ty,
    )


@base.builtin
def where(condition, x, y, _builder=None):
    """
    Returns a tensor of elements from either :code:`x` or :code:`y`, depending on :code:`condition`.

    Note that :code:`x` and :code:`y` are always evaluated regardless of the value of :code:`condition`.

    If you want to avoid unintended memory operations, use the :code:`mask` arguments in `triton.load` and `triton.store` instead.

    The shape of :code:`x` and :code:`y` are both broadcast to the shape of :code:`condition`.
    :code:`x` and :code:`y` must have the data type.

    :param condition: When True (nonzero), yield x, otherwise yield y.
    :type condition: Block of triton.bool
    :param x: values selected at indices where condition is True.
    :param y: values selected at indices where condition is False.
    """
    condition = base._to_tensor(condition, _builder)
    x = base._to_tensor(x, _builder)
    y = base._to_tensor(y, _builder)
    return _i_where(condition, x, y, _builder)


@jit
def minimum(x, y):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return where(x < y, x, y)
