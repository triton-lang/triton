from __future__ import division, annotations

from triton._C.libtriton.triton import ir

from triton import base, jitlib


@base.builtin
def where(condition, x, y, _builder: ir.builder = None) -> base.tensor:
    """
    Returns a tensor of elements from either :code:`x` or :code:`y`, depending on :code:`condition`.

    Note that :code:`x` and :code:`y` are always evaluated regardless of the value of :code:`condition`.

    If you want to avoid unintended memory operations, use the :code:`mask` arguments in `tr.load` and `tr.store` instead.

    The shape of :code:`x` and :code:`y` are both broadcast to the shape of :code:`condition`.
    :code:`x` and :code:`y` must have the data type.

    :param condition: When True (nonzero), yield x, otherwise yield y.
    :type condition: Block of tr.bool
    :param x: values selected at indices where condition is True.
    :param y: values selected at indices where condition is False.
    """
    condition = base._to_tensor(condition, _builder)
    x = base._to_tensor(x, _builder)
    y = base._to_tensor(y, _builder)
    return base._where(condition, x, y, _builder)


@jitlib.jit
def minimum(x, y):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return where(x < y, x, y)
