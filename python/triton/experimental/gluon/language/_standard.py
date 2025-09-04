from typing import TypeVar
from triton.runtime.jit import JITFunction
import triton.language.standard as tl_standard
from .._runtime import GluonJITFunction, jit
from triton import knobs
from . import _core as ttgl

T = TypeVar("T")


def _import_from_triton(fn: JITFunction[T]) -> GluonJITFunction[T]:
    assert knobs.runtime.interpret or isinstance(fn, JITFunction)
    # Wrap the function and preserve its original docstring
    gluon_fn = jit(fn.fn)
    gluon_fn.__doc__ = fn.__doc__
    return gluon_fn


cdiv = _import_from_triton(tl_standard.cdiv)
sum = _import_from_triton(tl_standard.sum)
max = _import_from_triton(tl_standard.max)
min = _import_from_triton(tl_standard.min)
reduce_or = _import_from_triton(tl_standard.reduce_or)
xor_sum = _import_from_triton(tl_standard.xor_sum)


@jit
def zeros(shape, dtype, layout=None):
    """
    Create a tensor filled with zeros.

    Args:
        shape (Sequence[int]): The shape of the tensor.
        dtype (dtype): The data type for the tensor.
        layout (Optional[DistributedLayout]): The distributed layout of the tensor, defaults to AutoLayout().

    Returns:
        tensor: A tensor where every element is zero.
    """
    return ttgl.full(shape, 0, dtype, layout)


@jit
def full_like(input, value, shape=None, dtype=None, layout=None):
    """
    Create a tensor with the same properties as a given tensor, filled with a specified value.

    Args:
        input (tensor): Reference tensor to infer default shape, dtype, and layout.
        value (int or float): The fill value.
        shape (Sequence[int], optional): Target shape. Defaults to input.shape.
        dtype (dtype, optional): Target data type. Defaults to input.dtype.
        layout (DistributedLayout, optional): Target layout. Defaults to input.layout.

    Returns:
        tensor: A tensor where every element equals value.
    """
    return ttgl.full(
        input.shape if shape is None else shape,
        value,
        input.dtype if dtype is None else dtype,
        input.type.layout if layout is None else layout,
    )


@jit
def zeros_like(input, shape=None, dtype=None, layout=None):
    """
    Create a tensor with the same properties as a given tensor, filled with zeros.

    Args:
        input (tensor): Reference tensor to infer default shape, dtype, and layout.
        shape (Sequence[int], optional): Target shape. Defaults to input.shape.
        dtype (dtype, optional): Target data type. Defaults to input.dtype.
        layout (DistributedLayout, optional): Target layout. Defaults to input.layout.

    Returns:
        tensor: A tensor where every element is zero.
    """
    return full_like(input, 0, shape=shape, dtype=dtype, layout=layout)
