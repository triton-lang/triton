# flake8: noqa
import triton
import triton.language.standard as tl_standard
from .._runtime import jit
from triton import knobs
from . import _core as ttgl

_IMPORT_FROM_TRITON = [
    "cdiv",
    "sum",
    "max",
    "min",
    "reduce_or",
    "xor_sum",
]

__all__ = [
    "full_like",
    "zeros",
    "zeros_like",
    *_IMPORT_FROM_TRITON,
]

for name in _IMPORT_FROM_TRITON:
    # Convert JITFunction -> GluonJitFunction
    fn = getattr(tl_standard, name)
    assert knobs.runtime.interpret or isinstance(fn, triton.runtime.JITFunction)
    # Wrap the function and preserve its original docstring
    gluon_fn = jit(fn.fn)
    gluon_fn.__doc__ = fn.__doc__
    globals()[name] = gluon_fn


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
