from __future__ import annotations

from ..runtime.jit import jit
from . import core

# -----------------------
# Standard library
# -----------------------


@jit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type input: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div


@jit
@core._add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    return 1 / (1 + core.exp(-x))


@jit
@core._add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding=False):
    z = x - core.max(x, 0)
    num = core.exp(z)
    den = core.sum(num, 0)
    return core.fdiv(num, den, ieee_rounding)


@jit
def ravel(x):
    """
    Returns a contiguous flattened view of :code:`x`.

    :param x: the input tensor
    :type x: Block
    """
    return core.view(x, [x.numel])


@jit
def swizzle2d(i, j, size_i, size_j, size_g):
    """
    Transforms indices of a row-major size_i*size_j matrix into those
    of one where indices are row major for each group of size_j rows.
    For example, for size_i = size_j = 4 and size_g = 2, it will transform
    [[0 , 1 , 2 , 3 ],
     [4 , 5 , 6 , 7 ],
     [8 , 9 , 10, 11],
     [12, 13, 14, 15]]
    into
    [[0, 2,  4 , 6 ],
     [1, 3,  5 , 7 ],
     [8, 10, 12, 14],
     [9, 11, 13, 15]]
    """
    # "unrolled index in array"
    ij = i * size_j + j
    # number of elements in `size_g` groups
    # of `size_j` columns
    size_gj = size_g * size_j
    # index of the group in which (i,j) is
    group_id = ij // size_gj
    # row-index of the first element of this group
    off_i = group_id * size_g
    # last group may have fewer rows
    size_g = core.minimum(size_i - off_i, size_g)
    # new row and column indices
    new_i = off_i + (ij % size_g)
    new_j = (ij % size_gj) // size_g
    return new_i, new_j


@jit
def zeros(shape, dtype):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    return core.full(shape, 0, dtype)


@jit
def zeros_like(input):
    return zeros(input.shape, input.dtype)
