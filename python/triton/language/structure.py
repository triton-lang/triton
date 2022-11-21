from __future__ import division, annotations

from typing import Tuple

import triton
from triton import language as tl


@tl.builtin
def reshape(
    input: tl.tensor,
    shape,
    _builder: tl.ir.builder = None,
) -> tl.tensor:
    """
    Tries to reshape the given tensor to a new shape.

    :param input: The input tensor.
    :type input:
    :param shape: The desired shape.
    :type shape: Tuple[int]

    """
    shape = [x.value for x in shape]
    return tl._reshape(input, shape, _builder)


@triton.jit
def ravel(x: tl.tensor) -> tl.tensor:
    """
    Returns a contiguous flattened view of :code:`x`

    :param x: the input tensor
    :type x: Block
    """
    return reshape(x, [x.numel])


@triton.jit
def swizzle2d(
    i: int,
    j: int,
    size_i: int,
    size_j: int,
    size_g: int,
) -> Tuple[int, int]:
    """
    transformes indices of a row-major size_i*size_j matrix into those
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
    size_g = tl.minimum(size_i - off_i, size_g)
    # new row and column indices
    new_i = off_i + (ij % size_g)
    new_j = (ij % size_gj) // size_g
    return new_i, new_j
