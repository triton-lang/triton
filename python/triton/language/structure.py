from __future__ import annotations

from typing import List

import triton
from triton import impl
import triton.language as tl


# -----------------------
# Shape Manipulation
# -----------------------


def _i_trans(
    input: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    if len(input.shape) != 2:
        raise ValueError("Only 2D tensors can be transposed")
    ret_type = tl.block_type(
        input.type.scalar,
        [input.shape[1], input.shape[0]],
    )
    return tl.tensor(
        builder.create_trans(input.handle),
        ret_type,
    )


@triton.builtin
def trans(input, _builder=None):
    return _i_trans(input, _builder)


def _i_view(
    input: tl.tensor,
    dst_shape: List[int],
    builder: tl.ir.builder,
) -> tl.tensor:
    # TODO: disable when TritonToTritonGPU handles views properly
    assert len(input.shape) == len(dst_shape)
    numel = 1
    for s in dst_shape:
        numel *= s
    if input.type.numel != numel:
        raise ValueError("cannot view block of different shape")
    ret_ty = tl.block_type(input.type.scalar, dst_shape)
    return tl.tensor(builder.create_view(input.handle, dst_shape), ret_ty)


@triton.builtin
def view(input, shape, _builder=None):
    """
    Returns a tensor with the same elements as `input` but a different shape.
    The order of the elements may not be preserved.

    :param input: The input tensor.
    :type input:
    :param shape: The desired shape.
    :type shape: Tuple[int]

    """
    shape = [x.value for x in shape]
    return impl._i_view(input, shape, _builder)


# -----------------------
# Conditioning
# -----------------------


@triton.jit
def ravel(x):
    """
    Returns a contiguous flattened view of :code:`x`

    :param x: the input tensor
    :type x: Block
    """
    return tl.view(x, [x.numel])


@triton.jit
def swizzle2d(
    i,
    j,
    size_i,
    size_j,
    size_g,
):
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
    size_g = tl.minimum(size_i - off_i, size_g)
    # new row and column indices
    new_i = off_i + (ij % size_g)
    new_j = (ij % size_gj) // size_g
    return new_i, new_j
