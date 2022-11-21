from __future__ import annotations

from typing import List

import triton
from triton import language as tl


def _arange(start: int, end: int, builder: tl.ir.builder) -> tl.tensor:
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("arange's arguments must be of type tl.constexpr")

    shape = [end - start]
    ret_ty = tl.block_type(tl.int32, shape)
    return tl.tensor(builder.get_range(start, end), ret_ty)


@tl.builtin
def arange(start, end, _builder: tl.ir.builder = None) -> tl.tensor:
    """
    Returns contiguous values within the open interval [:code:`start`, :code:`end`).

    :param start: Start of the interval. Must be a power of two.
    :type start: int
    :param stop: End of the interval. Must be a power of two >= start.
    :type stop: int
    """
    start = tl._constexpr_to_value(start)
    end = tl._constexpr_to_value(end)
    return _arange(start, end, _builder)


def _zeros(
    shape: List[int],
    dtype: tl.dtype,
    builder: tl.ir.builder,
) -> tl.tensor:
    _0 = tl.ir.constant.get_null_value(dtype.to_ir(builder))
    ret_ty = tl.block_type(dtype, shape)
    return tl.tensor(builder.create_splat(_0, shape), ret_ty)


@tl.builtin
def zeros(
    shape,
    dtype: tl.dtype,
    _builder: tl.ir.builder = None,
) -> tl.tensor:
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`float16`
    :type dtype: DType
    """
    for i, d in enumerate(shape):
        if not isinstance(d, tl.constexpr):
            raise TypeError(f"Shape element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(
                f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]"
            )
    shape = [x.value for x in shape]
    dtype = tl._constexpr_to_value(dtype)
    return tl._zeros(shape, dtype, _builder)


@triton.jit
def zeros_like(input: tl.tensor) -> tl.tensor:
    return zeros(input.shape, input.dtype)


def _cat(
    lhs: tl.tensor,
    rhs: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()
    assert lhs.type.shape[1:] == rhs.type.shape[1:]
    ret_shape = [lhs.type.shape[0] + rhs.type.shape[0]]
    ret_ty = tl.block_type(lhs.type.scalar, ret_shape)
    return tl.tensor(
        builder.create_cat(lhs.handle, rhs.handle),
        ret_ty,
    )


@tl.builtin
def cat(
    input: tl.tensor,
    other: tl.tensor,
    _builder: tl.ir.builder = None,
) -> tl.tensor:
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input:
    :param other: The second input tensor.
    :type other:
    """
    return _cat(input, other, _builder)
