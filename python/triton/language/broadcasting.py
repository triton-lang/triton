from __future__ import annotations

from typing import Tuple

import triton.language as tl


@tl.builtin
def broadcast(
    input: tl.tensor,
    other: tl.tensor,
    _builder: tl.ir.builder = None,
) -> Tuple[tl.tensor, tl.tensor]:
    """
    Tries to broadcast the two given blocks to a common compatible shape.

    :param input: The first input tensor.
    :type input: Block
    :param other: The second input tensor.
    :type other: Block
    """
    return tl._broadcast_impl_value(input, other, _builder)


@tl.builtin
def broadcast_to(input: tl.tensor, shape, _builder: tl.ir.builder = None) -> tl.tensor:
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape: Tuple[int]
    """
    return tl._broadcast_impl_shape(input, shape, _builder)
