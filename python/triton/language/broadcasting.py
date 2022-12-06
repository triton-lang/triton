from ..impl import base
import triton.language as tl


@tl.builtin
def broadcast(input, other, _builder=None):
    """
    Tries to broadcast the two given blocks to a common compatible shape.

    :param input: The first input tensor.
    :type input: Block
    :param other: The second input tensor.
    :type other: Block
    """
    return base._broadcast_impl_value(input, other, _builder)


@tl.builtin
def broadcast_to(input, shape, _builder=None):
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape: Tuple[int]
    """
    return base._broadcast_impl_shape(input, shape, _builder)
