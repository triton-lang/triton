import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# fmt: off

def create_ragged_descriptor(T, block_shape):
    """
    Given a 2- or 3-dimensional tensor T, this creates a 'ragged descriptor'
    which behaves like a concatenation (along the first axis) of subarrays
    of potentially unequal size.

    The load_ragged and store_ragged device functions can be used to read
    and write from subarrays T[batch_offset : batch_offset + batch_size]
    with hardware bounds-checking preventing any sort of leakage outside
    the subarray.
    """

    block_shape = list(block_shape)
    tensor_shape = list(T.shape)

    assert 2 <= len(tensor_shape) <= 3, "ragged tensors must have dimension 2 or 3"
    assert len(tensor_shape) == len(block_shape), "block shape must match tensor shape"

    max_int = 0x7fff0000
    billion = 0x40000000  # == 2**30

    assert tensor_shape[0] <= billion, "number of rows may not exceed 2**30"

    # we prepend an extra two dimensions and rely on the fact that pointers
    # have 64-bit wraparound semantics:
    tma_stride = [2**34 - T.stride(0), T.stride(0)] + [T.stride(i) for i in range(len(tensor_shape))]
    tma_shape  = [max_int, max_int, billion] + tensor_shape[1:]
    box_shape  = [1, 1] + block_shape

    return TensorDescriptor(T, tma_shape, tma_stride, box_shape)


@triton.jit
def to_ragged_indices(batch_offset, batch_size, row):
    """
    Helper function for load_ragged and store_ragged.
    """

    billion = 0x40000000  # == 2**30
    x = billion - batch_size + row
    y = batch_offset + batch_size

    return billion, y, x


@triton.jit
def load_ragged(TMA, batch_offset, batch_size, coords):
    """
    Read from a subarray T[batch_offset : batch_offset + batch_size] with
    hardware bounds-checking, where reading outside the subarray gives zeros.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.load().
    """

    c0, c1, c2 = to_ragged_indices(batch_offset, batch_size, coords[0])
    data = TMA.load([c0, c1, c2] + coords[1:])
    data = tl.reshape(data, data.shape[2:])
    return data


@triton.jit
def store_ragged(TMA, batch_offset, batch_size, coords, data):
    """
    Write to a subarray T[batch_offset : batch_offset + batch_size] with
    hardware bounds-checking, where writes outside the subarray are masked
    correctly.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.store().
    """

    c0, c1, c2 = to_ragged_indices(batch_offset, batch_size, coords[0])
    data = tl.reshape(data, [1, 1] + data.shape)
    TMA.store([c0, c1, c2] + coords[1:], data)
