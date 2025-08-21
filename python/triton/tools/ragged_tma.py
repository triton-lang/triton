import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# fmt: off


def create_ragged_descriptor(T, block_shape, ragged_dim=0):
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
    rank = len(tensor_shape)

    if ragged_dim < 0:
        ragged_dim += rank

    assert 0 <= ragged_dim < rank - 1, "last dimension cannot be ragged"
    assert rank <= 3, "read-write ragged descriptors must have at most 3 dimensions"

    assert len(block_shape) == rank, "block shape must have same length as tensor shape"

    max_int = 0x7fff0000
    billion = 0x40000000  # == 2**30

    assert tensor_shape[ragged_dim] <= billion, "number of rows may not exceed 2**30"
    tensor_shape[ragged_dim] = billion
    ragged_stride = T.stride(ragged_dim)

    # we prepend an extra two dimensions and rely on the fact that pointers
    # have 64-bit wraparound semantics:
    tma_stride = [2**34 - ragged_stride, ragged_stride] + [T.stride(i) for i in range(rank)]
    tma_shape  = [max_int, max_int] + tensor_shape
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
def load_ragged(TMA, batch_offset, batch_size, coords, ragged_dim: tl.constexpr = 0):
    """
    Read from a subarray T[batch_offset : batch_offset + batch_size] with
    hardware bounds-checking, where reading outside the subarray gives zeros.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.load().
    """

    tl.static_assert(len(TMA.shape) == len(coords) + 2, "TMA must be a read-write ragged descriptor")

    c0, c1, c2 = to_ragged_indices(batch_offset, batch_size, coords[ragged_dim])
    data = TMA.load([c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:])
    data = tl.reshape(data, data.shape[2:])
    return data


@triton.jit
def store_ragged(TMA, batch_offset, batch_size, coords, data, ragged_dim: tl.constexpr = 0):
    """
    Write to a subarray T[batch_offset : batch_offset + batch_size] with
    hardware bounds-checking, where writes outside the subarray are masked
    correctly.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.store().
    """

    c0, c1, c2 = to_ragged_indices(batch_offset, batch_size, coords[ragged_dim])
    data = tl.reshape(data, [1, 1] + data.shape)
    TMA.store([c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:], data)
