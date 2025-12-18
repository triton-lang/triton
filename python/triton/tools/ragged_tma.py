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
    and write from subarrays T[slice_off : slice_off + slice_size]
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
def _compute_ragged_descriptor_params_2d(
    shape_0, shape_1,
    stride_0, stride_1: tl.constexpr,
    ragged_dim: tl.constexpr
):
    tl.static_assert(
        ragged_dim < 1,
        "Using last dim as ragged dim is not supported"
    )

    max_int: tl.constexpr = 0x7fff0000
    billion: tl.constexpr = 0x40000000
    two_to_34 = tl.to_tensor(2**34)
    return (
        [max_int, max_int, billion, shape_1],
        [two_to_34 - stride_0, stride_0, stride_0, stride_1],
    )


@triton.jit
def _compute_ragged_descriptor_params_3d(
    shape_0, shape_1, shape_2,
    stride_0, stride_1, stride_2: tl.constexpr,
    ragged_dim: tl.constexpr
):
    tl.static_assert(
        ragged_dim < 2,
        "Using last dim as ragged dim is not supported"
    )

    max_int: tl.constexpr = 0x7fff0000
    billion: tl.constexpr = 0x40000000
    two_to_34 = tl.to_tensor(2**34)
    if ragged_dim == 0:
        return (
            [max_int, max_int, billion, shape_1, shape_2],
            [two_to_34 - stride_0, stride_0, stride_0, stride_1, stride_2],
        )
    else:
        return (
            [max_int, max_int, shape_0, billion, shape_2],
            [two_to_34 - stride_1, stride_1, stride_0, stride_1, stride_2],
        )


@triton.jit
def create_ragged_descriptor_device_2d(
    base_ptr,
    shape_0, shape_1,
    stride_0, stride_1: tl.constexpr,
    block_shape_0: tl.constexpr, block_shape_1: tl.constexpr,
    ragged_dim: tl.constexpr
):
    shape, stride = _compute_ragged_descriptor_params_2d(
        shape_0, shape_1,
        stride_0, stride_1,
        ragged_dim
    )
    one: tl.constexpr = 1
    return tl.make_tensor_descriptor(
        base_ptr,
        shape=shape,
        strides=[stride[0], stride[1], stride[2], stride_1],
        block_shape=[one, one, block_shape_0, block_shape_1],
    )


@triton.jit
def create_ragged_descriptor_device_3d(
    base_ptr,
    shape_0, shape_1, shape_2,
    stride_0, stride_1, stride_2: tl.constexpr,
    block_shape_0: tl.constexpr, block_shape_1: tl.constexpr, block_shape_2: tl.constexpr,
    ragged_dim: tl.constexpr
):
    shape, stride = _compute_ragged_descriptor_params_3d(
        shape_0, shape_1, shape_2,
        stride_0, stride_1, stride_2,
        ragged_dim
    )
    one: tl.constexpr = 1
    return tl.make_tensor_descriptor(
        base_ptr,
        shape=shape,
        strides=[stride[0], stride[1], stride[2], stride[3], stride_2],
        block_shape=[one, one, block_shape_0, block_shape_1, block_shape_2],
    )


@triton.jit
def to_ragged_indices(slice_off, slice_size, row):
    """
    Helper function for load_ragged and store_ragged.
    """
    billion = 0x40000000  # == 2**30
    x = billion - slice_size + row
    y = slice_off + slice_size
    return billion, y, x


@triton.jit
def load_ragged(TMA, slice_off, slice_size, coords, ragged_dim: tl.constexpr = 0):
    """
    Read from a subarray T[slice_off : slice_off + slice_size] with
    hardware bounds-checking, where reading outside the subarray gives zeros.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.load().
    """

    tl.static_assert(len(TMA.shape) == len(coords) + 2, "TMA must be a read-write ragged descriptor")

    c0, c1, c2 = to_ragged_indices(slice_off, slice_size, coords[ragged_dim])
    data = TMA.load([c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:])
    data = tl.reshape(data, data.shape[2:])
    return data


@triton.jit
def store_ragged(TMA, slice_off, slice_size, coords, data, ragged_dim: tl.constexpr = 0):
    """
    Write to a subarray T[slice_off : slice_off + slice_size] with
    hardware bounds-checking, where writes outside the subarray are masked
    correctly.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.store().
    """

    c0, c1, c2 = to_ragged_indices(slice_off, slice_size, coords[ragged_dim])
    data = tl.reshape(data, [1, 1] + data.shape)
    TMA.store([c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:], data)


@triton.jit
def atomic_add_ragged(TMA, slice_off, slice_size, coords, data, ragged_dim: tl.constexpr = 0):
    """
    Atomic add into a subarray T[slice_off : slice_off + slice_size] with
    hardware bounds-checking, where adds outside the subarray are masked
    correctly.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.atomic_add().
    """

    c0, c1, c2 = to_ragged_indices(slice_off, slice_size, coords[ragged_dim])
    data = tl.reshape(data, [1, 1] + data.shape)
    TMA.atomic_add([c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:], data)
