from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.hopper import tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

# fmt: off


def create_ragged_descriptor_host(T, block_shape, layout, ragged_dim=0):
    """
    Given a 2- or 3-dimensional tensor T, this creates a 'ragged descriptor'
    which behaves like a concatenation (along the first axis) of subarrays
    of potentially unequal size.

    The load_ragged and store_ragged device functions can be used to read
    and write from subarrays T[slice_off : slice_off + slice_size]
    with hardware bounds-checking preventing any sort of leakage outside
    the subarray.

    This is the host-side version that creates a descriptor on the host.
    """

    block_shape = list(block_shape)
    tensor_shape = list(T.shape)
    rank = len(tensor_shape)

    if ragged_dim < 0:
        ragged_dim += rank

    assert 0 <= ragged_dim < rank - 1, "last dimension cannot be ragged"
    assert rank <= 3, "read-write ragged descriptors must have at most 3 dimensions"
    assert rank >= 2, "read-write ragged descriptors must have at least 2 dimensions"
    assert len(block_shape) == rank, "block shape must have same length as tensor shape"

    if rank == 2:
        assert ragged_dim == 0, f"For 2D tensors, only ragged_dim=0 is supported, got ragged_dim={ragged_dim}"
    elif rank == 3:
        assert ragged_dim in [0, 1], f"For 3D tensors, only ragged_dim=0 or ragged_dim=1 is supported, got ragged_dim={ragged_dim}"

    max_int = 0x7fff0000
    billion = 0x40000000  # == 2**30

    assert tensor_shape[ragged_dim] <= billion, "number of rows may not exceed 2**30"
    tensor_shape[ragged_dim] = billion
    ragged_stride = T.stride(ragged_dim)

    # we prepend an extra two dimensions and rely on the fact that pointers
    # have 64-bit wraparound semantics:
    tma_strides = [2**34 - ragged_stride, ragged_stride] + [T.stride(i) for i in range(rank)]
    tma_shape = [max_int, max_int] + tensor_shape
    box_shape = [1, 1] + block_shape

    return TensorDescriptor(T, tma_shape, tma_strides, box_shape, layout)


@gluon.jit
def create_ragged_descriptor_device_2d(
    base_ptr,
    dim0: ttgl.constexpr, dim1: ttgl.constexpr,
    stride0: ttgl.constexpr, stride1: ttgl.constexpr,
    block0: ttgl.constexpr, block1: ttgl.constexpr,
    layout,
    ragged_dim: ttgl.constexpr
):
    """
    Create a ragged descriptor for 2D tensors inside a kernel (device-side version).

    Args:
        base_ptr: Pointer to the tensor data
        dim0, dim1: Tensor dimensions
        stride0, stride1: Tensor strides
        block0, block1: Block shape for TMA
        layout: NVMMASharedLayout for shared memory
        ragged_dim: Which dimension is ragged (only 0 is supported for 2D)
    """
    ttgl.static_assert(ragged_dim == 0, "For 2D tensors, only ragged_dim=0 is supported")

    max_int: ttgl.constexpr = 0x7fff0000
    billion: ttgl.constexpr = 0x40000000
    return tma.make_tensor_descriptor(
        base_ptr,
        shape=[max_int, max_int, billion, dim1],
        strides=[2**34 - stride0, stride0, stride0, stride1],
        block_shape=[1, 1, block0, block1],
        layout=layout,
    )


@gluon.jit
def create_ragged_descriptor_device_3d(
    base_ptr,
    dim0: ttgl.constexpr, dim1: ttgl.constexpr, dim2: ttgl.constexpr,
    stride0: ttgl.constexpr, stride1: ttgl.constexpr, stride2: ttgl.constexpr,
    block0: ttgl.constexpr, block1: ttgl.constexpr, block2: ttgl.constexpr,
    layout,
    ragged_dim: ttgl.constexpr
):
    """
    Create a ragged descriptor for 3D tensors inside a kernel (device-side version).

    Args:
        base_ptr: Pointer to the tensor data
        dim0, dim1, dim2: Tensor dimensions
        stride0, stride1, stride2: Tensor strides
        block0, block1, block2: Block shape for TMA
        layout: NVMMASharedLayout for shared memory
        ragged_dim: Which dimension is ragged (0 or 1 supported for 3D)
    """
    ttgl.static_assert(ragged_dim == 0 or ragged_dim == 1,
                      "For 3D tensors, only ragged_dim=0 or ragged_dim=1 is supported")

    max_int: ttgl.constexpr = 0x7fff0000
    billion: ttgl.constexpr = 0x40000000
    if ragged_dim == 0:
        return tma.make_tensor_descriptor(
            base_ptr,
            shape=[max_int, max_int, billion, dim1, dim2],
            strides=[2 ** 34 - stride0, stride0, stride0, stride1, stride2],
            block_shape=[1, 1, block0, block1, block2],
            layout=layout,
        )
    else:  # ragged_dim == 1
        return tma.make_tensor_descriptor(
            base_ptr,
            shape=[max_int, max_int, dim0, billion, dim2],
            strides=[2 ** 34 - stride1, stride1, stride0, stride1, stride2],
            block_shape=[1, 1, block0, block1, block2],
            layout=layout,
        )


@gluon.jit
def to_ragged_indices(slice_off, slice_size, row):
    """
    Helper function for load_ragged and store_ragged.
    """

    billion = 0x40000000  # == 2**30
    x = billion - slice_size + row
    y = slice_off + slice_size

    return billion, y, x


@gluon.jit
def load_ragged(TMA, slice_off, slice_size, coords, barrier, result, ragged_dim: ttgl.constexpr = 0, pred=True):
    """
    Read from a subarray T[slice_off : slice_off + slice_size] with
    hardware bounds-checking, where reading outside the subarray gives zeros.


    Coords should be an appropriately-sized list of integers, just like in TMA.load().
    """

    c0, c1, c2 = to_ragged_indices(slice_off, slice_size, coords[ragged_dim])
    full_coords = [c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:]
    tma.async_copy_global_to_shared(TMA, full_coords, barrier, result, pred)


@gluon.jit
def store_ragged(TMA, slice_off, slice_size, coords, data, ragged_dim: ttgl.constexpr = 0):
    """
    Write to a subarray T[slice_off : slice_off + slice_size] with
    hardware bounds-checking, where writes outside the subarray are masked
    correctly.

    Coords should be an appropriately-sized list of integers, just like in TMA.store().
    """

    c0, c1, c2 = to_ragged_indices(slice_off, slice_size, coords[ragged_dim])
    full_coords = [c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:]
    tma.async_copy_shared_to_global(TMA, full_coords, data)
