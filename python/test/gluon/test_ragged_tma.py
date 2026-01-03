import torch
import pytest
import triton

from triton._internal_testing import is_blackwell
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import tma, mbarrier
from triton.experimental.gluon.tools.ragged_tma import (
    create_ragged_descriptor_host,
    create_ragged_descriptor_device_2d,
    create_ragged_descriptor_device_3d,
    to_ragged_indices,
)


@gluon.jit
def example_load_store_kernel_host_desc(X, Y, x_off, y_off, num_slices, ragged_dim: ttgl.constexpr, ndim: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(X.dtype, X.block_shape, X.layout)

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.constexpr(mbarrier.MBarrierLayout()))
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, X.block_type.nbytes)

    c0_x, c1_x, c2_x = to_ragged_indices(x_off, num_slices, 0)

    if ndim == 2:
        if ragged_dim == 0:
            tma.async_copy_global_to_shared(X, [c0_x, c1_x, c2_x, 0], bar, smem)
    else:  # ndim == 3
        if ragged_dim == 0:
            tma.async_copy_global_to_shared(X, [c0_x, c1_x, c2_x, 0, 0], bar, smem)
        else:  # ragged_dim == 1
            tma.async_copy_global_to_shared(X, [c0_x, c1_x, 0, c2_x, 0], bar, smem)

    # Wait for load
    mbarrier.wait(bar, phase=0)

    # Store to ragged descriptor
    c0_y, c1_y, c2_y = to_ragged_indices(y_off, num_slices, 0)

    if ndim == 2:
        if ragged_dim == 0:
            tma.async_copy_shared_to_global(Y, [c0_y, c1_y, c2_y, 0], smem)
    else:  # ndim == 3
        if ragged_dim == 0:
            tma.async_copy_shared_to_global(Y, [c0_y, c1_y, c2_y, 0, 0], smem)
        else:  # ragged_dim == 1
            tma.async_copy_shared_to_global(Y, [c0_y, c1_y, 0, c2_y, 0], smem)
    tma.store_wait(0)

    smem._keep_alive()


@gluon.jit
def example_load_store_kernel_device_desc_2d(
    x_ptr, y_ptr,
    x_off, y_off, num_slices,
    dim0: ttgl.constexpr, dim1: ttgl.constexpr,
    stride0: ttgl.constexpr, stride1: ttgl.constexpr,
    block0: ttgl.constexpr, block1: ttgl.constexpr,
    layout: ttgl.constexpr,
    ragged_dim: ttgl.constexpr,
):
    X = create_ragged_descriptor_device_2d(x_ptr, dim0, dim1, stride0, stride1, block0, block1, layout, ragged_dim)
    Y = create_ragged_descriptor_device_2d(y_ptr, dim0, dim1, stride0, stride1, block0, block1, layout, ragged_dim)

    smem = ttgl.allocate_shared_memory(X.dtype, X.block_shape, layout)

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.constexpr(mbarrier.MBarrierLayout()))
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, X.block_type.nbytes)

    c0_x, c1_x, c2_x = to_ragged_indices(x_off, num_slices, 0)

    tma.async_copy_global_to_shared(X, [c0_x, c1_x, c2_x, 0], bar, smem)

    mbarrier.wait(bar, phase=0)

    c0_y, c1_y, c2_y = to_ragged_indices(y_off, num_slices, 0)

    tma.async_copy_shared_to_global(Y, [c0_y, c1_y, c2_y, 0], smem)

    smem._keep_alive()


@gluon.jit
def example_load_store_kernel_device_desc_3d(
    x_ptr, y_ptr,
    x_off, y_off, num_slices,
    dim0: ttgl.constexpr, dim1: ttgl.constexpr, dim2: ttgl.constexpr,
    stride0: ttgl.constexpr, stride1: ttgl.constexpr, stride2: ttgl.constexpr,
    block0: ttgl.constexpr, block1: ttgl.constexpr, block2: ttgl.constexpr,
    layout: ttgl.constexpr,
    ragged_dim: ttgl.constexpr,
):
    X = create_ragged_descriptor_device_3d(x_ptr, dim0, dim1, dim2, stride0, stride1, stride2, block0, block1, block2, layout, ragged_dim)
    Y = create_ragged_descriptor_device_3d(y_ptr, dim0, dim1, dim2, stride0, stride1, stride2, block0, block1, block2, layout, ragged_dim)

    smem = ttgl.allocate_shared_memory(X.dtype, X.block_shape, layout)

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.constexpr(mbarrier.MBarrierLayout()))
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, X.block_type.nbytes)

    c0_x, c1_x, c2_x = to_ragged_indices(x_off, num_slices, 0)

    if ragged_dim == 0:
        tma.async_copy_global_to_shared(X, [c0_x, c1_x, c2_x, 0, 0], bar, smem)
    else:  # ragged_dim == 1
        tma.async_copy_global_to_shared(X, [c0_x, c1_x, 0, c2_x, 0], bar, smem)

    mbarrier.wait(bar, phase=0)

    c0_y, c1_y, c2_y = to_ragged_indices(y_off, num_slices, 0)

    if ragged_dim == 0:
        tma.async_copy_shared_to_global(Y, [c0_y, c1_y, c2_y, 0, 0], smem)
    else:  # ragged_dim == 1
        tma.async_copy_shared_to_global(Y, [c0_y, c1_y, 0, c2_y, 0], smem)
    tma.store_wait(0)

    smem._keep_alive()


def _generate_test_params():
    dtypes = ["bfloat16", "float16", "float32", "int8", "int16", "int32"]
    modes = ["host", "device"]

    params = []
    for dtype in dtypes:
        for mode in modes:
            # 2D tensors: only ragged_dim=0 is valid
            params.append((dtype, mode, 2, 0))
            # 3D tensors: ragged_dim=0 and ragged_dim=1 are valid
            params.append((dtype, mode, 3, 0))
            params.append((dtype, mode, 3, 1))

    return params


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,descriptor_mode,ndim,ragged_dim", _generate_test_params())
def test_ragged_tma(dtype_name, descriptor_mode, ndim, ragged_dim):

    torch_dtype = getattr(torch, dtype_name)
    gluon_dtype = getattr(ttgl, dtype_name)

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    elem_size = torch.empty(0, dtype=torch_dtype).element_size()

    if ndim == 2:
        tensor_shape = (128, 80)
        block_shape = [32, 128]
        strides = [80, 1]
    else:  # ndim == 3
        if elem_size >= 4:  # float32, int32
            if ragged_dim == 0:
                tensor_shape = (64, 32, 32)
                block_shape = [16, 16, 32]
                strides = [32 * 32, 32, 1]
            else:  # ragged_dim == 1
                tensor_shape = (64, 32, 32)
                block_shape = [32, 16, 32]
                strides = [32 * 32, 32, 1]
        else:  # 2-byte or 1-byte types
            if ragged_dim == 0:
                tensor_shape = (64, 32, 64)
                block_shape = [16, 32, 64]
                strides = [32 * 64, 64, 1]
            else:  # ragged_dim == 1
                tensor_shape = (64, 32, 64)
                block_shape = [32, 16, 64]
                strides = [32 * 64, 64, 1]

    src = torch.ones(tensor_shape, dtype=torch_dtype, device="cuda")
    dst = torch.zeros(tensor_shape, dtype=torch_dtype, device="cuda")

    layout = ttgl.NVMMASharedLayout.get_default_for([1, 1] + block_shape, gluon_dtype)

    num_slices = min(block_shape[ragged_dim] - 1, tensor_shape[ragged_dim] // 3)
    x_off = 0
    y_off = (tensor_shape[ragged_dim] - num_slices) // 2

    if descriptor_mode == "host":
        X = create_ragged_descriptor_host(src, block_shape, layout, ragged_dim)
        Y = create_ragged_descriptor_host(dst, block_shape, layout, ragged_dim)

        example_load_store_kernel_host_desc[(1,)](
            X, Y, x_off, y_off, num_slices, ragged_dim, ndim,
            num_warps=4,
        )
    else:
        if ndim == 2:
            example_load_store_kernel_device_desc_2d[(1,)](
                src, dst,
                x_off, y_off, num_slices,
                tensor_shape[0], tensor_shape[1],
                strides[0], strides[1],
                block_shape[0], block_shape[1],
                layout, ragged_dim,
                num_warps=4,
            )
        else:  # ndim == 3
            example_load_store_kernel_device_desc_3d[(1,)](
                src, dst,
                x_off, y_off, num_slices,
                tensor_shape[0], tensor_shape[1], tensor_shape[2],
                strides[0], strides[1], strides[2],
                block_shape[0], block_shape[1], block_shape[2],
                layout, ragged_dim,
                num_warps=4,
            )

    if ragged_dim == 0:
        if ndim == 2:
            before = dst[:y_off, :block_shape[1]]
            copied = dst[y_off:y_off + num_slices, :block_shape[1]]
            after = dst[y_off + num_slices:, :block_shape[1]]
        else:  # ndim == 3
            before = dst[:y_off, :block_shape[1], :block_shape[2]]
            copied = dst[y_off:y_off + num_slices, :block_shape[1], :block_shape[2]]
            after = dst[y_off + num_slices:, :block_shape[1], :block_shape[2]]
    else:  # ragged_dim == 1
        before = dst[:block_shape[0], :y_off, :block_shape[2]]
        copied = dst[:block_shape[0], y_off:y_off + num_slices, :block_shape[2]]
        after = dst[:block_shape[0], y_off + num_slices:, :block_shape[2]]

    res0 = torch.all(before == 0.0).item()
    res1 = torch.all(copied == 1.0).item()
    res2 = torch.all(after == 0.0).item()

    assert [res0, res1, res2] == [True, True, True], \
        f"Failed for {ndim}D {descriptor_mode} mode ragged_dim={ragged_dim}: before={res0}, copied={res1}, after={res2}"
