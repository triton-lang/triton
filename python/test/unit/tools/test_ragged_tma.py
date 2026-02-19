import torch
import pytest
import triton
import triton.language as tl
from typing import Optional

from triton._internal_testing import is_hopper_or_newer
from triton.tools.ragged_tma import (
    create_ragged_descriptor,
    create_ragged_descriptor_device_2d,
    create_ragged_descriptor_device_3d,
    load_ragged,
    store_ragged,
)


@triton.jit
def example_load_store_kernel_host_desc(x_desc, y_desc, x_off, y_off, num_slices, ragged_dim: tl.constexpr,
                                        ndim: tl.constexpr):
    if ndim == 2:
        data = load_ragged(x_desc, x_off, num_slices, [0, 0], ragged_dim)
        store_ragged(y_desc, y_off, num_slices, [0, 0], data, ragged_dim)
    else:
        data = load_ragged(x_desc, x_off, num_slices, [0, 0, 0], ragged_dim)
        store_ragged(y_desc, y_off, num_slices, [0, 0, 0], data, ragged_dim)


@triton.jit
def example_load_store_kernel_device_desc_2d(
    x_ptr,
    y_ptr,
    x_off,
    y_off,
    num_slices,
    shape_0,
    shape_1,
    stride_0,
    stride_1,
    block_shape_0: tl.constexpr,
    block_shape_1: tl.constexpr,
    ragged_dim: tl.constexpr,
):
    x_desc = create_ragged_descriptor_device_2d(
        x_ptr,
        shape_0,
        shape_1,
        stride_0,
        stride_1,
        block_shape_0,
        block_shape_1,
        ragged_dim,
    )
    y_desc = create_ragged_descriptor_device_2d(
        y_ptr,
        shape_0,
        shape_1,
        stride_0,
        stride_1,
        block_shape_0,
        block_shape_1,
        ragged_dim,
    )

    data = load_ragged(x_desc, x_off, num_slices, [0, 0], ragged_dim)
    store_ragged(y_desc, y_off, num_slices, [0, 0], data, ragged_dim)


@triton.jit
def example_load_store_kernel_device_desc_3d(
    x_ptr,
    y_ptr,
    x_off,
    y_off,
    num_slices,
    shape_0,
    shape_1,
    shape_2,
    stride_0,
    stride_1,
    stride_2,
    block_shape_0: tl.constexpr,
    block_shape_1: tl.constexpr,
    block_shape_2: tl.constexpr,
    ragged_dim: tl.constexpr,
):
    x_desc = create_ragged_descriptor_device_3d(
        x_ptr,
        shape_0,
        shape_1,
        shape_2,
        stride_0,
        stride_1,
        stride_2,
        block_shape_0,
        block_shape_1,
        block_shape_2,
        ragged_dim,
    )
    y_desc = create_ragged_descriptor_device_3d(
        y_ptr,
        shape_0,
        shape_1,
        shape_2,
        stride_0,
        stride_1,
        stride_2,
        block_shape_0,
        block_shape_1,
        block_shape_2,
        ragged_dim,
    )

    data = load_ragged(x_desc, x_off, num_slices, [0, 0, 0], ragged_dim)
    store_ragged(y_desc, y_off, num_slices, [0, 0, 0], data, ragged_dim)


def _generate_test_params():
    dtypes = ["float16", "float32"]
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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
@pytest.mark.parametrize("dtype_name,descriptor_mode,ndim,ragged_dim", _generate_test_params())
def test_ragged_tma(dtype_name, descriptor_mode, ndim, ragged_dim):

    torch_dtype = getattr(torch, dtype_name)

    if ndim == 2:
        shape = [128, 80]
        strides = [80, 1]
        block_shape = [32, 128]
    else:  # ndim == 3
        if ragged_dim == 0:
            shape = [64, 32, 32]
            strides = [32 * 32, 32, 1]
            block_shape = [16, 16, 32]
        else:  # ragged_dim == 1
            shape = [64, 32, 32]
            strides = [32 * 32, 32, 1]
            block_shape = [32, 16, 32]

    src = torch.ones(shape, dtype=torch_dtype, device="cuda")
    dst = torch.zeros(shape, dtype=torch_dtype, device="cuda")

    num_slices = min(block_shape[ragged_dim] - 1, shape[ragged_dim] // 3)
    x_off = 0
    y_off = (shape[ragged_dim] - num_slices) // 2

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    if descriptor_mode == "host":
        x_desc = create_ragged_descriptor(src, block_shape, ragged_dim)
        y_desc = create_ragged_descriptor(dst, block_shape, ragged_dim)

        example_load_store_kernel_host_desc[(1, )](
            x_desc,
            y_desc,
            x_off,
            y_off,
            num_slices,
            ragged_dim,
            ndim,
        )
    else:
        if ndim == 2:
            example_load_store_kernel_device_desc_2d[(1, )](
                src,
                dst,
                x_off,
                y_off,
                num_slices,
                shape[0],
                shape[1],
                strides[0],
                strides[1],
                block_shape[0],
                block_shape[1],
                ragged_dim,
            )
        else:  # ndim == 3
            example_load_store_kernel_device_desc_3d[(1, )](
                src,
                dst,
                x_off,
                y_off,
                num_slices,
                shape[0],
                shape[1],
                shape[2],
                strides[0],
                strides[1],
                strides[2],
                block_shape[0],
                block_shape[1],
                block_shape[2],
                ragged_dim,
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

    assert [res0, res1, res2] == [
        True,
        True,
        True,
    ], f"Failed for {ndim}D {descriptor_mode} mode ragged_dim={ragged_dim}: before={res0}, copied={res1}, after={res2}"
