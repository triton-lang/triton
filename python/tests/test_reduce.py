import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8']  # PyTorch does not support uint16/uint32/uint64
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes
dtypes_with_bfloat16 = int_dtypes + uint_dtypes + float_dtypes
dtype_mapping = {dtype_str: torch.__dict__[dtype_str] for dtype_str in dtypes}


def get_reduced_dtype(dtype):
    if dtype in [torch.int8, torch.int16, torch.uint8]:
        return torch.int32
    if dtype in [torch.bfloat16]:
        return torch.float32
    return dtype


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel


@triton.jit
def reduce1d_kernel(x_ptr, z_ptr, block: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, block))
    tl.store(z_ptr, tl.OP(x, axis=0))


@triton.jit
def reduce2d_kernel(x_ptr, z_ptr, axis: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr):
    range_m = tl.arange(0, block_m)
    range_n = tl.arange(0, block_n)
    x = tl.load(x_ptr + range_m[:, None] * block_n + range_n[None, :])
    z = tl.OP(x, axis=axis)
    if axis == 0:
        tl.store(z_ptr + range_n, z)
    else:
        tl.store(z_ptr + range_m, z)


reduce1d_configs = [
    (op, dtype, shape)
    for op in ['sum', 'min', 'max']
    for dtype in dtypes
    for shape in [4, 8, 16, 32, 64, 128, 512, 1024]
]


@pytest.mark.parametrize('op, dtype, shape', reduce1d_configs)
def test_reduce1d(op, dtype, shape):
    dtype = dtype_mapping[dtype]
    reduced_dtype = get_reduced_dtype(dtype)

    if dtype.is_floating_point:
        x = torch.randn((shape,), device='cuda', dtype=dtype)
    elif dtype is torch.uint8:
        x = torch.randint(0, 20, (shape,), device='cuda', dtype=dtype)
    else:
        x = torch.randint(-20, 20, (shape,), device='cuda', dtype=dtype)
    z = torch.empty(
        tuple(),
        device=x.device,
        dtype=reduced_dtype,
    )

    kernel = patch_kernel(reduce1d_kernel, {'OP': op})
    grid = (1,)
    kernel[grid](x_ptr=x, z_ptr=z, block=shape)

    if op == 'sum':
        golden_z = torch.sum(x, dtype=reduced_dtype)
    elif op == 'min':
        golden_z = torch.min(x).to(reduced_dtype)
    else:
        golden_z = torch.max(x).to(reduced_dtype)

    if dtype.is_floating_point and op == 'sum':
        if shape >= 256:
            assert_close(z, golden_z, rtol=0.05, atol=0.1)
        elif shape >= 32:
            assert_close(z, golden_z, rtol=0.05, atol=0.02)
        else:
            assert_close(z, golden_z, rtol=0.01, atol=0.01)
    else:
        assert_close(z, golden_z, rtol=0.001, atol=0.001)


reduce2d_configs = [
    (op, dtype, shape, axis)
    for op in ['sum', 'min', 'max']
    for dtype in dtypes
    for shape in [(1, 4), (1, 8), (1, 16), (1, 32), (2, 32), (4, 32), (4, 128), (32, 64)]
    for axis in [0, 1]
]


@pytest.mark.parametrize('op, dtype, shape, axis', reduce2d_configs)
def test_reduce2d(op, dtype, shape, axis):
    dtype = dtype_mapping[dtype]
    reduced_dtype = get_reduced_dtype(dtype)
    reduced_shape = (shape[1 - axis],)

    if dtype.is_floating_point:
        x = torch.randn(shape, device='cuda', dtype=dtype)
    elif dtype is torch.uint8:
        x = torch.randint(0, 20, shape, device='cuda', dtype=dtype)
    else:
        x = torch.randint(-20, 20, shape, device='cuda', dtype=dtype)
    z = torch.empty(reduced_shape, device=x.device, dtype=reduced_dtype)

    kernel = patch_kernel(reduce2d_kernel, {'OP': op})
    kernel[(1,)](x_ptr=x, z_ptr=z, axis=axis, block_m=shape[0], block_n=shape[1])

    if op == 'sum':
        golden_z = torch.sum(x, dim=axis, keepdim=False, dtype=reduced_dtype)
    elif op == 'min':
        golden_z = torch.min(x, dim=axis, keepdim=False)[0].to(reduced_dtype)
    else:
        golden_z = torch.max(x, dim=axis, keepdim=False)[0].to(reduced_dtype)
    if dtype.is_floating_point and op == 'sum':
        if shape[axis] >= 256:
            assert_close(z, golden_z, rtol=0.05, atol=0.1)
        elif shape[axis] >= 32:
            assert_close(z, golden_z, rtol=0.05, atol=0.02)
        else:
            assert_close(z, golden_z, rtol=0.01, atol=0.01)
    else:
        assert_close(z, golden_z, rtol=0.001, atol=0.001)
