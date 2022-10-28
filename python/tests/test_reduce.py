import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}


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
    for dtype in ['float16', 'float32', 'float64']
    for shape in [4, 8, 16, 32, 64, 128, 512, 1024]
]


@pytest.mark.parametrize('op, dtype, shape', reduce1d_configs)
def test_reduce1d(op, dtype, shape):
    dtype = dtype_mapping[dtype]
    x = torch.randn((shape,), device='cuda', dtype=dtype)
    z = torch.empty(
        tuple(),
        device=x.device,
        dtype=dtype,
    )

    kernel = patch_kernel(reduce1d_kernel, {'OP': op})
    grid = (1,)
    kernel[grid](x_ptr=x, z_ptr=z, block=shape)

    if op == 'sum':
        golden_z = torch.sum(x, dtype=dtype)
    elif op == 'min':
        golden_z = torch.min(x)
    else:
        golden_z = torch.max(x)

    if op == 'sum':
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
    for dtype in ['float16', 'float32', 'float64']
    for shape in [(1, 4), (1, 8), (1, 16), (1, 32), (2, 32), (4, 32), (4, 128), (32, 64)]
    for axis in [0, 1]
]


@pytest.mark.parametrize('op, dtype, shape, axis', reduce2d_configs)
def test_reduce2d(op, dtype, shape, axis):
    dtype = dtype_mapping[dtype]
    x = torch.randn(shape, device='cuda', dtype=dtype)
    reduced_shape = (shape[1 - axis],)
    z = torch.empty(reduced_shape, device=x.device, dtype=dtype)

    kernel = patch_kernel(reduce2d_kernel, {'OP': op})
    grid = (1,)
    kernel[grid](x_ptr=x, z_ptr=z, axis=axis, block_m=shape[0], block_n=shape[1])

    if op == 'sum':
        golden_z = torch.sum(x, dim=axis, keepdim=False, dtype=dtype)
    elif op == 'min':
        golden_z = torch.min(x, dim=axis, keepdim=False)[0]
    else:
        golden_z = torch.max(x, dim=axis, keepdim=False)[0]

    if op == 'sum':
        if shape[axis] >= 256:
            assert_close(z, golden_z, rtol=0.05, atol=0.1)
        elif shape[axis] >= 32:
            assert_close(z, golden_z, rtol=0.05, atol=0.02)
        else:
            assert_close(z, golden_z, rtol=0.01, atol=0.01)
    else:
        assert_close(z, golden_z, rtol=0.001, atol=0.001)
