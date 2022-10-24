
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@pytest.mark.parametrize('num_warps, block_size, iter_size', [
    [4, 256, 1],
    [4, 1024, 256],
])
def test_sin_no_mask(num_warps, block_size, iter_size):
    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               block_size,
               iter_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        for i in range(0, block_size, iter_size):
            offset = pid * block_size + tl.arange(0, iter_size)
            x_ptrs = x_ptr + offset
            x = tl.load(x_ptrs)
            y = tl.libdevice.sin(x)
            y_ptrs = y_ptr + offset
            tl.store(y_ptrs, y)

            x_ptr += iter_size
            y_ptr += iter_size

    x = torch.randn((block_size,), device='cuda', dtype=torch.float32)
    y = torch.empty((block_size,), device=x.device, dtype=x.dtype)

    grid = lambda EA: (x.shape.numel() // (block_size),)
    kernel[grid](x_ptr=x, y_ptr=y,
                 block_size=x.shape[0], iter_size=iter_size, num_warps=num_warps)

    golden_y = torch.sin(x)
    assert_close(y, golden_y, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('num_warps, block_size, iter_size', [
    [4, 256, 1],
    [4, 1024, 256],
])
def test_fmin_no_mask(num_warps, block_size, iter_size):
    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               z_ptr,
               block_size,
               iter_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        for i in range(0, block_size, iter_size):
            offset = pid * block_size + tl.arange(0, iter_size)
            x_ptrs = x_ptr + offset
            y_ptrs = y_ptr + offset

            x = tl.load(x_ptrs)
            y = tl.load(y_ptrs)
            z = tl.libdevice.min(x, y)
            z_ptrs = z_ptr + offset
            tl.store(z_ptrs, z)

            x_ptr += iter_size
            y_ptr += iter_size
            z_ptr += iter_size

    x = torch.randn((block_size,), device='cuda', dtype=torch.float32)
    y = torch.randn((block_size,), device='cuda', dtype=torch.float32)
    z = torch.empty((block_size,), device=x.device, dtype=x.dtype)

    grid = lambda EA: (x.shape.numel() // (block_size),)
    kernel[grid](x_ptr=x, y_ptr=y, z_ptr=z,
                 block_size=x.shape[0], iter_size=iter_size, num_warps=num_warps)

    golden_z = torch.minimum(x, y)
    assert_close(z, golden_z, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('num_warps, block_size, iter_size', [
    [4, 256, 1],
    [4, 1024, 256],
])
def test_fmad_rn_no_mask(num_warps, block_size, iter_size):
    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               z_ptr,
               w_ptr,
               block_size,
               iter_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        for i in range(0, block_size, iter_size):
            offset = pid * block_size + tl.arange(0, iter_size)
            x_ptrs = x_ptr + offset
            y_ptrs = y_ptr + offset
            z_ptrs = z_ptr + offset

            x = tl.load(x_ptrs)
            y = tl.load(y_ptrs)
            z = tl.load(z_ptrs)

            w = tl.libdevice.fma_rn(x, y, z)
            w_ptrs = w_ptr + offset
            tl.store(w_ptrs, w)

            x_ptr += iter_size
            y_ptr += iter_size
            z_ptr += iter_size
            w_ptr += iter_size

    x = torch.randn((block_size,), device='cuda', dtype=torch.float64)
    y = torch.randn((block_size,), device='cuda', dtype=torch.float64)
    z = torch.randn((block_size,), device='cuda', dtype=torch.float64)
    w = torch.empty((block_size,), device=x.device, dtype=x.dtype)

    grid = lambda EA: (x.shape.numel() // (block_size),)
    kernel[grid](x_ptr=x, y_ptr=y, z_ptr=z, w_ptr=w,
                 block_size=x.shape[0], iter_size=iter_size, num_warps=num_warps)

    golden_w = x * y + z
    assert_close(w, golden_w, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("dtype_str, expr, lib_path",
                         [('float32', 'libdevice.pow', '/usr/local/cuda/nvvm/libdevice/libdevice.10.bc'),
                          ('int32', 'libdevice.ffs', ''),
                          ('float64', 'libdevice.norm4d', '')])
def test_libdevice(dtype_str, expr, lib_path):
    def patch_kernel(template, to_replace):
        kernel = triton.JITFunction(template.fn)
        for key, value in to_replace.items():
            kernel.src = kernel.src.replace(key, value)
        return kernel

    torch_type = {
        "int32": torch.int32,
        "float32": torch.float32,
        "float64": torch.float64
    }

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)

    shape = (128, )
    # limit the range of integers so that the sum does not overflow
    x = None
    if dtype_str == "int32":
        x = torch.randint(2**31 - 1, shape, dtype=torch_type[dtype_str], device="cuda")
    else:
        x = torch.randn(shape, dtype=torch_type[dtype_str], device="cuda")
    if expr == 'libdevice.ffs':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.ffs(x)'})
        y_ref = torch.zeros(shape, dtype=x.dtype, device="cuda")
        for i in range(shape[0]):
            y_ref[i] = (int(x[i]) & int(-x[i])).bit_length()
    elif expr == 'libdevice.pow':
        # numpy does not allow negative factors in power, so we use abs()
        x = torch.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.pow(x, x)'})
        y_ref = torch.pow(x, x)
    elif expr == 'libdevice.norm4d':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.norm4d(x, x, x, x)'})
        y_ref = torch.sqrt(4 * torch.pow(x, 2))

    # triton result
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    kernel[(1,)](x, y, BLOCK=shape[0], extern_libs={"libdevice": lib_path})
    # compare
    assert_close(y, y_ref)
