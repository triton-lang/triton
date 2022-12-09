import math
import random

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@pytest.mark.parametrize('num_warps, block_size, iter_size', [
    [4, 256, 1],
    [4, 1024, 256],
])
def test_vecadd_scf_no_mask(num_warps, block_size, iter_size):

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
            z = x + y
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

    golden_z = x + y
    assert_close(z, golden_z, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('shape, num_warps, block_size, iter_size', [
    [(127, 3), 2, 128, 1],
    [(127, 3), 2, 128, 32],
])
def test_vecadd_scf_mask(shape, num_warps, block_size, iter_size):
    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               z_ptr,
               num_elements,
               block_size: tl.constexpr,
               iter_size: tl.constexpr
               ):
        '''
        @block_size: size of a block
        @iter_size: size of the iteration, a block has multiple iterations
        @num_elements: number of elements
        '''
        pid = tl.program_id(axis=0)
        for i in range(tl.cdiv(block_size, iter_size)):
            # TODO: a bug here, if put the offset outside the forloop, there will be a GPU mis-aligned error.
            offset = pid * block_size + tl.arange(0, iter_size)
            x_ptrs = x_ptr + offset
            y_ptrs = y_ptr + offset

            x = tl.load(x_ptrs, mask=offset < num_elements)
            y = tl.load(y_ptrs, mask=offset < num_elements)
            z = x + y
            z_ptrs = z_ptr + offset
            tl.store(z_ptrs, z, mask=offset < num_elements)

            x_ptr += iter_size
            y_ptr += iter_size
            z_ptr += iter_size

    x = torch.randn(shape, device='cuda', dtype=torch.float32)
    y = torch.randn(shape, device='cuda', dtype=torch.float32)
    z = torch.empty(shape, device=x.device, dtype=x.dtype)

    grid = lambda EA: (math.ceil(x.numel() / block_size),)
    kernel[grid](x_ptr=x, y_ptr=y, z_ptr=z,
                 block_size=x.shape[0], iter_size=iter_size, num_warps=num_warps,
                 num_elements=x.numel())

    golden_z = x + y
    assert_close(z, golden_z, rtol=1e-7, atol=1e-7)


def vecadd_no_scf_tester(num_warps, block_size, shape):
    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               z_ptr,
               n_elements,
               block_size_N: tl.constexpr):
        pid = tl.program_id(axis=0)

        offset = pid * block_size_N + tl.arange(0, block_size_N)
        x_ptrs = x_ptr + offset
        y_ptrs = y_ptr + offset

        mask = offset < n_elements

        x = tl.load(x_ptrs, mask=mask)
        y = tl.load(y_ptrs, mask=mask)
        z = x + y
        z_ptrs = z_ptr + offset
        tl.store(z_ptrs, z, mask=mask)

    x = torch.randn(shape, device='cuda', dtype=torch.float32)
    y = torch.randn(shape, device='cuda', dtype=torch.float32)
    z = torch.empty(shape, device=x.device, dtype=x.dtype)

    grid = lambda EA: (math.ceil(x.shape.numel() / block_size),)
    kernel[grid](x_ptr=x, y_ptr=y, z_ptr=z, n_elements=x.shape.numel(), block_size_N=block_size, num_warps=num_warps)

    golden_z = x + y
    assert_close(z, golden_z, rtol=1e-7, atol=1e-7)


def vecadd_fcmp_no_scf_tester(num_warps, block_size, shape):
    '''
    vecadd tester with float comparison as load/store mask.
    '''
    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               z_ptr,
               n_elements,
               block_size_N: tl.constexpr):
        pid = tl.program_id(axis=0)

        offset = pid * block_size_N + tl.arange(0, block_size_N)
        x_ptrs = x_ptr + offset
        y_ptrs = y_ptr + offset

        io_mask = offset < n_elements
        x = tl.load(x_ptrs, mask=io_mask)
        y = tl.load(y_ptrs, mask=io_mask)

        z = x + y
        val_mask = offset < n_elements and (z < 0. or z > 1.)

        z_ptrs = z_ptr + offset
        tl.store(z_ptrs, z, mask=val_mask)

    x = torch.randn(shape, device='cuda', dtype=torch.float32)
    y = torch.randn(shape, device='cuda', dtype=torch.float32)
    z = torch.zeros(shape, device=x.device, dtype=x.dtype)

    grid = lambda EA: (math.ceil(x.shape.numel() / block_size),)
    kernel[grid](x_ptr=x, y_ptr=y, z_ptr=z, n_elements=x.shape.numel(), block_size_N=block_size, num_warps=num_warps)

    golden_z: torch.Tensor = x + y
    gz_data = torch.flatten(golden_z)
    for i in range(golden_z.numel()):
        gz_data[i] = gz_data[i] if gz_data[i] < 0. or gz_data[i] > 1. else 0.

    assert_close(z, golden_z, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('num_warps, block_size, shape', [
    [4, 256, (256,)],
    [2, 256, (256,)],
    [1, 256, (256,)],
    [4, 16, (256,)],
    [2, 64, (256,)],
    [1, 128, (256,)],
])
def test_vecadd_no_scf(num_warps, block_size, shape):
    vecadd_no_scf_tester(num_warps, block_size, shape)


@pytest.mark.parametrize('num_warps, block_size, shape', [
    [1, 128, (256 + 1,)],
    [1, 256, (256 + 1,)],
    [2, 256, (3, 256 + 7)],
    [4, 256, (3, 256 + 7)],
])
def test_vecadd_no_scf_masked(num_warps, block_size, shape):
    vecadd_no_scf_tester(num_warps, block_size, shape)


def test_vecadd_no_scf_masked_randomly():
    random.seed(0)  # fix seed to make random test reproducible
    for i in range(10):
        num_elements = random.randint(128, 2048)
        shape = (num_elements,)
        max_warps = num_elements // 32  # floor div
        for num_warps in range(1, max_warps):
            is_power2 = num_warps & (num_warps - 1) == 0 and num_warps != 0
            if not is_power2: continue
            block_size = min(32, num_warps * 32)
            vecadd_no_scf_tester(num_warps, block_size, shape)


@pytest.mark.parametrize('num_warps, block_size, shape', [
    [1, 128, (256 + 1,)],
    [1, 256, (256 + 1,)],
    [2, 256, (3, 256 + 7)],
    [4, 256, (3, 256 + 7)],
])
def test_vecadd_fcmp_no_scf_masked(num_warps, block_size, shape):
    vecadd_fcmp_no_scf_tester(num_warps, block_size, shape)
