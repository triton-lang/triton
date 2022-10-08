import pytest
import torch
from torch.testing import assert_allclose

import triton
import triton.language as tl


@pytest.mark.parametrize('NUM_WARPS, BLOCK_SIZE', [
    [4, 256],
    [2, 256],
    [1, 256],
])
def test_vecadd_no_mask(NUM_WARPS, BLOCK_SIZE):

    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               z_ptr,
               BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_ptrs = x_ptr + offset
        y_ptrs = y_ptr + offset
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        z = x + y
        z_ptrs = z_ptr + offset
        tl.store(z_ptrs, z)

    x = torch.randn((BLOCK_SIZE,), device='cuda', dtype=torch.float32)
    y = torch.randn((BLOCK_SIZE,), device='cuda', dtype=torch.float32)
    z = torch.empty((BLOCK_SIZE,), device=x.device, dtype=x.dtype)

    grid = lambda EA: (x.shape.numel() // BLOCK_SIZE,)
    kernel[grid](x_ptr=x, y_ptr=y, z_ptr=z, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)

    golden_z = x + y
    assert_allclose(z, golden_z, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('NUM_WARPS, BLOCK_SIZE, ITER_SIZE', [
    [4, 256, 1],
    [4, 1024, 256],
])
def test_vecadd_scf_no_mask(NUM_WARPS, BLOCK_SIZE, ITER_SIZE):

    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               z_ptr,
               BLOCK_SIZE,
               ITER_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        for i in range(0, BLOCK_SIZE, ITER_SIZE):
            offset = pid * BLOCK_SIZE + tl.arange(0, ITER_SIZE)
            x_ptrs = x_ptr + offset
            y_ptrs = y_ptr + offset
            x = tl.load(x_ptrs)
            y = tl.load(y_ptrs)
            z = x + y
            z_ptrs = z_ptr + offset
            tl.store(z_ptrs, z)
            x_ptr += ITER_SIZE
            y_ptr += ITER_SIZE
            z_ptr += ITER_SIZE

    x = torch.randn((BLOCK_SIZE,), device='cuda', dtype=torch.float32)
    y = torch.randn((BLOCK_SIZE,), device='cuda', dtype=torch.float32)
    z = torch.empty((BLOCK_SIZE,), device=x.device, dtype=x.dtype)

    grid = lambda EA: (x.shape.numel() // (BLOCK_SIZE),)
    kernel[grid](x_ptr=x, y_ptr=y, z_ptr=z,
                 BLOCK_SIZE=x.shape[0], ITER_SIZE=ITER_SIZE, num_warps=NUM_WARPS)

    golden_z = x + y
    assert_allclose(z, golden_z, rtol=1e-7, atol=1e-7)

# TODO: test_vecadd with mask
