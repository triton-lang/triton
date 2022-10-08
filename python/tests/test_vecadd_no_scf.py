import math

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


def vecadd_tester(num_warps, block_size, shape):
    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               z_ptr,
               n_elements,
               BLOCK_SIZE_N: tl.constexpr):
        pid = tl.program_id(axis=0)

        offset = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
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
    kernel[grid](x_ptr=x, y_ptr=y, z_ptr=z, n_elements=x.shape.numel(), BLOCK_SIZE_N=block_size, num_warps=num_warps)

    golden_z = x + y
    assert_close(z, golden_z, rtol=1e-7, atol=1e-7)


def vecadd_fcmp_tester(num_warps, block_size, shape):
    '''
    vecadd tester with float comparation as load/store mask.
    '''
    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               z_ptr,
               n_elements,
               BLOCK_SIZE_N: tl.constexpr):
        pid = tl.program_id(axis=0)

        offset = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_ptrs = x_ptr + offset
        y_ptrs = y_ptr + offset

        io_mask = offset < n_elements
        x = tl.load(x_ptrs, mask=io_mask)
        y = tl.load(y_ptrs, mask=io_mask)

        z = x + y
        val_mask = (offset < n_elements) and (z > 1.)

        z_ptrs = z_ptr + offset
        tl.store(z_ptrs, z, mask=val_mask)

    x = torch.randn(shape, device='cuda', dtype=torch.float32)
    y = torch.randn(shape, device='cuda', dtype=torch.float32)
    z = torch.zeros(shape, device=x.device, dtype=x.dtype)

    grid = lambda EA: (math.ceil(x.shape.numel() / block_size),)
    kernel[grid](x_ptr=x, y_ptr=y, z_ptr=z, n_elements=x.shape.numel(), BLOCK_SIZE_N=block_size, num_warps=num_warps)

    golden_z = x + y
    for i in range(shape[0]):
        golden_z[i] = golden_z[i] if golden_z[i] > 0. else 0.

    print('z:', z)
    print('goldern_z', golden_z)
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
    vecadd_tester(num_warps, block_size, shape)


@pytest.mark.parametrize('num_warps, block_size, shape', [
    [1, 128, (256 + 1,)],
    [1, 256, (256 + 1,)],
    [2, 256, (3, 256 + 7)],
    [4, 256, (3, 256 + 7)],
])
def test_vecadd__no_scf_masked(num_warps, block_size, shape):
    vecadd_tester(num_warps, block_size, shape)


@pytest.mark.parametrize('num_warps, block_size, shape', [
    [1, 128, (256 + 1,)],
    [1, 256, (256 + 1,)],
    [2, 256, (3, 256 + 7)],
    [4, 256, (3, 256 + 7)],
])
def test_vecadd_fcmp_no_scf_masked(num_warps, block_size, shape):
    vecadd_fcmp_tester(num_warps, block_size, shape)


if __name__ == '__main__':
    test_vecadd_fcmp_no_scf_masked(num_warps=1, block_size=128, shape=(256,))
