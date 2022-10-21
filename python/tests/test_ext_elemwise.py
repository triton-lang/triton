
import torch
import triton
import triton.language as tl
import pytest
from torch.testing import assert_close

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