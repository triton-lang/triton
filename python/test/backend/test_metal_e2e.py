
import pytest
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def test_add():
    size = 1024
    x = torch.rand(size, device='cpu')
    y = torch.rand(size, device='cpu')
    output = torch.empty(size, device='cpu')
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)
    
    assert torch.allclose(output, x + y)
