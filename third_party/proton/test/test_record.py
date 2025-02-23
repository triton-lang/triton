import torch
import pathlib

import triton
import triton.language as tl
import triton.profiler.language as pl
from typing import Optional



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
    pl.enter_scope("load0")
    y = tl.load(y_ptr + offsets, mask=mask)
    pl.exit_scope("load0")
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert size == 1024
    assert align == 0
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device="cuda")

triton.set_allocator(alloc_fn)
torch.manual_seed(0)
size = 256
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output = torch.empty_like(x)
n_elements = output.numel()
grid = (1, 1, 1)

pgm = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
ttgir = pgm.asm['ttgir']
#    print(ttgir)

