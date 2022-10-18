
import torch
import triton
import triton.language as tl
from torch.testing import assert_close


@triton.jit
def math_kernel(x1_ptr, y1_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid + tl.arange(0, 2)
    x1 = tl.load(x1_ptr + offset )
    y1 = tl.libdevice.sin(x1)
    tl.store(y1_ptr + offset, y1)
    

def math_ops(x1: torch.Tensor, y1: torch.Tensor):
    n_elements = x1.numel()
    grid = (n_elements,)
    math_kernel[grid](x1, y1, n_elements, BLOCK_SIZE=1)
    return

torch.manual_seed(0)
size = 32
x1 = torch.rand(size, device='cuda')
y1 = torch.zeros(size, device='cuda')
y2 = torch.sin(x1)
math_ops(x1, y1)
assert_close(y1, y2)
