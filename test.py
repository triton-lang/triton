import torch
import triton
from triton import language as tl
from triton.tools.experimental_descriptor import create_2d_tma_descriptor
from triton import cdiv

BLOCK_M: tl.constexpr = 128
BLOCK_N: tl.constexpr = 128


@triton.jit
def test_kernel(desc):
    off_n = tl.program_id(0) * BLOCK_N
    off_m = tl.program_id(1) * BLOCK_M
    tile = tl._experimental_descriptor_load(desc, [off_m, off_n], [BLOCK_M, BLOCK_N], tl.float32)
    tile += 1.0
    tl._experimental_descriptor_store(desc, tile, [off_m, off_n])


M = 256
N = 512
tensor = torch.zeros((M, N), device='cuda', dtype=torch.float32)
tma_desc = create_2d_tma_descriptor(tensor.data_ptr(), M, N, BLOCK_M, BLOCK_N, tensor.element_size())

val = torch.clone(tensor) + 1.0
test_kernel[(cdiv(N, BLOCK_N), cdiv(M, BLOCK_M))](tma_desc, num_warps=1)
assert torch.allclose(val, tensor)

print("byval tma desc passed!")
