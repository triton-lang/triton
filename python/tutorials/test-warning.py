import torch

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr  #
):
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                    block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)
    c = tl.dot(a, b)
    tl.store(c_block_ptr, c)


def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    # 1D launch kernel where each block gets its own program.
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_M, BLOCK_N, 128)
    return c


torch.manual_seed(0)
a = torch.randn((128, 128), device='cuda', dtype=torch.float16)
b = torch.randn((128, 128), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
