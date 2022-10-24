import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def matmul_no_scf_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    c = tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)

# TODO: num_warps could only be 4 for now


@pytest.mark.parametrize('SIZE_M,SIZE_N,SIZE_K,NUM_WARPS', [
    [128, 256, 32, 4],
    [256, 128, 16, 4],
    [128, 16, 32, 4],
    [32, 128, 64, 4],
])
def test_gemm_no_scf(SIZE_M, SIZE_N, SIZE_K, NUM_WARPS):
    a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)
    b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)
    c = torch.empty((SIZE_M, SIZE_N), device=a.device, dtype=torch.float32)
    grid = lambda META: (1, )
    matmul_no_scf_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c,
                               stride_am=a.stride(0), stride_ak=a.stride(1),
                               stride_bk=b.stride(0), stride_bn=b.stride(1),
                               stride_cm=c.stride(0), stride_cn=c.stride(1),
                               M=SIZE_M, N=SIZE_N, K=SIZE_K,
                               num_warps=NUM_WARPS)
    golden = torch.matmul(a, b)
    torch.set_printoptions(profile="full")
    assert_close(c, golden, rtol=1e-3, atol=1e-3, check_dtype=False)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)

# TODO: DotConversion in TritonGPUToLLVM cannot support non-splat C for the moment
# @pytest.mark.parametrize('SIZE_M,SIZE_N,SIZE_K,NUM_WARPS,BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K', [
#    [128, 256, 128, 4, 128, 256, 32],
#    # [256, 128, 64, 4, 256, 128, 16],
#    # [128, 16, 128, 4, 128, 16, 32],
#    # [32, 128, 256, 4, 32, 128, 64],
# ])
# def test_gemm(SIZE_M, SIZE_N, SIZE_K, NUM_WARPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
#    a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)
#    b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)
#    c = torch.empty((SIZE_M, SIZE_N), device=a.device, dtype=torch.float32)
#    grid = lambda META: (1, )
#    matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c,
#                        stride_am=a.stride(0), stride_ak=a.stride(1),
#                        stride_bk=b.stride(0), stride_bn=b.stride(1),
#                        stride_cm=c.stride(0), stride_cn=c.stride(1),
#                        M=a.shape[0], N=b.shape[1], K=a.shape[1],
#                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
#                        num_warps=NUM_WARPS)
#    golden = torch.matmul(a, b)
#    torch.set_printoptions(profile="full")
#    assert_close(c, golden, rtol=1e-3, atol=1e-3, check_dtype=False)
