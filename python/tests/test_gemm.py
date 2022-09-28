import pytest
import torch
from torch.testing import assert_allclose

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
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
def test_gemm_impl(SIZE_M, SIZE_N, SIZE_K, NUM_WARPS):
    a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)
    b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)
    c = torch.empty((SIZE_M, SIZE_N), device=a.device, dtype=torch.float32)
    grid = lambda META: (1, )
    matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c,
                        stride_am=a.stride(0), stride_ak=a.stride(1),
                        stride_bk=b.stride(0), stride_bn=b.stride(1),
                        stride_cm=c.stride(0), stride_cn=c.stride(1),
                        M=SIZE_M, N=SIZE_N, K=SIZE_K,
                        num_warps=NUM_WARPS)
    golden = torch.matmul(a, b)
    torch.set_printoptions(profile="full")
    assert_allclose(c, golden, rtol=1e-3, atol=1e-3)
