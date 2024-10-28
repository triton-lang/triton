"""
Mixed precision tests for matmul (tl.dot) with cast (tl.to)

issue: https://github.com/triton-lang/triton/issues/2523

TODO: float8 types
"""

import pytest
import torch

import triton
import triton.language as tl

input_dtypes = ["float16", "float32", "float64"]
out_dtypes = ["float16", "float32"]


@triton.jit
def matmul_kernel(A, B, C, M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  dot_out_dtype: tl.constexpr,  #
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  #
                  BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    # matrix multiplication
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


@pytest.mark.parametrize("M, K, N, w_dtype, x_dtype, out_dtype",
                         [(M, K, N, w, x, o)  #
                          for (M, K, N) in [(128, 128, 128), (1280, 768, 1024)]  #
                          for w in input_dtypes
                          for x in input_dtypes  #
                          for o in out_dtypes])
def test_cast_matmul(M, K, N, w_dtype, x_dtype, out_dtype):
    if x_dtype == w_dtype:
        pytest.skip("skip the same input dtype")
    device = torch.cuda.current_device()
    x_dtype = getattr(torch, x_dtype)
    w_dtype = getattr(torch, w_dtype)
    a = torch.randn((M, K), device=device, dtype=x_dtype)
    b = torch.randn((K, N), device=device, dtype=w_dtype)
    torch_dtype = getattr(torch, out_dtype)
    triton_dtype = getattr(tl, out_dtype)  # <- here force dot_out_dtype
    out_torch = torch.matmul(a.to(torch_dtype), b.to(torch_dtype))
    out_triton = torch.empty((M, N), device=device, dtype=torch_dtype)

    # launch kernel
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 32
    grid = ((triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), 1)

    matmul_kernel[grid](
        a, b, out_triton, M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        out_triton.stride(0), out_triton.stride(1), dot_out_dtype=triton_dtype,  #
        GROUP_M=8,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        BLOCK_K=BLOCK_K)

    torch.testing.assert_close(out_torch, out_triton, atol=0.3, rtol=0.01)
