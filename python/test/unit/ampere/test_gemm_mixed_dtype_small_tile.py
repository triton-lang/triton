import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def matmul_kernel_16bx8b(lhs_ptr,  # (M, K)
                         rhs_ptr,  # (K, N)
                         out_ptr,  # (M, N)
                         # shape information (strides)
                         M, N, K,
                         # block information
                         block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr, dtype_a: tl.constexpr,
                         dtype_b: tl.constexpr):
    start_m = tl.program_id(0)  # start (axis m)
    start_n = tl.program_id(1)  # start (axis n)

    acc = tl.zeros([block_m, block_n], dtype=tl.float32)

    for start_k in range(0, K, block_k):
        lhs_tile = (start_m * block_m + tl.arange(0, block_m))[:, None] * K + \
            (start_k + tl.arange(0, block_k))[None, :]

        rhs_tile = (start_k + tl.arange(0, block_k))[:, None] * N + \
            (start_n * block_n + tl.arange(0, block_n))[None, :]

        lhs_mask = ((start_m * block_m + tl.arange(0, block_m)) < M)[:, None] \
            * ((start_k + tl.arange(0, block_k)) < K)[None, :]

        rhs_mask = ((start_n * block_n + tl.arange(0, block_n)) < N)[None, :] \
            * ((start_k + tl.arange(0, block_k)) < K)[:, None]

        lhs = tl.load(lhs_ptr + lhs_tile, mask=lhs_mask, other=0.0)
        rhs = tl.load(rhs_ptr + rhs_tile, mask=rhs_mask, other=0.0)

        if dtype_a == torch.float16:
            acc = tl.dot(lhs, rhs.to(tl.float16), acc)
        elif dtype_a == torch.bfloat16:
            acc = tl.dot(lhs, rhs.to(tl.bfloat16), acc)

    out_tile = (start_m * block_m + tl.arange(0, block_m))[:, None] * N + \
        start_n * block_n + tl.arange(0, block_n)[None, :]

    mask = ((start_m * block_m + tl.arange(0, block_m)) < M)[:, None] & \
        ((start_n * block_n + tl.arange(0, block_n)) < N)[None, :]

    # print("acc ", acc)

    tl.store(out_ptr + out_tile, acc, mask=mask)


# this are the test cases for mixed dtype GEMM (16-bit and 8-bit) when the tile size is small (e.g., 16)
@pytest.mark.parametrize('M,N,K,block_m,block_n,block_k,dtype_a,dtype_b', [
    (256, 256, 256, block_m, block_n, block_k, dtype_a, dtype_b)
    for block_m in [16, 32, 64, 128]  # block size m
    for block_n in [16, 32, 64, 128]  # block size n
    for block_k in [16, 32]  # block size k
    for dtype_a in [torch.float16, torch.bfloat16]  # loop over different data types of 16-bit
    for dtype_b in [torch.int8, torch.float8_e5m2, torch.float8_e4m3fn]  # loop over different data types of 8-bit
])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 8, reason="Requires compute capability == 8")
def test_gemm_mixed_dtype_small_tile_16bx8b(M, N, K, block_m, block_n, block_k, dtype_a, dtype_b):
    lhs = torch.randn((M, K), dtype=dtype_a, device="cuda")
    if dtype_b == torch.int8:
        rhs = torch.randint(0, 127, (K, N), dtype=dtype_b, device="cuda")
    elif dtype_b == torch.float8_e5m2 or dtype_b == torch.float8_e4m3fn:
        rhs = torch.randn((K, N), dtype=torch.float16, device="cuda")
        rhs.to(dtype_b)
    out = torch.empty((M, N), dtype=dtype_a, device='cuda')

    matmul_kernel_16bx8b[(triton.cdiv(M, block_m), triton.cdiv(N,
                                                               block_n))](lhs, rhs, out, M=M, N=N, K=K, block_m=block_m,
                                                                          block_n=block_n, block_k=block_k,
                                                                          dtype_a=dtype_a, dtype_b=dtype_b)
    expected = torch.mm(lhs, rhs.to(dtype_a))
    assert_close(out.cpu(), expected.cpu(), rtol=1e-2, atol=1e-3, check_dtype=False)


@triton.jit
def matmul_kernel_8bx16b(lhs_ptr,  # (M, K)
                         rhs_ptr,  # (K, N)
                         out_ptr,  # (M, N)
                         # shape information (strides)
                         M, N, K,
                         # block information
                         block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr, dtype_a: tl.constexpr,
                         dtype_b: tl.constexpr):
    start_m = tl.program_id(0)  # start (axis m)
    start_n = tl.program_id(1)  # start (axis n)

    acc = tl.zeros([block_m, block_n], dtype=tl.float32)

    for start_k in range(0, K, block_k):
        lhs_tile = (start_m * block_m + tl.arange(0, block_m))[:, None] * K + \
            (start_k + tl.arange(0, block_k))[None, :]

        rhs_tile = (start_k + tl.arange(0, block_k))[:, None] * N + \
            (start_n * block_n + tl.arange(0, block_n))[None, :]

        lhs_mask = ((start_m * block_m + tl.arange(0, block_m)) < M)[:, None] \
            * ((start_k + tl.arange(0, block_k)) < K)[None, :]

        rhs_mask = ((start_n * block_n + tl.arange(0, block_n)) < N)[None, :] \
            * ((start_k + tl.arange(0, block_k)) < K)[:, None]

        lhs = tl.load(lhs_ptr + lhs_tile, mask=lhs_mask, other=0.0)
        rhs = tl.load(rhs_ptr + rhs_tile, mask=rhs_mask, other=0.0)

        if dtype_b == torch.float16:
            acc = tl.dot(lhs.to(tl.float16), rhs, acc)
        elif dtype_b == torch.bfloat16:
            acc = tl.dot(lhs.to(tl.bfloat16), rhs, acc)

    out_tile = (start_m * block_m + tl.arange(0, block_m))[:, None] * N + \
        start_n * block_n + tl.arange(0, block_n)[None, :]

    mask = ((start_m * block_m + tl.arange(0, block_m)) < M)[:, None] & \
        ((start_n * block_n + tl.arange(0, block_n)) < N)[None, :]

    # print("acc ", acc)

    tl.store(out_ptr + out_tile, acc, mask=mask)


# this are the test cases for mixed dtype GEMM (8-bit and 16-bit) when the tile size is small (e.g., 16)
@pytest.mark.parametrize('M,N,K,block_m,block_n,block_k,dtype_a,dtype_b', [
    (256, 256, 256, block_m, block_n, block_k, dtype_a, dtype_b)
    for block_m in [16, 32, 64, 128]  # block size m
    for block_n in [16, 32, 64, 128]  # block size n
    for block_k in [16, 32]  # block size k
    for dtype_a in [torch.int8, torch.float8_e5m2, torch.float8_e4m3fn]  # loop over different data types of 8-bit
    for dtype_b in [torch.float16, torch.bfloat16]  # loop over different data types of 16-bit
])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 8, reason="Requires compute capability == 8")
def test_gemm_mixed_dtype_small_tile_8bx16b(M, N, K, block_m, block_n, block_k, dtype_a, dtype_b):
    if dtype_a == torch.int8:
        lhs = torch.randint(0, 127, (K, N), dtype=dtype_a, device="cuda")
    elif dtype_a == torch.float8_e5m2 or dtype_a == torch.float8_e4m3fn:
        lhs = torch.randn((K, N), dtype=torch.float16, device="cuda")
        lhs.to(dtype_a)
    rhs = torch.randn((M, K), dtype=dtype_b, device="cuda")
    out = torch.empty((M, N), dtype=dtype_b, device='cuda')

    matmul_kernel_8bx16b[(triton.cdiv(M, block_m), triton.cdiv(N,
                                                               block_n))](lhs, rhs, out, M=M, N=N, K=K, block_m=block_m,
                                                                          block_n=block_n, block_k=block_k,
                                                                          dtype_a=dtype_a, dtype_b=dtype_b)
    expected = torch.mm(lhs.to(dtype_b), rhs)
    assert_close(out.cpu(), expected.cpu(), rtol=1e-2, atol=1e-3, check_dtype=False)
