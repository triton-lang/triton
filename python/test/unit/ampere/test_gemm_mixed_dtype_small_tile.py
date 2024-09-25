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
                         dtype_b: tl.constexpr, mode: tl.constexpr):
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

        if mode == '16bx8b':
            if dtype_a == torch.float16:
                acc = tl.dot(lhs, rhs.to(tl.float16), acc)
            elif dtype_a == torch.bfloat16:
                acc = tl.dot(lhs, rhs.to(tl.bfloat16), acc)
        elif mode == '8bx16b':
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


# These are the test cases for mixed dtype GEMM (16-bit and 8-bit) when the tile size is small (e.g., 16)
@pytest.mark.parametrize('M,N,K,block_m,block_n,block_k,dtype_16b,dtype_8b,mode', [
    (128, 128, 128, block_m, block_n, block_k, dtype_16b, dtype_8b, mode)
    for block_m in [16, 32, 64, 128]
    for block_n in [16, 32, 64, 128]  # block size n
    for block_k in [16, 32]  # block size k
    for dtype_16b in [torch.float16, torch.bfloat16]  # loop over different data types of 16-bit
    for dtype_8b in [torch.int8, torch.float8_e5m2, torch.float8_e4m3fn]  # loop over different data types of 8-bit,
    for mode in ['16bx8b', '8bx16b']  # loop over [16-bit x 8 bit, 8 bit x 16 bit] MMA
])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 8, reason="Requires compute capability == 8")
def test_gemm_mixed_dtype_small_tile_16bx8b(M, N, K, block_m, block_n, block_k, dtype_16b, dtype_8b, mode):
    assert M == K
    assert K == N  # let M == K == N for input matrix to simplify the testing logic, otherwise, we need to tranpose A and B for '8bx16b'
    operand_16b = torch.randn((M, K), dtype=dtype_16b, device="cuda")
    if dtype_8b == torch.int8:
        operand_8b = torch.randint(0, 127, (K, N), dtype=dtype_8b, device="cuda")
    elif dtype_8b == torch.float8_e5m2 or dtype_8b == torch.float8_e4m3fn:
        operand_8b = torch.randn((K, N), dtype=torch.float16, device="cuda")
        operand_8b.to(dtype_8b)

    dtype_A, dtype_B = dtype_16b, dtype_8b
    expected = torch.empty((M, N), dtype=dtype_16b, device='cuda')
    lhs, rhs = operand_16b, operand_8b

    if mode == '8bx16b':
        lhs, rhs = operand_8b, operand_16b
        expected = torch.mm(lhs.to(dtype_16b), rhs)
        block_m, block_n = block_n, block_m
        dtype_A, dtype_B = dtype_8b, dtype_16b
    elif mode == '16bx8b':
        expected = torch.mm(lhs, rhs.to(dtype_16b))

    out = torch.empty((M, N), dtype=dtype_16b, device='cuda')
    matmul_kernel_16bx8b[(triton.cdiv(M, block_m), triton.cdiv(N,
                                                               block_n))](lhs, rhs, out, M=M, N=N, K=K, block_m=block_m,
                                                                          block_n=block_n, block_k=block_k,
                                                                          dtype_a=dtype_A, dtype_b=dtype_B, mode=mode)

    assert_close(out.cpu(), expected.cpu(), rtol=1e-2, atol=1e-3, check_dtype=False)
