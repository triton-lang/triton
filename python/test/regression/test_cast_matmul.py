"""
Mixed precision tests for matmul (tl.dot) with cast (tl.to)

issue: https://github.com/triton-lang/triton/issues/2523

TODO: float8 types
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_hip_mi300, is_cuda, is_hip

input_dtypes = ["float16", "float32", "float64"]
if is_cuda():
    input_dtypes += ["int8", "float8_e5m2"]
    cc = torch.cuda.get_device_capability(0)
    if cc >= (8, 9):
        input_dtypes += ["float8_e4m3fn"]
elif is_hip_mi300():
    input_dtypes += [
        "int8",
        "float8_e5m2",
        # natively supported on mi300 (see CDNA3 ISA, section 7.2)
        "float8_e4m3fnuz",
    ]

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


@pytest.mark.parametrize("M, K, N, BLOCK_K, BLOCK_M, w_dtype, x_dtype, out_dtype",
                         [(M, K, N, BLOCK_K, BLOCK_M, w, x, o)  #
                          for BLOCK_K in [16, 32]  #
                          for BLOCK_M in [16, 64]  #
                          for (M, K, N) in [(128, 128, 128), (768, 768, 1024)]  #
                          for w in input_dtypes
                          for x in input_dtypes  #
                          for o in out_dtypes])
def test_cast_matmul(M, K, N, BLOCK_K, BLOCK_M, w_dtype, x_dtype, out_dtype):
    if x_dtype == w_dtype:
        pytest.skip("skip the same input dtype")
    if is_hip() and BLOCK_M == 64 and w_dtype in ["float8_e5m2", "float8_e4m3fnuz"]:
        pytest.skip("skip due to bug on HIP path")
    device = torch.cuda.current_device()
    x_dtype: torch.dtype = getattr(torch, x_dtype)
    w_dtype: torch.dtype = getattr(torch, w_dtype)

    def init_tensor(dtype, shape):
        if dtype == torch.int8:
            return torch.randint(0, 2, shape, device=device, dtype=dtype)
        elif dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2):
            return torch.randn(shape, device=device, dtype=torch.float16).to(dtype)
        else:
            return torch.randn(shape, device=device, dtype=dtype)

    torch.manual_seed(42)
    a = init_tensor(w_dtype, (M, K))
    b = init_tensor(x_dtype, (K, N))

    torch_dtype = getattr(torch, out_dtype)
    triton_dtype = getattr(tl, out_dtype)  # <- here force dot_out_dtype
    out_torch = torch.matmul(a.to(torch_dtype), b.to(torch_dtype))
    out_triton = torch.empty((M, N), device=device, dtype=torch_dtype)

    # launch kernel
    block_m, block_n, block_k = BLOCK_M, 16, BLOCK_K
    grid = ((triton.cdiv(M, block_m) * triton.cdiv(N, block_n)), 1)

    matmul_kernel[grid](
        a, b, out_triton, M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        out_triton.stride(0), out_triton.stride(1), dot_out_dtype=triton_dtype,  #
        GROUP_M=8,  #
        BLOCK_M=block_m,  #
        BLOCK_N=block_n,  #
        BLOCK_K=block_k)

    torch.testing.assert_close(out_torch, out_triton, atol=0.3, rtol=0.01)


@triton.jit
def matmul_kernel_16bx8b(lhs_ptr,  # (M, K)
                         rhs_ptr,  # (K, N)
                         out_ptr,  # (M, N)
                         # shape information (strides)
                         M, N, K,
                         # block information
                         block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
                         # different test cases
                         dtype_a: tl.constexpr, dtype_b: tl.constexpr, mode: tl.constexpr):
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
    for dtype_16b in [torch.bfloat16]  # loop over different data types of 16-bit
    for dtype_8b in [torch.int8, torch.float8_e5m2, torch.float8_e4m3fn]  # loop over different data types of 8-bit,
    for mode in ['16bx8b', '8bx16b']  # loop over [16-bit x 8 bit, 8 bit x 16 bit] MMA
])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 8, reason="Requires compute capability == 8")
def test_gemm_mixed_dtype_small_tile_16bx8b(M, N, K, block_m, block_n, block_k, dtype_16b, dtype_8b, mode):
    assert M == K
    assert K == N  # let M == K == N for input matrix to simplify the data initialization; Otherwise, we need to tranpose A and B for '8bx16b';
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

    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-2, atol=1e-3, check_dtype=False)
