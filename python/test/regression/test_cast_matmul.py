"""
issue: https://github.com/openai/triton/issues/2523
fused type convert and matmul, base on triton matmul, the different with matmul:
1. force C's dtype=dot_out_dtype to ["float16"]
2. accept A and B with dtype=[torch.float32, torch.float16]

"""
from itertools import product

import pytest
import torch
import torch.nn.functional as F

import triton.language as tl
from triton import cdiv, jit

input_dtypes = [torch.float32, torch.float16,]
out_dtypes = ["float16"]
if torch.cuda.is_bf16_supported():
    input_dtypes.append(torch.bfloat16)
    # out_dtypes.append("bfloat16") # TODO: maybe tl.dot can support bfloat16
# product w and x dtype: 3x3 = 9 cases


@pytest.mark.parametrize(
    "w_dtype,x_dtype,out_dtype",
    product(input_dtypes, input_dtypes, out_dtypes)
)
def test_cast_matmul(w_dtype, x_dtype, out_dtype):

    batch, seq, hidden = 20, 64, 768
    output = 1024
    device = torch.cuda.current_device()
    a = torch.randn(batch * seq, hidden, device=device,
                    dtype=x_dtype,
                    requires_grad=True)
    w = torch.randn(output, hidden, device=device,
                    dtype=w_dtype,
                    requires_grad=True)
    torch_dtype = getattr(torch, out_dtype)
    out_torch = F.linear(a.to(torch_dtype), w.to(torch_dtype))
    b = w.T
    c_dtype = getattr(torch, out_dtype)
    dot_out_dtype = getattr(tl, out_dtype)  # <- here force dot_out_dtype
    M, K = a.shape
    _, N = b.shape

    out_triton = torch.empty((M, N), device=device, dtype=c_dtype)

    ab_dtype = True
    allow_tf32 = True
    fp8_fast_accum = True
    # launch kernel
    grid = lambda META: (cdiv(M, META['BLOCK_M']) * cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K = 16, 32, 16, 1
    EVEN_K = K % (BLOCK_K * SPLIT_K) == 0

    @jit
    def matmul_kernel(A, B, C, M, N, K,
                      stride_am, stride_ak,
                      stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      dot_out_dtype: tl.constexpr,
                      allow_tf32: tl.constexpr,
                      fp8_fast_accum: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, AB_DTYPE: tl.constexpr
                      ):
        # matrix multiplication
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
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
        rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
        # pointers
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                k_remaining = K - k * (BLOCK_K * SPLIT_K)
                _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
                a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
                b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
            if AB_DTYPE:
                a = a.to(C.dtype.element_ty)
                b = b.to(C.dtype.element_ty)
            if fp8_fast_accum:
                acc = tl.dot(a, b, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
            else:
                acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
        acc = acc.to(C.dtype.element_ty)
        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        # handles write-back with reduction-splitting
        if SPLIT_K == 1:
            tl.store(C, acc, mask=mask)
        else:
            tl.atomic_add(C, acc, mask=mask)
    matmul_kernel[grid](a, b, out_triton, M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        out_triton.stride(0), out_triton.stride(1),
                        dot_out_dtype=dot_out_dtype,
                        allow_tf32=allow_tf32,
                        fp8_fast_accum=fp8_fast_accum,
                        GROUP_M=8, AB_DTYPE=ab_dtype,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_K=BLOCK_K,
                        SPLIT_K=SPLIT_K,
                        EVEN_K=EVEN_K,
                        )

    torch.testing.assert_close(out_torch, out_triton, atol=0.02, rtol=0.01)
