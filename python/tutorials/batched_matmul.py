import torch

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    q_ptr,
    k_ptr,
    o_ptr,
    stride_qb,
    stride_qm,
    stride_qk,
    stride_kb,
    stride_kk,
    stride_kn,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    q_ptrs = q_ptr + offs_b[:, None, None] * stride_qb + offs_m[None, :, None] * stride_qm + offs_k[None,
                                                                                                    None, :] * stride_qk
    k_ptrs = k_ptr + offs_b[:, None, None] * stride_kb + offs_k[None, :, None] * stride_kk + offs_n[None,
                                                                                                    None, :] * stride_kn

    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    qk = tl.dot(q, k)

    o_ptrs = o_ptr + offs_b[:, None, None] * stride_ob + offs_m[None, :, None] * stride_om + offs_n[None,
                                                                                                    None, :] * stride_on
    tl.store(o_ptrs, qk)


def matmul(q, k):
    # Check constraints.
    assert q.shape[-1] == k.shape[-2], "Incompatible dimensions"
    assert q.shape[0] == k.shape[0], "Incompatible batch dimensions"
    B, M, K = q.shape
    B, K, N = k.shape
    # Allocates output.
    o = torch.empty((B, M, N), device=q.device, dtype=q.dtype)

    BLOCK_B = B
    BLOCK_M = M
    BLOCK_N = N
    BLOCK_K = K

    grid = (
        triton.cdiv(M, BLOCK_M),
        1,
    )
    matmul_kernel[grid](
        q,
        k,
        o,  #
        q.stride(0),
        q.stride(1),  #
        q.stride(2),
        k.stride(0),
        k.stride(1),  #
        k.stride(2),
        o.stride(0),
        o.stride(1),  #
        o.stride(2),
        BLOCK_B=BLOCK_B,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,  #
        num_warps=B,
    )
    return o


B, M, K, N = 4, 64, 64, 64
q = torch.randn((B, M, K), device='cuda', dtype=torch.float16)
k = torch.randn((B, K, N), device='cuda', dtype=torch.float16)

triton_output = matmul(q, k)
torch_output = torch.matmul(q, k)

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
