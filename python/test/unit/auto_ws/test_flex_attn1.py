import pytest
import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor

# PyTorch reference implementation with sliding window and causal masking
def attention_sliding_window_pytorch(Q, K, V, window_size):
    B, H, M, D = Q.shape
    scores = torch.matmul(Q, K.transpose(2, 3)) / (D ** 0.5)
    q_idx = torch.arange(M, device=Q.device)[None, None, :, None]
    k_idx = torch.arange(M, device=Q.device)[None, None, None, :]
    causal_mask = q_idx >= k_idx
    window_mask = (q_idx - k_idx) <= window_size
    mask = causal_mask & window_mask
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output

# Triton kernel with combined causal and sliding window masking
@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_desc_ptr,
    k_desc_ptr,
    v_desc_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, D, window_size,
    scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_b = pid_b
    offs_h = pid_h
    offs_bh = offs_b * H + offs_h
    qvk_offset = offs_b.to(tl.int64) * stride_qb + offs_h.to(tl.int64) * stride_qh

    # FIXME: Broken in rebase
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [q_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [k_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [v_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )

    q = q_desc_ptr.load(
        [offs_bh * M + pid_m * BLOCK_SIZE_M, 0]
    )
    
    m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, D_HEAD), dtype=tl.float32)
    
    start_n = tl.maximum(0, (pid_m * BLOCK_SIZE_M) - window_size)
    end_n = (pid_m + 1) * BLOCK_SIZE_M
    start_block = start_n // BLOCK_SIZE_N
    end_block = tl.cdiv(end_n, BLOCK_SIZE_N)

    offs_kv = offs_bh * M + start_block * BLOCK_SIZE_N
    mask_q = offs_m < M
    
    for pid_n in range(start_block, end_block):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_k = offs_n < M

        k = k_desc_ptr.load(
            [offs_kv, 0],
        )

        scores = tl.dot(q, tl.trans(k)) * scale
        
        q_idx = offs_m[:, None]
        k_idx = offs_n[None, :]
        causal_mask = q_idx >= k_idx
        window_mask = (q_idx - k_idx) <= window_size
        mask = causal_mask & window_mask
        mask = mask & mask_q[:, None] & mask_k[None, :]
        scores = tl.where(mask, scores, float('-inf'))
        
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(scores - m_new[:, None])
        
        l_ij = tl.sum(beta, axis=1)
        l_new = alpha * l_i + l_ij
        
        v = v_desc_ptr.load(
            [offs_kv, 0],
        )

        beta = beta.to(v.dtype)
        pv = tl.dot(beta, v)
        
        acc = acc * alpha[:, None] + pv
        m_i = m_new
        l_i = l_new
        offs_kv += BLOCK_SIZE_N
    
    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + offs_b * stride_ob + offs_h * stride_oh + offs_m[:, None] * stride_om + tl.arange(0, D_HEAD)[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=mask_q[:, None])

# Cached fwd TMA descriptors
cached_desc_q = None
cached_desc_k = None
cached_desc_v = None
cached_args = None

def get_fwd_tma_descriptors(*args):
    global \
        cached_desc_q, \
        cached_desc_k, \
        cached_desc_v, \
        cached_args
    if args != cached_args:
        cached_args = args
        # reuse cached tma descriptors if input matches
        (
            q,
            k,
            v,
            Z,
            H,
            N_CTX,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            qkvo_element_size,
            fp8_v,
        ) = args
        cached_desc_q = TensorDescriptor(
            q,
            [Z * H * N_CTX, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_M, HEAD_DIM]
        )
        cached_desc_k = TensorDescriptor(
            k,
            [Z * H * N_CTX, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_N, HEAD_DIM]
        )
        cached_desc_v = TensorDescriptor(
            v,
            [Z * H * N_CTX, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_N, HEAD_DIM],
        )
    return cached_desc_q, cached_desc_k, cached_desc_v 


# Triton wrapper function
def attention_triton(Q, K, V, window_size):
    B, H, M, D = Q.shape
    output = torch.empty_like(Q)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = (B, H, triton.cdiv(M, BLOCK_SIZE_M))
    scale = 1.0 / math.sqrt(D)

    desc_q, desc_k, desc_v = get_fwd_tma_descriptors(
        Q,
        K,
        V,
        B,
        H,
        M,
        D,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        Q.element_size(),
        V.dtype == torch.float8_e5m2,
    )

    attention_kernel[grid](
        Q, K, V, output,
        desc_q,
        desc_k,
        desc_v,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        B, H, M, D, window_size,
        scale=scale,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        D_HEAD=D,
        num_warps=8,
        num_stages=3,
        mma_depth=1,
        enable_warp_specialization=True,
        math_wg_pipe=False,
    )
    return output

# Verification function
@pytest.mark.parametrize(
    "B, H, M, D, window_size", [
        (2, 2, 512, 128, 128)
    ]
) 
@pytest.mark.parametrize("do_bench", [False])
def test_flex_attn(B, H, M, D, window_size, do_bench):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Test requires Hopper target.")
    torch.manual_seed(0)
    B, H, M, D = 4, 4, 8192, 128
    window_size = 128
    Q = torch.randn((B, H, M, D), device='cuda', dtype=torch.float16)
    K = torch.randn((B, H, M, D), device='cuda', dtype=torch.float16)
    V = torch.randn((B, H, M, D), device='cuda', dtype=torch.float16)
    
    # Compute reference output
    torch_out = attention_sliding_window_pytorch(Q, K, V, window_size)
    
    # Compute Triton output
    triton_out = attention_triton(Q, K, V, window_size)
    
    # Check correctness
    assert torch.allclose(triton_out, torch_out, atol=1e-2, rtol=1e-2)
    print("PASS")
    
    # Performance comparison
    if do_bench:
        torch_time = triton.testing.do_bench(lambda: attention_sliding_window_pytorch(Q, K, V, window_size))
        triton_time = triton.testing.do_bench(lambda: attention_triton(Q, K, V, window_size))
        print(f"PyTorch time: {torch_time:.3f}ms")
        print(f"Triton time: {triton_time:.3f}ms")
        print(f"Speedup: {torch_time / triton_time:.2f}x")

if __name__ == "__main__":
    test_flex_attn(4, 4, 8192, 128, 128, True)
