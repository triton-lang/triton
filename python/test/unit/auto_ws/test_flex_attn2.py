import pytest
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# PyTorch reference implementation with causal mask
def torch_attention(Q, K, V):
    scale = Q.size(-1) ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    seq_len = Q.size(-2)
    # Create causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device), diagonal=0).bool()
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, V)
    return output

# Triton kernel with causal masking using TMA loads
@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    q_desc_ptr, k_desc_ptr, v_desc_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, DMODEL,  # Z: batch, H: heads, M: seq_len, DMODEL: d_model
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Compute program ids for batch, head and block of sequence
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Adjust the pointers to the current [batch, head]
    Q_ptr += pid_z * stride_qz + pid_h * stride_qh
    K_ptr += pid_z * stride_kz + pid_h * stride_kh
    V_ptr += pid_z * stride_vz + pid_h * stride_vh
    Out_ptr += pid_z * stride_oz + pid_h * stride_oh

    # Combine batch and head indices for TMA descriptor offset
    off_bh = pid_z * H + pid_h

    # FIXME: broken in rebase
    # Acquire TMA descriptors for Q, K, and V
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // dummy",
    #     "=r, l",
    #     [q_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // dummy",
    #     "=r, l",
    #     [k_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // dummy",
    #     "=r, l",
    #     [v_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )

    # Load Q with TMA load: block offset is computed as (off_bh * M + start_m)
    q = q_desc_ptr.load(
        [off_bh * M + start_m, 0],
    )

    # Initialize accumulators and running max/sum for softmax normalization
    m_prev = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
    l_prev = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # Iterate over blocks in the sequence dimension for K and V
    for pid_n in range(tl.cdiv(M, BLOCK_N)):
        start_n = pid_n * BLOCK_N
        offs_n_current = start_n + offs_n  # indices for current block in sequence

        # TMA load for K: use block offset (off_bh * M + start_n)
        k = k_desc_ptr.load(
            [off_bh * M + start_n, 0],
        )

        # Compute attention scores: [BLOCK_M, BLOCK_N] = q dot k^T
        scores = tl.dot(q, tl.trans(k)) * scale

        # Apply causal mask: only allow positions where query index >= key index
        mask = (offs_m[:, None] >= offs_n_current[None, :])
        scores = tl.where(mask, scores, float('-inf'))

        # Update softmax numerically stable accumulators
        m_curr = tl.maximum(tl.max(scores, axis=1), m_prev)
        alpha = tl.exp(m_prev - m_curr)
        p = tl.exp(scores - m_curr[:, None])
        l_curr = alpha * l_prev + tl.sum(p, axis=1)

        # TMA load for V: similar to K load
        v = v_desc_ptr.load(
            [off_bh * M + start_n, 0],
        )

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)

        m_prev = m_curr
        l_prev = l_curr

    # Final normalization of output
    acc = acc / l_prev[:, None]

    # Store the result with bounds check for last block
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_d[None, :] < DMODEL))

# Function to create TMA descriptors for Q, K, V
def get_fwd_tma_descriptors(Q, K, V, batch, heads, seq_len, d_model, BLOCK_M, BLOCK_N, element_size):
    total_rows = batch * heads * seq_len
    desc_q = TensorDescriptor(
        Q,
        [total_rows, d_model],
        [d_model, 1],
        [BLOCK_M, d_model]
    )
    desc_k = TensorDescriptor(
        K,
        [total_rows, d_model],
        [d_model, 1],
        [BLOCK_N, d_model]
    )
    desc_v = TensorDescriptor(
        V,
        [total_rows, d_model],
        [d_model, 1],
        [BLOCK_N, d_model]
    )
    return desc_q, desc_k, desc_v

def triton_attention(Q, K, V):
    # Ensure inputs are float16
    assert Q.dtype == K.dtype == V.dtype == torch.float16
    batch, heads, seq_len, d_model = Q.shape
    output = torch.empty_like(Q)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = d_model

    grid = (batch, heads, triton.cdiv(seq_len, BLOCK_M))
    scale = d_model ** -0.5

    # Create TMA descriptors for Q, K, V.
    desc_q, desc_k, desc_v = get_fwd_tma_descriptors(
        Q, K, V,
        batch, heads, seq_len, d_model,
        BLOCK_M, BLOCK_N,
        Q.element_size()
    )

    attention_kernel[grid](
        Q, K, V, output,
        desc_q, desc_k, desc_v,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch, heads, seq_len, d_model,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=8,
        num_stages=2,
        # mma_depth=1,
        enable_warp_specialization=True,
        math_wg_pipe=False,
    )
    return output

@pytest.mark.parametrize(
    "batch, heads, seq_len, d_model", [(2, 2, 512, 128)]
)
@pytest.mark.parametrize("do_bench", [False])
# Verification function
def test_flex_attn(batch, heads, seq_len, d_model, do_bench):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Test requires at least Hopper target.")
    batch, heads, seq_len, d_model = 4, 4, 8192, 128
    Q = torch.randn(batch, heads, seq_len, d_model, device='cuda', dtype=torch.float16)
    K = torch.randn(batch, heads, seq_len, d_model, device='cuda', dtype=torch.float16)
    V = torch.randn(batch, heads, seq_len, d_model, device='cuda', dtype=torch.float16)

    torch_out = torch_attention(Q, K, V)
    triton_out = triton_attention(Q, K, V)

    assert torch.allclose(torch_out, triton_out, atol=1e-2, rtol=1e-2)
    print("PASS")

    if do_bench:
        torch_time = triton.testing.do_bench(lambda: torch_attention(Q, K, V))
        triton_time = triton.testing.do_bench(lambda: triton_attention(Q, K, V))
        print(f"PyTorch time: {torch_time:.3f} ms")
        print(f"Triton time: {triton_time:.3f} ms")
        print(f"Speedup: {torch_time / triton_time:.2f}x")

if __name__ == "__main__":
    test_flex_attn(4, 4, 8192, 128, True)
