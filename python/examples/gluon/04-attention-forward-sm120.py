"""
Flash Attention Forward v2 for SM120 (Consumer Blackwell)
=========================================================

This example implements flash attention using Gluon on SM120 GPUs
(DGX Spark, RTX 5090, GB10). SM120 lacks tcgen05/TMEM/WGMMA but has
MMAv2 (mma.sync.aligned) tensor cores and cp.async for async data movement.

Features:
- Double-buffered K pipelining with FIFO-correct async groups:
  K[i+1] prefetch overlaps QK compute + softmax, V[i] load overlaps softmax.
  Both wait_group calls use compile-time constant 1.
- Causal masking with early loop termination
- BF16 and FP8 (E5M2) support
- Online softmax with numerically stable rescaling

Contributed by Second Nature Computing (https://joinsecondnature.com)
"""

import itertools

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from triton.experimental.gluon.language.nvidia.ampere import (
    async_copy as cp,
    mma_v2,
)


def is_sm12x():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 12


if __name__ == "__main__" and not is_sm12x():
    raise RuntimeError("This example requires an SM120 NVIDIA GPU (DGX Spark, RTX 5090, GB10)")


# ---------------------------------------------------------------------------
# Configuration aggregate — defines all layouts and block sizes
# ---------------------------------------------------------------------------

@aggregate
class AttentionConfig:
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    HEAD_DIM: gl.constexpr
    NUM_WARPS: gl.constexpr
    IS_FP8: gl.constexpr

    # MMA accumulator layout
    mma_layout: gl.constexpr
    # Operand layouts for Q*K^T
    q_dot_layout: gl.constexpr
    k_dot_layout: gl.constexpr
    # Operand layouts for P*V
    p_dot_layout: gl.constexpr
    v_dot_layout: gl.constexpr
    # Blocked layout for global loads/stores
    blocked_layout: gl.constexpr
    # Shared memory layouts
    q_smem_layout: gl.constexpr
    kv_smem_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, BLOCK_M, BLOCK_N, HEAD_DIM, NUM_WARPS, IS_FP8):
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.IS_FP8 = gl.constexpr(IS_FP8)

        # MMAv2 accumulator layout: mma.sync.aligned.m16n8k{16,32}
        self.mma_layout = gl.constexpr(gl.NVMMADistributedLayout(
            version=[2, 0],
            warps_per_cta=[NUM_WARPS, 1],
            instr_shape=[16, 8],
        ))

        # BF16: 8 elements per 128-bit register group (8*16=128 bits)
        # FP8:  16 elements per 128-bit register group (16*8=128 bits)
        k_width = 16 if IS_FP8 else 8

        # Q*K^T operand layouts
        self.q_dot_layout = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=self.mma_layout, k_width=k_width))
        self.k_dot_layout = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=self.mma_layout, k_width=k_width))

        # P*V operand layouts
        self.p_dot_layout = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=self.mma_layout, k_width=k_width))
        self.v_dot_layout = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=self.mma_layout, k_width=k_width))

        # Blocked layout for global memory access
        self.blocked_layout = gl.constexpr(
            gl.BlockedLayout([1, 1], [32, 1], [NUM_WARPS, 1], [1, 0]))

        # Shared memory layouts (swizzled for bank-conflict-free MMA access)
        smem_dtype = gl.float8e5 if IS_FP8 else gl.bfloat16
        self.q_smem_layout = gl.constexpr(
            gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], smem_dtype))
        self.kv_smem_layout = gl.constexpr(
            gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], smem_dtype))


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

RCP_LN2 = gl.constexpr(1.4426950408889634)  # 1/ln(2) for exp2-based softmax


@gluon.jit
def attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale,
    SEQ_LEN_Q: gl.constexpr,
    SEQ_LEN_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_FP8: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = gl.num_warps()
    cfg = AttentionConfig(BLOCK_M, BLOCK_N, HEAD_DIM, NUM_WARPS, IS_FP8)

    SMEM_DTYPE: gl.constexpr = gl.float8e5 if IS_FP8 else gl.bfloat16

    # Program indices: [batch*heads, seq_blocks]
    off_zh = gl.program_id(0)
    off_m = gl.program_id(1) * BLOCK_M

    # Compute batch and head offsets
    q_offset = off_zh * stride_qh
    k_offset = off_zh * stride_kh
    v_offset = off_zh * stride_vh
    o_offset = off_zh * stride_oh

    # --- Load Q block to shared memory, then to registers ---
    q_smem = gl.allocate_shared_memory(
        SMEM_DTYPE, [BLOCK_M, HEAD_DIM], layout=cfg.q_smem_layout)

    q_offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.blocked_layout))[:, None]
    q_offs_k = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, cfg.blocked_layout))[None, :]
    q_ptrs = q_ptr + q_offset + (off_m + q_offs_m) * stride_qm + q_offs_k * stride_qk
    q_mask = (off_m + q_offs_m) < SEQ_LEN_Q

    cp.async_copy_global_to_shared(q_smem, q_ptrs, mask=q_mask)
    cp.commit_group()
    cp.wait_group(0)
    gl.barrier()

    q = q_smem.load(cfg.q_dot_layout)

    # --- Initialize online softmax state ---
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32,
                  layout=gl.SliceLayout(1, cfg.mma_layout))
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32,
                  layout=gl.SliceLayout(1, cfg.mma_layout))
    acc = gl.zeros([BLOCK_M, HEAD_DIM], dtype=gl.float32, layout=cfg.mma_layout)

    # --- Allocate K/V shared memory ---
    # K is double-buffered for async pipelining; V is single-buffered
    k_smem = gl.allocate_shared_memory(
        SMEM_DTYPE, [2, BLOCK_N, HEAD_DIM], layout=cfg.kv_smem_layout)
    v_smem = gl.allocate_shared_memory(
        SMEM_DTYPE, [BLOCK_N, HEAD_DIM], layout=cfg.kv_smem_layout)

    # Precompute K/V offset patterns
    kv_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, cfg.blocked_layout))[:, None]
    kv_offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, cfg.blocked_layout))[None, :]

    # Scaling factor
    qk_scale = sm_scale * RCP_LN2

    # --- Causal loop bound ---
    if IS_CAUSAL:
        loop_end = min(off_m + BLOCK_M, SEQ_LEN_K)
    else:
        loop_end = SEQ_LEN_K

    # Precompute m_offs for causal mask (only used inside loop if IS_CAUSAL)
    m_offs = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.mma_layout))

    # --- Prologue: prefetch K[0] into buffer 0 ---
    k_ptrs_0 = k_ptr + k_offset + kv_offs_n * stride_kn + kv_offs_d * stride_kk
    k_mask_0 = kv_offs_n < SEQ_LEN_K
    cp.async_copy_global_to_shared(k_smem.index(0), k_ptrs_0, mask=k_mask_0)
    cp.commit_group()
    # Outstanding FIFO: [K[0]]

    # --- Main loop over K/V blocks ---
    # Double-buffer pipelining: K[i] prefetched in prologue/previous iter,
    # V[i] issued first so it's newest in FIFO. Both wait_group(1) calls
    # use a compile-time constant, avoiding the runtime-value limitation.
    for start_n in range(0, loop_end, BLOCK_N):
        # Current/next K buffer (alternates 0, 1, 0, 1, ...)
        cur_buf = (start_n // BLOCK_N) % 2
        next_buf = 1 - cur_buf

        # Issue V[i] load (async) — newest in FIFO
        v_ptrs = v_ptr + v_offset + (start_n + kv_offs_n) * stride_vn + kv_offs_d * stride_vk
        v_mask = (start_n + kv_offs_n) < SEQ_LEN_K
        cp.async_copy_global_to_shared(v_smem, v_ptrs, mask=v_mask)
        cp.commit_group()
        # FIFO: [K[i], V[i]]

        # Wait for K[i] (oldest completes), V[i] stays in flight
        cp.wait_group(1)
        gl.barrier()

        # Load K[i]: [BLOCK_N, HEAD_DIM] → transposed for Q @ K^T
        k = k_smem.index(cur_buf).permute([1, 0]).load(cfg.k_dot_layout)

        # Prefetch K[i+1] into next buffer (OOB on last iter → mask handles it)
        next_start = start_n + BLOCK_N
        k_next_ptrs = k_ptr + k_offset + (next_start + kv_offs_n) * stride_kn + kv_offs_d * stride_kk
        k_next_mask = (next_start + kv_offs_n) < SEQ_LEN_K
        cp.async_copy_global_to_shared(k_smem.index(next_buf), k_next_ptrs, mask=k_next_mask)
        cp.commit_group()
        # FIFO: [V[i], K[i+1]]

        # QK = Q @ K^T: [BLOCK_M, BLOCK_N]
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=cfg.mma_layout)
        qk = mma_v2(q, k, qk)

        # Mask out-of-bounds positions
        n_offs = start_n + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, cfg.mma_layout))[None, :]
        qk = gl.where(n_offs < SEQ_LEN_K, qk, float("-inf"))

        # Causal mask: mask positions where query position < key position
        if IS_CAUSAL:
            causal_mask = (off_m + m_offs[:, None]) >= n_offs
            qk = gl.where(causal_mask, qk, float("-inf"))

        # --- Online softmax (overlaps V[i] and K[i+1] loads) ---
        m_ij = gl.maximum(m_i, gl.max(qk, 1))
        m_ij_scaled = m_ij * qk_scale

        q_shifted = qk * qk_scale - m_ij_scaled[:, None]
        p = gl.exp2(q_shifted)

        m_diff = m_i * qk_scale - m_ij_scaled
        alpha = gl.exp2(m_diff)

        l_ij = gl.sum(p, 1)

        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # --- P * V ---
        p_cast = p.to(SMEM_DTYPE, fp_downcast_rounding="rtz")
        p_dot = gl.convert_layout(p_cast, cfg.p_dot_layout)

        # Wait for V[i] (oldest completes), K[i+1] stays in flight
        cp.wait_group(1)
        gl.barrier()

        v = v_smem.load(cfg.v_dot_layout)

        acc = mma_v2(p_dot, v, acc)

    # Drain last K prefetch (unused — loop has ended)
    cp.wait_group(0)

    # --- Final normalization ---
    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]

    # --- Store output ---
    out = acc.to(SMEM_DTYPE, fp_downcast_rounding="rtz")
    out_blocked = gl.convert_layout(out, cfg.blocked_layout)

    o_offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.blocked_layout))[:, None]
    o_offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, cfg.blocked_layout))[None, :]
    o_ptrs = out_ptr + o_offset + (off_m + o_offs_m) * stride_om + o_offs_d * stride_ok
    o_mask = (off_m + o_offs_m) < SEQ_LEN_Q
    gl.store(o_ptrs, out_blocked, mask=o_mask)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

def attention_forward(q, k, v, sm_scale, causal=False, BLOCK_M=128, BLOCK_N=64, num_warps=4):
    """Run flash attention forward pass.

    Args:
        q: [B, H, Sq, D] bfloat16 or float8_e5m2
        k: [B, H, Sk, D] bfloat16 or float8_e5m2
        v: [B, H, Sk, D] bfloat16 or float8_e5m2
        sm_scale: softmax scale (typically 1/sqrt(D))
        causal: whether to apply causal masking

    Returns:
        out: [B, H, Sq, D] same dtype as input
    """
    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    is_fp8 = q.dtype == torch.float8_e5m2
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in (torch.bfloat16, torch.float8_e5m2)

    out = torch.empty_like(q)

    # Grid: [B*H, ceil(Sq/BLOCK_M)]
    grid = (B * H, triton.cdiv(Sq, BLOCK_M))

    attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sm_scale,
        SEQ_LEN_Q=Sq,
        SEQ_LEN_K=Sk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
        IS_FP8=is_fp8,
        num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def reference_attention(q, k, v, sm_scale, causal=False):
    """Torch reference implementation."""
    # SDPA needs float16/bfloat16 — upcast FP8 inputs
    if q.dtype == torch.float8_e5m2:
        q_ref = q.to(torch.bfloat16)
        k_ref = k.to(torch.bfloat16)
        v_ref = v.to(torch.bfloat16)
    else:
        q_ref, k_ref, v_ref = q, k, v
    return torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, scale=sm_scale, is_causal=causal)


# --- BF16 non-causal tests ---
@pytest.mark.parametrize("B, H, Sq, Sk, D", [
    (1, 4, 128, 128, 64),
    (1, 4, 256, 256, 64),
    (1, 4, 128, 128, 128),
    (1, 4, 256, 256, 128),
    (2, 8, 128, 128, 64),
    (2, 8, 512, 512, 64),
    (1, 4, 256, 512, 64),   # non-square: Sq < Sk
    (1, 4, 512, 256, 64),   # non-square: Sq > Sk
    (2, 4, 1024, 1024, 64),
    (2, 4, 1024, 1024, 128),
])
@pytest.mark.skipif(not is_sm12x(), reason="Requires SM120 GPU")
def test_attention_bf16(B, H, Sq, Sk, D):
    torch.manual_seed(42)
    q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.bfloat16)
    sm_scale = 1.0 / (D ** 0.5)

    ref = reference_attention(q, k, v, sm_scale, causal=False)
    out = attention_forward(q, k, v, sm_scale, causal=False)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print(f"  PASS [BF16]: B={B} H={H} Sq={Sq} Sk={Sk} D={D}")


# --- BF16 causal tests ---
@pytest.mark.parametrize("B, H, S, D", [
    (1, 4, 128, 64),
    (1, 4, 256, 64),
    (1, 4, 128, 128),
    (1, 4, 256, 128),
    (2, 8, 512, 64),
    (2, 4, 1024, 64),
    (2, 4, 1024, 128),
])
@pytest.mark.skipif(not is_sm12x(), reason="Requires SM120 GPU")
def test_attention_bf16_causal(B, H, S, D):
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    sm_scale = 1.0 / (D ** 0.5)

    ref = reference_attention(q, k, v, sm_scale, causal=True)
    out = attention_forward(q, k, v, sm_scale, causal=True)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print(f"  PASS [BF16 causal]: B={B} H={H} S={S} D={D}")


# --- FP8 non-causal tests ---
@pytest.mark.parametrize("B, H, Sq, Sk, D", [
    (1, 4, 128, 128, 64),
    (1, 4, 256, 256, 64),
    (1, 4, 128, 128, 128),
    (1, 4, 256, 256, 128),
    (2, 8, 512, 512, 64),
    (1, 4, 256, 512, 64),
    (2, 4, 1024, 1024, 64),
])
@pytest.mark.skipif(not is_sm12x(), reason="Requires SM120 GPU")
def test_attention_fp8(B, H, Sq, Sk, D):
    torch.manual_seed(42)
    # Generate in BF16 then cast — FP8 has limited range
    q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    k = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    v = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    sm_scale = 1.0 / (D ** 0.5)

    ref = reference_attention(q, k, v, sm_scale, causal=False)
    out = attention_forward(q, k, v, sm_scale, causal=False)

    # FP8 E5M2 has only 2 mantissa bits — errors compound through matmuls
    out_bf16 = out.to(torch.bfloat16)
    ref_bf16 = ref.to(torch.bfloat16)
    torch.testing.assert_close(out_bf16, ref_bf16, rtol=1e-1, atol=4e-1)
    print(f"  PASS [FP8]: B={B} H={H} Sq={Sq} Sk={Sk} D={D}")


# --- FP8 causal tests ---
@pytest.mark.parametrize("B, H, S, D", [
    (1, 4, 128, 64),
    (1, 4, 256, 64),
    (1, 4, 128, 128),
    (1, 4, 256, 128),
    (2, 8, 512, 64),
    (2, 4, 1024, 64),
])
@pytest.mark.skipif(not is_sm12x(), reason="Requires SM120 GPU")
def test_attention_fp8_causal(B, H, S, D):
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    sm_scale = 1.0 / (D ** 0.5)

    ref = reference_attention(q, k, v, sm_scale, causal=True)
    out = attention_forward(q, k, v, sm_scale, causal=True)

    out_bf16 = out.to(torch.bfloat16)
    ref_bf16 = ref.to(torch.bfloat16)
    torch.testing.assert_close(out_bf16, ref_bf16, rtol=1e-1, atol=4e-1)
    print(f"  PASS [FP8 causal]: B={B} H={H} S={S} D={D}")


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

BATCH = [1]
N_HEADS = [16]
HEAD_DIM = [64, 128]
causal = [False, True]
providers = ["triton-bf16", "triton-fp8", "torch-bf16"]
N_CTX = [2**i for i in range(8, 13)]  # 256 to 4096

bench_configs = []
for Z, H, D, is_causal in itertools.product(BATCH, N_HEADS, HEAD_DIM, causal):
    config = triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=N_CTX,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name=f"SM120-Attention Z={Z} H={H} D={D} causal={is_causal}",
        args={
            "Z": Z,
            "H": H,
            "HEAD_DIM": D,
            "causal": is_causal,
        },
    )
    bench_configs.append(config)


@triton.testing.perf_report(bench_configs)
def bench(Z, H, N_CTX, HEAD_DIM, causal, provider):
    device = "cuda"
    torch.manual_seed(42)

    provider_name, dtype_str = provider.split("-")
    if dtype_str == "bf16":
        dtype = torch.bfloat16
    elif dtype_str == "fp8":
        dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    if dtype == torch.float8_e5m2:
        q = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=torch.bfloat16).to(dtype)
        k = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=torch.bfloat16).to(dtype)
        v = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=torch.bfloat16).to(dtype)
    else:
        q = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
        k = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
        v = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    if provider_name == "triton":
        fn = lambda: attention_forward(q, k, v, sm_scale, causal=causal)
    elif provider_name == "torch":
        # SDPA needs BF16 — upcast FP8 inputs
        q_ref = q.to(torch.bfloat16) if dtype == torch.float8_e5m2 else q
        k_ref = k.to(torch.bfloat16) if dtype == torch.float8_e5m2 else k
        v_ref = v.to(torch.bfloat16) if dtype == torch.float8_e5m2 else v
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, scale=sm_scale, is_causal=causal)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * Z * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench.run(save_path=".", print_data=True)
