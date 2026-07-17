"""
Fused Flash-Attention with native Grouped-Query / Multi-Query support (Triton).

This module implements memory-efficient, IO-aware exact attention (Flash
Attention v2 style) as Triton kernels, with first-class support for
Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) *inside* the
kernel -- i.e. without materialising a K/V tensor broadcast to the full number
of query heads. The forward and backward passes are both fused.

Layout
------
    q : [Z, H,    N_CTX, HEAD_DIM]   (fp16 / bf16 / fp32)
    k : [Z, H_KV, N_CTX, HEAD_DIM]
    v : [Z, H_KV, N_CTX, HEAD_DIM]
where ``H % H_KV == 0``. ``H_KV == H`` is standard MHA, ``H_KV == 1`` is MQA,
and ``1 < H_KV < H`` is GQA. Each query head ``h`` reads the KV head
``h // (H // H_KV)``.

Features
--------
* Online-softmax (running max/denominator) -- never materialises the
  ``N_CTX x N_CTX`` score matrix in HBM.
* Optional causal masking with the standard two-region (full / diagonal)
  split so masked blocks are skipped entirely.
* GQA/MQA handled by pointer arithmetic on the KV-head index -- zero extra
  memory traffic and no explicit ``repeat_interleave``.
* Backward pass computes dQ, dK, dV, correctly *reducing* the query-head
  gradients back onto the shared KV heads via atomic accumulation.
* Autograd-integrated ``torch.autograd.Function`` so it is a drop-in for
  training.

The kernels run on GPU (Triton codegen -> PTX/cubin) and are equally runnable
on CPU via ``TRITON_INTERPRET=1`` for correctness testing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Forward
# --------------------------------------------------------------------------- #
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,  # logsumexp, [Z, H, N_CTX] -- saved for backward
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    H_KV,
    N_CTX,
    GQA_GROUP: tl.constexpr,  # H // H_KV
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // GQA_GROUP  # GQA / MQA head mapping

    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_hkv * stride_kh
    v_base = V + off_z * stride_vz + off_hkv * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    m_mask = offs_m < N_CTX
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504089  # multiply by log2(e); use exp2

    # Upper bound of key blocks to visit.
    if IS_CAUSAL:
        hi = (start_m + 1) * BLOCK_M
    else:
        hi = N_CTX

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < N_CTX

        k_ptrs = k_base + (n_idx[None, :] * stride_kn + offs_d[:, None] * stride_kk)
        v_ptrs = v_base + (n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, k)
        qk = qk * qk_scale

        # masking: out-of-range keys and (optionally) causal
        qk = tl.where(n_mask[None, :], qk, float("-inf"))
        if IS_CAUSAL:
            causal = offs_m[:, None] >= n_idx[None, :]
            qk = tl.where(causal, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # finalise
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l[:, None]

    # logsumexp in natural log, for backward
    lse = m_i / 1.44269504089 + tl.math.log(safe_l)
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, lse, mask=m_mask)

    o_base = Out + off_z * stride_oz + off_h * stride_oh
    o_ptrs = o_base + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=m_mask[:, None])


# --------------------------------------------------------------------------- #
# Backward -- preprocessing: delta = rowsum(dO * O)
# --------------------------------------------------------------------------- #
@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N_CTX

    base = off_z * stride_oz + off_h * stride_oh
    o_ptrs = Out + base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    do_ptrs = DO + base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + offs_m, delta, mask=m_mask)


# --------------------------------------------------------------------------- #
# Backward -- main. One program per (batch*head, key-block). Iterates query
# blocks, accumulates dK/dV locally, and atomically scatters dQ.
# --------------------------------------------------------------------------- #
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    L,
    Delta,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    Z,
    H,
    H_KV,
    N_CTX,
    GQA_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // GQA_GROUP

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    n_mask = offs_n < N_CTX

    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_hkv * stride_kh
    v_base = V + off_z * stride_vz + off_hkv * stride_vh
    do_base = DO + off_z * stride_qz + off_h * stride_qh
    dq_base = DQ + off_z * stride_qz + off_h * stride_qh

    k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    RCP_LN2 = 1.44269504089
    if IS_CAUSAL:
        lo = start_n * BLOCK_N
    else:
        lo = 0

    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < N_CTX
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = do_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)
        lse = tl.load(L + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)
        delta = tl.load(Delta + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)

        # recompute p = softmax(qk) using saved lse
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        valid = m_mask[:, None] & n_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, tl.math.exp2((qk - lse[:, None]) * RCP_LN2), 0.0)

        # dv += p^T @ do
        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        # dp = do @ v^T ; ds = p * (dp - delta) * sm_scale
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid, ds, 0.0)
        # dk += ds^T @ q
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)
        # dq += ds @ k   (scatter with atomics: many key-blocks write same dq rows)
        dq_contrib = tl.dot(ds.to(k.dtype), k)
        dq_ptrs = dq_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        tl.atomic_add(dq_ptrs, dq_contrib, mask=m_mask[:, None])

    # scatter dk/dv onto shared KV head (GQA: multiple q-heads -> one kv head)
    dk_base = DK + off_z * stride_kz + off_hkv * stride_kh
    dv_base = DV + off_z * stride_vz + off_hkv * stride_vh
    dk_ptrs = dk_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    dv_ptrs = dv_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    tl.atomic_add(dk_ptrs, dk, mask=n_mask[:, None])
    tl.atomic_add(dv_ptrs, dv, mask=n_mask[:, None])


# --------------------------------------------------------------------------- #
# Autograd wrapper
# --------------------------------------------------------------------------- #
class _FlashAttnGQA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, block_m, block_n):
        Z, H, N_CTX, HEAD_DIM = q.shape
        H_KV = k.shape[1]
        assert H % H_KV == 0, "num_heads must be divisible by num_kv_heads"
        assert k.shape == (Z, H_KV, N_CTX, HEAD_DIM)
        assert v.shape == (Z, H_KV, N_CTX, HEAD_DIM)
        if sm_scale is None:
            sm_scale = 1.0 / (HEAD_DIM**0.5)
        GQA_GROUP = H // H_KV

        o = torch.empty_like(q)
        L = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(N_CTX, block_m), Z * H, 1)
        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            L,
            o,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *o.stride(),
            Z,
            H,
            H_KV,
            N_CTX,
            GQA_GROUP=GQA_GROUP,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=HEAD_DIM,
            IS_CAUSAL=causal,
        )
        ctx.save_for_backward(q, k, v, o, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.dims = (Z, H, H_KV, N_CTX, HEAD_DIM, GQA_GROUP)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        Z, H, H_KV, N_CTX, HEAD_DIM, GQA_GROUP = ctx.dims
        do = do.contiguous()

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        delta = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)

        pre_grid = (triton.cdiv(N_CTX, ctx.block_m), Z * H)
        _bwd_preprocess[pre_grid](
            o,
            do,
            delta,
            *o.stride(),
            Z,
            H,
            N_CTX,
            BLOCK_M=ctx.block_m,
            HEAD_DIM=HEAD_DIM,
        )
        grid = (triton.cdiv(N_CTX, ctx.block_n), Z * H, 1)
        _bwd_kernel[grid](
            q,
            k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,
            L,
            delta,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            Z,
            H,
            H_KV,
            N_CTX,
            GQA_GROUP=GQA_GROUP,
            BLOCK_M=ctx.block_m,
            BLOCK_N=ctx.block_n,
            HEAD_DIM=HEAD_DIM,
            IS_CAUSAL=ctx.causal,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def flash_attention(q, k, v, causal=False, sm_scale=None, block_m=32, block_n=32):
    """Fused Flash-Attention with GQA/MQA support.

    Args:
        q: [Z, H, N_CTX, D]
        k, v: [Z, H_KV, N_CTX, D]  (H % H_KV == 0)
        causal: apply causal mask.
        sm_scale: softmax scale (default 1/sqrt(D)).
        block_m, block_n: tile sizes.
    Returns:
        [Z, H, N_CTX, D]
    """
    return _FlashAttnGQA.apply(q, k, v, causal, sm_scale, block_m, block_n)
