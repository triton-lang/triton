"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import pytest
import torch

import triton
import triton.language as tl

torch_dtype: tl.constexpr = torch.float16
TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2fnuz')
if TORCH_HAS_FP8E5:
    torch_dtype: tl.constexpr = torch.float8_e5m2fnuz


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX,
    pre_load_v: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        if pre_load_v:
            v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(v.dtype), v)
        # -- update m_i and l_i
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2, 'slice_k_tile': 0, 'pre_load_v': False},
                      num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'slice_k_tile': 0, 'pre_load_v': False},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'waves_per_eu': 2, 'slice_k_tile': 0, 'pre_load_v': False},
                      num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'slice_k_tile': 0, 'pre_load_v': True},
                      num_stages=1, num_warps=4),  # d64-False
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'slice_k_tile': 0, 'pre_load_v': False},
                      num_stages=1, num_warps=4),  # d64-True
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2, 'slice_k_tile': 32, 'pre_load_v': False},
                      num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'slice_k_tile': 32, 'pre_load_v': False},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'waves_per_eu': 2, 'slice_k_tile': 32, 'pre_load_v': False},
                      num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'slice_k_tile': 32, 'pre_load_v': True},
                      num_stages=1, num_warps=4),  # d64-False
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'slice_k_tile': 32, 'pre_load_v': False},
                      num_stages=1, num_warps=4),  # d64-True
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2, 'slice_k_tile': 64, 'pre_load_v': False},
                      num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'slice_k_tile': 64, 'pre_load_v': False},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'waves_per_eu': 2, 'slice_k_tile': 64, 'pre_load_v': False},
                      num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'slice_k_tile': 64, 'pre_load_v': True},
                      num_stages=1, num_warps=4),  # d64-False
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'slice_k_tile': 64, 'pre_load_v': False},
                      num_stages=1, num_warps=4),  # d64-True
    ],
    key=['Z', 'H', 'N_CTX', 'STAGE', 'BLOCK_DMODEL'],
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
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
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_DMODEL: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qkv_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(base=Q + qkv_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qkv_offset, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn),
                                    offsets=(0, 0), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qkv_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_vk, stride_vn),
                                    offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(0, 1))
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout on NV GPUs but in VGPRs on AMD GPUs
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(q.dtype)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            start_m,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            pre_load_v,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compiler to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            start_m,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
            pre_load_v,
        )
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(base=Out + qkv_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_on),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    NewDO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    D,
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
    stride_vk,
    stride_vn,
    Z,
    H,
    N_CTX,
    P_SEQ,
    num_block_q,
    num_block_kv,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    qk_scale = sm_scale * 1.44269504
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_kz + off_h * stride_kh
    DV += off_z * stride_vz + off_h * stride_vh
    # See fwd pass above for explanation.
    qk_scale = sm_scale * 1.44269504
    for start_n in range(0, num_block_kv):
        if CAUSAL:
            lo = tl.math.max(start_n * BLOCK_M - P_SEQ, 0)
        else:
            lo = 0
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[None, :] * stride_qm + offs_k[:, None] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        # initialize dk amd dv
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block_q * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            if CAUSAL:
                qk = tl.where(P_SEQ + offs_m_curr[:, None] >= (offs_n[None, :]), float(0.), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            l_i = tl.load(l_ptrs + offs_m_curr)
            p = tl.math.exp2(qk * qk_scale - l_i[:, None])
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, v)
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            # compute dq
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dq_ptrs, dq)
            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        tl.store(dk_ptrs, dk)
        tl.store(dv_ptrs, dv)


@triton.jit
def _bwd_kernel_dk_dv(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DK,
    DV,
    L,
    D,
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
    stride_vk,
    stride_vn,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.
    qvk_offset = off_hz * stride_qh
    qdo_offset = qvk_offset + start_m * BLOCK_M * stride_qm
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # offs_d = tl.arange(0, BLOCK_DMODEL)
    # Initialize pointers to Q, K, V
    Q_block_ptr = tl.make_block_ptr(base=Q + qdo_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
                                    offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn),
                                    offsets=(0, start_m * BLOCK_M), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qvk_offset, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_vn, stride_vk),
                                    offsets=(0, start_m * BLOCK_M), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    DO_block_ptr = tl.make_block_ptr(base=DO + qdo_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
                                     offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    qk_scale = sm_scale * 1.44269504
    # load k and v: they will stay in SRAM throughout
    k = tl.load(K_block_ptr)
    k = (k * qk_scale).to(k.dtype)
    v = tl.load(V_block_ptr)
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # This lower loop bound is because of the causal mask. We create a lower triangular
    # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
    # be ignored in the GEMM.
    lo = start_m * BLOCK_M
    hi = N_CTX
    # loop over q, do
    for start_n in range(lo, hi, BLOCK_N):
        offs_m_curr = offs_n[:, None] + start_n
        # -- load q, do --
        q = tl.load(Q_block_ptr)
        do = tl.load(DO_block_ptr)
        # -- compute qk ----
        qk = tl.dot(q, k)
        qk = tl.where(offs_m_curr >= offs_m[None, :], qk, float("-inf"))
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i)
        # -- compute dv ----
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m_curr)
        dp = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32) - Di
        dp += tl.dot(do, v)
        # compute ds = p * (dp - delta[:, None])
        ds = p * dp
        # compute dk
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        # update pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_N, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_N, 0))
    # initialize pointers to output
    DK_block_ptr = tl.make_block_ptr(base=DK + qvk_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_kn, stride_kk),
                                     offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    DV_block_ptr = tl.make_block_ptr(base=DV + qvk_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_vk, stride_vn),
                                     offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    tl.store(DK_block_ptr, (dk * sm_scale).to(k.dtype))
    tl.store(DV_block_ptr, dv.to(v.dtype))


@triton.jit
def _bwd_kernel_dq(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    L,
    D,
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
    stride_vk,
    stride_vn,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # offs_d = tl.arange(0, BLOCK_DMODEL)
    # Initialize pointers to Q, K, V
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn),
                                    offsets=(0, 0), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qvk_offset, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_vn, stride_vk),
                                    offsets=(0, 0), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    DO_block_ptr = tl.make_block_ptr(base=DO + qvk_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
                                     offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    qk_scale = sm_scale * 1.44269504
    # load q and do: they will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(q.dtype)
    do = tl.load(DO_block_ptr)
    Di = tl.load(D_ptrs + offs_m)
    l_i = tl.load(l_ptrs + offs_m)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # loop over k, v
    lo = 0
    hi = (start_m + 1) * BLOCK_M
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ----
        qk = tl.dot(q, k)
        qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float("-inf"))
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp += tl.dot(do, v)
        # compute ds = p * (dp - delta[:, None])
        ds = p * dp
        # compute dq. Unfortunately we cannot avoid transpose here as this loop
        # uses k both normal and transpose.
        dq += tl.dot(ds.to(Q.dtype.element_ty), tl.trans(k))
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
    # initialize pointers to output
    DQ_block_ptr = tl.make_block_ptr(base=DQ + qvk_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
                                     offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    tl.store(DQ_block_ptr, (dq * sm_scale).to(q.dtype))


empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, split_kernel=False):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-2]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        if torch.version.hip is None:
            # BLOCK_M = 128
            # BLOCK_N = 64 if Lk <= 64 else 32
            # num_stages = 4 if Lk <= 64 else 3
            # num_warps = 4 if Lk <= 64 else 8
            pass

        stage = 3 if causal else 1
        grid = lambda META: (triton.cdiv(q.shape[2], META['BLOCK_M']), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        _attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            M,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            N_CTX=q.shape[2],
            BLOCK_DMODEL=Lk,
            STAGE=stage,
        )

        ## restore the grid for bwd kernel
        best_config = _attn_fwd.get_best_config()
        block_m = int(best_config.__str__().split(",")[0].split("BLOCK_M:")[1])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[0] * q.shape[1], 1)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.split_kernel = split_kernel
        return o

    @staticmethod
    def backward(ctx, do):
        # configuration is not supported
        assert (not (ctx.split_kernel and not ctx.causal))
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        q, k, v, o, L = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        do_scaled = torch.empty_like(do)
        # Figure out what BLOCK size fwd used and adjust num_blocks accordingly.
        # If the two are the same, we don't need this but the bwd pass block size
        # is smaller than the fwd so we need this scaling to ensure we loop over all
        # values and don't skip some blocks.
        # Alternatively we could compute a new grid but this keeps it consistent
        # with fwd and easier to reason about.
        block_scale = (q.shape[2] // ctx.grid[0]) // BLOCK
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o,
            do,
            do_scaled,
            delta,
            BLOCK_M=block_scale * BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        if not ctx.split_kernel:
            _bwd_kernel[(ctx.grid[1], )](
                q,
                k,
                v,
                ctx.sm_scale,
                o,
                do_scaled,
                dq,
                dk,
                dv,
                L,
                delta,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                q.shape[0],
                q.shape[1],
                q.shape[2],
                block_scale * ctx.grid[0],
                BLOCK_M=BLOCK,
                BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL,
                num_warps=4,
                CAUSAL=ctx.causal,
                num_stages=1,
            )
        else:
            dq = torch.zeros_like(q)
            _bwd_kernel_dk_dv[(block_scale * ctx.grid[0], ctx.grid[1])](
                q,
                k,
                v,
                ctx.sm_scale,
                o,
                do_scaled,
                dk,
                dv,
                L,
                delta,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                q.shape[0],
                q.shape[1],
                q.shape[2],
                BLOCK_M=BLOCK,
                BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL,
                num_warps=4,
                num_stages=1,
            )
            _bwd_kernel_dq[ctx.grid](
                q,
                k,
                v,
                ctx.sm_scale,
                o,
                do_scaled,
                dq,
                L,
                delta,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                q.shape[0],
                q.shape[1],
                q.shape[2],
                BLOCK_M=2 * BLOCK,
                BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL,
                num_warps=4,
                waves_per_eu=1,
                num_stages=1,
            )
        # print(h.asm["ttgir"])
        return dq, dk, dv, None, None, None


attention = _attention.apply


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [
    (4, 48, 1024, 64),
    (4, 48, 2048, 64),
    (4, 48, 4096, 64),
    (4, 48, 1024, 128),
    (4, 48, 2048, 128),
    (4, 48, 4096, 128),
    #(4, 48, 8192, 64),
    #(4, 48, 16384, 64)
])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_op_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_())
    v = (torch.empty((Z, H, D_HEAD, N_CTX), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_())
    sm_scale = 0.5
    # dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(v.dtype)
    ref_out = torch.matmul(p, v.transpose(2, 3))
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale)
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [
    (4, 48, 1024, 64),
    (4, 48, 2048, 64),
    (4, 48, 4096, 64),
    (1, 16, 8192, 64),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_op_bwd(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    causal = True
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = 0, 5
    split_kernel = True
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(v.dtype)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, split_kernel)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    if torch.version.hip is None:
        assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        assert torch.allclose(ref_dv, tri_dv, atol=5e-2, rtol=0)
    assert torch.allclose(ref_dk, tri_dk, atol=5e-2, rtol=0)
    assert torch.allclose(ref_dq, tri_dq, atol=5e-2, rtol=0)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None

name_to_torch_types = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

# vary seq length for fixed head and batch=4
configs = []
for mode in ['fwd']:
    for dtype in ["fp16", "bf16"]:
        for D_HEAD in [128, 64]:
            for causal in [False, True]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=['BATCH', 'H', 'N_CTX'], x_vals=[
                            (16, 16, 1024),
                            (8, 16, 2048),
                            (4, 16, 4096),
                            (2, 16, 8192),
                            (1, 16, 16384),
                            (4, 48, 1024),
                            (4, 48, 2048),
                            (4, 48, 4096),
                            (4, 48, 8192),
                            (4, 48, 16384),
                        ], line_arg='provider', line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
                        line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []), styles=[('red', '-'),
                                                                                                       ('blue', '-')],
                        ylabel='ms', plot_name=f'fused-attention-d{D_HEAD}-{mode}-causal={causal}-{dtype}',
                        args={'D_HEAD': D_HEAD, 'dtype': dtype, 'mode': mode, 'causal': causal}))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype, device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 100
    init_dtype = name_to_torch_types[dtype]
    split_kernel = False
    # Bwd pass only supports causal=True right now
    if mode == 'bwd':
        causal = True
        split_kernel = True
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, D_HEAD, N_CTX), dtype=init_dtype, device="cuda", requires_grad=True)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, split_kernel)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=init_dtype, device=device, requires_grad=True)
        if FLASH_VER == 1:
            lengths = torch.full((BATCH, ), fill_value=N_CTX, device=device)
            cu_seqlens = torch.zeros((BATCH + 1, ), device=device, dtype=torch.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            qkv = qkv.reshape(BATCH * N_CTX, 3, H, D_HEAD)
            fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=causal)
        elif FLASH_VER == 2:
            fn = lambda: flash_attn_func(qkv, causal=causal)
        else:
            raise ValueError(f'unknown {FLASH_VER = }')
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == 'bwd':
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


# only works on post-Ampere GPUs right now
bench_flash_attention.run(save_path='.', print_data=True)
