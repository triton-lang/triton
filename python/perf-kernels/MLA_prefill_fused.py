# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

import torch
import triton
import triton.language as tl
from utils.rotary_embedding import DeepseekScalingRotaryEmbedding
from utils.sglang_ref import extend_attention_fwd as extend_attention_fwd_ref
import argparse
import sys
import pytest

is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

is_hip_ = is_hip()

@triton.jit
def _fwd_kernel_fused(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    W_KC,
    W_VC,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_w_kch,
    stride_w_kcc,
    stride_w_vch,
    stride_w_vcc,
    logit_cap: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    # offs_c = tl.arange(0, BLOCK_C)
    offs_c_k = tl.arange(0, BLOCK_K)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < D
    mask_dv = offs_dv < DV
    mask_k = offs_c_k < C

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_D + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        offs_kv_loc = tl.load(
            kv_indices + cur_seq_kv_start_idx + start_n + offs_n, mask=mask_n, other=0
        )

        # load k in transposed way
        offs_buf_k = (
            offs_kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_c_k[:, None]
        )

        offs_buf_w_kc = (
            cur_head * stride_w_kch + offs_c_k[None, :] * stride_w_kcc + offs_d[:, None]
        )

        k = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
        # tile the C dim
        for _ in range(tl.cdiv(BLOCK_C, BLOCK_K)):
            w_kc = tl.load(W_KC + offs_buf_w_kc, mask=(mask_d[:, None]) & (mask_k[None, :]), other=0.0)
            k_pre_proj = tl.load(
                K_Buffer + offs_buf_k, mask=(mask_n[None, :]) & (mask_k[:, None]), other=0.0
            )

            k += tl.dot(w_kc, k_pre_proj).to(k_pre_proj.dtype)

            offs_buf_k += BLOCK_K
            offs_buf_w_kc += BLOCK_K * stride_w_kcc

        qk = tl.dot(q, k.to(q.dtype))
        if BLOCK_DPE > 0:
            offs_kpe = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Buffer + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe.to(kpe.dtype), kpe)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(custom_mask, qk, float("-inf"))
        else:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_buf_v = (
            offs_kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_c_k[None, :]
        )

        offs_buf_w_vc = (
            cur_head * stride_w_vch
            + offs_c_k[:, None] * stride_w_vcc
            + offs_dv[None, :]
        )

        v = tl.zeros([BLOCK_N, BLOCK_DV], dtype=tl.float32)
        for _ in range(tl.cdiv(BLOCK_DV, BLOCK_K)):
            v_pre_proj = tl.load(
                V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_k[None, :], other=0.0
            )
            w_vc = tl.load(W_VC + offs_buf_w_vc, mask=mask_dv[None, :] & mask_k[:, None], other=0.0)

            v += tl.dot(v_pre_proj, w_vc).to(v_pre_proj.dtype)
            offs_buf_v += BLOCK_K
            offs_buf_w_vc += BLOCK_K * stride_w_vcc

        p = p.to(q.dtype)
        v = v.to(q.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    # stage 2: compute the triangle part

    cur_block_m_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        # load k in transposed way
        offs_k = (
            (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_c_k[:, None]
        )

        offs_w_kc = (
            cur_head * stride_w_kch + offs_d[:, None] + offs_c_k[None, :] * stride_w_kcc
        )

        k = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
        # tile the C dim
        for _ in range(tl.cdiv(BLOCK_C, BLOCK_K)):
            w_kc = tl.load(W_KC + offs_w_kc, mask=(mask_d[:, None]) &(mask_k[None, :]), other=0.0)
            k_pre_proj = tl.load(
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_k[:, None]), other=0.0
            )

            k += tl.dot(w_kc, k_pre_proj).to(k_pre_proj.dtype)

            offs_k += BLOCK_K
            offs_w_kc += BLOCK_K * stride_w_kcc

        qk = tl.dot(q, k.to(q.dtype), out_dtype=tl.float32)
        if BLOCK_DPE > 0:
            offs_kpe = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Extend + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(custom_mask, qk, float("-inf"))
        else:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(mask_causual, qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_c_k[None, :]
        )
        offs_w_vc = (
            cur_head * stride_w_vch
            + offs_c_k[:, None] * stride_w_vcc
            + offs_dv[None, :]
        )

        v = tl.zeros([BLOCK_N, BLOCK_DV], dtype=tl.float32)
        for _ in range(tl.cdiv(BLOCK_DV, BLOCK_K)):
            v_pre_proj = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_k[None, :], other=0.0
            )
            w_vc = tl.load(W_VC + offs_w_vc, mask=mask_dv[None, :] & mask_k[:, None], other=0.0)

            v += tl.dot(v_pre_proj, w_vc).to(v_pre_proj.dtype)
            offs_v += BLOCK_K
            offs_w_vc += BLOCK_K * stride_w_vcc

        p = p.to(q.dtype)
        v = v.to(q.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / deno[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )


def extend_attention_fwd_fused(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    w_kc,
    w_vc,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    mask_indptr,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )

    D = w_kc.shape[-1]
    DV = w_vc.shape[-1]
    C = w_kc.shape[-2]

    if Lk == 576:
        BLOCK_C = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_C = 256
        BLOCK_DPE = 32
    elif Lk == 192:
        BLOCK_C = 128
        BLOCK_DPE = 64
    else:
        BLOCK_C = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_DV = triton.next_power_of_2(DV)

    if is_hip_:
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4

    else:
        if is_cuda_available and CUDA_CAPABILITY[0] >= 9:
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (128, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        elif is_cuda_available and CUDA_CAPABILITY[0] >= 8:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        else:
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

        num_warps = 4 if Lk <= 64 else 8

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    _fwd_kernel_fused[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        w_kc,
        w_vc,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        w_kc.stride(0),
        w_kc.stride(1),
        w_vc.stride(0),
        w_vc.stride(1),
        logit_cap=logit_cap,
        BLOCK_D=BLOCK_D,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_C=BLOCK_C,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=128,
        D=D,
        DV=DV,
        C=C,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        STORE_TRANSPOSE=is_hip_,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )


"""
forward_normal
q_extend.shape torch.Size([2048, 16, 192])
k_extend.shape torch.Size([2048, 16, 192])
v_extend.shape torch.Size([2048, 16, 128])

k_buffer.shape torch.Size([1044998, 1, 576])
v_buffer.shape torch.Size([1044998, 1, 512])

qo_indptr: tensor([   0, 2048], device='cuda:5', dtype=torch.int32)
kv_indptr: tensor([0, 0], device='cuda:1', dtype=torch.int32)
"""
"""
our forward_normal 
q_extend.shape: torch.Size([2048, 16, 192])
k_extend.shape: torch.Size([2048, 16, 192])
v_extend.shape: torch.Size([2048, 16, 128])
k_buffer.shape: torch.Size([0, 1, 576])
v_buffer.shape: torch.Size([0, 1, 576])
qo_indptr: tensor([   0, 2048], device='cuda:0')
kv_indptr: tensor([0, 0], device='cuda:0')
MLA-decode:
"""
"""
forward_absorb
q_extend.shape torch.Size([2048, 1, 576])
k_extend.shape torch.Size([2048, 1, 576])
v_extend.shape torch.Size([2048, 1, 512])

k_buffer.shape torch.Size([1044998, 1, 576])
v_buffer.shape torch.Size([1044998, 1, 512])

qo_indptr: tensor([   0, 2048], device='cuda:5', dtype=torch.int32)
kv_indptr: tensor([0, 2048], device='cuda:1', dtype=torch.int32)
"""
def input_helper(B, H, S_prefix, S_extend, kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, dtype, device):
    q = torch.randn(B * S_extend, H, qk_nope_head_dim + qk_rope_head_dim, dtype=dtype, device=device)
    kv_cache = torch.randn(B * S_extend, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)

    k_buffer = torch.randn(B * S_prefix, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype)
    v_buffer = torch.randn(B * S_prefix, 1, kv_lora_rank, dtype=dtype)

    # interlancing [batch_start_off, batch_seq_len, batch_start_off, batch_seq_len, ...,]
    qo_indptr = torch.arange(B + 1, device=device) * S_extend
    kv_indptr = torch.arange(B + 1, device=device) * S_prefix # 0, prefix_length, prefix_length*2
    kv_indices = torch.arange(B * (S_prefix), device=device)

    # o_extend = torch.empty(B * S_extend, H, kv_lora_rank, dtype=dtype, device=device)
    w_kc = torch.randn(H, kv_lora_rank, qk_nope_head_dim, dtype=dtype, device=device)
    w_vc = torch.randn(H, kv_lora_rank, v_head_dim, dtype=dtype, device=device)

    rotary_emb = DeepseekScalingRotaryEmbedding(
        qk_rope_head_dim,
        rotary_dim=qk_rope_head_dim,
        max_position_embeddings=16324,
        base=10,
        is_neox_style=True,
        scaling_factor=1.0,
        dtype=q.dtype,
        device=device,
    )

    positions = torch.tensor([S_extend], device=device).unsqueeze(0).repeat(B, 1)  # k positions and q position as last

    return q, kv_cache, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, w_kc, w_vc, rotary_emb, positions


def kv_b_proj(kv_a, w_kc, w_vc):
    kv_lora_rank = kv_a.shape[-1]
    qk_nope_head_dim = w_kc.shape[-1]
    v_head_dim = w_vc.shape[-1]
    num_heads = w_kc.shape[0]
    w = torch.cat((w_kc, w_vc), dim=-1).transpose(0, 1).reshape(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
    return torch.matmul(kv_a, w).type_as(kv_a)


"""
kv_a.shape: torch.Size([2048, 512])
kv.shape: torch.Size([2048, 4096])
"""
def forward_normal_ref(
    q,
    latent_cache,
    k_buffer,
    v_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    w_kc,
    w_vc,
    H,
    kv_lora_rank,
    qk_nope_head_dim,
    v_head_dim,
    qk_rope_head_dim,
    rotary_emb,
    positions
):
    _, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    kv_a, _ = latent_cache.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    # projection
    # kv_a = self.kv_a_layernorm(kv_a.contiguous())
    kv = kv_b_proj(kv_a, w_kc, w_vc)
    kv = kv.view(-1, H, qk_nope_head_dim + v_head_dim)
    k_nope = kv[..., : qk_nope_head_dim]
    v = kv[..., qk_nope_head_dim :]
    k_pe = latent_cache[:, :, kv_lora_rank :]
    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
    q[..., qk_nope_head_dim :] = q_pe
    k = torch.empty_like(q)
    k[..., : qk_nope_head_dim] = k_nope
    k[..., qk_nope_head_dim :] = k_pe

    latent_cache[:, :, : kv_lora_rank] = kv_a
    latent_cache[:, :, kv_lora_rank :] = k_pe

    extend_attention_fwd_ref(
        q,
        k,
        v,
        o,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask=None,
        mask_indptr=None,
        max_len_extend=qo_indptr[1]
    )
    attn_output = o
    attn_output = attn_output.reshape(-1, H * v_head_dim)
    return attn_output


def forward_normal_fused(
        q,
        latent_cache,
        k_buffer,
        v_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        w_kc,
        w_vc,
        H,
        kv_lora_rank,
        qk_nope_head_dim,
        v_head_dim,
        qk_rope_head_dim,
        rotary_emb,
        positions
    ):
    _, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    kv_a, _ = latent_cache.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    # projection
    # kv_a = self.kv_a_layernorm(kv_a.contiguous())
    # kv = kv_b_proj(kv_a, w_kc, w_vc)
    # kv = kv.view(-1, H, qk_nope_head_dim + v_head_dim)
    k_nope = v = kv_a[..., : kv_lora_rank]
    # k_nope = kv[..., : qk_nope_head_dim]
    # v = kv[..., qk_nope_head_dim :]
    k_pe = latent_cache[:, :, kv_lora_rank :]
    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
    q[..., qk_nope_head_dim :] = q_pe
    # TODO check this
    k_pe = k_pe.squeeze(0)
    k = torch.cat((k_nope, k_pe), dim=-1)

    latent_cache[:, :, : kv_lora_rank] = kv_a
    latent_cache[:, :, kv_lora_rank :] = k_pe

    extend_attention_fwd_fused(
        q,
        k,
        v,
        o,
        k_buffer,
        v_buffer,
        w_kc,
        w_vc,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask=None,
        mask_indptr=None,
        max_len_extend=qo_indptr[1]
    )
    attn_output = o
    attn_output = attn_output.reshape(-1, H * v_head_dim)
    return attn_output



@pytest.mark.parametrize('B, H, S_prefix, S_extend, kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim', [
    (1, 16, 0, 2048, 512, 128, 128, 64)
])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_op_fwd(B, H, S_prefix, S_extend, kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, dtype, device="cuda"):
    torch.manual_seed(0)

    q, kv_cache, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, w_kc, w_vc, rotary_emb, positions = input_helper(
        B,
        H,
        S_prefix,
        S_extend,
        kv_lora_rank,
        qk_nope_head_dim,
        v_head_dim,
        qk_rope_head_dim,
        dtype,
        device
    )

    tri_o = torch.empty(qo_indptr[-1], H, v_head_dim, dtype=q.dtype, device=q.device)
    ref_o = torch.empty(qo_indptr[-1], H, v_head_dim, dtype=q.dtype, device=q.device)

    tri_output = forward_normal_fused(
        q,
        kv_cache,
        k_buffer,
        v_buffer,
        tri_o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        w_kc,
        w_vc,
        H,
        kv_lora_rank,
        qk_nope_head_dim,
        v_head_dim,
        qk_rope_head_dim,
        rotary_emb,
        positions
    )

    ref_output = forward_normal_ref(
        q,
        kv_cache,
        k_buffer,
        v_buffer,
        ref_o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        w_kc,
        w_vc,
        H,
        kv_lora_rank,
        qk_nope_head_dim,
        v_head_dim,
        qk_rope_head_dim,
        rotary_emb,
        positions)
    print(ref_output)
    print(tri_output)
    torch.testing.assert_close(ref_output, tri_output, atol=1e-2, rtol=1e-2)


# forward_batch.extend_prefix_lens.sum() == 0 => forward_normal
def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    configs = []

    # prefill
    x_vals_list = [(args.B, 16, 0, 2048, 512, 128, 128, 64)]
    x_names = ["B", "H", "S_prefix", "S_extend", "kv_lora_rank", "qk_nope_head_dim", "v_head_dim", "qk_rope_head_dim"]
    line_vals = ["ref", "fused"]
    plot_name = "MLA-decode"

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-')], ylabel='ms',
                                 plot_name=plot_name, args={'sm_scale': 1.0, 'logit_cap': 0.0, 'device': args.device}))

    @triton.testing.perf_report(configs)
    def bench_MLA(B, H, S_prefix, S_extend, kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, sm_scale, logit_cap, device,
                  provider):
        warmup = 25
        rep = 100

        q, kv_cache, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, w_kc, w_vc, rotary_emb, positions = input_helper(
            B,
            H,
            S_prefix,
            S_extend,
            kv_lora_rank,
            qk_nope_head_dim,
            v_head_dim,
            qk_rope_head_dim,
            dtype,
            device)

        o = torch.empty(qo_indptr[-1], H, v_head_dim, dtype=q.dtype, device=q.device)

        if "ref" in provider:
            fn = lambda: {
                forward_normal_ref(q, kv_cache, k_buffer, v_buffer, o, qo_indptr, kv_indptr, kv_indices, w_kc, w_vc, H, kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, rotary_emb, positions)
            }
        if "fused" in provider:
            fn = lambda: {
                forward_normal_fused(q, kv_cache, k_buffer, v_buffer, o, qo_indptr, kv_indptr, kv_indices, w_kc, w_vc, H, kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, rotary_emb, positions)
            }

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_MLA.run(save_path=".", print_data=True, show_plots=False)


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA",
        allow_abbrev=False,
    )

    parser.add_argument("-dtype", default='bf16', help="data type")
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-B", type=int, default=1)
    return parser.parse_args()


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def main():
    torch.manual_seed(0)
    args = parse_args()
    torch.set_default_device(args.device)
    benchmark(args)


if __name__ == '__main__':
    sys.exit(main())