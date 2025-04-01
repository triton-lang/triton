"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm
See https://tridao.me/publications/flash2/flash2.pdf

Credits:
AMD Triton kernels team
OpenAI kernel team

Currently only the forward kernel is supported, and contains these features:

1) Fwd with causal masking
2) Arbitrary Q and KV sequence lengths
3) Arbitrary head sizes
4) Multi and grouped query attention
5) Variable sequence lengths
6) ALiBi and matrix bias

"""

import argparse
import subprocess
import pytest
import sys
import torch

import triton
import triton.language as tl
from utils.benchmark_utils import get_available_models, get_model_configs


class MetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    persistent = None
    num_contexts = 0
    varlen = False
    int8 = False
    layout = None
    dropout_p, return_encoded_softmax = 0.0, False

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

    def set_persistent(self, persistent):
        self.persistent = persistent

    def set_int8_params(self, q_descale, k_descale, v_descale, p_scale, p_descale):
        self.int8 = True
        self.q_descale = q_descale
        self.k_descale = k_descale
        self.v_descale = v_descale
        self.p_scale = p_scale
        self.p_descale = p_descale
        self.use_p_scale = (p_scale is not None) and (p_descale is not None) and (v_descale is not None)
        self.int8_kv = (q_descale is None) and (k_descale is not None) and (v_descale is not None)

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_dropout(self, dropout_p, return_encoded_softmax):
        self.dropout_p = dropout_p
        self.return_encoded_softmax = return_encoded_softmax

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()

        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, self)
        if self.varlen:
            assert q.dim() == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias is None
            # TODO:Remove once dropout is supported with varlen
            assert self.dropout_p == 0.0
            assert not self.return_encoded_softmax
        else:
            assert q.dim() == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        if self.int8:
            if self.int8_kv:
                assert v.dtype == k.dtype and k.dtype == torch.int8
                assert q.dtype != k.dtype
                assert (self.v_descale is not None) and (self.k_descale is not None)
            else:
                assert q.dtype == k.dtype and q.dtype == v.dtype and q.dtype == torch.int8
                assert (self.q_descale is not None) and (self.k_descale is not None) and (self.v_descale is not None)
                if self.use_p_scale:
                    assert (self.p_scale is not None) and (self.p_descale is not None)
        else:
            assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == 'thd' or not self.varlen


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def print_gpu(prefix, val=None):
    if (tl.program_id(0) == 0) and ((tl.program_id(1) == 0) and (tl.program_id(2) == 0)):
        if val is not None:
            tl.device_print(prefix, val)
        else:
            tl.device_print(prefix)


@triton.jit
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


def compute_alibi_tensor(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn, start_m,
                    actual_seqlen_k, actual_seqlen_q, dropout_p, philox_seed, batch_philox_offset, encoded_sm_ptrs,
                    block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope, q_descale,
                    k_descale, v_descale, p_scale, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, OFFS_M: tl.constexpr, OFFS_N: tl.constexpr,
                    PRE_LOAD_V: tl.constexpr, MASK_STEPS: tl.constexpr, ENABLE_DROPOUT: tl.constexpr,
                    RETURN_ENCODED_SOFTMAX: tl.constexpr, PADDED_HEAD: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr,
                    QK_SCALE: tl.constexpr, INT8_GEMM: tl.constexpr, USE_P_SCALE: tl.constexpr, INT8_KV: tl.constexpr):
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL, actual_seqlen_k)
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        # -- compute qk ----
        if INT8_GEMM:
            qk += ((((tl.dot(q, k).to(tl.float32) * q_descale)) * k_descale) * QK_SCALE)
        else:
            if INT8_KV:
                k = (k * k_descale).to(q.type.element_ty)
            qk += (tl.dot(q, k) * QK_SCALE)

        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q, actual_seqlen_k)
            # While bias is added after multiplying qk with sm_scale,
            # our optimization to use 2^x instead of e^x results in an additional
            # scale factor of log2(e) which we must also multiply the bias with.
            qk += (bias * 1.44269504089)

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q, actual_seqlen_k, global_m_positions,
                                              global_n_positions)
            qk += (alibi_block * 1.44269504089)  # scale factor of log2(e)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                tl.store(encoded_sm_ptrs, tl.where(keep, p, -p).to(encoded_sm_ptrs.type.element_ty))
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_sm_ptrs, p.to(encoded_sm_ptrs.type.element_ty))
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        if INT8_GEMM:
            if USE_P_SCALE:
                p = (p * p_scale).to(tl.int8)
                # They are all int8
                acc += tl.dot(p, v)
            else:
                # v is in int8 but p is not, we want the gemm in p's type
                acc += tl.dot(p, v.to(p.type.element_ty))
        else:
            if INT8_KV:
                v = (v * v_descale).to(p.type.element_ty)
            acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_ENCODED_SOFTMAX:
            encoded_sm_ptrs += BLOCK_N
    return acc, l_i, m_i


def get_gfx_version():
    try:
        # Run the rocminfo command
        result = subprocess.run(['rocminfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout

        # Parse the output to find the gfx version
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("Name: gfx"):
                gfx_version = line.split("Name:")[1].strip()
                return gfx_version
    except Exception as e:
        print(f"Error: {e}")
    return None


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def is_rdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ("gfx1030", "gfx1100", "gfx1101",
                                                                                   "gfx1102", "gfx1200", "gfx1201")


def get_cdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK']


def get_rdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 4, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 4, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 4, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
        # Fall-back config.
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=2),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK']


def get_autotune_configs():
    if is_rdna():
        return get_rdna_autotune_configs()
    elif is_cdna():
        return get_cdna_autotune_configs()
    else:
        raise ValueError("Unknown Device Type")


autotune_configs, autotune_keys = get_autotune_configs()


@triton.autotune(
    configs=autotune_configs,
    key=autotune_keys,
    use_cuda_graph=True,
)
@triton.jit
def attn_fwd(Q, K, V, bias, SM_SCALE: tl.constexpr, L, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
             stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh,
             stride_om, stride_on, stride_bz, stride_bh, stride_bm, stride_bn, stride_az, stride_ah, Q_descale,
             K_descale, P_scale, P_descale, V_descale, cu_seqlens_q, cu_seqlens_k, dropout_p, philox_seed,
             PERSISTENT: tl.constexpr, PERSISTENT_DYNAMIC: tl.constexpr, atomic_counter, NUM_CU: tl.constexpr,
             GRID_CU_MULTIP: tl.constexpr, B: tl.constexpr, philox_offset_base, encoded_softmax, alibi_slopes,
             HQ: tl.constexpr, HK: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr,
             MAX_SEQLENS_K: tl.constexpr, VARLEN: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, USE_BIAS: tl.constexpr,
             ENABLE_DROPOUT: tl.constexpr, RETURN_ENCODED_SOFTMAX: tl.constexpr, USE_ALIBI: tl.constexpr,
             INT8: tl.constexpr, USE_P_SCALE: tl.constexpr, INT8_KV: tl.constexpr):

    tl.assume(stride_qz >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_qm >= 0)
    tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_kn >= 0)
    tl.assume(stride_kk >= 0)
    tl.assume(stride_bz >= 0)
    tl.assume(stride_bh >= 0)
    tl.assume(stride_bm >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_vz >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_vk >= 0)
    tl.assume(stride_vn >= 0)
    tl.assume(stride_oz >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_om >= 0)
    tl.assume(stride_on >= 0)

    if PERSISTENT:  # if persistent, kernel loops over multiple tiles
        NUM_WG = NUM_CU * GRID_CU_MULTIP  # number of workgroups launched
        num_tiles_per_head = tl.cdiv(MAX_SEQLENS_Q, BLOCK_M)  # the number of work units (tiles) of a single head
        num_tiles_per_sample = num_tiles_per_head * HQ  # times the number of heads
        num_tiles_total = num_tiles_per_sample * B  # times the number of samples
        if PERSISTENT_DYNAMIC:
            tile_id = atomic_counter.atomic_add(1)  # retuns the value BEFORE the atomic operation
        else:
            tile_id = tl.program_id(0)
    else:  # standard, kernel processes only one tile
        tile_id = 0
        num_tiles_total = 1

    while tile_id < num_tiles_total:  # loops more than once only if PERSISTENT
        if PERSISTENT:
            # tile id basically tells us the Q block we are handling
            off_z = tile_id // num_tiles_per_sample  # at which batch sample are we
            off_h_q = tile_id % num_tiles_per_sample // num_tiles_per_head  # at which head are we inside the sample
            start_m = tile_id % num_tiles_per_sample % num_tiles_per_head  # at which tile are we inside the head
        else:
            start_m = tl.program_id(0)
            off_h_q = tl.program_id(1)
            off_z = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        continue_condition = True  # as we can't have return statements inside while loop in Triton

        if VARLEN:
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            # We have a one-size-fits-all grid in id(0). Some seqlens might be too
            # small for all start_m so for those we return early.
            if start_m * BLOCK_M > seqlen_q:
                continue_condition = False
                # return
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        else:
            cu_seqlens_q_start = 0
            cu_seqlens_k_start = 0
            seqlen_q = MAX_SEQLENS_Q
            seqlen_k = MAX_SEQLENS_K

        if continue_condition:
            # Now we compute whether we need to exit early due to causal masking.
            # This is because for seqlen_q > seqlen_k, M rows of the attn scores
            # are completely masked, resulting in 0s written to the output, and
            # inf written to LSE. We don't need to do any GEMMs in this case.
            # This block of code determines what N is, and if this WG is operating
            # on those M rows.
            n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
            if (IS_CAUSAL):
                # If seqlen_q == seqlen_k, the attn scores are a square matrix.
                # If seqlen_q != seqlen_k, attn scores are rectangular which means
                # the causal mask boundary is bottom right aligned, and ends at either
                # the top edge (seqlen_q < seqlen_k) or left edge.
                # This captures the decrease in n_blocks if we have a rectangular attn matrix
                n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
                # This is what adjusts the block_max for the current WG, only
                # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
                n_blocks = min(n_blocks, n_blocks_seqlen)
                # If we have no blocks after adjusting for seqlen deltas, this WG is part of
                # the blocks that are all 0. We exit early.
                if n_blocks <= 0:
                    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
                    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
                    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
                    o_ptrs_mask = (offs_m[:, None] < seqlen_q).broadcast_to([BLOCK_M, BLOCK_DMODEL])
                    # We still need to write 0s to the result
                    tl.store(o_ptrs, acc, mask=o_ptrs_mask)
                    # The tensor allocated for L is based on MAX_SEQLENS_Q as that is
                    # statically known.
                    l_ptrs = L + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
                    # We store inf to LSE, not -inf because in the bwd pass, we subtract this
                    # from qk which makes it -inf, such that exp(qk - inf) = 0 for these masked blocks.
                    l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
                    l_ptrs_mask = offs_m < MAX_SEQLENS_Q
                    tl.store(l_ptrs, l, mask=l_ptrs_mask)
                    # TODO: Should dropout and return encoded softmax be handled here too?
                    continue_condition = False
                    # return

            if continue_condition:
                # If MQA / GQA, set the K and V head offsets appropriately.
                GROUP_SIZE: tl.constexpr = HQ // HK
                if GROUP_SIZE != 1:
                    off_h_k = off_h_q // GROUP_SIZE
                else:
                    off_h_k = off_h_q

                n_extra_tokens = 0
                if seqlen_k < BLOCK_N:
                    n_extra_tokens = BLOCK_N - seqlen_k
                elif seqlen_k % BLOCK_N:
                    n_extra_tokens = seqlen_k % BLOCK_N
                PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

                # Compute pointers for all the tensors used in this kernel.
                q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
                q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
                k_offset = K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
                k_ptrs = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
                v_offset = V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
                v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
                # Compute pointers for all the scale tensors used in this kernel.

                INT8_GEMM: tl.constexpr = INT8 & (not INT8_KV)
                if INT8:
                    k_descale_ptrs = K_descale + off_h_k
                    v_descale_ptrs = V_descale + off_h_k
                    if not INT8_KV:
                        q_descale_ptrs = Q_descale + off_h_q
                    if USE_P_SCALE:
                        p_scale_ptrs = P_scale + off_h_q
                        p_descale_ptrs = P_descale + off_h_q

                if USE_BIAS:
                    # Note: this might get large enough to overflow on some configs
                    bias_offset = off_h_q * stride_bh
                    bias_ptrs = bias + bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
                else:
                    bias_ptrs = None

                if USE_ALIBI:
                    a_offset = off_z * stride_az + off_h_q * stride_ah
                    alibi_slope = tl.load(alibi_slopes + a_offset)
                else:
                    alibi_slope = None

                if ENABLE_DROPOUT:
                    off_hz = off_z * HQ + off_h_q
                    batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
                else:
                    batch_philox_offset = 0
                # We can ask to return the dropout mask without actually doing any dropout. In
                # this case, we return an invalid pointer so indicate the mask is not valid.
                if RETURN_ENCODED_SOFTMAX:
                    encoded_sm_base = encoded_softmax + off_h_q * seqlen_q * seqlen_k
                    encoded_sm_ptrs = encoded_sm_base + offs_m[:, None] * seqlen_k + offs_n[None, :]
                else:
                    encoded_sm_ptrs = None
                # initialize pointer to m and l
                m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
                l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
                acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
                # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
                # have native e^x support in HW.
                QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504089
                # Q is loaded once at the beginning and shared by all N blocks.
                q_ptrs_mask = offs_m[:, None] < seqlen_q
                if PADDED_HEAD:
                    q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

                if INT8:
                    k_descale = tl.load(k_descale_ptrs)
                    v_descale = tl.load(v_descale_ptrs)
                    if not INT8_KV:
                        q_descale = tl.load(q_descale_ptrs)
                    else:
                        q_descale = None
                    if USE_P_SCALE:
                        p_scale = tl.load(p_scale_ptrs)
                        p_descale = tl.load(p_descale_ptrs)
                    else:
                        p_scale = None
                        p_descale = None
                else:
                    q_descale = None
                    k_descale = None
                    v_descale = None
                    p_scale = None
                    p_descale = None
                # Here we compute how many full and masked blocks we have.
                padded_block_k = n_extra_tokens != 0
                is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
                if IS_CAUSAL:
                    # There are always at least BLOCK_M // BLOCK_N masked blocks.
                    # Additionally there might be one more due to dissimilar seqlens.
                    masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
                else:
                    # Padding on Q does not need to be masked in the FA loop.
                    masked_blocks = padded_block_k
                # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
                # In this case we might exceed n_blocks so pick the min.
                masked_blocks = min(masked_blocks, n_blocks)
                n_full_blocks = n_blocks - masked_blocks
                block_min = 0
                block_max = n_blocks * BLOCK_N
                # Compute for full blocks. Here we set causal to false regardless of its actual
                # value because there is no masking. Similarly we do not need padding.
                if n_full_blocks > 0:
                    block_max = (n_blocks - masked_blocks) * BLOCK_N
                    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk,
                                                    stride_bn, start_m, seqlen_k, seqlen_q, dropout_p, philox_seed,
                                                    batch_philox_offset, encoded_sm_ptrs,
                                                    # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
                                                    block_min, block_max, 0, 0, 0, alibi_slope, q_descale, k_descale,
                                                    v_descale, p_scale,
                                                    # IS_CAUSAL, ....
                                                    False, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                                    # _, MASK_STEPS, ...
                                                    PRE_LOAD_V, False, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX,
                                                    PADDED_HEAD, ACTUAL_BLOCK_DMODEL, QK_SCALE, INT8_GEMM, USE_P_SCALE,
                                                    INT8_KV)
                    block_min = block_max
                    block_max = n_blocks * BLOCK_N

                tl.debug_barrier()
                # Remaining blocks, if any, are full / not masked.
                if (masked_blocks > 0):
                    if IS_CAUSAL:
                        offs_n_causal = offs_n + (seqlen_q - seqlen_k)
                    else:
                        offs_n_causal = 0
                    k_ptrs += n_full_blocks * BLOCK_N * stride_kn
                    v_ptrs += n_full_blocks * BLOCK_N * stride_vk
                    if USE_BIAS:
                        bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
                    if RETURN_ENCODED_SOFTMAX:
                        encoded_sm_ptrs += n_full_blocks * BLOCK_N
                    acc, l_i, m_i = _attn_fwd_inner(
                        acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn, start_m, seqlen_k,
                        seqlen_q, dropout_p, philox_seed, batch_philox_offset, encoded_sm_ptrs, block_min, block_max,
                        offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope, q_descale, k_descale, v_descale,
                        p_scale, IS_CAUSAL, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                        # _, MASK_STEPS, ...
                        PRE_LOAD_V, True, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, PADDED_HEAD, ACTUAL_BLOCK_DMODEL,
                        QK_SCALE, INT8_GEMM, USE_P_SCALE, INT8_KV)

                if INT8 and not INT8_KV:
                    if USE_P_SCALE:
                        acc *= p_descale
                    acc *= v_descale

                # epilogue
                # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
                l_recip = 1 / l_i[:, None]
                acc = acc * l_recip

                if ENABLE_DROPOUT:
                    acc = acc / (1 - dropout_p)
                # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
                # then we have one block with a row of all NaNs which come from computing
                # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
                # and store 0s where there are NaNs as these rows should've been zeroed out.
                end_m_idx = (start_m + 1) * BLOCK_M
                start_m_idx = start_m * BLOCK_M
                causal_start_idx = seqlen_q - seqlen_k
                acc = acc.to(Out.type.element_ty)
                if IS_CAUSAL:
                    if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
                        out_mask_boundary = tl.full((BLOCK_DMODEL, ), causal_start_idx, dtype=tl.int32)
                        mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
                        out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
                        z = 0.0
                        acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
                # write back LSE
                l_ptrs = L + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
                # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
                # This is only true for the last M block. For others, overflow_size will be -ve
                overflow_size = end_m_idx - seqlen_q
                if overflow_size > 0:
                    boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
                    l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
                    tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
                else:
                    tl.store(l_ptrs, m_i + tl.math.log2(l_i))

                # write back O
                o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
                o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
                o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
                if overflow_size > 0:
                    o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
                if PADDED_HEAD:
                    o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)

        if PERSISTENT:
            if PERSISTENT_DYNAMIC:
                tile_id = atomic_counter.atomic_add(1)
            else:
                tile_id += NUM_WG
        else:
            tile_id = num_tiles_total  # break after single tile


@triton.jit
def _attn_bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    seqlen_q,
    head_dim,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    # off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    # off_n = tl.arange(0, D_HEAD)
    off_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1)  # head index
    off_z = tl.program_id(2)  # batch index
    num_h = tl.num_programs(1)
    o_offset = off_h * stride_oh + off_z * stride_oz
    O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=(seqlen_q, head_dim), strides=(stride_om, stride_on),
                                    offsets=(off_m, 0), block_shape=(BLOCK_M, D_HEAD), order=(1, 0))
    do_offset = off_h * stride_doh + off_z * stride_doz
    DO_block_ptr = tl.make_block_ptr(base=DO + do_offset, shape=(seqlen_q, head_dim), strides=(stride_dom, stride_don),
                                     offsets=(off_m, 0), block_shape=(BLOCK_M, D_HEAD), order=(1, 0))
    # load
    # o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back, shape (q.shape[0] * q.shape[1], q.shape[2])
    off_zh = off_z * num_h + off_h * 1
    # Check for OOB accesses
    delta_ptrs = Delta + off_zh * seqlen_q + off_m + tl.arange(0, BLOCK_M)
    overflow = off_m + BLOCK_M - seqlen_q
    if overflow > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow, dtype=tl.int32)
        mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(delta_ptrs, delta, mask=mask)
    else:
        tl.store(delta_ptrs, delta)


@triton.jit
def _bwd_kernel_dk_dv(dk, dv, Q, k, v, sm_scale, alibi_slope, DO, M, D,
                      # shared by Q/K/V/DO.
                      stride_tok, stride_d, H, N_CTX, BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
                      BLOCK_DMODEL: tl.constexpr,
                      # Filled in by the wrapper.
                      start_n, start_m, num_steps, MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    # offs_k = tl.arange(0, BLOCK_DMODEL)
    QT_block_ptr = tl.make_block_ptr(base=Q, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_d, stride_tok),
                                     offsets=(0, start_m), block_shape=(BLOCK_DMODEL, BLOCK_M1), order=(0, 1))
    DO_block_ptr = tl.make_block_ptr(base=DO, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_tok, stride_d),
                                     offsets=(start_m, 0), block_shape=(BLOCK_M1, BLOCK_DMODEL), order=(1, 0))
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(QT_block_ptr)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        kqT = tl.dot(k, qT)
        if alibi_slope is not None:
            alibi_block = compute_alibi_block(alibi_slope, N_CTX, N_CTX, offs_m, offs_n, True)
            kqT += alibi_block * 1.44269504089

        pT = tl.math.exp2(kqT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(DO_block_ptr)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        QT_block_ptr = tl.advance(QT_block_ptr, (0, step_m))
        DO_block_ptr = tl.advance(DO_block_ptr, (step_m, 0))
    return dk, dv


@triton.jit
def _bwd_kernel_dq(dq, q, K, V, do, m, D, alibi_slope,
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d, H, N_CTX, BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
                   BLOCK_DMODEL: tl.constexpr,
                   # Filled in by the wrapper.
                   start_m, start_n, num_steps, MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    # offs_k = tl.arange(0, BLOCK_DMODEL)
    KT_block_ptr = tl.make_block_ptr(base=K, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_d, stride_tok),
                                     offsets=(0, start_n), block_shape=(BLOCK_DMODEL, BLOCK_N2), order=(0, 1))
    VT_block_ptr = tl.make_block_ptr(base=V, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_d, stride_tok),
                                     offsets=(0, start_n), block_shape=(BLOCK_DMODEL, BLOCK_N2), order=(0, 1))
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(KT_block_ptr)
        qk = tl.dot(q, kT)
        if alibi_slope is not None:
            alibi_block = compute_alibi_block(alibi_slope, N_CTX, N_CTX, offs_m, offs_n)
            qk += alibi_block * 1.44269504089

        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        vT = tl.load(VT_block_ptr)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.0.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        KT_block_ptr = tl.advance(KT_block_ptr, (0, step_n))
        VT_block_ptr = tl.advance(VT_block_ptr, (0, step_n))
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale, alibi_slopes, DO, DQ, DK, DV, M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,
              # H = 16, N_CTX = 1024
              H, N_CTX, BLOCK_DMODEL: tl.constexpr, BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
              BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, BLK_SLICE_FACTOR: tl.constexpr, USE_ALIBI: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # offs_k = tl.arange(0, BLOCK_DMODEL)

    start_n = pid * BLOCK_N1
    # This assignment is important. It is what allows us to pick the diagonal
    # blocks. Later, when we want to do the lower triangular, we update start_m
    # after the first dkdv call.
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    # offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)

    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_tok, stride_d),
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_tok, stride_d),
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1, 0),
    )

    # load K and V: they stay in SRAM throughout the inner loop for dkdv.
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    if USE_ALIBI:
        a_offset = bhid
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    # compute dK and dV for blocks close to the diagonal that need to be masked
    num_steps = BLOCK_N1 // MASK_BLOCK_M1
    dk, dv = _bwd_kernel_dk_dv(dk, dv, Q, k, v, sm_scale, alibi_slope, DO, M, D, stride_tok, stride_d, H, N_CTX,
                               MASK_BLOCK_M1, BLOCK_N1, BLOCK_DMODEL, start_n, start_m, num_steps, MASK=True)

    # compute dK and dV for blocks that don't need masking further from the diagonal
    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    dk, dv = _bwd_kernel_dk_dv(dk, dv, Q, k, v, sm_scale, alibi_slope, DO, M, D, stride_tok, stride_d, H, N_CTX,
                               BLOCK_M1, BLOCK_N1, BLOCK_DMODEL, start_n, start_m, num_steps, MASK=False)

    DV_block_ptrs = tl.make_block_ptr(base=DV, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_tok, stride_d),
                                      offsets=(start_n, 0), block_shape=(BLOCK_N1, BLOCK_DMODEL), order=(1, 0))
    tl.store(DV_block_ptrs, dv.to(v.dtype))

    # Write back dK.
    dk *= sm_scale
    DK_block_ptrs = tl.make_block_ptr(base=DK, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_tok, stride_d),
                                      offsets=(start_n, 0), block_shape=(BLOCK_N1, BLOCK_DMODEL), order=(1, 0))
    tl.store(DK_block_ptrs, dk.to(k.dtype))

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    Q_block_ptr = tl.make_block_ptr(base=Q, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_tok, stride_d),
                                    offsets=(start_m, 0), block_shape=(BLOCK_M2, BLOCK_DMODEL), order=(1, 0))

    DO_block_ptr = tl.make_block_ptr(base=DO, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_tok, stride_d),
                                     offsets=(start_m, 0), block_shape=(BLOCK_M2, BLOCK_DMODEL), order=(1, 0))
    q = tl.load(Q_block_ptr)
    do = tl.load(DO_block_ptr)
    dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _bwd_kernel_dq(dq, q, K, V, do, m, D, alibi_slope, stride_tok, stride_d, H, N_CTX, BLOCK_M2, MASK_BLOCK_N2,
                        BLOCK_DMODEL, start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps, MASK=True)
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _bwd_kernel_dq(dq, q, K, V, do, m, D, alibi_slope, stride_tok, stride_d, H, N_CTX, BLOCK_M2, BLOCK_N2,
                        BLOCK_DMODEL, start_m, end_n - num_steps * BLOCK_N2, num_steps, MASK=False)
    # Write back dQ.
    DQ_block_ptr = tl.make_block_ptr(base=DQ, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_tok, stride_d),
                                     offsets=(start_m, 0), block_shape=(BLOCK_M2, BLOCK_DMODEL), order=(1, 0))
    dq *= LN2
    tl.store(DQ_block_ptr, dq.to(q.dtype))


def get_shape_from_layout(q, k, metadata):
    if metadata.layout == 'thd':
        nheads_q, nheads_k = q.shape[1], k.shape[1]
        head_size = q.shape[-1]
        batch = metadata.num_contexts
    elif metadata.layout == 'bhsd':
        batch, nheads_q, _, head_size = q.shape
        nheads_k = k.shape[1]
    elif metadata.layout == 'bshd':
        batch, _, nheads_q, head_size = q.shape
        nheads_k = k.shape[2]
    else:
        assert False, "Got unsupported layout."
    return batch, nheads_q, nheads_k, head_size


# TODO: This can probably optimized to have fewer lines of code.
def get_strides_from_layout(q, k, v, o, metadata):
    if metadata.layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    elif metadata.layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif metadata.layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return q_strides, k_strides, v_strides, o_strides


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, o, metadata: MetaData):
        # NOTE: a large bias tensor leads to overflow during pointer arithmetic
        if (metadata.bias is not None):
            assert (metadata.bias.numel() < 2**31)

        if o is None:
            if not metadata.int8:
                o = torch.empty_like(q, dtype=v.dtype)
            else:
                o = torch.empty_like(q, dtype=torch.float16)

        metadata.check_args(q, k, v, o)

        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, metadata)
        q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, metadata)

        # Get closest power of 2 over or equal to 32.
        padded_d_model = 1 << (head_size - 1).bit_length()
        # Smallest head_dim supported is 16. If smaller, the tile in the
        # kernel is padded - there is no padding in memory for any dims.
        padded_d_model = max(padded_d_model, 16)

        # encoded_softmax is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
        # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
        # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
        # only.  This return holds no useful output aside from debugging.
        if metadata.return_encoded_softmax:
            encoded_softmax = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device,
                                          dtype=torch.float32)
        else:
            encoded_softmax = None

        M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32)

        # Seed the RNG so we get reproducible results for testing.
        philox_seed = 0x1BF52
        philox_offset = 0x1D4B42

        if metadata.bias is not None:
            bias_strides = (metadata.bias.stride(0), metadata.bias.stride(1), metadata.bias.stride(2),
                            metadata.bias.stride(3))
        else:
            bias_strides = (0, 0, 0, 0)

        if metadata.alibi_slopes is not None:
            alibi_strides = (metadata.alibi_slopes.stride(0), metadata.alibi_slopes.stride(1))
        else:
            alibi_strides = (0, 0)

        if metadata.int8:
            q_descale, k_descale, p_scale, p_descale, v_descale = metadata.q_descale, metadata.k_descale, metadata.p_scale, metadata.p_descale, metadata.v_descale
        else:
            q_descale = k_descale = p_scale = p_descale = v_descale = None

        # number of compute units available
        NUM_CU = torch.cuda.get_device_properties("cuda").multi_processor_count

        if metadata.persistent is not None:
            grid = lambda META: (min(NUM_CU * META['GRID_CU_MULTIP'],
                                     triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']) * nheads_q * batch), )
        else:
            grid = lambda META: (triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']), nheads_q, batch)

        atomic_counter = torch.zeros([1], device=q.device, dtype=torch.int32)

        attn_fwd[grid](q, k, v, metadata.bias, metadata.sm_scale, M, o, *q_strides, *k_strides, *v_strides, *o_strides,
                       *bias_strides, *alibi_strides, q_descale, k_descale, p_scale, p_descale, v_descale,
                       metadata.cu_seqlens_q, metadata.cu_seqlens_k, dropout_p=metadata.dropout_p,
                       philox_seed=philox_seed, philox_offset_base=philox_offset, encoded_softmax=encoded_softmax,
                       alibi_slopes=metadata.alibi_slopes, HQ=nheads_q, HK=nheads_k, ACTUAL_BLOCK_DMODEL=head_size,
                       MAX_SEQLENS_Q=metadata.max_seqlens_q, MAX_SEQLENS_K=metadata.max_seqlens_k,
                       IS_CAUSAL=metadata.causal, VARLEN=metadata.varlen, BLOCK_DMODEL=padded_d_model,
                       USE_BIAS=False if metadata.bias is None else True,
                       USE_ALIBI=False if metadata.alibi_slopes is None else True, ENABLE_DROPOUT=metadata.dropout_p
                       > 0.0, RETURN_ENCODED_SOFTMAX=metadata.return_encoded_softmax, INT8=metadata.int8,
                       USE_P_SCALE=metadata.int8 and metadata.use_p_scale, INT8_KV=metadata.int8 and metadata.int8_kv,
                       PERSISTENT=metadata.persistent is not None, PERSISTENT_DYNAMIC=metadata.persistent == "dynamic",
                       NUM_CU=NUM_CU, atomic_counter=atomic_counter, B=batch)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = metadata.sm_scale
        ctx.BLOCK_DMODEL = head_size
        ctx.causal = metadata.causal
        ctx.alibi_slopes = metadata.alibi_slopes
        ctx.dropout_p = metadata.dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax
        ctx.return_encoded_softmax = metadata.return_encoded_softmax
        return o, encoded_softmax, attn_fwd.best_config

    @staticmethod
    def backward(ctx, *gradients):
        do = gradients[0]
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        seqlen_q = q.shape[2]
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        # NUM_WARPS, NUM_STAGES = 4, 1
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        assert N_CTX % PRE_BLOCK == 0
        delta = torch.empty_like(M)
        _, Lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]
        # padded_head = (Lk != ctx.BLOCK_DMODEL)
        grid_preprocess = (triton.cdiv(do.shape[2], BLOCK), do.shape[1], do.shape[0])
        _attn_bwd_preprocess[grid_preprocess](
            o,
            do,
            delta,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            seqlen_q,
            head_dim=Lk,
            BLOCK_M=BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_N1']), 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            ctx.alibi_slopes,
            do,
            dq,
            dk,
            dv,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            N_HEAD,
            N_CTX,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            USE_ALIBI=False if ctx.alibi_slopes is None else True,
        )

        return dq, dk, dv, None, None


attention = _attention.apply

INT8_MAX = 127


def quantize_int8(tensor: torch.Tensor, dim) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_vals = tensor.abs().amax(dim=[i for i in range(tensor.dim()) if i != dim], keepdim=True)

    # Avoid division by zero
    max_vals[max_vals == 0] = 1e-8

    # Compute scale factors for each channel
    scale = INT8_MAX / max_vals.to(torch.float32)

    # Quantize the tensor
    tensor = tensor * scale
    tensor = tensor.round_()
    tensor.clamp_(-INT8_MAX, INT8_MAX)
    tensor_quantized = tensor.to(torch.int8)

    return tensor_quantized, scale, 1 / scale


def quantize_input(q, k, v, input_metadata: MetaData, quantize_p=False, int8_kv=False):
    assert not (quantize_p and int8_kv)
    if input_metadata.layout == 'bhsd':
        qunatization_dim = 1
    elif input_metadata.layout == 'bshd':
        qunatization_dim = 2
    else:
        assert False, 'Got unsupported tensor layout'
    assert not (quantize_p and int8_kv)

    q_descale = None
    if not int8_kv:
        q, _, q_descale = quantize_int8(q, dim=qunatization_dim)
    k, _, k_descale = quantize_int8(k, dim=qunatization_dim)
    v, _, v_descale = quantize_int8(v, dim=qunatization_dim)

    # In real world use case, the p scale would be a parameter trained by the model.
    p_scale = p_descale = None
    # The p shape is always bhqk
    if quantize_p:
        _, nheads_q, _, _ = get_shape_from_layout(q, k, input_metadata)
        p_scale = torch.full((1, nheads_q, 1, 1), 127, dtype=torch.float32, device="cuda")
        p_descale = 1 / p_scale

    # We are not multiplying the scales togather to get qk_desale / o_descale e.g.
    # qk_desale = q_descale * k_descale
    # o_desale = p_descale * v_descale
    # it results in very small fp e.g. 0,0002, losing precision. They are applied on the run.
    input_metadata.set_int8_params(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                                   # By default p_scaling is not enabled
                                   p_scale=p_scale, p_descale=p_descale)

    return q, k, v


def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, requires_grad=True):
    torch.manual_seed(20)

    # Initialize q, k, v
    if layout == 'bhsd':
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == 'bshd':
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)
    else:
        assert False, 'Got unsupported tensor layout'
    q = torch.randn(q_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
    k = torch.randn(k_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
    v = torch.randn(k_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)

    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    return q, k, v, input_metadata


def varlen_input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, equal_seqlens=False):
    torch.manual_seed(20)

    # Random sequence lengths. Using N_CTX as kind of max of sum of individual seqs
    if not equal_seqlens:
        max_seqlens_q = N_CTX_Q // Z
        max_seqlens_k = N_CTX_K // Z
        if N_CTX_Q == N_CTX_K:
            seqlens_q = torch.randint(1, max_seqlens_q + 1, (Z, ), dtype=torch.int32)
            seqlens_k = seqlens_q
        else:
            seqlens_q = torch.randint(1, max_seqlens_q + 1, (Z, ), dtype=torch.int32)
            seqlens_k = torch.randint(1, max_seqlens_k + 1, (Z, ), dtype=torch.int32)
    else:
        seqlens_q = torch.full((Z, ), N_CTX_Q // Z)
        seqlens_k = torch.full((Z, ), N_CTX_K // Z)

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_q.cumsum(dim=0, dtype=torch.int32)])
    cu_seqlens_k = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_k.cumsum(dim=0, dtype=torch.int32)])
    cu_seqlens_q = cu_seqlens_q.to(device="cuda")
    cu_seqlens_k = cu_seqlens_k.to(device="cuda")

    # Initialize q, k, v with variable lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    q = torch.randn((total_q, HQ, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.randn((total_k, HK, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.randn((total_k, HK, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)
    return q, k, v, input_metadata


@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 24, 1024, 1024, 64),
    (1, 24, 6, 8192, 8192, 64),
    (1, 4, 2, 16384, 16384, 128),
    (2, 16, 4, 1020, 987, 128),
    (2, 16, 4, 15498, 2, 128),
    (2, 16, 2, 7, 16219, 64),
    (4, 48, 12, 1, 1, 64),
    (4, 48, 48, 1, 1, 128),
    (4, 48, 24, 3, 3, 128),
    (4, 48, 48, 1001, 990, 64),
    (1, 8, 8, 8081, 7099, 64),
    (1, 4, 4, 16330, 15989, 128),
    (4, 4, 1, 1024, 1024, 33),
    (4, 4, 2, 65, 1018, 65),
    (4, 4, 4, 128, 128, 65),
    (4, 4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_alibi', [True, False])
@pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
def test_op_fwd(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_alibi, layout, dtype=torch.float16):
    torch.manual_seed(20)
    q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout)
    if causal:
        input_metadata.need_causal()

    if use_alibi:
        # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
        alibi_slopes = torch.tensor([2**(-8 / HQ * i) for i in range(1, HQ + 1)], dtype=torch.float32,
                                    device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, HQ)
    else:
        alibi_slopes = None

    o = torch.empty_like(q)

    # triton implementation
    tri_out, _, _ = attention(q, k, v, o, input_metadata)

    # Transpose here if layout is bshd so we have same reference code for all layouts
    if layout == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()
    # Replicate K and V if using MQA/GQA
    if HQ != HK:
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[2],
                   k.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(k.shape[0], -1, k.shape[2], k.shape[3])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[2],
                   v.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(v.shape[0], -1, v.shape[2], v.shape[3])

    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * input_metadata.sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    if use_alibi:
        scores += compute_alibi_tensor(alibi_slopes, N_CTX_Q, N_CTX_K)

    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
        # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
        # this by converting the NaNs to 0s, which is what they should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    if layout == 'bshd':
        ref_out = ref_out.transpose(1, 2).clone()
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 24, 1024, 1024, 64),
    (1, 24, 6, 8192, 8192, 64),
    (1, 4, 2, 16384, 16384, 128),
    (2, 16, 4, 1020, 987, 128),
    (2, 16, 4, 15498, 2, 128),
    (2, 16, 2, 7, 16219, 64),
    (4, 48, 12, 1, 1, 64),
    (4, 48, 48, 1, 1, 128),
    (4, 48, 24, 3, 3, 128),
    (4, 48, 48, 1001, 990, 64),
    (1, 8, 8, 8081, 7099, 64),
    (1, 4, 4, 16330, 15989, 128),
    (4, 4, 1, 1024, 1024, 33),
    (4, 4, 2, 65, 1018, 65),
    (4, 4, 4, 128, 128, 65),
    (4, 4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_alibi', [True, False])
@pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
@pytest.mark.parametrize('persistent', ['fixed', 'dynamic'])
def test_op_persistent_fwd(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_alibi, layout, persistent,
                           dtype=torch.float16):
    torch.manual_seed(20)
    q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout)
    if causal:
        input_metadata.need_causal()

    if use_alibi:
        # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
        alibi_slopes = torch.tensor([2**(-8 / HQ * i) for i in range(1, HQ + 1)], dtype=torch.float32,
                                    device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, HQ)
    else:
        alibi_slopes = None

    input_metadata.set_persistent(persistent)

    o = torch.empty_like(q)

    # triton implementation
    tri_out, _, _ = attention(q, k, v, o, input_metadata)

    # Transpose here if layout is bshd so we have same reference code for all layouts
    if layout == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()
    # Replicate K and V if using MQA/GQA
    if HQ != HK:
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[2],
                   k.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(k.shape[0], -1, k.shape[2], k.shape[3])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[2],
                   v.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(v.shape[0], -1, v.shape[2], v.shape[3])

    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * input_metadata.sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    if use_alibi:
        scores += compute_alibi_tensor(alibi_slopes, N_CTX_Q, N_CTX_K)

    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
        # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
        # this by converting the NaNs to 0s, which is what they should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    if layout == 'bshd':
        ref_out = ref_out.transpose(1, 2).clone()
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1024, 1024, 64),
    (4, 12, 8192, 8192, 64),
    (2, 4, 16384, 16384, 128),
    (2, 16, 1020, 987, 128),
    (2, 4, 7, 16219, 64),
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 48, 1001, 990, 64),
    (1, 8, 8081, 7099, 64),
    (1, 8, 16330, 15989, 128),
    (4, 4, 1024, 1024, 33),
    (4, 4, 65, 1019, 65),
    (4, 4, 128, 128, 65),
    (4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('quantize_p', [True, False])
@pytest.mark.parametrize('layout', ['bhsd'])
def test_op_fwd_int8(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, quantize_p, layout, dtype=torch.float16):
    torch.manual_seed(20)

    # Disable grad to save memeory it won't run into OOM on CI machine.
    q, k, v, input_metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, requires_grad=False)
    if causal:
        input_metadata.need_causal()

    o = torch.empty_like(q)

    q_quantized, k_quantized, v_quantized = quantize_input(q, k, v, input_metadata, quantize_p=quantize_p)

    tri_out, _, best_configs = attention(q_quantized, k_quantized, v_quantized, o, input_metadata)

    # Compute scores
    q_descale, k_descale, v_descale = input_metadata.q_descale, input_metadata.k_descale, input_metadata.v_descale
    scores = (torch.einsum('bhqd,bhkd->bhqk', q_quantized.half(), k_quantized.half()) * q_descale *
              k_descale) * input_metadata.sm_scale

    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")

    # Quantization with tiling
    if quantize_p:
        tile_size = best_configs.kwargs["BLOCK_N"]  # We need the tiling to match Block_N to work
        m_i = torch.full((Z, H, N_CTX_Q), float('-inf'), device='cuda', dtype=torch.float32)
        acc = torch.zeros((Z, H, N_CTX_Q, D_HEAD), device='cuda', dtype=torch.float32)
        l_i = torch.zeros_like(m_i)

        for i in range(0, N_CTX_K, tile_size):
            qk_tile = scores[:, :, :, i:i + tile_size]
            v_tile = v_quantized[:, :, i:i + tile_size]
            m_ij = torch.max(m_i, torch.max(qk_tile, dim=-1).values)
            qk_tile -= m_ij.unsqueeze(-1)
            p_tile = torch.exp(qk_tile)
            l_ij = torch.sum(p_tile, dim=-1)
            p_tile = (p_tile * input_metadata.p_scale).to(torch.int8)

            alpha = torch.exp(m_i - m_ij)
            # We need float here since both p and v are quantized. So they might overflow the fp16 range.
            acc = acc * alpha.unsqueeze(-1) + torch.einsum('bhqk,bhkd->bhqd', p_tile.float(), v_tile.float())
            m_i = m_ij
            l_i = alpha * l_i + l_ij

        l_recip = 1 / l_i.unsqueeze(-1)
        acc = acc * input_metadata.p_descale * input_metadata.v_descale * l_recip
        ref_out = acc.to(torch.float16)
    else:
        p = torch.softmax(scores, dim=-1)
        ref_out = (torch.einsum('bhqk,bhkd->bhqd', p.float(), v_quantized.float()) * v_descale).to(torch.float16)

    if causal:
        nan_mask = torch.isnan(ref_out)
        ref_out[nan_mask] = 0

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1024, 1024, 64),
    (4, 12, 8192, 8192, 64),
    (2, 4, 16384, 16384, 128),
    (2, 16, 1020, 987, 128),
    (2, 4, 7, 16219, 64),
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 48, 1001, 990, 64),
    (1, 8, 8081, 7099, 64),
    (1, 8, 16330, 15989, 128),
    (4, 4, 1024, 1024, 33),
    (4, 4, 65, 1019, 65),
    (4, 4, 128, 128, 65),
    (4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('layout', ['bhsd'])
def test_op_fwd_int8_kv(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, layout, dtype=torch.float16):
    torch.manual_seed(20)

    q, k, v, input_metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout)
    if causal:
        input_metadata.need_causal()

    o = torch.empty_like(q)

    _, k_quantized, v_quantized = quantize_input(q, k, v, input_metadata, int8_kv=True)
    k_descale, v_descale = input_metadata.k_descale, input_metadata.v_descale
    k_dequantized = (k_quantized * k_descale).half()
    v_dequantized = (v_quantized * v_descale).half()

    tri_out, _, _ = attention(q, k_quantized, v_quantized, o, input_metadata)

    # Compute scores
    scores = torch.einsum('bhqd,bhkd->bhqk', q, k_dequantized).float() * input_metadata.sm_scale

    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")

    p = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v_dequantized).to(torch.float16)

    if causal:
        nan_mask = torch.isnan(ref_out)
        ref_out[nan_mask] = 0

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1024, 1024, 64),
    (4, 12, 8192, 8192, 64),
    (2, 4, 16384, 16384, 128),
    (2, 16, 1020, 987, 128),
    (2, 4, 7, 16219, 64),
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 48, 1001, 990, 64),
    (1, 8, 8081, 7099, 64),
    (1, 8, 16330, 15989, 128),
    (4, 4, 1024, 1024, 33),
    (4, 4, 65, 1019, 65),
    (4, 4, 128, 128, 65),
    # TODO: This config fails. Disabled until triaged and fixed.
    # (4, 4, 113, 123, 1),
    # (2, 16, 15498, 2, 128),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_bias', [True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_op_fwd_bias(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_bias, dtype):
    torch.manual_seed(20)
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    q, k, v, input_metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout='bhsd')
    if causal:
        input_metadata.need_causal()
    if use_bias:
        bias = torch.randn((1, H, N_CTX_Q, N_CTX_K), dtype=dtype, device="cuda")
        input_metadata.need_bias(bias, Z, H, N_CTX_Q, N_CTX_K)
    else:
        bias = None
    o = torch.empty_like(q)

    # triton implementation
    tri_out, _, _ = attention(q, k, v, o, input_metadata)
    # reference implementation:171

    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    if use_bias:
        scores += input_metadata.bias
    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
        # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
        # this by converting the NaNs to 0s, which is what they should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0

    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.to(dtype), v)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(4, 48, 8192, 64), (4, 48, 256, 64), (4, 48, 512, 64),
                                                 (4, 48, 1024, 64), (8, 48, 4096, 64), (4, 48, 8192, 64),
                                                 (4, 48, 128, 128), (4, 48, 4096, 128), (4, 48, 16384, 128),
                                                 (4, 16, 1024, 128), (4, 16, 8192, 128), (32, 48, 8192, 128)])
@pytest.mark.parametrize('causal', [True, False])
def test_op_varlen_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):

    q, k, v, input_metadata = varlen_input_helper(Z, H, H, N_CTX, N_CTX, D_HEAD, dtype)

    tri_out = torch.empty_like(q)
    ref_out = torch.empty_like(q)

    for i in range(0, input_metadata.num_contexts):
        start_q, start_k = input_metadata.cu_seqlens_q[i], input_metadata.cu_seqlens_k[i]
        end_q, end_k = input_metadata.cu_seqlens_q[i + 1], input_metadata.cu_seqlens_k[i + 1]
        scores = torch.einsum('qhd,khd->qhk', q[start_q:end_q], k[start_k:end_k]).float()
        p = torch.softmax(scores * input_metadata.sm_scale, dim=-1).half()
        ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p, v[start_k:end_k])
    attention(q, k, v, tri_out, input_metadata)
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize('Z, HQ, HK, N_CTX, D_HEAD', [(2, 48, 24, 128, 64), (4, 48, 12, 256, 64), (4, 48, 4, 512, 64),
                                                      (4, 48, 2, 1024, 64), (8, 48, 6, 4096, 64), (4, 48, 8, 16384, 64),
                                                      (4, 64, 16, 128, 128), (4, 64, 4, 4096, 128),
                                                      (4, 64, 8, 16384, 128), (4, 16, 4, 1024, 128),
                                                      (4, 16, 2, 8192, 128), (32, 128, 32, 8192, 128)])
@pytest.mark.parametrize('causal', [False])
def test_op_varlen_mqa_fwd(Z, HQ, HK, N_CTX, D_HEAD, causal, dtype=torch.float16):
    q, k, v, input_metadata = varlen_input_helper(Z, HQ, HK, N_CTX, N_CTX, D_HEAD, dtype)
    ref_out = torch.empty_like(q)
    tri_out = torch.empty_like(q)
    # Make KV look like HQ/HK "groups" of HK. Later, we will reshape so the
    # size aligns with Q.
    k_ref = k.view(k.shape[0], k.shape[1], 1, k.shape[2]).expand(-1, -1, HQ // HK, -1)
    v_ref = v.view(v.shape[0], v.shape[1], 1, v.shape[2]).expand(-1, -1, HQ // HK, -1)
    for i in range(0, input_metadata.num_contexts):
        start_q, start_k = input_metadata.cu_seqlens_q[i], input_metadata.cu_seqlens_k[i]
        end_q, end_k = input_metadata.cu_seqlens_q[i + 1], input_metadata.cu_seqlens_k[i + 1]
        k_curr = k_ref[start_k:end_k]
        k_curr = k_curr.reshape(k_curr.shape[0], -1, k_curr.shape[3])
        v_curr = v_ref[start_k:end_k]
        v_curr = v_curr.reshape(v_curr.shape[0], -1, v_curr.shape[3])
        scores = torch.einsum('qhd,khd->qhk', q[start_q:end_q], k_curr).float()
        p = torch.softmax(scores * input_metadata.sm_scale, dim=-1).half()
        ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p, v_curr)
    attention(q, k, v, tri_out, input_metadata)
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [
    (4, 48, 1024, 64),
    (4, 48, 2048, 64),
    (2, 48, 4096, 64),
    (1, 16, 1024, 64),
    (1, 16, 1024, 128),
    #(1, 16, 8192, 63),
    #(1, 16, 1022, 64),
])
@pytest.mark.parametrize('qseqlen_not_equal_kseqlen', [None])
@pytest.mark.parametrize('torch_sdpa_test', [False, True])
@pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize('use_alibi', [False, True])
def test_op_bwd(Z, H, N_CTX, D_HEAD, qseqlen_not_equal_kseqlen, causal, torch_sdpa_test, use_alibi,
                dtype=torch.float16):
    pytest.skip()
    torch.manual_seed(20)
    if qseqlen_not_equal_kseqlen is not None:
        seqlen_q = qseqlen_not_equal_kseqlen
    else:
        seqlen_q = N_CTX
    seqlen_k = N_CTX

    if causal and ((N_CTX - 1) & N_CTX):
        pytest.skip()
    if causal and seqlen_q != seqlen_k:
        pytest.skip()

    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = seqlen_q
    input_metadata.max_seqlens_k = seqlen_k

    dropout_p = 0
    q = (torch.empty((Z, H, seqlen_q, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    o = torch.empty_like(q)

    if causal:
        input_metadata.need_causal()

    if use_alibi and not torch_sdpa_test:
        # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
        alibi_slopes = torch.tensor([2**(-8 / H * i) for i in range(1, H + 1)], dtype=torch.float32,
                                    device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, H)
    dout = torch.randn_like(q)
    # reference implementation
    if torch_sdpa_test:
        ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q, k, v, dropout_p=dropout_p,
                                                                                 is_causal=causal, scale=sm_scale,
                                                                                 dropout_mask=None)
        ref_out.backward(dout.to(device=ref_out.device, dtype=ref_out.dtype))
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
    else:
        M = torch.tril(torch.ones((seqlen_q, seqlen_k), device="cuda"))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if use_alibi:
            p += compute_alibi_tensor(alibi_slopes, N_CTX, N_CTX)
        if causal:
            p[:, :, M == 0] = float("-inf")

        p = torch.softmax(p.float(), dim=-1).type(dtype=p.dtype)
        ref_out = torch.matmul(p, v)
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None

    # # triton implementation
    tri_out, _, _ = attention(q, k, v, o, input_metadata)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # test
    #print("reference")
    #print(ref_dv)
    #print("tri")
    #print(tri_dv)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    # The current block size for gfx90a and gfx908 series is 64x64. This results in
    # larger differences in float results due to rounding.

    if dtype == torch.bfloat16:
        ATOL = 1e-1 * max(1.0, (seqlen_q + D_HEAD) / 64.0)
    if dtype == torch.float32:
        ATOL = 1e-3 * max(1.0, (seqlen_q + D_HEAD) / 64.0)
    else:
        ATOL = 1e-1 * max(1.0, (seqlen_q + D_HEAD) / 64.0)

    RTOL = 0

    torch.testing.assert_close(ref_dv, tri_dv, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(ref_dk, tri_dk, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(ref_dq, tri_dq, atol=ATOL, rtol=RTOL)


def nonvarlen_benchmark_configs():
    configs = [
        (16, 16, 16, 1024, 1024),
        (8, 16, 16, 2048, 2048),
        (4, 16, 16, 4096, 4096),
        (2, 16, 16, 8192, 8192),
        (1, 16, 16, 16384, 16384),
        (2, 48, 48, 1024, 1024),
        (2, 48, 48, 2048, 1024),
        (2, 48, 48, 4096, 8192),
        (2, 48, 48, 8192, 4096),
        (2, 48, 48, 16384, 8192),
        (8, 16, 16, 1989, 15344),
        (4, 16, 16, 4097, 163),
        (2, 16, 16, 8122, 2159),
        (1, 16, 16, 16281, 7),
        (2, 48, 48, 1021, 1020),
        (2, 48, 48, 2001, 2048),
        (2, 48, 48, 3996, 9639),
        (2, 48, 48, 8181, 1021),
    ]
    return configs


def varlen_benchmark_configs():
    configs = [
        (2, 16, 4, 1024, 1024),
        (8, 16, 2, 2048, 2048),
        (4, 16, 8, 4096, 4096),
        (2, 16, 4, 8192, 8192),
        (2, 16, 8, 16384, 16384),
        (2, 48, 12, 1024, 1024),
        (2, 48, 24, 2048, 2048),
        (2, 48, 8, 4096, 4096),
        (2, 48, 4, 8192, 8192),
        (2, 48, 2, 16384, 16384),
        (2, 64, 32, 1024, 1024),
        (4, 64, 16, 2048, 2048),
        (4, 64, 8, 4096, 4096),
        (4, 64, 32, 8192, 8192),
        (4, 128, 16, 16384, 16384),
    ]
    return configs


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, model_families=["llama3"], model=args.model)
    fa_configs = []
    batch_size = args.b if args.b else 1

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        HK = HQ if config["num_key_value_heads"] is None else config["num_key_value_heads"]
        N_CTX_Q = args.sq if args.sq else 8192
        N_CTX_K = args.sk if args.sk else N_CTX_Q
        HEAD_DIM = config["hidden_size"] // HQ
        fa_configs.append((model_name, batch_size, HQ, HK, N_CTX_Q, N_CTX_K, HEAD_DIM))

    return fa_configs


def run_benchmark(custom, args):

    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    mode = 'fwd'
    x_names = ['BATCH', 'HQ', 'HK', 'N_CTX_Q', 'N_CTX_K']
    causal = args.causal if not args.model else True
    int8 = args.int8
    quantize_p = args.quantize_p and int8
    int8_kv = args.int8_kv and int8
    varlen = True if args.model else args.layout == 'thd'
    configs = []
    plot_name = f'fused-attention-{mode}-d{head_size}-layout{args.layout}'
    extra_args = {'D_HEAD': head_size, 'dtype': dtype, 'causal': causal, 'mode': mode}
    if custom:
        x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    else:
        if varlen:
            x_vals_list = varlen_benchmark_configs()
        else:
            x_vals_list = nonvarlen_benchmark_configs()

        if mode == 'bwd':
            # Only those with N_CTX_Q == N_CTX_K work
            new_x = []
            for v in x_vals_list:
                if v[-1] == v[-2]:
                    new_x.append(v)
            x_vals_list = new_x

        if args.model:
            x_vals_list = model_benchmark_configs(args)
            x_names = ['model', 'BATCH', 'HQ', 'HK', 'N_CTX_Q', 'N_CTX_K', 'D_HEAD']
            plot_name = f'fused-attention-{mode}-layout{args.layout}'
            extra_args = {'dtype': dtype, 'causal': causal, 'mode': mode}

    print_time = args.return_time
    line_vals = ['triton', 'torch']  # 'Time (ms)' if print_time else 'TFLOPS'
    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('green', '-'), ('red', '-')],
                                 ylabel='Time (ms)' if print_time else 'TFLOPS', plot_name=plot_name, args=extra_args))

    @triton.testing.perf_report(configs)
    def bench_flash_attention(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, causal, mode, provider, device="cuda",
                              model=None):
        assert mode in ["fwd", "bwd"]
        assert not (int8_kv and quantize_p)
        warmup = 25
        rep = 100
        # TODO: Enable bias after testing.
        # if use_bias:
        #     bias = torch.randn((1, H, N_CTX, N_CTX), dtype=torch.float32, device="cuda")
        #     input_metadata.need_bias(bias, BATCH, H, N_CTX, N_CTX)
        # else:
        #     bias = None
        # bias = None

        # Bwd pass only supports causal=True right now
        if mode == 'bwd':
            causal = True

        flops_per_matmul = 0
        if varlen:
            q, k, v, input_metadata = varlen_input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype,
                                                          args.equal_seqlens)
            for i in range(0, input_metadata.num_contexts):
                seqlen_q = (input_metadata.cu_seqlens_q[i + 1] - input_metadata.cu_seqlens_q[i]).item()
                seqlen_k = (input_metadata.cu_seqlens_k[i + 1] - input_metadata.cu_seqlens_k[i]).item()
                # x2 in both cases for 2 GEMMs
                if causal:
                    # If seqlen_q != seqlen_k then the causal mask ignores computation
                    # depending on which seqlen is larger. Either the lower triangle, or right triangle
                    # If seqlen_q is greater than seqlen_k, the lower triangle is non zero
                    # where the last row has seqlen_k valid element, the second last row has
                    # seqlen_k - 1 valid elements and so on until one element is valid in the
                    # seqlen_q - seqlen_k row, hence total valid elements are 1+2+...+seqlen_k
                    # which is seqlen_k*(seqlen_k+1)/2
                    # If seqlen_q is less than seqlen_k, then we count the zero elements
                    # the first row has seqlen_q-1 zero elements, the second row has seqlen_q-2
                    # zero elements and so on until the second last row has 1 zero element
                    # Total zero elements are 1+2+...+(seqlen_q-1) = seqlen_q*(seqlen_q-1)/2
                    # Total non zero elements are seqlen_q*seqlen_k - (seqlen_q*(seqlen_q-1)/2)
                    valid_out_elements = ((seqlen_k**2 + seqlen_k) / 2) if seqlen_q > seqlen_k else \
                            (seqlen_q * seqlen_k - ((seqlen_q**2 - seqlen_q) / 2))
                    flops_per_matmul += valid_out_elements * HQ * D_HEAD * 2
                else:
                    flops_per_matmul += seqlen_q * seqlen_k * HQ * D_HEAD * 2
        else:
            q, k, v, input_metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, args.layout)
            if causal:
                # Same calculation as if varlen/if causal above
                valid_out_elements = ((N_CTX_K**2 + N_CTX_K) / 2) if N_CTX_Q > N_CTX_K else \
                        (N_CTX_Q * N_CTX_K - ((N_CTX_Q**2 - N_CTX_Q) / 2))
                flops_per_matmul = 2.0 * BATCH * HQ * valid_out_elements * D_HEAD
            else:
                flops_per_matmul = 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * D_HEAD
        if causal:
            input_metadata.need_causal()

        if "triton" in provider:
            o = torch.empty_like(q)
            if int8:
                q, k, v = quantize_input(q, k, v, input_metadata, quantize_p=quantize_p, int8_kv=int8_kv)
            input_metadata.set_persistent(args.persistent)
            fn = lambda: attention(q, k, v, o, input_metadata)
            if mode == 'bwd':
                o, _, _ = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)

        elif "torch" in provider and args.layout in ["thd", "bhsd", "bshd"]:
            # torch requires the layout to be (b (optional),...,h,s,d)
            if args.layout in ["thd", "bshd"]:
                q = q.transpose(-3, -2)
                k = k.transpose(-3, -2)
                v = v.transpose(-3, -2)
            # check if GQA
            HQ = q.shape[-3]
            HK = k.shape[-3]
            if HQ != HK:  # TODO: sdpa(..., enable_gqa=True work) should work
                k = k.repeat_interleave(q.size(-3) // k.size(-3), -3)
                v = v.repeat_interleave(q.size(-3) // v.size(-3), -3)

            fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=input_metadata.sm_scale)
        else:
            assert False, f"Unknown provider {provider} in flash-attention."

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        total_flops = 2 * flops_per_matmul
        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        if print_time:
            return ms
        else:
            return total_flops / ms * 1e-9

    bench_flash_attention.run(save_path=".", print_data=True, show_plots=True)


def supported_layouts():
    layouts = \
        'bhsd: Q, K, V are individual tensors of [batch, num_heads, seqlen_q/k, head_size]' \
        'bshd: Q, K, V are individual tensors of [batch, seqlen_q/k, num_heads, head_size]' \
        'thd: Q, K, V are individual tensors of [total_q/k, num_heads, head_size]' \
        'This layout is sometimes called "varlen" or "grouped" layout.'
    return layouts


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark FlashAttention",
        allow_abbrev=False,
    )
    parser.add_argument('-model_configs', type=str, default="model_configs.json", help="Model config json file.")

    available_models = get_available_models(model_families=["llama3"])  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: [" + ", ".join(available_models) +
        "]. Use 'all' to benchmark all models. Not providing runs the default benchmark script with custom configs.")
    parser.add_argument('-model', type=str, default=None, help=model_help)
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument("-equal_seqlens", action='store_true', default=False,
                        help='If specified, each context within the thd layout' \
                            ' has same seqlen as sq and sk')
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-causal", action='store_true', default=False)
    parser.add_argument("-int8", action='store_true', default=False)
    parser.add_argument("-quantize_p", action='store_true', default=False)
    parser.add_argument("-int8_kv", action='store_true', default=False)
    parser.add_argument("-dtype", default='fp16')
    parser.add_argument("-return_time", action='store_true', default=False)
    parser.add_argument("-layout", type=str, default='bhsd', help=supported_layouts())
    parser.add_argument(
        "-persistent", nargs='?', const='fixed', choices=['fixed', 'dynamic'], default=None,
        help="Enable persistent kernels. Use '-persistent dynamic' for dynamic scheduling of the tiles.")
    return parser.parse_args()


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def main():
    args = parse_args()
    custom_config = False
    assert args.layout == 'thd' or not args.equal_seqlens or args.model, \
           "Equal sequence lengths arg must be used with the thd layout or a model config."
    if args.hq or args.hk or args.d:
        custom_config = True
        assert args.b and args.hq and args.sq and args.d, \
               "If custom config is specified, please provide \
                all of batch, number of Q heads, Q sequence length \
                and head size."

    if args.model:
        assert not (args.hq or args.hk or args.d), \
                "Specifying model fixes hq, hk and d already. Do not provide them!"

    assert args.dtype in arg_to_torch_dtype, \
           "Only fp16, bf16 and f32 types currently supported."

    if args.model:
        print("Note: Model config sets causal masking and THD layout (varlen) by default.")

    run_benchmark(custom_config, args)


if __name__ == '__main__':
    sys.exit(main())
