"""
This file implements a BSHD Flash Attention and tests against torch reference.
"""

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import pytest


def compute_split_factor(batch, num_q_heads, seq_len_k):
    target_total_wrkgrps = 1024
    tasks = batch * num_q_heads
    if tasks == 0: return 1
    ideal_split = target_total_wrkgrps // tasks
    min_chunk_size = 64  # arbitrary lower bound
    max_possible_splits = (seq_len_k + min_chunk_size - 1) // min_chunk_size

    split_factor = min(ideal_split, max_possible_splits)
    split_factor = max(split_factor, 1)
    return split_factor


# Handle imports for both pytest (module context) and direct execution
try:
    from .gfx1250_utils import static_profile
except ImportError:
    from gfx1250_utils import static_profile


@gluon.aggregate
class AttentionConfig:
    SEQLEN_Q: gl.constexpr
    SEQLEN_K: gl.constexpr
    HEAD_SZ: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_BUFFERS: gl.constexpr

    qk_layout: gl.constexpr
    pv_layout: gl.constexpr

    k_smem_layout: gl.constexpr
    v_smem_layout: gl.constexpr

    q_layout: gl.constexpr
    k_layout: gl.constexpr
    v_layout: gl.constexpr
    p_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS):

        # constants
        self.SEQLEN_Q = gl.constexpr(SEQLEN_Q)
        self.SEQLEN_K = gl.constexpr(SEQLEN_K)
        self.HEAD_SZ = gl.constexpr(HEAD_SZ)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_BUFFERS = gl.constexpr(NUM_BUFFERS)

        assert NUM_WARPS == 4 or NUM_WARPS == 8
        if NUM_WARPS == 4:
            warp_bases = [[1, 0], [2, 0]]
        else:
            warp_bases = [[1, 0], [2, 0], [4, 0]]

        # operator layouts
        self.qk_layout = gl.constexpr(
            gl.amd.AMDWMMALayout(3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 32]))
        self.pv_layout = gl.constexpr(
            gl.amd.AMDWMMALayout(3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 32]))

        # tensor layouts
        self.k_smem_layout = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for([[HEAD_SZ, 8]], [BLOCK_N, HEAD_SZ], [1, 0]))
        self.v_smem_layout = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for([[HEAD_SZ, 16]], [BLOCK_N, HEAD_SZ], [1, 0]))

        self.q_layout = gl.constexpr(gl.DotOperandLayout(0, self.qk_layout, 8))
        self.k_layout = gl.constexpr(gl.DotOperandLayout(1, self.qk_layout, 8))
        self.v_layout = gl.constexpr(gl.DotOperandLayout(1, self.pv_layout, 8))
        self.p_layout = gl.constexpr(gl.DotOperandLayout(0, self.pv_layout, 8))


@gluon.aggregate
class AttentionProgram:
    cfg: AttentionConfig

    q: gl.tensor

    k_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    k_buffer: gl.shared_memory_descriptor

    v_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    v_buffer: gl.shared_memory_descriptor

    o_ptr: gl.tensor
    o_offs: gl.tensor
    o_mask: gl.tensor

    sm_scale: gl.constexpr
    rcp_ln2: gl.constexpr
    sm_scale_dot_rcp_ln2: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, cfg,  #
                 q,  #
                 k_desc, k_buffer,  #
                 v_desc, v_buffer,  #
                 o_ptr, o_offs, o_mask,  #
                 sm_scale):

        self.cfg = cfg
        self.q = q

        self.k_desc = k_desc
        self.k_buffer = k_buffer
        self.v_desc = v_desc
        self.v_buffer = v_buffer

        self.o_ptr = o_ptr
        self.o_offs = o_offs
        self.o_mask = o_mask

        self.sm_scale = gl.constexpr(sm_scale)
        self.rcp_ln2 = gl.constexpr(1.4426950408889634)
        self.sm_scale_dot_rcp_ln2: gl.constexpr = self.sm_scale * self.rcp_ln2

    @gluon.jit
    def initialize_decode(cfg, q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
                          stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, SM_SCALE):
        off_z = gl.program_id(0)
        off_q_head = gl.program_id(1)
        off_k_head = off_q_head
        off_m = 0  # Decode always processes the first (and only) Q block per instance

        # q [BLOCK_M, HEAD_SZ]
        q_offs = (stride_qz * off_z + stride_qh * off_q_head + stride_qm *
                  (off_m + gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)))[:, None] + stride_qk *
                  (gl.arange(0, cfg.HEAD_SZ, layout=gl.SliceLayout(0, cfg.q_layout)))[None, :])

        # k [HEAD_SZ, BLOCK_N]
        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=k_ptr + stride_kz * off_z + stride_kh * off_k_head,
                                                           shape=(cfg.SEQLEN_K, cfg.HEAD_SZ),
                                                           strides=(stride_kn, stride_kk),
                                                           block_shape=(cfg.BLOCK_N, cfg.HEAD_SZ),
                                                           layout=cfg.k_smem_layout)
        k_buffer = gl.allocate_shared_memory(k_desc.dtype, shape=[2] + k_desc.block_shape, layout=k_desc.layout)

        # v [BLOCK_N, BLOCK_DMODEL]
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=v_ptr + stride_vz * off_z + stride_vh * off_k_head,
                                                           shape=(cfg.SEQLEN_K, cfg.HEAD_SZ),
                                                           strides=(stride_vn, stride_vk),
                                                           block_shape=(cfg.BLOCK_N, cfg.HEAD_SZ),
                                                           layout=cfg.v_smem_layout)
        v_buffer = gl.allocate_shared_memory(v_desc.dtype, shape=[2] + v_desc.block_shape, layout=v_desc.layout)

        q_mask = (off_m + gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)))[:, None] < cfg.SEQLEN_Q
        q = gl.amd.gfx1250.buffer_load(q_ptr, q_offs, mask=q_mask)

        # dummy values for Program struct (unused in Decode FWD)
        o_offs = gl.zeros([cfg.BLOCK_M, cfg.HEAD_SZ], dtype=gl.int32)
        o_mask = gl.zeros([cfg.BLOCK_M, 1], dtype=gl.int1)

        return AttentionProgram(cfg, q, k_desc, k_buffer, v_desc, v_buffer, o_ptr, o_offs, o_mask, SM_SCALE)

    @gluon.jit
    def initialize(cfg,  #
                   q_ptr, k_ptr, v_ptr, o_ptr,  #
                   stride_qz, stride_qh, stride_qm, stride_qk,  #
                   stride_kz, stride_kh, stride_kn, stride_kk,  #
                   stride_vz, stride_vh, stride_vn, stride_vk,  #
                   stride_oz, stride_oh, stride_om, stride_on,  #
                   sm_scale: gl.constexpr):
        SEQLEN_K: gl.constexpr = cfg.SEQLEN_K
        SEQLEN_Q: gl.constexpr = cfg.SEQLEN_Q
        HEAD_SZ: gl.constexpr = cfg.HEAD_SZ
        BLOCK_M: gl.constexpr = cfg.BLOCK_M
        BLOCK_N: gl.constexpr = cfg.BLOCK_N

        # workgroup offsets
        off_z = gl.program_id(0)
        off_q_head = gl.program_id(1)
        off_k_head = off_q_head
        off_m = gl.program_id(2) * BLOCK_M

        # q [BLOCK_M, HEAD_SZ]
        q_offs = (stride_qz * off_z + stride_qh * off_q_head + stride_qm *
                  (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)))[:, None] + stride_qk *
                  (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.q_layout)))[None, :])

        # k [HEAD_SZ, BLOCK_N]
        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(  #
            base=k_ptr + stride_kz * off_z + stride_kh * off_k_head,  #
            shape=(SEQLEN_K, HEAD_SZ),  #
            strides=(stride_kn, stride_kk),  #
            block_shape=(BLOCK_N, HEAD_SZ),  #
            layout=cfg.k_smem_layout)
        k_buffer = gl.allocate_shared_memory(k_desc.dtype, shape=[2] + k_desc.block_shape, layout=k_desc.layout)

        # v [BLOCK_N, BLOCK_DMODEL]
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(  #
            base=v_ptr + stride_vz * off_z + stride_vh * off_k_head,  #
            shape=(SEQLEN_K, HEAD_SZ),  #
            strides=(stride_vn, stride_vk),  #
            block_shape=(BLOCK_N, HEAD_SZ),  #
            layout=cfg.v_smem_layout)
        v_buffer = gl.allocate_shared_memory(v_desc.dtype, shape=[2] + v_desc.block_shape, layout=v_desc.layout)

        q_mask = (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)))[:, None] < SEQLEN_Q
        q = gl.amd.gfx1250.buffer_load(q_ptr, q_offs, mask=q_mask)

        o_offs = (stride_oz * off_z + stride_oh * off_q_head + stride_om *
                  (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)))[:, None] + stride_on *
                  (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.pv_layout)))[None, :])

        o_mask = (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)))[:, None] < SEQLEN_Q

        # create the program
        return AttentionProgram(cfg, q,  #
                                k_desc, k_buffer,  #
                                v_desc, v_buffer,  #
                                o_ptr, o_offs, o_mask,  #
                                sm_scale)

    @gluon.jit
    def tdm_shared_load_k(self, buffer_id, wait_count):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        return self.k_buffer.index(buffer_id).permute([1, 0]).load(layout=self.cfg.k_layout)

    @gluon.jit
    def tdm_shared_load_v(self, buffer_id, wait_count):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        return self.v_buffer.index(buffer_id).load(layout=self.cfg.v_layout)

    @gluon.jit
    def tdm_load_global_to_shared_k(self, offset, buffer_index):
        gl.amd.gfx1250.tdm.async_load(self.k_desc, offset, self.k_buffer.index(buffer_index))

    @gluon.jit
    def tdm_load_global_to_shared_v(self, offset, buffer_index):
        gl.amd.gfx1250.tdm.async_load(self.v_desc, offset, self.v_buffer.index(buffer_index))

    @gluon.jit
    def compute_qk(self, k, cur_seq):
        qk = gl.zeros([self.cfg.BLOCK_M, self.cfg.BLOCK_N], dtype=gl.float32, layout=self.cfg.qk_layout)
        qk = gl.amd.gfx1250.wmma(self.q, k, qk)
        # Handle/pad unaligned M and K2 ids for QK.
        qk_mask = (
            cur_seq +
            gl.arange(0, self.cfg.BLOCK_N, layout=gl.SliceLayout(0, self.cfg.qk_layout)))[None, :] < self.cfg.SEQLEN_K
        qk = gl.where(qk_mask, qk, float("-inf"))
        return qk

    @gluon.jit
    def compute_qk_no_mask(self, k):
        qk = gl.zeros([self.cfg.BLOCK_M, self.cfg.BLOCK_N], dtype=gl.float32, layout=self.cfg.qk_layout)
        qk = gl.amd.gfx1250.wmma(self.q, k, qk)
        return qk

    @gluon.jit
    def softmax_part0(self, qk, m_i):
        # get max scores so far
        m_ij = gl.maximum(m_i, gl.max(qk, 1))
        m_ij_scaled = m_ij * self.sm_scale_dot_rcp_ln2

        # scale and subtract max
        q_shifted = self.sm_scale_dot_rcp_ln2 * qk - m_ij_scaled[:, None]

        # Compute scaled QK and softmax probabilities
        p = gl.exp2(q_shifted)

        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff_scaled = self.sm_scale_dot_rcp_ln2 * m_i - m_ij_scaled
        alpha = gl.exp2(m_diff_scaled)

        return p, alpha, m_ij

    @gluon.jit
    def compute_pv(self, p, v, acc):
        p = gl.convert_layout(p, self.cfg.p_layout)
        return gl.amd.gfx1250.wmma(p, v, acc)

    @gluon.jit
    def softmax_part1(self, p, l_i, acc, alpha):
        # update l_ij before applying dropout
        l_ij = gl.sum(p, 1)

        # update output accumulator
        updated_acc = acc * alpha[:, None]
        updated_p = p.to(gl.bfloat16, fp_downcast_rounding="rtz")

        # Update l_i
        updated_l_i = l_i * alpha + l_ij
        return updated_p, updated_l_i, updated_acc

    @gluon.jit
    def store_output(self, out):
        casted_out = out.to(self.o_ptr.dtype.element_ty)
        gl.amd.gfx1250.buffer_store(casted_out, self.o_ptr, self.o_offs, mask=self.o_mask)


@gluon.jit
def attn_decode_fwd_kernel(q_ptr, k_ptr, v_ptr, mid_o_ptr, mid_l_ptr, mid_m_ptr, stride_qz, stride_qh, stride_qm,
                           stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn,
                           stride_vk, stride_mid_oz, stride_mid_oh, stride_mid_os, stride_mid_om, stride_mid_on,
                           stride_mid_lz, stride_mid_lh, stride_mid_ls, stride_mid_lm, stride_mid_mz, stride_mid_mh,
                           stride_mid_ms, stride_mid_mm, SM_SCALE: gl.constexpr, SEQLEN_Q: gl.constexpr,
                           SEQLEN_K: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_SZ: gl.constexpr,
                           SPLIT_FACTOR: gl.constexpr, CHUNK_SIZE: gl.constexpr):
    NUM_BUFFERS: gl.constexpr = 1
    NUM_WARPS: gl.constexpr = 4
    # use standard Config with Block_M. For decode SEQLEN_Q=1 but we use BLOCK_M tile.
    cfg = AttentionConfig(SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS)

    pgm = AttentionProgram.initialize_decode(cfg, q_ptr, k_ptr, v_ptr, mid_o_ptr, stride_qz, stride_qh, stride_qm,
                                             stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                                             stride_vh, stride_vn, stride_vk, SM_SCALE)

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    l_i = gl.full([BLOCK_M], 0.0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    acc = gl.zeros([BLOCK_M, HEAD_SZ], dtype=gl.float32, layout=cfg.pv_layout)

    split_id = gl.program_id(2)
    start_k = split_id * CHUNK_SIZE
    end_k = min(start_k + CHUNK_SIZE, SEQLEN_K)

    for current_k in range(start_k, end_k, BLOCK_N):
        pgm.tdm_load_global_to_shared_k([current_k, 0], buffer_index=0)
        k = pgm.tdm_shared_load_k(0, wait_count=0)

        qk = pgm.compute_qk(k, current_k)

        # mask out keys that belong to the next split
        # qk is [BLOCK_M, BLOCK_N]
        # We want to mask columns where current_k + col >= end_k
        # re-use qk_layout for the mask
        extra_mask = (current_k + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)))[None, :] < end_k
        qk = gl.where(extra_mask, qk, float("-inf"))

        p, alpha, m_i = pgm.softmax_part0(qk, m_i)
        p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

        pgm.tdm_load_global_to_shared_v([current_k, 0], 0)
        v = pgm.tdm_shared_load_v(0, wait_count=0)

        acc = pgm.compute_pv(p, v, acc)

    off_z = gl.program_id(0)
    off_h = gl.program_id(1)

    mid_o_offs = (off_z * stride_mid_oz + off_h * stride_mid_oh + split_id * stride_mid_os +
                  (gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)))[:, None] * stride_mid_om +
                  (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.pv_layout)))[None, :] * stride_mid_on)

    casted_acc = acc.to(mid_o_ptr.dtype.element_ty)
    gl.amd.gfx1250.buffer_store(casted_acc, mid_o_ptr, mid_o_offs)

    mid_l_base = mid_l_ptr + off_z * stride_mid_lz + off_h * stride_mid_lh + split_id * stride_mid_ls
    mid_m_base = mid_m_ptr + off_z * stride_mid_mz + off_h * stride_mid_mh + split_id * stride_mid_ms

    # offs_vec = gl.arange(0, BLOCK_M) * stride_mid_lm
    offs_vec = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)) * stride_mid_lm

    gl.store(mid_l_base + offs_vec, l_i)
    gl.store(mid_m_base + offs_vec, m_i)


@gluon.jit
def attn_decode_reduce_kernel(mid_o_ptr, mid_l_ptr, mid_m_ptr, out_ptr, stride_mid_oz, stride_mid_oh, stride_mid_os,
                              stride_mid_om, stride_mid_on, stride_mid_lz, stride_mid_lh, stride_mid_ls, stride_mid_lm,
                              stride_mid_mz, stride_mid_mh, stride_mid_ms, stride_mid_mm, stride_oz, stride_oh,
                              stride_om, stride_on, SM_SCALE: gl.constexpr, SPLIT_FACTOR: gl.constexpr,
                              BLOCK_M: gl.constexpr, HEAD_SZ: gl.constexpr, SEQLEN_Q: gl.constexpr,
                              SEQLEN_K: gl.constexpr, BLOCK_N: gl.constexpr):
    # Grid: [BATCH, HEADS, 1]
    off_z = gl.program_id(0)
    off_h = gl.program_id(1)

    # Initialize Global Accumulators
    # Use actual config params
    NUM_BUFFERS: gl.constexpr = 1
    NUM_WARPS: gl.constexpr = 4
    cfg = AttentionConfig(SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS)

    m_global = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    l_global = gl.full([BLOCK_M], 0.0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    acc_global = gl.zeros([BLOCK_M, HEAD_SZ], dtype=gl.float32, layout=cfg.pv_layout)

    rcp_ln2 = 1.4426950408889634

    for s in range(SPLIT_FACTOR):
        # Load m_s, l_s, acc_s

        # Offsets
        off_l_base = off_z * stride_mid_lz + off_h * stride_mid_lh + s * stride_mid_ls
        off_m_base = off_z * stride_mid_mz + off_h * stride_mid_mh + s * stride_mid_ms
        off_o_base = off_z * stride_mid_oz + off_h * stride_mid_oh + s * stride_mid_os

        offs_vec = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)) * stride_mid_lm

        m_s = gl.load(mid_m_ptr + off_m_base + offs_vec)
        l_s = gl.load(mid_l_ptr + off_l_base + offs_vec)

        mid_o_offs = (off_o_base +
                      (gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)))[:, None] * stride_mid_om +
                      (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.pv_layout)))[None, :] * stride_mid_on)

        acc_s = gl.amd.gfx1250.buffer_load(mid_o_ptr, mid_o_offs)
        acc_s = acc_s.to(gl.float32)

        m_new = gl.maximum(m_global, m_s)

        alpha = gl.exp2((m_global - m_new) * SM_SCALE * rcp_ln2)
        beta = gl.exp2((m_s - m_new) * SM_SCALE * rcp_ln2)

        l_global = l_global * alpha + l_s * beta
        acc_global = acc_global * alpha[:, None] + acc_s * beta[:, None]
        m_global = m_new

    l_recip = 1.0 / l_global[:, None]
    acc_global = acc_global * l_recip

    o_offs = (stride_oz * off_z + stride_oh * off_h +
              (gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)))[:, None] * stride_om +
              (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.pv_layout)))[None, :] * stride_on)

    o_mask = (gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)))[:, None] < SEQLEN_Q

    casted_out = acc_global.to(out_ptr.dtype.element_ty)
    gl.amd.gfx1250.buffer_store(casted_out, out_ptr, o_offs, mask=o_mask)


@gluon.jit
def attn_fwd_kernel(q_ptr, k_ptr, v_ptr, out_ptr,  #
                    stride_qz, stride_qh, stride_qm, stride_qk,  #
                    stride_kz, stride_kh, stride_kn, stride_kk,  #
                    stride_vz, stride_vh, stride_vn, stride_vk,  #
                    stride_oz, stride_oh, stride_om, stride_on,  #
                    SM_SCALE: gl.constexpr,  #
                    SEQLEN_Q: gl.constexpr,  #
                    SEQLEN_K: gl.constexpr,  #
                    BLOCK_M: gl.constexpr,  #
                    BLOCK_N: gl.constexpr,  #
                    HEAD_SZ: gl.constexpr,  #
                    ):

    NUM_BUFFERS: gl.constexpr = 1
    NUM_WARPS: gl.constexpr = 4
    cfg = AttentionConfig(SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS)
    pgm = AttentionProgram.initialize(  #
        cfg, q_ptr, k_ptr, v_ptr, out_ptr,  #
        stride_qz, stride_qh, stride_qm, stride_qk,  #
        stride_kz, stride_kh, stride_kn, stride_kk,  #
        stride_vz, stride_vh, stride_vn, stride_vk,  #
        stride_oz, stride_oh, stride_om, stride_on,  #
        SM_SCALE)

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    acc = gl.zeros([BLOCK_M, HEAD_SZ], dtype=gl.float32, layout=cfg.pv_layout)

    n_blocks_n = (SEQLEN_K + BLOCK_N - 1) // BLOCK_N
    block_min = 0
    block_max = n_blocks_n * BLOCK_N
    for block_id in range(block_min, block_max, BLOCK_N):
        pgm.tdm_load_global_to_shared_k([block_id, 0], buffer_index=0)
        k = pgm.tdm_shared_load_k(0, wait_count=0)

        qk = pgm.compute_qk(k, block_id)

        p, alpha, m_i = pgm.softmax_part0(qk, m_i)
        p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

        pgm.tdm_load_global_to_shared_v([block_id, 0], 0)
        v = pgm.tdm_shared_load_v(0, wait_count=0)

        acc = pgm.compute_pv(p, v, acc)

    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    pgm.store_output(acc)


@gluon.jit
def attn_fwd_pipelined_kernel(q_ptr, k_ptr, v_ptr, out_ptr,  #
                              stride_qz, stride_qh, stride_qm, stride_qk,  #
                              stride_kz, stride_kh, stride_kn, stride_kk,  #
                              stride_vz, stride_vh, stride_vn, stride_vk,  #
                              stride_oz, stride_oh, stride_om, stride_on,  #
                              SM_SCALE: gl.constexpr,  #
                              SEQLEN_Q: gl.constexpr,  #
                              SEQLEN_K: gl.constexpr,  #
                              BLOCK_M: gl.constexpr,  #
                              BLOCK_N: gl.constexpr,  #
                              HEAD_SZ: gl.constexpr,  #
                              ):
    NUM_BUFFERS: gl.constexpr = 2
    NUM_WARPS: gl.constexpr = 4
    cfg = AttentionConfig(SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS)
    pgm = AttentionProgram.initialize(  #
        cfg, q_ptr, k_ptr, v_ptr, out_ptr,  #
        stride_qz, stride_qh, stride_qm, stride_qk,  #
        stride_kz, stride_kh, stride_kn, stride_kk,  #
        stride_vz, stride_vh, stride_vn, stride_vk,  #
        stride_oz, stride_oh, stride_om, stride_on,  #
        SM_SCALE)

    ITERS_IN_PROLOGUE_EPILOGUE: gl.constexpr = 3
    n_blocks_n = max((SEQLEN_K + BLOCK_N - 1) // BLOCK_N - ITERS_IN_PROLOGUE_EPILOGUE, 1)

    # Since QK from the final iteration is already peeled into the epilogue,
    # we only need to handle case where SEQLEN_K < ITERS_IN_PROLOGUE_EPILOGUE * BLOCK_N.
    has_remainder: gl.constexpr = SEQLEN_K < (ITERS_IN_PROLOGUE_EPILOGUE) * BLOCK_N
    REMAINDER_PEELED_ITERS = 1
    if has_remainder:
        n_blocks_n = n_blocks_n - REMAINDER_PEELED_ITERS

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    acc = gl.zeros([BLOCK_M, HEAD_SZ], dtype=gl.float32, layout=cfg.pv_layout)

    block_min = 0
    block_max = n_blocks_n * BLOCK_N
    """
    Prologue:
    t = i           t = i+1          t = i+2
    [GLDS_K]
    [LR_K, GLDS_V], [GLDS_K]
    [QK, SM0],      [LR_K, GLDS_V],  [GLDS_K]
    """
    # GLDS_K_t0, GLDS_K_t1, GLDS_V_t0
    pgm.tdm_load_global_to_shared_k([0, 0], buffer_index=0)
    pgm.tdm_load_global_to_shared_k([BLOCK_N, 0], buffer_index=1)
    pgm.tdm_load_global_to_shared_v([0, 0], buffer_index=0)

    # LR_K_t0
    k = pgm.tdm_shared_load_k(0, wait_count=2)

    # QK_t0
    qk = pgm.compute_qk(k, 0)

    # SM0_t0
    p, alpha, m_i = pgm.softmax_part0(qk, m_i)

    # GLDS_K_t2, GLDS_V_t1,
    pgm.tdm_load_global_to_shared_k([2 * BLOCK_N, 0], buffer_index=0)
    pgm.tdm_load_global_to_shared_v([BLOCK_N, 0], buffer_index=1)

    # LR_K_t1
    k = pgm.tdm_shared_load_k(1, wait_count=3)

    iter_id = 0
    # TODO: Re-unroll to factor of 2 to optimize rotating register, once LLVM codegen can handle it better.
    for block_id in range(block_min, block_max, BLOCK_N):
        """
        Steady State (Hot Loop - No Masking):
        t = i              t = i+1         t = i+2         t = i+3
        [SM1, LR_V, PV],   [QK, SM0],    [LR_K, GLDS_V]     [GLDS_K]
        """
        t_1 = block_id + BLOCK_N
        t_2 = block_id + 2 * BLOCK_N
        t_3 = block_id + 3 * BLOCK_N

        # QK, SM1, LR_V (no mask needed - all blocks in hot loop are full)
        qk = pgm.compute_qk_no_mask(k)

        p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

        v = pgm.tdm_shared_load_v(iter_id % NUM_BUFFERS, wait_count=2)

        # GLDS_K
        pgm.tdm_load_global_to_shared_k([t_3, 0], (iter_id + 1) % NUM_BUFFERS)

        # PV, SM0, LR_K
        acc = pgm.compute_pv(p, v, acc)

        p, alpha, m_i = pgm.softmax_part0(qk, m_i)

        k = pgm.tdm_shared_load_k(iter_id % NUM_BUFFERS, wait_count=2)

        # GLDS_V
        pgm.tdm_load_global_to_shared_v([t_2, 0], iter_id % NUM_BUFFERS)
        iter_id += 1
    """
    Final iteration of steady state that requires masking.(if masking is required)
    """
    if has_remainder:
        t_1 = iter_id * BLOCK_N + BLOCK_N
        t_2 = iter_id * BLOCK_N + 2 * BLOCK_N
        t_3 = iter_id * BLOCK_N + 3 * BLOCK_N

        # Process the remainder block with masking
        qk = pgm.compute_qk(k, t_1)

        p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

        v = pgm.tdm_shared_load_v(iter_id % NUM_BUFFERS, wait_count=2)

        # GLDS_K
        pgm.tdm_load_global_to_shared_k([t_3, 0], (iter_id + 1) % NUM_BUFFERS)

        # PV, SM0, LR_K
        acc = pgm.compute_pv(p, v, acc)

        p, alpha, m_i = pgm.softmax_part0(qk, m_i)

        k = pgm.tdm_shared_load_k(iter_id % NUM_BUFFERS, wait_count=2)

        # GLDS_V
        pgm.tdm_load_global_to_shared_v([t_2, 0], iter_id % NUM_BUFFERS)
        iter_id += 1
    """
    Epilogue:
    t = i+1              t = i+2              t = i+3
    [SM1, LR_V, PV],    [QK, SM0],          [LR_K, GLDS_V]
                        [SM1, LR_V, PV],    [QK, SM0]
                                            [SM1, LR_V, PV]
    """
    epilogue_offset = (iter_id - 1) * BLOCK_N
    t_2 = epilogue_offset + 2 * BLOCK_N
    t_3 = epilogue_offset + 3 * BLOCK_N
    # SM1_t1, LR_V_t1, PV_t1
    p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

    v = pgm.tdm_shared_load_v(iter_id % NUM_BUFFERS, wait_count=2)

    acc = pgm.compute_pv(p, v, acc)

    # QK_t2, SM0_t2
    qk = pgm.compute_qk(k, t_2)
    p, alpha, m_i = pgm.softmax_part0(qk, m_i)

    # LR_K_t3, GLDS_V_t3
    k = pgm.tdm_shared_load_k(iter_id % NUM_BUFFERS, wait_count=1)

    pgm.tdm_load_global_to_shared_v([t_3, 0], iter_id % NUM_BUFFERS)

    # QK_t3, SM1_t2, LR_V_t2
    qk = pgm.compute_qk(k, t_3)

    p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

    v = pgm.tdm_shared_load_v((iter_id + 1) % NUM_BUFFERS, wait_count=1)

    # PV_t_2, SM0_t_3, SM1_t_3, LR_V_t3
    acc = pgm.compute_pv(p, v, acc)

    p, alpha, m_i = pgm.softmax_part0(qk, m_i)
    p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

    v = pgm.tdm_shared_load_v(iter_id % NUM_BUFFERS, wait_count=0)

    # PV_t_3
    acc = pgm.compute_pv(p, v, acc)

    # Post loop scaling and output

    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    pgm.store_output(acc)


@gluon.jit
def attn_fwd_pingpong_pipelined_kernel(q_ptr, k_ptr, v_ptr, out_ptr,  #
                                       stride_qz, stride_qh, stride_qm, stride_qk,  #
                                       stride_kz, stride_kh, stride_kn, stride_kk,  #
                                       stride_vz, stride_vh, stride_vn, stride_vk,  #
                                       stride_oz, stride_oh, stride_om, stride_on,  #
                                       SM_SCALE: gl.constexpr,  #
                                       SEQLEN_Q: gl.constexpr,  #
                                       SEQLEN_K: gl.constexpr,  #
                                       BLOCK_M: gl.constexpr,  #
                                       BLOCK_N: gl.constexpr,  #
                                       HEAD_SZ: gl.constexpr,  #
                                       ):
    NUM_BUFFERS: gl.constexpr = 2
    NUM_WARPS: gl.constexpr = 8
    cfg = AttentionConfig(SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS)
    pgm = AttentionProgram.initialize(  #
        cfg, q_ptr, k_ptr, v_ptr, out_ptr,  #
        stride_qz, stride_qh, stride_qm, stride_qk,  #
        stride_kz, stride_kh, stride_kn, stride_kk,  #
        stride_vz, stride_vh, stride_vn, stride_vk,  #
        stride_oz, stride_oh, stride_om, stride_on,  #
        SM_SCALE)

    ITERS_IN_PROLOGUE_EPILOGUE: gl.constexpr = 3
    n_blocks_n = max((SEQLEN_K + BLOCK_N - 1) // BLOCK_N - ITERS_IN_PROLOGUE_EPILOGUE, 1)

    # Since QK from the final iteration is already peeled into the epilogue,
    # we only need to handle case where SEQLEN_K < ITERS_IN_PROLOGUE_EPILOGUE * BLOCK_N.
    has_remainder: gl.constexpr = SEQLEN_K < (ITERS_IN_PROLOGUE_EPILOGUE) * BLOCK_N
    REMAINDER_PEELED_ITERS = 1
    if has_remainder:
        n_blocks_n = n_blocks_n - REMAINDER_PEELED_ITERS

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    acc = gl.zeros([BLOCK_M, HEAD_SZ], dtype=gl.float32, layout=cfg.pv_layout)

    block_min = 0
    block_max = n_blocks_n * BLOCK_N
    """
    Prologue:
    t = i           t = i+1          t = i+2
    [GLDS_K]
    [LR_K, GLDS_V], [GLDS_K]
    [QK, SM0],      [LR_K, GLDS_V],  [GLDS_K]
    """
    # GLDS_K_t0, GLDS_K_t1, GLDS_V_t0
    pgm.tdm_load_global_to_shared_k([0, 0], buffer_index=0)
    pgm.tdm_load_global_to_shared_k([BLOCK_N, 0], buffer_index=1)
    pgm.tdm_load_global_to_shared_v([0, 0], buffer_index=0)

    # LR_K_t0
    k = pgm.tdm_shared_load_k(0, wait_count=2)

    # QK_t0
    qk = pgm.compute_qk(k, 0)

    # SM0_t0
    p, alpha, m_i = pgm.softmax_part0(qk, m_i)

    # GLDS_K_t2, GLDS_V_t1
    pgm.tdm_load_global_to_shared_k([2 * BLOCK_N, 0], buffer_index=0)
    pgm.tdm_load_global_to_shared_v([BLOCK_N, 0], buffer_index=1)

    # LR_K_t1
    k = pgm.tdm_shared_load_k(1, wait_count=3)
    iter_id = 0
    for block_id in range(block_min, block_max, BLOCK_N):
        """
        Steady State (Hot Loop - No Masking):
        t = i              t = i+1         t = i+2         t = i+3
        [SM1, LR_V, PV],   [QK, SM0],    [LR_K, GLDS_V]     [GLDS_K]

        unroll_factor=2 to save computation wrt iter_id and arithmetic computation
        for rotating registers.
        """
        """
        1/2 of unrolled loop
        """

        # QK, SM1, LR_V (no mask needed - all blocks in hot loop are full)
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            t_1 = block_id + BLOCK_N
            t_2 = block_id + 2 * BLOCK_N
            t_3 = block_id + 3 * BLOCK_N
            qk = pgm.compute_qk_no_mask(k)

        gl.amd.gfx1250.tdm.async_wait(2)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            # v = pgm.tdm_shared_load_v(iter_id % NUM_BUFFERS, wait_count=2)
            p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)
            v = pgm.v_buffer.index(iter_id % NUM_BUFFERS).load(layout=pgm.cfg.v_layout)
            pgm.tdm_load_global_to_shared_k([t_3, 0], (iter_id + 1) % NUM_BUFFERS)

        # PV, SM0, LR_K
        with gl.amd.warp_pipeline_stage("stage2", priority=0):
            acc = pgm.compute_pv(p, v, acc)

        gl.amd.gfx1250.tdm.async_wait(2)
        with gl.amd.warp_pipeline_stage("stage3", priority=1):
            # k = pgm.tdm_shared_load_k(iter_id % NUM_BUFFERS, wait_count=2)
            p, alpha, m_i = pgm.softmax_part0(qk, m_i)
            k = pgm.k_buffer.index(iter_id % NUM_BUFFERS).permute([1, 0]).load(layout=pgm.cfg.k_layout)
            pgm.tdm_load_global_to_shared_v([t_2, 0], iter_id % NUM_BUFFERS)
            iter_id += 1
    """
    Final iteration of steady state that requires masking.(if masking is required)
    """
    if has_remainder:
        t_1 = iter_id * BLOCK_N + BLOCK_N
        t_2 = iter_id * BLOCK_N + 2 * BLOCK_N
        t_3 = iter_id * BLOCK_N + 3 * BLOCK_N

        # Process the remainder block with masking
        qk = pgm.compute_qk(k, t_1)

        p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

        v = pgm.tdm_shared_load_v(iter_id % NUM_BUFFERS, wait_count=2)

        # GLDS_K
        pgm.tdm_load_global_to_shared_k([t_3, 0], (iter_id + 1) % NUM_BUFFERS)

        # PV, SM0, LR_K
        acc = pgm.compute_pv(p, v, acc)

        p, alpha, m_i = pgm.softmax_part0(qk, m_i)

        k = pgm.tdm_shared_load_k(iter_id % NUM_BUFFERS, wait_count=2)

        # GLDS_V
        pgm.tdm_load_global_to_shared_v([t_2, 0], iter_id % NUM_BUFFERS)
        iter_id += 1
    """
    Epilogue:
    t = i+1              t = i+2              t = i+3
    [SM1, LR_V, PV],    [QK, SM0],          [LR_K, GLDS_V]
                        [SM1, LR_V, PV],    [QK, SM0]
                                            [SM1, LR_V, PV]
    """
    epilogue_offset = (iter_id - 1) * BLOCK_N
    t_2 = epilogue_offset + 2 * BLOCK_N
    t_3 = epilogue_offset + 3 * BLOCK_N
    # SM1_t1, LR_V_t1, PV_t1
    p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

    v = pgm.tdm_shared_load_v(iter_id % NUM_BUFFERS, wait_count=2)

    acc = pgm.compute_pv(p, v, acc)

    # QK_t2, SM0_t2
    qk = pgm.compute_qk(k, t_2)
    p, alpha, m_i = pgm.softmax_part0(qk, m_i)

    # LR_K_t3, GLDS_V_t3
    k = pgm.tdm_shared_load_k(iter_id % NUM_BUFFERS, wait_count=1)

    pgm.tdm_load_global_to_shared_v([t_3, 0], iter_id % NUM_BUFFERS)

    # QK_t3, SM1_t2, LR_V_t2
    qk = pgm.compute_qk(k, t_3)

    p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

    v = pgm.tdm_shared_load_v((iter_id + 1) % NUM_BUFFERS, wait_count=1)

    # PV_t_2, SM0_t_3, SM1_t_3, LR_V_t3
    acc = pgm.compute_pv(p, v, acc)

    p, alpha, m_i = pgm.softmax_part0(qk, m_i)
    p, l_i, acc = pgm.softmax_part1(p, l_i, acc, alpha)

    v = pgm.tdm_shared_load_v(iter_id % NUM_BUFFERS, wait_count=0)

    # PV_t_3
    acc = pgm.compute_pv(p, v, acc)

    # Post loop scaling and output

    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    pgm.store_output(acc)


def generate_configs():
    base_configs = [
        # Tests for pipelined attention fwd kernel
        pytest.param({
            "BATCH": 8, "SEQLEN_Q": 512, "SEQLEN_K": 512, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 128, "BLOCK_M":
            128, "BLOCK_N": 64, "ATTN_FN": "pipeline"
        }),
        pytest.param({
            "BATCH": 8, "SEQLEN_Q": 1024, "SEQLEN_K": 1024, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 64,
            "BLOCK_M": 128, "BLOCK_N": 128, "ATTN_FN": "pipeline"
        }),
        pytest.param({
            "BATCH": 4, "SEQLEN_Q": 2000, "SEQLEN_K": 2000, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 64,
            "BLOCK_M": 128, "BLOCK_N": 128, "ATTN_FN": "pipeline"
        }),
        pytest.param({
            "BATCH": 1, "SEQLEN_Q": 3, "SEQLEN_K": 32, "NUM_Q_HEADS": 4, "NUM_K_HEADS": 4, "HEAD_SZ": 128, "BLOCK_M":
            128, "BLOCK_N": 32, "ATTN_FN": "pipeline"
        }),
        pytest.param({
            "BATCH": 4, "SEQLEN_Q": 1, "SEQLEN_K": 100, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 32, "BLOCK_M":
            128, "BLOCK_N": 32, "ATTN_FN": "pipeline"
        }),
        pytest.param({
            "BATCH": 1, "SEQLEN_Q": 1, "SEQLEN_K": 30, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 32, "BLOCK_M":
            128, "BLOCK_N": 32, "ATTN_FN": "pipeline"
        }),
        # Tests for pingpong pipelined attention fwd kernel
        pytest.param({
            "BATCH": 8, "SEQLEN_Q": 1024, "SEQLEN_K": 1024, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 128,
            "BLOCK_M": 256, "BLOCK_N": 64, "ATTN_FN": "pingpong"
        }),
        pytest.param({
            "BATCH": 1, "SEQLEN_Q": 300, "SEQLEN_K": 300, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 64, "BLOCK_M":
            256, "BLOCK_N": 32, "ATTN_FN": "pingpong"
        }),
        # decode
        pytest.param({
            "BATCH": 8, "SEQLEN_Q": 1, "SEQLEN_K": 1024, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 128, "BLOCK_M":
            128, "BLOCK_N": 32, "ATTN_FN": "decode"
        }),
        # Tests for non-pipelined attention fwd kernel
        pytest.param({
            "BATCH": 8, "SEQLEN_Q": 512, "SEQLEN_K": 512, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 128, "BLOCK_M":
            128, "BLOCK_N": 32, "ATTN_FN": "default"
        }),
        pytest.param({
            "BATCH": 1, "SEQLEN_Q": 1, "SEQLEN_K": 30, "NUM_Q_HEADS": 8, "NUM_K_HEADS": 8, "HEAD_SZ": 32, "BLOCK_M":
            128, "BLOCK_N": 32, "ATTN_FN": "default"
        }),
    ]
    return base_configs


_KERNEL_NUM_WARPS = {attn_fwd_kernel: 4, attn_fwd_pipelined_kernel: 4, attn_fwd_pingpong_pipelined_kernel: 8}

_ATTN_TYPE_TO_KERNEL_FN = {
    "default": attn_fwd_kernel, "pipeline": attn_fwd_pipelined_kernel, "pingpong": attn_fwd_pingpong_pipelined_kernel,
    "decode": attn_decode_fwd_kernel
}


def run_decode_attention(config, q, k, v, o, sm_scale):
    BATCH = config["BATCH"]
    SEQLEN_Q = config["SEQLEN_Q"]
    SEQLEN_K = config["SEQLEN_K"]
    NUM_Q_HEADS = config["NUM_Q_HEADS"]
    HEAD_SZ = config["HEAD_SZ"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]

    split_factor = compute_split_factor(BATCH, NUM_Q_HEADS, SEQLEN_K)
    chunk_size = (SEQLEN_K + split_factor - 1) // split_factor

    mid_o = torch.zeros((BATCH, NUM_Q_HEADS, split_factor, BLOCK_M, HEAD_SZ), dtype=torch.float32).cuda()
    mid_l = torch.zeros((BATCH, NUM_Q_HEADS, split_factor, BLOCK_M), dtype=torch.float32).cuda()
    mid_m = torch.full((BATCH, NUM_Q_HEADS, split_factor, BLOCK_M), float("-inf"), dtype=torch.float32).cuda()

    print(f"Launching Decode FWD: Split Factor {split_factor}, Chunk Size {chunk_size}")
    attn_stage1 = attn_decode_fwd_kernel[(BATCH, NUM_Q_HEADS,
                                          split_factor)](q, k, v, mid_o, mid_l, mid_m, *q.stride(), *k.stride(),
                                                         *v.stride(), *mid_o.stride(), *mid_l.stride(), *mid_m.stride(),
                                                         sm_scale, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N, HEAD_SZ,
                                                         split_factor, chunk_size, num_warps=4, waves_per_eu=1)

    attn_stage2 = attn_decode_reduce_kernel[(BATCH, NUM_Q_HEADS, 1)](mid_o, mid_l, mid_m, o, *mid_o.stride(),
                                                                     *mid_l.stride(), *mid_m.stride(), *o.stride(),
                                                                     sm_scale, split_factor, 16, HEAD_SZ, SEQLEN_Q,
                                                                     SEQLEN_K, BLOCK_N, num_warps=4, waves_per_eu=1)
    return (attn_stage1, attn_stage2)


def run_prefill_attention(config, q, k, v, o, sm_scale):
    BATCH = config["BATCH"]
    SEQLEN_Q = config["SEQLEN_Q"]
    SEQLEN_K = config["SEQLEN_K"]
    NUM_Q_HEADS = config["NUM_Q_HEADS"]
    HEAD_SZ = config["HEAD_SZ"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    attn_fn = _ATTN_TYPE_TO_KERNEL_FN[config["ATTN_FN"]]

    num_warps = _KERNEL_NUM_WARPS[attn_fn]

    grid = (
        BATCH,
        NUM_Q_HEADS,
        ((SEQLEN_Q + BLOCK_M - 1) // BLOCK_M),
    )
    attn_kernel = attn_fn[grid](
        q, k, v, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        sm_scale, SEQLEN_Q, SEQLEN_K,  #
        BLOCK_M, BLOCK_N,  #
        HEAD_SZ, num_warps=num_warps, waves_per_eu=1)
    return (attn_kernel, )


def run_attention(config, check=True):
    BATCH = config["BATCH"]
    SEQLEN_Q = config["SEQLEN_Q"]
    SEQLEN_K = config["SEQLEN_K"]
    NUM_Q_HEADS = config["NUM_Q_HEADS"]
    NUM_K_HEADS = config["NUM_K_HEADS"]
    HEAD_SZ = config["HEAD_SZ"]
    attn_fn = config["ATTN_FN"]

    dtype = torch.bfloat16
    torch.random.manual_seed(0)
    q = torch.randn((BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ), dtype=dtype)
    k = torch.randn((BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ), dtype=dtype)
    v = torch.randn((BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ), dtype=dtype)
    sm_scale = 1.0 / (HEAD_SZ**0.5)

    o = torch.zeros_like(q, dtype=torch.float32)
    if check:
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    o = o.cuda()

    if attn_fn == "decode":
        attn_kernel = run_decode_attention(config, q, k, v, o, sm_scale)
    else:
        attn_kernel = run_prefill_attention(config, q, k, v, o, sm_scale)

    torch.cuda.synchronize()
    o = o.cpu()
    rtol = 0.004
    atol = 0.004
    if check:
        torch.testing.assert_allclose(o, ref, rtol=rtol, atol=atol)
    return attn_kernel


@pytest.mark.parametrize("config", generate_configs())
def test_attention(config):
    run_attention(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", type=int, default=1, help='batch size')
    parser.add_argument("--seqlen-q", type=int, default=1024, help='Q sequence length')
    parser.add_argument("--seqlen-k", type=int, default=1024, help='K/V sequence length')
    parser.add_argument("--num-heads-q", type=int, default=8, help='Number of Q heads')
    parser.add_argument("--num-heads-k", type=int, default=8, help='Number of K/V heads')
    parser.add_argument("--head-size", type=int, default=128, help='Q/K/V head size')
    parser.add_argument("--block-m", type=int, default=128, help='BLOCK_M size')
    parser.add_argument("--block-n", type=int, default=128, help='BLOCK_N size')
    parser.add_argument(
        "--attention-type",
        type=str,
        choices=["default", "pipeline", "pingpong", "decode"],
        default="default",
        help="Attention Kernel Type",
    )
    args = parser.parse_args()
    config = {
        "BATCH": args.b,  #
        "SEQLEN_Q": args.seqlen_q, "SEQLEN_K": args.seqlen_k,  #
        "NUM_Q_HEADS": args.num_heads_q, "NUM_K_HEADS": args.num_heads_k,  #
        "HEAD_SZ": args.head_size,  #
        "BLOCK_M": args.block_m, "BLOCK_N": args.block_n,  #
        "ATTN_FN": args.attention_type,  #
    }
    print(config)
    attn_kernel = run_attention(config)
    [static_profile(kernel) for kernel in attn_kernel]
