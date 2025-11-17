"""
Multi-head attention kernel in Gluon
"""
# ruff: noqa: E402
import hip

# Needed for internal dev flow for now; will remove later
hip.hip.hipInit(0)

import argparse
import re
import pytest
import torch
import math
from einops import repeat

from triton import cdiv
from triton.language.core import _aggregate as aggregate
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

from triton.experimental.gluon.language.amd.gfx1250 import wmma_scaled
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.experimental.gluon.language.amd.gfx1250 import buffer_load, buffer_store
from triton.experimental.gluon.language.amd.gfx1250 import async_copy as cp

torch.random.manual_seed(0)

# ===-----------------------------------------------------------------------===#
# Scale Preshuffling Utilities
#
# In scaled wmma instruction, scales takes shapes of
# - [M, K // 32] for scaleA
# - [N, K // 32] for scaleB
# in global memory.

# To have vectorized memory access, it's better to store scales in a packed block scale layout.

# In this layout, scales are stored in the shape
# - scaleA: [M // 32 // 4, K // 32 // 4, 32, 4, 4]
# - scaleB: [N // 32 // 4, K // 32 // 4, 32, 4, 4]

# In this way, we can load scales from global memory in a more vectorized way.

# Then inside the kernel, we permute and reshape scales to canonical shapes required by scaled wmma.
# ===-----------------------------------------------------------------------===#


def _preshuffle_scale(x, preshuffle_factor: int = 128):
    b, h, NON_K, K_SCALE = x.shape
    num_chunk_m = NON_K // preshuffle_factor
    SCALE_KWIDTH = 4 if K_SCALE >= 4 else K_SCALE
    num_chunk_k = K_SCALE // SCALE_KWIDTH

    x = x.view(b, h, num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, SCALE_KWIDTH)
    x = x.permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    return x.view(b, h, NON_K // preshuffle_factor, K_SCALE * preshuffle_factor)


@gluon.constexpr_function
def _get_v_scale_preshuffle_factor(non_k_dim):
    # !Note that we mix non_k_dims of tensor and block here as a workaround specifically for this kernel. It may not work properly if non_k_dims of tensor and block don't match
    if non_k_dim >= 128:
        # This is the ideal factor we should use
        return 128
    elif non_k_dim == 64:
        # When the non_k_dim of a tensor is 64, we can only use factor 64.
        return 64
    else:
        raise ValueError(f'NonKDim={non_k_dim} is too small for current scale preshuffling schema')


# ===-----------------------------------------------------------------------===#
# Kernel Utilities
# ===-----------------------------------------------------------------------===#


def composition(cls):
    """ A decorator lets aggregate type to directly access attributes from its aggregate member. """

    def __getattr__(self, name):
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        for member in self.__dict__.values():
            if getattr(member, "__triton_aggregate__", False) and not hasattr(member, name):
                continue
            return getattr(member, name)
        raise AttributeError(f"{type(self).__name__} object has no attribute '{name}'")

    cls.__getattr__ = __getattr__
    return cls


@gluon.constexpr_function
def get_padded_shared_layout(outer_dim, inner_dim, transposed=False):
    shape = [outer_dim, inner_dim]
    padding_interval = inner_dim
    padding_amount = 16 if transposed else 8
    return ttgl.PaddedSharedLayout.with_identity_for([[padding_interval, padding_amount]], shape, [1, 0])


@aggregate
class AttentionConfigBase:
    Q_TYPE: ttgl.constexpr  # the data type for Q, either 'e5m2' or 'e4m3'
    P_TYPE: ttgl.constexpr  # the data type for P; we always assume P_TYPE == Q_TYPE
    KV_TYPE: ttgl.constexpr  # the data type for K and V, either 'e5m2', 'e4m3' or 'e2m1'
    SEQLEN_Q: ttgl.constexpr
    SEQLEN_K: ttgl.constexpr
    NUM_Q_HEADS: ttgl.constexpr
    NUM_K_HEADS: ttgl.constexpr
    HEAD_SZ: ttgl.constexpr
    BLOCK_M: ttgl.constexpr
    BLOCK_N: ttgl.constexpr
    NUM_WARPS: ttgl.constexpr
    NUM_BUFFERS: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N,
                 NUM_WARPS, NUM_BUFFERS):
        self.Q_TYPE = ttgl.constexpr(Q_TYPE)
        self.P_TYPE = ttgl.constexpr(Q_TYPE)
        self.KV_TYPE = ttgl.constexpr(KV_TYPE)
        self.SEQLEN_Q = ttgl.constexpr(SEQLEN_Q)
        self.SEQLEN_K = ttgl.constexpr(SEQLEN_K)
        self.NUM_Q_HEADS = ttgl.constexpr(NUM_Q_HEADS)
        self.NUM_K_HEADS = ttgl.constexpr(NUM_K_HEADS)
        self.HEAD_SZ = ttgl.constexpr(HEAD_SZ)
        self.BLOCK_M = ttgl.constexpr(BLOCK_M)
        self.BLOCK_N = ttgl.constexpr(BLOCK_N)
        self.NUM_WARPS = ttgl.constexpr(NUM_WARPS)
        self.NUM_BUFFERS = ttgl.constexpr(NUM_BUFFERS)


@composition
@aggregate
class GlobalScaledAttentionConfig:
    base: AttentionConfigBase

    q_layout: ttgl.constexpr
    k_smem_layout: ttgl.constexpr
    k_layout: ttgl.constexpr
    p_layout: ttgl.constexpr
    v_smem_layout: ttgl.constexpr
    v_layout: ttgl.constexpr
    acc_layout: ttgl.constexpr

    @gluon.constexpr_function
    def get_operand_smem_layout(outer_dim, inner_dim, transposed):
        shape = [outer_dim, inner_dim]
        padding_interval = inner_dim
        padding_amount = 16 if transposed else 8
        return ttgl.PaddedSharedLayout.with_identity_for([[padding_interval, padding_amount]], shape, [1, 0])

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N,
                 NUM_WARPS, NUM_BUFFERS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert KV_TYPE in ['e5m2', 'e4m3']
        assert NUM_WARPS == 4 or NUM_WARPS == 8

        base = AttentionConfigBase(Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M,
                                   BLOCK_N, NUM_WARPS, NUM_BUFFERS)
        self.base = base

        wmma_layout: ttgl.constexpr = ttgl.amd.AMDWMMALayout(  #
            version=3, transposed=True, warps_per_cta=[NUM_WARPS, 1], instr_shape=[16, 16, 128])
        self.q_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, 16))
        self.k_smem_layout = ttgl.constexpr(get_padded_shared_layout(BLOCK_N, HEAD_SZ))
        self.k_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, 16))
        self.p_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, 16))
        self.v_smem_layout = ttgl.constexpr(get_padded_shared_layout(HEAD_SZ, BLOCK_N))
        self.v_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, 16))
        self.acc_layout = ttgl.constexpr(wmma_layout)


@composition
@aggregate
class BlockScaledAttentionConfig:
    base: AttentionConfigBase

    P_SCALING: ttgl.constexpr  # whether to use per-block scaling for P; if False, use an uniform scale of 1.0
    KV_PACK_DIV: ttgl.constexpr  # Packing factor for mxfp operands. It's 2 for e2m1, and 1 for others
    NON_K_PRESHUFFLE_BLOCK_SIZE_K: ttgl.constexpr  # Divisor of nonK dim in preshuffling for k_scale
    NON_K_PRESHUFFLE_BLOCK_SIZE_V: ttgl.constexpr  # Divisor of nonK dim in preshuffling for v_scale
    SCALE_PRESHUFFLED: ttgl.constexpr

    q_layout: ttgl.constexpr
    q_scale_layout: ttgl.constexpr

    k_smem_layout: ttgl.constexpr
    k_layout: ttgl.constexpr
    k_scale_load_layout: ttgl.constexpr
    k_scale_smem_layout: ttgl.constexpr
    k_scale_layout: ttgl.constexpr

    p_layout: ttgl.constexpr
    p_scale_layout: ttgl.constexpr

    v_smem_layout: ttgl.constexpr
    v_layout: ttgl.constexpr
    v_scale_load_layout: ttgl.constexpr
    v_scale_smem_layout: ttgl.constexpr
    v_scale_layout: ttgl.constexpr

    acc_layout: ttgl.constexpr

    @gluon.constexpr_function
    def get_acc_layout(tiles_per_warp, num_warps):
        wmma_layout = ttgl.amd.AMDWMMALayout(version=3,  #
                                             transposed=True,  #
                                             warps_per_cta=[num_warps, 1],  #
                                             instr_shape=[16, 16, 128],  #
                                             tiles_per_warp=tiles_per_warp)
        return wmma_layout

    @gluon.constexpr_function
    def get_operand_reg_layout(operand, tiles_per_warp, num_warps, packed):
        wmma_layout = BlockScaledAttentionConfig.get_acc_layout(tiles_per_warp, num_warps)
        wmma_layout_packed = BlockScaledAttentionConfig.get_acc_layout(tiles_per_warp, num_warps)
        wmma_layout_packed.instr_shape[-1] //= 2
        return ttgl.DotOperandLayout(operand, wmma_layout_packed if packed else wmma_layout, 16)

    @gluon.constexpr_function
    def get_scale_load_layout(shape, num_warps):
        _, inner_dim = shape
        assert inner_dim in [64, 128, 256, 512]
        if inner_dim == 512:
            return ttgl.BlockedLayout([1, 16], [1, 32], [num_warps, 1], [1, 0])
        if inner_dim == 256:
            return ttgl.BlockedLayout([1, 8], [1, 32], [num_warps, 1], [1, 0])
        elif inner_dim == 128:
            return ttgl.BlockedLayout([1, 4], [1, 32], [num_warps, 1], [1, 0])
        else:
            return ttgl.BlockedLayout([1, 4], [2, 16], [num_warps, 1], [1, 0])

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N,
                 NUM_WARPS, P_SCALING, SCALE_PRESHUFFLED, NUM_BUFFERS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert KV_TYPE in ['e5m2', 'e4m3', 'e2m1']
        assert NUM_WARPS == 4 or NUM_WARPS == 8

        base = AttentionConfigBase(Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M,
                                   BLOCK_N, NUM_WARPS, NUM_BUFFERS)
        self.base = base

        self.P_SCALING = ttgl.constexpr(P_SCALING)
        self.SCALE_PRESHUFFLED = ttgl.constexpr(SCALE_PRESHUFFLED)

        BLOCK_K_SCALE_QK = HEAD_SZ // 32
        BLOCK_K_SCALE_PV = BLOCK_N // 32
        self.NON_K_PRESHUFFLE_BLOCK_SIZE_K = ttgl.constexpr(128 if SCALE_PRESHUFFLED else 1)
        self.NON_K_PRESHUFFLE_BLOCK_SIZE_V = ttgl.constexpr(
            _get_v_scale_preshuffle_factor(HEAD_SZ) if SCALE_PRESHUFFLED else 1)

        tiles_per_warp: ttgl.constexpr = [2, 2] if SCALE_PRESHUFFLED else [1, 1]
        num_warps: ttgl.constexpr = NUM_WARPS

        KV_PACK_DIV = ttgl.constexpr(2 if KV_TYPE == 'e2m1' else 1)
        self.KV_PACK_DIV = KV_PACK_DIV

        self.q_layout = ttgl.constexpr(
            BlockScaledAttentionConfig.get_operand_reg_layout(0, tiles_per_warp, num_warps, packed=False))
        self.q_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.q_layout, [self.BLOCK_M, BLOCK_K_SCALE_QK]))

        self.k_smem_layout = ttgl.constexpr(get_padded_shared_layout(BLOCK_N, HEAD_SZ // KV_PACK_DIV))
        self.k_layout = ttgl.constexpr(
            BlockScaledAttentionConfig.get_operand_reg_layout(1, tiles_per_warp, num_warps, packed=(KV_TYPE == 'e2m1')))
        self.k_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.k_layout, [self.BLOCK_N, BLOCK_K_SCALE_QK]))
        self.k_scale_smem_layout = ttgl.constexpr(ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0]))

        # Only for non-preshuffling case
        self.k_scale_load_layout = ttgl.constexpr(
            BlockScaledAttentionConfig.get_scale_load_layout((BLOCK_K_SCALE_QK, BLOCK_N), num_warps))

        self.p_layout = ttgl.constexpr(
            BlockScaledAttentionConfig.get_operand_reg_layout(0, tiles_per_warp, num_warps, packed=False))
        self.p_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.p_layout, [self.BLOCK_M, BLOCK_K_SCALE_PV]))

        self.v_smem_layout = ttgl.constexpr(get_padded_shared_layout(HEAD_SZ, BLOCK_N // KV_PACK_DIV))
        self.v_layout = ttgl.constexpr(
            BlockScaledAttentionConfig.get_operand_reg_layout(1, tiles_per_warp, num_warps, packed=(KV_TYPE == 'e2m1')))
        self.v_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.v_layout, [self.HEAD_SZ, BLOCK_K_SCALE_PV]))
        self.v_scale_smem_layout = ttgl.constexpr(ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0]))

        # Only for non-preshuffling case
        self.v_scale_load_layout = ttgl.constexpr(
            BlockScaledAttentionConfig.get_scale_load_layout((BLOCK_K_SCALE_PV, HEAD_SZ), num_warps))

        self.acc_layout = ttgl.constexpr(BlockScaledAttentionConfig.get_acc_layout(tiles_per_warp, num_warps))


# ===-----------------------------------------------------------------------===#
# Kernel Primitives
# ===-----------------------------------------------------------------------===#


@aggregate
class GlobalScaledAttentionProgram:
    cfg: GlobalScaledAttentionConfig

    q: ttgl.tensor
    q_scale: ttgl.tensor
    k_desc: tdm.tensor_descriptor
    k_buffer: ttgl.shared_memory_descriptor
    k_step: ttgl.constexpr
    k_scale: ttgl.tensor
    v_desc: tdm.tensor_descriptor
    v_buffer: ttgl.shared_memory_descriptor
    v_step: ttgl.constexpr
    v_scale: ttgl.tensor
    o_ptr: ttgl.tensor
    o_offs: ttgl.tensor
    o_mask: ttgl.tensor
    sm_scale: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg,  #
                 q, q_scale,  #
                 k_desc, k_buffer, k_step, k_scale,  #
                 v_desc, v_buffer, v_step, v_scale,  #
                 o_ptr, o_offs, o_mask,  #
                 sm_scale):
        self.cfg = cfg
        self.q = q
        self.q_scale = q_scale
        self.k_desc = k_desc
        self.k_buffer = k_buffer
        self.k_step = ttgl.constexpr(k_step)
        self.k_scale = k_scale
        self.v_desc = v_desc
        self.v_buffer = v_buffer
        self.v_step = ttgl.constexpr(v_step)
        self.v_scale = v_scale
        self.o_ptr = o_ptr
        self.o_offs = o_offs
        self.o_mask = o_mask
        self.sm_scale = sm_scale

    @gluon.jit
    def initialize(cfg, q_ptr, q_scale, k_ptr, k_scale, v_ptr, v_scale, o_ptr, sm_scale):
        ttgl.static_assert(isinstance(cfg, GlobalScaledAttentionConfig))
        SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
        SEQLEN_Q: ttgl.constexpr = cfg.SEQLEN_Q
        HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
        NUM_Q_HEADS: ttgl.constexpr = cfg.NUM_Q_HEADS
        NUM_K_HEADS: ttgl.constexpr = cfg.NUM_K_HEADS
        BLOCK_M: ttgl.constexpr = cfg.BLOCK_M
        BLOCK_N: ttgl.constexpr = cfg.BLOCK_N
        NUM_BUFFERS: ttgl.constexpr = cfg.NUM_BUFFERS

        # programs: (NUM_Q_HEADS, NUM_BLOCKS, BATCH)
        off_h = ttgl.program_id(0)
        off_m = ttgl.program_id(1)
        off_z = ttgl.program_id(2)

        # compute offsets for q
        # q       [BLOCK_M, HEAD_SZ]
        q_off_zh = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h)
        q_offs_m = BLOCK_M * off_m + \
                   ttgl.arange(0, BLOCK_M, ttgl.SliceLayout(1, cfg.q_layout))
        q_offs_d = ttgl.arange(0, HEAD_SZ, ttgl.SliceLayout(0, cfg.q_layout))
        q_offs = q_off_zh + \
                q_offs_m[:, None] * HEAD_SZ + \
                q_offs_d[None, :]

        ttgl.static_assert(NUM_Q_HEADS % NUM_K_HEADS == 0)
        GROUP_SIZE: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
        off_hk = off_h // GROUP_SIZE

        # create descriptor and buffer for k
        # shape: [HEAD_SZ, BLOCK_N]
        k_off_zh = SEQLEN_K * (HEAD_SZ) * (NUM_K_HEADS * off_z + off_hk)
        k_desc = tdm.make_tensor_descriptor(  #
            base=k_off_zh + k_ptr,  #
            shape=[SEQLEN_K, HEAD_SZ],  #
            strides=[HEAD_SZ, 1],  #
            block_shape=[BLOCK_N, HEAD_SZ],  #
            layout=cfg.k_smem_layout)
        k_buffer = ttgl.allocate_shared_memory(  #
            k_desc.dtype,  #
            [NUM_BUFFERS] + k_desc.block_shape,  #
            k_desc.layout)
        k_step: ttgl.constexpr = BLOCK_N

        # create descriptor and buffer for v
        # shape: [BLOCK_N, HEAD_SZ]
        v_off_zh = (SEQLEN_K) * HEAD_SZ * (NUM_K_HEADS * off_z + off_hk)
        v_desc = tdm.make_tensor_descriptor(  #
            base=v_off_zh + v_ptr,  #
            shape=[HEAD_SZ, SEQLEN_K],  #
            strides=[SEQLEN_K, 1],  #
            block_shape=[HEAD_SZ, BLOCK_N],  #
            layout=cfg.v_smem_layout)
        v_buffer = ttgl.allocate_shared_memory(  #
            v_desc.dtype,  #
            [NUM_BUFFERS] + v_desc.block_shape,  #
            v_desc.layout)
        v_step: ttgl.constexpr = BLOCK_N

        # output [BLOCK_M, HEAD_SZ]
        o_offs_zh = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h)
        o_offs_m = BLOCK_M * off_m + \
                ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, cfg.acc_layout))
        o_offs_n = ttgl.arange(0, HEAD_SZ, layout=ttgl.SliceLayout(0, cfg.acc_layout))
        o_offs = o_offs_zh + \
                o_offs_m[:, None] * HEAD_SZ + \
                o_offs_n[None, :]
        o_mask = o_offs_m[:, None] < SEQLEN_Q

        # load q
        q_mask = q_offs_m[:, None] < SEQLEN_Q
        q = buffer_load(q_ptr, q_offs, mask=q_mask, other=0.0)

        # create the program
        return GlobalScaledAttentionProgram(  #
            cfg,  #
            q, q_scale,  #
            k_desc, k_buffer, k_step, k_scale,  #
            v_desc, v_buffer, v_step, v_scale,  #
            o_ptr, o_offs, o_mask,  #
            sm_scale)

    @gluon.jit
    def issue_global_load_k(self, i, buf):
        k_step: ttgl.constexpr = self.k_step
        k_buffer = self.k_buffer.index(buf)
        tdm.async_load(self.k_desc, [i * k_step, 0], k_buffer)

    @gluon.jit
    def issue_global_load_v(self, i, buf):
        v_step: ttgl.constexpr = self.v_step
        v_buffer = self.v_buffer.index(buf)
        tdm.async_load(self.v_desc, [0, i * v_step], v_buffer)

    @gluon.jit
    def shared_load_k(self, buf, wait_count):
        cfg = self.cfg

        k_buffer = self.k_buffer.index(buf).permute((1, 0))

        tdm.async_wait(wait_count)
        k = k_buffer.load(cfg.k_layout)
        k_scale = self.k_scale
        return k, k_scale

    @gluon.jit
    def shared_load_v(self, buf, wait_count):
        cfg = self.cfg

        v_buffer = self.v_buffer.index(buf).permute((1, 0))

        tdm.async_wait(wait_count)
        v = v_buffer.load(cfg.v_layout)
        v_scale = self.v_scale
        return v, v_scale

    @gluon.jit
    def compute_qk(self, k, k_scale):
        cfg = self.cfg
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)

        qk = wmma_scaled(self.q, self.q_scale, cfg.Q_TYPE, k, k_scale, cfg.KV_TYPE, zero)
        return qk

    @gluon.jit
    def compute_pv(self, p, p_scale, v, v_scale, acc):
        cfg = self.cfg

        acc = wmma_scaled(p, p_scale, cfg.P_TYPE, v, v_scale, cfg.KV_TYPE, acc)
        return acc

    @gluon.jit
    def softmax0(self, qk, m_i):
        sm_scale = self.sm_scale

        m_ij = ttgl.maximum(m_i, ttgl.max(qk, 1))

        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
        p = ttgl.exp2(qk_shifted)

        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)

        return p, alpha, m_ij

    @gluon.jit
    def softmax1(self, p, alpha, acc, l_i):
        cfg = self.cfg

        l_ij = ttgl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij

        p = p.to(ttgl.float8e4nv if cfg.P_TYPE == 'e4m3' else ttgl.float8e5)
        p = ttgl.convert_layout(p, cfg.p_layout)
        p_scale = 0x7F

        return p, p_scale, acc, l_i

    @gluon.jit
    def store_output(self, acc):
        o = acc.to(self.o_ptr.dtype.element_ty)
        buffer_store(o, self.o_ptr, self.o_offs, mask=self.o_mask)


@aggregate
class BlockScaledAttentionProgram:
    cfg: BlockScaledAttentionConfig

    q: ttgl.tensor
    q_scale: ttgl.tensor

    k_desc: tdm.tensor_descriptor
    k_scale_desc: tdm.tensor_descriptor
    k_scale_ptr: ttgl.tensor
    k_scale_offs: ttgl.tensor
    k_buffer: ttgl.shared_memory_descriptor
    k_scale_buffer: ttgl.shared_memory_descriptor
    k_step: ttgl.constexpr
    k_scale_step: ttgl.constexpr

    v_desc: tdm.tensor_descriptor
    v_scale_desc: tdm.tensor_descriptor
    v_scale_ptr: ttgl.tensor
    v_scale_offs: ttgl.tensor
    v_buffer: ttgl.shared_memory_descriptor
    v_scale_buffer: ttgl.shared_memory_descriptor
    v_step: ttgl.constexpr
    v_scale_step: ttgl.constexpr

    o_ptr: ttgl.tensor
    o_offs: ttgl.tensor
    o_mask: ttgl.tensor

    # TODO: sm_scale should be a constexpr but the current llvm can not properly
    # fuse v_fma for literal operands, so we are using tensor here to ensure
    # it is in a register. Change it back to constexpr once the llvm is fixed.
    sm_scale: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg,  #
                 q, q_scale,  #
                 k_desc, k_scale_desc, k_scale_ptr, k_scale_offs, k_buffer, k_scale_buffer, k_step, k_scale_step,  #
                 v_desc, v_scale_desc, v_scale_ptr, v_scale_offs, v_buffer, v_scale_buffer, v_step, v_scale_step,  #
                 o_ptr, o_offs, o_mask,  #
                 sm_scale):
        self.cfg = cfg
        self.q = q
        self.q_scale = q_scale
        self.k_desc = k_desc
        self.k_scale_desc = k_scale_desc
        self.k_scale_ptr = k_scale_ptr
        self.k_scale_offs = k_scale_offs
        self.k_buffer = k_buffer
        self.k_scale_buffer = k_scale_buffer
        self.k_step = ttgl.constexpr(k_step)
        self.k_scale_step = ttgl.constexpr(k_scale_step)
        self.v_desc = v_desc
        self.v_scale_desc = v_scale_desc
        self.v_scale_ptr = v_scale_ptr
        self.v_scale_offs = v_scale_offs
        self.v_buffer = v_buffer
        self.v_scale_buffer = v_scale_buffer
        self.v_step = ttgl.constexpr(v_step)
        self.v_scale_step = ttgl.constexpr(v_scale_step)
        self.o_ptr = o_ptr
        self.o_offs = o_offs
        self.o_mask = o_mask
        self.sm_scale = sm_scale

    @gluon.jit
    def initialize_k_scale_offsets(cfg, off_z, off_hk):
        assert not cfg.SCALE_PRESHUFFLED, "Only use async load for scales when preshuffling is disabled"
        k_scale_off_zh = cfg.SEQLEN_K * (cfg.HEAD_SZ // 32) * (cfg.NUM_K_HEADS * off_z + off_hk)
        k_scale_offs_d = ttgl.arange(0, cfg.HEAD_SZ // 32, ttgl.SliceLayout(1, cfg.k_scale_load_layout))
        k_scale_offs_n = ttgl.arange(0, cfg.BLOCK_N, ttgl.SliceLayout(0, cfg.k_scale_load_layout))
        # in non-preshuffled case, k_scale is transposed
        # shape: [HEAD_SZ / 32, BLOCK_N]
        k_scale_offs = k_scale_off_zh + \
                    k_scale_offs_d[:, None] * cfg.SEQLEN_K + \
                    k_scale_offs_n[None, :]

        return k_scale_offs

    @gluon.jit
    def initialize_k_scale_descriptor(cfg, k_scale_ptr, off_z, off_hk):
        assert cfg.SCALE_PRESHUFFLED, "Only use TDM load for scales when preshuffling is enabled"
        BLOCK_K_SCALE_PRESHUFFLED: ttgl.constexpr = cfg.HEAD_SZ // 32 * cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_K
        BLOCK_N_PRESHUFFLED: ttgl.constexpr = cfg.BLOCK_N // cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_K
        k_scale_off_zh = (cfg.SEQLEN_K // cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_K) * BLOCK_K_SCALE_PRESHUFFLED * (
            cfg.NUM_K_HEADS * off_z + off_hk)
        # shape: [BLOCK_N / 128, HEAD_SZ * 4]
        return tdm.make_tensor_descriptor(  #
            base=k_scale_off_zh + k_scale_ptr,  #
            shape=[cfg.SEQLEN_K // cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_K, BLOCK_K_SCALE_PRESHUFFLED],  #
            strides=[BLOCK_K_SCALE_PRESHUFFLED, 1],  #
            block_shape=[BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED],  #
            layout=cfg.k_scale_smem_layout)

    @gluon.jit
    def initialize_v_scale_offsets(cfg, off_z, off_hk):
        assert not cfg.SCALE_PRESHUFFLED, "Only use async load for scales when preshuffling is disabled"
        v_scale_off_zh = (cfg.SEQLEN_K // 32) * cfg.HEAD_SZ * (cfg.NUM_K_HEADS * off_z + off_hk)
        v_scale_offs_n = ttgl.arange(0, cfg.BLOCK_N // 32, ttgl.SliceLayout(1, cfg.v_scale_load_layout))
        v_scale_offs_d = ttgl.arange(0, cfg.HEAD_SZ, ttgl.SliceLayout(0, cfg.v_scale_load_layout))
        # in non-preshuffled case, v_scale is transposed
        # shape: [BLOCK_N / 32, HEAD_SZ]
        v_scale_offs = v_scale_off_zh + \
                    v_scale_offs_n[:, None] * cfg.HEAD_SZ + \
                    v_scale_offs_d[None, :]

        return v_scale_offs

    @gluon.jit
    def initialize_v_scale_descriptor(cfg, v_scale_ptr, off_z, off_hk):
        assert cfg.SCALE_PRESHUFFLED, "Only use TDM load for scales when preshuffling is enabled"
        BLOCK_K_SCALE_PRESHUFFLED: ttgl.constexpr = cfg.BLOCK_N // 32 * cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_V
        BLOCK_N_PRESHUFFLED: ttgl.constexpr = cfg.HEAD_SZ // cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_V
        v_scale_off_zh = (cfg.SEQLEN_K // 32 *
                          cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_V) * BLOCK_N_PRESHUFFLED * (cfg.NUM_K_HEADS * off_z + off_hk)
        # shape(head_sz=128): [HEAD_SZ / 128, BLOCK_N * 4]
        # shape(head_sz=64):  [HEAD_SZ / 64, BLOCK_N * 2]
        return tdm.make_tensor_descriptor(  #
            base=v_scale_off_zh + v_scale_ptr,  #
            shape=[BLOCK_N_PRESHUFFLED, cfg.SEQLEN_K // 32 * cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_V],  #
            strides=[cfg.SEQLEN_K // 32 * cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_V, 1],  #
            block_shape=[BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED],  #
            layout=cfg.v_scale_smem_layout)

    @gluon.jit
    def initialize_scale_buffer(cfg, scale_ptr, outer_dim, inner_dim, smem_layout):
        if cfg.SCALE_PRESHUFFLED:
            block_shape: ttgl.constexpr = [outer_dim, inner_dim]
        else:
            # scale is transposed for non-preshuffled case
            block_shape: ttgl.constexpr = [inner_dim, outer_dim]
        return ttgl.allocate_shared_memory(  #
            scale_ptr.dtype.element_ty,  #
            [cfg.NUM_BUFFERS] + block_shape,  #
            smem_layout)

    @gluon.jit
    def initialize(cfg,  #
                   q_ptr, q_scale_ptr,  #
                   k_ptr, k_scale_ptr,  #
                   v_ptr, v_scale_ptr,  #
                   o_ptr,  #
                   sm_scale):
        ttgl.static_assert(isinstance(cfg, BlockScaledAttentionConfig))
        SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
        SEQLEN_Q: ttgl.constexpr = cfg.SEQLEN_Q
        HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
        NUM_Q_HEADS: ttgl.constexpr = cfg.NUM_Q_HEADS
        NUM_K_HEADS: ttgl.constexpr = cfg.NUM_K_HEADS
        BLOCK_M: ttgl.constexpr = cfg.BLOCK_M
        BLOCK_N: ttgl.constexpr = cfg.BLOCK_N
        KV_PACK_DIV: ttgl.constexpr = cfg.KV_PACK_DIV
        NUM_BUFFERS: ttgl.constexpr = cfg.NUM_BUFFERS

        # programs: (NUM_Q_HEADS, NUM_BLOCKS, BATCH)
        off_h = ttgl.program_id(0)
        off_m = ttgl.program_id(1)
        off_z = ttgl.program_id(2)

        # compute offsets for q and q_scale
        # q       [BLOCK_M, HEAD_SZ]
        # q_scale [BLOCK_M, HEAD_SZ / 32]
        q_off_zh = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h)
        q_offs_m = BLOCK_M * off_m + \
                   ttgl.arange(0, BLOCK_M, ttgl.SliceLayout(1, cfg.q_layout))
        q_offs_d = ttgl.arange(0, HEAD_SZ, ttgl.SliceLayout(0, cfg.q_layout))
        q_offs = q_off_zh + \
                q_offs_m[:, None] * HEAD_SZ + \
                q_offs_d[None, :]

        q_scale_off_zh = SEQLEN_Q * (HEAD_SZ // 32) * (NUM_Q_HEADS * off_z + off_h)
        q_scale_offs_m = BLOCK_M * off_m + \
                        ttgl.arange(0, BLOCK_M, ttgl.SliceLayout(1, cfg.q_scale_layout))
        q_scale_offs_d = ttgl.arange(0, HEAD_SZ // 32, ttgl.SliceLayout(0, cfg.q_scale_layout))
        q_scale_offs = q_scale_off_zh + \
                    q_scale_offs_m[:, None] * (HEAD_SZ // 32) + \
                    q_scale_offs_d[None, :]

        ttgl.static_assert(NUM_Q_HEADS % NUM_K_HEADS == 0)
        GROUP_SIZE: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
        off_hk = off_h // GROUP_SIZE

        # create descriptor and buffer for k
        k_off_zh = SEQLEN_K * (HEAD_SZ // KV_PACK_DIV) * (NUM_K_HEADS * off_z + off_hk)
        k_desc = tdm.make_tensor_descriptor(  #
            base=k_off_zh + k_ptr,  #
            shape=[SEQLEN_K, HEAD_SZ // KV_PACK_DIV],  #
            strides=[HEAD_SZ // KV_PACK_DIV, 1],  #
            block_shape=[BLOCK_N, HEAD_SZ // KV_PACK_DIV],  #
            layout=cfg.k_smem_layout)
        k_buffer = ttgl.allocate_shared_memory(  #
            k_desc.dtype,  #
            [NUM_BUFFERS] + k_desc.block_shape,  #
            k_desc.layout)
        k_step: ttgl.constexpr = BLOCK_N

        # create buffer and offsets for k_scale
        k_scale_desc = BlockScaledAttentionProgram.initialize_k_scale_descriptor(cfg, k_scale_ptr, off_z, off_hk)
        k_scale_offs = BlockScaledAttentionProgram.initialize_k_scale_offsets(cfg, off_z, off_hk)
        k_scale_buffer = BlockScaledAttentionProgram.initialize_scale_buffer(
            cfg, k_scale_ptr,  #
            BLOCK_N // cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_K,  #
            (HEAD_SZ // 32) * cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_K,  #
            cfg.k_scale_smem_layout)
        k_scale_step: ttgl.constexpr = BLOCK_N // cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_K

        # create descriptor and buffer for v
        v_off_zh = (SEQLEN_K // KV_PACK_DIV) * HEAD_SZ * (NUM_K_HEADS * off_z + off_hk)
        v_desc = tdm.make_tensor_descriptor(  #
            base=v_off_zh + v_ptr,  #
            shape=[HEAD_SZ, SEQLEN_K // KV_PACK_DIV],  #
            strides=[SEQLEN_K // KV_PACK_DIV, 1],  #
            block_shape=[HEAD_SZ, BLOCK_N // KV_PACK_DIV],  #
            layout=cfg.v_smem_layout)
        v_buffer = ttgl.allocate_shared_memory(  #
            v_desc.dtype,  #
            [NUM_BUFFERS] + v_desc.block_shape,  #
            v_desc.layout)
        v_step: ttgl.constexpr = BLOCK_N // KV_PACK_DIV

        # create buffer and offsets for v_scale
        v_scale_desc = BlockScaledAttentionProgram.initialize_v_scale_descriptor(cfg, v_scale_ptr, off_z, off_hk)
        v_scale_offs = BlockScaledAttentionProgram.initialize_v_scale_offsets(cfg, off_z, off_hk)
        v_scale_buffer = BlockScaledAttentionProgram.initialize_scale_buffer(
            cfg, v_scale_ptr,  #
            HEAD_SZ // cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_V,  #
            (BLOCK_N // 32) * cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_V,  #
            cfg.v_scale_smem_layout)
        if cfg.SCALE_PRESHUFFLED:
            v_scale_step: ttgl.constexpr = cfg.BLOCK_N // 32 * cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_V
        else:
            v_scale_step: ttgl.constexpr = (BLOCK_N // 32) * HEAD_SZ

        # output [BLOCK_M, HEAD_SZ]
        o_offs_zh = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h)
        o_offs_m = BLOCK_M * off_m + \
                ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, cfg.acc_layout))
        o_offs_n = ttgl.arange(0, HEAD_SZ, layout=ttgl.SliceLayout(0, cfg.acc_layout))
        o_offs = o_offs_zh + \
                o_offs_m[:, None] * HEAD_SZ + \
                o_offs_n[None, :]
        o_mask = o_offs_m[:, None] < SEQLEN_Q

        # load q and q_scale
        q_mask = q_offs_m[:, None] < SEQLEN_Q
        q = buffer_load(q_ptr, q_offs, mask=q_mask, other=0.0)
        q_scale_mask = q_scale_offs_m[:, None] < SEQLEN_Q
        q_scale = buffer_load(q_scale_ptr, q_scale_offs, mask=q_scale_mask, other=0x7F)

        # create the program
        return BlockScaledAttentionProgram(  #
            cfg,  #
            q, q_scale,  #
            k_desc, k_scale_desc, k_scale_ptr, k_scale_offs, k_buffer,  #
            k_scale_buffer, k_step, k_scale_step,  #
            v_desc, v_scale_desc, v_scale_ptr, v_scale_offs, v_buffer,  #
            v_scale_buffer, v_step, v_scale_step,  #
            o_ptr, o_offs, o_mask,  #
            sm_scale)

    @gluon.jit
    def issue_global_load_k(self, i, buf):
        cfg = self.cfg
        k_step: ttgl.constexpr = self.k_step
        k_scale_step: ttgl.constexpr = self.k_scale_step

        k_buffer = self.k_buffer.index(buf)
        k_scale_buffer = self.k_scale_buffer.index(buf)

        tdm.async_load(self.k_desc, [i * k_step, 0], k_buffer)
        if cfg.SCALE_PRESHUFFLED:
            tdm.async_load(self.k_scale_desc, [i * k_scale_step, 0], k_scale_buffer)
        else:
            # We use TDM to avoid register spills for preshuffling, but TDM increases
            # register usage for non-preshuffling case, we will converge later.
            # Need to keep this line and enable or remove this after investigation.
            # tdm.async_load(self.k_scale_desc, [0, i * k_scale_step], k_scale_buffer)
            k_scale_ptrs = (self.k_scale_ptr + i * k_scale_step) + self.k_scale_offs
            cp.global_to_shared(k_scale_buffer, k_scale_ptrs)
            cp.commit_group()

    @gluon.jit
    def issue_global_load_v(self, i, buf):
        cfg = self.cfg
        v_step: ttgl.constexpr = self.v_step
        v_scale_step: ttgl.constexpr = self.v_scale_step

        v_buffer = self.v_buffer.index(buf)
        v_scale_buffer = self.v_scale_buffer.index(buf)

        tdm.async_load(self.v_desc, [0, i * v_step], v_buffer)
        if cfg.SCALE_PRESHUFFLED:
            tdm.async_load(self.v_scale_desc, [0, i * v_scale_step], v_scale_buffer)
        else:
            # We use TDM to avoid register spills for preshuffling, but TDM increases
            # register usage for non-preshuffling case, we will converge later.
            # Need to keep this line and enable or remove this after investigation.
            # tdm.async_load(self.v_scale_desc, [i * v_scale_step, 0], v_scale_buffer)
            v_scale_ptrs = (self.v_scale_ptr + i * v_scale_step) + self.v_scale_offs
            cp.global_to_shared(v_scale_buffer, v_scale_ptrs)
            cp.commit_group()

    @gluon.jit
    def shared_load_k(self, buf, wait_count):
        cfg = self.cfg

        k_buffer = self.k_buffer.index(buf).permute((1, 0))
        k_scale_buffer = self.k_scale_buffer.index(buf)
        if cfg.SCALE_PRESHUFFLED:
            k_scale_buffer = self._unshuffle_scale_subview(k_scale_buffer, cfg.BLOCK_N, cfg.HEAD_SZ // 32,
                                                           cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_K)
        else:
            k_scale_buffer = k_scale_buffer.permute((1, 0))

        self._async_wait(wait_count)
        k = k_buffer.load(cfg.k_layout)
        k_scale = k_scale_buffer.load(cfg.k_scale_layout)
        return k, k_scale

    @gluon.jit
    def shared_load_v(self, buf, wait_count):
        cfg = self.cfg

        v_buffer = self.v_buffer.index(buf).permute((1, 0))
        v_scale_buffer = self.v_scale_buffer.index(buf)
        if cfg.SCALE_PRESHUFFLED:
            v_scale_buffer = self._unshuffle_scale_subview(v_scale_buffer, cfg.HEAD_SZ, cfg.BLOCK_N // 32,
                                                           cfg.NON_K_PRESHUFFLE_BLOCK_SIZE_V)
        else:
            v_scale_buffer = v_scale_buffer.permute((1, 0))

        self._async_wait(wait_count)
        v = v_buffer.load(cfg.v_layout)
        v_scale = v_scale_buffer.load(cfg.v_scale_layout)
        return v, v_scale

    @gluon.jit
    def compute_qk(self, k, k_scale):
        cfg = self.cfg
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)

        qk = wmma_scaled(self.q, self.q_scale, cfg.Q_TYPE, k, k_scale, cfg.KV_TYPE, zero)
        return qk

    @gluon.jit
    def compute_pv(self, p, p_scale, v, v_scale, acc):
        cfg = self.cfg

        acc = wmma_scaled(p, p_scale, cfg.P_TYPE, v, v_scale, cfg.KV_TYPE, acc)
        return acc

    @gluon.jit
    def softmax0(self, qk, m_i):
        sm_scale = self.sm_scale

        m_ij = ttgl.maximum(m_i, ttgl.max(qk, 1))

        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
        p = ttgl.exp2(qk_shifted)

        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)

        return p, alpha, m_ij

    @gluon.jit
    def softmax1(self, p, alpha, acc, l_i):
        cfg = self.cfg

        l_ij = ttgl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij

        if cfg.P_SCALING:
            p, p_scale = self._downcast_fp32_to_mxfp8(p, cfg.P_TYPE, [cfg.BLOCK_M, cfg.BLOCK_N])
            p = ttgl.convert_layout(p, cfg.p_layout)
            p_scale = ttgl.convert_layout(p_scale, cfg.p_scale_layout)
        else:
            p = self._downcast_fp32_to_fp8(p, cfg.P_TYPE)
            p = ttgl.convert_layout(p, cfg.p_layout)
            p_scale = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 32], 0x7F, ttgl.uint8, cfg.p_scale_layout)

        return p, p_scale, acc, l_i

    @gluon.jit
    def store_output(self, acc):
        o = acc.to(self.o_ptr.dtype.element_ty)
        buffer_store(o, self.o_ptr, self.o_offs, mask=self.o_mask)

    @gluon.jit
    def _async_wait(self, count):
        if self.cfg.SCALE_PRESHUFFLED:
            tdm.async_wait(count * 2)
        else:
            tdm.async_wait(count)
            cp.wait_group(count)

    @gluon.jit
    def _downcast_fp32_to_mxfp8(self, x, x_format: ttgl.constexpr, shape: ttgl.constexpr):
        block_size: ttgl.constexpr = 32
        outer_dim: ttgl.constexpr = shape[0]
        inner_dim: ttgl.constexpr = shape[1]

        ttgl.static_assert(x_format == 'e4m3' or x_format == 'e5m2')
        dtype: ttgl.constexpr = ttgl.float8e4nv if x_format == 'e4m3' else ttgl.float8e5
        fp8_max: ttgl.constexpr = 57344.0 if dtype == 'e5m2' else 448.0

        ttgl.static_assert(x.dtype == ttgl.float32)
        x = ttgl.reshape(x, [outer_dim, inner_dim // block_size, block_size])
        x_abs = ttgl.abs(x)
        x_max = ttgl.max(x_abs, axis=2)

        dequant_scale = x_max / fp8_max
        dequant_scale = (dequant_scale.to(ttgl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000

        dequant_scale_fp32 = dequant_scale.to(ttgl.float32, bitcast=True)
        quant_scale = ttgl.where(dequant_scale_fp32 == 0.0, 0, 1.0 / dequant_scale_fp32)

        x = x * quant_scale[:, :, None]
        x = ttgl.reshape(x, [outer_dim, inner_dim])
        x = x.to(dtype)

        dequant_scale = (dequant_scale >> 23).to(ttgl.uint8)
        return x, dequant_scale

    @gluon.jit
    def _downcast_fp32_to_fp8(self, x, x_format: ttgl.constexpr):
        if x_format == 'e4m3':
            return x.to(ttgl.float8e4nv)
        else:
            assert x_format == 'e5m2'
            return x.to(ttgl.float8e5)

    @gluon.jit
    def _unshuffle_scale_subview(self, buffer, non_k_dim, k_dim, NON_K_PRESHUFFLE_BLOCK_SIZE):
        BLOCK_NONK_PRESHUFFLED: ttgl.constexpr = non_k_dim // NON_K_PRESHUFFLE_BLOCK_SIZE
        SCALE_KWIDTH: ttgl.constexpr = 4 if k_dim >= 4 else k_dim
        return buffer.reshape((
            BLOCK_NONK_PRESHUFFLED,  #
            k_dim // SCALE_KWIDTH,  #
            NON_K_PRESHUFFLE_BLOCK_SIZE // 4,  #
            4,  #
            SCALE_KWIDTH  #
        )).permute((0, 3, 2, 1, 4)).reshape((non_k_dim, k_dim))


@gluon.jit
def get_program(q_ptr, k_ptr, v_ptr,  #
                q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
                o_ptr,  #
                sm_scale,  #
                Q_TYPE: ttgl.constexpr,  #
                KV_TYPE: ttgl.constexpr,  #
                SEQLEN_Q: ttgl.constexpr,  #
                SEQLEN_K: ttgl.constexpr,  #
                NUM_Q_HEADS: ttgl.constexpr,  #
                NUM_K_HEADS: ttgl.constexpr,  #
                HEAD_SZ: ttgl.constexpr,  #
                BLOCK_M: ttgl.constexpr,  #
                BLOCK_N: ttgl.constexpr,  #
                BLOCK_SCALING: ttgl.constexpr,  #
                SCALE_PRESHUFFLED: ttgl.constexpr,  #
                P_SCALING: ttgl.constexpr,  #
                NUM_BUFFERS: ttgl.constexpr):
    NUM_WARPS: ttgl.constexpr = ttgl.num_warps()
    if BLOCK_SCALING:
        cfg = BlockScaledAttentionConfig(  #
            Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_WARPS,
            P_SCALING, SCALE_PRESHUFFLED, NUM_BUFFERS)
        pgm = BlockScaledAttentionProgram.initialize(  #
            cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale)
    else:
        cfg = GlobalScaledAttentionConfig(  #
            Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_WARPS,
            NUM_BUFFERS)
        pgm = GlobalScaledAttentionProgram.initialize(  #
            cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale)
    return pgm


# ===-----------------------------------------------------------------------===#
# Gluon Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_kernel(q_ptr, k_ptr, v_ptr,  #
                    q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
                    o_ptr,  #
                    sm_scale,  #
                    Q_TYPE: ttgl.constexpr,  #
                    KV_TYPE: ttgl.constexpr,  #
                    SEQLEN_Q: ttgl.constexpr,  #
                    SEQLEN_K: ttgl.constexpr,  #
                    NUM_Q_HEADS: ttgl.constexpr,  #
                    NUM_K_HEADS: ttgl.constexpr,  #
                    HEAD_SZ: ttgl.constexpr,  #
                    BLOCK_M: ttgl.constexpr,  #
                    BLOCK_N: ttgl.constexpr,  #
                    BLOCK_SCALING: ttgl.constexpr,  #
                    SCALE_PRESHUFFLED: ttgl.constexpr,  #
                    P_SCALING: ttgl.constexpr):
    end = ttgl.cdiv(SEQLEN_K, BLOCK_N)

    # init program
    pgm = get_program(  #
        q_ptr, k_ptr, v_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, o_ptr, sm_scale,  #
        Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, BLOCK_SCALING,
        SCALE_PRESHUFFLED, P_SCALING, NUM_BUFFERS=1)
    cfg = pgm.cfg

    # init accumulator and softmax state
    m_i = ttgl.full([BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    l_i = ttgl.full([BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    acc = ttgl.full([BLOCK_M, HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

    for i in range(0, end):
        pgm.issue_global_load_k(i, buf=0)
        k, k_scale = pgm.shared_load_k(buf=0, wait_count=0)
        p = pgm.compute_qk(k, k_scale)
        p, alpha, m_i = pgm.softmax0(p, m_i)
        p, p_scale, acc, l_i = pgm.softmax1(p, alpha, acc, l_i)
        pgm.issue_global_load_v(i, buf=0)
        v, v_scale = pgm.shared_load_v(buf=0, wait_count=0)
        acc = pgm.compute_pv(p, p_scale, v, v_scale, acc)

    acc = acc / l_i[:, None]
    pgm.store_output(acc)


@gluon.jit
def attn_fwd_pipelined_kernel(q_ptr, k_ptr, v_ptr,  #
                              q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
                              o_ptr,  #
                              sm_scale,  #
                              Q_TYPE: ttgl.constexpr,  #
                              KV_TYPE: ttgl.constexpr,  #
                              SEQLEN_Q: ttgl.constexpr,  #
                              SEQLEN_K: ttgl.constexpr,  #
                              NUM_Q_HEADS: ttgl.constexpr,  #
                              NUM_K_HEADS: ttgl.constexpr,  #
                              HEAD_SZ: ttgl.constexpr,  #
                              BLOCK_M: ttgl.constexpr,  #
                              BLOCK_N: ttgl.constexpr,  #
                              BLOCK_SCALING: ttgl.constexpr,  #
                              SCALE_PRESHUFFLED: ttgl.constexpr,  #
                              P_SCALING: ttgl.constexpr):
    end = ttgl.cdiv(SEQLEN_K, BLOCK_N)

    # init program
    pgm = get_program(  #
        q_ptr, k_ptr, v_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, o_ptr, sm_scale,  #
        Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, BLOCK_SCALING,
        SCALE_PRESHUFFLED, P_SCALING, NUM_BUFFERS=2)
    cfg = pgm.cfg

    # init accumulator and softmax state
    m_i = ttgl.full([BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    l_i = ttgl.full([BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
    acc = ttgl.full([BLOCK_M, HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

    # pipeline prologue, loop -3
    pgm.issue_global_load_k(0, buf=0)

    # pipeline prologue, loop -2
    pgm.issue_global_load_k(1, buf=1)

    k, k_scale = pgm.shared_load_k(buf=0, wait_count=1)

    pgm.issue_global_load_v(0, buf=0)

    # pipeline prologue, loop -1
    qk = pgm.compute_qk(k, k_scale)

    pgm.issue_global_load_k(2, buf=0)

    p, alpha, m_i = pgm.softmax0(qk, m_i)
    k, k_scale = pgm.shared_load_k(buf=1, wait_count=2)

    pgm.issue_global_load_v(1, buf=1)

    # main loop, loop 0 to end-3, unrolled by 2
    for i in range(0, end - 2, 2):
        # loop i
        qk = pgm.compute_qk(k, k_scale)
        p, p_scale, acc, l_i = pgm.softmax1(p, alpha, acc, l_i)
        v, v_scale = pgm.shared_load_v(buf=0, wait_count=2)

        pgm.issue_global_load_k(i + 3, buf=1)

        acc = pgm.compute_pv(p, p_scale, v, v_scale, acc)
        p, alpha, m_i = pgm.softmax0(qk, m_i)
        k, k_scale = pgm.shared_load_k(buf=0, wait_count=2)

        pgm.issue_global_load_v(i + 2, buf=0)

        # loop i+1
        qk = pgm.compute_qk(k, k_scale)
        p, p_scale, acc, l_i = pgm.softmax1(p, alpha, acc, l_i)
        v, v_scale = pgm.shared_load_v(buf=1, wait_count=2)

        if i + 4 < end:
            pgm.issue_global_load_k(i + 4, buf=0)

        acc = pgm.compute_pv(p, p_scale, v, v_scale, acc)
        p, alpha, m_i = pgm.softmax0(qk, m_i)
        k, k_scale = pgm.shared_load_k(buf=1, wait_count=2)

        pgm.issue_global_load_v(i + 3, buf=1)

    # pipeline epilogue, loop end-2
    qk = pgm.compute_qk(k, k_scale)
    p, p_scale, acc, l_i = pgm.softmax1(p, alpha, acc, l_i)
    v, v_scale = pgm.shared_load_v(buf=0, wait_count=1)

    acc = pgm.compute_pv(p, p_scale, v, v_scale, acc)
    p, alpha, m_i = pgm.softmax0(qk, m_i)

    # pipeline epilogue, loop end-1
    p, p_scale, acc, l_i = pgm.softmax1(p, alpha, acc, l_i)
    v, v_scale = pgm.shared_load_v(buf=1, wait_count=0)

    acc = pgm.compute_pv(p, p_scale, v, v_scale, acc)

    # write output
    acc = acc / l_i[:, None]
    pgm.store_output(acc)


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


def attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
             q_scale: torch.Tensor | int, k_scale: torch.Tensor | int, v_scale: torch.Tensor | int,  #
             q_type: str, kv_type: str, block_m: int, block_n: int,  #
             block_scaling: bool, pipelined: bool, scale_preshuffled: bool, p_scaling: bool = False):
    batch, seqlen_q, num_q_heads, head_sz = q.shape
    _, seqlen_k, num_k_heads, _ = k.shape
    sm_scale = head_sz**(-0.5) * 1.4426950408889634  # 1 / ln(2)

    # q: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ]
    # k: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ / KV_PACK_DIV]
    # v: [BATCH, NUM_K_HEADS, HEAD_SZ, SEQLEN_K / KV_PACK_DIV]
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 3, 1).contiguous()
    if block_scaling:
        # q_scale: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ / 32]
        q_scale = q_scale.permute(0, 2, 1, 3).contiguous()
        if scale_preshuffled:
            # k_scale:              [BATCH, NUM_K_HEADS, SEQLEN_K / 128, HEAD_SZ * 4]
            # v_scale(head_sz=128): [BATCH, NUM_K_HEADS, HEAD_SZ / 128, SEQLEN_K * 4]
            # v_scale(head_sz=64):  [BATCH, NUM_K_HEADS, HEAD_SZ / 64, SEQLEN_K * 2]
            k_scale = _preshuffle_scale(k_scale.permute(0, 2, 1, 3).contiguous())
            v_scale = _preshuffle_scale(
                v_scale.permute(0, 2, 3, 1).contiguous(), _get_v_scale_preshuffle_factor(head_sz))
        else:
            # NOTE: We transposed the last 2 dims of k_scale and v_scale for contiguous elements in async copy.
            # k_scale: [BATCH, NUM_K_HEADS, HEAD_SZ / 32, SEQLEN_K]
            # v_scale: [BATCH, NUM_K_HEADS, SEQLEN_K / 32, HEAD_SZ]
            k_scale = k_scale.permute(0, 2, 3, 1).contiguous()
            v_scale = v_scale.permute(0, 2, 1, 3).contiguous()
    else:
        assert scale_preshuffled is False
    # o: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ]
    o = torch.zeros_like(q, dtype=torch.float32)

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    if block_scaling:
        q_scale = q_scale.cuda()
        k_scale = k_scale.cuda()
        v_scale = v_scale.cuda()
    o = o.cuda()

    # Use (NUM_Q_HEADS, NUM_BLOCKS, BATCH) for better xcd locality
    grid = (num_q_heads, cdiv(seqlen_q, block_m), batch)
    kargs = [
        q, k, v, q_scale, k_scale, v_scale, o, sm_scale,  #
        q_type, kv_type, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, block_m, block_n,  #
        block_scaling, scale_preshuffled, p_scaling
    ]
    if pipelined:
        assert cdiv(seqlen_k, block_n) > 4
        assert cdiv(seqlen_k, block_n) % 2 == 0
        kernel = attn_fwd_pipelined_kernel[grid](*kargs, num_warps=4, waves_per_eu=1)
    else:
        kernel = attn_fwd_kernel[grid](*kargs, num_warps=4, waves_per_eu=1)

    return o.cpu().permute(0, 2, 1, 3), kernel


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def _attn_fwd_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
                  q_scale: torch.Tensor | float, k_scale: torch.Tensor | float,
                  v_scale: torch.Tensor | float) -> torch.Tensor:

    q = q * q_scale
    k = k * k_scale
    v = v * v_scale

    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]

    scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output


def _create_operand(dtype: str, b: int, s: int, h: int, d: int, pack_dim: int = -1):
    size = (b, s, h, d)
    if dtype == 'e4m3':
        sig = torch.randint(0, 2, size, dtype=torch.uint8)
        exp = torch.randint(0, 2**4, size, dtype=torch.uint8)
        man = torch.randint(0, 2**3, size, dtype=torch.uint8)
        v = ((sig << 7) | (exp << 3) | man).type(torch.uint8)
        v[(exp << 3) | man == 0x7F] = 0x00  # avoid NaN
        v = v.view(torch.float8_e4m3fn)
        v_ref = v.view(torch.float8_e4m3fn).to(torch.float32)
    elif dtype == 'e5m2':
        sig = torch.randint(0, 2, size, dtype=torch.uint8)
        exp = torch.randint(0, 2**5, size, dtype=torch.uint8)
        man = torch.randint(0, 2**2, size, dtype=torch.uint8)
        v = ((sig << 7) | (exp << 2) | man).type(torch.uint8)
        v[(exp << 2) | man >= 0x7C] = 0x00  # avoid NaN and Inf
        v = v.view(torch.float8_e5m2)
        v_ref = v.view(torch.float8_e5m2).to(torch.float32)
    else:
        assert dtype == 'e2m1'
        assert pack_dim >= 0
        v_mxfp4 = MXFP4Tensor(size=size).random()
        v = v_mxfp4.to_packed_tensor(pack_dim)
        v_ref = v_mxfp4.to(torch.float32)
    return v, v_ref


def _create_block_scale(dtype: str, b: int, s: int, h: int, d: int, scale_dim: int):
    # Limit scale to an empirical range for accuracy
    if dtype == 'e4m3':
        low, high = 1 / 16, 2
    elif dtype == 'e5m2':
        low, high = 1 / 16, 2
    else:
        assert dtype == 'e2m1'
        low, high = 1 / 4, 16
    size = [b, s, h, d]
    size[scale_dim] //= 32
    scale = MXScaleTensor(size=tuple(size)).random(low, high)
    scale_ref = scale.to(torch.float32).repeat_interleave(32, dim=scale_dim)
    return scale.data, scale_ref


def _create_global_scale(dtype: str):
    assert dtype in ['e4m3', 'e5m2']
    low, high = (0x7F - 1), 0x7F + 1
    scale = torch.randint(low, high + 1, (), dtype=torch.uint8).item()
    scale_ref = 2**(scale - 0x7F)
    return scale, scale_ref


def static_profile(kernel):
    amdgcn = kernel.asm['amdgcn']

    sgpr_count = int(re.search(r'\.sgpr_count:\s+(\d+)', amdgcn).group(1))
    sgpr_spill_count = int(re.search(r'\.sgpr_spill_count:\s+(\d+)', amdgcn).group(1))
    vgpr_count = int(re.search(r'\.vgpr_count:\s+(\d+)', amdgcn).group(1))
    vgpr_spill_count = int(re.search(r'\.vgpr_spill_count:\s+(\d+)', amdgcn).group(1))
    scratch_size = int(re.search(r';\s+ScratchSize:\s+(\d+)', amdgcn).group(1))
    code_len_in_byte = int(re.search(r';\s+codeLenInByte\s+=\s+(\d+)', amdgcn).group(1))
    occupancy = int(re.search(r';\s+Occupancy:\s+(\d+)', amdgcn).group(1))

    print(f"- sgpr_count: {sgpr_count}\n"
          f"- sgpr_spill_count: {sgpr_spill_count}\n"
          f"- vgpr_count: {vgpr_count}\n"
          f"- vgpr_spill_count: {vgpr_spill_count}\n"
          f"- scratch_size: {scratch_size}\n"
          f"- code_len_in_byte: {code_len_in_byte}\n"
          f"- occupancy: {occupancy}\n")


@pytest.mark.parametrize("q_type,kv_type", [("e4m3", "e4m3"), ("e4m3", "e2m1")])
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen_q", [256])
@pytest.mark.parametrize("seqlen_k", [1024])
@pytest.mark.parametrize("num_q_heads,num_k_heads", [(1, 1), (4, 1), (4, 2)])
@pytest.mark.parametrize("head_sz", [64, 128])
@pytest.mark.parametrize("block_m", [128])
@pytest.mark.parametrize("block_n", [128])
@pytest.mark.parametrize("pipelined", [False, True])
@pytest.mark.parametrize("scale_preshuffled", [False, True])
def test_block_scaled_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, block_m,
                               block_n, pipelined, scale_preshuffled):
    q, q_ref = _create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
    k, k_ref = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=3)
    v, v_ref = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=1)
    q_scale, q_scale_ref = _create_block_scale(q_type, batch, seqlen_q, num_q_heads, head_sz, scale_dim=3)
    k_scale, k_scale_ref = _create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=3)
    v_scale, v_scale_ref = _create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=1)

    o, kernel = attn_fwd(q, k, v, q_scale, k_scale, v_scale, q_type, kv_type, block_m, block_n,  #
                         block_scaling=True, pipelined=pipelined, scale_preshuffled=scale_preshuffled, p_scaling=False)
    o = o.to(torch.float32)

    o_ref = _attn_fwd_ref(q_ref, k_ref, v_ref, q_scale_ref, k_scale_ref, v_scale_ref)
    o_ref = o_ref.to(torch.float32)

    amdgcn = kernel.asm['amdgcn']

    # check use correct wmma scaled instruction
    wmma_instrs = re.search(r'v_wmma_[^ ]+', amdgcn)
    for instr in wmma_instrs.groups():
        assert instr == 'v_wmma_scale_f32_16x16x128_f8f6f4'

    # check there is no convert layout for P via shared memory
    ds_store_instrs = re.findall(r'ds_store_[^ ]+', amdgcn)
    assert len(ds_store_instrs) == 0

    # TODO: Reenable this for scale preshuffling after tweaking layouts
    if not scale_preshuffled:
        # check use non-transposed load of k, v and transposed load of k_scale, v_scale from shared memory
        ds_load_instrs = re.findall(r'ds_load_[^ ]+', amdgcn)
        assert set(ds_load_instrs) == {'ds_load_tr8_b64', 'ds_load_b128'}

    # check async global load is vectorized with scale preshuffling
    if scale_preshuffled:
        async_load_instrs = re.findall(r'global_load_async_to_lds_[^ ]+', amdgcn)
        if head_sz == 128:
            for instr in async_load_instrs:
                assert instr == "global_load_async_to_lds_b128"
        elif head_sz == 64:
            for instr in async_load_instrs:
                assert instr == "global_load_async_to_lds_b64"

    # check output correctness
    matches = torch.isclose(o, o_ref, atol=0.1, rtol=0.1)
    total = o.numel()
    mismatches = total - matches.sum().item()
    mismatch_ratio = mismatches / total
    assert mismatches < 10, f"Mismatched elements: {mismatches} / {total} ({mismatch_ratio:.6%})"


@pytest.mark.parametrize("q_type,kv_type", [("e4m3", "e4m3")])
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen_q", [256])
@pytest.mark.parametrize("seqlen_k", [1024])
@pytest.mark.parametrize("num_q_heads,num_k_heads", [(1, 1), (4, 1), (4, 2)])
@pytest.mark.parametrize("head_sz", [64, 128])
@pytest.mark.parametrize("block_m", [128])
@pytest.mark.parametrize("block_n", [128])
@pytest.mark.parametrize("pipelined", [False, True])
def test_global_scaled_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, block_m,
                                block_n, pipelined):
    q, q_ref = _create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
    k, k_ref = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz)
    v, v_ref = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz)
    q_scale, q_scale_ref = _create_global_scale(q_type)
    k_scale, k_scale_ref = _create_global_scale(kv_type)
    v_scale, v_scale_ref = _create_global_scale(kv_type)

    o, kernel = attn_fwd(q, k, v, q_scale, k_scale, v_scale, q_type, kv_type, block_m, block_n,  #
                         block_scaling=False, pipelined=pipelined, scale_preshuffled=False)
    o = o.to(torch.float32)

    o_ref = _attn_fwd_ref(q_ref, k_ref, v_ref, q_scale_ref, k_scale_ref, v_scale_ref)
    o_ref = o_ref.to(torch.float32)

    amdgcn = kernel.asm['amdgcn']

    # check use correct wmma scaled instruction
    wmma_instrs = re.findall(r'v_wmma_[^ ]+', amdgcn)
    assert len(wmma_instrs) > 0 and all(instr == 'v_wmma_scale_f32_16x16x128_f8f6f4' for instr in wmma_instrs)

    # check there is no convert layout for p via shared memory
    ds_store_instrs = re.findall(r'ds_store_[^ ]+', amdgcn)
    assert len(ds_store_instrs) == 0

    # check always use non-transposed load of k and v from shared memory
    ds_load_instrs = re.findall(r'ds_load_[^ ]+', amdgcn)
    assert all(instr == 'ds_load_b128' for instr in ds_load_instrs)

    # check output correctness
    matches = torch.isclose(o, o_ref, atol=0.25, rtol=0.25)
    total = o.numel()
    mismatches = total - matches.sum().item()
    mismatch_ratio = mismatches / total
    assert mismatches < 10, f"Mismatched elements: {mismatches} / {total} ({mismatch_ratio:.6%})"


if __name__ == "__main__":

    def launch(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, block_m, block_n,
               scale_type, pipelined, scale_preshuffled, disable_p_scaling):
        q, _ = _create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
        k, _ = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=3)
        v, _ = _create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=1)
        if scale_type == 'block':
            q_scale, _ = _create_block_scale(q_type, batch, seqlen_q, num_q_heads, head_sz, scale_dim=3)
            k_scale, _ = _create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=3)
            v_scale, _ = _create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=1)
        else:
            assert scale_type == 'global'
            q_scale, _ = _create_global_scale(q_type)
            k_scale, _ = _create_global_scale(kv_type)
            v_scale, _ = _create_global_scale(kv_type)

        _, kernel = attn_fwd(q, k, v, q_scale, k_scale, v_scale, q_type, kv_type, block_m, block_n,  #
                             scale_type == 'block', pipelined, scale_preshuffled, not disable_p_scaling)
        static_profile(kernel)

    parser = argparse.ArgumentParser()
    parser.add_argument("--q_type", type=str, choices=['e4m3', 'e5m2'], required=True)
    parser.add_argument("--kv_type", type=str, choices=['e4m3', 'e5m2', 'e2m1'], required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seqlen_q", type=int, required=True)
    parser.add_argument("--seqlen_k", type=int, required=True)
    parser.add_argument("--num_q_heads", type=int, required=True)
    parser.add_argument("--num_k_heads", type=int, required=True)
    parser.add_argument("--head_sz", type=int, required=True)
    parser.add_argument("--block_m", type=int, required=True)
    parser.add_argument("--block_n", type=int, required=True)
    parser.add_argument("--pipelined", action="store_true")
    parser.add_argument(
        "--scale_preshuffled", action="store_true",
        help="When set, we will preshuffle the K/V scales before passing to the kernel. "
        "Only works for block scaling.")
    parser.add_argument(
        "--disable_p_scaling", action="store_true", help="When set, we will use a fixed scale of 1.0 for all P blocks. "
        "Otherwise, we will compute and apply per-block scaling for the P matrix tensor. "
        "Only apply when block scaling is enabled. Ignored for global scaling.")
    parser.add_argument(
        "--scale_type", type=str, choices=['block', 'global'], required=True,
        help="`block` = use block scaling where 32 elements share a scale; "
        "`global` = use a single global scale for all elements")
    args = parser.parse_args()
    args = vars(args)
    launch(**args)
