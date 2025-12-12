"""
Multi-head attention kernel in Gluon
"""
# ruff: noqa: E402
import hip

# Needed for internal dev flow for now; will remove later
hip.hip.hipInit(0)

import os
import sys
import inspect
import argparse
import re
import pytest
import torch
import math

from triton import cdiv
from triton.language.core import _aggregate as aggregate
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

from triton.experimental.gluon.language.amd.gfx1250 import wmma_scaled
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.experimental.gluon.language.amd.gfx1250 import buffer_load, buffer_store
from triton.experimental.gluon.language.amd.gfx1250 import async_copy as cp

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
def get_padded_shared_layout(shape, transposed=False):
    """ Get a padded shared layout without back conflict for a given tensor shape. """
    _, inner_dim = shape
    ## Here we assume the elements in LDS is 8-bit (for mxfp4, 2 mxfp4
    ## are packed in 1 8-bit elements). Then 256 elements can occupy
    ## 64 banks. Therefore, we want the padding_interval to be at
    ## least 256 elements.
    ## On the other hand, we only need to add padding after a row of
    ## elements. So we also want the padding_interval to be at least inner_dim.
    padding_interval = max(inner_dim, 256)
    ## For K tensor, we use ds_load_b128 and 16 x 8-bit element is the vector size
    ## For V tensor, there are 3 cases
    ## 1. V is HEAD_SZ contiguous. In this case, ds_load_tr8_b64 is
    ##    used. And the padding_amount should be the number of elements
    ##    from 2 threads, i.e. 16 elements.
    ## 2. V is seq_len contiguous and kWidth=16. In this case,
    ##    ds_load_b128 is used, and padding_amount should be 16 as for K tensor.
    ## 3. V is seq_len contiguous and kWidth=8. In this case,
    ##    ds_load_b64 is used. In this case, we can also use 16 as the padding_amount.
    padding_amount = 16
    return ttgl.PaddedSharedLayout.with_identity_for([[padding_interval, padding_amount]], shape, [1, 0])


@gluon.constexpr_function
def get_load_layout(shape, num_warps):
    """ Get a layout with better vectorized access for a given tensor shape. """
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


@aggregate
class MemoryBlock:
    """
    MemoryBlock groups variables to describe a block of 2D tensor in global memory.
    """
    dtype: ttgl.constexpr
    ptr: ttgl.tensor
    offs: ttgl.tensor
    mask: ttgl.tensor
    shape: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, ptr, offs, mask, shape):
        self.dtype = ttgl.constexpr(ptr.dtype.element_ty)
        self.ptr = ptr
        self.offs = offs
        self.mask = mask
        self.shape = ttgl.constexpr(shape)

    @gluon.jit
    def initialize(base, shape, block_shape, layout):
        ttgl.static_assert(len(block_shape) == 2 and len(shape) == 2)

        offs_m = ttgl.arange(0, block_shape[0], ttgl.SliceLayout(1, layout))
        offs_n = ttgl.arange(0, block_shape[1], ttgl.SliceLayout(0, layout))
        offs = offs_m[:, None] * shape[1] + offs_n[None, :]
        mask = (offs_m < shape[0])[:, None] & (offs_n < shape[1])[None, :]

        return MemoryBlock(base, offs, mask, block_shape)


@aggregate
class MemoryUnit:
    """
    MemoryUnit abstracts the logic of transferring data from global memory to shared memory for 2D tensor.
    It supports 2 methods:

    - `issue_tdm_load`: issue an async load via TDM from global memory to shared memory.
    - `issue_async_copy`: issue an async copy from global memory to shared memory.

    To help use a MemoryUnit in a loop, it supports load with an `idx` argument, meaning loading the `idx`-th block
    along the `axis` dimension. This requires the one dimension of the tensor shape equals to the block size, and we
    will slide the block along the other dimension.
    """
    smem: ttgl.shared_memory_descriptor
    desc: tdm.tensor_descriptor
    block: MemoryBlock

    strides: ttgl.constexpr
    axis: ttgl.constexpr
    sub_axis: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, smem, desc, block,  #
                 strides, axis, sub_axis):
        self.smem = smem
        self.desc = desc
        self.block = block
        self.strides = ttgl.constexpr(strides)
        self.axis = ttgl.constexpr(axis)
        self.sub_axis = ttgl.constexpr(sub_axis)

    @gluon.jit
    def _compute_axis_offset(self, idx, sub_idx):
        axis: ttgl.constexpr = self.axis
        sub_axis: ttgl.constexpr = self.sub_axis

        if sub_axis is None:
            step: ttgl.constexpr = self.block.shape[axis]
            off = [idx * step, 0] if axis == 0 else [0, idx * step]
        else:
            step: ttgl.constexpr = self.block.shape[axis]
            if sub_axis == axis:
                step *= 2
            off = [idx * step, 0] if axis == 0 else [0, idx * step]

            sub_step: ttgl.constexpr = self.block.shape[sub_axis]
            off = [off[0] + sub_idx * sub_step, off[1]] if sub_axis == 0 else \
                  [off[0], off[1] + sub_idx * sub_step]

        return off

    @gluon.jit
    def issue_tdm_load(self, idx, sub_idx=0, buf=0, pred=True):
        axis_off = self._compute_axis_offset(idx, sub_idx)
        num_subtile: ttgl.constexpr = 2 if self.sub_axis is not None else 1
        smem = self.smem.index(buf * num_subtile + sub_idx)
        tdm.async_load(self.desc, axis_off, smem, pred)

    @gluon.jit
    def issue_async_copy(self, idx, sub_idx=0, buf=0):
        axis_off = self._compute_axis_offset(idx, sub_idx)
        off = axis_off[0] * self.strides[0] + axis_off[1] * self.strides[1]
        num_subtile: ttgl.constexpr = 2 if self.sub_axis is not None else 1
        smem = self.smem.index(buf * num_subtile + sub_idx)
        cp.global_to_shared(smem, self.block.ptr + off + self.block.offs)
        cp.commit_group()

    @gluon.jit
    def initialize(base, shape, block_shape, layout, smem_layout, num_buffers=1,  #
                   sub_axis=None):
        ttgl.static_assert(len(block_shape) == 2 and len(shape) == 2)

        dtype: ttgl.constexpr = base.dtype.element_ty

        ttgl.static_assert(block_shape[0] <= shape[0] and block_shape[1] <= shape[1])
        if shape[0] > block_shape[0]:
            ttgl.static_assert(shape[1] == block_shape[1])
            axis: ttgl.constexpr = 0
        else:
            axis: ttgl.constexpr = 1

        sub_block_m: ttgl.constexpr = block_shape[0] if sub_axis != 0 else block_shape[0] // 2
        sub_block_n: ttgl.constexpr = block_shape[1] if sub_axis != 1 else block_shape[1] // 2
        num_subtile: ttgl.constexpr = 2 if sub_axis is not None else 1

        desc = tdm.make_tensor_descriptor(  #
            base=base,  #
            shape=shape,  #
            strides=[shape[1], 1],  #
            block_shape=[sub_block_m, sub_block_n],  #
            layout=smem_layout)
        block = MemoryBlock.initialize(base, shape, [sub_block_m, sub_block_n], layout)
        smem = ttgl.allocate_shared_memory(  #
            dtype,  #
            [num_buffers * num_subtile] + [sub_block_m, sub_block_n],  #
            smem_layout)

        return MemoryUnit(smem, desc, block,  #
                          [shape[1], 1], axis, sub_axis)


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
    NUM_BUFFERS: ttgl.constexpr
    NUM_WARPS: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N,
                 NUM_BUFFERS, NUM_WARPS):
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
        self.NUM_BUFFERS = ttgl.constexpr(NUM_BUFFERS)
        self.NUM_WARPS = ttgl.constexpr(NUM_WARPS)


# ===-----------------------------------------------------------------------===#
# Global Scaled Attention Program
# ===-----------------------------------------------------------------------===#


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

    # Whether the layout convert between QK and P is trivial - no data movement. This can happen when we use
    # k_width=8 for P and V, which effectively makes QK and P have the same layout.
    CONVERT_LAYOUT_TRIVIAL: ttgl.constexpr
    # Whether to subtile K and V
    SUBTILE: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N,
                 P_K_WIDTH, SUBTILE, NUM_BUFFERS, NUM_WARPS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert KV_TYPE in ['e5m2', 'e4m3']
        assert NUM_WARPS == 4 or NUM_WARPS == 8
        assert P_K_WIDTH == 16 or P_K_WIDTH == 8

        self.base = AttentionConfigBase(Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M,
                                        BLOCK_N, NUM_BUFFERS, NUM_WARPS)

        wmma_layout: ttgl.constexpr = ttgl.amd.AMDWMMALayout(  #
            version=3, transposed=True, warps_per_cta=[NUM_WARPS, 1], instr_shape=[16, 16, 128])
        self.q_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, 16))
        self.k_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, 16))
        self.p_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, P_K_WIDTH))
        self.v_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, P_K_WIDTH))
        self.k_smem_layout = ttgl.constexpr(get_padded_shared_layout([BLOCK_N, HEAD_SZ]))
        self.v_smem_layout = ttgl.constexpr(get_padded_shared_layout([BLOCK_N, HEAD_SZ], transposed=True))
        if SUBTILE:
            self.k_smem_layout = ttgl.constexpr(get_padded_shared_layout([BLOCK_N // 2, HEAD_SZ]))
            self.v_smem_layout = ttgl.constexpr(get_padded_shared_layout([BLOCK_N, HEAD_SZ // 2], transposed=True))
        self.acc_layout = ttgl.constexpr(wmma_layout)

        self.CONVERT_LAYOUT_TRIVIAL = ttgl.constexpr(True if P_K_WIDTH == 8 else False)
        self.SUBTILE = ttgl.constexpr(SUBTILE)


@aggregate
class GlobalScaledAttentionProgram:
    cfg: GlobalScaledAttentionConfig

    q: ttgl.tensor
    q_scale: ttgl.tensor
    k_mem: MemoryUnit
    k_scale: ttgl.tensor
    v_mem: MemoryUnit
    v_scale: ttgl.tensor
    o_blk: MemoryBlock
    # TODO: sm_scale should be a constexpr but the current llvm can not properly
    # fuse v_fma for literal operands, so we are using tensor here to ensure
    # it is in a register. Change it back to constexpr once the llvm is fixed.
    sm_scale: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg,  #
                 q, q_scale,  #
                 k_mem, k_scale,  #
                 v_mem, v_scale,  #
                 o_blk,  #
                 sm_scale):
        self.cfg = cfg
        self.q = q
        self.q_scale = q_scale
        self.k_mem = k_mem
        self.k_scale = k_scale
        self.v_mem = v_mem
        self.v_scale = v_scale
        self.o_blk = o_blk
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
        SUBTILE: ttgl.constexpr = cfg.SUBTILE

        off_h = ttgl.program_id(0)  # NUM_Q_HEADS
        off_m = ttgl.program_id(1)  # NUM_BLOCKS
        off_z = ttgl.program_id(2)  # BATCH

        ttgl.static_assert(NUM_Q_HEADS % NUM_K_HEADS == 0)
        group_sz: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
        off_hk = off_h // group_sz

        q_off = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h) +\
                BLOCK_M * off_m * HEAD_SZ
        q_blk = MemoryBlock.initialize(  #
            q_ptr + q_off,  #
            shape=[SEQLEN_Q, HEAD_SZ],  #
            block_shape=[BLOCK_M, HEAD_SZ],  #
            layout=cfg.q_layout)

        k_off = SEQLEN_K * HEAD_SZ * (NUM_K_HEADS * off_z + off_hk)
        k_mem = MemoryUnit.initialize(  #
            base=k_ptr + k_off,  #
            shape=[SEQLEN_K, HEAD_SZ],  #
            block_shape=[BLOCK_N, HEAD_SZ],  #
            layout=cfg.k_layout,  #
            smem_layout=cfg.k_smem_layout,  #
            num_buffers=NUM_BUFFERS,  #
            sub_axis=0 if SUBTILE else None)

        v_mem = MemoryUnit.initialize(  #
            base=v_ptr + k_off,  #
            shape=[SEQLEN_K, HEAD_SZ],  #
            block_shape=[BLOCK_N, HEAD_SZ],  #
            layout=cfg.v_layout,  #
            smem_layout=cfg.v_smem_layout,  #
            num_buffers=NUM_BUFFERS,  #
            sub_axis=1 if SUBTILE else None)

        o_blk = MemoryBlock.initialize(  #
            o_ptr + q_off,  #
            shape=[SEQLEN_Q, HEAD_SZ],  #
            block_shape=[BLOCK_M, HEAD_SZ],  #
            layout=cfg.acc_layout)

        q = buffer_load(q_blk.ptr, q_blk.offs, q_blk.mask, other=0.0)

        return GlobalScaledAttentionProgram(  #
            cfg,  #
            q, q_scale,  #
            k_mem, k_scale,  #
            v_mem, v_scale,  #
            o_blk,  #
            sm_scale)

    @gluon.jit
    def issue_global_load_k(self, idx, sub_idx=0, buf=0, pred=True):
        self.k_mem.issue_tdm_load(idx, sub_idx, buf, pred)

    @gluon.jit
    def issue_global_load_v(self, idx, sub_idx=0, buf=0, pred=True):
        self.v_mem.issue_tdm_load(idx, sub_idx, buf, pred)

    @gluon.jit
    def shared_load_k(self, sub_idx=0, buf=0):
        cfg = self.cfg

        k_buffer = self.k_mem.smem.index(buf).permute((1, 0))
        if cfg.SUBTILE:
            k_buffer = self.k_mem.smem.index(buf * 2 + sub_idx).permute((1, 0))
        k = k_buffer.load(cfg.k_layout)
        return k

    @gluon.jit
    def shared_load_v(self, sub_idx=0, buf=0):
        cfg = self.cfg

        v_buffer = self.v_mem.smem.index(buf)
        if cfg.SUBTILE:
            v_buffer = self.v_mem.smem.index(buf * 2 + sub_idx)
        v = v_buffer.load(cfg.v_layout)
        return v

    @gluon.jit
    def compute_qk(self, k, k_scale, acc):
        cfg = self.cfg

        qk = wmma_scaled(self.q, self.q_scale, cfg.Q_TYPE, k, k_scale, cfg.KV_TYPE, acc)
        return qk

    @gluon.jit
    def compute_pv(self, p, p_scale, v, v_scale, acc):
        cfg = self.cfg

        acc = wmma_scaled(p, p_scale, cfg.P_TYPE, v, v_scale, cfg.KV_TYPE, acc)
        return acc

    @gluon.jit
    def downcast_p(self, p):
        cfg = self.cfg

        p = p.to(ttgl.float8e4nv if cfg.P_TYPE == 'e4m3' else ttgl.float8e5)
        p = ttgl.convert_layout(p, cfg.p_layout, cfg.CONVERT_LAYOUT_TRIVIAL)
        return p

    @gluon.jit
    def store_output(self, acc):
        o_blk = self.o_blk
        o = acc.to(o_blk.dtype)
        buffer_store(o, o_blk.ptr, o_blk.offs, o_blk.mask)

    @gluon.jit
    def concat_subtile(self, x, y):
        cfg = self.cfg
        layout: ttgl.constexpr = cfg.acc_layout
        shape: ttgl.constexpr = [x.shape[0], x.shape[1] + y.shape[1]]
        a = ttgl.join(x, y)
        a = a.permute(0, 2, 1).reshape(shape)
        a = ttgl.convert_layout(a, layout, assert_trivial=True)
        return a

    @gluon.jit
    def async_wait(self, count):
        tdm.async_wait(count)

    @gluon.jit
    def fwd_loop(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
        acc = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

        sm_scale = self.sm_scale
        k_scale = self.k_scale
        v_scale = self.v_scale
        p_scale = 0x7F

        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end):
            self.issue_global_load_k(i)

            self.async_wait(0)
            k = self.shared_load_k()

            qk = self.compute_qk(k, k_scale, zero)

            m = ttgl.max(qk, 1)
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
            p = ttgl.exp2(qk_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            m_i = m_ij
            alpha = ttgl.exp2(m_diff)
            l_ij = ttgl.sum(p, 1)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            p = self.downcast_p(p)

            self.issue_global_load_v(i)

            self.async_wait(0)
            v = self.shared_load_v()

            acc = self.compute_pv(p, p_scale, v, v_scale, acc)

        acc = acc / l_i[:, None]
        self.store_output(acc)

    @gluon.jit
    def fwd_loop_pipeline(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
        acc = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

        sm_scale = self.sm_scale
        k_scale = self.k_scale
        v_scale = self.v_scale
        p_scale = 0x7F

        # pipeline prologue, iter -3
        self.issue_global_load_k(0, buf=0)  # ................................. iter 0

        # pipeline prologue, iter -2
        self.issue_global_load_k(1, buf=1)  # ................................. iter 1

        self.async_wait(1)  # ................................................. iter 0
        k = self.shared_load_k(buf=0)
        self.issue_global_load_v(0, buf=0)  # ................................. iter 0

        # pipeline prologue, iter -1
        qk = self.compute_qk(k, k_scale, zero)  # ............................. iter 0

        self.issue_global_load_k(2, buf=0)  # ................................. iter 2

        m = ttgl.max(qk, 1)  # ................................................ iter 0
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        self.async_wait(2)  # ................................................. iter 0
        k = self.shared_load_k(buf=1)
        self.issue_global_load_v(1, buf=1)  # ................................. iter 1

        # main loop from 0 to end-3
        # TODO: Ideally we should unroll the loop by 2 to remove the buffer index
        # update, but our current codegen in llvm does not perform well. Re-enable
        # unroll when fixed.
        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 2
            b = 1 - a

            qk = self.compute_qk(k, k_scale, zero)  # ......................... iter i+1
            l_ij = ttgl.sum(p, 1)  # .......................................... iter i
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            p = self.downcast_p(p)

            self.async_wait(2)  # ............................................. iter i
            v = self.shared_load_v(buf=a)
            self.issue_global_load_k(i + 3, buf=b, pred=i != end - 3)  # ...... iter i+3

            acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ............. iter i
            m = ttgl.max(qk, 1)  # ............................................ iter i+1
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
            p = ttgl.exp2(qk_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            alpha = ttgl.exp2(m_diff)
            m_i = m_ij

            self.async_wait(2)  # ............................................. iter i+2
            k = self.shared_load_k(buf=a)
            self.issue_global_load_v(i + 2, buf=a)  # ......................... iter i+2

        # pipeline epilogue, iter end-2
        qk = self.compute_qk(k, k_scale, zero)  # ............................. iter end-1
        l_ij = ttgl.sum(p, 1)  # .............................................. iter end-2
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(2)  # ................................................. iter end-2
        v = self.shared_load_v(buf=0)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-2
        m = ttgl.max(qk, 1)  # ................................................ iter end-1
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        # pipeline epilogue, iter end-1
        l_ij = ttgl.sum(p, 1)  # .............................................. iter end-1
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(0)  # ................................................. iter end-1
        v = self.shared_load_v(buf=1)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-1

        # write output
        l_recip = 1 / l_i
        acc = acc * l_recip[:, None]
        self.store_output(acc)

    @gluon.jit
    def fwd_subtile(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc0 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc1 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)

        sm_scale = self.sm_scale
        k_scale = self.k_scale
        v_scale = self.v_scale
        p_scale = 0x7F

        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end):
            self.issue_global_load_k(i, sub_idx=0)
            self.issue_global_load_k(i, sub_idx=1)

            self.async_wait(0)
            k0 = self.shared_load_k(sub_idx=0)
            k1 = self.shared_load_k(sub_idx=1)

            qk0 = self.compute_qk(k0, k_scale, zero)
            qk1 = self.compute_qk(k1, k_scale, zero)

            qk = self.concat_subtile(qk0, qk1)
            m = ttgl.max(qk, 1)
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]
            qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
            p0 = ttgl.exp2(qk0_shifted)
            p1 = ttgl.exp2(qk1_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            m_i = m_ij
            alpha = ttgl.exp2(m_diff)
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            p = self.concat_subtile(p0, p1)
            l_ij = ttgl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            p = self.downcast_p(p)

            self.issue_global_load_v(i, sub_idx=0)
            self.issue_global_load_v(i, sub_idx=1)

            self.async_wait(0)
            v0 = self.shared_load_v(sub_idx=0)
            v1 = self.shared_load_v(sub_idx=1)

            acc0 = self.compute_pv(p, p_scale, v0, v_scale, acc0)
            acc1 = self.compute_pv(p, p_scale, v1, v_scale, acc1)

        acc = self.concat_subtile(acc0, acc1)
        acc = acc / l_i[:, None]
        self.store_output(acc)

    @gluon.jit
    def fwd_subtile_pipeline(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc0 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc1 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)

        sm_scale = self.sm_scale
        k_scale = self.k_scale
        v_scale = self.v_scale
        p_scale = 0x7F

        # pipeline prologue, iter -3
        self.issue_global_load_k(0, sub_idx=0, buf=0)  # ...................... iter 0

        self.issue_global_load_k(0, sub_idx=1, buf=0)  # ...................... iter 0

        # pipeline prologue, iter -2
        self.issue_global_load_k(1, sub_idx=0, buf=1)  # ...................... iter 1

        self.async_wait(2)  # ................................................. iter 0
        k0 = self.shared_load_k(sub_idx=0, buf=0)
        self.issue_global_load_k(1, sub_idx=1, buf=1)  # ...................... iter 1

        # pipeline prologue, iter -1
        qk0 = self.compute_qk(k0, k_scale, zero)  # ........................... iter 0
        self.async_wait(2)  # ................................................. iter 0
        k1 = self.shared_load_k(sub_idx=1, buf=0)
        self.issue_global_load_v(0, sub_idx=0, buf=0)  # ...................... iter 0

        qk1 = self.compute_qk(k1, k_scale, zero)  # ........................... iter 0
        self.issue_global_load_v(0, sub_idx=1, buf=0)  # ...................... iter 0

        qk = self.concat_subtile(qk0, qk1)  # ................................. iter 0
        m = ttgl.max(qk, 1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        self.issue_global_load_k(2, sub_idx=0, buf=0)  # ...................... iter 2

        self.async_wait(4)  # ................................................. iter 1
        k0 = self.shared_load_k(sub_idx=0, buf=1)
        qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]  # ................ iter 0
        qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
        p0 = ttgl.exp2(qk0_shifted)
        self.issue_global_load_k(2, sub_idx=1, buf=0)  # ...................... iter 2

        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 2
            b = 1 - a
            pred = (i != end - 3)

            qk0 = self.compute_qk(k0, k_scale, zero)  # ....................... iter i+1
            self.async_wait(4)  # ............................................. iter i+1
            k1 = self.shared_load_k(sub_idx=1, buf=b)
            p1 = ttgl.exp2(qk1_shifted)  # .................................... iter i
            m_diff = m_i * sm_scale - m_ij_scaled
            m_i = m_ij
            alpha = ttgl.exp2(m_diff)
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            self.issue_global_load_v(i + 1, sub_idx=0, buf=b)  # .............. iter i+1

            qk1 = self.compute_qk(k1, k_scale, zero)  # ....................... iter i+1
            self.async_wait(4)  # ............................................. iter i
            v0 = self.shared_load_v(sub_idx=0, buf=a)
            p = self.concat_subtile(p0, p1)  # ................................ iter i
            l_ij = ttgl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            p = self.downcast_p(p)
            self.issue_global_load_v(i + 1, sub_idx=1, buf=b)  # .............. iter i+1

            acc0 = self.compute_pv(p, p_scale, v0, v_scale, acc0)  # .......... iter i
            self.async_wait(4)  # ............................................. iter i
            v1 = self.shared_load_v(sub_idx=1, buf=a)
            qk = self.concat_subtile(qk0, qk1)  # ............................. iter i+1
            m = ttgl.max(qk, 1)
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            self.issue_global_load_k(i + 3, sub_idx=0, buf=b, pred=pred)  # ... iter i+3

            acc1 = self.compute_pv(p, p_scale, v1, v_scale, acc1)  # .......... iter i
            self.async_wait(4)  # ............................................. iter i+2
            k0 = self.shared_load_k(sub_idx=0, buf=a)
            qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]  # ............ iter i+1
            qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
            p0 = ttgl.exp2(qk0_shifted)
            self.issue_global_load_k(i + 3, sub_idx=1, buf=b, pred=pred)  # ... iter i+3

        # pipeline epilogue iter end-2
        self.issue_global_load_v(end - 1, sub_idx=0, buf=1)
        self.issue_global_load_v(end - 1, sub_idx=1, buf=1)

        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(2)
        v0 = self.shared_load_v(sub_idx=0, buf=0)
        v1 = self.shared_load_v(sub_idx=1, buf=0)

        acc0 = self.compute_pv(p, p_scale, v0, v_scale, acc0)
        acc1 = self.compute_pv(p, p_scale, v1, v_scale, acc1)

        # pipeline epilogue iter end-1
        qk0 = self.compute_qk(k0, k_scale, zero)
        k1 = self.shared_load_k(sub_idx=1, buf=1)
        qk1 = self.compute_qk(k1, k_scale, zero)

        qk = self.concat_subtile(qk0, qk1)
        m = ttgl.max(qk, 1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]
        qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
        p0 = ttgl.exp2(qk0_shifted)
        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(0)
        v0 = self.shared_load_v(sub_idx=0, buf=1)
        v1 = self.shared_load_v(sub_idx=1, buf=1)

        acc0 = self.compute_pv(p, p_scale, v0, v_scale, acc0)
        acc1 = self.compute_pv(p, p_scale, v1, v_scale, acc1)

        # write output
        acc = self.concat_subtile(acc0, acc1)
        l_recip = 1 / l_i
        acc = acc * l_recip[:, None]
        self.store_output(acc)


# ===-----------------------------------------------------------------------===#
# Block Scaled Attention Program
# ===-----------------------------------------------------------------------===#


@composition
@aggregate
class BlockScaledAttentionConfig:
    base: AttentionConfigBase

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

    # Whether to use per-block scaling for P; if False, use an uniform scale of 1.0.
    P_SCALING: ttgl.constexpr
    # Whether the layout convert between QK and P is trivial - no data movement. This can happen when we use
    # k_width=8 for P and V, which effectively makes QK and P have the same layout. But note we can use k_width=8 for
    # V when it is a mxfp4, so this only applies when KV_TYPE is not 'e2m1'.
    CONVERT_LAYOUT_TRIVIAL: ttgl.constexpr
    # Whether to subtile K and V
    SUBTILE: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N,
                 P_SCALING, P_K_WIDTH, SUBTILE, NUM_BUFFERS, NUM_WARPS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert KV_TYPE in ['e5m2', 'e4m3', 'e2m1']
        assert NUM_WARPS == 4 or NUM_WARPS == 8
        assert P_K_WIDTH == 16 or (KV_TYPE != 'e2m1' and P_K_WIDTH == 8)
        KV_PACK_DIV: ttgl.constexpr = 2 if KV_TYPE == 'e2m1' else 1

        self.base = AttentionConfigBase(Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M,
                                        BLOCK_N, NUM_BUFFERS, NUM_WARPS)

        tiles_per_warp: ttgl.constexpr = [2, 2]
        num_warps: ttgl.constexpr = NUM_WARPS

        wmma_layout: ttgl.constexpr = ttgl.amd.AMDWMMALayout(  #
            version=3, transposed=True, warps_per_cta=[num_warps, 1], instr_shape=[16, 16, 128],
            tiles_per_warp=tiles_per_warp)
        wmma_layout_packed: ttgl.constexpr = ttgl.amd.AMDWMMALayout(  #
            version=3, transposed=True, warps_per_cta=[num_warps, 1], instr_shape=[16, 16, 64],
            tiles_per_warp=tiles_per_warp)

        self.q_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, k_width=16))
        if KV_TYPE == 'e2m1':
            self.k_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout_packed, k_width=16))
            self.p_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, k_width=16))
            self.v_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout_packed, k_width=16))
            self.CONVERT_LAYOUT_TRIVIAL = ttgl.constexpr(False)
        else:
            self.k_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, k_width=16))
            self.p_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, k_width=P_K_WIDTH))
            self.v_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, k_width=P_K_WIDTH))
            self.CONVERT_LAYOUT_TRIVIAL = ttgl.constexpr(True if P_K_WIDTH == 8 else False)

        self.q_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.q_layout, [BLOCK_M, HEAD_SZ // 32]))
        self.k_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.k_layout, [BLOCK_N, HEAD_SZ // 32]))
        self.p_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.p_layout, [BLOCK_M, BLOCK_N // 32]))
        self.v_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.v_layout, [HEAD_SZ, BLOCK_N // 32]))

        self.k_scale_load_layout = ttgl.constexpr(get_load_layout([HEAD_SZ // 32, BLOCK_N], num_warps))
        self.v_scale_load_layout = ttgl.constexpr(get_load_layout([BLOCK_N // 32, HEAD_SZ], num_warps))

        self.k_smem_layout = ttgl.constexpr(  #
            get_padded_shared_layout([BLOCK_N, HEAD_SZ // KV_PACK_DIV]))
        self.v_smem_layout = ttgl.constexpr(  #
            get_padded_shared_layout([BLOCK_N // KV_PACK_DIV, HEAD_SZ], transposed=True))
        if SUBTILE:
            self.k_smem_layout = ttgl.constexpr(  #
                get_padded_shared_layout([BLOCK_N // 2, HEAD_SZ // KV_PACK_DIV]))
            self.v_smem_layout = ttgl.constexpr(  #
                get_padded_shared_layout([BLOCK_N // KV_PACK_DIV, HEAD_SZ // 2], transposed=True))
        self.k_scale_smem_layout = ttgl.constexpr(ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        self.v_scale_smem_layout = ttgl.constexpr(ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0]))

        self.acc_layout = ttgl.constexpr(wmma_layout)

        self.P_SCALING = ttgl.constexpr(P_SCALING)
        self.SUBTILE = ttgl.constexpr(SUBTILE)


@aggregate
class BlockScaledAttentionProgram:
    cfg: BlockScaledAttentionConfig

    q: ttgl.tensor
    q_scale: ttgl.tensor
    k_mem: MemoryUnit
    k_scale_mem: MemoryUnit
    v_mem: MemoryUnit
    v_scale_mem: MemoryUnit
    o_blk: MemoryBlock
    # TODO: sm_scale should be a constexpr but the current llvm can not properly
    # fuse v_fma for literal operands, so we are using tensor here to ensure
    # it is in a register. Change it back to constexpr once the llvm is fixed.
    sm_scale: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg,  #
                 q, q_scale,  #
                 k_mem, k_scale_mem,  #
                 v_mem, v_scale_mem,  #
                 o_blk,  #
                 sm_scale):
        self.cfg = cfg
        self.q = q
        self.q_scale = q_scale
        self.k_mem = k_mem
        self.k_scale_mem = k_scale_mem
        self.v_mem = v_mem
        self.v_scale_mem = v_scale_mem
        self.o_blk = o_blk
        self.sm_scale = sm_scale

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
        KV_PACK_DIV: ttgl.constexpr = 2 if cfg.KV_TYPE == 'e2m1' else 1
        NUM_BUFFERS: ttgl.constexpr = cfg.NUM_BUFFERS
        SUBTILE: ttgl.constexpr = cfg.SUBTILE

        off_h = ttgl.program_id(0)  # NUM_Q_HEADS
        off_m = ttgl.program_id(1)  # NUM_BLOCKS
        off_z = ttgl.program_id(2)  # BATCH

        ttgl.static_assert(NUM_Q_HEADS % NUM_K_HEADS == 0)
        group_sz: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
        off_hk = off_h // group_sz

        q_off = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h) + \
                BLOCK_M * off_m * HEAD_SZ
        q_blk = MemoryBlock.initialize(  #
            base=q_ptr + q_off,  #
            shape=[SEQLEN_Q, HEAD_SZ],  #
            block_shape=[BLOCK_M, HEAD_SZ],  #
            layout=cfg.q_layout)

        q_scale_off = SEQLEN_Q * (HEAD_SZ // 32) * (NUM_Q_HEADS * off_z + off_h) + \
                      BLOCK_M * off_m * (HEAD_SZ // 32)
        q_scale_blk = MemoryBlock.initialize(  #
            base=q_scale_ptr + q_scale_off,  #
            shape=[SEQLEN_Q, HEAD_SZ // 32],  #
            block_shape=[BLOCK_M, HEAD_SZ // 32],  #
            layout=cfg.q_scale_layout)

        k_off = SEQLEN_K * (HEAD_SZ // KV_PACK_DIV) * (NUM_K_HEADS * off_z + off_hk)
        k_mem = MemoryUnit.initialize(  #
            base=k_ptr + k_off,  #
            shape=[SEQLEN_K, HEAD_SZ // KV_PACK_DIV],  #
            block_shape=[BLOCK_N, HEAD_SZ // KV_PACK_DIV],  #
            layout=cfg.k_layout,  #
            smem_layout=cfg.k_smem_layout,  #
            num_buffers=NUM_BUFFERS,  #
            sub_axis=0 if SUBTILE else None)

        K_SCALE_DIV: ttgl.constexpr = 128
        k_scale_off = (SEQLEN_K // K_SCALE_DIV) * (HEAD_SZ // 32 * K_SCALE_DIV) * (NUM_K_HEADS * off_z + off_hk)
        k_scale_mem = MemoryUnit.initialize(  #
            base=k_scale_ptr + k_scale_off,  #
            shape=[SEQLEN_K // K_SCALE_DIV, HEAD_SZ // 32 * K_SCALE_DIV],  #
            block_shape=[BLOCK_N // K_SCALE_DIV, HEAD_SZ // 32 * K_SCALE_DIV],  #
            layout=cfg.k_scale_layout,  #
            smem_layout=cfg.k_scale_smem_layout,  #
            num_buffers=NUM_BUFFERS)

        v_off = (SEQLEN_K // KV_PACK_DIV) * HEAD_SZ * (NUM_K_HEADS * off_z + off_hk)
        v_mem = MemoryUnit.initialize(  #
            base=v_ptr + v_off,  #
            shape=[SEQLEN_K // KV_PACK_DIV, HEAD_SZ],  #
            block_shape=[BLOCK_N // KV_PACK_DIV, HEAD_SZ],  #
            layout=cfg.v_layout,  #
            smem_layout=cfg.v_smem_layout,  #
            num_buffers=NUM_BUFFERS, sub_axis=1 if SUBTILE else None)

        V_SCALE_DIV: ttgl.constexpr = 128 if HEAD_SZ == 128 else 64
        v_scale_off = (SEQLEN_K // 32 * V_SCALE_DIV) * (HEAD_SZ // V_SCALE_DIV) * (NUM_K_HEADS * off_z + off_hk)
        v_scale_mem = MemoryUnit.initialize(  #
            base=v_scale_ptr + v_scale_off,  #
            shape=[HEAD_SZ // V_SCALE_DIV, SEQLEN_K // 32 * V_SCALE_DIV],  #
            block_shape=[HEAD_SZ // V_SCALE_DIV, BLOCK_N // 32 * V_SCALE_DIV],  #
            layout=cfg.v_scale_layout,  #
            smem_layout=cfg.v_scale_smem_layout,  #
            num_buffers=NUM_BUFFERS)

        o_blk = MemoryBlock.initialize(  #
            o_ptr + q_off,  #
            shape=[SEQLEN_Q, HEAD_SZ],  #
            block_shape=[BLOCK_M, HEAD_SZ],  #
            layout=cfg.acc_layout)

        q = buffer_load(q_blk.ptr, q_blk.offs, q_blk.mask, other=0.0)
        q_scale = buffer_load(q_scale_blk.ptr, q_scale_blk.offs, q_scale_blk.mask, other=0x7F)

        return BlockScaledAttentionProgram(  #
            cfg,  #
            q, q_scale,  #
            k_mem, k_scale_mem,  #
            v_mem, v_scale_mem,  #
            o_blk,  #
            sm_scale)

    @gluon.jit
    def issue_global_load_k(self, idx, sub_idx=0, buf=0, pred=True):
        self.k_mem.issue_tdm_load(idx, sub_idx, buf, pred)

    @gluon.jit
    def issue_global_load_v(self, idx, sub_idx=0, buf=0, pred=True):
        self.v_mem.issue_tdm_load(idx, sub_idx, buf, pred)

    @gluon.jit
    def issue_global_load_k_scale(self, idx, buf=0, pred=True):
        self.k_scale_mem.issue_tdm_load(idx, buf=buf, pred=pred)

    @gluon.jit
    def issue_global_load_v_scale(self, idx, buf=0, pred=True):
        self.v_scale_mem.issue_tdm_load(idx, buf=buf, pred=pred)

    @gluon.jit
    def shared_load_k(self, sub_idx=0, buf=0):
        cfg = self.cfg

        k_buffer = self.k_mem.smem.index(buf).permute((1, 0))
        if cfg.SUBTILE:
            k_buffer = self.k_mem.smem.index(buf * 2 + sub_idx).permute((1, 0))
        k = k_buffer.load(cfg.k_layout)
        return k

    @gluon.jit
    def shared_load_v(self, sub_idx=0, buf=0):
        cfg = self.cfg

        v_buffer = self.v_mem.smem.index(buf)
        if cfg.SUBTILE:
            v_buffer = self.v_mem.smem.index(buf * 2 + sub_idx)
        v = v_buffer.load(cfg.v_layout)
        return v

    @gluon.jit
    def shared_load_k_scale(self, buf=0):
        cfg = self.cfg

        K_SCALE_DIV: ttgl.constexpr = 128
        k_scale_buffer = self.k_scale_mem.smem.index(buf)
        k_scale_buffer = self.unshuffle_scale(k_scale_buffer, cfg.BLOCK_N, cfg.HEAD_SZ // 32, K_SCALE_DIV)
        k_scale = k_scale_buffer.load(cfg.k_scale_layout)
        return k_scale

    @gluon.jit
    def shared_load_v_scale(self, buf=0):
        cfg = self.cfg

        V_SCALE_DIV: ttgl.constexpr = 128 if cfg.HEAD_SZ == 128 else 64
        v_scale_buffer = self.v_scale_mem.smem.index(buf)
        v_scale_buffer = self.unshuffle_scale(v_scale_buffer, cfg.HEAD_SZ, cfg.BLOCK_N // 32, V_SCALE_DIV)
        v_scale = v_scale_buffer.load(cfg.v_scale_layout)
        return v_scale

    @gluon.jit
    def compute_qk(self, k, k_scale, acc):
        cfg = self.cfg

        qk = wmma_scaled(self.q, self.q_scale, cfg.Q_TYPE, k, k_scale, cfg.KV_TYPE, acc)
        return qk

    @gluon.jit
    def compute_pv(self, p, p_scale, v, v_scale, acc):
        cfg = self.cfg

        acc = wmma_scaled(p, p_scale, cfg.P_TYPE, v, v_scale, cfg.KV_TYPE, acc)
        return acc

    @gluon.jit
    def downcast_p(self, p):
        cfg = self.cfg

        if cfg.P_SCALING:
            p, p_scale = self.downcast_fp32_to_mxfp8(p, cfg.P_TYPE, [cfg.BLOCK_M, cfg.BLOCK_N])
            p_scale = ttgl.convert_layout(p_scale, cfg.p_scale_layout)
        else:
            p = self.downcast_fp32_to_fp8(p, cfg.P_TYPE)
            p_scale = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 32], 0x7F, ttgl.uint8, cfg.p_scale_layout)
        p = ttgl.convert_layout(p, cfg.p_layout, cfg.CONVERT_LAYOUT_TRIVIAL)

        return p, p_scale

    @gluon.jit
    def store_output(self, acc):
        o_blk = self.o_blk
        o = acc.to(o_blk.dtype)
        buffer_store(o, o_blk.ptr, o_blk.offs, o_blk.mask)

    @gluon.jit
    def async_wait(self, count):
        tdm.async_wait(count)

    @gluon.jit
    def downcast_fp32_to_mxfp8(self, x, x_format: ttgl.constexpr, shape: ttgl.constexpr):
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
    def downcast_fp32_to_fp8(self, x, x_format: ttgl.constexpr):
        if x_format == 'e4m3':
            return x.to(ttgl.float8e4nv)
        else:
            assert x_format == 'e5m2'
            return x.to(ttgl.float8e5)

    @gluon.jit
    def unshuffle_scale(self, buffer, non_k_dim, k_dim, non_k_div):
        block_non_k: ttgl.constexpr = non_k_dim // non_k_div
        kwidth: ttgl.constexpr = 4 if k_dim >= 4 else k_dim
        return (buffer  #
                .reshape((block_non_k, k_dim // kwidth, non_k_div // 4, 4, kwidth))  #
                .permute((0, 3, 2, 1, 4))  #
                .reshape((non_k_dim, k_dim)))

    @gluon.jit
    def concat_subtile(self, x, y):
        ttgl.static_assert(x.type.layout == y.type.layout)
        layout: ttgl.constexpr = x.type.layout
        shape: ttgl.constexpr = [x.shape[0], x.shape[1] + y.shape[1]]
        a = ttgl.join(x, y)
        a = a.permute(0, 2, 1).reshape(shape)
        a = ttgl.convert_layout(a, layout, assert_trivial=True)
        return a

    @gluon.jit
    def split_scale(self, x):
        layout: ttgl.constexpr = x.type.layout
        a0, a1 = x.reshape([2, x.shape[0] // 2, x.shape[1]]).permute(1, 2, 0).split()
        a0 = ttgl.convert_layout(a0, layout, assert_trivial=True)
        a1 = ttgl.convert_layout(a1, layout, assert_trivial=True)
        return a0, a1

    @gluon.jit
    def fwd_loop(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
        acc = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)
        sm_scale = self.sm_scale

        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end):
            self.issue_global_load_k(i)
            self.issue_global_load_k_scale(i)

            self.async_wait(0)
            k = self.shared_load_k()
            k_scale = self.shared_load_k_scale()

            qk = self.compute_qk(k, k_scale, zero)

            m = ttgl.max(qk, 1)
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
            p = ttgl.exp2(qk_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            m_i = m_ij
            alpha = ttgl.exp2(m_diff)
            l_ij = ttgl.sum(p, 1)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            p, p_scale = self.downcast_p(p)

            self.issue_global_load_v(i)
            self.issue_global_load_v_scale(i)

            self.async_wait(0)
            v = self.shared_load_v()
            v_scale = self.shared_load_v_scale()

            acc = self.compute_pv(p, p_scale, v, v_scale, acc)

        acc = acc / l_i[:, None]
        self.store_output(acc)

    @gluon.jit
    def fwd_loop_pipeline(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
        acc = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)
        sm_scale = self.sm_scale

        # pipeline prologue, iter -3
        self.issue_global_load_k(0, buf=0)  # ................................. iter 0
        self.issue_global_load_k_scale(0, buf=0)  # ........................... iter 0

        # pipeline prologue, iter -2
        self.issue_global_load_k(1, buf=1)  # ................................. iter 1
        self.issue_global_load_k_scale(1, buf=1)  # ........................... iter 1

        self.async_wait(1 * 2)  # ............................................. iter 0
        k = self.shared_load_k(buf=0)
        k_scale = self.shared_load_k_scale(buf=0)
        self.issue_global_load_v(0, buf=0)  # ................................. iter 0
        self.issue_global_load_v_scale(0, buf=0)  # ........................... iter 0

        # pipeline prologue, iter -1
        qk = self.compute_qk(k, k_scale, zero)  # ............................. iter 0

        self.issue_global_load_k(2, buf=0)  # ................................. iter 2
        self.issue_global_load_k_scale(2, buf=0)  # ........................... iter 2

        m = ttgl.max(qk, 1)  # ................................................ iter 0
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        self.async_wait(2 * 2)  # ............................................. iter 0
        k = self.shared_load_k(buf=1)
        k_scale = self.shared_load_k_scale(buf=1)
        self.issue_global_load_v(1, buf=1)  # ................................. iter 1
        self.issue_global_load_v_scale(1, buf=1)  # ........................... iter 1

        # main loop from 0 to end-3
        # TODO: Ideally we should unroll the loop by 2 to remove the buffer index
        # update, but our current codegen in llvm does not perform well. Re-enable
        # unroll when fixed.
        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 2
            b = 1 - a
            pred = (i != end - 3)

            qk = self.compute_qk(k, k_scale, zero)  # ......................... iter i+1
            l_ij = ttgl.sum(p, 1)  # .......................................... iter i
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            p, p_scale = self.downcast_p(p)

            self.async_wait(2 * 2)  # ......................................... iter i
            v = self.shared_load_v(buf=a)
            v_scale = self.shared_load_v_scale(buf=a)
            self.issue_global_load_k(i + 3, buf=b, pred=pred)  # .............. iter i+3
            self.issue_global_load_k_scale(i + 3, buf=b, pred=pred)  # ........ iter i+3

            acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ............. iter i
            m = ttgl.max(qk, 1)  # ............................................ iter i+1
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
            p = ttgl.exp2(qk_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            alpha = ttgl.exp2(m_diff)
            m_i = m_ij

            self.async_wait(2 * 2)  # ......................................... iter i+2
            k = self.shared_load_k(buf=a)
            k_scale = self.shared_load_k_scale(buf=a)
            self.issue_global_load_v(i + 2, buf=a)  # ......................... iter i+2
            self.issue_global_load_v_scale(i + 2, buf=a)  # ................... iter i+2

        # pipeline epilogue, iter end-2
        qk = self.compute_qk(k, k_scale, zero)  # ............................. iter end-1
        l_ij = ttgl.sum(p, 1)  # .............................................. iter end-2
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(2 * 2)  # ............................................. iter end-2
        v = self.shared_load_v(buf=0)
        v_scale = self.shared_load_v_scale(buf=0)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-2
        m = ttgl.max(qk, 1)  # ................................................ iter end-1
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - m_ij_scaled[:, None]
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        # pipeline epilogue, iter end-1
        l_ij = ttgl.sum(p, 1)  # .............................................. iter end-1
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(0)  # ................................................. iter end-1
        v = self.shared_load_v(buf=1)
        v_scale = self.shared_load_v_scale(buf=1)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-1

        # write output
        l_recip = 1 / l_i
        acc = acc * l_recip[:, None]
        self.store_output(acc)

    @gluon.jit
    def fwd_subtile(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc0 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc1 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        sm_scale = self.sm_scale

        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end):
            self.issue_global_load_k(i, sub_idx=0)
            self.issue_global_load_k(i, sub_idx=1)
            self.issue_global_load_k_scale(i)

            self.async_wait(0)
            k0 = self.shared_load_k(sub_idx=0)
            k1 = self.shared_load_k(sub_idx=1)
            k_scale = self.shared_load_k_scale()
            k0_scale, k1_scale = self.split_scale(k_scale)

            qk0 = self.compute_qk(k0, k0_scale, zero)
            qk1 = self.compute_qk(k1, k1_scale, zero)

            qk = self.concat_subtile(qk0, qk1)
            m = ttgl.max(qk, 1)
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]
            qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
            p0 = ttgl.exp2(qk0_shifted)
            p1 = ttgl.exp2(qk1_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            m_i = m_ij
            alpha = ttgl.exp2(m_diff)
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            p = self.concat_subtile(p0, p1)
            l_ij = ttgl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            p, p_scale = self.downcast_p(p)

            self.issue_global_load_v(i, sub_idx=0)
            self.issue_global_load_v(i, sub_idx=1)
            self.issue_global_load_v_scale(i)

            self.async_wait(0)
            v0 = self.shared_load_v(sub_idx=0)
            v1 = self.shared_load_v(sub_idx=1)
            v_scale = self.shared_load_v_scale()
            v0_scale, v1_scale = self.split_scale(v_scale)

            acc0 = self.compute_pv(p, p_scale, v0, v0_scale, acc0)
            acc1 = self.compute_pv(p, p_scale, v1, v1_scale, acc1)

        acc = self.concat_subtile(acc0, acc1)
        acc = acc / l_i[:, None]
        self.store_output(acc)

    @gluon.jit
    def fwd_subtile_pipeline(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc0 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc1 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        sm_scale = self.sm_scale

        # pipeline prologue, iter -3
        self.issue_global_load_k(0, sub_idx=0, buf=0)  # ...................... iter 0
        self.issue_global_load_k_scale(0, buf=0)  # ........................... iter 0

        self.issue_global_load_k(0, sub_idx=1, buf=0)  # ...................... iter 0

        # pipeline prologue, iter -2
        self.issue_global_load_k(1, sub_idx=0, buf=1)  # ...................... iter 1
        self.issue_global_load_k_scale(1, buf=1)  # ........................... iter 1

        self.async_wait(4)  # ................................................. iter 0
        k0 = self.shared_load_k(sub_idx=0, buf=0)
        self.async_wait(3)  # ................................................. iter 0
        k_scale = self.shared_load_k_scale(buf=0)
        k0_scale, k1_scale = self.split_scale(k_scale)
        self.issue_global_load_k(1, sub_idx=1, buf=1)  # ...................... iter 1

        # pipeline prologue, iter -1
        qk0 = self.compute_qk(k0, k0_scale, zero)  # .......................... iter 0
        self.async_wait(3)  # ................................................. iter 0
        k1 = self.shared_load_k(sub_idx=1, buf=0)
        self.issue_global_load_v(0, sub_idx=0, buf=0)  # ...................... iter 0
        self.issue_global_load_v_scale(0, buf=0)  # ........................... iter 0

        qk1 = self.compute_qk(k1, k1_scale, zero)  # .......................... iter 0
        self.issue_global_load_v(0, sub_idx=1, buf=0)  # ...................... iter 0

        qk = self.concat_subtile(qk0, qk1)  # ................................. iter 0
        m = ttgl.max(qk, 1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        self.issue_global_load_k(2, sub_idx=0, buf=0)  # ...................... iter 2
        self.issue_global_load_k_scale(2, buf=0)  # ........................... iter 2

        self.async_wait(6)  # ................................................. iter 1
        k0 = self.shared_load_k(sub_idx=0, buf=1)
        self.async_wait(5)  # ................................................. iter 1
        k_scale = self.shared_load_k_scale(buf=1)
        k0_scale, k1_scale = self.split_scale(k_scale)
        qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]  # ................ iter 0
        qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
        p0 = ttgl.exp2(qk0_shifted)
        self.issue_global_load_k(2, sub_idx=1, buf=0)  # ...................... iter 2

        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 2
            b = 1 - a
            pred = (i != end - 3)

            qk0 = self.compute_qk(k0, k0_scale, zero)  # ...................... iter i+1
            self.async_wait(5)  # ............................................. iter i+1
            k1 = self.shared_load_k(sub_idx=1, buf=b)
            p1 = ttgl.exp2(qk1_shifted)  # .................................... iter i
            m_diff = m_i * sm_scale - m_ij_scaled
            m_i = m_ij
            alpha = ttgl.exp2(m_diff)
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            self.issue_global_load_v(i + 1, sub_idx=0, buf=b)  # .............. iter i+1
            self.issue_global_load_v_scale(i + 1, buf=b)  # ................... iter i+1

            qk1 = self.compute_qk(k1, k1_scale, zero)  # ...................... iter i+1
            self.async_wait(6)  # ............................................. iter i
            v0 = self.shared_load_v(sub_idx=0, buf=a)
            self.async_wait(5)  # ............................................. iter i
            v_scale = self.shared_load_v_scale(buf=a)
            v0_scale, v1_scale = self.split_scale(v_scale)
            p = self.concat_subtile(p0, p1)  # ................................ iter i
            l_ij = ttgl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            p, p_scale = self.downcast_p(p)
            self.issue_global_load_v(i + 1, sub_idx=1, buf=b)  # .............. iter i+1

            acc0 = self.compute_pv(p, p_scale, v0, v0_scale, acc0)  # ......... iter i
            self.async_wait(5)  # ............................................. iter i
            v1 = self.shared_load_v(sub_idx=1, buf=a)
            qk = self.concat_subtile(qk0, qk1)  # ............................. iter i+1
            m = ttgl.max(qk, 1)
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            self.issue_global_load_k(i + 3, sub_idx=0, buf=b, pred=pred)  # ... iter i+3
            self.issue_global_load_k_scale(i + 3, buf=b, pred=pred)  # ........ iter i+3

            acc1 = self.compute_pv(p, p_scale, v1, v1_scale, acc1)  # ......... iter i
            self.async_wait(6)  # ............................................. iter i+2
            k0 = self.shared_load_k(sub_idx=0, buf=a)
            self.async_wait(5)  # ............................................. iter i+2
            k_scale = self.shared_load_k_scale(buf=a)
            k0_scale, k1_scale = self.split_scale(k_scale)
            qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]  # ............ iter i+1
            qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
            p0 = ttgl.exp2(qk0_shifted)
            self.issue_global_load_k(i + 3, sub_idx=1, buf=b, pred=pred)  # ... iter i+3

        # pipeline epilogue iter end-2
        self.issue_global_load_v(end - 1, sub_idx=0, buf=1)
        self.issue_global_load_v(end - 1, sub_idx=1, buf=1)
        self.issue_global_load_v_scale(end - 1, buf=1)

        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(3)
        v0 = self.shared_load_v(sub_idx=0, buf=0)
        v1 = self.shared_load_v(sub_idx=1, buf=0)
        v_scale = self.shared_load_v_scale(buf=0)
        v0_scale, v1_scale = self.split_scale(v_scale)

        acc0 = self.compute_pv(p, p_scale, v0, v0_scale, acc0)
        acc1 = self.compute_pv(p, p_scale, v1, v1_scale, acc1)

        # pipeline epilogue iter end-1
        k1 = self.shared_load_k(sub_idx=1, buf=1)
        qk0 = self.compute_qk(k0, k0_scale, zero)
        qk1 = self.compute_qk(k1, k1_scale, zero)

        qk = self.concat_subtile(qk0, qk1)
        m = ttgl.max(qk, 1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale

        qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]
        qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
        p0 = ttgl.exp2(qk0_shifted)

        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(0)
        v0 = self.shared_load_v(sub_idx=0, buf=1)
        v1 = self.shared_load_v(sub_idx=1, buf=1)
        v_scale = self.shared_load_v_scale(buf=1)
        v0_scale, v1_scale = self.split_scale(v_scale)

        acc0 = self.compute_pv(p, p_scale, v0, v0_scale, acc0)
        acc1 = self.compute_pv(p, p_scale, v1, v1_scale, acc1)

        # write output
        acc = self.concat_subtile(acc0, acc1)
        l_recip = 1 / l_i
        acc = acc * l_recip[:, None]
        self.store_output(acc)


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_kernel(  #
        q_ptr, k_ptr, v_ptr,  #
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
        SUBTILE: ttgl.constexpr,  #
        PIPELINED: ttgl.constexpr,  #
        P_SCALING: ttgl.constexpr,  #
        P_K_WIDTH: ttgl.constexpr):

    NUM_WARPS: ttgl.constexpr = ttgl.num_warps()
    NUM_BUFFERS: ttgl.constexpr = 2 if PIPELINED else 1
    if BLOCK_SCALING:
        cfg = BlockScaledAttentionConfig(  #
            Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, P_SCALING,
            P_K_WIDTH, SUBTILE, NUM_BUFFERS, NUM_WARPS)
        pgm = BlockScaledAttentionProgram.initialize(  #
            cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale)
    else:
        cfg = GlobalScaledAttentionConfig(  #
            Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N, P_K_WIDTH,
            SUBTILE, NUM_BUFFERS, NUM_WARPS)
        pgm = GlobalScaledAttentionProgram.initialize(  #
            cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale)

    if SUBTILE and PIPELINED:
        pgm.fwd_subtile_pipeline()
    if SUBTILE and not PIPELINED:
        pgm.fwd_subtile()
    if not SUBTILE and PIPELINED:
        pgm.fwd_loop_pipeline()
    if not SUBTILE and not PIPELINED:
        pgm.fwd_loop()


def attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
             q_scale: torch.Tensor | int, k_scale: torch.Tensor | int, v_scale: torch.Tensor | int,  #
             q_type: str, kv_type: str, block_m: int, block_n: int,  #
             block_scaling: bool, subtile: bool, pipelined: bool, p_scaling: bool, p_k_width: int, num_warps: int = 4):
    batch, seqlen_q, num_q_heads, head_sz = q.shape
    _, seqlen_k, num_k_heads, _ = k.shape
    sm_scale = head_sz**(-0.5) * 1.4426950408889634  # 1 / ln(2)
    assert head_sz in {64, 128}
    assert block_n == 128
    if pipelined:
        assert cdiv(seqlen_k, block_n) > 4
        assert cdiv(seqlen_k, block_n) % 2 == 0
    if subtile:
        assert head_sz == 128

    # q: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ]
    # k: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
    # v: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    if block_scaling:
        # q_scale: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ / 32]
        q_scale = q_scale.permute(0, 2, 1, 3).contiguous()

        # In scaled wmma instruction, scales takes following shapes in global memory:
        # - a_scale: [M, K // 32]
        # - b_scale: [N, K // 32]
        #
        # To have vectorized memory access, it's better to store scales in a packed block scale layout. In this
        # layout, scales are stored in the shape:
        # - a_scale: [M // 32 // 4, K // 32 // 4, 32, 4, 4]
        # - b_scale: [N // 32 // 4, K // 32 // 4, 32, 4, 4]
        #
        # In this way, we can load scales from global memory in a more vectorized way. Then inside the kernel, we
        # permute and reshape scales to canonical shapes required by scaled wmma.
        def _preshuffle_scale(x: torch.Tensor, preshuffle_factor: int):
            b, h, non_k, k = x.shape
            num_chunk_m = non_k // preshuffle_factor
            scale_kwidth = 4 if k >= 4 else k
            num_chunk_k = k // scale_kwidth

            x = x.view(b, h, num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, scale_kwidth)
            x = x.permute(0, 1, 2, 5, 4, 3, 6).contiguous()
            return x.view(b, h, non_k // preshuffle_factor, k * preshuffle_factor)

        # k_scale:              [BATCH, NUM_K_HEADS, SEQLEN_K / 128, HEAD_SZ * 4]
        # v_scale(head_sz=128): [BATCH, NUM_K_HEADS, HEAD_SZ / 128, SEQLEN_K * 4]
        # v_scale(head_sz=64):  [BATCH, NUM_K_HEADS, HEAD_SZ / 64, SEQLEN_K * 2]
        k_scale = _preshuffle_scale(k_scale.permute(0, 2, 1, 3), 128)
        v_scale = _preshuffle_scale(v_scale.permute(0, 2, 3, 1), 128 if head_sz == 128 else 64)
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
    args = [
        q, k, v, q_scale, k_scale, v_scale, o, sm_scale,  #
        q_type, kv_type, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, block_m, block_n,  #
        block_scaling, subtile, pipelined, p_scaling, p_k_width
    ]
    kwargs = {"num_warps": num_warps, "waves_per_eu": 1}
    kernel = attn_fwd_kernel[grid](*args, **kwargs)

    return o.cpu().permute(0, 2, 1, 3), kernel


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def attn_fwd_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
                 q_scale: torch.Tensor | float, k_scale: torch.Tensor | float,
                 v_scale: torch.Tensor | float) -> torch.Tensor:

    q = q * q_scale
    k = k * k_scale
    v = v * v_scale

    g = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    d = q.shape[-1]

    scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output


def create_operand(dtype: str, b: int, s: int, h: int, d: int, pack_dim: int = -1):
    size = (b, s, h, d)
    # Limit operand to an empirical range for accuracy
    if dtype == 'e4m3':
        low, high = 0x38 - 15, 0x38 + 5  # [0.2812, 1.6250]
        v = torch.randint(low, high + 1, size, dtype=torch.uint8)
        v = v.view(torch.float8_e4m3fn)
        v_ref = v.to(torch.float32)
    elif dtype == 'e5m2':
        low, high = 0x3C - 15, 0x3C + 5  # [0.0781, 2.500]
        v = torch.randint(low, high + 1, size, dtype=torch.uint8)
        v = v.view(torch.float8_e5m2)
        v_ref = v.to(torch.float32)
    else:
        assert dtype == 'e2m1'
        assert pack_dim >= 0
        low, high = 1 / 16, 16
        v_data = (low - high) * torch.rand(size) + low
        v_mxfp4 = MXFP4Tensor(v_data)
        v = v_mxfp4.to_packed_tensor(pack_dim)
        v_ref = v_mxfp4.to(torch.float32)
    return v, v_ref


def create_block_scale(dtype: str, b: int, s: int, h: int, d: int, scale_dim: int):
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


def create_global_scale(dtype: str):
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


def get_source_mapping(block_scaling, subtile, pipelined, amdgcn):
    """
    Create a mapping from amdgcn assembly to source code lines:

    mapping = { (line_no, code): [instr1, instr2, ...] }

    For call stack: fn1 -> fn2
    line_no = "line1 -> line2 -> ..."
    code    = "code1 -> code2 -> ..."

    Only collect instructions inside the main loop of the kernel.
    """
    mapping = {}

    mod = sys.modules.get(__name__)
    src_lines = inspect.getsource(mod).splitlines()

    pgm = BlockScaledAttentionProgram if block_scaling else GlobalScaledAttentionProgram
    func_map = {
        (True, True): pgm.fwd_subtile_pipeline,
        (True, False): pgm.fwd_subtile,
        (False, True): pgm.fwd_loop_pipeline,
        (False, False): pgm.fwd_loop,
    }
    func = func_map[(subtile, pipelined)]
    func_start, func_end = func.starting_line_number + 1, func.starting_line_number + len(func.raw_src) - 1

    def is_in_loop(line_no: int, base_indent: int) -> bool:
        if line_no < func_start or line_no > func_end:
            return False
        line = src_lines[line_no - 1]
        indent = len(line) - len(line.lstrip())
        return indent >= base_indent + 4

    lines = amdgcn.splitlines()
    start_idx = next((i for i, line in enumerate(lines) if re.match(r'^\s*\.cfi_startproc', line)), None)
    end_idx = next((i for i, line in enumerate(lines) if re.match(r'^\s*\.cfi_endproc', line)), None)
    if start_idx is None or end_idx is None:
        return mapping

    loc = None
    loc_in_loop = False
    for line in lines[start_idx + 1:end_idx]:
        # Look for .loc directive
        if re.match(r'^\s*\.loc\s+', line):
            loc_str = line.split(';')[-1].strip()
            # Find location strings like 'file:line:column'
            locs = re.findall(r'([^\s\[\]@]+:\d+:\d+)', loc_str)
            callstack = []
            for loc_item in locs:
                file, line_no, _ = loc_item.split(':')
                # Only map locations from current file
                if file == os.path.basename(__file__):
                    code_line = src_lines[int(line_no) - 1].strip()
                    callstack.append((int(line_no), code_line))
            if not callstack:
                loc = None
                continue

            # Decide whether the current loc is in loop
            loc_in_loop = any(is_in_loop(l[0], 8) for l in callstack)

            # Build call stack string (reverse for deepest call first)
            callstack.reverse()
            line_no_str = " -> ".join(str(l[0]) for l in callstack)
            code_str = " -> ".join(l[1] for l in callstack)
            loc = (line_no_str, code_str)
            mapping.setdefault(loc, [])
            continue

        if loc is None:
            continue

        # Clean up instruction line
        instr = line.strip()
        instr = re.sub(r'\s/\*.*?\*/', '', instr).strip()
        if not instr or instr.startswith('.') or instr.startswith(';'):
            continue

        # Append instruction to the corresponding source code location
        if loc_in_loop:
            mapping[loc].append(instr)

    # remove empty entries
    mapping = {loc: instrs for loc, instrs in mapping.items() if instrs}

    return mapping


@pytest.mark.parametrize(
    "q_type,kv_type,batch,seqlen_q,seqlen_k,num_q_heads,num_k_heads,head_sz,"
    "block_m,block_n,subtile,pipelined,p_k_width",
    [(*test, *config)  #
     for test in [[q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz]
                  for q_type, kv_type in [("e4m3", "e4m3"), ("e4m3", "e2m1")]
                  for batch in [1]
                  for seqlen_q in [1, 1024]  # Prefill, Decode
                  for seqlen_k in [1024]
                  for num_q_heads, num_k_heads in [(1, 1), (4, 1), (4, 2)]  # MHA, MQA, GQA
                  for head_sz in [64, 128]]
     for config in [[128, 128, False, False, 16],  # baseline
                    [128, 128, False, True, 16],  # pipeline
                    [128, 128, False, True, 8],  # pipeline + layout optimization
                    [256, 128, True, False, 8],  # subtile + layout optimization
                    [256, 128, True, True, 8]  # subtile + pipeline + layout optimization
                    ]
     # only run optimized config for decode mha with head_sz=128
     if not (config != [128, 128, False, False, False, 16] and test[3:] != [1024, 1024, 1, 1, 128])])
def test_block_scaled_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz,  #
                               block_m, block_n, subtile, pipelined, p_k_width):
    if kv_type == 'e2m1' and p_k_width == 8:
        pytest.skip("e2m1 can not use k_width=8 for p")

    torch.manual_seed(0)

    q, q_ref = create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
    k, k_ref = create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=3)
    v, v_ref = create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=1)
    q_scale, q_scale_ref = create_block_scale(q_type, batch, seqlen_q, num_q_heads, head_sz, scale_dim=3)
    k_scale, k_scale_ref = create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=3)
    v_scale, v_scale_ref = create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=1)

    o, kernel = attn_fwd(q, k, v,  #
                         q_scale, k_scale, v_scale,  #
                         q_type, kv_type, block_m, block_n,  #
                         True, subtile, pipelined, False, p_k_width)
    o = o.to(torch.float32)

    o_ref = attn_fwd_ref(q_ref, k_ref, v_ref, q_scale_ref, k_scale_ref, v_scale_ref)
    o_ref = o_ref.to(torch.float32)

    # check output correctness
    matches = torch.isclose(o, o_ref, atol=0.1, rtol=0.1)
    total = o.numel()
    mismatches = total - matches.sum().item()
    mismatch_ratio = mismatches / total
    assert mismatches < 10, f"Mismatched elements: {mismatches} / {total} ({mismatch_ratio:.6%})"

    # check code generation
    amdgcn = kernel.asm['amdgcn']
    mapping = get_source_mapping(True, subtile, pipelined, amdgcn)

    groups = {
        'qk': r'.*compute_qk.*',
        'pv': r'.*compute_pv.*',
        'ds_load_k': r'.*shared_load_k.* -> .*k_buffer.load',
        'ds_load_v': r'.*shared_load_v.* -> .*v_buffer.load',
        'convert_layout': r'.*ttgl.convert_layout.*',
    }
    for g in groups.keys():
        code = [loc[1] for loc in mapping.keys() if re.match(groups[g], loc[1])]
        # check when k_width=8, there is no convert layout
        if g == 'convert_layout' and p_k_width == 8:
            assert len(code) == 0
            continue
        # check all groups exist
        assert len(code) > 0
    for loc, instrs in mapping.items():
        _, code = loc
        # check use correct wmma instruction
        if re.match(groups['pv'], code) or re.match(groups['qk'], code):
            wmma_instrs = [instr for instr in instrs if re.match(r'v_wmma_*', instr)]
            assert len(wmma_instrs) > 0 and all(
                instr.startswith("v_wmma_scale_f32_16x16x128_f8f6f4") for instr in wmma_instrs)
        # check always use ds_load_b128 to load k
        if re.match(groups['ds_load_k'], code):
            ds_load_instrs = [instr for instr in instrs if re.match(r'ds_load_', instr)]
            assert len(ds_load_instrs) > 0 and all(instr.startswith("ds_load_b128") for instr in ds_load_instrs)
        # check always use ds_load_tr8_b64 to load v
        if re.match(groups['ds_load_v'], code):
            ds_load_instrs = [instr for instr in instrs if re.match(r'ds_load_', instr)]
            assert len(ds_load_instrs) > 0 and all(instr.startswith("ds_load_tr8_b64") for instr in ds_load_instrs)
        # check use v_permlane16_swap for convert layout
        if p_k_width == 16 and re.match(groups['convert_layout'], code):
            v_permlane_instrs = [instr for instr in instrs if re.match(r'v_permlane_*', instr)]
            assert len(v_permlane_instrs) > 0 and all(
                instr.startswith("v_permlane16_swap") for instr in v_permlane_instrs)


@pytest.mark.parametrize(
    "q_type,kv_type,batch,seqlen_q,seqlen_k,num_q_heads,num_k_heads,head_sz,"
    "block_m,block_n,subtile,pipelined,p_k_width",
    [(*test, *config)  #
     for test in [[q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz]
                  for q_type, kv_type in [("e4m3", "e4m3")]
                  for batch in [1]
                  for seqlen_q in [1, 1024]  # Prefill, Decode
                  for seqlen_k in [1024]
                  for num_q_heads, num_k_heads in [(1, 1), (4, 1), (4, 2)]  # MHA, MQA, GQA
                  for head_sz in [64, 128]]
     for config in [[128, 128, False, False, 16],  # baseline
                    [128, 128, False, True, 8],  # pipeline + layout optimization
                    [256, 128, True, False, 8],  # subtile + layout optimization
                    [256, 128, True, True, 8],  # subtile + pipeline + layout optimization
                    ]
     # only run optimized config for decode mha with head_sz=128
     if not (config != [128, 128, False, False, 16] and test[3:] != [1024, 1024, 1, 1, 128])])
def test_global_scaled_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz,  #
                                block_m, block_n, subtile, pipelined, p_k_width):
    torch.manual_seed(0)

    q, q_ref = create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
    k, k_ref = create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz)
    v, v_ref = create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz)
    q_scale, q_scale_ref = create_global_scale(q_type)
    k_scale, k_scale_ref = create_global_scale(kv_type)
    v_scale, v_scale_ref = create_global_scale(kv_type)

    o, kernel = attn_fwd(q, k, v,  #
                         q_scale, k_scale, v_scale,  #
                         q_type, kv_type, block_m, block_n,  #
                         False, subtile, pipelined, False, p_k_width)
    o = o.to(torch.float32)

    o_ref = attn_fwd_ref(q_ref, k_ref, v_ref, q_scale_ref, k_scale_ref, v_scale_ref)
    o_ref = o_ref.to(torch.float32)

    # check output correctness
    matches = torch.isclose(o, o_ref, atol=0.25, rtol=0.25)
    total = o.numel()
    mismatches = total - matches.sum().item()
    mismatch_ratio = mismatches / total
    assert mismatches < 10, f"Mismatched elements: {mismatches} / {total} ({mismatch_ratio:.6%})"

    # check code generation
    amdgcn = kernel.asm['amdgcn']
    mapping = get_source_mapping(False, subtile, pipelined, amdgcn)

    groups = {
        'qk': r'.*compute_qk.*',
        'pv': r'.*compute_pv.*',
        'ds_load_k': r'.*shared_load_k.* -> .*k_buffer.load',
        'ds_load_v': r'.*shared_load_v.* -> .*v_buffer.load',
        'convert_layout': r'.*ttgl.convert_layout.*',
    }
    for g in groups.keys():
        code = [loc[1] for loc in mapping.keys() if re.match(groups[g], loc[1])]
        # check when k_width=8, there is no convert layout
        if g == 'convert_layout' and p_k_width == 8:
            assert len(code) == 0
            continue
        # check all groups exist
        assert len(code) > 0
    for loc, instrs in mapping.items():
        _, code = loc
        # check use correct wmma instruction
        if re.match(groups['pv'], code) or re.match(groups['qk'], code):
            wmma_instrs = [instr for instr in instrs if re.match(r'v_wmma_*', instr)]
            assert len(wmma_instrs) > 0 and all(
                instr.startswith("v_wmma_scale_f32_16x16x128_f8f6f4") for instr in wmma_instrs)
        # check always use ds_load_b128 to load k
        if re.match(groups['ds_load_k'], code):
            ds_load_instrs = [instr for instr in instrs if re.match(r'ds_load_', instr)]
            assert len(ds_load_instrs) > 0 and all(instr.startswith("ds_load_b128") for instr in ds_load_instrs)
        # check always use ds_load_tr8_b64 to load v
        if re.match(groups['ds_load_v'], code):
            ds_load_instrs = [instr for instr in instrs if re.match(r'ds_load_', instr)]
            assert len(ds_load_instrs) > 0 and all(instr.startswith("ds_load_tr8_b64") for instr in ds_load_instrs)
        # check use v_permlane16_swap for convert layout
        if p_k_width == 16 and re.match(groups['convert_layout'], code):
            v_permlane_instrs = [instr for instr in instrs if re.match(r'v_permlane_*', instr)]
            assert len(v_permlane_instrs) > 0 and all(
                instr.startswith("v_permlane16_swap") for instr in v_permlane_instrs)


def run_attention(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, block_m, block_n,
                  scale_type, subtile, pipelined, disable_p_scaling, p_k_width):
    if kv_type == 'e2m1' and p_k_width == 8:
        raise RuntimeError("e2m1 can not use k_width=8 for p")

    q, _ = create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
    k, _ = create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=3)
    v, _ = create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=1)
    if scale_type == 'block':
        q_scale, _ = create_block_scale(q_type, batch, seqlen_q, num_q_heads, head_sz, scale_dim=3)
        k_scale, _ = create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=3)
        v_scale, _ = create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=1)
    else:
        assert scale_type == 'global'
        q_scale, _ = create_global_scale(q_type)
        k_scale, _ = create_global_scale(kv_type)
        v_scale, _ = create_global_scale(kv_type)

    _, kernel = attn_fwd(q, k, v,  #
                         q_scale, k_scale, v_scale,  #
                         q_type, kv_type, block_m, block_n,  #
                         scale_type == 'block', subtile, pipelined, not disable_p_scaling, p_k_width)
    return kernel


if __name__ == "__main__":

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
    parser.add_argument(
        "--scale_type", type=str, choices=['block', 'global'], required=True,
        help="`block` = use block scaling where 32 elements share a scale; "
        "`global` = use a single global scale for all elements")
    parser.add_argument("--subtile", action="store_true")
    parser.add_argument("--pipelined", action="store_true")
    parser.add_argument(
        "--disable_p_scaling", action="store_true", help="When set, we will use a fixed scale of 1.0 for all P blocks. "
        "Otherwise, we will compute and apply per-block scaling for the P matrix tensor. "
        "Only apply when block scaling is enabled. Ignored for global scaling.")
    parser.add_argument(
        "--p_k_width", type=int, choices=[8, 16], required=True,
        help="The K width (in elements) for p. When set to 8, we can remove the layout conversion for p")
    args = parser.parse_args()
    args = vars(args)

    kernel = run_attention(**args)
    static_profile(kernel)
