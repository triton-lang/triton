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

from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.experimental.gluon.language.amd.gfx1250 import wmma_scaled
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.experimental.gluon.language.amd.gfx1250 import buffer_load, buffer_store
from triton.experimental.gluon.language.amd.gfx1250 import async_copy as cp

# Handle imports for both pytest (module context) and direct execution
try:
    from .gfx1250_utils import static_profile, composition
except ImportError:
    from gfx1250_utils import static_profile, composition

# ===-----------------------------------------------------------------------===#
# Kernel Utilities
# ===-----------------------------------------------------------------------===#


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
def get_wmma_layout(shape, num_warps, packed=False, preshuffled=False, warp_axis=0):
    warps_per_cta = [num_warps, 1] if warp_axis == 0 else [1, num_warps]
    tiles_per_warp = [1, 1]

    if preshuffled:
        if shape[1] > 16 * warps_per_cta[1]:
            tiles_per_warp[1] = 2
        if shape[0] > 16 * warps_per_cta[0]:
            tiles_per_warp[0] = 2

    reg_bases = []
    if tiles_per_warp[1] > 1:
        reg_bases.append([0, 1])
    if tiles_per_warp[0] > 1:
        reg_bases.append([1, 0])

    warp_bases = []
    warps_n = 1
    tiles_n = tiles_per_warp[1]
    while warps_n < warps_per_cta[1]:
        warp_bases.append([0, tiles_n])
        warps_n <<= 1
        tiles_n <<= 1
    warps_m = 1
    tiles_m = tiles_per_warp[0]
    while warps_m < warps_per_cta[0]:
        warp_bases.append([tiles_m, 0])
        warps_m <<= 1
        tiles_m <<= 1

    instr_shape = [16, 16, 128] if not packed else [16, 16, 64]
    return ttgl.amd.AMDWMMALayout(3, True, warp_bases, reg_bases, instr_shape)


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
    will slide the block along the other dimension. For example, for a tensor with shape [1024, 256] and block size
    [256, 256], we will automatically determine `axis=0`, and `idx=0` means loading [0:256, :].

    MemoryUnit also supports split a block into 2 sub-tiles along the `sub_axis` axis. For example, for a tensor with
    shape [1024, 256] and block size [512, 256]:
    - when `sub_axis=0`, we will split the block into 2 sub-tiles with shape [256, 256]
    - when `sub_axis=1`, we will split the block into 2 sub-tiles with shape [512, 128]
    When `sub_axis` is set to None, no sub-tiling is performed.
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
    def issue_tdm_load(self, idx, sub_idx=0, buf=0, pred=1):
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
    def initialize(base, shape, block_shape, layout, padding=False, num_buffers=1, sub_axis=None):
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

        if padding:
            smem_layout: ttgl.constexpr = get_padded_shared_layout([sub_block_m, sub_block_n])
        else:
            smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])

        desc = tdm.make_tensor_descriptor(  #
            base=base,  #
            shape=shape,  #
            strides=[shape[1], 1],  #
            block_shape=[sub_block_m, sub_block_n],  #
            layout=smem_layout)
        block = MemoryBlock.initialize(  #
            base=base,  #
            shape=shape,  #
            block_shape=[sub_block_m, sub_block_n],  #
            layout=layout)
        smem = ttgl.allocate_shared_memory(  #
            dtype,  #
            [num_buffers * num_subtile] + [sub_block_m, sub_block_n],  #
            smem_layout)

        return MemoryUnit(smem, desc, block, [shape[1], 1], axis, sub_axis)


def preshuffle_scale(x: torch.Tensor, preshuffle_factor: int = 128):
    """ Preshuffle scales for scaled wmma instruction.
    In scaled wmma instruction, scales takes following shapes in global memory:
    - a_scale: [M, K // 32]
    - b_scale: [N, K // 32]

    To have vectorized memory access, it's better to store scales in a packed block scale layout. In this
    layout, scales are stored contiguously in the shape of:
    - a_scale: [M // 32 // 4, K // 32 // 4, 32, 4, 4]
    - b_scale: [N // 32 // 4, K // 32 // 4, 32, 4, 4]

    The output shape will be
    - a_scale: [M // preshuffle_factor, K * preshuffle_factor]
    - b_scale: [N // preshuffle_factor, K * preshuffle_factor]

    In this way, we can load scales from global memory in a more vectorized way. Then inside the kernel, we
    permute and reshape scales to canonical shapes required by scaled wmma.
    """
    *prefix, non_k, k = x.shape
    scale_kwidth = 4 if k >= 4 else k
    num_chunk_m = non_k // preshuffle_factor
    num_chunk_k = k // scale_kwidth

    batch = 1
    for d in prefix:
        batch *= d
    x = x.reshape(batch, non_k, k)

    x = x.view(batch, num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, scale_kwidth)
    x = x.permute(0, 1, 4, 3, 2, 5).contiguous()
    x = x.view(batch, num_chunk_m, k * preshuffle_factor)

    return x.view(*prefix, non_k // preshuffle_factor, k * preshuffle_factor)


@gluon.jit
def unshuffle_scale(buffer, non_k_dim, k_dim, preshuffle_factor=128):
    """ Unshuffle scales inside the kernel to restore the original shape. """
    block_non_k: ttgl.constexpr = non_k_dim // preshuffle_factor
    kwidth: ttgl.constexpr = 4 if k_dim >= 4 else k_dim
    return (buffer  #
            .reshape((block_non_k, k_dim // kwidth, preshuffle_factor // 4, 4, kwidth))  #
            .permute((0, 3, 2, 1, 4))  #
            .reshape((non_k_dim, k_dim)))


def preshuffle_operand(x: torch.Tensor, block_shape: list[int], sub_axis: int | None = None):
    """ Preshuffle operand for better TDM performance.

    To get better performance from TDM, we need to make sure the inner-most dim of the target block is 256B.
    For a given tensor `x` with shape [*, dim_outer, dim_inner], we will reshape it into
    [*, dim_outer * dim_inner // 256, 256] from the host side, then restore it inside the kernel (`unshuffle_operand`).

    When we do subtile for the operand (sux_axis is not None), depending on the sub_axis:
    - When `sub_axis==0`, we are subtiling the outer dim, this works the same as no subtile case.
    - When `sub_axis==1`, we are subtiling the inner dim, we need to first permute subtiles before reshaping.
    """
    block_dim_outer, block_dim_inner = block_shape

    elem_bits = x.element_size() * 8
    assert elem_bits == 8  # Only support 8-bit elements for now
    elems = 256
    *prefix, dim_outer, dim_inner = x.shape
    assert block_dim_inner == dim_inner

    if sub_axis == 0 or sub_axis is None:
        x = x.contiguous().reshape(*prefix, dim_outer * dim_inner // elems, elems)
        return x
    else:
        assert sub_axis == 1
        batch = 1
        for d in prefix:
            batch *= d
        x = x.reshape(batch, dim_outer, dim_inner)

        x = x.view(batch, dim_outer // block_dim_outer, block_dim_outer, 2, dim_inner // 2)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(*prefix, dim_outer * dim_inner // elems, elems)
        return x


@gluon.jit
def unshuffle_operand(buffer, block_shape, sub_axis=None):
    """
    Unshuffle the operand's shared memory to restore the original shape. Use in pair with `preshuffle_operand`. The
    `block_shape` and `sub_axis` should be the same as those used in `preshuffle_operand` to get the correct original
    shape.
    """
    if sub_axis is None:
        return buffer.reshape(block_shape)
    elif sub_axis == 0:
        return buffer.reshape([block_shape[0] // 2, block_shape[1]])
    else:
        return buffer.reshape([block_shape[0], block_shape[1] // 2])


@gluon.jit
def initialize_kv_mem(base, shape, block_shape, layout, num_buffers=1, subtile=False):
    """
    Initialize the MemoryUnit for K or V. This is a specialized version of MemoryUnit for K or V. It considers the
    preshuffle and subtile logic, and will deduce the correct block shape accordingly. After preshuffling, a block
    is always subtiled along the outer dim (sub_axis=0).
    """
    elem_bits: ttgl.constexpr = base.dtype.element_ty.primitive_bitwidth
    ttgl.static_assert(elem_bits == 8)  # Only support 8-bit elements for now
    elems: ttgl.constexpr = 256
    return MemoryUnit.initialize(  #
        base=base,  #
        shape=[shape[0] * shape[1] // elems, elems],  #
        block_shape=[block_shape[0] * block_shape[1] // elems, elems],  #
        layout=layout,  #
        padding=True,  #
        num_buffers=num_buffers,  #
        sub_axis=0 if subtile else None)


@gluon.jit
def get_kv_buffer(mem, sub_idx, buf, block_shape, sub_axis=None):
    """
    Get the shared memory buffer from K/V memory unit. This function should be used in pair with `initialize_kv_mem` to
    get the correct shared memory shape.
    """
    smem = mem.smem
    buffer = smem.index((buf * 2 + sub_idx) if sub_axis is not None else buf)
    buffer = unshuffle_operand(buffer, block_shape, sub_axis)
    return buffer


@gluon.jit
def initialize_kv_scale_mem(base, shape, block_shape, layout, num_buffers=1, preshuffle_factor=128):
    """
    Initialize the MemoryUnit for K or V scales. This is a specialized version of MemoryUnit for K or V scales. It
    considers the preshuffle for K, V scales and will deduce the correct block shape accordingly.
    """
    return MemoryUnit.initialize(  #
        base=base,  #
        shape=[shape[0] // preshuffle_factor, shape[1] * preshuffle_factor],  #
        block_shape=[block_shape[0] // preshuffle_factor, block_shape[1] * preshuffle_factor],  #
        layout=layout,  #
        num_buffers=num_buffers)


@gluon.jit
def get_kv_scale_buffer(mem, buf, block_shape, preshuffle_factor=128, slice=None):
    """
    Get the shared memory buffer for K or V scales. This function should be used in pair with `initialize_kv_scale_mem`
    to get the correct shared memory buffer by reshaping the preshuffled data.
    """
    smem = mem.smem
    buffer = smem.index(buf)
    buffer = unshuffle_scale(buffer, block_shape[0], block_shape[1], preshuffle_factor)
    if slice is not None:
        buffer = buffer.slice(slice * (block_shape[0] // 2), (block_shape[0] // 2))
    return buffer


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
    k_layout: ttgl.constexpr
    p_layout: ttgl.constexpr
    v_layout: ttgl.constexpr
    acc_layout: ttgl.constexpr

    # Whether the layout convert between QK and P is trivial - no data movement. This can happen when we use
    # k_width=8 for P and V, which effectively makes QK and P have the same layout.
    CONVERT_LAYOUT_TRIVIAL: ttgl.constexpr
    # Whether to subtile K and V.
    SUBTILE: ttgl.constexpr
    # Whether to use pingpong schedule
    PINGPONG: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ,  #
                 BLOCK_M, BLOCK_N, SUBTILE, PINGPONG, WARP_REDUCE, P_K_WIDTH, NUM_BUFFERS, NUM_WARPS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert KV_TYPE in ['e5m2', 'e4m3']
        assert P_K_WIDTH == 16 or P_K_WIDTH == 8

        self.base = AttentionConfigBase(Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M,
                                        BLOCK_N, NUM_BUFFERS, NUM_WARPS)

        warp_axis = 0 if not WARP_REDUCE else 1
        wmma_shape = [BLOCK_M, min(BLOCK_N, HEAD_SZ)]
        if SUBTILE:
            wmma_shape = [BLOCK_M, min(BLOCK_N // 2, HEAD_SZ // 2)]
        wmma_layout: ttgl.constexpr = get_wmma_layout(wmma_shape, NUM_WARPS, warp_axis=warp_axis)

        self.q_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, 16))
        self.k_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, 16))
        self.p_layout = ttgl.constexpr(ttgl.DotOperandLayout(0, wmma_layout, P_K_WIDTH))
        self.v_layout = ttgl.constexpr(ttgl.DotOperandLayout(1, wmma_layout, P_K_WIDTH))
        self.acc_layout = ttgl.constexpr(wmma_layout)

        self.CONVERT_LAYOUT_TRIVIAL = ttgl.constexpr(True if P_K_WIDTH == 8 and not WARP_REDUCE else False)
        self.SUBTILE = ttgl.constexpr(SUBTILE)
        self.PINGPONG = ttgl.constexpr(PINGPONG)


@aggregate
class GlobalScaledAttentionProgram:
    cfg: GlobalScaledAttentionConfig

    q_blk: MemoryBlock
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
                 q_blk, q_scale,  #
                 k_mem, k_scale,  #
                 v_mem, v_scale,  #
                 o_blk,  #
                 sm_scale):
        self.cfg = cfg
        self.q_blk = q_blk
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

        off_h = ttgl.program_id(0)
        off_m = ttgl.program_id(1)
        off_z = ttgl.program_id(2)

        if SEQLEN_Q == SEQLEN_K:
            GROUP_SZ: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
            off_hk = off_h // GROUP_SZ

            # q_off =
            #   off_z * stride_z (NUM_Q_HEADS * SEQLEN_Q * HEAD_SZ) +
            #   off_h * stride_h (SEQLEN_Q * HEAD_SZ) +
            #   off_m * stride_m (BLOCK_M * HEAD_SZ)
            q_off = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h) + \
                    BLOCK_M * HEAD_SZ * off_m
            q_blk = MemoryBlock.initialize(  #
                q_ptr + q_off,  #
                shape=[SEQLEN_Q, HEAD_SZ],  #
                block_shape=[BLOCK_M, HEAD_SZ],  #
                layout=cfg.q_layout)

            o_blk = MemoryBlock.initialize(  #
                o_ptr + q_off,  #
                shape=[SEQLEN_Q, HEAD_SZ],  #
                block_shape=[BLOCK_M, HEAD_SZ],  #
                layout=cfg.acc_layout)
        else:
            GROUP_SZ: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
            NUM_GROUPS: ttgl.constexpr = NUM_K_HEADS
            off_hk = off_h

            # q_off =
            #   off_z * stride_z (NUM_GROUPS * GROUP_SZ * HEAD_SZ) +
            #   off_h * stride_h (GROUP_SZ * HEAD_SZ) +
            #   off_m * stride_m (BLOCK_M * HEAD_SZ)
            q_off = GROUP_SZ * HEAD_SZ * (NUM_GROUPS * off_z + off_h) + \
                    BLOCK_M * HEAD_SZ * off_m
            q_blk = MemoryBlock.initialize(  #
                q_ptr + q_off,  #
                shape=[GROUP_SZ, HEAD_SZ],  #
                block_shape=[BLOCK_M, HEAD_SZ],  #
                layout=cfg.q_layout)

            o_off = q_off
            o_blk = MemoryBlock.initialize(  #
                o_ptr + o_off,  #
                shape=[GROUP_SZ, HEAD_SZ],  #
                block_shape=[BLOCK_M, HEAD_SZ],  #
                layout=cfg.acc_layout)

        k_off = SEQLEN_K * HEAD_SZ * (NUM_K_HEADS * off_z + off_hk)
        k_mem = initialize_kv_mem(  #
            base=k_ptr + k_off,  #
            shape=[SEQLEN_K, HEAD_SZ],  #
            block_shape=[BLOCK_N, HEAD_SZ],  #
            layout=cfg.k_layout,  #
            num_buffers=NUM_BUFFERS,  #
            subtile=SUBTILE)

        v_off = k_off
        v_mem = initialize_kv_mem(  #
            base=v_ptr + v_off,  #
            shape=[SEQLEN_K, HEAD_SZ],  #
            block_shape=[BLOCK_N, HEAD_SZ],  #
            layout=cfg.v_layout,  #
            num_buffers=NUM_BUFFERS,  #
            subtile=SUBTILE)

        return GlobalScaledAttentionProgram(  #
            cfg,  #
            q_blk, q_scale,  #
            k_mem, k_scale,  #
            v_mem, v_scale,  #
            o_blk,  #
            sm_scale)

    @gluon.jit
    def global_load_q(self):
        q_blk = self.q_blk
        q = buffer_load(q_blk.ptr, q_blk.offs, q_blk.mask, other=0.0)
        return q

    @gluon.jit
    def issue_global_load_k(self, idx, sub_idx=0, buf=0, pred=1):
        self.k_mem.issue_tdm_load(idx, sub_idx, buf, pred)

    @gluon.jit
    def issue_global_load_v(self, idx, sub_idx=0, buf=0, pred=1):
        self.v_mem.issue_tdm_load(idx, sub_idx, buf, pred)

    @gluon.jit
    def shared_load_k(self, sub_idx=0, buf=0):
        cfg = self.cfg

        k_buffer = get_kv_buffer(self.k_mem, sub_idx, buf,  #
                                 block_shape=[cfg.BLOCK_N, cfg.HEAD_SZ],  #
                                 sub_axis=0 if cfg.SUBTILE else None)
        k_buffer = k_buffer.permute((1, 0))
        k = k_buffer.load(cfg.k_layout)
        return k

    @gluon.jit
    def shared_load_v(self, sub_idx=0, buf=0):
        cfg = self.cfg

        v_buffer = get_kv_buffer(self.v_mem, sub_idx, buf,  #
                                 block_shape=[cfg.BLOCK_N, cfg.HEAD_SZ],  #
                                 sub_axis=1 if cfg.SUBTILE else None)
        v = v_buffer.load(cfg.v_layout)
        return v

    @gluon.jit
    def compute_qk(self, q, q_scale, k, k_scale, acc):
        cfg = self.cfg

        qk = wmma_scaled(q, q_scale, cfg.Q_TYPE, k, k_scale, cfg.KV_TYPE, acc)
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
    def split_subtile(self, x):
        layout: ttgl.constexpr = x.type.layout
        a0, a1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        a0 = ttgl.convert_layout(a0, layout, assert_trivial=True)
        a1 = ttgl.convert_layout(a1, layout, assert_trivial=True)
        return a0, a1

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
        q_scale = self.q_scale
        k_scale = self.k_scale
        p_scale = 0x7F
        v_scale = self.v_scale

        q = self.global_load_q()

        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end):
            self.issue_global_load_k(i)

            self.async_wait(0)
            k = self.shared_load_k()

            qk = self.compute_qk(q, q_scale, k, k_scale, zero)

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
    def fwd_pipeline(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
        acc = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

        sm_scale = self.sm_scale
        q_scale = self.q_scale
        k_scale = self.k_scale
        p_scale = 0x7F
        v_scale = self.v_scale

        q = self.global_load_q()

        # pipeline prologue, iter -3
        self.issue_global_load_k(0, buf=0)  # ................................. iter 0

        # pipeline prologue, iter -2
        self.issue_global_load_k(1, buf=1)  # ................................. iter 1

        self.async_wait(1)  # ................................................. iter 0
        k = self.shared_load_k(buf=0)
        self.issue_global_load_v(0, buf=0)  # ................................. iter 0

        # pipeline prologue, iter -1
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter 0

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
            pred = i - end + 3
            pred = (pred >> 31) & 1

            qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ............. iter i+1
            l_ij = ttgl.sum(p, 1)  # .......................................... iter i
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            p = self.downcast_p(p)

            self.async_wait(2)  # ............................................. iter i
            v = self.shared_load_v(buf=a)
            self.issue_global_load_k(i + 3, buf=b, pred=pred)  # .............. iter i+3

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
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter end-1
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
    def fwd_pipeline_subtile(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc0 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc1 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)

        sm_scale = self.sm_scale
        q_scale = self.q_scale
        k_scale = self.k_scale
        p_scale = 0x7F
        v_scale = self.v_scale

        q = self.global_load_q()

        # pipeline prologue, iter -3
        self.issue_global_load_k(0, sub_idx=0, buf=0)  # ...................... iter 0

        self.issue_global_load_k(0, sub_idx=1, buf=0)  # ...................... iter 0

        # pipeline prologue, iter -2
        self.issue_global_load_k(1, sub_idx=0, buf=1)  # ...................... iter 1

        self.async_wait(2)  # ................................................. iter 0
        k0 = self.shared_load_k(sub_idx=0, buf=0)
        self.issue_global_load_k(1, sub_idx=1, buf=1)  # ...................... iter 1

        # pipeline prologue, iter -1
        qk0 = self.compute_qk(q, q_scale, k0, k_scale, zero)  # ............... iter 0
        self.async_wait(2)  # ................................................. iter 0
        k1 = self.shared_load_k(sub_idx=1, buf=0)
        self.issue_global_load_v(0, sub_idx=0, buf=0)  # ...................... iter 0

        qk1 = self.compute_qk(q, q_scale, k1, k_scale, zero)  # ............... iter 0
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
            pred = i - end + 3
            pred = (pred >> 31) & 1

            qk0 = self.compute_qk(q, q_scale, k0, k_scale, zero)  # ........... iter i+1
            self.async_wait(4)  # ............................................. iter i+1
            k1 = self.shared_load_k(sub_idx=1, buf=b)
            p1 = ttgl.exp2(qk1_shifted)  # .................................... iter i
            m_diff = m_i * sm_scale - m_ij_scaled
            m_i = m_ij
            alpha = ttgl.exp2(m_diff)
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            self.issue_global_load_v(i + 1, sub_idx=0, buf=b)  # .............. iter i+1

            qk1 = self.compute_qk(q, q_scale, k1, k_scale, zero)  # ........... iter i+1
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
        qk0 = self.compute_qk(q, q_scale, k0, k_scale, zero)
        k1 = self.shared_load_k(sub_idx=1, buf=1)
        qk1 = self.compute_qk(q, q_scale, k1, k_scale, zero)

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

    @gluon.jit
    def fwd_pipeline_pingpong(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
        acc = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

        sm_scale = self.sm_scale
        q_scale = self.q_scale
        k_scale = self.k_scale
        p_scale = 0x7F
        v_scale = self.v_scale

        q = self.global_load_q()

        # pipeline prologue, iter -3
        self.issue_global_load_k(0, buf=0)  # ................................. iter 0

        # pipeline prologue, iter -2
        self.issue_global_load_k(1, buf=1)  # ................................. iter 1

        self.async_wait(1)  # ................................................. iter 0
        k = self.shared_load_k(buf=0)
        self.issue_global_load_v(0, buf=0)  # ................................. iter 0

        # pipeline prologue, iter -1
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter 0

        self.issue_global_load_k(2, buf=0)  # ................................. iter 2

        m = ttgl.max(qk, 1)  # ................................................ iter 0
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk0, qk1 = self.split_subtile(qk)
        qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]
        qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
        p0 = ttgl.exp2(qk0_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        self.async_wait(2)  # ................................................. iter 0
        k = self.shared_load_k(buf=1)
        self.issue_global_load_v(1, buf=1)  # ................................. iter 1

        # main loop from 0 to end-3
        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 2
            b = 1 - a
            pred = i - end + 3
            pred = (pred >> 31) & 1

            with warp_pipeline_stage("stage0"):
                qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ......... iter i+1
                p1 = ttgl.exp2(qk1_shifted)  # ................................ iter i
                p = self.concat_subtile(p0, p1)
                l_ij = ttgl.sum(p, 1)
                acc = acc * alpha[:, None]
                l_i = l_i * alpha + l_ij
                p = self.downcast_p(p)

            self.async_wait(2)
            with warp_pipeline_stage("stage1"):
                v = self.shared_load_v(buf=a)  # .............................. iter i
                self.issue_global_load_k(i + 3, buf=b, pred=pred)  # .......... iter i+3

            with warp_pipeline_stage("stage2"):
                acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ......... iter i
                m = ttgl.max(qk, 1)  # ........................................ iter i+1
                m_ij = ttgl.maximum(m_i, m)
                m_ij_scaled = m_ij * sm_scale
                qk0, qk1 = self.split_subtile(qk)
                qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]
                qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
                p0 = ttgl.exp2(qk0_shifted)
                m_diff = m_i * sm_scale - m_ij_scaled
                alpha = ttgl.exp2(m_diff)
                m_i = m_ij

            self.async_wait(2)
            with warp_pipeline_stage("stage3"):
                k = self.shared_load_k(buf=a)  # .............................. iter i+2
                self.issue_global_load_v(i + 2, buf=a)  # ..................... iter i+2

        # pipeline epilogue, iter end-2
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter end-1
        p1 = ttgl.exp2(qk1_shifted)  # ........................................ iter end-2
        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(2)  # ................................................. iter end-2
        v = self.shared_load_v(buf=0)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-2
        m = ttgl.max(qk, 1)  # ................................................ iter end-1
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk0, qk1 = self.split_subtile(qk)
        qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]
        qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
        p0 = ttgl.exp2(qk0_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        # pipeline epilogue, iter end-1
        p1 = ttgl.exp2(qk1_shifted)  # ........................................ iter end-1
        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, 1)
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


# ===-----------------------------------------------------------------------===#
# Block Scaled Attention Program
# ===-----------------------------------------------------------------------===#


@composition
@aggregate
class BlockScaledAttentionConfig:
    base: AttentionConfigBase

    q_layout: ttgl.constexpr
    q_scale_layout: ttgl.constexpr

    k_layout: ttgl.constexpr
    k_scale_layout: ttgl.constexpr

    p_layout: ttgl.constexpr
    p_scale_layout: ttgl.constexpr

    v_layout: ttgl.constexpr
    v_scale_layout: ttgl.constexpr

    acc_layout: ttgl.constexpr

    # Whether to use per-block scaling for P; if False, use an uniform scale of 1.0.
    P_SCALING: ttgl.constexpr
    # Whether the layout convert between QK and P is trivial - no data movement. This can happen when we use
    # k_width=8 for P and V, which effectively makes QK and P have the same layout. But note we can use k_width=8 for
    # V when it is a mxfp4, so this only applies when KV_TYPE is not 'e2m1'.
    CONVERT_LAYOUT_TRIVIAL: ttgl.constexpr
    # Whether to subtile K and V.
    SUBTILE: ttgl.constexpr
    # The divisor for packed K, V
    KV_PACK_DIV: ttgl.constexpr
    # Whether to use pingpong schedule
    PINGPONG: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, P_SCALING,  #
                 BLOCK_M, BLOCK_N, SUBTILE, PINGPONG, WARP_REDUCE, P_K_WIDTH, NUM_BUFFERS, NUM_WARPS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert KV_TYPE in ['e5m2', 'e4m3', 'e2m1']
        assert P_K_WIDTH == 16 or (KV_TYPE != 'e2m1' and P_K_WIDTH == 8)

        KV_PACK_DIV: ttgl.constexpr = 2 if KV_TYPE == 'e2m1' else 1
        self.KV_PACK_DIV = ttgl.constexpr(KV_PACK_DIV)
        self.base = AttentionConfigBase(Q_TYPE, KV_TYPE, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M,
                                        BLOCK_N, NUM_BUFFERS, NUM_WARPS)

        warp_axis = 0 if not WARP_REDUCE else 1
        wmma_shape = [BLOCK_M, min(BLOCK_N, HEAD_SZ)]
        if SUBTILE:
            wmma_shape = [BLOCK_M, min(BLOCK_N // 2, HEAD_SZ // 2)]
        wmma_layout = get_wmma_layout(wmma_shape, NUM_WARPS, preshuffled=True, warp_axis=warp_axis)
        wmma_layout_packed = get_wmma_layout(wmma_shape, NUM_WARPS, packed=True, preshuffled=True, warp_axis=warp_axis)

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
            self.CONVERT_LAYOUT_TRIVIAL = ttgl.constexpr(True if P_K_WIDTH == 8 and not WARP_REDUCE else False)

        self.q_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.q_layout, [BLOCK_M, HEAD_SZ // 32]))
        self.k_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.k_layout, [BLOCK_N, HEAD_SZ // 32]))
        self.p_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.p_layout, [BLOCK_M, BLOCK_N // 32]))
        self.v_scale_layout = ttgl.constexpr(
            ttgl.amd.gfx1250.get_wmma_scale_layout(self.v_layout, [HEAD_SZ, BLOCK_N // 32]))

        self.acc_layout = ttgl.constexpr(wmma_layout)

        self.P_SCALING = ttgl.constexpr(P_SCALING)
        self.SUBTILE = ttgl.constexpr(SUBTILE)
        self.PINGPONG = ttgl.constexpr(PINGPONG)


@aggregate
class BlockScaledAttentionProgram:
    cfg: BlockScaledAttentionConfig

    q_blk: MemoryBlock
    q_scale_blk: MemoryBlock
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
                 q_blk, q_scale_blk,  #
                 k_mem, k_scale_mem,  #
                 v_mem, v_scale_mem,  #
                 o_blk,  #
                 sm_scale):
        self.cfg = cfg
        self.q_blk = q_blk
        self.q_scale_blk = q_scale_blk
        self.k_mem = k_mem
        self.k_scale_mem = k_scale_mem
        self.v_mem = v_mem
        self.v_scale_mem = v_scale_mem
        self.o_blk = o_blk
        self.sm_scale = sm_scale

    @gluon.jit
    def initialize(cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale):
        ttgl.static_assert(isinstance(cfg, BlockScaledAttentionConfig))

        SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
        SEQLEN_Q: ttgl.constexpr = cfg.SEQLEN_Q
        HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
        NUM_Q_HEADS: ttgl.constexpr = cfg.NUM_Q_HEADS
        NUM_K_HEADS: ttgl.constexpr = cfg.NUM_K_HEADS
        BLOCK_M: ttgl.constexpr = cfg.BLOCK_M
        BLOCK_N: ttgl.constexpr = cfg.BLOCK_N
        NUM_BUFFERS: ttgl.constexpr = cfg.NUM_BUFFERS
        SUBTILE: ttgl.constexpr = cfg.SUBTILE
        KV_PACK_DIV: ttgl.constexpr = cfg.KV_PACK_DIV

        off_h = ttgl.program_id(0)
        off_m = ttgl.program_id(1)
        off_z = ttgl.program_id(2)

        if SEQLEN_Q == SEQLEN_K:
            GROUP_SZ: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
            off_hk = off_h // GROUP_SZ

            # q_off =
            #   off_z * stride_z (NUM_Q_HEADS * SEQLEN_Q * HEAD_SZ) +
            #   off_h * stride_h (SEQLEN_Q * HEAD_SZ) +
            #   off_m * stride_m (BLOCK_M * HEAD_SZ)
            q_off = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h) + \
                    BLOCK_M * HEAD_SZ * off_m
            q_blk = MemoryBlock.initialize(  #
                base=q_ptr + q_off,  #
                shape=[SEQLEN_Q, HEAD_SZ],  #
                block_shape=[BLOCK_M, HEAD_SZ],  #
                layout=cfg.q_layout)

            # q_scale_off =
            #   off_z * stride_z (NUM_Q_HEADS * SEQLEN_Q * HEAD_SZ // 32) +
            #   off_h * stride_h (SEQLEN_Q * HEAD_SZ // 32) +
            #   off_m * stride_m (BLOCK_M * HEAD_SZ // 32)
            q_scale_off = SEQLEN_Q * (HEAD_SZ // 32) * (NUM_Q_HEADS * off_z + off_h) + \
                          BLOCK_M * (HEAD_SZ // 32) * off_m
            q_scale_blk = MemoryBlock.initialize(  #
                base=q_scale_ptr + q_scale_off,  #
                shape=[SEQLEN_Q, HEAD_SZ // 32],  #
                block_shape=[BLOCK_M, HEAD_SZ // 32],  #
                layout=cfg.q_scale_layout)

            o_off = q_off
            o_blk = MemoryBlock.initialize(  #
                o_ptr + o_off,  #
                shape=[SEQLEN_Q, HEAD_SZ],  #
                block_shape=[BLOCK_M, HEAD_SZ],  #
                layout=cfg.acc_layout)
        else:
            GROUP_SZ: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
            NUM_GROUPS: ttgl.constexpr = NUM_K_HEADS
            off_hk = off_h

            # q_off =
            #   off_z * stride_z (NUM_GROUPS * GROUP_SZ * HEAD_SZ) +
            #   off_h * stride_h (GROUP_SZ * HEAD_SZ) +
            #   off_m * stride_m (BLOCK_M * HEAD_SZ)
            q_off = GROUP_SZ * HEAD_SZ * (NUM_GROUPS * off_z + off_h) + \
                    BLOCK_M * HEAD_SZ * off_m
            q_blk = MemoryBlock.initialize(  #
                q_ptr + q_off,  #
                shape=[GROUP_SZ, HEAD_SZ],  #
                block_shape=[BLOCK_M, HEAD_SZ],  #
                layout=cfg.q_layout)

            # q_scale_off =
            #   off_z * stride_z (NUM_GROUPS * GROUP_SZ * HEAD_SZ // 32) +
            #   off_h * stride_h (GROUP_SZ * HEAD_SZ // 32) +
            #   off_m * stride_m (BLOCK_M * HEAD_SZ // 32)
            q_scale_off = GROUP_SZ * (HEAD_SZ // 32) * (NUM_GROUPS * off_z + off_h) + \
                          BLOCK_M * (HEAD_SZ // 32) * off_m
            q_scale_blk = MemoryBlock.initialize(  #
                base=q_scale_ptr + q_scale_off,  #
                shape=[GROUP_SZ, HEAD_SZ // 32],  #
                block_shape=[BLOCK_M, HEAD_SZ // 32],  #
                layout=cfg.q_scale_layout)

            o_off = q_off
            o_blk = MemoryBlock.initialize(  #
                o_ptr + o_off,  #
                shape=[GROUP_SZ, HEAD_SZ],  #
                block_shape=[BLOCK_M, HEAD_SZ],  #
                layout=cfg.acc_layout)

        k_off = SEQLEN_K * (HEAD_SZ // KV_PACK_DIV) * (NUM_K_HEADS * off_z + off_hk)
        k_mem = initialize_kv_mem(  #
            base=k_ptr + k_off,  #
            shape=[SEQLEN_K, HEAD_SZ // KV_PACK_DIV],  #
            block_shape=[BLOCK_N, HEAD_SZ // KV_PACK_DIV],  #
            layout=cfg.k_layout,  #
            num_buffers=NUM_BUFFERS,  #
            subtile=SUBTILE)

        k_scale_off = (SEQLEN_K) * (HEAD_SZ // 32) * (NUM_K_HEADS * off_z + off_hk)
        k_scale_mem = initialize_kv_scale_mem(  #
            base=k_scale_ptr + k_scale_off,  #
            shape=[SEQLEN_K, HEAD_SZ // 32],  #
            block_shape=[BLOCK_N, HEAD_SZ // 32],  #
            layout=cfg.k_scale_layout,  #
            num_buffers=NUM_BUFFERS)

        v_off = (SEQLEN_K // KV_PACK_DIV) * HEAD_SZ * (NUM_K_HEADS * off_z + off_hk)
        v_mem = initialize_kv_mem(  #
            base=v_ptr + v_off,  #
            shape=[SEQLEN_K // KV_PACK_DIV, HEAD_SZ],  #
            block_shape=[BLOCK_N // KV_PACK_DIV, HEAD_SZ],  #
            layout=cfg.v_layout,  #
            num_buffers=NUM_BUFFERS,  #
            subtile=SUBTILE)

        v_scale_off = (SEQLEN_K // 32) * (HEAD_SZ) * (NUM_K_HEADS * off_z + off_hk)
        v_scale_mem = initialize_kv_scale_mem(  #
            base=v_scale_ptr + v_scale_off,  #
            shape=[HEAD_SZ, SEQLEN_K // 32],  #
            block_shape=[HEAD_SZ, BLOCK_N // 32],  #
            layout=cfg.v_scale_layout,  #
            num_buffers=NUM_BUFFERS,  #
            preshuffle_factor=128 if HEAD_SZ == 128 else 64)

        return BlockScaledAttentionProgram(  #
            cfg,  #
            q_blk, q_scale_blk,  #
            k_mem, k_scale_mem,  #
            v_mem, v_scale_mem,  #
            o_blk,  #
            sm_scale)

    @gluon.jit
    def global_load_q(self):
        q_blk = self.q_blk
        q = buffer_load(q_blk.ptr, q_blk.offs, q_blk.mask, other=0.0)
        return q

    @gluon.jit
    def global_load_q_scale(self):
        q_scale_blk = self.q_scale_blk
        q_scale = buffer_load(q_scale_blk.ptr, q_scale_blk.offs, q_scale_blk.mask, other=0x7F)
        return q_scale

    @gluon.jit
    def issue_global_load_k(self, idx, sub_idx=0, buf=0, pred=1):
        self.k_mem.issue_tdm_load(idx, sub_idx, buf, pred)

    @gluon.jit
    def issue_global_load_v(self, idx, sub_idx=0, buf=0, pred=1):
        self.v_mem.issue_tdm_load(idx, sub_idx, buf, pred)

    @gluon.jit
    def issue_global_load_k_scale(self, idx, buf=0, pred=1):
        self.k_scale_mem.issue_tdm_load(idx, buf=buf, pred=pred)

    @gluon.jit
    def issue_global_load_v_scale(self, idx, buf=0, pred=1):
        self.v_scale_mem.issue_tdm_load(idx, buf=buf, pred=pred)

    @gluon.jit
    def shared_load_k(self, sub_idx=0, buf=0):
        cfg = self.cfg

        k_buffer = get_kv_buffer(self.k_mem, sub_idx, buf,  #
                                 block_shape=[cfg.BLOCK_N, cfg.HEAD_SZ // cfg.KV_PACK_DIV],  #
                                 sub_axis=0 if cfg.SUBTILE else None)
        k_buffer = k_buffer.permute((1, 0))
        k = k_buffer.load(cfg.k_layout)
        return k

    @gluon.jit
    def shared_load_v(self, sub_idx=0, buf=0):
        cfg = self.cfg

        v_buffer = get_kv_buffer(self.v_mem, sub_idx, buf,  #
                                 block_shape=[cfg.BLOCK_N // cfg.KV_PACK_DIV, cfg.HEAD_SZ],  #
                                 sub_axis=1 if cfg.SUBTILE else None)
        v = v_buffer.load(cfg.v_layout)
        return v

    @gluon.jit
    def shared_load_k_scale(self, buf=0, slice=None):
        cfg = self.cfg

        k_scale_buffer = get_kv_scale_buffer(self.k_scale_mem, buf,  #
                                             [cfg.BLOCK_N, cfg.HEAD_SZ // 32],  #
                                             slice=slice)
        k_scale = k_scale_buffer.load(cfg.k_scale_layout)
        return k_scale

    @gluon.jit
    def shared_load_v_scale(self, buf=0, slice=None):
        cfg = self.cfg

        v_scale_buffer = get_kv_scale_buffer(self.v_scale_mem, buf,  #
                                             [cfg.HEAD_SZ, cfg.BLOCK_N // 32],
                                             preshuffle_factor=128 if cfg.HEAD_SZ == 128 else 64,  #
                                             slice=slice)
        v_scale = v_scale_buffer.load(cfg.v_scale_layout)
        return v_scale

    @gluon.jit
    def compute_qk(self, q, q_scale, k, k_scale, acc):
        cfg = self.cfg

        qk = wmma_scaled(q, q_scale, cfg.Q_TYPE, k, k_scale, cfg.KV_TYPE, acc)
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
    def concat_subtile(self, x, y):
        ttgl.static_assert(x.type.layout == y.type.layout)
        layout: ttgl.constexpr = x.type.layout
        shape: ttgl.constexpr = [x.shape[0], x.shape[1] + y.shape[1]]
        a = ttgl.join(x, y)
        a = a.permute(0, 2, 1).reshape(shape)
        a = ttgl.convert_layout(a, layout, assert_trivial=True)
        return a

    @gluon.jit
    def split_subtile(self, x):
        layout: ttgl.constexpr = x.type.layout
        a0, a1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
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

        q = self.global_load_q()
        q_scale = self.global_load_q_scale()

        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end):
            self.issue_global_load_k(i)
            self.issue_global_load_k_scale(i)

            self.async_wait(0)
            k = self.shared_load_k()
            k_scale = self.shared_load_k_scale()

            qk = self.compute_qk(q, q_scale, k, k_scale, zero)

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
    def fwd_pipeline(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
        acc = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)
        sm_scale = self.sm_scale

        q = self.global_load_q()
        q_scale = self.global_load_q_scale()

        # pipeline prologue, iter -3
        self.issue_global_load_k(0, buf=0)  # ................................. iter 0
        self.issue_global_load_k_scale(0, buf=0)  # ........................... iter 0

        # pipeline prologue, iter -2
        self.issue_global_load_k(1, buf=1)  # ................................. iter 1
        self.issue_global_load_k_scale(1, buf=1)  # ........................... iter 1

        self.async_wait(2)  # ................................................. iter 0
        k = self.shared_load_k(buf=0)
        k_scale = self.shared_load_k_scale(buf=0)
        self.issue_global_load_v(0, buf=0)  # ................................. iter 0
        self.issue_global_load_v_scale(0, buf=0)  # ........................... iter 0

        # pipeline prologue, iter -1
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter 0

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

        self.async_wait(4)  # ................................................. iter 0
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
            pred = i - end + 3
            pred = (pred >> 31) & 1

            qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ............. iter i+1
            l_ij = ttgl.sum(p, 1)  # .......................................... iter i
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            p, p_scale = self.downcast_p(p)

            self.async_wait(4)  # ............................................. iter i
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

            self.async_wait(4)  # ............................................. iter i+2
            k = self.shared_load_k(buf=a)
            k_scale = self.shared_load_k_scale(buf=a)
            self.issue_global_load_v(i + 2, buf=a)  # ......................... iter i+2
            self.issue_global_load_v_scale(i + 2, buf=a)  # ................... iter i+2

        # pipeline epilogue, iter end-2
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter end-1
        l_ij = ttgl.sum(p, 1)  # .............................................. iter end-2
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(4)  # ................................................. iter end-2
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
    def fwd_pipeline_subtile(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc0 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc1 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        sm_scale = self.sm_scale

        q = self.global_load_q()
        q_scale = self.global_load_q_scale()

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
        k0_scale = self.shared_load_k_scale(buf=0, slice=0)
        k1_scale = self.shared_load_k_scale(buf=0, slice=1)
        self.issue_global_load_k(1, sub_idx=1, buf=1)  # ...................... iter 1

        # pipeline prologue, iter -1
        qk0 = self.compute_qk(q, q_scale, k0, k0_scale, zero)  # .............. iter 0
        self.async_wait(3)  # ................................................. iter 0
        k1 = self.shared_load_k(sub_idx=1, buf=0)
        self.issue_global_load_v(0, sub_idx=0, buf=0)  # ...................... iter 0
        self.issue_global_load_v_scale(0, buf=0)  # ........................... iter 0

        qk1 = self.compute_qk(q, q_scale, k1, k1_scale, zero)  # .............. iter 0
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
        k0_scale = self.shared_load_k_scale(buf=1, slice=0)
        k1_scale = self.shared_load_k_scale(buf=1, slice=1)
        qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]  # ................ iter 0
        qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
        p0 = ttgl.exp2(qk0_shifted)
        self.issue_global_load_k(2, sub_idx=1, buf=0)  # ...................... iter 2

        end = ttgl.cdiv(cfg.SEQLEN_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 2
            b = 1 - a
            pred = i - end + 3
            pred = (pred >> 31) & 1

            qk0 = self.compute_qk(q, q_scale, k0, k0_scale, zero)  # .......... iter i+1
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

            qk1 = self.compute_qk(q, q_scale, k1, k1_scale, zero)  # .......... iter i+1
            self.async_wait(6)  # ............................................. iter i
            v0 = self.shared_load_v(sub_idx=0, buf=a)
            self.async_wait(5)  # ............................................. iter i
            v0_scale = self.shared_load_v_scale(buf=a, slice=0)
            v1_scale = self.shared_load_v_scale(buf=a, slice=1)
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
            k0_scale = self.shared_load_k_scale(buf=a, slice=0)
            k1_scale = self.shared_load_k_scale(buf=a, slice=1)
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
        v0_scale = self.shared_load_v_scale(buf=0, slice=0)
        v1_scale = self.shared_load_v_scale(buf=0, slice=1)

        acc0 = self.compute_pv(p, p_scale, v0, v0_scale, acc0)
        acc1 = self.compute_pv(p, p_scale, v1, v1_scale, acc1)

        # pipeline epilogue iter end-1
        k1 = self.shared_load_k(sub_idx=1, buf=1)
        qk0 = self.compute_qk(q, q_scale, k0, k0_scale, zero)
        qk1 = self.compute_qk(q, q_scale, k1, k1_scale, zero)

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
        v0_scale = self.shared_load_v_scale(buf=1, slice=0)
        v1_scale = self.shared_load_v_scale(buf=1, slice=1)

        acc0 = self.compute_pv(p, p_scale, v0, v0_scale, acc0)
        acc1 = self.compute_pv(p, p_scale, v1, v1_scale, acc1)

        # write output
        acc = self.concat_subtile(acc0, acc1)
        l_recip = 1 / l_i
        acc = acc * l_recip[:, None]
        self.store_output(acc)

    @gluon.jit
    def fwd_pipeline_pingpong(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
        acc = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)
        sm_scale = self.sm_scale

        q = self.global_load_q()
        q_scale = self.global_load_q_scale()

        # pipeline prologue, iter -3
        self.issue_global_load_k(0, buf=0)  # ................................. iter 0
        self.issue_global_load_k_scale(0, buf=0)  # ........................... iter 0

        # pipeline prologue, iter -2
        self.issue_global_load_k(1, buf=1)  # ................................. iter 1
        self.issue_global_load_k_scale(1, buf=1)  # ........................... iter 1

        self.async_wait(2)  # ................................................. iter 0
        k = self.shared_load_k(buf=0)
        k_scale = self.shared_load_k_scale(buf=0)
        self.issue_global_load_v(0, buf=0)  # ................................. iter 0
        self.issue_global_load_v_scale(0, buf=0)  # ........................... iter 0

        # pipeline prologue, iter -1
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter 0

        self.issue_global_load_k(2, buf=0)  # ................................. iter 2
        self.issue_global_load_k_scale(2, buf=0)  # ........................... iter 2

        m = ttgl.max(qk, 1)  # ................................................ iter 0
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk0, qk1 = self.split_subtile(qk)
        qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]
        qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
        p0 = ttgl.exp2(qk0_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        self.async_wait(4)  # ................................................. iter 0
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
            pred = i - end + 3
            pred = (pred >> 31) & 1

            with warp_pipeline_stage("stage0"):
                qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ......... iter i+1
                p1 = ttgl.exp2(qk1_shifted)  # ................................ iter i
                p = self.concat_subtile(p0, p1)
                l_ij = ttgl.sum(p, 1)
                acc = acc * alpha[:, None]
                l_i = l_i * alpha + l_ij
                p, p_scale = self.downcast_p(p)

            self.async_wait(4)
            with warp_pipeline_stage("stage1"):
                v = self.shared_load_v(buf=a)  # .............................. iter i
                v_scale = self.shared_load_v_scale(buf=a)
                self.issue_global_load_k(i + 3, buf=b, pred=pred)  # .......... iter i+3
                self.issue_global_load_k_scale(i + 3, buf=b, pred=pred)

            with warp_pipeline_stage("stage2"):
                acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ......... iter i
                m = ttgl.max(qk, 1)  # ........................................ iter i+1
                m_ij = ttgl.maximum(m_i, m)
                m_ij_scaled = m_ij * sm_scale
                qk0, qk1 = self.split_subtile(qk)
                qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]
                qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
                p0 = ttgl.exp2(qk0_shifted)
                m_diff = m_i * sm_scale - m_ij_scaled
                alpha = ttgl.exp2(m_diff)
                m_i = m_ij

            self.async_wait(4)
            with warp_pipeline_stage("stage3"):
                k = self.shared_load_k(buf=a)  # .............................. iter i+2
                k_scale = self.shared_load_k_scale(buf=a)
                self.issue_global_load_v(i + 2, buf=a)  # ..................... iter i+2
                self.issue_global_load_v_scale(i + 2, buf=a)

        # pipeline epilogue, iter end-2
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter end-1
        p1 = ttgl.exp2(qk1_shifted)  # ........................................ iter end-2
        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(4)  # ................................................. iter end-2
        v = self.shared_load_v(buf=0)
        v_scale = self.shared_load_v_scale(buf=0)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-2
        m = ttgl.max(qk, 1)  # ................................................ iter end-1
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk0, qk1 = self.split_subtile(qk)
        qk0_shifted = qk0 * sm_scale - m_ij_scaled[:, None]
        qk1_shifted = qk1 * sm_scale - m_ij_scaled[:, None]
        p0 = ttgl.exp2(qk0_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        # pipeline epilogue, iter end-1
        p1 = ttgl.exp2(qk1_shifted)  # ........................................ iter end-1
        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, 1)
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


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit
def mxfp_attn_fwd_kernel(  #
        q_ptr, k_ptr, v_ptr,  #
        q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
        o_ptr,  #
        sm_scale,  #
        cfg: ttgl.constexpr):

    # Select the target program
    BLOCK_SCALING: ttgl.constexpr = isinstance(cfg, BlockScaledAttentionConfig)
    if not BLOCK_SCALING:
        pgm = GlobalScaledAttentionProgram.initialize(  #
            cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale)
    else:
        pgm = BlockScaledAttentionProgram.initialize(  #
            cfg, q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, v_scale_ptr, o_ptr, sm_scale)

    # Select the target schedule
    if cfg.NUM_BUFFERS == 1:
        pgm.fwd_loop()
    elif cfg.NUM_BUFFERS == 2:
        if cfg.SUBTILE:
            pgm.fwd_pipeline_subtile()
        elif cfg.PINGPONG:
            pgm.fwd_pipeline_pingpong()
        else:
            pgm.fwd_pipeline()


def get_attn_schedule(cfg):
    if isinstance(cfg, BlockScaledAttentionConfig):
        pgm = BlockScaledAttentionProgram
    else:
        pgm = GlobalScaledAttentionProgram

    if cfg.NUM_BUFFERS == 1:
        return pgm.fwd_loop
    elif cfg.NUM_BUFFERS == 2:
        if cfg.SUBTILE:
            return pgm.fwd_pipeline_subtile
        elif cfg.PINGPONG:
            return pgm.fwd_pipeline_pingpong
        else:
            return pgm.fwd_pipeline


def get_attn_config(  #
        q_type, kv_type, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, block_scaling, p_scaling,  #
        block_m, block_n, pipelined, num_warps):

    # When we have a large block_m for pipeline, we will subtile K/V to
    # save registers
    subtile = pipelined and block_m >= 256
    # When pipelined, we need double buffer for K/V
    num_buffers = 1 if not pipelined else 2
    # When kv_type if mxfp8 (e4m3 or e5m2), we can use p_k_width of 8,
    # which makes QK and P share the same layout.
    p_k_width = 16 if kv_type == 'e2m1' else 8
    # We can use pingpong schedule where there are 8 or more warps
    pingpong = pipelined and num_warps >= 8
    # TODO: Currently pingpong schedule will have register spill for
    # block_m=256.
    if block_m >= 256:
        pingpong = False
    # Disable warp reduce as it does not show performance benefit.
    warp_reduce = False

    if block_scaling:
        cfg = BlockScaledAttentionConfig(  #
            q_type, kv_type, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, p_scaling,  #
            block_m, block_n, subtile, pingpong, warp_reduce, p_k_width, num_buffers, num_warps)
    else:
        cfg = GlobalScaledAttentionConfig(  #
            q_type, kv_type, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz,  #
            block_m, block_n, subtile, pingpong, warp_reduce, p_k_width, num_buffers, num_warps)

    return cfg


def attn_fwd(  #
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
        q_scale: torch.Tensor | int, k_scale: torch.Tensor | int, v_scale: torch.Tensor | int,  #
        q_type: str, kv_type: str, block_scaling: bool, p_scaling: bool,  #
        block_m: int, block_n: int, pipelined: bool, num_warps: int):

    batch, seqlen_q, num_q_heads, head_sz = q.shape
    _, seqlen_k, num_k_heads, _ = k.shape
    dtype = torch.float32
    assert seqlen_q == 1 or seqlen_q == seqlen_k
    assert num_q_heads >= num_k_heads and num_q_heads % num_k_heads == 0
    assert head_sz in {64, 128}
    if pipelined:
        assert cdiv(seqlen_k, block_n) > 4

    cfg = get_attn_config(  #
        q_type, kv_type, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, block_scaling, p_scaling,  #
        block_m, block_n, pipelined, num_warps)
    subtile = cfg.SUBTILE
    kv_pack_div = 2 if kv_type == 'e2m1' else 1

    if seqlen_q == seqlen_k:
        # q: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ]
        # k: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
        # v: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
        # o: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ]
        q = q.permute(0, 2, 1, 3).contiguous()
        k = preshuffle_operand(k.permute(0, 2, 1, 3),  #
                               block_shape=[block_n, head_sz // kv_pack_div],  #
                               sub_axis=0 if subtile else None)
        v = preshuffle_operand(v.permute(0, 2, 1, 3),  #
                               block_shape=[block_n // kv_pack_div, head_sz],  #
                               sub_axis=1 if subtile else None)
        o = torch.zeros_like(q, dtype=dtype)

        # q_scale: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ / 32]
        # k_scale: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ / 32]
        # v_scale: [BATCH, NUM_K_HEADS, HEAD_SZ / 32, SEQLEN_K]
        if block_scaling:
            q_scale = q_scale.permute(0, 2, 1, 3).contiguous()
            k_scale = preshuffle_scale(k_scale.permute(0, 2, 1, 3), preshuffle_factor=128)
            v_scale = preshuffle_scale(v_scale.permute(0, 2, 3, 1), preshuffle_factor=128 if head_sz == 128 else 64)

        grid = (num_q_heads, cdiv(seqlen_q, block_m), batch)
    else:
        group_sz = num_q_heads // num_k_heads
        num_groups = num_k_heads
        # q: [BATCH, NUM_GROUPS, GROUP_SZ, HEAD_SZ]
        # k: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
        # v: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
        # o: [BATCH, NUM_GROUPS, GROUP_SZ, HEAD_SZ]
        q = q.permute(0, 2, 1, 3).view(batch, num_groups, group_sz, head_sz).contiguous()
        k = preshuffle_operand(k.permute(0, 2, 1, 3),  #
                               block_shape=[block_n, head_sz // kv_pack_div],  #
                               sub_axis=0 if subtile else None)
        v = preshuffle_operand(v.permute(0, 2, 1, 3),  #
                               block_shape=[block_n // kv_pack_div, head_sz],  #
                               sub_axis=1 if subtile else None)
        o = torch.zeros_like(q, dtype=dtype)

        # q_scale: [BATCH, NUM_GROUPS, GROUP_SZ, HEAD_SZ / 32]
        # k_scale: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ / 32]
        # v_scale: [BATCH, NUM_K_HEADS, HEAD_SZ / 32, SEQLEN_K]
        if block_scaling:
            q_scale = q_scale.permute(0, 2, 1, 3).view(batch, num_groups, group_sz, head_sz // 32).contiguous()
            k_scale = preshuffle_scale(k_scale.permute(0, 2, 1, 3), preshuffle_factor=128)
            v_scale = preshuffle_scale(v_scale.permute(0, 2, 3, 1), preshuffle_factor=128 if head_sz == 128 else 64)

        grid = (num_groups, cdiv(group_sz, block_m), batch)

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    if block_scaling:
        q_scale = q_scale.cuda()
        k_scale = k_scale.cuda()
        v_scale = v_scale.cuda()
    o = o.cuda()

    sm_scale = head_sz**(-0.5) * 1.4426950408889634  # 1 / ln(2)
    args = [q, k, v, q_scale, k_scale, v_scale, o, sm_scale, cfg]
    kwargs = {"num_warps": num_warps, "waves_per_eu": 1}

    kernel = mxfp_attn_fwd_kernel[grid](*args, **kwargs)
    out = o.cpu()
    if seqlen_q == seqlen_k:
        out = out.permute(0, 2, 1, 3)
    else:
        out = out.view(batch, num_q_heads, seqlen_q, head_sz).permute(0, 2, 1, 3)

    return out, kernel, cfg


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


def get_source_mapping(amdgcn, cfg):
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

    func = get_attn_schedule(cfg)
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


def get_attn_fwd_configs():
    # block_m,block_n,pipelined,num_warps
    configs = {
        "4warp_128x128_loop": [128, 128, False, 4],
        "4warp_128x128_pipeline": [128, 128, True, 4],
        "4warp_256x128_pipeline": [256, 128, True, 4],
        "1warp_16x128_loop": [16, 128, False, 1],
        "1warp_16x128_pipeline": [16, 128, True, 1],
        "4warp_64x128_loop": [64, 128, False, 4],
        "4warp_64x128_pipeline": [64, 128, True, 4],
    }

    return configs


def get_fwd_test_cases(block_scaling: bool):
    dtypes = [("e4m3", "e4m3"), ("e4m3", "e2m1")] if block_scaling else [("e4m3", "e4m3")]
    tests = [[q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz]
             for q_type, kv_type in dtypes
             for batch in [1]
             for seqlen_q, seqlen_k, num_q_heads, num_k_heads in [
                 (1024, 1024, 1, 1),
                 (1, 1024, 1, 1),
                 (1, 1024, 64, 1),
             ]
             for head_sz in [64, 128]]
    configs = get_attn_fwd_configs()

    param = []
    for test in tests:
        seqlen_q, seqlen_k, num_q_heads, num_k_heads = test[3:7]
        if seqlen_q == seqlen_k:
            # MHA Prefill
            param.append((*test, *configs["4warp_128x128_loop"]))
            param.append((*test, *configs["4warp_128x128_pipeline"]))
            param.append((*test, *configs["4warp_256x128_pipeline"]))
        else:
            assert seqlen_q == 1
            if num_q_heads == num_k_heads:
                # MHA Decode
                param.append((*test, *configs["1warp_16x128_loop"]))
                param.append((*test, *configs["1warp_16x128_pipeline"]))
            else:
                assert num_q_heads // num_k_heads == 64
                # MQA Decode
                param.append((*test, *configs["4warp_64x128_loop"]))
                param.append((*test, *configs["4warp_64x128_pipeline"]))
    return param


@pytest.mark.parametrize(
    "q_type,kv_type,batch,seqlen_q,seqlen_k,num_q_heads,num_k_heads,head_sz,"
    "block_m,block_n,pipelined,num_warps",  #
    get_fwd_test_cases(True))
def test_block_scaled_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz,  #
                               block_m, block_n, pipelined, num_warps):
    torch.manual_seed(0)

    q, q_ref = create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
    k, k_ref = create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=3)
    v, v_ref = create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz, pack_dim=1)
    q_scale, q_scale_ref = create_block_scale(q_type, batch, seqlen_q, num_q_heads, head_sz, scale_dim=3)
    k_scale, k_scale_ref = create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=3)
    v_scale, v_scale_ref = create_block_scale(kv_type, batch, seqlen_k, num_k_heads, head_sz, scale_dim=1)

    o, kernel, cfg = attn_fwd(  #
        q, k, v,  #
        q_scale, k_scale, v_scale,  #
        q_type, kv_type, True, False,  #
        block_m, block_n, pipelined, num_warps)
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
    mapping = get_source_mapping(amdgcn, cfg)

    groups = {
        'qk': r'.*compute_qk.*',
        'pv': r'.*compute_pv.*',
        'ds_load_k': r'.*shared_load_k.* -> .*k_buffer.load',
        'ds_load_v': r'.*shared_load_v.* -> .*v_buffer.load',
        'convert_layout': r'.*ttgl.convert_layout.*',
    }
    for g in groups.keys():
        code = [loc[1] for loc in mapping.keys() if re.match(groups[g], loc[1])]
        # check convert layout
        if g == 'convert_layout' and cfg.CONVERT_LAYOUT_TRIVIAL:
            assert len(code) == 0
            continue
        # check all other groups exist
        assert len(code) > 0

    for loc, instrs in mapping.items():
        _, code = loc
        # check use correct wmma instruction
        if re.match(groups['pv'], code) or re.match(groups['qk'], code):
            wmma_instrs = [instr for instr in instrs if re.match(r'v_wmma_*', instr)]
            assert len(wmma_instrs) > 0 and \
                all(instr.startswith("v_wmma_scale_f32_16x16x128_f8f6f4") for instr in wmma_instrs)
        # check always use ds_load_b128 to load k and all instructions are using the same vgpr for address
        if re.match(groups['ds_load_k'], code):
            ds_load_instrs = [instr for instr in instrs if re.match(r'ds_load_', instr)]
            assert len(ds_load_instrs) > 0 and all(instr.startswith("ds_load_b128") for instr in ds_load_instrs)
            sources = [instr.split()[2] for instr in ds_load_instrs]
            assert all(source == sources[0] for source in sources)
        # check always use ds_load_tr8_b64 to load v and all instructions are using the same vgpr for address
        if re.match(groups['ds_load_v'], code):
            ds_load_instrs = [instr for instr in instrs if re.match(r'ds_load_', instr)]
            assert len(ds_load_instrs) > 0 and all(instr.startswith("ds_load_tr8_b64") for instr in ds_load_instrs)
            sources = [instr.split()[2] for instr in ds_load_instrs]
            assert all(source == sources[0] for source in sources)
        # check use v_permlane16_swap for convert layout
        if re.match(groups['convert_layout'], code):
            v_permlane_instrs = [instr for instr in instrs if re.match(r'v_permlane_*', instr)]
            assert len(v_permlane_instrs) > 0 and all(
                instr.startswith("v_permlane16_swap") for instr in v_permlane_instrs)
        # check there is no v_readfirstlane
        assert all(not re.match(r'v_readfirstlane', instr) for instr in instrs)


@pytest.mark.parametrize(
    "q_type,kv_type,batch,seqlen_q,seqlen_k,num_q_heads,num_k_heads,head_sz,"
    "block_m,block_n,pipelined,num_warps",  #
    get_fwd_test_cases(False))
def test_global_scaled_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz,  #
                                block_m, block_n, pipelined, num_warps):
    torch.manual_seed(0)

    q, q_ref = create_operand(q_type, batch, seqlen_q, num_q_heads, head_sz)
    k, k_ref = create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz)
    v, v_ref = create_operand(kv_type, batch, seqlen_k, num_k_heads, head_sz)
    q_scale, q_scale_ref = create_global_scale(q_type)
    k_scale, k_scale_ref = create_global_scale(kv_type)
    v_scale, v_scale_ref = create_global_scale(kv_type)

    o, kernel, cfg = attn_fwd(  #
        q, k, v,  #
        q_scale, k_scale, v_scale,  #
        q_type, kv_type, False, False,  #
        block_m, block_n, pipelined, num_warps)
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
    mapping = get_source_mapping(amdgcn, cfg)

    groups = {
        'qk': r'.*compute_qk.*',
        'pv': r'.*compute_pv.*',
        'ds_load_k': r'.*shared_load_k.* -> .*k_buffer.load',
        'ds_load_v': r'.*shared_load_v.* -> .*v_buffer.load',
        'convert_layout': r'.*ttgl.convert_layout.*',
    }
    for g in groups.keys():
        code = [loc[1] for loc in mapping.keys() if re.match(groups[g], loc[1])]
        # check convert layout
        if g == 'convert_layout' and cfg.CONVERT_LAYOUT_TRIVIAL:
            assert len(code) == 0
            continue
        # check all other groups exist
        assert len(code) > 0

    for loc, instrs in mapping.items():
        _, code = loc
        # check use correct wmma instruction
        if re.match(groups['pv'], code) or re.match(groups['qk'], code):
            wmma_instrs = [instr for instr in instrs if re.match(r'v_wmma_*', instr)]
            assert len(wmma_instrs) > 0 and \
                all(instr.startswith("v_wmma_scale_f32_16x16x128_f8f6f4") for instr in wmma_instrs)
        # check always use ds_load_b128 to load k and all instructions are using the same vgpr for address
        if re.match(groups['ds_load_k'], code):
            ds_load_instrs = [instr for instr in instrs if re.match(r'ds_load_', instr)]
            assert len(ds_load_instrs) > 0 and all(instr.startswith("ds_load_b128") for instr in ds_load_instrs)
            sources = [instr.split()[2] for instr in ds_load_instrs]
            assert all(source == sources[0] for source in sources)
        # check always use ds_load_tr8_b64 to load v and all instructions are using the same vgpr for address
        if re.match(groups['ds_load_v'], code):
            ds_load_instrs = [instr for instr in instrs if re.match(r'ds_load_', instr)]
            assert len(ds_load_instrs) > 0 and all(instr.startswith("ds_load_tr8_b64") for instr in ds_load_instrs)
            sources = [instr.split()[2] for instr in ds_load_instrs]
            assert all(source == sources[0] for source in sources)
        # check use v_permlane16_swap for convert layout
        if re.match(groups['convert_layout'], code):
            v_permlane_instrs = [instr for instr in instrs if re.match(r'v_permlane_*', instr)]
            assert len(v_permlane_instrs) > 0 and all(
                instr.startswith("v_permlane16_swap") for instr in v_permlane_instrs)
        # check there is no v_readfirstlane
        assert all(not re.match(r'v_readfirstlane', instr) for instr in instrs)


def run_attention(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, scale_type,
                  disable_p_scaling, block_m, block_n, pipelined, num_warps):
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

    _, kernel, _ = attn_fwd(  #
        q, k, v,  #
        q_scale, k_scale, v_scale,  #
        q_type, kv_type, scale_type == 'block', not disable_p_scaling,  #
        block_m, block_n, pipelined, num_warps)
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
    parser.add_argument(
        "--disable_p_scaling", action="store_true", help="When set, we will use a fixed scale of 1.0 for all P blocks. "
        "Otherwise, we will compute and apply per-block scaling for the P matrix tensor. "
        "Only apply when block scaling is enabled. Ignored for global scaling.")
    parser.add_argument("--pipelined", action="store_true")
    parser.add_argument("--num_warps", type=int, required=True)
    args = parser.parse_args()
    args = vars(args)

    kernel = run_attention(**args)
    static_profile(kernel)
