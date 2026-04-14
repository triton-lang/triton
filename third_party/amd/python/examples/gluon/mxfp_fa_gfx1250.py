"""
Multi-head attention kernel in Gluon
"""
import os
import sys
import inspect
import argparse
import re
from functools import partial
import pytest
import torch
import math

from triton import cdiv
from triton.language.core import _aggregate as aggregate
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
from triton.experimental.gluon.language import expand_dims

from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.experimental.gluon.language.amd.gfx1250 import wmma_scaled
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.experimental.gluon.language.amd.gfx1250 import buffer_load
from triton.experimental.gluon.language.amd.gfx1250 import get_wmma_scale_layout

# Handle imports for both pytest (module context) and direct execution
try:
    from .gfx1250_utils import static_profile, composition
except ImportError:
    from gfx1250_utils import static_profile, composition

# ===-----------------------------------------------------------------------===#
# Kernel Utilities
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def get_shared_layout(shape, padding=False, transposed=False, clamp=False):
    """Default shared memory layout for TDM.

    When `padding=True`, use a padded shared memory layout to reduce LDS bank
    conflicts. When `clamp=True`, we will clamp the padding_interval to be no
    more than the inner dimension of the block.
    """
    if not padding:
        return ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])

    _, inner_dim = shape
    ## Here we assume the elements in LDS is 8-bit (for mxfp4, 2 mxfp4
    ## are packed in 1 8-bit elements). Then 256 elements can occupy
    ## 64 banks. Therefore, we want the padding_interval to be at
    ## least 256 elements.
    ## On the other hand, we only need to add padding after a row of
    ## elements. So we also want the padding_interval to be at least inner_dim.
    padding_interval = inner_dim if clamp else max(inner_dim, 256)
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
    rank = len(shape)
    assert rank == 2 or rank == 3
    warps_per_cta = [1] * rank
    warps_per_cta[warp_axis] = num_warps
    tiles_per_warp = [1] * rank

    # When use preshuffle, we should try to increase tiles_per_warp to 2 for each dimension if possible.
    if preshuffled:
        if 16 * warps_per_cta[-1] < shape[-1]:
            tiles_per_warp[-1] = 2
        if 16 * warps_per_cta[-2] < shape[-2]:
            tiles_per_warp[-2] = 2

    # Translate tiles_per_warp to reg_bases for linear layout
    reg_bases = []
    if tiles_per_warp[-1] > 1:
        base = [0] * rank
        base[-1] = 1
        reg_bases.append(base)
    if tiles_per_warp[-2] > 1:
        base = [0] * rank
        base[-2] = 1
        reg_bases.append(base)

    # Translate warps_per_cta to warp_bases for linear layout
    warp_bases = []
    warps = 1
    tiles = tiles_per_warp[warp_axis]
    tile_size = 16 if warp_axis >= rank - 2 else 1
    while warps < warps_per_cta[warp_axis]:
        base = [0] * rank
        if tiles * tile_size < shape[warp_axis]:
            base[warp_axis] = tiles
            tiles <<= 1
        warp_bases.append(base)
        warps <<= 1

    instr_shape = [16, 16, 128] if not packed else [16, 16, 64]
    return ttgl.amd.AMDWMMALayout(3, True, warp_bases, reg_bases, instr_shape, rank=rank)


@aggregate
class MemoryBlock:
    """
    MemoryBlock groups variables to describe a block of 2D/3D tensor in global memory.
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
        rank: ttgl.constexpr = len(block_shape)
        ttgl.static_assert(rank == len(shape))
        ttgl.static_assert(rank == 2 or rank == 3)

        if rank == 2:
            offs_m = ttgl.arange(0, block_shape[0], ttgl.SliceLayout(1, layout))
            offs_n = ttgl.arange(0, block_shape[1], ttgl.SliceLayout(0, layout))
            offs = expand_dims(offs_m, -1) * shape[1] + expand_dims(offs_n, -2)
            mask = expand_dims(offs_m < shape[0], -1) & expand_dims(offs_n < shape[1], -2)
        else:
            offs_b = ttgl.arange(0, block_shape[0], ttgl.SliceLayout(1, ttgl.SliceLayout(2, layout)))
            offs_m = ttgl.arange(0, block_shape[1], ttgl.SliceLayout(0, ttgl.SliceLayout(2, layout)))
            offs_n = ttgl.arange(0, block_shape[2], ttgl.SliceLayout(0, ttgl.SliceLayout(1, layout)))

            offs = offs_b[:, None, None] * (shape[1] * shape[2]) + \
                   offs_m[None, :, None] * shape[2] + \
                   offs_n[None, None, :]
            mask = (offs_b < shape[0])[:, None, None] & \
                   (offs_m < shape[1])[None, :, None] & \
                   (offs_n < shape[2])[None, None, :]

        return MemoryBlock(base, offs, mask, block_shape)


@aggregate
class MemoryUnit:
    """
    MemoryUnit wraps a global-memory tensor descriptor and its corresponding shared-memory slots.
    """
    smem: ttgl.shared_memory_descriptor
    desc: tdm.tensor_descriptor

    @gluon.constexpr_function
    def __init__(self, smem, desc):
        self.smem = smem
        self.desc = desc

    @gluon.jit
    def initialize(base, shape, block_shape, padding=False, num_slots=1):
        ttgl.static_assert(len(block_shape) == 2 and len(shape) == 2)

        dtype: ttgl.constexpr = base.dtype.element_ty

        smem_layout: ttgl.constexpr = get_shared_layout(block_shape, padding=padding)

        shape0 = shape[0]
        shape1 = shape[1]
        desc = tdm.make_tensor_descriptor(  #
            base=base,  #
            shape=[shape0, shape1],  #
            strides=[shape1, 1],  #
            block_shape=[block_shape[0], block_shape[1]],  #
            layout=smem_layout)
        smem = ttgl.allocate_shared_memory(  #
            dtype,  #
            [num_slots] + block_shape,  #
            smem_layout)

        return MemoryUnit(smem, desc)


@aggregate
class KVMemory:
    k_mem: MemoryUnit
    v_mem: MemoryUnit
    k_shape: ttgl.constexpr
    v_shape: ttgl.constexpr
    cfg: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, k_mem, v_mem, k_shape, v_shape, cfg):
        self.k_mem = k_mem
        self.v_mem = v_mem
        self.k_shape = ttgl.constexpr(k_shape)
        self.v_shape = ttgl.constexpr(v_shape)
        self.cfg = ttgl.constexpr(cfg)

    @gluon.constexpr_function
    def preshuffle(x: torch.Tensor, block_shape: list[int], sub_axis: int | None = None):
        """ Preshuffle operand for better TDM performance.

        To get better performance from TDM, we need to make sure the inner-most dim of the target block is 256B.
        For a given tensor `x` with shape [*, dim_outer, dim_inner], we will reshape it into
        [*, dim_outer * dim_inner // 256, 256] from the host side, then restore it inside the kernel (`unshuffle`).

        When we do subtile for the operand (sub_axis is not None), depending on the sub_axis:
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
            batch = math.prod(prefix)
            x = x.reshape(batch, dim_outer, dim_inner)

            x = x.view(batch, dim_outer // block_dim_outer, block_dim_outer, 2, dim_inner // 2)
            x = x.permute(0, 1, 3, 2, 4).contiguous()
            x = x.reshape(*prefix, dim_outer * dim_inner // elems, elems)
            return x

    @gluon.jit
    def unshuffle(buffer, block_shape, sub_axis=None):
        """
        Unshuffle the operand's shared memory to restore the original shape by reshaping.
        """
        if sub_axis is None:
            return buffer.reshape(block_shape)
        elif sub_axis == 0:
            return buffer.reshape([block_shape[0] // 2, block_shape[1]])
        else:
            return buffer.reshape([block_shape[0], block_shape[1] // 2])

    @gluon.constexpr_function
    def get_shuffle_shape(shape):
        elems = 256
        *prefix, dim_inner = shape
        return [*prefix[:-1], prefix[-1] * dim_inner // elems, elems]

    @gluon.constexpr_function
    def get_flat_shape(shape):
        *prefix, dim_inner = shape
        return [math.prod(prefix), dim_inner]

    @gluon.jit
    def initialize(k_base, v_base, cfg):
        BATCH: ttgl.constexpr = cfg.BATCH
        SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
        HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
        NUM_K_HEADS: ttgl.constexpr = cfg.NUM_K_HEADS
        BLOCK_N: ttgl.constexpr = cfg.BLOCK_N
        SPLIT_K: ttgl.constexpr = cfg.SPLIT_K
        NUM_BUFFERS: ttgl.constexpr = cfg.NUM_BUFFERS
        SUBTILE: ttgl.constexpr = cfg.SUBTILE
        KV_PACK_DIV: ttgl.constexpr = cfg.KV_PACK_DIV
        NUM_SUBTILES: ttgl.constexpr = 2 if SUBTILE else 1

        k_shape: ttgl.constexpr = KVMemory.get_shuffle_shape([BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ // KV_PACK_DIV])
        k_shape_flat: ttgl.constexpr = KVMemory.get_flat_shape(k_shape)
        k_block_shape: ttgl.constexpr = KVMemory.get_shuffle_shape(
            [BLOCK_N * SPLIT_K, HEAD_SZ // KV_PACK_DIV] if not SUBTILE else \
            [BLOCK_N * SPLIT_K // 2, HEAD_SZ // KV_PACK_DIV])
        k_mem = MemoryUnit.initialize(  #
            base=k_base,  #
            shape=k_shape_flat,  #
            block_shape=k_block_shape,  #
            padding=True,  #
            num_slots=NUM_BUFFERS * NUM_SUBTILES)

        v_shape: ttgl.constexpr = KVMemory.get_shuffle_shape([BATCH, NUM_K_HEADS, SEQLEN_K // KV_PACK_DIV, HEAD_SZ])
        v_shape_flat: ttgl.constexpr = KVMemory.get_flat_shape(v_shape)
        v_block_shape: ttgl.constexpr = KVMemory.get_shuffle_shape(
            [BLOCK_N * SPLIT_K // KV_PACK_DIV, HEAD_SZ] if not SUBTILE else \
            [BLOCK_N * SPLIT_K // KV_PACK_DIV, HEAD_SZ // 2])
        v_mem = MemoryUnit.initialize(  #
            base=v_base,  #
            shape=v_shape_flat,  #
            block_shape=v_block_shape,  #
            padding=True,  #
            num_slots=NUM_BUFFERS * NUM_SUBTILES)

        return KVMemory(k_mem, v_mem, k_shape, v_shape, cfg)

    @gluon.jit
    def issue_load_k(self, off, idx, sub_idx=0, buf=0, pred=1):
        SUBTILE: ttgl.constexpr = self.cfg.SUBTILE

        block_shape: ttgl.constexpr = self.k_mem.desc.block_shape
        num_subtiles: ttgl.constexpr = 2 if SUBTILE else 1
        smem_idx = buf * num_subtiles + sub_idx

        smem = self.k_mem.smem.index(smem_idx)
        off_m = off[0] + idx * num_subtiles * block_shape[0] + sub_idx * block_shape[0]
        off_n = off[1]
        tdm.async_load(self.k_mem.desc, [off_m, off_n], smem, pred)

    @gluon.jit
    def issue_load_v(self, off, idx, sub_idx=0, buf=0, pred=1):
        SUBTILE: ttgl.constexpr = self.cfg.SUBTILE

        block_shape: ttgl.constexpr = self.v_mem.desc.block_shape
        num_subtiles: ttgl.constexpr = 2 if SUBTILE else 1
        smem_idx = buf * num_subtiles + sub_idx

        smem = self.v_mem.smem.index(smem_idx)
        off_m = off[0] + idx * num_subtiles * block_shape[0] + sub_idx * block_shape[0]
        off_n = off[1]
        tdm.async_load(self.v_mem.desc, [off_m, off_n], smem, pred)

    @gluon.jit
    def get_k_buffer(self, sub_idx, buf):
        cfg = self.cfg
        sub_axis: ttgl.constexpr = 0 if cfg.SUBTILE else None
        block_shape: ttgl.constexpr = [cfg.BLOCK_N * cfg.SPLIT_K, cfg.HEAD_SZ // cfg.KV_PACK_DIV]
        buffer = self.k_mem.smem.index((buf * 2 + sub_idx) if sub_axis is not None else buf)
        buffer = KVMemory.unshuffle(buffer, block_shape, sub_axis)
        if cfg.SPLIT_K == 1:
            buffer = buffer.permute([1, 0])
        else:
            buffer = buffer.reshape([cfg.SPLIT_K, block_shape[0] // cfg.SPLIT_K, block_shape[1]])
            buffer = buffer.permute([0, 2, 1])
        return buffer

    @gluon.jit
    def get_v_buffer(self, sub_idx, buf):
        cfg = self.cfg
        sub_axis: ttgl.constexpr = 1 if cfg.SUBTILE else None
        block_shape: ttgl.constexpr = [cfg.SPLIT_K * cfg.BLOCK_N // cfg.KV_PACK_DIV, cfg.HEAD_SZ]
        buffer = self.v_mem.smem.index((buf * 2 + sub_idx) if sub_axis is not None else buf)
        buffer = KVMemory.unshuffle(buffer, block_shape, sub_axis)
        if cfg.SPLIT_K > 1:
            buffer = buffer.reshape([cfg.SPLIT_K, block_shape[0] // cfg.SPLIT_K, block_shape[1]])
        return buffer


@aggregate
class KVScaleMemory:
    k_mem: MemoryUnit
    v_mem: MemoryUnit
    k_shape: ttgl.constexpr
    v_shape: ttgl.constexpr
    cfg: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, k_mem, v_mem, k_shape, v_shape, cfg):
        self.k_mem = k_mem
        self.v_mem = v_mem
        self.k_shape = ttgl.constexpr(k_shape)
        self.v_shape = ttgl.constexpr(v_shape)
        self.cfg = ttgl.constexpr(cfg)

    @gluon.constexpr_function
    def preshuffle(x: torch.Tensor):
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
        preshuffle_factor = min(128, non_k)
        scale_kwidth = 4 if k >= 4 else k
        num_chunk_m = non_k // preshuffle_factor
        num_chunk_k = k // scale_kwidth

        batch = math.prod(prefix)
        x = x.reshape(batch, non_k, k)

        x = x.view(batch, num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, scale_kwidth)
        x = x.permute(0, 1, 4, 3, 2, 5).contiguous()
        x = x.view(batch, num_chunk_m, k * preshuffle_factor)

        return x.view(*prefix, non_k // preshuffle_factor, k * preshuffle_factor)

    @gluon.jit
    def unshuffle(buffer, block_shape):
        """
        Unshuffle scales inside the kernel to restore the original shape.
        """
        non_k_dim: ttgl.constexpr = block_shape[0]
        k_dim: ttgl.constexpr = block_shape[1]
        preshuffle_factor: ttgl.constexpr = 128 if non_k_dim >= 128 else non_k_dim
        block_non_k: ttgl.constexpr = non_k_dim // preshuffle_factor
        kwidth: ttgl.constexpr = 4 if k_dim >= 4 else k_dim
        return (buffer  #
                .reshape((block_non_k, k_dim // kwidth, preshuffle_factor // 4, 4, kwidth))  #
                .permute((0, 3, 2, 1, 4))  #
                .reshape((non_k_dim, k_dim)))

    @gluon.constexpr_function
    def get_shuffle_shape(shape, factor):
        *prefix, dim_inner = shape
        return [*prefix[:-1], prefix[-1] // factor, dim_inner * factor]

    @gluon.constexpr_function
    def get_flat_shape(shape):
        *prefix, dim_inner = shape
        return [math.prod(prefix), dim_inner]

    @gluon.jit
    def initialize(k_base, v_base, cfg):
        BATCH: ttgl.constexpr = cfg.BATCH
        SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
        HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
        NUM_K_HEADS: ttgl.constexpr = cfg.NUM_K_HEADS
        BLOCK_N: ttgl.constexpr = cfg.BLOCK_N
        SPLIT_K: ttgl.constexpr = cfg.SPLIT_K
        NUM_BUFFERS: ttgl.constexpr = cfg.NUM_BUFFERS

        k_preshuffle_factor: ttgl.constexpr = 128 if BLOCK_N * SPLIT_K >= 128 else BLOCK_N * SPLIT_K
        k_shape: ttgl.constexpr = KVScaleMemory.get_shuffle_shape([BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ // 32],
                                                                  k_preshuffle_factor)
        k_shape_flat: ttgl.constexpr = KVScaleMemory.get_flat_shape(k_shape)
        k_block_shape: ttgl.constexpr = KVScaleMemory.get_shuffle_shape([BLOCK_N * SPLIT_K, HEAD_SZ // 32],
                                                                        k_preshuffle_factor)
        k_mem = MemoryUnit.initialize(  #
            base=k_base,  #
            shape=k_shape_flat,  #
            block_shape=k_block_shape,  #
            num_slots=NUM_BUFFERS)

        v_preshuffle_factor: ttgl.constexpr = 128 if HEAD_SZ >= 128 else HEAD_SZ
        v_shape: ttgl.constexpr = KVScaleMemory.get_shuffle_shape([BATCH, NUM_K_HEADS, HEAD_SZ, SEQLEN_K // 32],
                                                                  v_preshuffle_factor)
        v_shape_flat: ttgl.constexpr = KVScaleMemory.get_flat_shape(v_shape)
        v_block_shape: ttgl.constexpr = KVScaleMemory.get_shuffle_shape([HEAD_SZ, BLOCK_N * SPLIT_K // 32],
                                                                        v_preshuffle_factor)
        v_mem = MemoryUnit.initialize(  #
            base=v_base,  #
            shape=v_shape_flat,  #
            block_shape=v_block_shape,  #
            num_slots=NUM_BUFFERS)

        return KVScaleMemory(k_mem, v_mem, k_shape, v_shape, cfg)

    @gluon.jit
    def issue_load_k(self, off, idx, buf=0, pred=1):
        block_shape: ttgl.constexpr = self.k_mem.desc.block_shape
        off_m = off[0] + idx * block_shape[0]
        off_n = off[1]
        smem = self.k_mem.smem.index(buf)
        tdm.async_load(self.k_mem.desc, [off_m, off_n], smem, pred)

    @gluon.jit
    def issue_load_v(self, off, idx, buf=0, pred=1):
        block_shape: ttgl.constexpr = self.v_mem.desc.block_shape
        off_m = off[0]
        off_n = off[1] + idx * block_shape[1]
        smem = self.v_mem.smem.index(buf)
        tdm.async_load(self.v_mem.desc, [off_m, off_n], smem, pred)

    @gluon.jit
    def get_k_buffer(self, buf, slice=None):
        cfg = self.cfg
        block_shape: ttgl.constexpr = [cfg.BLOCK_N * cfg.SPLIT_K, cfg.HEAD_SZ // 32]
        buffer = self.k_mem.smem.index(buf)
        buffer = KVScaleMemory.unshuffle(buffer, block_shape)
        if cfg.SPLIT_K > 1:
            buffer = buffer.reshape([cfg.SPLIT_K, block_shape[0] // cfg.SPLIT_K, block_shape[1]])
        if slice is not None:
            slice_size: ttgl.constexpr = buffer.shape[-2] // 2
            buffer = buffer.slice(start=slice * slice_size, length=slice_size, dim=-2)
        return buffer

    @gluon.jit
    def get_v_buffer(self, buf, slice=None):
        cfg = self.cfg
        block_shape: ttgl.constexpr = [cfg.HEAD_SZ, cfg.BLOCK_N * cfg.SPLIT_K // 32]
        buffer = self.v_mem.smem.index(buf)
        buffer = KVScaleMemory.unshuffle(buffer, block_shape)
        if cfg.SPLIT_K > 1:
            buffer = buffer.reshape([block_shape[0], cfg.SPLIT_K, block_shape[1] // cfg.SPLIT_K])
            buffer = buffer.permute([1, 0, 2])
        if slice is not None:
            slice_size: ttgl.constexpr = buffer.shape[-2] // 2
            buffer = buffer.slice(start=slice * slice_size, length=slice_size, dim=-2)
        return buffer


@aggregate
class AttentionConfigBase:
    Q_TYPE: ttgl.constexpr  # the data type for Q, either 'e5m2' or 'e4m3'
    P_TYPE: ttgl.constexpr  # the data type for P; we always assume P_TYPE == Q_TYPE
    KV_TYPE: ttgl.constexpr  # the data type for K and V, either 'e5m2', 'e4m3' or 'e2m1'
    BATCH: ttgl.constexpr
    SEQLEN_Q: ttgl.constexpr
    SEQLEN_K: ttgl.constexpr
    NUM_Q_HEADS: ttgl.constexpr
    NUM_K_HEADS: ttgl.constexpr
    HEAD_SZ: ttgl.constexpr
    BLOCK_M: ttgl.constexpr
    BLOCK_N: ttgl.constexpr
    SPLIT_K: ttgl.constexpr
    NUM_BUFFERS: ttgl.constexpr
    NUM_WARPS: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, BLOCK_M, BLOCK_N,
                 SPLIT_K, NUM_BUFFERS, NUM_WARPS):
        self.Q_TYPE = ttgl.constexpr(Q_TYPE)
        self.P_TYPE = ttgl.constexpr(Q_TYPE)
        self.KV_TYPE = ttgl.constexpr(KV_TYPE)
        self.BATCH = ttgl.constexpr(BATCH)
        self.SEQLEN_Q = ttgl.constexpr(SEQLEN_Q)
        self.SEQLEN_K = ttgl.constexpr(SEQLEN_K)
        self.NUM_Q_HEADS = ttgl.constexpr(NUM_Q_HEADS)
        self.NUM_K_HEADS = ttgl.constexpr(NUM_K_HEADS)
        self.HEAD_SZ = ttgl.constexpr(HEAD_SZ)
        self.BLOCK_M = ttgl.constexpr(BLOCK_M)
        self.BLOCK_N = ttgl.constexpr(BLOCK_N)
        self.SPLIT_K = ttgl.constexpr(SPLIT_K)
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
    # The divisor for packed K, V; always 1 for global-scaled (mxfp8).
    KV_PACK_DIV: ttgl.constexpr
    # Whether to use pingpong schedule
    PINGPONG: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, Q_TYPE, KV_TYPE, BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ,  #
                 BLOCK_M, BLOCK_N, SPLIT_K, SUBTILE, PINGPONG, P_K_WIDTH, NUM_BUFFERS, NUM_WARPS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert KV_TYPE in ['e5m2', 'e4m3']
        assert P_K_WIDTH == 16 or P_K_WIDTH == 8

        self.base = AttentionConfigBase(Q_TYPE, KV_TYPE, BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ,
                                        BLOCK_M, BLOCK_N, SPLIT_K, NUM_BUFFERS, NUM_WARPS)

        shape = [BLOCK_M, min(BLOCK_N, HEAD_SZ)] if not SUBTILE else \
                [BLOCK_M, min(BLOCK_N // 2, HEAD_SZ // 2)]

        wmma_layout = partial(get_wmma_layout, num_warps=NUM_WARPS)
        if SPLIT_K == 1:

            q_layout = ttgl.DotOperandLayout(0, wmma_layout(shape), k_width=16)
            k_layout = ttgl.DotOperandLayout(1, wmma_layout(shape), k_width=16)
            p_layout = ttgl.DotOperandLayout(0, wmma_layout(shape), k_width=P_K_WIDTH)
            v_layout = ttgl.DotOperandLayout(1, wmma_layout(shape), k_width=P_K_WIDTH)

            acc_layout = wmma_layout(shape)
        else:
            z = SPLIT_K

            q_layout = ttgl.DotOperandLayout(0, wmma_layout([1, *shape]), k_width=16)
            k_layout = ttgl.DotOperandLayout(1, wmma_layout([z, *shape]), k_width=16)
            p_layout = ttgl.DotOperandLayout(0, wmma_layout([z, *shape]), k_width=P_K_WIDTH)
            v_layout = ttgl.DotOperandLayout(1, wmma_layout([z, *shape]), k_width=P_K_WIDTH)

            acc_layout = wmma_layout([z, *shape])

        self.q_layout = ttgl.constexpr(q_layout)
        self.k_layout = ttgl.constexpr(k_layout)
        self.p_layout = ttgl.constexpr(p_layout)
        self.v_layout = ttgl.constexpr(v_layout)
        self.acc_layout = ttgl.constexpr(acc_layout)

        self.KV_PACK_DIV = ttgl.constexpr(2 if KV_TYPE == 'e2m1' else 1)
        self.SUBTILE = ttgl.constexpr(SUBTILE)
        self.PINGPONG = ttgl.constexpr(PINGPONG)
        self.CONVERT_LAYOUT_TRIVIAL = ttgl.constexpr(True if P_K_WIDTH == 8 else False)


@aggregate
class GlobalScaledAttentionProgram:
    cfg: GlobalScaledAttentionConfig

    q_blk: MemoryBlock
    q_scale: ttgl.tensor
    kv_mem: KVMemory
    k_off: ttgl.tuple
    v_off: ttgl.tuple
    k_scale: ttgl.tensor
    v_scale: ttgl.tensor
    # TODO: sm_scale should be a constexpr but the current llvm can not properly
    # fuse v_fma for literal operands, so we are using tensor here to ensure
    # it is in a register. Change it back to constexpr once the llvm is fixed.
    sm_scale: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg,  #
                 q_blk, q_scale,  #
                 kv_mem, k_off, v_off,  #
                 k_scale, v_scale,  #
                 sm_scale):
        self.cfg = cfg
        self.q_blk = q_blk
        self.q_scale = q_scale
        self.kv_mem = kv_mem
        self.k_off = k_off
        self.v_off = v_off
        self.k_scale = k_scale
        self.v_scale = v_scale
        self.sm_scale = sm_scale

    @gluon.jit
    def initialize(cfg, q_ptr, q_scale, kv_mem, k_scale, v_scale, sm_scale):
        ttgl.static_assert(isinstance(cfg, GlobalScaledAttentionConfig))

        SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
        SEQLEN_Q: ttgl.constexpr = cfg.SEQLEN_Q
        HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
        NUM_Q_HEADS: ttgl.constexpr = cfg.NUM_Q_HEADS
        NUM_K_HEADS: ttgl.constexpr = cfg.NUM_K_HEADS
        BLOCK_M: ttgl.constexpr = cfg.BLOCK_M
        SPLIT_K: ttgl.constexpr = cfg.SPLIT_K
        GROUP_SZ: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
        NUM_GROUPS: ttgl.constexpr = NUM_K_HEADS

        off_h = ttgl.program_id(0)
        off_m = ttgl.program_id(1)
        off_z = ttgl.program_id(2)

        ttgl.static_assert(SPLIT_K > 0)
        ttgl.static_assert(SEQLEN_K % SPLIT_K == 0)

        if SEQLEN_Q == SEQLEN_K:
            off_hk = off_h // GROUP_SZ

            q_off = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h) + \
                    BLOCK_M * HEAD_SZ * off_m
            q_blk = MemoryBlock.initialize(  #
                q_ptr + q_off,  #
                shape=[SEQLEN_Q, HEAD_SZ],  #
                block_shape=[BLOCK_M, HEAD_SZ],  #
                layout=cfg.q_layout)

        else:
            off_hk = off_h

            q_off = GROUP_SZ * HEAD_SZ * (NUM_GROUPS * off_z + off_h) + \
                    BLOCK_M * HEAD_SZ * off_m

            if SPLIT_K == 1:
                q_blk = MemoryBlock.initialize(  #
                    q_ptr + q_off,  #
                    shape=[GROUP_SZ, HEAD_SZ],  #
                    block_shape=[BLOCK_M, HEAD_SZ],  #
                    layout=cfg.q_layout)
            else:
                q_blk = MemoryBlock.initialize(  #
                    q_ptr + q_off,  #
                    shape=[1, GROUP_SZ, HEAD_SZ],  #
                    block_shape=[1, BLOCK_M, HEAD_SZ],  #
                    layout=cfg.q_layout)

        k_off = [kv_mem.k_shape[2] * (kv_mem.k_shape[1] * off_z + off_hk), 0]
        v_off = [kv_mem.v_shape[2] * (kv_mem.v_shape[1] * off_z + off_hk), 0]

        return GlobalScaledAttentionProgram(  #
            cfg,  #
            q_blk, q_scale,  #
            kv_mem,  #
            k_off, v_off,  #
            k_scale, v_scale,  #
            sm_scale)

    @gluon.jit
    def global_load_q(self):
        q_blk = self.q_blk
        q = buffer_load(q_blk.ptr, q_blk.offs, q_blk.mask, other=0.0)
        return q

    @gluon.jit
    def issue_global_load_k(self, idx, sub_idx=0, buf=0, pred=1):
        self.kv_mem.issue_load_k(self.k_off, idx, sub_idx, buf, pred)

    @gluon.jit
    def issue_global_load_v(self, idx, sub_idx=0, buf=0, pred=1):
        self.kv_mem.issue_load_v(self.v_off, idx, sub_idx, buf, pred)

    @gluon.jit
    def shared_load_k(self, sub_idx=0, buf=0):
        cfg = self.cfg

        k_buffer = self.kv_mem.get_k_buffer(sub_idx, buf)
        k = k_buffer.load(cfg.k_layout)
        return k

    @gluon.jit
    def shared_load_v(self, sub_idx=0, buf=0):
        cfg = self.cfg

        v_buffer = self.kv_mem.get_v_buffer(sub_idx, buf)
        v = v_buffer.load(cfg.v_layout)
        return v

    @gluon.jit
    def compute_qk(self, q, q_scale, k, k_scale, acc):
        cfg = self.cfg

        if cfg.SPLIT_K > 1:
            q = q.broadcast_to([cfg.SPLIT_K, q.shape[1], q.shape[2]])
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
    def create_acc(self):
        cfg = self.cfg

        if cfg.SPLIT_K == 1:
            m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
            l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
            zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
            acc = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)
        else:
            m_i = ttgl.full([cfg.SPLIT_K, cfg.BLOCK_M], float("-inf"), ttgl.float32,
                            ttgl.SliceLayout(2, cfg.acc_layout))
            l_i = ttgl.full([cfg.SPLIT_K, cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(2, cfg.acc_layout))
            zero = ttgl.full([cfg.SPLIT_K, cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
            acc = ttgl.full([cfg.SPLIT_K, cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

        return m_i, l_i, zero, acc

    @gluon.jit
    def fwd_loop(self):
        cfg = self.cfg

        m_i, l_i, zero, acc = self.create_acc()

        sm_scale = self.sm_scale
        q_scale = self.q_scale.to(ttgl.uint8)
        k_scale = self.k_scale.to(ttgl.uint8)
        p_scale = 0x7F
        p_scale = p_scale.to(ttgl.uint8)
        v_scale = self.v_scale.to(ttgl.uint8)

        q = self.global_load_q()

        end = ttgl.cdiv(cfg.SEQLEN_K // cfg.SPLIT_K, cfg.BLOCK_N)
        for i in range(0, end):
            self.issue_global_load_k(i)

            self.async_wait(0)
            k = self.shared_load_k()

            qk = self.compute_qk(q, q_scale, k, k_scale, zero)

            m = ttgl.max(qk, -1)
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
            p = ttgl.exp2(qk_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            m_i = m_ij
            alpha = ttgl.exp2(m_diff)
            l_ij = ttgl.sum(p, -1)
            acc = acc * expand_dims(alpha, -1)
            l_i = l_i * alpha + l_ij
            p = self.downcast_p(p)

            self.issue_global_load_v(i)

            self.async_wait(0)
            v = self.shared_load_v()

            acc = self.compute_pv(p, p_scale, v, v_scale, acc)

        return acc, l_i, m_i

    @gluon.jit
    def fwd_pipeline(self):
        cfg = self.cfg

        m_i, l_i, zero, acc = self.create_acc()

        sm_scale = self.sm_scale
        q_scale = self.q_scale.to(ttgl.uint8)
        k_scale = self.k_scale.to(ttgl.uint8)
        p_scale = 0x7F
        p_scale = p_scale.to(ttgl.uint8)
        v_scale = self.v_scale.to(ttgl.uint8)

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

        m = ttgl.max(qk, -1)  # ............................................... iter 0
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
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
        end = ttgl.cdiv(cfg.SEQLEN_K // cfg.SPLIT_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 2
            b = 1 - a
            pred = i - end + 3
            pred = (pred >> 31) & 1

            qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ............. iter i+1
            l_ij = ttgl.sum(p, -1)  # ......................................... iter i
            acc = acc * expand_dims(alpha, -1)
            l_i = l_i * alpha + l_ij
            p = self.downcast_p(p)

            self.async_wait(2)  # ............................................. iter i
            v = self.shared_load_v(buf=a)
            self.issue_global_load_k(i + 3, buf=b, pred=pred)  # .............. iter i+3

            acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ............. iter i
            m = ttgl.max(qk, -1)  # ........................................... iter i+1
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
            p = ttgl.exp2(qk_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            alpha = ttgl.exp2(m_diff)
            m_i = m_ij

            self.async_wait(2)  # ............................................. iter i+2
            k = self.shared_load_k(buf=a)
            self.issue_global_load_v(i + 2, buf=a)  # ......................... iter i+2

        # pipeline epilogue, iter end-2
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter end-1
        l_ij = ttgl.sum(p, -1)  # ............................................. iter end-2
        acc = acc * expand_dims(alpha, -1)
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(2)  # ................................................. iter end-2
        v = self.shared_load_v(buf=0)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-2
        m = ttgl.max(qk, -1)  # ............................................... iter end-1
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        # pipeline epilogue, iter end-1
        l_ij = ttgl.sum(p, -1)  # ............................................. iter end-1
        acc = acc * expand_dims(alpha, -1)
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(0)  # ................................................. iter end-1
        v = self.shared_load_v(buf=1)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-1

        return acc, l_i, m_i

    @gluon.jit
    def fwd_pipeline_subtile(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc0 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc1 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)

        sm_scale = self.sm_scale
        q_scale = self.q_scale.to(ttgl.uint8)
        k_scale = self.k_scale.to(ttgl.uint8)
        p_scale = 0x7F
        p_scale = p_scale.to(ttgl.uint8)
        v_scale = self.v_scale.to(ttgl.uint8)

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
        m = ttgl.max(qk, -1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        self.issue_global_load_k(2, sub_idx=0, buf=0)  # ...................... iter 2

        self.async_wait(4)  # ................................................. iter 1
        k0 = self.shared_load_k(sub_idx=0, buf=1)
        qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)  # ........ iter 0
        qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
        p0 = ttgl.exp2(qk0_shifted)
        self.issue_global_load_k(2, sub_idx=1, buf=0)  # ...................... iter 2

        end = ttgl.cdiv(cfg.SEQLEN_K // cfg.SPLIT_K, cfg.BLOCK_N)
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
            acc0 = acc0 * expand_dims(alpha, -1)
            acc1 = acc1 * expand_dims(alpha, -1)
            self.issue_global_load_v(i + 1, sub_idx=0, buf=b)  # .............. iter i+1

            qk1 = self.compute_qk(q, q_scale, k1, k_scale, zero)  # ........... iter i+1
            self.async_wait(4)  # ............................................. iter i
            v0 = self.shared_load_v(sub_idx=0, buf=a)
            p = self.concat_subtile(p0, p1)  # ................................ iter i
            l_ij = ttgl.sum(p, -1)
            l_i = l_i * alpha + l_ij
            p = self.downcast_p(p)
            self.issue_global_load_v(i + 1, sub_idx=1, buf=b)  # .............. iter i+1

            acc0 = self.compute_pv(p, p_scale, v0, v_scale, acc0)  # .......... iter i
            self.async_wait(4)  # ............................................. iter i
            v1 = self.shared_load_v(sub_idx=1, buf=a)
            qk = self.concat_subtile(qk0, qk1)  # ............................. iter i+1
            m = ttgl.max(qk, -1)
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            self.issue_global_load_k(i + 3, sub_idx=0, buf=b, pred=pred)  # ... iter i+3

            acc1 = self.compute_pv(p, p_scale, v1, v_scale, acc1)  # .......... iter i
            self.async_wait(4)  # ............................................. iter i+2
            k0 = self.shared_load_k(sub_idx=0, buf=a)
            qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)  # .... iter i+1
            qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
            p0 = ttgl.exp2(qk0_shifted)
            self.issue_global_load_k(i + 3, sub_idx=1, buf=b, pred=pred)  # ... iter i+3

        # pipeline epilogue iter end-2
        self.issue_global_load_v(end - 1, sub_idx=0, buf=1)
        self.issue_global_load_v(end - 1, sub_idx=1, buf=1)

        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * expand_dims(alpha, -1)
        acc1 = acc1 * expand_dims(alpha, -1)

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, -1)
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
        m = ttgl.max(qk, -1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)
        qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
        p0 = ttgl.exp2(qk0_shifted)
        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * expand_dims(alpha, -1)
        acc1 = acc1 * expand_dims(alpha, -1)

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, -1)
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(0)
        v0 = self.shared_load_v(sub_idx=0, buf=1)
        v1 = self.shared_load_v(sub_idx=1, buf=1)

        acc0 = self.compute_pv(p, p_scale, v0, v_scale, acc0)
        acc1 = self.compute_pv(p, p_scale, v1, v_scale, acc1)

        # write output
        acc = self.concat_subtile(acc0, acc1)
        return acc, l_i, m_i

    @gluon.jit
    def fwd_pipeline_subtile_pingpong(self):
        cfg = self.cfg

        m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
        zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc0 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)
        acc1 = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ // 2], 0.0, ttgl.float32, cfg.acc_layout)

        sm_scale = self.sm_scale
        q_scale = self.q_scale.to(ttgl.uint8)
        k_scale = self.k_scale.to(ttgl.uint8)
        p_scale = 0x7F
        p_scale = p_scale.to(ttgl.uint8)
        v_scale = self.v_scale.to(ttgl.uint8)

        q = self.global_load_q()

        # pipeline prologue, iter -3
        self.issue_global_load_k(0, sub_idx=0, buf=0)  # ...................... iter 0

        self.issue_global_load_k(0, sub_idx=1, buf=0)  # ...................... iter 0

        # pipeline prologue, iter -2
        self.issue_global_load_k(1, sub_idx=0, buf=1)  # ...................... iter 1

        self.async_wait(2)
        k0 = self.shared_load_k(sub_idx=0, buf=0)  # .......................... iter 0
        self.issue_global_load_k(1, sub_idx=1, buf=1)  # ...................... iter 1

        # pipeline prologue, iter -1
        qk0 = self.compute_qk(q, q_scale, k0, k_scale, zero)  # ............... iter 0
        self.async_wait(2)
        k1 = self.shared_load_k(sub_idx=1, buf=0)  # .......................... iter 0
        self.issue_global_load_v(0, sub_idx=0, buf=0)  # ...................... iter 0

        qk1 = self.compute_qk(q, q_scale, k1, k_scale, zero)  # ............... iter 0
        self.issue_global_load_v(0, sub_idx=1, buf=0)  # ...................... iter 0

        qk = self.concat_subtile(qk0, qk1)  # ................................. iter 0
        m = ttgl.max(qk, -1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        self.issue_global_load_k(2, sub_idx=0, buf=0)  # ...................... iter 2

        self.async_wait(4)
        k0 = self.shared_load_k(sub_idx=0, buf=1)  # .......................... iter 1
        qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)  # ........ iter 0
        qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
        p0 = ttgl.exp2(qk0_shifted)
        self.issue_global_load_k(2, sub_idx=1, buf=0)  # ...................... iter 2

        end = ttgl.cdiv(cfg.SEQLEN_K // cfg.SPLIT_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 2
            b = 1 - a
            pred = i - end + 3
            pred = (pred >> 31) & 1

            with warp_pipeline_stage("compute0"):
                qk0 = self.compute_qk(q, q_scale, k0, k_scale, zero)  # ....... iter i+1
                p1 = ttgl.exp2(qk1_shifted)  # ................................ iter i
                m_diff = m_i * sm_scale - m_ij_scaled
                m_i = m_ij
                alpha = ttgl.exp2(m_diff)
                acc0 = acc0 * expand_dims(alpha, -1)
                acc1 = acc1 * expand_dims(alpha, -1)

            self.async_wait(4)
            with warp_pipeline_stage("memory0"):
                k1 = self.shared_load_k(sub_idx=1, buf=b)  # .................. iter i+1
                self.issue_global_load_v(i + 1, sub_idx=0, buf=b)  # .......... iter i+1

            with warp_pipeline_stage("compute1"):
                qk1 = self.compute_qk(q, q_scale, k1, k_scale, zero)  # ....... iter i+1
                p = self.concat_subtile(p0, p1)  # ............................ iter i
                l_ij = ttgl.sum(p, -1)
                l_i = l_i * alpha + l_ij
                p = self.downcast_p(p)

            self.async_wait(4)
            with warp_pipeline_stage("memory1"):
                v0 = self.shared_load_v(sub_idx=0, buf=a)  # .................. iter i
                self.issue_global_load_v(i + 1, sub_idx=1, buf=b)  # .......... iter i+1

            with warp_pipeline_stage("compute2"):
                acc0 = self.compute_pv(p, p_scale, v0, v_scale, acc0)  # ...... iter i
                qk = self.concat_subtile(qk0, qk1)  # ......................... iter i+1
                m = ttgl.max(qk, -1)
                m_ij = ttgl.maximum(m_i, m)
                m_ij_scaled = m_ij * sm_scale

            self.async_wait(4)
            with warp_pipeline_stage("memory2"):
                v1 = self.shared_load_v(sub_idx=1, buf=a)  # .................. iter i
                self.issue_global_load_k(i + 3, sub_idx=0, buf=b, pred=pred)  # iter i+3

            with warp_pipeline_stage("compute3"):
                acc1 = self.compute_pv(p, p_scale, v1, v_scale, acc1)  # ...... iter i
                qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)  # iter i+1
                qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
                p0 = ttgl.exp2(qk0_shifted)

            self.async_wait(4)
            with warp_pipeline_stage("memory3"):
                k0 = self.shared_load_k(sub_idx=0, buf=a)  # .................. iter i+2
                self.issue_global_load_k(i + 3, sub_idx=1, buf=b, pred=pred)  # iter i+3

        # pipeline epilogue iter end-2
        self.issue_global_load_v(end - 1, sub_idx=0, buf=1)
        self.issue_global_load_v(end - 1, sub_idx=1, buf=1)

        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * expand_dims(alpha, -1)
        acc1 = acc1 * expand_dims(alpha, -1)

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, -1)
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
        m = ttgl.max(qk, -1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale

        qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)
        qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
        p0 = ttgl.exp2(qk0_shifted)

        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * expand_dims(alpha, -1)
        acc1 = acc1 * expand_dims(alpha, -1)

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, -1)
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(0)
        v0 = self.shared_load_v(sub_idx=0, buf=1)
        v1 = self.shared_load_v(sub_idx=1, buf=1)

        acc0 = self.compute_pv(p, p_scale, v0, v_scale, acc0)
        acc1 = self.compute_pv(p, p_scale, v1, v_scale, acc1)

        acc = self.concat_subtile(acc0, acc1)
        return acc, l_i, m_i

    @gluon.jit
    def fwd_pipeline_triplebuf(self):
        cfg = self.cfg

        m_i, l_i, zero, acc = self.create_acc()

        sm_scale = self.sm_scale
        q_scale = self.q_scale.to(ttgl.uint8)
        k_scale = self.k_scale.to(ttgl.uint8)
        p_scale = 0x7F
        p_scale = p_scale.to(ttgl.uint8)
        v_scale = self.v_scale.to(ttgl.uint8)

        q = self.global_load_q()

        # pipeline prologue, iter -4
        self.issue_global_load_k(0, buf=0)  # ................................. iter 0

        # pipeline prologue, iter -3
        self.issue_global_load_k(1, buf=1)  # ................................. iter 1

        # pipeline prologue, iter -2
        self.issue_global_load_v(0, buf=0)  # ................................. iter 0

        self.async_wait(2)
        self.issue_global_load_k(2, buf=2)  # ................................. iter 2
        k = self.shared_load_k(buf=0)  # ...................................... iter 0

        # pipeline prologue, iter -1
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter 0

        self.issue_global_load_v(1, buf=1)  # ................................. iter 1

        m = ttgl.max(qk, -1)  # ............................................... iter 0
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        self.async_wait(3)
        self.issue_global_load_k(3, buf=0)  # ................................. iter 3
        k = self.shared_load_k(buf=1)  # ...................................... iter 1

        # main loop from 0 to end-3
        end = ttgl.cdiv(cfg.SEQLEN_K // cfg.SPLIT_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 3
            b = (i + 1) % 3
            c = (i + 2) % 3
            pred = i - end + 4
            pred = (pred >> 31) & 1

            qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ............. iter i+1
            l_ij = ttgl.sum(p, -1)  # ......................................... iter i
            acc = acc * expand_dims(alpha, -1)
            l_i = l_i * alpha + l_ij
            p = self.downcast_p(p)

            self.async_wait(3)
            self.issue_global_load_v(i + 2, buf=c)  # ......................... iter i+2
            v = self.shared_load_v(buf=a)  # .................................. iter i

            acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ............. iter i
            m = ttgl.max(qk, -1)  # ........................................... iter i+1
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
            p = ttgl.exp2(qk_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            alpha = ttgl.exp2(m_diff)
            m_i = m_ij

            self.async_wait(3)
            self.issue_global_load_k(i + 4, buf=b, pred=pred)  # .............. iter i+4
            k = self.shared_load_k(buf=c)  # .................................. iter i+2

        # pipeline epilogue, iter end-2
        a = (end - 1) % 3

        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter end-1
        l_ij = ttgl.sum(p, -1)  # ............................................. iter end-2
        acc = acc * expand_dims(alpha, -1)
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(1)
        v = self.shared_load_v(buf=a)  # ...................................... iter end-2

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-2
        m = ttgl.max(qk, -1)  # ............................................... iter end-1
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        # pipeline epilogue, iter end-1
        a = (end - 1) % 3

        l_ij = ttgl.sum(p, -1)
        acc = acc * expand_dims(alpha, -1)
        l_i = l_i * alpha + l_ij
        p = self.downcast_p(p)

        self.async_wait(0)
        v = self.shared_load_v(buf=a)  # ...................................... iter end-1

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-1

        return acc, l_i, m_i


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
    def __init__(self, Q_TYPE, KV_TYPE, BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, P_SCALING,  #
                 BLOCK_M, BLOCK_N, SPLIT_K, SUBTILE, PINGPONG, P_K_WIDTH, NUM_BUFFERS, NUM_WARPS):
        assert Q_TYPE in ['e5m2', 'e4m3']
        assert KV_TYPE in ['e5m2', 'e4m3', 'e2m1']
        assert P_K_WIDTH == 16 or (KV_TYPE != 'e2m1' and P_K_WIDTH == 8)
        self.base = AttentionConfigBase(Q_TYPE, KV_TYPE, BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ,
                                        BLOCK_M, BLOCK_N, SPLIT_K, NUM_BUFFERS, NUM_WARPS)

        packed = (KV_TYPE == 'e2m1')
        shape = [BLOCK_M, min(BLOCK_N, HEAD_SZ)] if not SUBTILE else \
                [BLOCK_M, min(BLOCK_N // 2, HEAD_SZ // 2)]

        wmma_layout = partial(get_wmma_layout, num_warps=NUM_WARPS, preshuffled=True)
        if SPLIT_K == 1:
            q_layout = ttgl.DotOperandLayout(0, wmma_layout(shape), k_width=16)
            k_layout = ttgl.DotOperandLayout(1, wmma_layout(shape, packed=packed), k_width=16)
            p_layout = ttgl.DotOperandLayout(0, wmma_layout(shape), k_width=P_K_WIDTH)
            v_layout = ttgl.DotOperandLayout(1, wmma_layout(shape, packed=packed), k_width=P_K_WIDTH)

            q_scale_layout = get_wmma_scale_layout(q_layout, [BLOCK_M, HEAD_SZ // 32])
            k_scale_layout = get_wmma_scale_layout(k_layout, [BLOCK_N, HEAD_SZ // 32])
            p_scale_layout = get_wmma_scale_layout(p_layout, [BLOCK_M, BLOCK_N // 32])
            v_scale_layout = get_wmma_scale_layout(v_layout, [HEAD_SZ, BLOCK_N // 32])

            acc_layout = wmma_layout(shape)
        else:
            z = SPLIT_K

            q_layout = ttgl.DotOperandLayout(0, wmma_layout([1, *shape]), k_width=16)
            k_layout = ttgl.DotOperandLayout(1, wmma_layout([z, *shape], packed=packed), k_width=16)
            p_layout = ttgl.DotOperandLayout(0, wmma_layout([z, *shape]), k_width=P_K_WIDTH)
            v_layout = ttgl.DotOperandLayout(1, wmma_layout([z, *shape], packed=packed), k_width=P_K_WIDTH)

            q_scale_layout = get_wmma_scale_layout(q_layout, [1, BLOCK_M, HEAD_SZ // 32])
            k_scale_layout = get_wmma_scale_layout(k_layout, [SPLIT_K, BLOCK_N, HEAD_SZ // 32])
            p_scale_layout = get_wmma_scale_layout(p_layout, [SPLIT_K, BLOCK_M, BLOCK_N // 32])
            v_scale_layout = get_wmma_scale_layout(v_layout, [SPLIT_K, HEAD_SZ, BLOCK_N // 32])

            acc_layout = wmma_layout([z, *shape])

        self.q_layout = ttgl.constexpr(q_layout)
        self.k_layout = ttgl.constexpr(k_layout)
        self.p_layout = ttgl.constexpr(p_layout)
        self.v_layout = ttgl.constexpr(v_layout)
        self.q_scale_layout = ttgl.constexpr(q_scale_layout)
        self.k_scale_layout = ttgl.constexpr(k_scale_layout)
        self.p_scale_layout = ttgl.constexpr(p_scale_layout)
        self.v_scale_layout = ttgl.constexpr(v_scale_layout)
        self.acc_layout = ttgl.constexpr(acc_layout)

        self.KV_PACK_DIV = ttgl.constexpr(2 if KV_TYPE == 'e2m1' else 1)
        self.SUBTILE = ttgl.constexpr(SUBTILE)
        self.PINGPONG = ttgl.constexpr(PINGPONG)
        self.CONVERT_LAYOUT_TRIVIAL = ttgl.constexpr(True if P_K_WIDTH == 8 else False)
        self.P_SCALING = ttgl.constexpr(P_SCALING)


@aggregate
class BlockScaledAttentionProgram:
    cfg: BlockScaledAttentionConfig

    q_blk: MemoryBlock
    q_scale_blk: MemoryBlock
    kv_mem: KVMemory
    kv_scale_mem: KVScaleMemory
    k_off: ttgl.tuple
    v_off: ttgl.tuple
    k_scale_off: ttgl.tuple
    v_scale_off: ttgl.tuple
    # TODO: sm_scale should be a constexpr but the current llvm can not properly
    # fuse v_fma for literal operands, so we are using tensor here to ensure
    # it is in a register. Change it back to constexpr once the llvm is fixed.
    sm_scale: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg,  #
                 q_blk, q_scale_blk,  #
                 kv_mem, kv_scale_mem,  #
                 k_off, v_off,  #
                 k_scale_off, v_scale_off,  #
                 sm_scale):
        self.cfg = cfg
        self.q_blk = q_blk
        self.q_scale_blk = q_scale_blk
        self.kv_mem = kv_mem
        self.kv_scale_mem = kv_scale_mem
        self.k_off = k_off
        self.v_off = v_off
        self.k_scale_off = k_scale_off
        self.v_scale_off = v_scale_off
        self.sm_scale = sm_scale

    @gluon.jit
    def initialize(cfg, q_ptr, q_scale_ptr, kv_mem, kv_scale_mem, sm_scale):
        ttgl.static_assert(isinstance(cfg, BlockScaledAttentionConfig))

        SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
        SEQLEN_Q: ttgl.constexpr = cfg.SEQLEN_Q
        HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
        NUM_Q_HEADS: ttgl.constexpr = cfg.NUM_Q_HEADS
        NUM_K_HEADS: ttgl.constexpr = cfg.NUM_K_HEADS
        BLOCK_M: ttgl.constexpr = cfg.BLOCK_M
        SPLIT_K: ttgl.constexpr = cfg.SPLIT_K
        GROUP_SZ: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
        NUM_GROUPS: ttgl.constexpr = NUM_K_HEADS

        off_h = ttgl.program_id(0)
        off_m = ttgl.program_id(1)
        off_z = ttgl.program_id(2)

        ttgl.static_assert(SPLIT_K > 0)
        ttgl.static_assert(SEQLEN_K % SPLIT_K == 0)

        if SEQLEN_Q == SEQLEN_K:
            off_hk = off_h // GROUP_SZ

            q_off = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h) + \
                    BLOCK_M * HEAD_SZ * off_m
            q_blk = MemoryBlock.initialize(  #
                base=q_ptr + q_off,  #
                shape=[SEQLEN_Q, HEAD_SZ],  #
                block_shape=[BLOCK_M, HEAD_SZ],  #
                layout=cfg.q_layout)

            q_scale_off = SEQLEN_Q * (HEAD_SZ // 32) * (NUM_Q_HEADS * off_z + off_h) + \
                          BLOCK_M * (HEAD_SZ // 32) * off_m
            q_scale_blk = MemoryBlock.initialize(  #
                base=q_scale_ptr + q_scale_off,  #
                shape=[SEQLEN_Q, HEAD_SZ // 32],  #
                block_shape=[BLOCK_M, HEAD_SZ // 32],  #
                layout=cfg.q_scale_layout)

        else:
            off_hk = off_h

            q_off = GROUP_SZ * HEAD_SZ * (NUM_GROUPS * off_z + off_h) + \
                    BLOCK_M * HEAD_SZ * off_m
            q_scale_off = GROUP_SZ * (HEAD_SZ // 32) * (NUM_GROUPS * off_z + off_h) + \
                          BLOCK_M * (HEAD_SZ // 32) * off_m

            if SPLIT_K == 1:
                q_blk = MemoryBlock.initialize(  #
                    q_ptr + q_off,  #
                    shape=[GROUP_SZ, HEAD_SZ],  #
                    block_shape=[BLOCK_M, HEAD_SZ],  #
                    layout=cfg.q_layout)
                q_scale_blk = MemoryBlock.initialize(  #
                    base=q_scale_ptr + q_scale_off,  #
                    shape=[GROUP_SZ, HEAD_SZ // 32],  #
                    block_shape=[BLOCK_M, HEAD_SZ // 32],  #
                    layout=cfg.q_scale_layout)
            else:
                q_blk = MemoryBlock.initialize(  #
                    q_ptr + q_off,  #
                    shape=[1, GROUP_SZ, HEAD_SZ],  #
                    block_shape=[1, BLOCK_M, HEAD_SZ],  #
                    layout=cfg.q_layout)
                q_scale_blk = MemoryBlock.initialize(  #
                    base=q_scale_ptr + q_scale_off,  #
                    shape=[1, GROUP_SZ, HEAD_SZ // 32],  #
                    block_shape=[1, BLOCK_M, HEAD_SZ // 32],  #
                    layout=cfg.q_scale_layout)

        k_off = [kv_mem.k_shape[2] * (kv_mem.k_shape[1] * off_z + off_hk), 0]
        v_off = [kv_mem.v_shape[2] * (kv_mem.v_shape[1] * off_z + off_hk), 0]

        k_scale_off = [kv_scale_mem.k_shape[2] * (kv_scale_mem.k_shape[1] * off_z + off_hk), 0]
        v_scale_off = [kv_scale_mem.v_shape[2] * (kv_scale_mem.v_shape[1] * off_z + off_hk), 0]

        return BlockScaledAttentionProgram(  #
            cfg,  #
            q_blk, q_scale_blk,  #
            kv_mem, kv_scale_mem,  #
            k_off, v_off,  #
            k_scale_off, v_scale_off,  #
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
        self.kv_mem.issue_load_k(self.k_off, idx, sub_idx, buf, pred)

    @gluon.jit
    def issue_global_load_v(self, idx, sub_idx=0, buf=0, pred=1):
        self.kv_mem.issue_load_v(self.v_off, idx, sub_idx, buf, pred)

    @gluon.jit
    def issue_global_load_k_scale(self, idx, buf=0, pred=1):
        self.kv_scale_mem.issue_load_k(self.k_scale_off, idx, buf, pred)

    @gluon.jit
    def issue_global_load_v_scale(self, idx, buf=0, pred=1):
        self.kv_scale_mem.issue_load_v(self.v_scale_off, idx, buf, pred)

    @gluon.jit
    def shared_load_k(self, sub_idx=0, buf=0):
        cfg = self.cfg

        k_buffer = self.kv_mem.get_k_buffer(sub_idx, buf)
        k = k_buffer.load(cfg.k_layout)
        return k

    @gluon.jit
    def shared_load_v(self, sub_idx=0, buf=0):
        cfg = self.cfg

        v_buffer = self.kv_mem.get_v_buffer(sub_idx, buf)
        v = v_buffer.load(cfg.v_layout)
        return v

    @gluon.jit
    def shared_load_k_scale(self, buf=0, slice=None):
        cfg = self.cfg

        k_scale_buffer = self.kv_scale_mem.get_k_buffer(buf, slice=slice)
        k_scale = k_scale_buffer.load(cfg.k_scale_layout)
        return k_scale

    @gluon.jit
    def shared_load_v_scale(self, buf=0, slice=None):
        cfg = self.cfg

        v_scale_buffer = self.kv_scale_mem.get_v_buffer(buf, slice=slice)
        v_scale = v_scale_buffer.load(cfg.v_scale_layout)
        return v_scale

    @gluon.jit
    def compute_qk(self, q, q_scale, k, k_scale, acc):
        cfg = self.cfg

        if cfg.SPLIT_K > 1:
            q = q.broadcast_to([cfg.SPLIT_K, q.shape[1], q.shape[2]])
            q_scale = q_scale.broadcast_to([cfg.SPLIT_K, q_scale.shape[1], q_scale.shape[2]])
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
            # Flatten leading dims to reduce 3D (SPLIT_K > 1) to 2D
            if cfg.SPLIT_K > 1:
                p = ttgl.reshape(p, [cfg.SPLIT_K * cfg.BLOCK_M, cfg.BLOCK_N])
            p, p_scale = self.downcast_fp32_to_mxfp8(p, cfg.P_TYPE, [cfg.SPLIT_K * cfg.BLOCK_M, cfg.BLOCK_N])
            # Unflatten back to 3D
            if cfg.SPLIT_K > 1:
                p = ttgl.reshape(p, [cfg.SPLIT_K, cfg.BLOCK_M, cfg.BLOCK_N])
                p_scale = ttgl.reshape(p_scale, [cfg.SPLIT_K, cfg.BLOCK_M, cfg.BLOCK_N // 32])
            p_scale = ttgl.convert_layout(p_scale, cfg.p_scale_layout)
        else:
            if cfg.SPLIT_K == 1:
                p = self.downcast_fp32_to_fp8(p, cfg.P_TYPE)
                p_scale = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N // 32], 0x7F, ttgl.uint8, cfg.p_scale_layout)
            else:
                p = self.downcast_fp32_to_fp8(p, cfg.P_TYPE)
                p_scale = ttgl.full([cfg.SPLIT_K, cfg.BLOCK_M, cfg.BLOCK_N // 32], 0x7F, ttgl.uint8, cfg.p_scale_layout)
        p = ttgl.convert_layout(p, cfg.p_layout, cfg.CONVERT_LAYOUT_TRIVIAL)

        return p, p_scale

    @gluon.jit
    def async_wait(self, count):
        tdm.async_wait(count)

    @gluon.jit
    def downcast_fp32_to_mxfp8(self, x, x_format: ttgl.constexpr, shape: ttgl.constexpr):
        block_size: ttgl.constexpr = 32

        ttgl.static_assert(x_format == 'e4m3' or x_format == 'e5m2')
        dtype: ttgl.constexpr = ttgl.float8e4nv if x_format == 'e4m3' else ttgl.float8e5
        fp8_max: ttgl.constexpr = 57344.0 if dtype == 'e5m2' else 448.0

        ttgl.static_assert(x.dtype == ttgl.float32)
        ttgl.static_assert(len(shape) == 2)

        outer_dim: ttgl.constexpr = shape[0]
        inner_dim: ttgl.constexpr = shape[1]

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
    def create_acc(self):
        cfg = self.cfg

        if cfg.SPLIT_K == 1:
            m_i = ttgl.full([cfg.BLOCK_M], float("-inf"), ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
            l_i = ttgl.full([cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(1, cfg.acc_layout))
            zero = ttgl.full([cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
            acc = ttgl.full([cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)
        else:
            m_i = ttgl.full([cfg.SPLIT_K, cfg.BLOCK_M], float("-inf"), ttgl.float32,
                            ttgl.SliceLayout(2, cfg.acc_layout))
            l_i = ttgl.full([cfg.SPLIT_K, cfg.BLOCK_M], 1.0, ttgl.float32, ttgl.SliceLayout(2, cfg.acc_layout))
            zero = ttgl.full([cfg.SPLIT_K, cfg.BLOCK_M, cfg.BLOCK_N], 0.0, ttgl.float32, cfg.acc_layout)
            acc = ttgl.full([cfg.SPLIT_K, cfg.BLOCK_M, cfg.HEAD_SZ], 0.0, ttgl.float32, cfg.acc_layout)

        return m_i, l_i, zero, acc

    @gluon.jit
    def fwd_loop(self):
        cfg = self.cfg

        m_i, l_i, zero, acc = self.create_acc()
        sm_scale = self.sm_scale

        q = self.global_load_q()
        q_scale = self.global_load_q_scale()

        end = ttgl.cdiv(cfg.SEQLEN_K // cfg.SPLIT_K, cfg.BLOCK_N)
        for i in range(0, end):
            self.issue_global_load_k(i)
            self.issue_global_load_k_scale(i)

            self.async_wait(0)
            k = self.shared_load_k()
            k_scale = self.shared_load_k_scale()

            qk = self.compute_qk(q, q_scale, k, k_scale, zero)

            m = ttgl.max(qk, -1)
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
            p = ttgl.exp2(qk_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            m_i = m_ij
            alpha = ttgl.exp2(m_diff)
            l_ij = ttgl.sum(p, -1)
            acc = acc * expand_dims(alpha, -1)
            l_i = l_i * alpha + l_ij
            p, p_scale = self.downcast_p(p)

            self.issue_global_load_v(i)
            self.issue_global_load_v_scale(i)

            self.async_wait(0)
            v = self.shared_load_v()
            v_scale = self.shared_load_v_scale()

            acc = self.compute_pv(p, p_scale, v, v_scale, acc)

        return acc, l_i, m_i

    @gluon.jit
    def fwd_pipeline(self):
        cfg = self.cfg

        m_i, l_i, zero, acc = self.create_acc()
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

        m = ttgl.max(qk, -1)  # ............................................... iter 0
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
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
        end = ttgl.cdiv(cfg.SEQLEN_K // cfg.SPLIT_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 2
            b = 1 - a
            pred = i - end + 3
            pred = (pred >> 31) & 1

            qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ............. iter i+1
            l_ij = ttgl.sum(p, -1)  # ......................................... iter i
            acc = acc * expand_dims(alpha, -1)
            l_i = l_i * alpha + l_ij
            p, p_scale = self.downcast_p(p)

            self.async_wait(4)  # ............................................. iter i
            v = self.shared_load_v(buf=a)
            v_scale = self.shared_load_v_scale(buf=a)
            self.issue_global_load_k(i + 3, buf=b, pred=pred)  # .............. iter i+3
            self.issue_global_load_k_scale(i + 3, buf=b, pred=pred)  # ........ iter i+3

            acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ............. iter i
            m = ttgl.max(qk, -1)  # ........................................... iter i+1
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
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
        l_ij = ttgl.sum(p, -1)  # ............................................. iter end-2
        acc = acc * expand_dims(alpha, -1)
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(4)  # ................................................. iter end-2
        v = self.shared_load_v(buf=0)
        v_scale = self.shared_load_v_scale(buf=0)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-2
        m = ttgl.max(qk, -1)  # ............................................... iter end-1
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        # pipeline epilogue, iter end-1
        l_ij = ttgl.sum(p, -1)  # ............................................. iter end-1
        acc = acc * expand_dims(alpha, -1)
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(0)  # ................................................. iter end-1
        v = self.shared_load_v(buf=1)
        v_scale = self.shared_load_v_scale(buf=1)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-1

        return acc, l_i, m_i

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
        m = ttgl.max(qk, -1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        self.issue_global_load_k(2, sub_idx=0, buf=0)  # ...................... iter 2
        self.issue_global_load_k_scale(2, buf=0)  # ........................... iter 2

        self.async_wait(6)  # ................................................. iter 1
        k0 = self.shared_load_k(sub_idx=0, buf=1)
        self.async_wait(5)  # ................................................. iter 1
        k0_scale = self.shared_load_k_scale(buf=1, slice=0)
        k1_scale = self.shared_load_k_scale(buf=1, slice=1)
        qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)  # ........ iter 0
        qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
        p0 = ttgl.exp2(qk0_shifted)
        self.issue_global_load_k(2, sub_idx=1, buf=0)  # ...................... iter 2

        end = ttgl.cdiv(cfg.SEQLEN_K // cfg.SPLIT_K, cfg.BLOCK_N)
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
            acc0 = acc0 * expand_dims(alpha, -1)
            acc1 = acc1 * expand_dims(alpha, -1)
            self.issue_global_load_v(i + 1, sub_idx=0, buf=b)  # .............. iter i+1
            self.issue_global_load_v_scale(i + 1, buf=b)  # ................... iter i+1

            qk1 = self.compute_qk(q, q_scale, k1, k1_scale, zero)  # .......... iter i+1
            self.async_wait(6)  # ............................................. iter i
            v0 = self.shared_load_v(sub_idx=0, buf=a)
            self.async_wait(5)  # ............................................. iter i
            v0_scale = self.shared_load_v_scale(buf=a, slice=0)
            v1_scale = self.shared_load_v_scale(buf=a, slice=1)
            p = self.concat_subtile(p0, p1)  # ................................ iter i
            l_ij = ttgl.sum(p, -1)
            l_i = l_i * alpha + l_ij
            p, p_scale = self.downcast_p(p)
            self.issue_global_load_v(i + 1, sub_idx=1, buf=b)  # .............. iter i+1

            acc0 = self.compute_pv(p, p_scale, v0, v0_scale, acc0)  # ......... iter i
            self.async_wait(5)  # ............................................. iter i
            v1 = self.shared_load_v(sub_idx=1, buf=a)
            qk = self.concat_subtile(qk0, qk1)  # ............................. iter i+1
            m = ttgl.max(qk, -1)
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
            qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)  # .... iter i+1
            qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
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
        acc0 = acc0 * expand_dims(alpha, -1)
        acc1 = acc1 * expand_dims(alpha, -1)

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, -1)
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
        m = ttgl.max(qk, -1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale

        qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)
        qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
        p0 = ttgl.exp2(qk0_shifted)

        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * expand_dims(alpha, -1)
        acc1 = acc1 * expand_dims(alpha, -1)

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, -1)
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
        return acc, l_i, m_i

    @gluon.jit
    def fwd_pipeline_subtile_pingpong(self):
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

        self.async_wait(5)
        k0 = self.shared_load_k(sub_idx=0, buf=0)  # .......................... iter 0
        k0_scale = self.shared_load_k_scale(buf=0, slice=0)
        k1_scale = self.shared_load_k_scale(buf=0, slice=1)
        self.issue_global_load_k(1, sub_idx=1, buf=1)  # ...................... iter 1

        # pipeline prologue, iter -1
        qk0 = self.compute_qk(q, q_scale, k0, k0_scale, zero)  # .............. iter 0
        self.async_wait(5)
        k1 = self.shared_load_k(sub_idx=1, buf=0)  # .......................... iter 0
        self.issue_global_load_v(0, sub_idx=0, buf=0)  # ...................... iter 0
        self.issue_global_load_v_scale(0, buf=0)  # ........................... iter 0

        qk1 = self.compute_qk(q, q_scale, k1, k1_scale, zero)  # .............. iter 0
        self.issue_global_load_v(0, sub_idx=1, buf=0)  # ...................... iter 0

        qk = self.concat_subtile(qk0, qk1)  # ................................. iter 0
        m = ttgl.max(qk, -1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        self.issue_global_load_k(2, sub_idx=0, buf=0)  # ...................... iter 2
        self.issue_global_load_k_scale(2, buf=0)  # ........................... iter 2

        self.async_wait(7)
        k0 = self.shared_load_k(sub_idx=0, buf=1)  # .......................... iter 1
        k0_scale = self.shared_load_k_scale(buf=1, slice=0)
        k1_scale = self.shared_load_k_scale(buf=1, slice=1)
        qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)  # ........ iter 0
        qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
        p0 = ttgl.exp2(qk0_shifted)
        self.issue_global_load_k(2, sub_idx=1, buf=0)  # ...................... iter 2

        end = ttgl.cdiv(cfg.SEQLEN_K // cfg.SPLIT_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 2
            b = 1 - a
            pred = i - end + 3
            pred = (pred >> 31) & 1

            with warp_pipeline_stage("compute0"):
                qk0 = self.compute_qk(q, q_scale, k0, k0_scale, zero)  # ...... iter i+1
                p1 = ttgl.exp2(qk1_shifted)  # ................................ iter i
                m_diff = m_i * sm_scale - m_ij_scaled
                m_i = m_ij
                alpha = ttgl.exp2(m_diff)
                acc0 = acc0 * expand_dims(alpha, -1)
                acc1 = acc1 * expand_dims(alpha, -1)

            self.async_wait(7)
            with warp_pipeline_stage("memory0"):
                k1 = self.shared_load_k(sub_idx=1, buf=b)  # .................. iter i+1
                self.issue_global_load_v(i + 1, sub_idx=0, buf=b)  # .......... iter i+1
                self.issue_global_load_v_scale(i + 1, buf=b)  # ............... iter i+1

            with warp_pipeline_stage("compute1"):
                qk1 = self.compute_qk(q, q_scale, k1, k1_scale, zero)  # ...... iter i+1
                p = self.concat_subtile(p0, p1)  # ............................ iter i
                l_ij = ttgl.sum(p, -1)
                l_i = l_i * alpha + l_ij
                p, p_scale = self.downcast_p(p)

            self.async_wait(7)
            with warp_pipeline_stage("memory1"):
                v0 = self.shared_load_v(sub_idx=0, buf=a)  # .................. iter i
                v0_scale = self.shared_load_v_scale(buf=a, slice=0)  # ........ iter i
                v1_scale = self.shared_load_v_scale(buf=a, slice=1)
                self.issue_global_load_v(i + 1, sub_idx=1, buf=b)  # .......... iter i+1

            with warp_pipeline_stage("compute2"):
                acc0 = self.compute_pv(p, p_scale, v0, v0_scale, acc0)  # ..... iter i
                qk = self.concat_subtile(qk0, qk1)  # ......................... iter i+1
                m = ttgl.max(qk, -1)
                m_ij = ttgl.maximum(m_i, m)
                m_ij_scaled = m_ij * sm_scale

            self.async_wait(7)
            with warp_pipeline_stage("memory2"):
                v1 = self.shared_load_v(sub_idx=1, buf=a)  # .................. iter i
                self.issue_global_load_k(i + 3, sub_idx=0, buf=b, pred=pred)  # iter i+3
                self.issue_global_load_k_scale(i + 3, buf=b, pred=pred)  # .... iter i+3

            with warp_pipeline_stage("compute3"):
                acc1 = self.compute_pv(p, p_scale, v1, v1_scale, acc1)  # ..... iter i
                qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)  # iter i+1
                qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
                p0 = ttgl.exp2(qk0_shifted)

            self.async_wait(7)
            with warp_pipeline_stage("memory3"):
                k0 = self.shared_load_k(sub_idx=0, buf=a)  # .................. iter i+2
                k0_scale = self.shared_load_k_scale(buf=a, slice=0)  # ........ iter i+2
                k1_scale = self.shared_load_k_scale(buf=a, slice=1)
                self.issue_global_load_k(i + 3, sub_idx=1, buf=b, pred=pred)  # iter i+3

        # pipeline epilogue iter end-2
        self.issue_global_load_v(end - 1, sub_idx=0, buf=1)
        self.issue_global_load_v(end - 1, sub_idx=1, buf=1)
        self.issue_global_load_v_scale(end - 1, buf=1)

        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * expand_dims(alpha, -1)
        acc1 = acc1 * expand_dims(alpha, -1)

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, -1)
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(5)
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
        m = ttgl.max(qk, -1)
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale

        qk0_shifted = qk0 * sm_scale - expand_dims(m_ij_scaled, -1)
        qk1_shifted = qk1 * sm_scale - expand_dims(m_ij_scaled, -1)
        p0 = ttgl.exp2(qk0_shifted)

        p1 = ttgl.exp2(qk1_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        m_i = m_ij
        alpha = ttgl.exp2(m_diff)
        acc0 = acc0 * expand_dims(alpha, -1)
        acc1 = acc1 * expand_dims(alpha, -1)

        p = self.concat_subtile(p0, p1)
        l_ij = ttgl.sum(p, -1)
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(0)
        v0 = self.shared_load_v(sub_idx=0, buf=1)
        v1 = self.shared_load_v(sub_idx=1, buf=1)
        v0_scale = self.shared_load_v_scale(buf=1, slice=0)
        v1_scale = self.shared_load_v_scale(buf=1, slice=1)

        acc0 = self.compute_pv(p, p_scale, v0, v0_scale, acc0)
        acc1 = self.compute_pv(p, p_scale, v1, v1_scale, acc1)

        acc = self.concat_subtile(acc0, acc1)
        return acc, l_i, m_i

    @gluon.jit
    def fwd_pipeline_triplebuf(self):
        cfg = self.cfg

        m_i, l_i, zero, acc = self.create_acc()
        sm_scale = self.sm_scale

        q = self.global_load_q()
        q_scale = self.global_load_q_scale()

        # pipeline prologue, iter -4
        self.issue_global_load_k(0, buf=0)  # ................................. iter 0
        self.issue_global_load_k_scale(0, buf=0)  # ........................... iter 0

        # pipeline prologue, iter -3
        self.issue_global_load_k(1, buf=1)  # ................................. iter 1
        self.issue_global_load_k_scale(1, buf=1)  # ........................... iter 1

        # pipeline prologue, iter -2
        self.issue_global_load_v(0, buf=0)  # ................................. iter 0
        self.issue_global_load_v_scale(0, buf=0)  # ........................... iter 0

        self.async_wait(4)
        self.issue_global_load_k(2, buf=2)  # ................................. iter 2
        self.issue_global_load_k_scale(2, buf=2)  # ........................... iter 2
        k = self.shared_load_k(buf=0)  # ...................................... iter 0
        k_scale = self.shared_load_k_scale(buf=0)

        # pipeline prologue, iter -1
        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter 0

        self.issue_global_load_v(1, buf=1)  # ................................. iter 1
        self.issue_global_load_v_scale(1, buf=1)  # ........................... iter 1

        m = ttgl.max(qk, -1)  # ............................................... iter 0
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        self.async_wait(6)
        self.issue_global_load_k(3, buf=0)  # ................................. iter 3
        self.issue_global_load_k_scale(3, buf=0)  # ........................... iter 3
        k = self.shared_load_k(buf=1)  # ...................................... iter 1
        k_scale = self.shared_load_k_scale(buf=1)

        # main loop from 0 to end-3
        end = ttgl.cdiv(cfg.SEQLEN_K // cfg.SPLIT_K, cfg.BLOCK_N)
        for i in range(0, end - 2):
            a = i % 3
            b = (i + 1) % 3
            c = (i + 2) % 3
            pred = i - end + 4
            pred = (pred >> 31) & 1

            qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ............. iter i+1
            l_ij = ttgl.sum(p, -1)  # ......................................... iter i
            acc = acc * expand_dims(alpha, -1)
            l_i = l_i * alpha + l_ij
            p, p_scale = self.downcast_p(p)

            self.async_wait(6)
            self.issue_global_load_v(i + 2, buf=c)  # ......................... iter i+2
            self.issue_global_load_v_scale(i + 2, buf=c)  # ................... iter i+2
            v = self.shared_load_v(buf=a)  # .................................. iter i
            v_scale = self.shared_load_v_scale(buf=a)

            acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ............. iter i
            m = ttgl.max(qk, -1)  # ........................................... iter i+1
            m_ij = ttgl.maximum(m_i, m)
            m_ij_scaled = m_ij * sm_scale
            qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
            p = ttgl.exp2(qk_shifted)
            m_diff = m_i * sm_scale - m_ij_scaled
            alpha = ttgl.exp2(m_diff)
            m_i = m_ij

            self.async_wait(6)
            self.issue_global_load_k(i + 4, buf=b, pred=pred)  # .............. iter i+4
            self.issue_global_load_k_scale(i + 4, buf=b, pred=pred)  # ........ iter i+4
            k = self.shared_load_k(buf=c)  # .................................. iter i+2
            k_scale = self.shared_load_k_scale(buf=c)

        # pipeline epilogue, iter end-2
        a = (end - 2) % 3

        qk = self.compute_qk(q, q_scale, k, k_scale, zero)  # ................. iter end-1
        l_ij = ttgl.sum(p, -1)  # ............................................. iter end-2
        acc = acc * expand_dims(alpha, -1)
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(2)
        v = self.shared_load_v(buf=a)  # ...................................... iter end-2
        v_scale = self.shared_load_v_scale(buf=a)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-2
        m = ttgl.max(qk, -1)  # ............................................... iter end-1
        m_ij = ttgl.maximum(m_i, m)
        m_ij_scaled = m_ij * sm_scale
        qk_shifted = qk * sm_scale - expand_dims(m_ij_scaled, -1)
        p = ttgl.exp2(qk_shifted)
        m_diff = m_i * sm_scale - m_ij_scaled
        alpha = ttgl.exp2(m_diff)
        m_i = m_ij

        # pipeline epilogue, iter end-1
        a = (end - 1) % 3

        l_ij = ttgl.sum(p, -1)  # ............................................. iter end-1
        acc = acc * expand_dims(alpha, -1)
        l_i = l_i * alpha + l_ij
        p, p_scale = self.downcast_p(p)

        self.async_wait(0)
        v = self.shared_load_v(buf=a)  # ...................................... iter end-1
        v_scale = self.shared_load_v_scale(buf=a)

        acc = self.compute_pv(p, p_scale, v, v_scale, acc)  # ................. iter end-1

        return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit
def store_output(  #
        o_ptr,  #
        acc, l_i, m_i,  #
        sm_scale,  #
        cfg: ttgl.constexpr):
    SEQLEN_K: ttgl.constexpr = cfg.SEQLEN_K
    SEQLEN_Q: ttgl.constexpr = cfg.SEQLEN_Q
    HEAD_SZ: ttgl.constexpr = cfg.HEAD_SZ
    NUM_Q_HEADS: ttgl.constexpr = cfg.NUM_Q_HEADS
    NUM_K_HEADS: ttgl.constexpr = cfg.NUM_K_HEADS
    BLOCK_M: ttgl.constexpr = cfg.BLOCK_M
    SPLIT_K: ttgl.constexpr = cfg.SPLIT_K
    NUM_WARPS: ttgl.constexpr = cfg.NUM_WARPS

    off_h = ttgl.program_id(0)
    off_m = ttgl.program_id(1)
    off_z = ttgl.program_id(2)

    if SEQLEN_Q == SEQLEN_K:
        ttgl.static_assert(SPLIT_K == 1)

        l_recip = 1 / l_i
        acc = acc * expand_dims(l_recip, -1)

        o_base = SEQLEN_Q * HEAD_SZ * (NUM_Q_HEADS * off_z + off_h)
        o_shape = [SEQLEN_Q, HEAD_SZ]

    else:
        GROUP_SZ: ttgl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
        NUM_GROUPS: ttgl.constexpr = NUM_K_HEADS

        if SPLIT_K == 1:
            l_recip = 1 / l_i
            acc = acc * expand_dims(l_recip, -1)

        else:
            m_ij = ttgl.max(m_i, 0)
            m_ij_scaled = m_ij * sm_scale
            m_diff = m_i * sm_scale - expand_dims(m_ij_scaled, 0)
            alpha = ttgl.exp2(m_diff)

            shape: ttgl.constexpr = [SPLIT_K * BLOCK_M, HEAD_SZ]
            acc = acc * expand_dims(alpha, -1)
            acc = acc.reshape(shape)

            acc_smem_layout: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[HEAD_SZ, 4]], shape, [1, 0])
            acc_smem = ttgl.allocate_shared_memory(acc.dtype, shape, acc_smem_layout)
            acc_smem.store(acc)

            acc_layout: ttgl.constexpr = ttgl.BlockedLayout([1, HEAD_SZ // NUM_WARPS // 2], [16, 2], [1, NUM_WARPS],
                                                            [1, 0])
            acc = ttgl.zeros([BLOCK_M, HEAD_SZ], acc.dtype, acc_layout)
            for i in ttgl.static_range(SPLIT_K):
                acc += acc_smem.slice(i * BLOCK_M, BLOCK_M).load(acc_layout)

            l_i = ttgl.sum(l_i * alpha, 0)
            l_recip = 1 / l_i
            l_recip = expand_dims(l_recip, 0)
            l_recip = ttgl.permute(l_recip, [1, 0])
            l_recip = ttgl.convert_layout(l_recip, acc.type.layout)
            acc = acc * l_recip

        o_base = GROUP_SZ * HEAD_SZ * (NUM_GROUPS * off_z + off_h)
        o_shape = [GROUP_SZ, HEAD_SZ]

    o_smem_layout: ttgl.constexpr = get_shared_layout([BLOCK_M, HEAD_SZ], padding=True, clamp=True)
    o_desc = tdm.make_tensor_descriptor(  #
        base=o_ptr + o_base,  #
        shape=o_shape,  #
        strides=[HEAD_SZ, 1],  #
        block_shape=[BLOCK_M, HEAD_SZ],  #
        layout=o_smem_layout)

    o = acc.to(o_ptr.dtype.element_ty)
    o_smem = ttgl.allocate_shared_memory(o_ptr.dtype.element_ty, [BLOCK_M, HEAD_SZ], o_smem_layout)
    o_smem.store(o)
    tdm.async_store(o_desc, [off_m * BLOCK_M, 0], o_smem)


@gluon.jit
def mxfp_attn_fwd_kernel(  #
        q_ptr, k_ptr, v_ptr,  #
        q_scale_ptr, k_scale_ptr, v_scale_ptr,  #
        o_ptr,  #
        sm_scale,  #
        cfg: ttgl.constexpr):

    # Select the target program
    BLOCK_SCALING: ttgl.constexpr = isinstance(cfg, BlockScaledAttentionConfig)
    kv_mem = KVMemory.initialize(k_ptr, v_ptr, cfg)
    if not BLOCK_SCALING:
        pgm = GlobalScaledAttentionProgram.initialize(  #
            cfg, q_ptr, q_scale_ptr, kv_mem, k_scale_ptr, v_scale_ptr, sm_scale)
    else:
        kv_scale_mem = KVScaleMemory.initialize(k_scale_ptr, v_scale_ptr, cfg)
        pgm = BlockScaledAttentionProgram.initialize(  #
            cfg, q_ptr, q_scale_ptr, kv_mem, kv_scale_mem, sm_scale)

    # Select the target schedule
    if cfg.NUM_BUFFERS == 1:
        acc, l_i, m_i = pgm.fwd_loop()
    elif cfg.NUM_BUFFERS == 2:
        if cfg.SUBTILE:
            if cfg.PINGPONG:
                acc, l_i, m_i = pgm.fwd_pipeline_subtile_pingpong()
            else:
                acc, l_i, m_i = pgm.fwd_pipeline_subtile()
        else:
            acc, l_i, m_i = pgm.fwd_pipeline()
    elif cfg.NUM_BUFFERS == 3:
        ttgl.static_assert(not cfg.SUBTILE)
        acc, l_i, m_i = pgm.fwd_pipeline_triplebuf()

    store_output(o_ptr, acc, l_i, m_i, sm_scale, cfg)


def get_attn_schedule(cfg):
    if isinstance(cfg, BlockScaledAttentionConfig):
        pgm = BlockScaledAttentionProgram
    else:
        pgm = GlobalScaledAttentionProgram

    if cfg.NUM_BUFFERS == 1:
        return pgm.fwd_loop
    elif cfg.NUM_BUFFERS == 2:
        if cfg.SUBTILE:
            if cfg.PINGPONG:
                return pgm.fwd_pipeline_subtile_pingpong
            else:
                return pgm.fwd_pipeline_subtile
        else:
            return pgm.fwd_pipeline
    elif cfg.NUM_BUFFERS == 3:
        assert not cfg.SUBTILE
        return pgm.fwd_pipeline_triplebuf


def attn_fwd(  #
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  #
        q_scale: torch.Tensor | int, k_scale: torch.Tensor | int, v_scale: torch.Tensor | int,  #
        q_type: str, kv_type: str, block_scaling: bool, p_scaling: bool,  #
        block_m: int, block_n: int, split_k: int, pipelined: bool, num_warps: int):

    batch, seqlen_q, num_q_heads, head_sz = q.shape
    _, seqlen_k, num_k_heads, _ = k.shape
    out_dtype = torch.float32
    assert seqlen_q == 1 or seqlen_q == seqlen_k
    assert num_q_heads >= num_k_heads and num_q_heads % num_k_heads == 0
    assert head_sz in {64, 128}
    assert block_n >= 128
    assert block_m >= 16
    assert seqlen_k % block_n == 0
    if split_k > 1:
        # Split k partitions along multiple warps
        assert split_k == num_warps
    assert (not pipelined) or cdiv(seqlen_k, block_n) > 4
    kv_pack_div = 2 if kv_type == 'e2m1' else 1

    # When we have a large block_m for pipeline, we will subtile K/V to
    # save registers
    subtile = pipelined and block_m >= 256
    # We can use pingpong schedule where there are 8 or more warps
    pingpong = pipelined and num_warps >= 8
    # Decide the number of buffers for pipeline
    num_buffers = 1
    if pipelined:
        num_buffers = 2
        # When block_m is small, the kernel becomes memory bound, and we need
        # to increase the number of outstanding memory requests to improve
        # memory utilization. This can be achieved by using triple buffering
        # where one additional buffer can be used for an immediate memory
        # request, without waiting for the data in the buffer to be consumed.
        if block_m <= 64:
            num_buffers = 3
        group_sz = num_q_heads // num_k_heads
        # For MHA decode, we are using a single wave per workgroup, and needs
        # multiple workgroups to be scheduled on the same processor to achieve
        # good occupancy. However, occupancy can be limited by the LDS size -
        # which can roughly be computed by
        # ```
        # 2 * BLOCK_N * HEAD_SZ * NUM_BUFFERS / KV_PACK_DIV
        # ```
        # We only have 320KB per processor. For mxfp8 and head_sz=128,
        # 1 workgroup needs 2 * 128 * 128 * 3 = 96KB, and 4 workgroups
        # can use 384KB which exceeds the limit. So we will fallback to
        # double buffering in this case.
        if seqlen_q == 1 and group_sz == 1 and num_warps == 1:
            if kv_type != 'e2m1' and head_sz == 128:
                num_buffers = 2
        # For MQA decode with split-k, we will increase the LDS usage for
        # k partitions, which can also exceed the LDS limit for mxfp8 with
        # head_sz=128.
        if seqlen_q == 1 and split_k > 1:
            if kv_type != 'e2m1' and head_sz == 128:
                num_buffers = 2
    # When kv_type is mxfp8 (e4m3 or e5m2), we can use p_k_width of 8,
    # which makes QK and P share the same layout.
    p_k_width = 16 if kv_type == 'e2m1' else 8

    if block_scaling:
        cfg = BlockScaledAttentionConfig(  #
            q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, p_scaling,  #
            block_m, block_n, split_k, subtile, pingpong, p_k_width, num_buffers, num_warps)
    else:
        cfg = GlobalScaledAttentionConfig(  #
            q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz,  #
            block_m, block_n, split_k, subtile, pingpong, p_k_width, num_buffers, num_warps)

    if seqlen_q == seqlen_k:
        assert split_k == 1
        # q: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ]
        # k: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
        # v: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
        # o: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ]
        q = q.permute(0, 2, 1, 3).contiguous()
        k = KVMemory.preshuffle(k.permute(0, 2, 1, 3),  #
                                block_shape=[block_n, head_sz // kv_pack_div],  #
                                sub_axis=0 if subtile else None)
        v = KVMemory.preshuffle(v.permute(0, 2, 1, 3),  #
                                block_shape=[block_n // kv_pack_div, head_sz],  #
                                sub_axis=1 if subtile else None)
        o = torch.zeros_like(q, dtype=out_dtype)

        # q_scale: [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ / 32]
        # k_scale: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ / 32]
        # v_scale: [BATCH, NUM_K_HEADS, HEAD_SZ, SEQLEN_K / 32]
        if block_scaling:
            q_scale = q_scale.permute(0, 2, 1, 3).contiguous()
            k_scale = KVScaleMemory.preshuffle(k_scale.permute(0, 2, 1, 3))
            v_scale = KVScaleMemory.preshuffle(v_scale.permute(0, 2, 3, 1))

        grid = (num_q_heads, cdiv(seqlen_q, block_m), batch)

    else:
        group_sz = num_q_heads // num_k_heads
        num_groups = num_k_heads
        # q: [BATCH, NUM_GROUPS, GROUP_SZ, HEAD_SZ]
        # k: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
        # v: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
        # o: [BATCH, NUM_GROUPS, GROUP_SZ, HEAD_SZ]
        q = q.permute(0, 2, 1, 3).view(batch, num_groups, group_sz, head_sz).contiguous()
        k = KVMemory.preshuffle(k.permute(0, 2, 1, 3),  #
                                block_shape=[block_n, head_sz // kv_pack_div],  #
                                sub_axis=0 if subtile else None)
        v = KVMemory.preshuffle(v.permute(0, 2, 1, 3),  #
                                block_shape=[block_n // kv_pack_div, head_sz],  #
                                sub_axis=1 if subtile else None)
        o = torch.zeros_like(q, dtype=out_dtype)

        # q_scale: [BATCH, NUM_GROUPS, GROUP_SZ, HEAD_SZ / 32]
        # k_scale: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ / 32]
        # v_scale: [BATCH, NUM_K_HEADS, HEAD_SZ, SEQLEN_K / 32]
        if block_scaling:
            q_scale = q_scale.permute(0, 2, 1, 3).view(batch, num_groups, group_sz, head_sz // 32).contiguous()
            k_scale = KVScaleMemory.preshuffle(k_scale.permute(0, 2, 1, 3))
            v_scale = KVScaleMemory.preshuffle(v_scale.permute(0, 2, 3, 1))

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
        out = out.reshape(batch, num_q_heads, seqlen_q, head_sz).permute(0, 2, 1, 3)

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


def get_fwd_test_cases(block_scaling: bool):
    dtypes = [("e4m3", "e4m3"), ("e4m3", "e2m1")] if block_scaling else [("e4m3", "e4m3")]
    tests = [[q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz]
             for q_type, kv_type in dtypes
             for batch in [1]
             for seqlen_q, seqlen_k, num_q_heads, num_k_heads in [
                 (1024, 1024, 1, 1),
                 (1024, 1024, 4, 1),
                 (1024, 1024, 4, 2),
                 (1, 1024, 1, 1),
                 (1, 8192, 64, 1),
                 (1, 8192, 64, 2),
             ]
             for head_sz in [64, 128]]

    # block_m,block_n,split_k,pipelined,num_warps
    configs = {
        "4warp_128x128_loop": [128, 128, 1, False, 4],
        "4warp_128x128_pipeline": [128, 128, 1, True, 4],
        "4warp_256x128_pipeline": [256, 128, 1, True, 4],
        "8warp_256x128_pipeline": [256, 128, 1, True, 8],
        "1warp_16x128_loop": [16, 128, 1, False, 1],
        "1warp_16x128_pipeline": [16, 128, 1, True, 1],
        "4warp_16x128_loop_split4": [16, 128, 4, False, 4],
        "4warp_16x128_pipeline_split4": [16, 128, 4, True, 4],
    }

    param = []
    for test in tests:
        seqlen_q, seqlen_k, num_q_heads, num_k_heads = test[3:7]
        if seqlen_q == seqlen_k:
            # MHA/GQA Prefill
            param.append((*test, *configs["4warp_128x128_loop"]))
            param.append((*test, *configs["4warp_128x128_pipeline"]))
            param.append((*test, *configs["4warp_256x128_pipeline"]))
            param.append((*test, *configs["8warp_256x128_pipeline"]))
        else:
            assert seqlen_q == 1
            if num_q_heads == num_k_heads:
                # MHA Decode
                param.append((*test, *configs["1warp_16x128_loop"]))
                param.append((*test, *configs["1warp_16x128_pipeline"]))
            else:
                # MQA Decode
                param.append((*test, *configs["4warp_16x128_loop_split4"]))
                param.append((*test, *configs["4warp_16x128_pipeline_split4"]))
    return param


@pytest.mark.parametrize(
    "q_type,kv_type,batch,seqlen_q,seqlen_k,num_q_heads,num_k_heads,head_sz,"
    "block_m,block_n,split_k,pipelined,num_warps",  #
    get_fwd_test_cases(True))
def test_block_scaled_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz,  #
                               block_m, block_n, split_k, pipelined, num_warps):
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
        block_m, block_n, split_k, pipelined, num_warps)
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
            # TODO: For some cases, we can have generated code using 2 different source vgprs for address.
            assert len(set(sources)) <= 2
        # check use v_permlane16_swap for convert layout
        if re.match(groups['convert_layout'], code):
            v_permlane_instrs = [instr for instr in instrs if re.match(r'v_permlane_*', instr)]
            assert len(v_permlane_instrs) > 0 and all(
                instr.startswith("v_permlane16_swap") for instr in v_permlane_instrs)
        # check there is no v_readfirstlane
        assert all(not re.match(r'v_readfirstlane', instr) for instr in instrs)


@pytest.mark.parametrize(
    "q_type,kv_type,batch,seqlen_q,seqlen_k,num_q_heads,num_k_heads,head_sz,"
    "block_m,block_n,split_k,pipelined,num_warps",  #
    get_fwd_test_cases(False))
def test_global_scaled_attn_fwd(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz,  #
                                block_m, block_n, split_k, pipelined, num_warps):
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
        block_m, block_n, split_k, pipelined, num_warps)
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
            # TODO: For some cases, we can have generated code using 2 different source vgprs for address.
            assert len(set(sources)) <= 2
        # check use v_permlane16_swap for convert layout
        if re.match(groups['convert_layout'], code):
            v_permlane_instrs = [instr for instr in instrs if re.match(r'v_permlane_*', instr)]
            assert len(v_permlane_instrs) > 0 and all(
                instr.startswith("v_permlane16_swap") for instr in v_permlane_instrs)
        # check there is no v_readfirstlane
        assert all(not re.match(r'v_readfirstlane', instr) for instr in instrs)


def run_attention(q_type, kv_type, batch, seqlen_q, seqlen_k, num_q_heads, num_k_heads, head_sz, scale_type,
                  disable_p_scaling, block_m, block_n, split_k, pipelined, num_warps):
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
        block_m, block_n, split_k, pipelined, num_warps)
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
    parser.add_argument("--split_k", type=int, default=1)
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
