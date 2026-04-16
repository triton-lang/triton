"""
2CTA Block-Scaled Matrix Multiplication
=======================================

High-performance 2CTA warp-specialized block-scaled MMA.
Two CTAs cooperate per output tile, sharing operands to
increase arithmetic intensity and reduce the per-CTA SMEM
footprint.
"""

import argparse
import itertools
import pytest
import torch

import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton.language.core import _aggregate as aggregate

from triton._C.libtriton import nvidia

from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    clc,
    tcgen05_copy,
    tcgen05_commit,
    tcgen05_mma_scaled,
    mbarrier,
    tma,
)

# ---------------------------------------------------------------------------
# Tile scheduler
# ---------------------------------------------------------------------------


@gluon.jit
def _planar_snake(lin_idx, m_tiles, n_tiles, minor_dim: gl.constexpr, tile_width: gl.constexpr):
    major_size = n_tiles if minor_dim == 0 else m_tiles
    minor_size = m_tiles if minor_dim == 0 else n_tiles

    full_minor_tiles = minor_size // tile_width
    full_minor_size = full_minor_tiles * tile_width
    full_elements = full_minor_tiles * tile_width * major_size

    minor_tile_idx = lin_idx // (tile_width * major_size)

    full_minor_within = lin_idx % tile_width
    full_major_within = (lin_idx // tile_width) % major_size
    full_minor = minor_tile_idx * tile_width + full_minor_within
    full_major = gl.where((minor_tile_idx % 2) == 0, full_major_within, major_size - 1 - full_major_within)

    partial_width = minor_size - full_minor_size
    partial_width = gl.where(partial_width > 0, partial_width, 1)
    partial_lin = lin_idx - full_elements
    partial_minor_within = partial_lin % partial_width
    partial_major_within = (partial_lin // partial_width) % major_size
    partial_minor = minor_tile_idx * tile_width + partial_minor_within
    partial_major = gl.where((minor_tile_idx % 2) == 0, partial_major_within, major_size - 1 - partial_major_within)

    in_full_tile = lin_idx < full_elements
    minor = gl.where(in_full_tile, full_minor, partial_minor)
    major = gl.where(in_full_tile, full_major, partial_major)

    if minor_dim == 0:
        return minor, major
    return major, minor


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


@gluon.constexpr_function
def get_split_dim(cga_layout, dim):
    return 1 << sum(b[dim] != 0 for b in cga_layout)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def random_quantized_tensor(MN, K, format):
    assert format in ["mxfp4", "mxfp8", "nvfp4"]
    VEC_SIZE = 16 if format == "nvfp4" else 32

    # Generate a random quantized tensor and its scale factors, assuming we are
    # scaling along the K dimension.
    base = MXFP4Tensor(size=(MN, K), device="cuda").random()
    scale = MXScaleTensor(size=(MN, K // VEC_SIZE), device="cuda").random(low=1 / 128, high=2.0)

    # Compute the dequantized tensor to use for testing.
    ref = base.to(torch.float32)
    scale_ref = scale.to(torch.float32)
    value = ref * scale_ref.repeat_interleave(VEC_SIZE, dim=1)

    if format == "mxfp8":
        # For mxfp8, convert the tensor to a regular float8 torch tensor.
        return ref.to(torch.float8_e4m3fn), scale.data, value
    elif format == "mxfp4":
        # For mxfp4, pack the elements along the K dimension.
        return base.to_packed_tensor(dim=1), scale.data, value
    else:
        # For nvfp4, pack the elements along the K dimension, and convert the
        # scale factors to float8_e4m3fn.
        return base.to_packed_tensor(dim=1), scale_ref.to(torch.float8_e4m3fn), value


def align_to(a, b):
    # Return next multiple of `b` greater than or equal to `a`.
    return triton.cdiv(a, b) * b


def swizzle_scales_packed_block(scales: torch.Tensor):
    # When the scale tensor is not an even multiple of [128, 4], we need to pad
    # the scale tensor so it can use the packed block format.
    PAD_MN = align_to(scales.shape[0], 128) - scales.shape[0]
    PAD_K = align_to(scales.shape[1], 4) - scales.shape[1]
    scales = torch.nn.functional.pad(scales, (0, PAD_K, 0, PAD_MN))

    MN, SCALE_K = scales.shape[0], scales.shape[1]
    REP_MN = MN // 128
    REP_K = SCALE_K // 4
    scales = scales.reshape(REP_MN, 4, 32, REP_K, 4)
    scales = scales.permute(0, 3, 2, 1, 4)
    return scales.contiguous()


# ---------------------------------------------------------------------------
# Autotuning configs and hook
# ---------------------------------------------------------------------------


def mma_scaled_get_configs(pre_hook=None, cga_layouts=None):
    if cga_layouts is None:
        cga_layouts = [(), ((1, 0), )]
    return [
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "EPILOGUE_BLOCK_N": epilogue_n,
                "num_buffers": stages,
                "num_acc_buffers": acc_buffers,
                "GRID_MINOR_DIM": minor_dim,
                "GRID_TILE_WIDTH": grid_tile_width,
                "CGA_LAYOUT": cga_layout,
            },
            num_warps=4,
            num_ctas=2**len(cga_layout),
            pre_hook=pre_hook,
        )
        for BM in (128, 256)
        for BN in (128, 256)
        for BK in (128, 256)
        for epilogue_n in (64, BN)
        for minor_dim in (0, 1)
        for grid_tile_width in (4, 8, 16)
        for stages in (3, 4, 5)
        for acc_buffers in (1, 2)
        for cga_layout in cga_layouts
        # tcgen05_mma_scaled requires BLOCK_M_PER_CTA == 128
        if BM // (2**len(cga_layout)) == 128 if epilogue_n <= BN
    ]


def mma_scaled_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    epilogue_n = nargs["EPILOGUE_BLOCK_N"]
    cga_layout = nargs["CGA_LAYOUT"]

    a_base = nargs["a_desc"].base
    b_base = nargs["b_desc"].base
    a_is_fp4 = a_base.dtype == torch.uint8
    b_is_fp4 = b_base.dtype == torch.uint8
    mixed_prec = a_is_fp4 != b_is_fp4
    a_elem_per_byte = 2 if a_is_fp4 else 1
    b_elem_per_byte = 2 if b_is_fp4 else 1

    a_block = [block_m, block_k // a_elem_per_byte]
    b_block = [block_n, block_k // b_elem_per_byte]
    c_block = [block_m, epilogue_n]

    nargs["a_desc"].block_shape = a_block
    nargs["b_desc"].block_shape = b_block
    nargs["c_desc"].block_shape = c_block

    cga = tuple(tuple(x) for x in cga_layout) if cga_layout else None
    nargs["a_desc"].layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.uint8 if a_is_fp4 else gl.float8e4nv,
                                                                  fp4_padded=(mixed_prec and a_is_fp4), cga_layout=cga)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(b_block, gl.uint8 if b_is_fp4 else gl.float8e4nv,
                                                                  fp4_padded=(mixed_prec and b_is_fp4), cga_layout=cga)
    c_dtype = getattr(gl, str(nargs["c_desc"].base.dtype).split('.')[1])
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(c_block, c_dtype, cga_layout=cga)

    a_scale_base = nargs["a_scale_desc"].base
    is_nvfp4 = a_scale_base.dtype == torch.float8_e4m3fn
    vec_size = 16 if is_nvfp4 else 32
    rep_m = block_m // 128
    rep_n = block_n // 128
    rep_k = block_k // (vec_size * 4)
    nargs["a_scale_desc"].block_shape = [1, rep_m, rep_k, 2, 256]
    nargs["b_scale_desc"].block_shape = [1, rep_n, rep_k, 2, 256]

    if cga_layout:
        cga_a_scale = [[0, 1, 0, 0, 0]]
        cga_b_scale = [[0, 0, 0, 0, 0]]
        nargs["a_scale_desc"].layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5,
                                                            cga_layout=cga_a_scale)
        nargs["b_scale_desc"].layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5,
                                                            cga_layout=cga_b_scale)
    else:
        no_swizzle = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
        nargs["a_scale_desc"].layout = no_swizzle
        nargs["b_scale_desc"].layout = no_swizzle


@gluon.jit
def unswizzle_scales_shared_memory(smem, BLOCK_MN: gl.constexpr, BLOCK_K: gl.constexpr, VEC_SIZE: gl.constexpr):
    smem = smem.reshape((smem.shape[1], smem.shape[2], 32, 4, 4))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((BLOCK_MN, BLOCK_K // VEC_SIZE))


@gluon.jit
def async_mma_scaled_impl(a_smem, b_smem, a_scale_smem, b_scale_smem, acc_tmem, use_acc, pred):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_smem.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = a_smem.shape[0]
    BLOCK_N: gl.constexpr = b_smem.shape[0]
    BLOCK_K: gl.constexpr = a_smem.shape[1] * A_ELEM_PER_BYTE
    # Recall we use `uint8` to represent fp4 elements.
    VEC_SIZE: gl.constexpr = 32 if a_scale_smem.dtype == gl.uint8 else 16

    a_scale = unswizzle_scales_shared_memory(a_scale_smem, BLOCK_M, BLOCK_K, VEC_SIZE)
    b_scale = unswizzle_scales_shared_memory(b_scale_smem, BLOCK_N, BLOCK_K, VEC_SIZE)

    # We don't need to hoist the scales tensor memory allocations outside of the loop,
    # so we can pull them into this helper function.
    two_ctas: gl.constexpr = acc_tmem.type.layout.two_ctas
    a_scale_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=[[1, 0]] if two_ctas else [])
    b_scale_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=[[0, 0]] if two_ctas else [])
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype, a_scale.type.shape, a_scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype, b_scale.type.shape, b_scale_layout)
    tcgen05_copy(a_scale, a_scale_tmem)
    tcgen05_copy(b_scale, b_scale_tmem)

    a_format: gl.constexpr = "e2m1" if a_smem.dtype == gl.uint8 else "e4m3"
    b_format: gl.constexpr = "e2m1" if b_smem.dtype == gl.uint8 else "e4m3"
    tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                       use_acc=use_acc, pred=pred)


# This helper function computes all the load indexing and issues the async loads
# based on the current `pid_m`, `pid_n`, and `k` indices. The compiler will run
# loop-invariant code motion to hoist code that does not depend on `k`, like
# `pid_m * BLOCK_M`, outside of the inner loop, so we can safely abstract the
# load indexing without performance loss.
#
# Encapsulating the load indexing logic will help keep our pipelined kernel code
# clean, as pipelining can get messy.
@gluon.jit
def issue_loads(producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs,
                b_scale_bufs, bars, pred, multicast_b_scale: gl.constexpr = False):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if b_desc.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = a_desc.block_shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_shape[0]
    BLOCK_K: gl.constexpr = a_desc.block_shape[1] * A_ELEM_PER_BYTE
    REP_M: gl.constexpr = a_scale_desc.block_shape[1]
    REP_N: gl.constexpr = b_scale_desc.block_shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_shape[2]
    B_REP_K: gl.constexpr = b_scale_desc.block_shape[2]

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N
    off_k_a = k // A_ELEM_PER_BYTE
    off_k_b = k // B_ELEM_PER_BYTE
    off_k_a_scale = (k // BLOCK_K) * A_REP_K
    off_k_b_scale = (k // BLOCK_K) * B_REP_K

    index = producer.index
    bar = bars.index(index)
    mbarrier.expect(
        bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta + a_scale_desc.nbytes_per_cta + b_scale_desc.nbytes_per_cta,
        pred)
    tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_bufs.index(index), pred)
    tma.async_copy_global_to_shared(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], bar,
                                    a_scale_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], bar,
                                    b_scale_bufs.index(index), pred, multicast=multicast_b_scale)
    return producer.next(pred)


@gluon.jit
def issue_mma(consumer, c_bars, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs, producer, p_bars, acc_tmem, use_acc, pred):
    c_index = consumer.index
    mbarrier.wait(c_bars.index(c_index), consumer.phase, pred)
    async_mma_scaled_impl(a_bufs.index(c_index), b_bufs.index(c_index), a_scale_bufs.index(c_index),
                          b_scale_bufs.index(c_index), acc_tmem, use_acc, pred)
    tcgen05_commit(p_bars.index(producer.index), pred)
    return consumer.next(pred), producer.next(pred)


@aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)


@aggregate
class ClcTileSchedulerConsumer:
    has_work: gl.tensor
    tile_id: gl.tensor
    pid_m: gl.tensor
    pid_n: gl.tensor
    num_pid_m: gl.tensor
    num_pid_n: gl.tensor
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_planar_pid_buffers: gl.shared_memory_descriptor
    clc_planar_ready_bars: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor
    counter: Counter
    consumed_counter: Counter

    @gluon.jit
    def initialize(M, N, TILE_M: gl.constexpr, TILE_N: gl.constexpr, MINOR_DIM: gl.constexpr,
                   GRID_TILE_WIDTH: gl.constexpr, clc_result_buffers, clc_barriers, clc_planar_pid_buffers,
                   clc_planar_ready_bars, clc_consumed_bars):
        tile_id = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(M, TILE_M)
        num_pid_n = gl.cdiv(N, TILE_N)
        pid_m, pid_n = _planar_snake(tile_id, num_pid_m, num_pid_n, MINOR_DIM, GRID_TILE_WIDTH)
        has_work = gl.to_tensor(True)
        counter = Counter.create(0, clc_barriers.shape[0])
        consumed_counter = Counter.create(0, clc_barriers.shape[0])
        return ClcTileSchedulerConsumer(
            has_work,
            tile_id,
            pid_m,
            pid_n,
            num_pid_m,
            num_pid_n,
            TILE_M,
            TILE_N,
            MINOR_DIM,
            GRID_TILE_WIDTH,
            clc_result_buffers,
            clc_barriers,
            clc_planar_pid_buffers,
            clc_planar_ready_bars,
            clc_consumed_bars,
            counter,
            consumed_counter,
        )

    @gluon.jit
    def get_offsets(self):
        return self.pid_m * self.TILE_M, self.pid_n * self.TILE_N

    @gluon.jit
    def step(self, iteration):
        # The 0-th iteration uses the program_id as the tile_id.
        # At the end of each iteration we prefetch the next tile.
        # As such we must signal the consumed slot at the end of
        # each iteration skipping the first one.
        consumed_counter = self.consumed_counter
        if iteration > 0:
            mbarrier.arrive(self.clc_consumed_bars.index(consumed_counter.index))
            consumed_counter = consumed_counter.next()
        counter = self.counter
        barrier = self.clc_barriers.index(counter.index)
        result = self.clc_result_buffers.index(counter.index)
        mbarrier.wait(barrier, counter.phase)
        clc_res = clc.load_result(result)
        mbarrier.wait(self.clc_planar_ready_bars.index(counter.index), counter.phase)
        planar_slot = self.clc_planar_pid_buffers.index(counter.index)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        packed_pid = planar_slot.load(planar_layout).reshape([])
        pid_m = ((packed_pid >> 32) & 0xFFFFFFFF).to(gl.int32)
        pid_n = (packed_pid & 0xFFFFFFFF).to(gl.int32)
        has_work = clc_res.is_canceled()
        tile_id = self.tile_id
        if has_work:
            tile_id = clc_res.program_id(0)
        return ClcTileSchedulerConsumer(
            has_work,
            tile_id,
            pid_m,
            pid_n,
            self.num_pid_m,
            self.num_pid_n,
            self.TILE_M,
            self.TILE_N,
            self.MINOR_DIM,
            self.GRID_TILE_WIDTH,
            self.clc_result_buffers,
            self.clc_barriers,
            self.clc_planar_pid_buffers,
            self.clc_planar_ready_bars,
            self.clc_consumed_bars,
            counter.next(),
            consumed_counter,
        )


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------


@aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_scale_desc: tma.tensor_descriptor
    b_scale_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    a_scale_bufs: gl.shared_memory_descriptor
    b_scale_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_planar_pid_buffers: gl.shared_memory_descriptor
    clc_planar_ready_bars: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr

    @gluon.jit
    def get_clc_consumer(self):
        return ClcTileSchedulerConsumer.initialize(
            self.c_desc.shape[0],
            self.c_desc.shape[1],
            self.a_desc.block_shape[0],
            self.b_desc.block_shape[0],
            self.MINOR_DIM,
            self.GRID_TILE_WIDTH,
            self.clc_result_buffers,
            self.clc_barriers,
            self.clc_planar_pid_buffers,
            self.clc_planar_ready_bars,
            self.clc_consumed_bars,
        )


@gluon.jit
def mma_scaled_load_partition(p):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if p.a_desc.dtype == gl.uint8 else 1
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    K = p.a_desc.shape[1] * A_ELEM_PER_BYTE
    state = Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = p.get_clc_consumer()
    i = 0
    while scheduler.has_work:
        for k in range(0, K, BLOCK_K):
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            state = issue_loads(state, scheduler.pid_m, scheduler.pid_n, k, p.a_desc, p.b_desc, p.a_scale_desc,
                                p.b_scale_desc, p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs, p.load_ready_bars,
                                pred=True, multicast_b_scale=gl.num_ctas() > 1)
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def mma_scaled_mma_partition(p):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if p.a_desc.dtype == gl.uint8 else 1
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    K = p.a_desc.shape[1] * A_ELEM_PER_BYTE
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.get_clc_consumer()
    i = 0
    while scheduler.has_work:
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for k in range(0, K, BLOCK_K):
            _, load_state = issue_mma(load_state, p.load_ready_bars, p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs,
                                      load_state, p.load_empty_bars, acc_buf, use_acc, pred=True)
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def mma_scaled_epilogue_partition(p):
    tile_m: gl.constexpr = p.c_desc.block_shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_shape[0]
    EPILOGUE_BLOCK_N: gl.constexpr = p.c_desc.block_shape[1]
    subtile_factor: gl.constexpr = BLOCK_N // EPILOGUE_BLOCK_N
    subtile_stages: gl.constexpr = 1 if subtile_factor == 1 else 2
    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    acc_smems = gl.allocate_shared_memory(p.c_desc.dtype, [subtile_stages, tile_m, EPILOGUE_BLOCK_N], p.c_desc.layout)
    sub_acc_state = Counter.create(0, subtile_stages)
    scheduler = p.get_clc_consumer()
    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)

        for s in gl.static_range(subtile_factor):
            acc_sub = acc_buf.slice(EPILOGUE_BLOCK_N * s, EPILOGUE_BLOCK_N)
            acc_smem = acc_smems.index(sub_acc_state.index)
            acc = acc_sub.load().to(p.c_desc.dtype)
            tma.store_wait(pendings=subtile_stages - 1)
            acc_smem.store(acc)
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + EPILOGUE_BLOCK_N * s], acc_smem)
            sub_acc_state = sub_acc_state.next()
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1
    tma.store_wait(0)


@gluon.jit
def mma_scaled_clc_partition(p):
    TILE_M: gl.constexpr = p.a_desc.block_shape[0]
    TILE_N: gl.constexpr = p.b_desc.block_shape[0]
    has_work = gl.to_tensor(True)
    num_pid_m = gl.cdiv(p.c_desc.shape[0], TILE_M)
    num_pid_n = gl.cdiv(p.c_desc.shape[1], TILE_N)
    state = Counter.create(0, p.clc_barriers.shape[0])
    consumed_state = Counter.create(1, p.clc_barriers.shape[0])
    ACC_STAGES: gl.constexpr = p.clc_barriers.shape[0]
    i = 0
    while has_work:
        # Reuse the slot only after all consumer partitions signaled consumed.
        mbarrier.wait(p.clc_consumed_bars.index(consumed_state.index), consumed_state.phase, pred=(i >= ACC_STAGES))
        barrier = p.clc_barriers.index(state.index)
        result = p.clc_result_buffers.index(state.index)
        # 16: clc.try_cancel has a `.b128` modifier
        mbarrier.expect(barrier, 16)
        clc.try_cancel(result, barrier)
        mbarrier.wait(barrier, state.phase)
        clc_res = clc.load_result(result)
        has_work = clc_res.is_canceled()
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            tile_id = clc_res.program_id(0)
            pid_m, pid_n = _planar_snake(tile_id, num_pid_m, num_pid_n, p.MINOR_DIM, p.GRID_TILE_WIDTH)
        packed_pid = (pid_m.to(gl.int64) << 32) | (pid_n.to(gl.int64) & 0xFFFFFFFF)
        planar_slot = p.clc_planar_pid_buffers.index(state.index)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        planar_slot.store(gl.full([1], packed_pid, gl.int64, layout=planar_layout))
        mbarrier.arrive(p.clc_planar_ready_bars.index(state.index))
        state = state.next()
        consumed_state = consumed_state.next()
        i += 1


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@gluon.jit
def mma_scaled_warp_specialized_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, M, N, K, A_ELEM_PER_BYTE,
                                       num_buffers: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                       BLOCK_K: gl.constexpr, EPILOGUE_BLOCK_N: gl.constexpr,
                                       num_acc_buffers: gl.constexpr, GRID_MINOR_DIM: gl.constexpr,
                                       GRID_TILE_WIDTH: gl.constexpr, CGA_LAYOUT: gl.constexpr):
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    gl.static_assert(BLOCK_M_PER_CTA == 64 or BLOCK_M_PER_CTA == 128)
    N_PARTITIONS: gl.constexpr = 4

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_shape,
                                             b_scale_desc.layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M_PER_CTA, BLOCK_N], col_stride=1, cga_layout=CGA_LAYOUT,
                                                   two_ctas=TWO_CTAS)

    load_empty_bars = mbarrier.allocate_mbarrier(batch=num_buffers)
    load_ready_bars = mbarrier.allocate_mbarrier(batch=num_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    acc_empty_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers, two_ctas=TWO_CTAS)
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    clc_barriers = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    clc_planar_ready_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    clc_consumed_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(clc_barriers.index(i), count=1)
        mbarrier.init(clc_planar_ready_bars.index(i), count=1)
        mbarrier.init(clc_consumed_bars.index(i), count=N_PARTITIONS - 1)

    cga_layout_clc: gl.constexpr = [[0]] * (gl.num_ctas().bit_length() - 1)
    clc_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0], cga_layout=cga_layout_clc)
    clc_result_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 2], clc_layout)
    clc_planar_pid_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 1], clc_layout)

    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)
    p = PartitionArgs(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                      load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars, clc_result_buffers,
                      clc_barriers, clc_planar_pid_buffers, clc_planar_ready_bars, clc_consumed_bars, GRID_MINOR_DIM,
                      GRID_TILE_WIDTH)

    gl.warp_specialize([
        (mma_scaled_epilogue_partition, (p, )),
        (mma_scaled_mma_partition, (p, )),
        (mma_scaled_load_partition, (p, )),
        (mma_scaled_clc_partition, (p, )),
    ], [1, 1, 1], [24, 24, 24])


mma_scaled_kernel = triton.autotune(
    configs=mma_scaled_get_configs(pre_hook=mma_scaled_tma_set_block_size_hook),
    key=["M", "N", "K", "A_ELEM_PER_BYTE"],
)(mma_scaled_warp_specialized_kernel)

mma_scaled_1cta_kernel = triton.autotune(
    configs=mma_scaled_get_configs(pre_hook=mma_scaled_tma_set_block_size_hook, cga_layouts=[()]),
    key=["M", "N", "K", "A_ELEM_PER_BYTE"],
)(mma_scaled_warp_specialized_kernel)

mma_scaled_2cta_kernel = triton.autotune(
    configs=mma_scaled_get_configs(pre_hook=mma_scaled_tma_set_block_size_hook, cga_layouts=[((1, 0), )]),
    key=["M", "N", "K", "A_ELEM_PER_BYTE"],
)(mma_scaled_warp_specialized_kernel)

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def make_dummy_descriptors(A, B, A_scale, B_scale, out_dtype, M, N):
    """Create TMA descriptors with dummy block shapes; the hook sets the real ones."""
    dummy_block_2d = [1, 1]
    dummy_layout_2d = gl.NVMMASharedLayout.get_default_for(dummy_block_2d, gl.float8e4nv)
    a_desc = TensorDescriptor.from_tensor(A, dummy_block_2d, dummy_layout_2d)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block_2d, dummy_layout_2d)

    C = torch.empty(M, N, device="cuda", dtype=out_dtype)
    C_dtype = getattr(gl, str(out_dtype).split('.')[1])
    c_layout = gl.NVMMASharedLayout.get_default_for(dummy_block_2d, C_dtype)
    c_desc = TensorDescriptor.from_tensor(C, dummy_block_2d, c_layout)

    A_scale_5d = A_scale.reshape(1, A_scale.shape[0], A_scale.shape[1], 2, 256)
    B_scale_5d = B_scale.reshape(1, B_scale.shape[0], B_scale.shape[1], 2, 256)
    dummy_block_5d = [1, 1, 1, 2, 256]
    dummy_layout_5d = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
    a_scale_desc = TensorDescriptor.from_tensor(A_scale_5d, dummy_block_5d, dummy_layout_5d)
    b_scale_desc = TensorDescriptor.from_tensor(B_scale_5d, dummy_block_5d, dummy_layout_5d)

    return a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc


def mma_scaled_warp_specialized(A, B, A_scale, B_scale, VEC_SIZE, GRID_MINOR_DIM=0, GRID_TILE_WIDTH=4,
                                out_dtype=torch.float16, BLOCK_M=128, BLOCK_N=256, BLOCK_K=None, EPILOGUE_BLOCK_N=None,
                                num_buffers=3, acc_buffers=None, num_ctas=1):
    """Warp-specialized block-scale MMA (supports 1CTA and 2CTA)."""
    if BLOCK_K is None:
        BLOCK_K = 128 if torch.float8_e4m3fn in [A.dtype, B.dtype] else 256
    if EPILOGUE_BLOCK_N is None:
        EPILOGUE_BLOCK_N = BLOCK_N
    if acc_buffers is None:
        acc_buffers = 2 if BLOCK_N < 256 else 1

    M, N = A.shape[0], B.shape[0]
    IS_FP4_A = A.dtype == torch.uint8
    K = A.shape[1] * (2 if IS_FP4_A else 1)
    cga_layout = ((1, 0), ) if num_ctas > 1 else ()

    A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc = make_dummy_descriptors(A, B, A_scale, B_scale, out_dtype, M, N)

    mma_scaled_tma_set_block_size_hook({
        "a_desc": A_desc,
        "b_desc": B_desc,
        "c_desc": C_desc,
        "a_scale_desc": A_scale_desc,
        "b_scale_desc": B_scale_desc,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "EPILOGUE_BLOCK_N": EPILOGUE_BLOCK_N,
        "CGA_LAYOUT": cga_layout,
    })

    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (num_pid, )
    A_ELEM_PER_BYTE = 2 if IS_FP4_A else 1
    mma_scaled_warp_specialized_kernel[grid](
        A_desc,
        B_desc,
        C_desc,
        A_scale_desc,
        B_scale_desc,
        M,
        N,
        K,
        A_ELEM_PER_BYTE,
        num_buffers,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EPILOGUE_BLOCK_N,
        acc_buffers,
        GRID_MINOR_DIM,
        GRID_TILE_WIDTH,
        cga_layout,
        num_ctas=num_ctas,
    )
    return C_desc.base


def mma_scaled_matmul(A, B, A_scale, B_scale, VEC_SIZE, out_dtype=torch.float16, num_ctas=None):
    """Autotuned block-scaled matmul.

    Args:
        num_ctas: None = autotune across all configs (1CTA and 2CTA),
                  1 = autotune 1CTA configs only,
                  2 = autotune 2CTA configs only.
    """
    M, N = A.shape[0], B.shape[0]
    IS_FP4_A = A.dtype == torch.uint8
    A_ELEM_PER_BYTE = 2 if IS_FP4_A else 1
    K = A.shape[1] * A_ELEM_PER_BYTE

    a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc = make_dummy_descriptors(A, B, A_scale, B_scale, out_dtype, M, N)

    def grid(meta):
        num_tiles = triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"])
        return (num_tiles, )

    kernel = {None: mma_scaled_kernel, 1: mma_scaled_1cta_kernel, 2: mma_scaled_2cta_kernel}[num_ctas]
    kernel[grid](a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, M, N, K, A_ELEM_PER_BYTE)
    return c_desc.base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [128, 640, 704, 1152, 4096])
@pytest.mark.parametrize("M, N", [(2048, 2048), (500, 600), (256, 256), (128, 128), (8192, 8192)])
@pytest.mark.parametrize("a_format, b_format",
                         list(itertools.product(["mxfp8", "mxfp4"], repeat=2)) + [("nvfp4", "nvfp4")])
@pytest.mark.parametrize("num_ctas, BLOCK_N, EPILOGUE_BLOCK_N, num_buffers", [
    (2, 256, 256, 4),
    (2, 256, 64, 5),
    (2, 128, 64, 6),
    (1, 256, 256, 3),
    (1, 256, 64, 3),
    (1, 128, 64, 5),
])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_mma_scaled_warp_specialized(M, N, K, a_format, b_format, num_ctas, BLOCK_N, EPILOGUE_BLOCK_N, num_buffers):
    if a_format != b_format and K % 128 != 0:
        pytest.skip("fp4 packed tensor descriptor requires K to be a multiple of 128")
    BLOCK_M = 256 if num_ctas > 1 else 128
    torch.manual_seed(0)
    A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
    B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
    VEC_SIZE = 16 if a_format == "nvfp4" else 32
    A_scale = swizzle_scales_packed_block(A_scale)
    B_scale = swizzle_scales_packed_block(B_scale)
    C_ref = A_ref @ B_ref.T
    C = mma_scaled_warp_specialized(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                    EPILOGUE_BLOCK_N=EPILOGUE_BLOCK_N, num_buffers=num_buffers, num_ctas=num_ctas)
    torch.testing.assert_close(C_ref, C.to(torch.float32), atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

if is_blackwell():
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None

CUBLAS_FORMATS = {"mxfp8", "nvfp4"}


def cublas_block_scaled_matmul(A, B, A_scale_flat, B_scale_flat, fmt):
    """cuBLAS block-scaled matmul. Supports mxfp8 and nvfp4 (mxfp4 not supported by cuBLAS)."""
    M, N = A.shape[0], B.shape[0]
    output = torch.empty((M, N), dtype=torch.float16, device="cuda")
    if fmt == "mxfp8":
        cublas.block_scaled_matmul_mxfp8(A, B, output, A_scale_flat, B_scale_flat)
    elif fmt == "nvfp4":
        cublas.block_scaled_matmul_nvfp4(A, B, output, A_scale_flat, B_scale_flat)
    else:
        raise ValueError(f"cuBLAS does not support format: {fmt}")
    return output


ALL_FORMATS = [("mxfp8", "mxfp8"), ("nvfp4", "nvfp4"), ("mxfp8", "mxfp4"), ("mxfp4", "mxfp4")]

MNK_VALS = [8192, 16384, 32768]

BEST_1CTA_CONFIG = dict(BLOCK_M=128, BLOCK_N=256, EPILOGUE_BLOCK_N=64, num_buffers=3, num_ctas=1, GRID_MINOR_DIM=1,
                        GRID_TILE_WIDTH=8)
BEST_2CTA_CONFIG = dict(BLOCK_M=256, BLOCK_N=256, EPILOGUE_BLOCK_N=64, num_buffers=5, num_ctas=2, GRID_MINOR_DIM=0,
                        GRID_TILE_WIDTH=8)


def make_fn(variant, A, B, A_scale, B_scale, VEC_SIZE, a_format, use_autotuned=False):
    """Build the callable for a given variant (1cta, 2cta, or cublas)."""
    if variant == "2cta":
        if use_autotuned:
            return lambda: mma_scaled_matmul(A, B, A_scale, B_scale, VEC_SIZE, num_ctas=2)
        return lambda: mma_scaled_warp_specialized(A, B, A_scale, B_scale, VEC_SIZE, **BEST_2CTA_CONFIG)
    elif variant == "1cta":
        if use_autotuned:
            return lambda: mma_scaled_matmul(A, B, A_scale, B_scale, VEC_SIZE, num_ctas=1)
        return lambda: mma_scaled_warp_specialized(A, B, A_scale, B_scale, VEC_SIZE, **BEST_1CTA_CONFIG)
    elif variant == "cublas":
        A_scale_flat = A_scale.contiguous().flatten()
        B_scale_flat = B_scale.contiguous().flatten()

        def cublas_fn():
            return cublas_block_scaled_matmul(A, B, A_scale_flat, B_scale_flat, a_format)

        return cublas_fn
    else:
        raise ValueError(f"Unknown variant: {variant}")


def make_tensors(MNK, a_format, b_format):
    """Allocate and prepare input tensors for a given size and format."""
    M = N = K = MNK
    torch.manual_seed(0)
    A, A_scale, _ = random_quantized_tensor(M, K, a_format)
    B, B_scale, _ = random_quantized_tensor(N, K, b_format)
    A_scale = swizzle_scales_packed_block(A_scale)
    B_scale = swizzle_scales_packed_block(B_scale)
    VEC_SIZE = 16 if a_format == "nvfp4" else 32
    return A, B, A_scale, B_scale, VEC_SIZE


def get_variants(a_format, b_format):
    """Return the list of variants available for a given format pair."""
    has_cublas = a_format == b_format and a_format in CUBLAS_FORMATS
    return ["1cta", "2cta", "cublas"] if has_cublas else ["1cta", "2cta"]


def print_table(label, variants, mnk_vals, results):
    """Print a formatted benchmark table with optional ratio columns."""
    has_cublas = "cublas" in variants
    col_w = 16
    header = f"{'MNK':>8}"
    header += f"  {'1cta (TFLOPS)':>{col_w}}"
    header += f"  {'2cta (TFLOPS)':>{col_w}}"
    header += f"  {'2cta/1cta':>{col_w}}"
    if has_cublas:
        header += f"  {'cublas (TFLOPS)':>{col_w}}"
        header += f"  {'2cta/cublas':>{col_w}}"
    print(f"block-scale-matmul-{label}:")
    print(header)
    for MNK in mnk_vals:
        t1 = results.get((label, "1cta", MNK))
        t2 = results.get((label, "2cta", MNK))
        ratio_2v1 = t2 / t1 if t1 and t2 else 0.0
        row = f"{MNK:>8}"
        row += f"  {t1:>{col_w}.1f}" if t1 else f"  {'--':>{col_w}}"
        row += f"  {t2:>{col_w}.1f}" if t2 else f"  {'--':>{col_w}}"
        row += f"  {ratio_2v1:>{col_w}.2f}"
        if has_cublas:
            tc = results.get((label, "cublas", MNK))
            ratio_2vc = t2 / tc if t2 and tc else 0.0
            row += f"  {tc:>{col_w}.1f}" if tc else f"  {'--':>{col_w}}"
            row += f"  {ratio_2vc:>{col_w}.2f}"
        print(row)
    print()


def format_config(cfg):
    """Format an autotuner Config as a concise string."""
    if cfg is None:
        return "(none)"
    kw = cfg.kwargs
    parts = [
        f"BM={kw['BLOCK_M']}", f"BN={kw['BLOCK_N']}", f"BK={kw['BLOCK_K']}", f"epilogue_N={kw['EPILOGUE_BLOCK_N']}",
        f"bufs={kw['num_buffers']}", f"acc_bufs={kw['num_acc_buffers']}", f"minor={kw['GRID_MINOR_DIM']}",
        f"tile_w={kw['GRID_TILE_WIDTH']}", f"cga={kw['CGA_LAYOUT']}"
    ]
    return ", ".join(parts)


def run_benchmark(use_autotuned=False):
    results = {}
    best_configs = {}
    for a_format, b_format in ALL_FORMATS:
        label = f"{a_format}-{b_format}"
        variants = get_variants(a_format, b_format)
        for MNK in MNK_VALS:
            A, B, A_scale, B_scale, VEC_SIZE = make_tensors(MNK, a_format, b_format)
            for variant in variants:
                if use_autotuned:
                    print(f"  {label} {variant} MNK={MNK}: ...", end="", flush=True)
                fn = make_fn(variant, A, B, A_scale, B_scale, VEC_SIZE, a_format, use_autotuned=use_autotuned)
                ms = triton.testing.do_bench(fn)
                tflops = 2.0 * MNK**3 * 1e-12 / (ms * 1e-3)
                results[(label, variant, MNK)] = tflops
                if use_autotuned:
                    print(f"\r  {label} {variant} MNK={MNK}: {tflops:.1f} TFLOPS")
                    if variant == "1cta":
                        best_configs[(label, "1cta", MNK)] = mma_scaled_1cta_kernel.best_config
                    elif variant == "2cta":
                        best_configs[(label, "2cta", MNK)] = mma_scaled_2cta_kernel.best_config

    if use_autotuned:
        largest_mnk = MNK_VALS[-1]
        print(f"\nBest autotuned configs (MNK={largest_mnk}):")
        for a_format, b_format in ALL_FORMATS:
            label = f"{a_format}-{b_format}"
            c1 = best_configs.get((label, "1cta", largest_mnk))
            c2 = best_configs.get((label, "2cta", largest_mnk))
            print(f"  {label}:")
            print(f"    1cta: {format_config(c1)}")
            print(f"    2cta: {format_config(c2)}")
        print()

    for a_format, b_format in ALL_FORMATS:
        label = f"{a_format}-{b_format}"
        variants = get_variants(a_format, b_format)
        print_table(label, variants, MNK_VALS, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block-scaled matmul benchmark")
    parser.add_argument(
        "--use-autotuned",
        action="store_true",
        help="Use autotuned mma_scaled_matmul() instead of mma_scaled_warp_specialized().",
    )
    args = parser.parse_args()
    run_benchmark(use_autotuned=args.use_autotuned)
