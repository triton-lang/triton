"""
2CTA Block-Scaled Matrix Multiplication
=======================================

High-performance 2CTA warp-specialized block-scaled MMA.
Two CTAs cooperate per output tile, sharing operands to
increase arithmetic intensity and reduce the per-CTA SMEM
footprint.
"""

import pytest
import torch

import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor
from triton.experimental.gluon.language.nvidia import blackwell as blackwell_lang
from triton.experimental.gluon.language.nvidia import rubin as rubin_lang
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    clc,
    tcgen05_copy,
    tcgen05_commit,
    tcgen05_mma_barrier_count,
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


def is_cuda():
    return torch.cuda.is_available() and triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] in (10, 11)


def is_rubin():
    return is_cuda() and torch.cuda.get_device_capability() == (10, 7)


IS_RUBIN = gl.constexpr(is_rubin())

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


SCHEDULER_CLC = gl.constexpr("cluster_launch_control")
SCHEDULER_SPS = gl.constexpr("static_persistent")


@gluon.jit
def unswizzle_scales_shared_memory(smem, BLOCK_MN: gl.constexpr, BLOCK_K: gl.constexpr, VEC_SIZE: gl.constexpr):
    smem = smem.reshape((smem.shape[1], smem.shape[2], 32, 4, 4))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((BLOCK_MN, BLOCK_K // VEC_SIZE))


@gluon.jit
def async_mma_scaled_impl(a_smem, b_smem, a_scale_smem, b_scale_smem, acc_tmem, a_empty_bar, b_empty_bar, use_acc,
                          pred):
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
    a_scale_cga: gl.constexpr = [[1, 0]] if two_ctas else []
    # Rubin requires kThenMn ordering for NVFP4 A scales when BLOCK_M_PER_CTA=256.
    if IS_RUBIN:
        BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // gl.num_ctas()
        a_scale_order: gl.constexpr = "kThenMn" if VEC_SIZE == 16 and BLOCK_M_PER_CTA == 256 else "mnThenK"
        a_scale_layout: gl.constexpr = rubin_lang.TensorMemoryScalesLayout(cga_layout=a_scale_cga,
                                                                           block_rep_order=a_scale_order)
        b_scale_layout: gl.constexpr = rubin_lang.TensorMemoryScalesLayout(cga_layout=[[0, 0]] if two_ctas else [])
    else:
        a_scale_layout: gl.constexpr = blackwell_lang.TensorMemoryScalesLayout(cga_layout=a_scale_cga)
        b_scale_layout: gl.constexpr = blackwell_lang.TensorMemoryScalesLayout(cga_layout=[[0, 0]] if two_ctas else [])
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype, a_scale.type.shape, a_scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype, b_scale.type.shape, b_scale_layout)
    tcgen05_copy(a_scale, a_scale_tmem)
    tcgen05_copy(b_scale, b_scale_tmem)

    a_format: gl.constexpr = "e2m1" if a_smem.dtype == gl.uint8 else "e4m3"
    b_format: gl.constexpr = "e2m1" if b_smem.dtype == gl.uint8 else "e4m3"
    tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                       use_acc=use_acc, pred=pred, multicast=True, mbarriers=[a_empty_bar, b_empty_bar])


# This helper function computes all the load indexing and issues the async loads
# based on the current `pid_m`, `pid_n`, and `k` indices. The compiler will run
# loop-invariant code motion to hoist code that does not depend on `k`, like
# `pid_m * BLOCK_M`, outside of the inner loop, so we can safely abstract the
# load indexing without performance loss.
#
# Encapsulating the load indexing logic will help keep our pipelined kernel code
# clean, as pipelining can get messy.
@gluon.jit
def issue_operand_load(desc, scale_desc, bufs, scale_bufs, empty_bars, ready_bars, state, pid, k, pred=True):
    ELEM_PER_BYTE: gl.constexpr = 2 if desc.dtype == gl.uint8 else 1
    BLOCK_K: gl.constexpr = desc.block_shape[1] * ELEM_PER_BYTE
    mbarrier.wait(empty_bars.index(state.index), state.phase, pred)
    bar = ready_bars.index(state.index)
    mbarrier.expect(bar, desc.nbytes_per_cta + scale_desc.nbytes_per_cta, pred)
    tma.async_load(desc, [pid * desc.block_shape[0], k // ELEM_PER_BYTE], bar, bufs.index(state.index), pred,
                   multicast=True)
    tma.async_load(scale_desc, [0, pid * scale_desc.block_shape[1], (k // BLOCK_K) * scale_desc.block_shape[2], 0, 0],
                   bar, scale_bufs.index(state.index), pred, multicast=True)
    return state.next(pred)


@gluon.aggregate
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


@gluon.aggregate
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


@gluon.aggregate
class SpsTileScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    pid_m: gl.tensor
    pid_n: gl.tensor
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr
    NUM_PID_M: gl.tensor
    NUM_PID_N: gl.tensor

    @gluon.jit
    def initialize(TILE_M: gl.constexpr, TILE_N: gl.constexpr, MINOR_DIM: gl.constexpr, GRID_TILE_WIDTH: gl.constexpr,
                   NUM_PID_M, NUM_PID_N):
        tile_id = gl.program_id(axis=0)
        has_work = tile_id < NUM_PID_M * NUM_PID_N
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            pid_m, pid_n = _planar_snake(tile_id, NUM_PID_M, NUM_PID_N, MINOR_DIM, GRID_TILE_WIDTH)
        return SpsTileScheduler(has_work, tile_id, pid_m, pid_n, TILE_M, TILE_N, MINOR_DIM, GRID_TILE_WIDTH, NUM_PID_M,
                                NUM_PID_N)

    @gluon.jit
    def get_offsets(self):
        return self.pid_m * self.TILE_M, self.pid_n * self.TILE_N

    @gluon.jit
    def step(self, iteration):
        next_tile_id = self.tile_id + gl.num_programs(axis=0)
        has_work = next_tile_id < self.NUM_PID_M * self.NUM_PID_N
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            pid_m, pid_n = _planar_snake(next_tile_id, self.NUM_PID_M, self.NUM_PID_N, self.MINOR_DIM,
                                         self.GRID_TILE_WIDTH)
        return SpsTileScheduler(has_work, next_tile_id, pid_m, pid_n, self.TILE_M, self.TILE_N, self.MINOR_DIM,
                                self.GRID_TILE_WIDTH, self.NUM_PID_M, self.NUM_PID_N)


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------


@gluon.aggregate
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
    a_empty_bars: gl.shared_memory_descriptor
    a_ready_bars: gl.shared_memory_descriptor
    b_empty_bars: gl.shared_memory_descriptor
    b_ready_bars: gl.shared_memory_descriptor
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
    NUM_PID_M: gl.tensor
    NUM_PID_N: gl.tensor
    scheduler: gl.constexpr

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
    def get_sps_scheduler(self):
        return SpsTileScheduler.initialize(
            self.a_desc.block_shape[0],
            self.b_desc.block_shape[0],
            self.MINOR_DIM,
            self.GRID_TILE_WIDTH,
            self.NUM_PID_M,
            self.NUM_PID_N,
        )


@gluon.jit
def mma_scaled_load_partition(p):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if p.a_desc.dtype == gl.uint8 else 1
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    K = p.a_desc.shape[1] * A_ELEM_PER_BYTE
    A_LEAD: gl.constexpr = max(p.a_bufs.shape[0] - p.b_bufs.shape[0], 0)
    B_LEAD: gl.constexpr = max(p.b_bufs.shape[0] - p.a_bufs.shape[0], 0)
    a_state = Counter.create(1, p.a_empty_bars.shape[0])
    b_state = Counter.create(1, p.b_empty_bars.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        # Fill the deeper ring's extra slots before advancing both rings.
        # This preserves their depth difference without another producer warp.
        for j in gl.static_range(A_LEAD):
            a_state = issue_operand_load(p.a_desc, p.a_scale_desc, p.a_bufs, p.a_scale_bufs, p.a_empty_bars,
                                         p.a_ready_bars, a_state, scheduler.pid_m, j * BLOCK_K, j * BLOCK_K < K)
        for j in gl.static_range(B_LEAD):
            b_state = issue_operand_load(p.b_desc, p.b_scale_desc, p.b_bufs, p.b_scale_bufs, p.b_empty_bars,
                                         p.b_ready_bars, b_state, scheduler.pid_n, j * BLOCK_K, j * BLOCK_K < K)
        for k in range(0, K, BLOCK_K):
            a_k = k + A_LEAD * BLOCK_K
            b_k = k + B_LEAD * BLOCK_K
            # Issue B and its scales before A and its scales for performance.
            b_state = issue_operand_load(p.b_desc, p.b_scale_desc, p.b_bufs, p.b_scale_bufs, p.b_empty_bars,
                                         p.b_ready_bars, b_state, scheduler.pid_n, b_k,
                                         True if B_LEAD == 0 else b_k < K)
            a_state = issue_operand_load(p.a_desc, p.a_scale_desc, p.a_bufs, p.a_scale_bufs, p.a_empty_bars,
                                         p.a_ready_bars, a_state, scheduler.pid_m, a_k,
                                         True if A_LEAD == 0 else a_k < K)
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def mma_scaled_mma_partition(p):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if p.a_desc.dtype == gl.uint8 else 1
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    K = p.a_desc.shape[1] * A_ELEM_PER_BYTE
    a_state = Counter.create(0, p.a_bufs.shape[0])
    b_state = Counter.create(0, p.b_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for k in range(0, K, BLOCK_K):
            mbarrier.wait(p.a_ready_bars.index(a_state.index), a_state.phase)
            mbarrier.wait(p.b_ready_bars.index(b_state.index), b_state.phase)
            async_mma_scaled_impl(p.a_bufs.index(a_state.index), p.b_bufs.index(b_state.index),
                                  p.a_scale_bufs.index(a_state.index), p.b_scale_bufs.index(b_state.index), acc_buf,
                                  a_empty_bar=p.a_empty_bars.index(a_state.index),
                                  b_empty_bar=p.b_empty_bars.index(b_state.index), use_acc=use_acc, pred=True)
            a_state = a_state.next()
            b_state = b_state.next()
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
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
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
            tma.async_store(p.c_desc, [off_m, off_n + EPILOGUE_BLOCK_N * s], acc_smem)
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
        mbarrier.expect(barrier, 16, from_cta=0x0)
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
def mma_scaled_warp_specialized_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, M, N,
                                       a_buffers: gl.constexpr, b_buffers: gl.constexpr, BLOCK_M: gl.constexpr,
                                       BLOCK_N: gl.constexpr, num_acc_buffers: gl.constexpr,
                                       GRID_MINOR_DIM: gl.constexpr, GRID_TILE_WIDTH: gl.constexpr,
                                       CGA_LAYOUT: gl.constexpr, scheduler: gl.constexpr):
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    N_PARTITIONS: gl.constexpr = 4 if scheduler == SCHEDULER_CLC else 3

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [a_buffers] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [b_buffers] + b_desc.block_shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [a_buffers] + a_scale_desc.block_shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [b_buffers] + b_scale_desc.block_shape,
                                             b_scale_desc.layout)

    # Rubin supports B reuse with BLOCK_M_PER_CTA=256.
    TMEM_BLOCK_M: gl.constexpr = min(BLOCK_M_PER_CTA, 128)
    tmem_layout: gl.constexpr = TensorMemoryLayout([TMEM_BLOCK_M, BLOCK_N], col_stride=1, cga_layout=CGA_LAYOUT,
                                                   two_ctas=TWO_CTAS)
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)

    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0),
         a_scale_bufs.index(0), b_scale_bufs.index(0)], multicast=True, two_ctas=acc_bufs.index(0).type.layout.two_ctas)

    a_empty_bars = mbarrier.allocate_mbarrier(batch=a_buffers)
    a_ready_bars = mbarrier.allocate_mbarrier(batch=a_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(a_buffers):
        mbarrier.init(a_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(a_ready_bars.index(i), count=1)

    b_empty_bars = mbarrier.allocate_mbarrier(batch=b_buffers)
    b_ready_bars = mbarrier.allocate_mbarrier(batch=b_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(b_buffers):
        mbarrier.init(b_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(b_ready_bars.index(i), count=1)

    acc_empty_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers, two_ctas=TWO_CTAS)
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    clc_barriers = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    clc_planar_ready_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    cga_layout_clc: gl.constexpr = [[0]] * (gl.num_ctas().bit_length() - 1)
    clc_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0], cga_layout=cga_layout_clc)
    clc_consumed_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], clc_layout)
    if scheduler == SCHEDULER_CLC:
        for i in gl.static_range(num_acc_buffers):
            mbarrier.init(clc_barriers.index(i), count=1)
            mbarrier.init(clc_planar_ready_bars.index(i), count=1)
            mbarrier.init(clc_consumed_bars.index(i), count=N_PARTITIONS - 1)

    clc_result_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 2], clc_layout)
    clc_planar_pid_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 1], clc_layout)

    p = PartitionArgs(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                      a_empty_bars, a_ready_bars, b_empty_bars, b_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars,
                      clc_result_buffers, clc_barriers, clc_planar_pid_buffers,
                      clc_planar_ready_bars, clc_consumed_bars, GRID_MINOR_DIM, GRID_TILE_WIDTH, gl.cdiv(M, BLOCK_M),
                      gl.cdiv(N, BLOCK_N), scheduler)

    if scheduler == SCHEDULER_CLC:
        gl.warp_specialize([
            (mma_scaled_epilogue_partition, (p, )),
            (mma_scaled_mma_partition, (p, )),
            (mma_scaled_load_partition, (p, )),
            (mma_scaled_clc_partition, (p, )),
        ], [1, 1, 1], [24, 24, 24])
    else:
        gl.warp_specialize([
            (mma_scaled_epilogue_partition, (p, )),
            (mma_scaled_mma_partition, (p, )),
            (mma_scaled_load_partition, (p, )),
        ], [1, 1], [24, 24])


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def make_descriptors(A, B, A_scale, B_scale, out_dtype, BLOCK_M, BLOCK_N, BLOCK_K, EPILOGUE_BLOCK_N, num_ctas):
    """Construct TMA descriptors with their final logical block shapes."""
    cga = ((1, 0), ) if num_ctas == 2 else None
    a_is_fp4, b_is_fp4 = A.dtype == torch.uint8, B.dtype == torch.uint8
    a_block = [BLOCK_M, BLOCK_K // (2 if a_is_fp4 else 1)]
    b_block = [BLOCK_N, BLOCK_K // (2 if b_is_fp4 else 1)]
    mixed = a_is_fp4 != b_is_fp4
    a_layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.uint8 if a_is_fp4 else gl.float8e4nv,
                                                    fp4_padded=not is_rubin() and mixed and a_is_fp4, cga_layout=cga)
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block, gl.uint8 if b_is_fp4 else gl.float8e4nv,
                                                    fp4_padded=not is_rubin() and mixed and b_is_fp4, cga_layout=cga)
    a_desc = TensorDescriptor.from_tensor(A, a_block, a_layout)
    b_desc = TensorDescriptor.from_tensor(B, b_block, b_layout)
    C = torch.empty((A.shape[0], B.shape[0]), device=A.device, dtype=out_dtype)
    c_block = [BLOCK_M, EPILOGUE_BLOCK_N]
    c_layout = gl.NVMMASharedLayout.get_default_for(c_block, getattr(gl, str(out_dtype).split('.')[1]), cga_layout=cga)
    c_desc = TensorDescriptor.from_tensor(C, c_block, c_layout)

    vec_size = 16 if A_scale.dtype == torch.float8_e4m3fn else 32
    a_scale_block = [1, BLOCK_M // 128, BLOCK_K // (vec_size * 4), 2, 256]
    b_scale_block = [1, BLOCK_N // 128, BLOCK_K // (vec_size * 4), 2, 256]
    a_scale_layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5,
                                          cga_layout=[[0, 1, 0, 0, 0]] if num_ctas == 2 else None)
    b_scale_layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5,
                                          cga_layout=[[0, 0, 0, 0, 0]] if num_ctas == 2 else None)
    a_scale_desc = TensorDescriptor.from_tensor(A_scale.reshape(1, A_scale.shape[0], A_scale.shape[1], 2, 256),
                                                a_scale_block, a_scale_layout)
    b_scale_desc = TensorDescriptor.from_tensor(B_scale.reshape(1, B_scale.shape[0], B_scale.shape[1], 2, 256),
                                                b_scale_block, b_scale_layout)
    return a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc


def mma_scaled_warp_specialized_grid(M, N, BLOCK_M, BLOCK_N, num_ctas, scheduler, device):
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    if scheduler == SCHEDULER_CLC:
        return (num_pid, )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return (max(1, min(num_pid, sm_count // num_ctas)), )


def mma_scaled_warp_specialized(A, B, A_scale, B_scale, *, scheduler, GRID_MINOR_DIM, GRID_TILE_WIDTH, out_dtype,
                                BLOCK_M, BLOCK_N, BLOCK_K, EPILOGUE_BLOCK_N, a_buffers: int, b_buffers: int,
                                acc_buffers, num_ctas, num_warps):
    """Warp-specialized block-scale MMA (supports 1CTA and 2CTA)."""
    assert a_buffers > 0 and b_buffers > 0, "pipeline depths must be positive"
    assert num_ctas in (1, 2), "num_ctas must be 1 or 2"
    assert scheduler in (SCHEDULER_CLC, SCHEDULER_SPS), "unsupported scheduler"
    assert BLOCK_M // num_ctas in ((64, 128, 256) if is_rubin() else (64, 128)), "unsupported BLOCK_M per CTA"
    assert BLOCK_M >= 128, "BLOCK_M must be at least 128 for scale descriptors"
    assert BLOCK_N % EPILOGUE_BLOCK_N == 0, "EPILOGUE_BLOCK_N must divide BLOCK_N"
    descs = make_descriptors(A, B, A_scale, B_scale, out_dtype, BLOCK_M, BLOCK_N, BLOCK_K, EPILOGUE_BLOCK_N, num_ctas)
    M, N = A.shape[0], B.shape[0]
    grid = mma_scaled_warp_specialized_grid(M, N, BLOCK_M, BLOCK_N, num_ctas, scheduler, A.device)
    mma_scaled_warp_specialized_kernel[grid](*descs, M, N, a_buffers, b_buffers, BLOCK_M, BLOCK_N, acc_buffers,
                                             GRID_MINOR_DIM, GRID_TILE_WIDTH, ((1, 0), ) if num_ctas == 2 else (),
                                             scheduler=scheduler, num_ctas=num_ctas, num_warps=num_warps)
    return descs[2].base


# ---------------------------------------------------------------------------
# Best configs
# ---------------------------------------------------------------------------

ALL_FORMATS = [("mxfp8", "mxfp8"), ("nvfp4", "nvfp4"), ("mxfp8", "mxfp4"), ("mxfp4", "mxfp4")]
CONFIG_FIELDS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "EPILOGUE_BLOCK_N", "num_buffers", "acc_buffers", "GRID_MINOR_DIM",
                 "GRID_TILE_WIDTH")

if is_rubin():
    # Rubin supports B reuse with BLOCK_M_PER_CTA=256.
    BEST_CONFIGS = {
        ("mxfp8-mxfp8", 1): (256, 256, 128, 32, 4, 1, 1, 16),
        ("mxfp8-mxfp8", 2): (512, 256, 128, 32, 5, 1, 1, 16),
        ("nvfp4-nvfp4", 1): (256, 256, 256, 32, 4, 1, 1, 16),
        ("nvfp4-nvfp4", 2): (512, 256, 256, 32, 5, 1, 1, 16),
        ("mxfp8-mxfp4", 1): (256, 256, 128, 32, 5, 1, 1, 16),
        ("mxfp8-mxfp4", 2): (512, 256, 128, 64, 6, 1, 1, 16),
        ("mxfp4-mxfp4", 1): (256, 256, 256, 32, 4, 1, 1, 16),
        ("mxfp4-mxfp4", 2): (512, 256, 256, 32, 5, 1, 1, 16),
    }
else:
    BEST_CONFIGS = {
        ("mxfp8-mxfp8", 1): (128, 256, 128, 64, 3, 1, 1, 8),
        ("mxfp8-mxfp8", 2): (256, 256, 128, 64, 5, 1, 0, 8),
        ("nvfp4-nvfp4", 1): (128, 256, 256, 64, 3, 1, 1, 8),
        ("nvfp4-nvfp4", 2): (256, 256, 256, 64, 5, 1, 0, 8),
        ("mxfp8-mxfp4", 1): (128, 256, 128, 64, 3, 1, 1, 8),
        ("mxfp8-mxfp4", 2): (256, 256, 128, 64, 5, 1, 0, 8),
        ("mxfp4-mxfp4", 1): (128, 256, 256, 64, 3, 1, 1, 8),
        ("mxfp4-mxfp4", 2): (256, 256, 256, 64, 5, 1, 0, 8),
    }


def scheduler_for_k(K):
    return SCHEDULER_SPS if K <= 8192 else SCHEDULER_CLC


def get_best_config(a_format, b_format, num_ctas, K):
    vals = BEST_CONFIGS[(f"{a_format}-{b_format}", num_ctas)]
    config = dict(zip(CONFIG_FIELDS, vals))
    config["num_ctas"] = num_ctas
    config["scheduler"] = scheduler_for_k(K)
    depth = config.pop("num_buffers")
    config.update(a_buffers=depth, b_buffers=depth)
    config.update(out_dtype=torch.float16, num_warps=4)
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M, N, K", [(8192, 8192, 8192), (128, 128, 128), (500, 600, 8320)])
@pytest.mark.parametrize("double_buffer", [False, True])
@pytest.mark.parametrize("a_format, b_format", ALL_FORMATS)
@pytest.mark.parametrize("a_buffers, b_buffers", [(None, None), (1, 3), (3, 1)])
@pytest.mark.parametrize("num_ctas", [1, 2])
@pytest.mark.parametrize("scheduler", [SCHEDULER_CLC, SCHEDULER_SPS])
@pytest.mark.skipif(not (is_blackwell() or is_rubin()), reason="Requires Blackwell or Rubin")
@pytest.mark.enable_warmup(min_capability=10)
def test_mma_scaled_best_config(M, N, K, double_buffer, a_format, b_format, a_buffers, b_buffers, num_ctas, scheduler):
    if double_buffer and not is_rubin():
        pytest.skip("Double-buffered 128x256 tiles require Rubin")
    config = get_best_config(a_format, b_format, num_ctas, K)
    config["scheduler"] = scheduler
    if double_buffer:
        config.update(BLOCK_M=128 * num_ctas, BLOCK_N=256, acc_buffers=2)
    if a_buffers is not None:
        config.update(a_buffers=a_buffers, b_buffers=b_buffers)
    torch.manual_seed(0)
    A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
    B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
    A_scale = swizzle_scales_packed_block(A_scale)
    B_scale = swizzle_scales_packed_block(B_scale)
    C_ref = A_ref @ B_ref.T
    C = mma_scaled_warp_specialized(A, B, A_scale, B_scale, **config)
    torch.testing.assert_close(C_ref, C.to(torch.float32), atol=2e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

K_VALS = [8192, 16384, 32768]


def run_benchmark():
    M = N = 16384 if is_rubin() else 8192
    print(f"M={M}, N={N}")
    print("format             K     1CTA TFLOP/s  2CTA TFLOP/s  2CTA/1CTA")
    for a_format, b_format in ALL_FORMATS:
        for K in K_VALS:
            torch.manual_seed(0)
            A, sa, _ = random_quantized_tensor(M, K, a_format)
            B, sb, _ = random_quantized_tensor(N, K, b_format)
            sa, sb = swizzle_scales_packed_block(sa), swizzle_scales_packed_block(sb)
            rates = []
            for ctas in (1, 2):
                config = get_best_config(a_format, b_format, ctas, K)
                # Prepare the output and descriptors before timing kernel launches.
                descs = make_descriptors(A, B, sa, sb, config["out_dtype"], config["BLOCK_M"], config["BLOCK_N"],
                                         config["BLOCK_K"], config["EPILOGUE_BLOCK_N"], ctas)
                grid = mma_scaled_warp_specialized_grid(M, N, config["BLOCK_M"], config["BLOCK_N"], ctas,
                                                        config["scheduler"], A.device)
                fn = lambda: mma_scaled_warp_specialized_kernel[grid](
                    *descs, M, N, config["a_buffers"], config["b_buffers"], config["BLOCK_M"], config["BLOCK_N"],
                    config["acc_buffers"], config["GRID_MINOR_DIM"], config["GRID_TILE_WIDTH"], ((1, 0), )
                    if ctas == 2 else (), scheduler=config["scheduler"], num_ctas=ctas, num_warps=config["num_warps"])
                ms = triton.testing.do_bench(fn)
                rates.append(2 * M * N * K * 1e-9 / ms)
            print(f"{a_format + '-' + b_format:15} {K:5d} {rates[0]:13.1f} {rates[1]:13.1f} {rates[1]/rates[0]:10.2f}")


if __name__ == "__main__":
    run_benchmark()
