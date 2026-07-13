"""
Overlapping Accumulator Block-Scaled Matrix Multiplication
==========================================================

This example demonstrates overlapping accumulator storage in TMEM for 2CTA
block scale MMA by comparing double buffered overlapping BLOCK_N=256
accumulators against a single buffered BLOCK_N=256 accumulator.
"""

import argparse
import pytest
import torch

import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tcgen05_copy,
    tcgen05_commit,
    tcgen05_mma_barrier_count,
    tcgen05_mma_scaled,
    mbarrier,
    tma,
)


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


# ---------------------------------------------------------------------------
# Fixed Configuration
# ---------------------------------------------------------------------------

FORMATS = [("mxfp8", "mxfp8"), ("mxfp8", "mxfp4"), ("mxfp4", "mxfp4"), ("nvfp4", "nvfp4")]
BLOCK_M = 256
BLOCK_N = 256
EPILOGUE_BLOCK_N = 32
NUM_BUFFERS = 5
NUM_CTAS = 2
CGA_LAYOUT = ((1, 0), )
OUT_DTYPE = torch.float16
TMEM_COLS = gl.constexpr(512)

# ---------------------------------------------------------------------------
# Overlapping Accumulator Layout
# ---------------------------------------------------------------------------

# yapf: disable
#
# With 512 TMEM columns, double buffered BLOCK_N=256 accumulators fully consume TMEM
# leaving no room for the A and B scales required for a block scale MMA workload.
#
# Gluon example 04 uses a single buffered BLOCK_N=256 accumulator, leaving
# some performance on the table.
#
# This example keeps the double buffered BLOCK_N=256 accumulators by overlapping them
# in TMEM by exactly the size of the A and B scales as shown here:
#
#   TMEM:                                     BLOCK_N                        TMEM_COLS
#                0                              256                             512
#                [---------------------------------------------------------------)
#   left accum   [-------------------------------)
#   right accum                            [-------------------------------)
#   A scales                                                               [-)
#   B scales                                                                 [---)
#
# The overlap path demonstrates three major enhancements over example 04:
#   1) Fully allocate TMEM and then slice it to hold accumulators and scales.
#   2) Epilogue drains overlapped columns first so the next MMA can proceed.
#   3) New ping-pong style barrier guards the overlapped columns.
#
# Tile 0 MMA starts immediately and writes the left accumulator:
#
#   left accum   [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)
#                start -> MMA
#
# Tile 0 epilogue drains the overlap first and releases the overlap barrier:
#
#                <--------- drain from high to low
#   left accum   [xxxxxxxxxxxxxxxxxxxxxxxxx------)
#                drain overlap -> arrive overlap_bar
#
# Tile 1 MMA waits for Tile 0 epilogue to release the overlap barrier, then writes the right accumulator:
#
#   left accum   [xxxxxxxxxxxxxxxxxxxxxxxxx------)
#   right accum                            [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)
#                                          wait overlap_bar -> MMA
#
# Tile 1 epilogue drains the overlap first and releases the overlap barrier for Tile 2:
#
#   left accum   [xxxxxxxxxxxxxxxxxxx------------)
#   right accum                            [-----xxxxxxxxxxxxxxxxxxxxxxxxxx)
#                                          drain from low to high -------->
#                                          drain overlap -> arrive overlap_bar
#
# The overlap barrier is now released for Tile 2, but Tile 2 MMA must wait for Tile 0 epilogue
# to fully drain the left accumulator. The legacy acc_empty / acc_ready barriers are used to
# communicate empty / full accumulator semantics between MMA and epilogue partitions.
#
# yapf: enable


class TmemOverlapStoragePlan:

    def __init__(self, left_accum_offset, left_accum_cols, right_accum_offset, right_accum_cols, a_scale_offset,
                 a_scale_cols, b_scale_offset, b_scale_cols, early_release_subtile):
        self.left_accum_offset = left_accum_offset
        self.left_accum_cols = left_accum_cols
        self.right_accum_offset = right_accum_offset
        self.right_accum_cols = right_accum_cols
        self.a_scale_offset = a_scale_offset
        self.a_scale_cols = a_scale_cols
        self.b_scale_offset = b_scale_offset
        self.b_scale_cols = b_scale_cols
        self.early_release_subtile = early_release_subtile


@gluon.constexpr_function
def tmem_overlap_storage_plan(BLOCK_K, EPILOGUE_BLOCK_N, VEC_SIZE, USE_OVERLAP: gl.constexpr):
    # Hard code BLOCK_N to 256 for this example.
    BLOCK_N: gl.constexpr = 256

    # Scale views have K / VEC_SIZE columns. A scales are M-sharded
    # across CTAs, so each CTA stores one TMEM column per scale-K column.
    # B scales are duplicated across CTAs by the scale layout; per CTA,
    # they store one scale-K column group for each 128-column N block.
    scale_k = BLOCK_K // VEC_SIZE
    a_scale_cols = scale_k
    b_scale_cols = scale_k * (BLOCK_N // 128)

    # The overlap path uses double buffered BLOCK_N=256 accumulators.
    if USE_OVERLAP:
        # The overlap is the combined size of the A and B scales
        # which cannot exceed the size of BLOCK_N.
        overlap_cols = a_scale_cols + b_scale_cols
        assert overlap_cols < BLOCK_N

        # The left accumulator begins at 0 and ends at BLOCK_N.
        # The right accumulator begins at BLOCK_N minus the overlap.
        right_accum_offset = BLOCK_N - overlap_cols

        # Scale views begin after the overlapping accumulators.
        a_scale_offset = TMEM_COLS - overlap_cols

        # Identify the zero-based index of the epilogue subtile containing the last overlap column.
        early_release_subtile = (overlap_cols - 1) // EPILOGUE_BLOCK_N

    # The comparison path uses a single buffered BLOCK_N=256 accumulator.
    else:
        right_accum_offset = 0
        a_scale_offset = BLOCK_N
        early_release_subtile = 0

    return TmemOverlapStoragePlan(0, BLOCK_N, right_accum_offset, BLOCK_N, a_scale_offset, a_scale_cols,
                                  a_scale_offset + a_scale_cols, b_scale_cols, early_release_subtile)


@gluon.constexpr_function
def tmem_overlap_storage_plan_from_partition(p):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if p.a_desc.dtype == gl.uint8 else 1
    BLOCK_N: gl.constexpr = p.b_desc.block_shape[0]
    assert BLOCK_N == 256
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    EPILOGUE_BLOCK_N: gl.constexpr = p.c_desc.block_shape[1]
    VEC_SIZE: gl.constexpr = 32 if p.a_scale_desc.dtype == gl.uint8 else 16
    return tmem_overlap_storage_plan(BLOCK_K, EPILOGUE_BLOCK_N, VEC_SIZE, p.USE_OVERLAP)


def test_tmem_overlap_accum():
    plan = tmem_overlap_storage_plan(BLOCK_K=256, EPILOGUE_BLOCK_N=32, VEC_SIZE=16, USE_OVERLAP=True)
    assert (plan.left_accum_offset, plan.left_accum_cols) == (0, 256)
    assert (plan.right_accum_offset, plan.right_accum_cols) == (208, 256)
    assert (plan.a_scale_offset, plan.a_scale_cols) == (464, 16)
    assert (plan.b_scale_offset, plan.b_scale_cols) == (480, 32)
    assert plan.early_release_subtile == 1

    plan = tmem_overlap_storage_plan(BLOCK_K=128, EPILOGUE_BLOCK_N=32, VEC_SIZE=32, USE_OVERLAP=True)
    assert (plan.right_accum_offset, plan.right_accum_cols) == (244, 256)
    assert (plan.a_scale_offset, plan.a_scale_cols) == (500, 4)
    assert (plan.b_scale_offset, plan.b_scale_cols) == (504, 8)
    assert plan.early_release_subtile == 0

    plan = tmem_overlap_storage_plan(BLOCK_K=256, EPILOGUE_BLOCK_N=32, VEC_SIZE=16, USE_OVERLAP=False)
    assert (plan.left_accum_offset, plan.left_accum_cols) == (0, 256)
    assert (plan.right_accum_offset, plan.right_accum_cols) == (0, 256)
    assert (plan.a_scale_offset, plan.a_scale_cols) == (256, 16)
    assert (plan.b_scale_offset, plan.b_scale_cols) == (272, 32)


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


@gluon.jit
def unswizzle_scales_shared_memory(smem, BLOCK_MN: gl.constexpr, BLOCK_K: gl.constexpr, VEC_SIZE: gl.constexpr):
    smem = smem.reshape((smem.shape[1], smem.shape[2], 32, 4, 4))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((BLOCK_MN, BLOCK_K // VEC_SIZE))


@gluon.jit
def async_mma_scaled_impl(a_smem, b_smem, a_scale_smem, b_scale_smem, acc_tmem, tmem_pool, tmem_plan, mma_bar,
                          use_acc, pred):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_smem.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = a_smem.shape[0]
    BLOCK_N: gl.constexpr = b_smem.shape[0]
    BLOCK_K: gl.constexpr = a_smem.shape[1] * A_ELEM_PER_BYTE
    # Recall we use `uint8` to represent fp4 elements.
    VEC_SIZE: gl.constexpr = 32 if a_scale_smem.dtype == gl.uint8 else 16

    a_scale = unswizzle_scales_shared_memory(a_scale_smem, BLOCK_M, BLOCK_K, VEC_SIZE)
    b_scale = unswizzle_scales_shared_memory(b_scale_smem, BLOCK_N, BLOCK_K, VEC_SIZE)

    two_ctas: gl.constexpr = acc_tmem.type.layout.two_ctas
    a_scale_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=[[1, 0]] if two_ctas else [])
    b_scale_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=[[0, 0]] if two_ctas else [])
    a_scale_tmem = tmem_pool.slice(tmem_plan.a_scale_offset,
                                   tmem_plan.a_scale_cols)._reinterpret(dtype=a_scale.dtype, shape=a_scale.shape,
                                                                          layout=a_scale_layout)
    b_scale_tmem = tmem_pool.slice(tmem_plan.b_scale_offset,
                                   tmem_plan.b_scale_cols)._reinterpret(dtype=b_scale.dtype, shape=b_scale.shape,
                                                                          layout=b_scale_layout)

    tcgen05_copy(a_scale, a_scale_tmem)
    tcgen05_copy(b_scale, b_scale_tmem)

    a_format: gl.constexpr = "e2m1" if a_smem.dtype == gl.uint8 else "e4m3"
    b_format: gl.constexpr = "e2m1" if b_smem.dtype == gl.uint8 else "e4m3"
    tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                       use_acc=use_acc, pred=pred, multicast=True, mbarriers=[mma_bar])


@gluon.jit
def issue_loads(producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs,
                b_scale_bufs, bars, pred):
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
    tma.async_load(a_desc, [off_m, off_k_a], bar, a_bufs.index(index), pred, multicast=True)
    tma.async_load(b_desc, [off_n, off_k_b], bar, b_bufs.index(index), pred, multicast=True)
    tma.async_load(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], bar, a_scale_bufs.index(index), pred,
                   multicast=True)
    tma.async_load(b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], bar, b_scale_bufs.index(index), pred,
                   multicast=True)
    return producer.next(pred)


@gluon.jit
def issue_mma(consumer, c_bars, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs, producer, p_bars, acc_tmem, tmem_pool,
              tmem_plan, use_acc, pred):
    c_index = consumer.index
    mbarrier.wait(c_bars.index(c_index), consumer.phase, pred)
    async_mma_scaled_impl(a_bufs.index(c_index), b_bufs.index(c_index), a_scale_bufs.index(c_index),
                          b_scale_bufs.index(c_index), acc_tmem, tmem_pool, tmem_plan, p_bars.index(producer.index),
                          use_acc, pred)
    return consumer.next(pred), producer.next(pred)


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
class SpsTileScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    pid_m: gl.tensor
    pid_n: gl.tensor
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    NUM_PID_M: gl.tensor
    NUM_PID_N: gl.tensor

    @gluon.jit
    def initialize(TILE_M: gl.constexpr, TILE_N: gl.constexpr, NUM_PID_M, NUM_PID_N):
        tile_id = gl.program_id(axis=0)
        has_work = tile_id < NUM_PID_M * NUM_PID_N
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            pid_m = tile_id // NUM_PID_N
            pid_n = tile_id % NUM_PID_N
        return SpsTileScheduler(has_work, tile_id, pid_m, pid_n, TILE_M, TILE_N, NUM_PID_M, NUM_PID_N)

    @gluon.jit
    def get_offsets(self):
        return self.pid_m * self.TILE_M, self.pid_n * self.TILE_N

    @gluon.jit
    def step(self):
        next_tile_id = self.tile_id + gl.num_programs(axis=0)
        has_work = next_tile_id < self.NUM_PID_M * self.NUM_PID_N
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            pid_m = next_tile_id // self.NUM_PID_N
            pid_n = next_tile_id % self.NUM_PID_N
        return SpsTileScheduler(has_work, next_tile_id, pid_m, pid_n, self.TILE_M, self.TILE_N, self.NUM_PID_M,
                                self.NUM_PID_N)


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
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    overlap_bar: gl.shared_memory_descriptor
    NUM_PID_M: gl.tensor
    NUM_PID_N: gl.tensor
    USE_OVERLAP: gl.constexpr

    @gluon.jit
    def get_sps_scheduler(self):
        return SpsTileScheduler.initialize(
            self.a_desc.block_shape[0],
            self.b_desc.block_shape[0],
            self.NUM_PID_M,
            self.NUM_PID_N,
        )


@gluon.jit
def mma_scaled_load_partition(p):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if p.a_desc.dtype == gl.uint8 else 1
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    K = p.a_desc.shape[1] * A_ELEM_PER_BYTE
    state = Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = p.get_sps_scheduler()
    while scheduler.has_work:
        for k in range(0, K, BLOCK_K):
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            state = issue_loads(state, scheduler.pid_m, scheduler.pid_n, k, p.a_desc, p.b_desc, p.a_scale_desc,
                                p.b_scale_desc, p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs, p.load_ready_bars,
                                pred=True)
        scheduler = scheduler.step()


@gluon.jit
def mma_scaled_mma_partition(p):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if p.a_desc.dtype == gl.uint8 else 1
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    K = p.a_desc.shape[1] * A_ELEM_PER_BYTE
    tmem_plan: gl.constexpr = tmem_overlap_storage_plan_from_partition(p)
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        # The overlap barrier only protects the shared columns. Before reusing an
        # accumulator window, wait until its epilogue has drained the entire window.
        # For example, Tile 2 cannot reuse the left window until Tile 0 releases it.
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)

        # MMA must wait for the preceding epilogue to drain the overlap.
        if p.USE_OVERLAP:
            mbarrier.wait(p.overlap_bar.index(0), (i + 1) % 2)

        # When overlap is disabled, always use the left (only) accumulator.
        # When overlap is enabled, even tiles use the left and odd tiles use the right accumulator.
        if p.USE_OVERLAP and i % 2 != 0:
            acc_tmem = p.acc_bufs.slice(tmem_plan.right_accum_offset, tmem_plan.right_accum_cols)
        else:
            acc_tmem = p.acc_bufs.slice(tmem_plan.left_accum_offset, tmem_plan.left_accum_cols)

        use_acc = False
        for k in range(0, K, BLOCK_K):
            _, load_state = issue_mma(load_state, p.load_ready_bars, p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs,
                                      load_state, p.load_empty_bars, acc_tmem, p.acc_bufs, tmem_plan, use_acc,
                                      pred=True)
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()

        scheduler = scheduler.step()
        i += 1


@gluon.jit
def mma_scaled_epilogue_partition(p):
    tile_m: gl.constexpr = p.c_desc.block_shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_shape[0]
    EPILOGUE_BLOCK_N: gl.constexpr = p.c_desc.block_shape[1]
    tmem_plan: gl.constexpr = tmem_overlap_storage_plan_from_partition(p)
    subtile_factor: gl.constexpr = BLOCK_N // EPILOGUE_BLOCK_N
    subtile_stages: gl.constexpr = 1 if subtile_factor == 1 else 2
    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    acc_smems = gl.allocate_shared_memory(p.c_desc.dtype, [subtile_stages, tile_m, EPILOGUE_BLOCK_N], p.c_desc.layout)
    sub_acc_state = Counter.create(0, subtile_stages)
    scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)

        # When overlap is disabled, always use the left (only) accumulator.
        # When overlap is enabled, even tiles use the left and odd tiles use the right accumulator.
        if p.USE_OVERLAP and i % 2 != 0:
            acc_tmem = p.acc_bufs.slice(tmem_plan.right_accum_offset, tmem_plan.right_accum_cols)
        else:
            acc_tmem = p.acc_bufs.slice(tmem_plan.left_accum_offset, tmem_plan.left_accum_cols)

        for s in gl.static_range(subtile_factor):
            # When overlap is disabled, always drain subtiles from low-to-high.
            # When overlap is enabled, drain the overlap first:
            #   Even tiles use the left accumulator and drain high-to-low.
            #   Odd tiles use the right accumulator and drain low-to-high.
            if p.USE_OVERLAP and i % 2 == 0:
                acc_sub = acc_tmem.slice(EPILOGUE_BLOCK_N * (subtile_factor - 1 - s), EPILOGUE_BLOCK_N)
                store_n = off_n + EPILOGUE_BLOCK_N * (subtile_factor - 1 - s)
            else:
                acc_sub = acc_tmem.slice(EPILOGUE_BLOCK_N * s, EPILOGUE_BLOCK_N)
                store_n = off_n + EPILOGUE_BLOCK_N * s

            acc_smem = acc_smems.index(sub_acc_state.index)
            acc = acc_sub.load().to(p.c_desc.dtype)

            # Signal the barrier once the epilogue drains the overlap so the next MMA can proceed.
            if p.USE_OVERLAP and s == tmem_plan.early_release_subtile:
                mbarrier.arrive(p.overlap_bar.index(0), count=1)

            tma.store_wait(pendings=subtile_stages - 1)
            acc_smem.store(acc)
            tma.async_copy_shared_to_global(p.c_desc, [off_m, store_n], acc_smem)
            sub_acc_state = sub_acc_state.next()
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()
        scheduler = scheduler.step()
        i += 1
    tma.store_wait(0)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@gluon.jit
def mma_scaled_warp_specialized_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, M, N, K, A_ELEM_PER_BYTE,
                                       num_buffers: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                       BLOCK_K: gl.constexpr, EPILOGUE_BLOCK_N: gl.constexpr,
                                       CGA_LAYOUT: gl.constexpr, USE_OVERLAP: gl.constexpr):
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    gl.static_assert(BLOCK_M_PER_CTA == 64 or BLOCK_M_PER_CTA == 128)
    num_acc_buffers: gl.constexpr = 2 if USE_OVERLAP else 1

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_shape,
                                             b_scale_desc.layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M_PER_CTA, BLOCK_N], col_stride=1, cga_layout=CGA_LAYOUT,
                                                   two_ctas=TWO_CTAS)

    # Allocate all of TMEM and then slice it (later) to accommodate accumulators and scales.
    acc_bufs = allocate_tensor_memory(gl.float32, [BLOCK_M, TMEM_COLS], tmem_layout)
    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0),
         a_scale_bufs.index(0), b_scale_bufs.index(0)], multicast=True, two_ctas=TWO_CTAS)

    load_empty_bars = mbarrier.allocate_mbarrier(batch=num_buffers)
    load_ready_bars = mbarrier.allocate_mbarrier(batch=num_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready_bars.index(i), count=1)

    acc_empty_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers, two_ctas=TWO_CTAS)
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)
    overlap_bar = mbarrier.allocate_mbarrier(batch=1, two_ctas=TWO_CTAS)
    mbarrier.init(overlap_bar.index(0), count=1)

    p = PartitionArgs(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                      load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars, overlap_bar,
                      gl.cdiv(M, BLOCK_M), gl.cdiv(N, BLOCK_N), USE_OVERLAP)

    gl.warp_specialize([
        (mma_scaled_epilogue_partition, (p, )),
        (mma_scaled_mma_partition, (p, )),
        (mma_scaled_load_partition, (p, )),
    ], [1, 1], [24, 24])


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def make_descriptors(A, B, A_scale, B_scale, M, N):
    """Create TMA descriptors for the fixed example configuration."""
    a_is_fp4 = A.dtype == torch.uint8
    b_is_fp4 = B.dtype == torch.uint8
    mixed_prec = a_is_fp4 != b_is_fp4
    a_elem_per_byte = 2 if a_is_fp4 else 1
    b_elem_per_byte = 2 if b_is_fp4 else 1

    BLOCK_K = 128 if torch.float8_e4m3fn in [A.dtype, B.dtype] else 256
    a_block = [BLOCK_M, BLOCK_K // a_elem_per_byte]
    b_block = [BLOCK_N, BLOCK_K // b_elem_per_byte]
    c_block = [BLOCK_M, EPILOGUE_BLOCK_N]

    cga = tuple(tuple(x) for x in CGA_LAYOUT)
    a_layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.uint8 if a_is_fp4 else gl.float8e4nv,
                                                    fp4_padded=(mixed_prec and a_is_fp4), cga_layout=cga)
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block, gl.uint8 if b_is_fp4 else gl.float8e4nv,
                                                    fp4_padded=(mixed_prec and b_is_fp4), cga_layout=cga)
    a_desc = TensorDescriptor.from_tensor(A, a_block, a_layout)
    b_desc = TensorDescriptor.from_tensor(B, b_block, b_layout)

    C = torch.empty(M, N, device="cuda", dtype=OUT_DTYPE)
    C_dtype = getattr(gl, str(OUT_DTYPE).split('.')[1])
    c_layout = gl.NVMMASharedLayout.get_default_for(c_block, C_dtype, cga_layout=cga)
    c_desc = TensorDescriptor.from_tensor(C, c_block, c_layout)

    is_nvfp4 = A_scale.dtype == torch.float8_e4m3fn
    vec_size = 16 if is_nvfp4 else 32
    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // (vec_size * 4)
    a_scale_block = [1, rep_m, rep_k, 2, 256]
    b_scale_block = [1, rep_n, rep_k, 2, 256]

    cga_a_scale = [[0, 1, 0, 0, 0]]
    cga_b_scale = [[0, 0, 0, 0, 0]]
    a_scale_layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5, cga_layout=cga_a_scale)
    b_scale_layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5, cga_layout=cga_b_scale)

    A_scale_5d = A_scale.reshape(1, A_scale.shape[0], A_scale.shape[1], 2, 256)
    B_scale_5d = B_scale.reshape(1, B_scale.shape[0], B_scale.shape[1], 2, 256)
    a_scale_desc = TensorDescriptor.from_tensor(A_scale_5d, a_scale_block, a_scale_layout)
    b_scale_desc = TensorDescriptor.from_tensor(B_scale_5d, b_scale_block, b_scale_layout)

    return a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, BLOCK_K


def mma_scaled_warp_specialized(A, B, A_scale, B_scale, use_overlap=True):
    """Warp-specialized block-scale MMA with optional overlapping accumulators."""
    M, N = A.shape[0], B.shape[0]
    A_ELEM_PER_BYTE = 2 if A.dtype == torch.uint8 else 1
    B_ELEM_PER_BYTE = 2 if B.dtype == torch.uint8 else 1
    K = A.shape[1] * A_ELEM_PER_BYTE
    assert K == B.shape[1] * B_ELEM_PER_BYTE
    A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc, BLOCK_K = make_descriptors(A, B, A_scale, B_scale, M, N)

    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    sm_count = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid = (max(1, min(num_pid, sm_count // NUM_CTAS)), )
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
        NUM_BUFFERS,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EPILOGUE_BLOCK_N,
        CGA_LAYOUT,
        use_overlap,
        num_ctas=NUM_CTAS,
    )
    return C_desc.base


def make_problem(M, N, K, a_format, b_format):
    A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
    B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
    A_scale = swizzle_scales_packed_block(A_scale)
    B_scale = swizzle_scales_packed_block(B_scale)
    return A, B, A_scale, B_scale, A_ref @ B_ref.T


def benchmark(M, N, K, a_format, b_format, use_overlap):
    A, B, A_scale, B_scale, _ = make_problem(M, N, K, a_format, b_format)

    def run():
        return mma_scaled_warp_specialized(A, B, A_scale, B_scale, use_overlap=use_overlap)

    ms = triton.testing.do_bench_cudagraph(run)
    return 2.0 * M * N * K * 1.0e-9 / ms


@pytest.mark.parametrize("a_format, b_format", FORMATS)
@pytest.mark.parametrize("use_overlap", [False, True])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_mma_scaled_overlap_accumulator(a_format, b_format, use_overlap):
    torch.manual_seed(0)
    A, B, A_scale, B_scale, C_ref = make_problem(8192, 8192, 8192, a_format, b_format)
    C = mma_scaled_warp_specialized(A, B, A_scale, B_scale, use_overlap=use_overlap)
    torch.testing.assert_close(C_ref, C.to(torch.float32), atol=1e-3, rtol=1e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    args = parser.parse_args()

    print("format            K        off         on    speedup")
    for a_format, b_format in FORMATS:
        label = f"{a_format}-{b_format}"
        for K in args.K:
            torch.manual_seed(0)
            off = benchmark(args.M, args.N, K, a_format, b_format, use_overlap=False)
            torch.manual_seed(0)
            on = benchmark(args.M, args.N, K, a_format, b_format, use_overlap=True)
            speedup = (on / off - 1.0) * 100.0
            print(f"{label:14s} {K:7d} {off:9.1f} {on:9.1f} {speedup:8.2f}%")
        print()


if __name__ == "__main__":
    main()
