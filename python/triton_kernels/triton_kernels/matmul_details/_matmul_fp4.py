"""Shared kernels for routed MXFP8 x MXFP4 matmul.

Two-CTA ragged FP8 x FP4 matmul, with overlapped TMEM accumulators.

Adapted from Triton python/examples/gluon/06-overlapping-accumulator.py
_at 18e7b43799a848c2bc12496e3f022e6d01978194. The ragged scheduler,
scale packing, bias epilogue and masked tail stores are local additions.

Copyright 2018-2020 Philippe Tillet; Copyright 2020-2022 OpenAI.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to
do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma_barrier_count,
    tcgen05_mma_scaled,
    tensor_memory_descriptor,
    tma,
)
from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor
from triton_kernels.matmul_details.opt_flags import _get_idle_sms, _get_opt_flags_constraints
from triton_kernels.tensor import FP4, RaggedTensorMetadata, Tensor
from triton_kernels.tensor_details.layout import (
    BlackwellActMXScaleLayout,
    BlackwellMXScaleLayout,
    BlackwellMXValueLayout,
    StridedLayout,
)

if TYPE_CHECKING:
    from triton_kernels.matmul import Epilogue, FusedActivation, FusedComm, PrecisionConfig

TMEM_COLS = gl.constexpr(512)


@triton.jit
def pack_routed_fp4_scales(
    X,
    Y,
    Idx,
    Sizes,
    Starts,
    BlockOffs,
    Schedule,
    E: tl.constexpr,
    K4: tl.constexpr,
    KS: tl.constexpr,
    KSTRIDE: tl.constexpr,
) -> None:
    block = tl.program_id(0)
    col = tl.program_id(1)
    valid = block < tl.load(BlockOffs + E)
    packed = tl.load(Schedule + block, valid, other=0).to(tl.uint32)
    expert = packed & 65535
    local = packed >> 16
    start = tl.load(Starts + expert, valid, other=0)
    size = tl.load(Sizes + expert, valid, other=0)
    row = tl.arange(0, 128)
    k = tl.arange(0, 4)
    live = valid & (local * 128 + row < size)
    if Idx is not None:
        source = tl.load(Idx + start + local * 128 + row, live, other=-1)
    else:
        source = start + local * 128 + row
    values = tl.load(
        X + source[:, None].to(tl.int64) * KS + (col * 4 + k[None, :]) * KSTRIDE,
        live[:, None] & (source[:, None] >= 0),
        other=0,
    )
    order = (row % 32) * 16 + (row // 32) * 4
    tl.store(Y + (block.to(tl.int64) * K4 + col) * 512 + order[:, None] + k[None, :], values)


class TmemOverlapStoragePlan:

    def __init__(
        self,
        left_accum_offset,
        left_accum_cols,
        right_accum_offset,
        right_accum_cols,
        a_scale_offset,
        a_scale_cols,
        b_scale_offset,
        b_scale_cols,
        early_release_subtile,
    ) -> None:
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
def _tmem_overlap_storage_plan(BLOCK_K, EPILOGUE_BLOCK_N, VEC_SIZE,
                               USE_OVERLAP: gl.constexpr) -> TmemOverlapStoragePlan:
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

        # The left accumulator begins _at 0 and ends _at BLOCK_N.
        # The right accumulator begins _at BLOCK_N minus the overlap.
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

    return TmemOverlapStoragePlan(
        0,
        BLOCK_N,
        right_accum_offset,
        BLOCK_N,
        a_scale_offset,
        a_scale_cols,
        a_scale_offset + a_scale_cols,
        b_scale_cols,
        early_release_subtile,
    )


@gluon.constexpr_function
def _tmem_overlap_storage_plan_from_partition(p) -> TmemOverlapStoragePlan:
    A_ELEM_PER_BYTE: gl.constexpr = 2 if p.a_desc.dtype == gl.uint8 else 1
    BLOCK_N: gl.constexpr = p.b_desc.block_shape[0]
    assert BLOCK_N == 256
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    EPILOGUE_BLOCK_N: gl.constexpr = p.c_desc.block_shape[1]
    VEC_SIZE: gl.constexpr = 32 if p.a_scale_desc.dtype == gl.uint8 else 16
    return _tmem_overlap_storage_plan(BLOCK_K, EPILOGUE_BLOCK_N, VEC_SIZE, p.USE_OVERLAP)


@gluon.jit
def _unswizzle_scales_shared_memory(smem, BLOCK_MN: gl.constexpr, BLOCK_K: gl.constexpr,
                                    VEC_SIZE: gl.constexpr) -> gl.shared_memory_descriptor:
    smem = smem.reshape((smem.shape[1], smem.shape[2], 32, 4, 4))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((BLOCK_MN, BLOCK_K // VEC_SIZE))


@gluon.jit
def _async_mma_scaled_impl(
    a_smem,
    b_smem,
    a_scale_smem,
    b_scale_smem,
    acc_tmem,
    tmem_pool,
    tmem_plan,
    mma_bar,
    use_acc,
    pred,
) -> None:
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_smem.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = a_smem.shape[0]
    BLOCK_N: gl.constexpr = b_smem.shape[0]
    BLOCK_K: gl.constexpr = a_smem.shape[1] * A_ELEM_PER_BYTE
    # Recall we use `uint8` to represent fp4 elements.
    VEC_SIZE: gl.constexpr = 32 if a_scale_smem.dtype == gl.uint8 else 16

    a_scale = _unswizzle_scales_shared_memory(a_scale_smem, BLOCK_M, BLOCK_K, VEC_SIZE)
    b_scale = _unswizzle_scales_shared_memory(b_scale_smem, BLOCK_N, BLOCK_K, VEC_SIZE)

    two_ctas: gl.constexpr = acc_tmem.type.layout.two_ctas
    a_scale_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=[[1, 0]] if two_ctas else [])
    b_scale_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=[[0, 0]] if two_ctas else [])
    a_scale_tmem = tmem_pool.slice(tmem_plan.a_scale_offset,
                                   tmem_plan.a_scale_cols).reinterpret(dtype=a_scale.dtype, shape=a_scale.shape,
                                                                       layout=a_scale_layout)
    b_scale_tmem = tmem_pool.slice(tmem_plan.b_scale_offset,
                                   tmem_plan.b_scale_cols).reinterpret(dtype=b_scale.dtype, shape=b_scale.shape,
                                                                       layout=b_scale_layout)

    tcgen05_copy(a_scale, a_scale_tmem)
    tcgen05_copy(b_scale, b_scale_tmem)

    a_format: gl.constexpr = "e2m1" if a_smem.dtype == gl.uint8 else "e4m3"
    b_format: gl.constexpr = "e2m1" if b_smem.dtype == gl.uint8 else "e4m3"
    tcgen05_mma_scaled(
        a_smem,
        b_smem.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_tmem,
        a_format,
        b_format,
        use_acc=use_acc,
        pred=pred,
        multicast=True,
        mbarriers=[mma_bar],
    )


@gluon.jit
def _issue_loads(
    producer,
    pid_n,
    k,
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    a_bufs,
    b_bufs,
    a_scale_bufs,
    b_scale_bufs,
    bars,
    off_m,
    row_end,
    expert,
    scale_m,
    gather_ptr,
    N,
    pred,
) -> "Counter":
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if b_desc.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = b_desc.block_shape[0]
    BLOCK_K: gl.constexpr = a_desc.block_shape[1] * A_ELEM_PER_BYTE
    B_REP_K: gl.constexpr = b_scale_desc.block_shape[2]

    off_n = expert * N + pid_n * BLOCK_N
    off_n_b_scale = off_n // 128
    off_k_a = k // A_ELEM_PER_BYTE
    off_k_b = k // B_ELEM_PER_BYTE
    off_k_b_scale = (k // BLOCK_K) * B_REP_K

    index = producer.index
    bar = bars.index(index)
    mbarrier.expect(
        bar,
        (BLOCK_M * BLOCK_K // 2) + b_desc.nbytes_per_cta + a_scale_desc.nbytes_per_cta + b_scale_desc.nbytes_per_cta,
        pred,
    )
    if gather_ptr is not None:
        index_layout: gl.constexpr = gl.SliceLayout(
            0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0], cga_layout=((0, 1), )))
        rows = off_m + gl.arange(0, 256, layout=index_layout)
        source_rows = gl.load(gather_ptr + rows, rows < row_end, other=-1)
        tma.async_gather(a_desc, source_rows, off_k_a, bar, a_bufs.index(index), pred, multicast=True)
    else:
        tma.async_load(a_desc, [off_m, off_k_a], bar, a_bufs.index(index), pred, multicast=True)
    tma.async_load(b_desc, [off_n, off_k_b], bar, b_bufs.index(index), pred, multicast=True)
    tma.async_load(
        a_scale_desc,
        [0, scale_m, k // 128, 0, 0],
        bar,
        a_scale_bufs.index(index),
        pred,
        multicast=True,
    )
    tma.async_load(
        b_scale_desc,
        [0, off_n_b_scale, off_k_b_scale, 0, 0],
        bar,
        b_scale_bufs.index(index),
        pred,
        multicast=True,
    )
    return producer._next(pred)


@gluon.jit
def _issue_mma(
    consumer,
    c_bars,
    a_bufs,
    b_bufs,
    a_scale_bufs,
    b_scale_bufs,
    producer,
    p_bars,
    acc_tmem,
    tmem_pool,
    tmem_plan,
    use_acc,
    pred,
) -> tuple["Counter", "Counter"]:
    c_index = consumer.index
    mbarrier.wait(c_bars.index(c_index), consumer.phase, pred)
    _async_mma_scaled_impl(
        a_bufs.index(c_index),
        b_bufs.index(c_index),
        a_scale_bufs.index(c_index),
        b_scale_bufs.index(c_index),
        acc_tmem,
        tmem_pool,
        tmem_plan,
        p_bars.index(producer.index),
        use_acc,
        pred,
    )
    return consumer._next(pred), producer._next(pred)


@gluon.aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.jit
    def _create(phase, num_barriers: gl.constexpr) -> "Counter":
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def _next(self, pred=True) -> "Counter":
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
    expert: gl.tensor
    off_m: gl.tensor
    row_end: gl.tensor
    scale_m: gl.tensor
    NUM_PID_M: gl.tensor
    NUM_PID_N: gl.tensor
    schedule: gl.tensor
    slice_offs: gl.tensor
    slice_sizes: gl.tensor
    scale_offs: gl.tensor

    @gluon.jit
    def _at(tile_id, NUM_PID_M, NUM_PID_N, schedule, slice_offs, slice_sizes, scale_offs) -> "SpsTileScheduler":
        # Reuse the public 128-row schedule. Each even local block consumes
        # two adjacent blocks; odd blocks are skipped rather than repacked.
        packed = gl.load(schedule + tile_id // NUM_PID_N, tile_id < NUM_PID_M * NUM_PID_N, other=-1).to(gl.uint32)
        valid = (packed != 0xFFFFFFFF) & (((packed >> 16) & 1) == 0)
        while (tile_id < NUM_PID_M * NUM_PID_N) & ~valid:
            tile_id += gl.num_programs(axis=0)
            packed = gl.load(schedule + tile_id // NUM_PID_N, tile_id < NUM_PID_M * NUM_PID_N, other=-1).to(gl.uint32)
            valid = (packed != 0xFFFFFFFF) & (((packed >> 16) & 1) == 0)
        has_work = tile_id < NUM_PID_M * NUM_PID_N
        expert = (packed & 65535).to(gl.int32)
        block = (packed >> 16).to(gl.int32)
        start = gl.load(slice_offs + expert, has_work, other=0)
        size = gl.load(slice_sizes + expert, has_work, other=0)
        scale_start = gl.load(scale_offs + expert, has_work, other=0)
        return SpsTileScheduler(
            has_work,
            tile_id,
            tile_id // NUM_PID_N,
            tile_id % NUM_PID_N,
            expert,
            start + block * 128,
            start + size,
            scale_start + block,
            NUM_PID_M,
            NUM_PID_N,
            schedule,
            slice_offs,
            slice_sizes,
            scale_offs,
        )

    @gluon.jit
    def _initialize(NUM_PID_M, NUM_PID_N, schedule, slice_offs, slice_sizes, scale_offs) -> "SpsTileScheduler":
        return SpsTileScheduler._at(gl.program_id(0), NUM_PID_M, NUM_PID_N, schedule, slice_offs, slice_sizes,
                                    scale_offs)

    @gluon.jit
    def _get_offsets(self) -> tuple[gl.tensor, gl.tensor]:
        return self.off_m, self.pid_n * 256

    @gluon.jit
    def _step(self) -> "SpsTileScheduler":
        return SpsTileScheduler._at(
            self.tile_id + gl.num_programs(0),
            self.NUM_PID_M,
            self.NUM_PID_N,
            self.schedule,
            self.slice_offs,
            self.slice_sizes,
            self.scale_offs,
        )


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
    schedule: gl.tensor
    slice_offs: gl.tensor
    slice_sizes: gl.tensor
    scale_offs: gl.tensor
    gather_ptr: gl.tensor | gl.constexpr
    out_ptr: gl.tensor
    bias_ptr: gl.tensor | gl.constexpr
    bias_stride: gl.tensor
    out_stride: gl.tensor
    beta_ptr: gl.tensor | gl.constexpr
    gamma_ptr: gl.tensor | gl.constexpr
    alpha: gl.tensor | gl.constexpr
    N: gl.tensor

    @gluon.jit
    def _get_sps_scheduler(self) -> "SpsTileScheduler":
        return SpsTileScheduler._initialize(
            self.NUM_PID_M,
            self.NUM_PID_N,
            self.schedule,
            self.slice_offs,
            self.slice_sizes,
            self.scale_offs,
        )


@gluon.jit
def _mma_scaled_load_partition(p) -> None:
    A_ELEM_PER_BYTE: gl.constexpr = 2 if p.a_desc.dtype == gl.uint8 else 1
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    K = p.a_desc.shape[1] * A_ELEM_PER_BYTE
    state = Counter._create(1, p.load_empty_bars.shape[0])
    scheduler = p._get_sps_scheduler()
    while scheduler.has_work:
        for k in range(0, K, BLOCK_K):
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            state = _issue_loads(
                state,
                scheduler.pid_n,
                k,
                p.a_desc,
                p.b_desc,
                p.a_scale_desc,
                p.b_scale_desc,
                p.a_bufs,
                p.b_bufs,
                p.a_scale_bufs,
                p.b_scale_bufs,
                p.load_ready_bars,
                scheduler.off_m,
                scheduler.row_end,
                scheduler.expert,
                scheduler.scale_m,
                p.gather_ptr,
                p.N,
                pred=True,
            )
        scheduler = scheduler._step()


@gluon.jit
def _mma_scaled_mma_partition(p) -> None:
    A_ELEM_PER_BYTE: gl.constexpr = 2 if p.a_desc.dtype == gl.uint8 else 1
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    K = p.a_desc.shape[1] * A_ELEM_PER_BYTE
    tmem_plan: gl.constexpr = _tmem_overlap_storage_plan_from_partition(p)
    load_state = Counter._create(0, p.load_empty_bars.shape[0])
    acc_state = Counter._create(1, p.acc_empty_bars.shape[0])
    scheduler = p._get_sps_scheduler()
    while scheduler.has_work:
        for parity in gl.static_range(2):
            if scheduler.has_work:
                # The overlap barrier only protects the shared columns. Before reusing an
                # accumulator window, wait until its epilogue has drained the entire window.
                # For example, Tile 2 cannot reuse the left window until Tile 0 releases it.
                mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)

                # MMA must wait for the preceding epilogue to drain the overlap.
                if p.USE_OVERLAP:
                    mbarrier.wait(p.overlap_bar.index(0), (parity + 1) % 2)

                # When overlap is disabled, always use the left (only) accumulator.
                # When overlap is enabled, even tiles use the left and odd tiles use the right accumulator.
                if p.USE_OVERLAP and parity != 0:
                    acc_tmem = p.acc_bufs.slice(tmem_plan.right_accum_offset, tmem_plan.right_accum_cols)
                else:
                    acc_tmem = p.acc_bufs.slice(tmem_plan.left_accum_offset, tmem_plan.left_accum_cols)

                use_acc = False
                for _k in range(0, K, BLOCK_K):
                    _, load_state = _issue_mma(
                        load_state,
                        p.load_ready_bars,
                        p.a_bufs,
                        p.b_bufs,
                        p.a_scale_bufs,
                        p.b_scale_bufs,
                        load_state,
                        p.load_empty_bars,
                        acc_tmem,
                        p.acc_bufs,
                        tmem_plan,
                        use_acc,
                        pred=True,
                    )
                    use_acc = True
                tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
                acc_state = acc_state._next()

                scheduler = scheduler._step()


@gluon.jit
def _mma_scaled_epilogue_partition(p) -> None:
    tile_m: gl.constexpr = p.c_desc.block_shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_shape[0]
    EPILOGUE_BLOCK_N: gl.constexpr = p.c_desc.block_shape[1]
    tmem_plan: gl.constexpr = _tmem_overlap_storage_plan_from_partition(p)
    subtile_factor: gl.constexpr = BLOCK_N // EPILOGUE_BLOCK_N
    subtile_stages: gl.constexpr = 1 if subtile_factor == 1 else 2
    acc_state = Counter._create(0, p.acc_empty_bars.shape[0])
    acc_smems = gl.allocate_shared_memory(p.c_desc.dtype, [subtile_stages, tile_m, EPILOGUE_BLOCK_N], p.c_desc.layout)
    sub_acc_state = Counter._create(0, subtile_stages)
    scheduler = p._get_sps_scheduler()
    while scheduler.has_work:
        for parity in gl.static_range(2):
            if scheduler.has_work:
                off_m, off_n = scheduler._get_offsets()
                mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)

                # When overlap is disabled, always use the left (only) accumulator.
                # When overlap is enabled, even tiles use the left and odd tiles use the right accumulator.
                if p.USE_OVERLAP and parity != 0:
                    acc_tmem = p.acc_bufs.slice(tmem_plan.right_accum_offset, tmem_plan.right_accum_cols)
                else:
                    acc_tmem = p.acc_bufs.slice(tmem_plan.left_accum_offset, tmem_plan.left_accum_cols)

                for s in gl.static_range(subtile_factor):
                    # When overlap is disabled, always drain subtiles from low-to-high.
                    # When overlap is enabled, drain the overlap first:
                    #   Even tiles use the left accumulator and drain high-to-low.
                    #   Odd tiles use the right accumulator and drain low-to-high.
                    if p.USE_OVERLAP and parity == 0:
                        acc_sub = acc_tmem.slice(EPILOGUE_BLOCK_N * (subtile_factor - 1 - s), EPILOGUE_BLOCK_N)
                        store_n = off_n + EPILOGUE_BLOCK_N * (subtile_factor - 1 - s)
                    else:
                        acc_sub = acc_tmem.slice(EPILOGUE_BLOCK_N * s, EPILOGUE_BLOCK_N)
                        store_n = off_n + EPILOGUE_BLOCK_N * s

                    acc_smem = acc_smems.index(sub_acc_state.index)
                    accumulator = acc_sub.load()
                    rows = off_m + gl.arange(0, tile_m, layout=gl.SliceLayout(1, accumulator.type.layout))
                    cols = store_n + gl.arange(0, EPILOGUE_BLOCK_N, layout=gl.SliceLayout(0, accumulator.type.layout))
                    if p.bias_ptr is not None:
                        bias = gl.load(
                            p.bias_ptr + scheduler.expert * p.bias_stride + cols,
                            cols < p.N,
                            other=0,
                        ).to(gl.float32)
                        if p.beta_ptr is not None:
                            beta = gl.load(p.beta_ptr + rows, rows < scheduler.row_end, other=0).to(gl.float32)
                            accumulator = gl.fma(bias[None, :], beta[:, None], accumulator)
                        else:
                            accumulator += bias[None, :]
                    if p.gamma_ptr is not None:
                        gamma = gl.load(p.gamma_ptr + rows, rows < scheduler.row_end, other=0).to(gl.float32)
                        accumulator *= gamma[:, None]
                    accumulator *= p.alpha
                    acc = accumulator.to(p.c_desc.dtype)

                    # Signal the barrier once the epilogue drains the overlap so the _next MMA can proceed.
                    if p.USE_OVERLAP and s == tmem_plan.early_release_subtile:
                        mbarrier.arrive(p.overlap_bar.index(0), count=1)

                    rows = off_m + gl.arange(0, tile_m, layout=gl.SliceLayout(1, acc.type.layout))
                    cols = store_n + gl.arange(0, EPILOGUE_BLOCK_N, layout=gl.SliceLayout(0, acc.type.layout))
                    if (off_m + tile_m <= scheduler.row_end) & (store_n + EPILOGUE_BLOCK_N <= p.N):
                        tma.store_wait(pendings=subtile_stages - 1)
                        acc_smem.store(acc)
                        tma.async_store(p.c_desc, [off_m, store_n], acc_smem)
                        sub_acc_state = sub_acc_state._next()
                    else:
                        gl.store(
                            p.out_ptr + rows[:, None].to(gl.int64) * p.out_stride + cols[None, :],
                            acc,
                            (rows[:, None] < scheduler.row_end) & (cols[None, :] < p.N),
                        )
                mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
                acc_state = acc_state._next()
                scheduler = scheduler._step()
    tma.store_wait(0)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@gluon.jit
def sparse_fp4_matmul_kernel(
    a_desc,
    b_desc,
    c_desc,
    a_scale_desc,
    b_scale_desc,
    N,
    num_buffers: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    CGA_LAYOUT: gl.constexpr,
    USE_OVERLAP: gl.constexpr,
    EXPERTS: gl.constexpr,
    schedule,
    slice_offs,
    slice_sizes,
    scale_offs,
    gather_ptr,
    out_ptr,
    bias_ptr,
    beta_ptr,
    gamma_ptr,
    alpha,
    bias_stride,
    out_stride,
) -> None:
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    gl.static_assert(BLOCK_M_PER_CTA == 64 or BLOCK_M_PER_CTA == 128)
    num_acc_buffers: gl.constexpr = 2 if USE_OVERLAP else 1

    # Gather's global descriptor is one replicated row. The selected rows in
    # shared memory, like a regular A tile, are split across the two CTAs.
    a_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2,
                                                  cga_layout=CGA_LAYOUT)
    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers, 256, BLOCK_K], a_layout)
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
         a_scale_bufs.index(0), b_scale_bufs.index(0)],
        multicast=True,
        two_ctas=TWO_CTAS,
    )

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

    p = PartitionArgs(
        a_desc,
        b_desc,
        c_desc,
        a_scale_desc,
        b_scale_desc,
        a_bufs,
        b_bufs,
        a_scale_bufs,
        b_scale_bufs,
        load_empty_bars,
        load_ready_bars,
        acc_bufs,
        acc_empty_bars,
        acc_ready_bars,
        overlap_bar,
        gl.load(scale_offs + EXPERTS),
        gl.cdiv(N, BLOCK_N),
        USE_OVERLAP,
        schedule,
        slice_offs,
        slice_sizes,
        scale_offs,
        gather_ptr,
        out_ptr,
        bias_ptr,
        bias_stride,
        out_stride,
        beta_ptr,
        gamma_ptr,
        alpha,
        N,
    )

    gl.warp_specialize(
        [
            (_mma_scaled_epilogue_partition, (p, )),
            (_mma_scaled_mma_partition, (p, )),
            (_mma_scaled_load_partition, (p, )),
        ],
        [1, 4],
        [24, 128],
    )


def matmul_fp4(
    a: torch.Tensor | Tensor,
    b: torch.Tensor | Tensor,
    bias: torch.Tensor | None,
    a_ragged_metadata: RaggedTensorMetadata | None,
    b_ragged_metadata: RaggedTensorMetadata | None,
    gather_indx: torch.Tensor | None,
    scatter_indx: torch.Tensor | None,
    precision_config: PrecisionConfig | None,
    betas: torch.Tensor | None,
    gammas: torch.Tensor | None,
    out_alpha: float | None,
    c: torch.Tensor | None,
    fused_comm: FusedComm | None,
    fused_activation: FusedActivation | None,
    epilogue: Epilogue | None,
    c_acc_in: torch.Tensor | None,
) -> torch.Tensor | None:
    pc = precision_config
    if not (isinstance(a, torch.Tensor) and isinstance(b, Tensor) and a.is_cuda and a.ndim == 2
            and a.dtype == torch.float8_e4m3fn and a.is_contiguous() and b.ndim == 3 and b.dtype == FP4
            and isinstance(b.storage.layout, BlackwellMXValueLayout) and a_ragged_metadata is not None
            and b_ragged_metadata is None and scatter_indx is None and fused_comm is None and fused_activation is None
            and (gammas is None or out_alpha in (None, 1.0)) and epilogue is None and c_acc_in is None
            and pc is not None and pc.out_dtype == torch.bfloat16 and not pc.enforce_bitwise_invariance
            and pc.a_microblock_size == pc.b_microblock_size == 32 and isinstance(pc.a_mx_scale, Tensor)
            and isinstance(pc.b_mx_scale, Tensor) and pc.a_mx_tensor_scale is None and pc.b_mx_tensor_scale is None
            and pc.c_mx_scale is None and pc.report_quantization_err_fn is None
            and isinstance(pc.b_mx_scale.storage.layout, BlackwellMXScaleLayout) and pc.intermediate_out_dtype is None
            and all(value is None for data in (
                pc.flex_ctx.lhs_data,
                pc.flex_ctx.rhs_data,
                pc.flex_ctx.out_data,
                pc.flex_ctx.acc_data,
            ) for value in vars(data).values()) and torch.cuda.get_device_capability(a.device)[0] == 10
            and not _get_opt_flags_constraints()):
        return None
    experts, k, n = b.shape
    rows = a.shape[0] if gather_indx is None else gather_indx.numel()
    if not (k == a.shape[1] and k >= 2048 and k % 256 == 0 and n % 128 == 0 and n >= 2048 and rows > 0 and experts > 0
            and experts == a_ragged_metadata.n_slices):
        return None
    expected_rows = a_ragged_metadata.expected_slice_size
    if (rows // experts if expected_rows is None else expected_rows) < 256:
        return None
    weights = b.storage.data.mT
    scales = pc.b_mx_scale.storage.data
    input_scales = pc.a_mx_scale.storage.data
    input_layout = pc.a_mx_scale.storage.layout
    native_scales = (isinstance(input_layout, BlackwellActMXScaleLayout)
                     and input_layout.ragged_metadata is a_ragged_metadata)
    if not (weights.is_contiguous() and scales.is_contiguous() and scales.dtype == input_scales.dtype == torch.uint8
            and weights.device == scales.device == input_scales.device == a.device and
            (native_scales or (isinstance(input_layout, StridedLayout) and input_scales.ndim == 2))):
        return None
    if c is not None and not (c.shape == (rows, n) and c.dtype == torch.bfloat16 and c.device == a.device
                              and c.is_contiguous() and not c.requires_grad):
        return None
    if bias is not None and not (bias.shape == (experts, n) and bias.stride(-1) == 1 and bias.dtype == torch.float32
                                 and bias.device == a.device):
        return None
    if any(t is not None and (t.device != a.device or t.numel() != rows or not t.is_contiguous())
           for t in (gather_indx, betas, gammas)):
        return None
    if c is not None and any(
            torch._C._overlaps(c, t)
            for t in (a, weights, scales, input_scales, bias, gather_indx, betas, gammas)
            if t is not None):
        return None
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count - _get_idle_sms()
    if sms < 2:
        return None
    metadata = a_ragged_metadata
    blocks = metadata.n_blocks(experts, rows, 128)
    schedule = metadata.block_schedule(128)
    if not blocks:
        return None
    if c is None:
        c = torch.empty((rows, n), device=a.device, dtype=torch.bfloat16)
    if native_scales:
        packed_scales = input_scales
    else:
        packed_scales = torch.empty((1, blocks, k // 128, 2, 256), device=a.device, dtype=torch.uint8)
        pack_routed_fp4_scales[(blocks, k // 128)](
            input_scales,
            packed_scales,
            gather_indx,
            metadata.slice_sizes,
            metadata.slice_offs,
            metadata.block_offs(128),
            schedule,
            E=experts,
            K4=k // 128,
            KS=input_scales.stride(0),
            KSTRIDE=input_scales.stride(1),
            num_warps=4,
        )
    a_desc = TensorDescriptor(
        a,
        list(a.shape),
        list(a.stride()),
        [256 if gather_indx is None else 1, 256],
        gl.NVMMASharedLayout(128, 8, cga_layout=[[1, 0]] if gather_indx is None else [[0, 0]]),
    )
    b_desc = TensorDescriptor(
        weights,
        [experts * n, k // 2],
        [k // 2, 1],
        [256, 128],
        gl.NVMMASharedLayout(128, 8, fp4_padded=True, cga_layout=[[1, 0]]),
    )
    c_desc = TensorDescriptor(c, [rows, n], [n, 1], [256, 32], gl.NVMMASharedLayout(64, 16, cga_layout=[[1, 0]]))
    scale_layout = gl.NVMMASharedLayout(0, 8, rank=5, cga_layout=[[0, 1, 0, 0, 0]])
    a_scale_desc = TensorDescriptor(
        packed_scales,
        list(packed_scales.shape),
        list(packed_scales.stride()),
        [1, 2, 2, 2, 256],
        scale_layout,
    )
    b_scale_desc = TensorDescriptor(
        scales,
        list(scales.shape),
        list(scales.stride()),
        [1, 2, 2, 2, 256],
        gl.NVMMASharedLayout(0, 8, rank=5, cga_layout=[[0, 0, 0, 0, 0]]),
    )
    grid = min(sms // 2, blocks * triton.cdiv(n, 256))
    sparse_fp4_matmul_kernel[(grid, )](
        a_desc,
        b_desc,
        c_desc,
        a_scale_desc,
        b_scale_desc,
        n,
        num_buffers=3,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=256,
        CGA_LAYOUT=((1, 0), ),
        USE_OVERLAP=True,
        EXPERTS=experts,
        schedule=schedule,
        slice_offs=metadata.slice_offs,
        slice_sizes=metadata.slice_sizes,
        scale_offs=metadata.block_offs(128),
        gather_ptr=gather_indx,
        out_ptr=c,
        bias_ptr=bias,
        beta_ptr=betas,
        gamma_ptr=gammas,
        alpha=1.0 if out_alpha is None else out_alpha,
        bias_stride=0 if bias is None else bias.stride(0),
        out_stride=n,
        num_ctas=2,
        num_warps=8,
        enable_fp_fusion=True,
    )
    return c
