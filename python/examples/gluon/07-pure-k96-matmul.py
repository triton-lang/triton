"""
Dense FP4 Matrix Multiplication with K96
======================================

High-performance 2CTA matrix multiplication for MXFP4 (block32 scales) and
NVFP4 (block16 scales). Three exact K256 producer stages feed eight K96 MMAs without
padding the reduction dimension. Accumulation is FP32; output defaults to FP16.
Requires sm103 (Blackwell Ultra) and K divisible by 768.

This builds on example 04's scale packing and tile schedulers. The MMA partition
uses typed descriptor ranges and continuations to span producer slots. NVFP4
stages data and scales in independent rings and drains the accumulator early.

Run correctness checks and benchmark both formats::

    python 07-pure-k96-matmul.py --M 16384 --N 16384 --K 16128

Compare output types (the operands remain FP4)::

    python 07-pure-k96-matmul.py --out-dtype float16 bfloat16 float32

Or run the example's tests::

    pytest -s 07-pure-k96-matmul.py

For frozen-binary comparisons and measured results, see
bench-tcgen05-pure-k96.py and K96_EXPLORATION.md.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tcgen05_copy,
    tcgen05_commit,
    tcgen05_mma_barrier_count,
    tcgen05_mma_scaled,
    mbarrier,
    tma,
)

# Reuse example 04's schedulers, scale packing, and coupled-ring pipeline.
_base_path = Path(__file__).with_name('04-2cta-block-scale-matmul.py')
_spec = importlib.util.spec_from_file_location('dense_k96_base', _base_path)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)
Counter = base.Counter
PartitionArgs = base.PartitionArgs
SCHEDULER_CLC = base.SCHEDULER_CLC
SCHEDULER_SPS = base.SCHEDULER_SPS
mma_scaled_load_partition = base.mma_scaled_load_partition
mma_scaled_epilogue_partition = base.mma_scaled_epilogue_partition
mma_scaled_clc_partition = base.mma_scaled_clc_partition
unswizzle_scales_shared_memory = base.unswizzle_scales_shared_memory

# ---------------------------------------------------------------------------
# MMA partition
# ---------------------------------------------------------------------------

# Each macrotile consumes three K256 producer slots with eight K96 MMAs:
#
#   Logical K range   Source slots             Release
#   [  0, 192)        slot 0                   -
#   [192, 288)        slot 0 tail -> slot 1     slot 0
#   [288, 480)        slot 1                   -
#   [480, 576)        slot 1 tail -> slot 2     slot 1
#   [576, 768)        slot 2                   slot 2
#
# k_range is relative to the current data descriptor; scale offsets are relative
# to the macrotile's TMEM scale stream. Continuation descriptors supply the next
# slot without assuming that ring slots are adjacent in shared memory.
#
# MXFP4 copies a full K256 of scales at a time. NVFP4 uses a separate scale ring
# and interleaves K128 scale copies with MMA issue. Splitting the first and third
# ranges above into individual K96 calls makes those copies ready just in time,
# without increasing the number of copies or bytes transferred.


@gluon.jit
def copy_scales_to_tmem(p, scale_idx, a_scale_tmem, b_scale_tmem, OFFSET: gl.constexpr, SCALE_BLOCK_SIZE: gl.constexpr,
                        SRC: gl.constexpr = 0, WIDTH: gl.constexpr = 256):
    a = unswizzle_scales_shared_memory(p.a_scale_bufs.index(scale_idx), 256, 256, SCALE_BLOCK_SIZE)
    b = unswizzle_scales_shared_memory(p.b_scale_bufs.index(scale_idx), 256, 256, SCALE_BLOCK_SIZE)
    tcgen05_copy(a.slice(SRC // SCALE_BLOCK_SIZE, WIDTH // SCALE_BLOCK_SIZE, 1),
                 a_scale_tmem.slice(OFFSET // SCALE_BLOCK_SIZE, WIDTH // SCALE_BLOCK_SIZE))
    tcgen05_copy(b.slice(SRC // SCALE_BLOCK_SIZE, WIDTH // SCALE_BLOCK_SIZE, 1),
                 b_scale_tmem.slice(OFFSET // SCALE_BLOCK_SIZE, WIDTH // SCALE_BLOCK_SIZE))


@gluon.jit
def mma_scaled_mma_partition(p, scale_empty_bars):
    SPLIT_SCALES: gl.constexpr = p.a_scale_bufs.shape[0] != p.a_bufs.shape[0]
    COPY_K: gl.constexpr = 128 if SPLIT_SCALES else 256
    scale_state = Counter.create(0, p.a_scale_bufs.shape[0])
    K = p.a_desc.shape[1] * 2
    SCALE_BLOCK_SIZE: gl.constexpr = 32 if p.a_scale_desc.dtype == gl.uint8 else 16
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    # TMEM allocations are power-of-two; only the scales for K768 are consumed.
    a_scale_tmem = allocate_tensor_memory(p.a_scale_desc.dtype, [256, 1024 // SCALE_BLOCK_SIZE],
                                          TensorMemoryScalesLayout([[1, 0]]))
    b_scale_tmem = allocate_tensor_memory(p.b_scale_desc.dtype, [256, 1024 // SCALE_BLOCK_SIZE],
                                          TensorMemoryScalesLayout([[0, 0]]))
    i = 0
    while scheduler.has_work:
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for k in range(0, K, 768):
            # Data and scale rings advance independently, including across tiles.
            scale0 = scale_state if SPLIT_SCALES else load_state
            scale1 = scale0.next()
            scale2 = scale1.next()
            data0 = load_state
            data1 = data0.next()
            data2 = data1.next()
            a0, b0 = p.a_bufs.index(data0.index), p.b_bufs.index(data0.index).permute((1, 0))
            a1, b1 = p.a_bufs.index(data1.index), p.b_bufs.index(data1.index).permute((1, 0))
            a2, b2 = p.a_bufs.index(data2.index), p.b_bufs.index(data2.index).permute((1, 0))
            mbarrier.wait(p.load_ready_bars.index(data0.index), data0.phase)
            copy_scales_to_tmem(p, scale0.index, a_scale_tmem, b_scale_tmem, 0, SCALE_BLOCK_SIZE, WIDTH=COPY_K)
            if SPLIT_SCALES:
                tcgen05_mma_scaled(a0, b0, acc, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1", use_acc=use_acc,
                                   k_range=(0, 96), instruction_k=96, scale_block_size=SCALE_BLOCK_SIZE,
                                   a_scale_offset=0, b_scale_offset=0, multicast=True, mbarriers=[], is_async=True)
                copy_scales_to_tmem(p, scale0.index, a_scale_tmem, b_scale_tmem, 128, SCALE_BLOCK_SIZE, SRC=128,
                                    WIDTH=128)
                tcgen05_commit(scale_empty_bars.index(scale0.index),
                               descs=[p.a_scale_bufs.index(scale0.index),
                                      p.b_scale_bufs.index(scale0.index)])
                tcgen05_mma_scaled(a0, b0, acc, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1", k_range=(96, 192),
                                   instruction_k=96, scale_block_size=SCALE_BLOCK_SIZE,
                                   a_scale_offset=96 // SCALE_BLOCK_SIZE, b_scale_offset=96 // SCALE_BLOCK_SIZE,
                                   multicast=True, mbarriers=[], is_async=True)
            else:
                tcgen05_mma_scaled(a0, b0, acc, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1", use_acc=use_acc,
                                   k_range=(0, 192), instruction_k=96, scale_block_size=SCALE_BLOCK_SIZE,
                                   a_scale_offset=0, b_scale_offset=0, multicast=True, mbarriers=[], is_async=True)
            # The crossing MMA needs both slots; release only the retiring slot.
            mbarrier.wait(p.load_ready_bars.index(data1.index), data1.phase)
            copy_scales_to_tmem(p, scale1.index, a_scale_tmem, b_scale_tmem, 256, SCALE_BLOCK_SIZE, WIDTH=COPY_K)
            tcgen05_mma_scaled(a0, b0, acc, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1", a_next=a1, b_next=b1,
                               k_range=(192, 288), instruction_k=96, scale_block_size=SCALE_BLOCK_SIZE,
                               a_scale_offset=192 // SCALE_BLOCK_SIZE, b_scale_offset=192 // SCALE_BLOCK_SIZE,
                               multicast=True, mbarriers=[p.load_empty_bars.index(data0.index)])
            if SPLIT_SCALES:
                tcgen05_mma_scaled(a1, b1, acc, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1", k_range=(32, 128),
                                   instruction_k=96, scale_block_size=SCALE_BLOCK_SIZE,
                                   a_scale_offset=288 // SCALE_BLOCK_SIZE, b_scale_offset=288 // SCALE_BLOCK_SIZE,
                                   multicast=True, mbarriers=[], is_async=True)
                copy_scales_to_tmem(p, scale1.index, a_scale_tmem, b_scale_tmem, 384, SCALE_BLOCK_SIZE, SRC=128,
                                    WIDTH=128)
                tcgen05_commit(scale_empty_bars.index(scale1.index),
                               descs=[p.a_scale_bufs.index(scale1.index),
                                      p.b_scale_bufs.index(scale1.index)])
                tcgen05_mma_scaled(a1, b1, acc, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1", k_range=(128, 224),
                                   instruction_k=96, scale_block_size=SCALE_BLOCK_SIZE,
                                   a_scale_offset=384 // SCALE_BLOCK_SIZE, b_scale_offset=384 // SCALE_BLOCK_SIZE,
                                   multicast=True, mbarriers=[], is_async=True)
            else:
                tcgen05_mma_scaled(a1, b1, acc, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1", k_range=(32, 224),
                                   instruction_k=96, scale_block_size=SCALE_BLOCK_SIZE,
                                   a_scale_offset=288 // SCALE_BLOCK_SIZE, b_scale_offset=288 // SCALE_BLOCK_SIZE,
                                   multicast=True, mbarriers=[], is_async=True)
            mbarrier.wait(p.load_ready_bars.index(data2.index), data2.phase)
            copy_scales_to_tmem(p, scale2.index, a_scale_tmem, b_scale_tmem, 512, SCALE_BLOCK_SIZE, WIDTH=COPY_K)
            tcgen05_mma_scaled(a1, b1, acc, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1", a_next=a2, b_next=b2,
                               k_range=(224, 320), instruction_k=96, scale_block_size=SCALE_BLOCK_SIZE,
                               a_scale_offset=480 // SCALE_BLOCK_SIZE, b_scale_offset=480 // SCALE_BLOCK_SIZE,
                               multicast=True, mbarriers=[p.load_empty_bars.index(data1.index)])
            if SPLIT_SCALES:
                copy_scales_to_tmem(p, scale2.index, a_scale_tmem, b_scale_tmem, 640, SCALE_BLOCK_SIZE, SRC=128,
                                    WIDTH=128)
                tcgen05_commit(scale_empty_bars.index(scale2.index),
                               descs=[p.a_scale_bufs.index(scale2.index),
                                      p.b_scale_bufs.index(scale2.index)])
            tcgen05_mma_scaled(a2, b2, acc, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1", k_range=(64, 256),
                               instruction_k=96, scale_block_size=SCALE_BLOCK_SIZE,
                               a_scale_offset=576 // SCALE_BLOCK_SIZE, b_scale_offset=576 // SCALE_BLOCK_SIZE,
                               multicast=True, mbarriers=[p.load_empty_bars.index(data2.index)])
            scale_state = scale2.next()
            load_state = data2.next()
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


# ---------------------------------------------------------------------------
# TMA producer with independent data and scale rings
# ---------------------------------------------------------------------------


@gluon.jit
def mma_scaled_split_load_partition(p, scale_empty_bars):
    K = p.a_desc.shape[1] * 2
    data = Counter.create(1, p.load_empty_bars.shape[0])
    scales = Counter.create(1, p.a_scale_bufs.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()
        for k in range(0, K, 256):
            mbarrier.wait(p.load_empty_bars.index(data.index), data.phase)
            bar = p.load_ready_bars.index(data.index)
            mbarrier.expect(
                bar, p.a_desc.nbytes_per_cta + p.b_desc.nbytes_per_cta + p.a_scale_desc.nbytes_per_cta +
                p.b_scale_desc.nbytes_per_cta)
            tma.async_load(p.a_desc, [off_m, k // 2], bar, p.a_bufs.index(data.index), multicast=True)
            tma.async_load(p.b_desc, [off_n, k // 2], bar, p.b_bufs.index(data.index), multicast=True)
            # Data has six slots. Only the independently staged scales wait
            # for their five-slot ring, after both data transfers are issued.
            mbarrier.wait(scale_empty_bars.index(scales.index), scales.phase)
            sk = k // 256 * p.a_scale_desc.block_shape[2]
            tma.async_load(p.b_scale_desc, [0, scheduler.pid_n * 2, sk, 0, 0], bar, p.b_scale_bufs.index(scales.index),
                           multicast=True)
            tma.async_load(p.a_scale_desc, [0, scheduler.pid_m * 2, sk, 0, 0], bar, p.a_scale_bufs.index(scales.index),
                           multicast=True)
            data = data.next()
            scales = scales.next()
        scheduler = scheduler.step(i)
        i += 1


# ---------------------------------------------------------------------------
# Early accumulator drain and TMA epilogue
# ---------------------------------------------------------------------------


@gluon.jit
def mma_scaled_split_epilogue_partition(p, c_final):
    EPILOGUE_BLOCK_N: gl.constexpr = p.c_desc.block_shape[1]
    buf = gl.allocate_shared_memory(p.c_desc.dtype, [256, EPILOGUE_BLOCK_N], p.c_desc.layout)
    state = Counter.create(0, p.acc_empty_bars.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()
        mbarrier.wait(p.acc_ready_bars.index(state.index), state.phase)
        acc = p.acc_bufs.index(state.index)
        # Convert each half before loading the next, keeping the complete
        # output in 16-bit registers without a 256-register FP32 live range.
        parts = ()
        for j in gl.static_range(2):
            # Split the N128 half into contiguous output chunks using only views.
            chunks = (acc.slice(j * 128, 128).load().to(p.c_desc.dtype), )
            for level in gl.static_range((128 // EPILOGUE_BLOCK_N).bit_length() - 1):
                next_chunks = ()
                for n in gl.static_range(len(chunks)):
                    left, right = chunks[n].reshape((256, 2, chunks[n].shape[1] // 2)).permute((0, 2, 1)).split()
                    next_chunks += (left, right)
                chunks = next_chunks
            parts += chunks
        mbarrier.arrive(p.acc_empty_bars.index(state.index), count=1)
        if p.scheduler == SCHEDULER_SPS:
            last = scheduler.tile_id + gl.num_programs(0) >= p.NUM_PID_M * p.NUM_PID_N
        else:
            last = False
        if last:
            tiles = parts
            for level in gl.static_range((256 // EPILOGUE_BLOCK_N).bit_length() - 1):
                joined = ()
                for j in gl.static_range(len(tiles) // 2):
                    first, second = tiles[2 * j], tiles[2 * j + 1]
                    joined += (gl.join(first, second).permute((0, 2, 1)).reshape((256, first.shape[1] * 2)), )
                tiles = joined
            # All input reads have completed, and this persistent CTA has no
            # future producer work. Four retired FP4 input slots fit one 16-bit
            # output tile, avoiding the narrow epilogue's tail. The
            # four epilogue warps each issue an N64 TMA box; this is not one
            # hardware transaction. No input storage is reused before completion.
            scratch = p.a_bufs.slice(0, 4, 0).reinterpret(p.c_desc.dtype, [256, 256], c_final.layout)
            scratch.store(tiles[0])
            tma.async_store(c_final, [off_m, off_n], scratch)
        else:
            for j in gl.static_range(256 // EPILOGUE_BLOCK_N):
                tma.store_wait(0)
                buf.store(parts[j])
                tma.async_store(p.c_desc, [off_m, off_n + j * EPILOGUE_BLOCK_N], buf)
        state = state.next()
        scheduler = scheduler.step(i)
        i += 1
    tma.store_wait(0)


# ---------------------------------------------------------------------------
# Warp-specialized kernel
# ---------------------------------------------------------------------------


@gluon.jit
def dense_k96_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, M, N, K, A_ELEM_PER_BYTE,
                     num_buffers: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                     EPILOGUE_BLOCK_N: gl.constexpr, num_acc_buffers: gl.constexpr, GRID_MINOR_DIM: gl.constexpr,
                     GRID_TILE_WIDTH: gl.constexpr, CGA_LAYOUT: gl.constexpr, scheduler: gl.constexpr,
                     SCALE_BUFFERS: gl.constexpr, c_final):
    SPLIT_SCALES: gl.constexpr = SCALE_BUFFERS != num_buffers
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    gl.static_assert(BLOCK_M_PER_CTA == 64 or BLOCK_M_PER_CTA == 128)
    N_PARTITIONS: gl.constexpr = 4 if scheduler == SCHEDULER_CLC else 3

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [SCALE_BUFFERS] + a_scale_desc.block_shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [SCALE_BUFFERS] + b_scale_desc.block_shape,
                                             b_scale_desc.layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M_PER_CTA, BLOCK_N], col_stride=1, cga_layout=CGA_LAYOUT,
                                                   two_ctas=TWO_CTAS)
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)

    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count([
        a_bufs.index(0),
        b_bufs.index(0),
        a_bufs.index(1),
        b_bufs.index(1),
        a_scale_bufs.index(0),
        b_scale_bufs.index(0)
    ], multicast=True, two_ctas=acc_bufs.index(0).type.layout.two_ctas)

    if SPLIT_SCALES:
        scale_empty_bars = mbarrier.allocate_mbarrier(batch=SCALE_BUFFERS)
        for slot in gl.static_range(SCALE_BUFFERS):
            mbarrier.init(scale_empty_bars.index(slot), count=mma_barrier_count)
    load_empty_bars = mbarrier.allocate_mbarrier(batch=num_buffers)
    if not SPLIT_SCALES:
        scale_empty_bars = load_empty_bars
    load_ready_bars = mbarrier.allocate_mbarrier(batch=num_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready_bars.index(i), count=1)

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
                      load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars, clc_result_buffers,
                      clc_barriers, clc_planar_pid_buffers, clc_planar_ready_bars, clc_consumed_bars, GRID_MINOR_DIM,
                      GRID_TILE_WIDTH, gl.cdiv(M, BLOCK_M), gl.cdiv(N, BLOCK_N), scheduler)

    if SPLIT_SCALES:
        if scheduler == SCHEDULER_CLC:
            gl.warp_specialize([
                (mma_scaled_split_epilogue_partition, (p, c_final)),
                (mma_scaled_mma_partition, (p, scale_empty_bars)),
                (mma_scaled_split_load_partition, (p, scale_empty_bars)),
                (mma_scaled_clc_partition, (p, )),
            ], [1, 1, 1], [48, 48, 24])
        else:
            gl.warp_specialize([
                (mma_scaled_split_epilogue_partition, (p, c_final)),
                (mma_scaled_mma_partition, (p, scale_empty_bars)),
                (mma_scaled_split_load_partition, (p, scale_empty_bars)),
            ], [1, 1], [48, 48])
    else:
        if scheduler == SCHEDULER_CLC:
            gl.warp_specialize([
                (mma_scaled_epilogue_partition, (p, )),
                (mma_scaled_mma_partition, (p, scale_empty_bars)),
                (mma_scaled_load_partition, (p, )),
                (mma_scaled_clc_partition, (p, )),
            ], [1, 1, 1], [24, 24, 24])
        else:
            gl.warp_specialize([
                (mma_scaled_epilogue_partition, (p, )),
                (mma_scaled_mma_partition, (p, scale_empty_bars)),
                (mma_scaled_load_partition, (p, )),
            ], [1, 1], [24, 24])


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------


def matmul(A, B, A_scale, B_scale, *, buffers=None, epilogue=None, scheduler=None, out_dtype=torch.float16,
           tile_width=None):
    """Compute A @ B.T from packed FP4 operands and prepacked block scales.

    A and B are uint8 tensors of shape (M, K // 2) and (N, K // 2).
    Pack their scale tensors with example 04's swizzle_scales_packed_block:
    uint8 E8M0 scales select MXFP4/block32, float8_e4m3fn selects NVFP4/block16.
    The result has shape (M, N). M/N tails are supported; K must be divisible by 768.

    Defaults retain six coupled data/scale slots for MXFP4 and six data/five
    scale slots for 16-bit NVFP4 output. Wider NVFP4 output uses five coupled
    slots. Buffer and epilogue overrides are useful for studying the pipeline;
    the default settings preserve the measured shared-memory/register budget.
    """
    assert torch.cuda.get_device_capability(A.device) == (10, 3), "K96 requires sm103"
    assert A.dtype == B.dtype == torch.uint8, "FP4 operands must use packed uint8 storage"
    assert A_scale.dtype == B_scale.dtype in (torch.uint8, torch.float8_e4m3fn)
    scale_block_size = 16 if A_scale.dtype == torch.float8_e4m3fn else 32
    if buffers is None:
        buffers = 6 if scale_block_size == 32 or out_dtype.itemsize == 2 else 5
    scale_buffers = min(buffers, 5) if scale_block_size == 16 else buffers
    if epilogue is None:
        epilogue = 16 if scale_buffers != buffers else (64 if scale_block_size == 32 else 128) // out_dtype.itemsize
    M, N, K = A.shape[0], B.shape[0], A.shape[1] * 2
    assert B.shape[1] * 2 == K and K % 768 == 0, "K must match and be divisible by 768"
    if scheduler is None:
        scheduler = SCHEDULER_SPS
    if tile_width is None:
        tile_width = 16 if K <= 16384 else 8

    a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc = base.make_dummy_descriptors(
        A, B, A_scale, B_scale, out_dtype, M, N)
    base.mma_scaled_tma_set_block_size_hook(
        dict(a_desc=a_desc, b_desc=b_desc, c_desc=c_desc, a_scale_desc=a_scale_desc, b_scale_desc=b_scale_desc,
             BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, EPILOGUE_BLOCK_N=epilogue, CGA_LAYOUT=((1, 0), )))
    c_dtype = getattr(gl, str(out_dtype).split('.')[1])
    final_layout = gl.NVMMASharedLayout.get_default_for([256, 256], c_dtype, cga_layout=((1, 0), ))
    c_final = base.TensorDescriptor.from_tensor(c_desc.base, [256, 256], final_layout)
    grid = base.mma_scaled_warp_specialized_grid(M, N, 256, 256, 2, scheduler, A.device)
    dense_k96_kernel[grid](a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, M, N, K, 2, buffers, 256, 256, 256,
                           epilogue, 1, 0, tile_width, ((1, 0), ), scheduler, scale_buffers, c_final, num_ctas=2)
    return c_desc.base


# ---------------------------------------------------------------------------
# Correctness and benchmark
# ---------------------------------------------------------------------------


def is_blackwell_ultra():
    return base.is_blackwell() and torch.cuda.get_device_capability() == (10, 3)


def make_problem(M, N, K, format):
    A, A_scale, A_ref = base.random_quantized_tensor(M, K, format)
    B, B_scale, B_ref = base.random_quantized_tensor(N, K, format)
    A_scale = base.swizzle_scales_packed_block(A_scale)
    B_scale = base.swizzle_scales_packed_block(B_scale)
    return A, B, A_scale, B_scale, A_ref @ B_ref.T


def benchmark(M, N, K, format, scheduler, rep_ms=500, tile_width=None, out_dtype=torch.float16):
    A, B, A_scale, B_scale, C_ref = make_problem(M, N, K, format)

    def run():
        return matmul(A, B, A_scale, B_scale, scheduler=scheduler, tile_width=tile_width, out_dtype=out_dtype)

    C = run()
    # Compare before output rounding: independent BF16 rounding can straddle a midpoint.
    rtol = max(1e-3, torch.finfo(out_dtype).eps / 2)
    torch.testing.assert_close(C.float(), C_ref, atol=2e-3, rtol=rtol)
    ms = triton.testing.do_bench_cudagraph(run, rep=rep_ms)
    return ms, 2.0 * M * N * K / (ms * 1.0e12)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires sm103 K96")
@pytest.mark.parametrize("format", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matmul(format, out_dtype):
    torch.manual_seed(0)
    A, B, A_scale, B_scale, C_ref = make_problem(512, 512, 2304, format)
    C = matmul(A, B, A_scale, B_scale, out_dtype=out_dtype)
    # Compare before output rounding: independent BF16 rounding can straddle a midpoint.
    rtol = max(1e-3, torch.finfo(out_dtype).eps / 2)
    torch.testing.assert_close(C.float(), C_ref, atol=2e-3, rtol=rtol)


def main():
    parser = argparse.ArgumentParser(description="Dense MXFP4/NVFP4 matmul with pure K96 on sm103")
    parser.add_argument("--M", type=int, default=16384)
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--K", type=int, default=16128, help="Reduction size, divisible by 768")
    parser.add_argument("--format", nargs="+", choices=["mxfp4", "nvfp4"], default=["mxfp4", "nvfp4"])
    parser.add_argument("--out-dtype", nargs="+", choices=["float16", "bfloat16", "float32"], default=["float16"],
                        help="Output type; operands remain packed FP4 and accumulation remains FP32")
    parser.add_argument("--scheduler", choices=["sps", "clc"], default="sps")
    parser.add_argument("--tile-width", type=int, help="Override the grouped tile traversal width")
    parser.add_argument("--rep-ms", type=int, default=500, help="CUDA-graph benchmark duration parameter")
    args = parser.parse_args()
    if not is_blackwell_ultra():
        parser.error("K96 requires an sm103 (Blackwell Ultra) GPU")
    if args.K % 768:
        parser.error("--K must be divisible by 768")
    scheduler = SCHEDULER_SPS if args.scheduler == "sps" else SCHEDULER_CLC
    print("format   output          M       N       K        ms   PFLOPS")
    for format in args.format:
        for dtype in args.out_dtype:
            torch.manual_seed(0)
            ms, pflops = benchmark(args.M, args.N, args.K, format, scheduler, args.rep_ms, args.tile_width,
                                   getattr(torch, dtype))
            print(f"{format:8s} {dtype:8s} {args.M:7d} {args.N:7d} {args.K:7d} {ms:9.4f} {pflops:8.4f}")


if __name__ == "__main__":
    main()
