"""
2CTA Block-Scaled Matrix Multiplication
=======================================

High-performance 2CTA warp-specialized block-scaled MMA.
Two CTAs cooperate per output tile, sharing operands to
increase arithmetic intensity and reduce the per-CTA SMEM
footprint.

Performance is benchmarked against a baseline 1CTA kernel
(from Gluon tutorial 11) and cuBLAS.  Supports mxfp8,
mxfp4, nvfp4, and mixed-precision (mxfp8 x mxfp4) formats.
"""

import itertools
import pytest
import torch

import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from dataclasses import replace
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton.language.core import _aggregate as aggregate

from triton._C.libtriton import nvidia

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tcgen05_copy,
    tcgen05_commit,
    tcgen05_mma_scaled,
    mbarrier,
    tma,
)

# ---------------------------------------------------------------------------
# Tile scheduler
# ---------------------------------------------------------------------------


def GroupedPersistentTileScheduler(GROUP_SIZE_M):
    # Bind this as a constexpr so it can be captured.
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

    # Like C++ templates!
    @aggregate
    class GroupedPersistentTileSchedulerImpl:
        start_pid: gl.tensor
        num_pid_m: gl.tensor
        num_pid_in_group: gl.tensor
        num_pid: gl.tensor

        @gluon.jit
        def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
            start_pid = gl.program_id(axis=0)
            num_pid_m = gl.cdiv(M, BLOCK_M)
            num_pid_n = gl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_pid = num_pid_m * num_pid_n
            return GroupedPersistentTileSchedulerImpl(start_pid, num_pid_m, num_pid_in_group, num_pid)

        @gluon.jit
        def get_num_tiles(self):
            return gl.cdiv(self.num_pid - self.start_pid, gl.num_programs(axis=0))

        @gluon.jit
        def get_tile(self, idx):
            tile_id = self.start_pid + idx * gl.num_programs(axis=0)
            group_id = tile_id // self.num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % self.num_pid_in_group) // group_size_m
            return pid_m, pid_n

    GroupedPersistentTileSchedulerImpl.__name__ = f"GroupedPersistentTileScheduler({GROUP_SIZE_M.value})"
    return GroupedPersistentTileSchedulerImpl


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


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


def make_operand_descriptor(value: torch.Tensor, BLOCK_MN: int, BLOCK_K: int, MIXED_PREC: bool, cga_layout=None):
    # If the operand dtype is fp4, they will be packed into uint8.
    IS_FP4 = value.dtype == torch.uint8
    ELEM_PER_BYTE = 2 if IS_FP4 else 1

    # When performing a mixed-precision `tcgen05_mma_scaled`, where one operand
    # is mxfp8 and the other is mxfp4, the fp4 operand is padded in shared memory.
    IS_MIXED_PREC_FP4 = MIXED_PREC and IS_FP4
    layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_MN, BLOCK_K // ELEM_PER_BYTE],
        gl.uint8 if IS_FP4 else gl.float8e4nv,
        fp4_padded=IS_MIXED_PREC_FP4,
        cga_layout=cga_layout,
    )
    return TensorDescriptor.from_tensor(value, [BLOCK_MN, BLOCK_K // ELEM_PER_BYTE], layout)


def make_output_descriptor(M: int, N: int, dtype: torch.dtype, BLOCK_M: int, BLOCK_N: int, cga_layout=None):
    C = torch.empty(M, N, device="cuda", dtype=dtype)
    C_dtype = getattr(gl, str(dtype).split('.')[1])
    C_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], C_dtype, cga_layout=cga_layout)
    return TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], C_desc_layout)


def make_scales_descriptor(scales: torch.Tensor, BLOCK_MN: int, BLOCK_K: int, VEC_SIZE: int, cga_layout=None):
    # Note that this 5D swizzling scheme has minimum block size requirements
    # of BLOCK_N >= 128 and BLOCK_K >= VEC_SIZE * 4 (64 for nvfp4 and 128 for MX).
    REP_MN = BLOCK_MN // 128
    REP_K = BLOCK_K // (VEC_SIZE * 4)
    # Use a 5D TMA descriptor with block shape [1, rep_m, rep_k, 2, 256] of uint8
    # elements. With 256 bytes along the inner dimension, we better utilize the
    # L2 cache and don't require the TMA engine to emit many small messages (16B)
    # as it would with 32x16xu8.
    block_shape = [1, REP_MN, REP_K, 2, 256]
    scales = scales.reshape(1, scales.shape[0], scales.shape[1], 2, 256)
    IS_NVFP4 = scales.dtype == torch.float8_e4m3fn
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float8e4nv if IS_NVFP4 else gl.uint8,
                                                  cga_layout=cga_layout)
    return TensorDescriptor.from_tensor(scales, block_shape, layout)


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
    BLOCK_M: gl.constexpr = a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    REP_M: gl.constexpr = a_scale_desc.block_type.shape[1]
    REP_N: gl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_type.shape[2]
    B_REP_K: gl.constexpr = b_scale_desc.block_type.shape[2]

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
    SchedulerImpl: gl.constexpr
    multicast_b_scale: gl.constexpr

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    M: gl.tensor
    N: gl.tensor
    K: gl.tensor


@gluon.jit
def mma_scaled_load_partition(p):
    state = Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.M, p.N, p.BLOCK_M, p.BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        for k in range(0, p.K, p.BLOCK_K):
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            state = issue_loads(state, pid_m, pid_n, k, p.a_desc, p.b_desc, p.a_scale_desc, p.b_scale_desc, p.a_bufs,
                                p.b_bufs, p.a_scale_bufs, p.b_scale_bufs, p.load_ready_bars, pred=True,
                                multicast_b_scale=p.multicast_b_scale)


@gluon.jit
def mma_scaled_mma_partition(p):
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.M, p.N, p.BLOCK_M, p.BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for k in range(0, p.K, p.BLOCK_K):
            _, load_state = issue_mma(load_state, p.load_ready_bars, p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs,
                                      load_state, p.load_empty_bars, acc_buf, use_acc, pred=True)
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def mma_scaled_epilogue_partition(p):
    tile_m: gl.constexpr = p.c_desc.block_type.shape[0]
    EPILOGUE_BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    subtile_factor: gl.constexpr = p.BLOCK_N // EPILOGUE_BLOCK_N
    subtile_stages: gl.constexpr = 1 if subtile_factor == 1 else 2
    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    acc_smems = gl.allocate_shared_memory(p.c_desc.dtype, [subtile_stages, tile_m, EPILOGUE_BLOCK_N], p.c_desc.layout)
    sub_acc_state = Counter.create(0, subtile_stages)
    scheduler = p.SchedulerImpl.initialize(p.M, p.N, p.BLOCK_M, p.BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)

        for s in gl.static_range(subtile_factor):
            acc_sub = acc_buf.slice(EPILOGUE_BLOCK_N * s, EPILOGUE_BLOCK_N)
            acc_smem = acc_smems.index(sub_acc_state.index)
            acc = acc_sub.load().to(p.c_desc.dtype)
            tma.store_wait(pendings=subtile_stages - 1)
            acc_smem.store(acc)
            tma.async_copy_shared_to_global(p.c_desc, [pid_m * p.BLOCK_M, pid_n * p.BLOCK_N + EPILOGUE_BLOCK_N * s],
                                            acc_smem)
            sub_acc_state = sub_acc_state.next()
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()
    tma.store_wait(0)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@gluon.jit
def mma_scaled_warp_specialized_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, num_buffers: gl.constexpr,
                                       BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                                       num_acc_buffers: gl.constexpr, SchedulerImpl: gl.constexpr,
                                       NUM_CTAS: gl.constexpr, block_layout_c: gl.constexpr):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    M = c_desc.shape[0]
    N = c_desc.shape[1]
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_type.shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_type.shape,
                                             b_scale_desc.layout)

    if NUM_CTAS > 1:
        bar_layout: gl.constexpr = mbarrier.MBarrierLayout.multicta(num_ctas=NUM_CTAS, two_cta=True)
        lead_bar_layout: gl.constexpr = mbarrier.MBarrierLayout.multicta(num_ctas=NUM_CTAS, two_cta=False)
        tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M // NUM_CTAS, BLOCK_N], col_stride=1,
                                                       cga_layout=block_layout_c.cga_layout, two_ctas=True)
    else:
        bar_layout: gl.constexpr = mbarrier.MBarrierLayout()
        lead_bar_layout: gl.constexpr = mbarrier.MBarrierLayout()
        tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)

    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, NUM_CTAS], lead_bar_layout)
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], bar_layout)
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], bar_layout)
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, NUM_CTAS], lead_bar_layout)
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)
    p = PartitionArgs(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                      load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars, SchedulerImpl,
                      NUM_CTAS > 1, BLOCK_M, BLOCK_N, BLOCK_K, M, N, K)

    gl.warp_specialize([
        (mma_scaled_epilogue_partition, (p, )),
        (mma_scaled_mma_partition, (p, )),
        (mma_scaled_load_partition, (p, )),
    ], [1, 1], [24, 24])


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def mma_scaled_warp_specialized(A, B, A_scale, B_scale, VEC_SIZE, GROUP_SIZE_M=8, out_dtype=torch.float16, BLOCK_M=128,
                                BLOCK_N=256, BLOCK_K=None, EPILOGUE_BLOCK_N=None, num_buffers=3, acc_buffers=None,
                                num_ctas=1):
    """Warp-specialized block-scale MMA (supports 1CTA and multi-CTA)."""
    if BLOCK_K is None:
        BLOCK_K = 128 if torch.float8_e4m3fn in [A.dtype, B.dtype] else 256
    if EPILOGUE_BLOCK_N is None:
        EPILOGUE_BLOCK_N = BLOCK_N
    if acc_buffers is None:
        acc_buffers = 2 if BLOCK_N < 256 else 1

    SchedulerImpl = GroupedPersistentTileScheduler(GROUP_SIZE_M)
    M, N = A.shape[0], B.shape[0]
    MIXED_PREC = A.dtype != B.dtype

    if num_ctas > 1:
        # split A/C along M; B along N across CTAs
        cga_layout = [[1, 0]]
        cga_layout_a_scale = [[0, 1, 0, 0, 0]]  # split A scales along M across CTAs
        cga_layout_b_scale = [[0, 0, 0, 0, 0]]  # broadcast B scales to both CTAs
        no_swizzle_a = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5,
                                            cga_layout=cga_layout_a_scale)
        no_swizzle_b = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5,
                                            cga_layout=cga_layout_b_scale)
        block_layout_c = gl.BlockedLayout([1, 8], [1, 32], warps_per_cta=[4, 1], order=[1, 0], cga_layout=cga_layout)
    else:
        cga_layout = None
        cga_layout_a_scale = None
        cga_layout_b_scale = None
        no_swizzle_a = no_swizzle_b = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
        block_layout_c = None

    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC, cga_layout=cga_layout)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC, cga_layout=cga_layout)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, EPILOGUE_BLOCK_N, cga_layout=cga_layout)
    A_scale_desc = make_scales_descriptor(A_scale, BLOCK_M, BLOCK_K, VEC_SIZE, cga_layout=cga_layout_a_scale)
    B_scale_desc = make_scales_descriptor(B_scale, BLOCK_N, BLOCK_K, VEC_SIZE, cga_layout=cga_layout_b_scale)
    A_scale_desc = replace(A_scale_desc, layout=no_swizzle_a)
    B_scale_desc = replace(B_scale_desc, layout=no_swizzle_b)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    mma_scaled_warp_specialized_kernel[grid](
        A_desc,
        B_desc,
        C_desc,
        A_scale_desc,
        B_scale_desc,
        num_buffers,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        acc_buffers,
        SchedulerImpl,
        num_ctas,
        block_layout_c,
        num_ctas=num_ctas,
    )
    return C_desc.base


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


def make_fn(variant, A, B, A_scale, B_scale, VEC_SIZE, a_format):
    """Build the callable for a given variant (1cta, 2cta, or cublas)."""
    # 2CTA: Shared B operand doubles arithmetic intensity and
    # halves per-CTA SMEM. Subtiled epilogue reclaims additional
    # SMEM from the C store path. Reduced SMEM footprint enables
    # a deeper pipeline.
    if variant == "2cta":
        return lambda: mma_scaled_warp_specialized(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M=256, BLOCK_N=256,
                                                   EPILOGUE_BLOCK_N=64, num_buffers=5, num_ctas=2)
    # 1CTA: Defaults from Gluon tutorial 11.
    elif variant == "1cta":
        return lambda: mma_scaled_warp_specialized(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M=128, BLOCK_N=256,
                                                   EPILOGUE_BLOCK_N=256, num_buffers=3, num_ctas=1)
    elif variant == "cublas":
        A_scale_flat = A_scale.contiguous().flatten()
        B_scale_flat = B_scale.contiguous().flatten()

        def cublas_fn():
            cublas.set_stream(torch.cuda.current_stream().cuda_stream)
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
    """Print a formatted benchmark table with optional ratio columns.

    Column order: MNK | 1cta | 2cta | 2cta/1cta | [cublas] | [2cta/cublas]
    """
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


# Uses do_bench_cudagraph to amortise persistent-kernel launch
# overhead.  These kernels launch a full SM grid and partition work
# cooperatively, so standalone launch cost dominates at small problem
# sizes.  CUDA graph replay isolates steady-state compute throughput,
# which is the production-relevant metric.
def run_benchmark():
    results = {}
    for a_format, b_format in ALL_FORMATS:
        label = f"{a_format}-{b_format}"
        variants = get_variants(a_format, b_format)
        for MNK in MNK_VALS:
            A, B, A_scale, B_scale, VEC_SIZE = make_tensors(MNK, a_format, b_format)
            for variant in variants:
                fn = make_fn(variant, A, B, A_scale, B_scale, VEC_SIZE, a_format)
                ms = triton.testing.do_bench_cudagraph(fn)
                tflops = 2.0 * MNK**3 * 1e-12 / (ms * 1e-3)
                results[(label, variant, MNK)] = tflops

    for a_format, b_format in ALL_FORMATS:
        label = f"{a_format}-{b_format}"
        variants = get_variants(a_format, b_format)
        print_table(label, variants, MNK_VALS, results)


if __name__ == "__main__":
    run_benchmark()
