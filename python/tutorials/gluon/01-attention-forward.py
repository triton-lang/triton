import torch
import triton
import triton.language as tl
import pytest
import itertools

from typing import Union

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper.tma import tensor_descriptor
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
)

# ===-----------------------------------------------------------------------===#
# Layout Utilities
# ===-----------------------------------------------------------------------===#


@tl.constexpr_function
def get_tmem_32x32b_reg_layout(instr_shape, shape, num_warps):
    assert len(shape) == 2, "expected a 2D tensor"
    assert num_warps in [4, 8], "expected 4 or 8 warps"
    M, N, _ = instr_shape

    blocks_per_tile = [shape[0] // M, shape[1] // N]
    num_blocks = blocks_per_tile[0] * blocks_per_tile[1]

    num_warp_groups = num_warps // 4
    if M == 64:
        threads_per_warp = [16, 2]
        if num_blocks == 1:
            size_per_thread = [1, N // (num_warp_groups * 2)]
            warps_per_cta = [4, num_warp_groups]
        else:
            size_per_thread = [1, N // 2]
            warps_per_cta = [4 * min(blocks_per_tile[0], num_warp_groups)]
            warps_per_cta.append(triton.cdiv(num_warp_groups, warps_per_cta[0] // 4))
    else:
        if shape[0] > 128:
            size_per_thread = [1, N]
            threads_per_warp = [32, 1]
            warps_per_cta = [4 * num_warp_groups, 1]
        else:
            size_per_thread = [1, N // num_warp_groups]
            threads_per_warp = [32, 1]
            warps_per_cta = [4, num_warp_groups]
    return gl.BlockedLayout(
        size_per_thread=size_per_thread,
        threads_per_warp=threads_per_warp,
        warps_per_cta=warps_per_cta,
        order=[0, 1],
    )


@tl.constexpr_function
def get_mma_instr_shape(shape, element_ty):
    m = 128 if shape[0] >= 128 else 64
    n = 256 if shape[1] >= 256 else shape[1]
    k = 256 // element_ty.primitive_bitwidth
    return (m, n, k)


@tl.constexpr_function
def get_nvmma_layout(shape, element_ty, order=[1, 0], fp4_padded=False):
    packing_factor = 2 if fp4_padded else 1

    contig_dim_size = shape[order[0]] * packing_factor * element_ty.primitive_bitwidth // 8
    if contig_dim_size >= 128 and contig_dim_size % 128 == 0:
        swizzle_byte_width = 128
    elif contig_dim_size >= 64 and contig_dim_size % 64 == 0:
        swizzle_byte_width = 64
    elif contig_dim_size >= 32 and contig_dim_size % 32 == 0:
        swizzle_byte_width = 32
    else:
        swizzle_byte_width = 0

    flatten_outer_dim = 1
    for i in range(1, len(shape)):
        flatten_outer_dim *= shape[order[i]]
    if len(shape) < 2 or flatten_outer_dim < 8:
        swizzle_byte_width = 0
    transposed = order[0] == 0

    return gl.NVMMASharedLayout(
        swizzle_byte_width=swizzle_byte_width,
        element_bitwidth=element_ty.primitive_bitwidth,
        rank=len(shape),
        transposed=transposed,
        fp4_padded=fp4_padded,
    )


@tl.constexpr_function
def get_mma_reg_layout(shape, num_warps, dtype=gl.float32):
    instr_shape = get_mma_instr_shape(shape, dtype)
    return get_tmem_32x32b_reg_layout(instr_shape, shape, num_warps)


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@tl.constexpr_function
def get_load_size_bytes(desc):
    size = desc.dtype.primitive_bitwidth // 8
    for dim in desc.block_type.shape:
        size *= dim
    return size


@gluon.jit
def load_tensor_desc_to_smem(desc, coord, smem):
    size: gl.constexpr = get_load_size_bytes(desc)
    barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    mbarrier.expect(barrier, size)
    tma.async_copy_global_to_shared(desc, coord, barrier, smem)
    mbarrier.wait(barrier, 0)
    mbarrier.invalidate(barrier)


@gluon.jit
def store_smem_to_tensor_desc(desc, coord, smem):
    tma.async_copy_shared_to_global(desc, coord, smem)
    tma.store_wait(0)


# ===-----------------------------------------------------------------------===#
# Data Abstractions
# ===-----------------------------------------------------------------------===#


@tl.constexpr_function
def Ref(T):

    @gl.aggregate
    class RefType:
        value: T

        def __init__(self, value):
            self.value = value

    return RefType


def Channel(T, alloc_fn):

    @gl.aggregate
    class ChannelType:
        mem: T
        ready_bars: gl.shared_memory_descriptor
        empty_bars: gl.shared_memory_descriptor
        num_buffers: gl.constexpr
        num_consumers: gl.constexpr

        def alloc(shape: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr, num_buffers: gl.constexpr,
                  num_consumers: gl.constexpr = 1):
            mem = alloc_fn(dtype, [num_buffers] + shape, layout)
            return ChannelType._borrow(mem, shape, dtype, layout, num_buffers, num_consumers)

        def _borrow(mem, shape: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr, num_buffers: gl.constexpr,
                    num_consumers: gl.constexpr = 1):
            mem = mem._reinterpret(dtype, [num_buffers] + shape, layout)
            return ChannelType(mem, num_buffers, num_consumers)

        def __init__(self, mem, num_buffers, num_consumers):
            self.mem = mem
            self.ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            self.empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            for i in tl.static_range(num_buffers):
                mbarrier.init(self.ready_bars.index(i), count=1)
                mbarrier.init(self.empty_bars.index(i), count=num_consumers)
            self.num_buffers: gl.constexpr = num_buffers
            self.num_consumers: gl.constexpr = num_consumers

        def increment(self, index, phase):
            if self.num_buffers == 1:
                return gl.to_tensor(0), phase ^ 1
            next_index = index + 1
            rollover = next_index == self.num_buffers
            index = gl.where(rollover, 0, next_index)
            phase = gl.where(rollover, phase ^ 1, phase)
            return index, phase

        def initialize_for_consumer(self):
            for i in tl.static_range(self.num_buffers):
                mbarrier.arrive(self.ready_bars.index(i), count=1)

        def initialize_for_producer(self):
            for i in tl.static_range(self.num_buffers):
                mbarrier.arrive(self.empty_bars.index(i), count=self.num_consumers)

        def acquire_producer(self, index, phase):
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)

            mbarrier.wait(empty_bar, phase)
            return mem, ready_bar

        def acquire_consumer(self, index, phase):
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)

            mbarrier.wait(ready_bar, phase)
            return mem, empty_bar

        def create_producer(self):
            return Producer(self)

        def create_consumer(self):
            return Consumer(self)

        def release(self):
            if isinstance(self.mem, gl.shared_memory_descriptor):
                self.mem._keep_alive()
            for i in tl.static_range(self.num_buffers):
                mbarrier.invalidate(self.ready_bars.index(i))
                mbarrier.invalidate(self.empty_bars.index(i))

    @gl.aggregate
    class Producer:
        channel: ChannelType
        phase: gl.tensor
        index: gl.tensor

        def __init__(self, channel):
            self.channel = channel
            self.phase = 0
            self.index = 0

        def acquire(self):
            mem, ready_bar = self.channel.acquire_producer(self.index, self.phase)
            self.index, self.phase = self.channel.increment(self.index, self.phase)
            return mem, ready_bar

        def emplace(self, value):
            mem, ready_bar = self.acquire()
            mem.store(value)
            mbarrier.arrive(ready_bar, count=1)

    @gl.aggregate
    class Consumer:
        channel: ChannelType
        phase: gl.tensor
        index: gl.tensor

        def __init__(self, channel):
            self.channel = channel
            self.phase = 0
            self.index = 0

        def acquire(self):
            mem, empty_bar = self.channel.acquire_consumer(self.index, self.phase)
            self.index, self.phase = self.channel.increment(self.index, self.phase)
            return mem, empty_bar

        def get(self, layout: gl.constexpr):
            mem, empty_bar = self.acquire()
            value = mem.load(layout)
            mbarrier.arrive(empty_bar, count=1)
            return value

    return ChannelType, Producer, Consumer


SharedMemoryChannel, SharedMemoryProducer, SharedMemoryConsumer = Channel(gl.shared_memory_descriptor,
                                                                          gl.allocate_shared_memory)
TensorMemoryChannel, TensorMemoryProducer, TensorMemoryConsumer = Channel(tensor_memory_descriptor,
                                                                          allocate_tensor_memory)


@gl.aggregate
class LoadContext:
    desc: tensor_descriptor
    channel: SharedMemoryChannel

    def __init__(self, desc, num_buffers: gl.constexpr, num_consumers: gl.constexpr = 1):
        self.desc = desc
        shape: gl.constexpr = desc.block_type.shape
        self.channel = SharedMemoryChannel.alloc(shape, desc.dtype, gl.constexpr(desc.layout), num_buffers,
                                                 num_consumers)
        self.channel.initialize_for_producer()

    def release(self):
        self.channel.release()


@gl.aggregate
class PipelinedLoadProducer:
    desc: tensor_descriptor
    impl: SharedMemoryProducer
    offset: gl.tensor
    step: gl.constexpr

    def __init__(self, ctx, offset, step: gl.constexpr):
        self.desc = ctx.desc
        self.impl = ctx.channel.create_producer()
        self.offset = offset
        self.step: gl.constexpr = step

    def wait_and_issue_next(self):
        smem, ready_bar = self.impl.acquire()

        size: gl.constexpr = get_load_size_bytes(self.desc)
        mbarrier.expect(ready_bar, size)
        tma.async_copy_global_to_shared(self.desc, [self.offset, 0], ready_bar, smem)

        self.offset += self.step


@gl.aggregate
class MMAContext:
    shape: gl.constexpr
    instr_shape: gl.constexpr
    channel: TensorMemoryChannel

    def __init__(self, shape: gl.constexpr, num_buffers: gl.constexpr, dtype: gl.constexpr = gl.float32):
        self.shape: gl.constexpr = shape
        self.instr_shape: gl.constexpr = get_mma_instr_shape(shape, dtype)
        tmem_layout: gl.constexpr = TensorMemoryLayout((self.instr_shape[0], self.instr_shape[1]), unpacked=True)

        self.channel = TensorMemoryChannel.alloc(shape, dtype, tmem_layout, num_buffers)

    def release(self):
        self.channel.release()

    @tl.constexpr_function
    def get_reg_layout(self, num_warps):
        return get_tmem_32x32b_reg_layout(self.instr_shape, self.shape, num_warps)


@gl.aggregate
class MMAProducer:
    producer: TensorMemoryProducer

    def __init__(self, ctx):
        self.producer = ctx.channel.create_producer()

    def wait_and_issue_next(self, a, b, release_bars, use_acc):
        tmem, bar = self.producer.acquire()
        tcgen05_mma(a, b, tmem, use_acc=use_acc, mbarriers=[bar] + release_bars,
                    mbarrier_preds=[True] * (len(release_bars) + 1))


# ===-----------------------------------------------------------------------===#
# Gluon Attention
# ===-----------------------------------------------------------------------===#


@gl.aggregate
class AttentionConfig:
    qk_scale: gl.tensor
    Z: gl.tensor
    H: gl.tensor
    N_CTX: gl.tensor

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    HEAD_DIM: gl.constexpr
    GROUP_SIZE_N: gl.constexpr
    NUM_SMS: gl.constexpr
    dtype: gl.constexpr
    num_warps: gl.constexpr

    SPLIT_D_FACTOR: gl.constexpr
    SPLIT_M: gl.constexpr
    SPLIT_D: gl.constexpr

    q_shape: gl.constexpr
    k_shape: gl.constexpr
    v_shape: gl.constexpr
    qk_shape: gl.constexpr
    o_shape: gl.constexpr

    qk_tmem_layout: gl.constexpr
    o_tmem_layout: gl.constexpr
    p_tmem_layout: gl.constexpr

    qk_layout: gl.constexpr
    o_layout: gl.constexpr
    o_splitn_layout: gl.constexpr
    mi_2d_layout: gl.constexpr

    mi_use_tmem: gl.constexpr

    def __init__(self, qk_scale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, GROUP_SIZE_N, NUM_SMS, dtype, num_warps,
                 SPLIT_D_FACTOR):
        self.qk_scale = qk_scale
        self.Z = Z
        self.H = H
        self.N_CTX = N_CTX

        self.BLOCK_M: gl.constexpr = BLOCK_M
        self.BLOCK_N: gl.constexpr = BLOCK_N
        self.HEAD_DIM: gl.constexpr = HEAD_DIM
        self.GROUP_SIZE_N: gl.constexpr = GROUP_SIZE_N
        self.NUM_SMS: gl.constexpr = NUM_SMS
        self.dtype: gl.constexpr = dtype
        self.num_warps: gl.constexpr = num_warps

        self.SPLIT_D_FACTOR: gl.constexpr = SPLIT_D_FACTOR
        self.SPLIT_M: gl.constexpr = self.BLOCK_M // 2
        self.SPLIT_D: gl.constexpr = self.HEAD_DIM // self.SPLIT_D_FACTOR

        self.q_shape: gl.constexpr = [self.SPLIT_M, self.HEAD_DIM]
        self.k_shape: gl.constexpr = [self.BLOCK_N, self.HEAD_DIM]
        self.qk_shape: gl.constexpr = [self.SPLIT_M, self.BLOCK_N]
        self.v_shape: gl.constexpr = [self.BLOCK_N, self.HEAD_DIM]
        self.o_shape: gl.constexpr = [self.SPLIT_M, self.HEAD_DIM]

        qk_instr_shape: gl.constexpr = get_mma_instr_shape(self.qk_shape, gl.float32)
        o_instr_shape: gl.constexpr = get_mma_instr_shape(self.o_shape, gl.float32)
        self.qk_tmem_layout: gl.constexpr = TensorMemoryLayout((qk_instr_shape[0], qk_instr_shape[1]), unpacked=True)
        self.o_tmem_layout: gl.constexpr = TensorMemoryLayout((o_instr_shape[0], o_instr_shape[1]), unpacked=True)
        self.p_tmem_layout: gl.constexpr = TensorMemoryLayout((qk_instr_shape[0], qk_instr_shape[1]), unpacked=False)

        self.qk_layout: gl.constexpr = gl.constexpr(
            get_tmem_32x32b_reg_layout(qk_instr_shape, self.qk_shape, self.num_warps))
        self.o_layout: gl.constexpr = gl.constexpr(
            get_tmem_32x32b_reg_layout(o_instr_shape, self.o_shape, self.num_warps))
        self.o_splitn_layout: gl.constexpr = get_tmem_32x32b_reg_layout(
            (o_instr_shape[0], o_instr_shape[1] // self.SPLIT_D_FACTOR, o_instr_shape[2]),
            (self.o_shape[0], self.o_shape[1] // self.SPLIT_D_FACTOR), self.num_warps)
        self.mi_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])

        self.mi_use_tmem: gl.constexpr = True

    def get_program(self, pid_m, pid_n):
        start_m = pid_m
        off_hz = pid_n
        off_z = off_hz // self.H
        off_h = off_hz % self.H

        offset_y = off_z * (self.N_CTX * self.H) + off_h * self.N_CTX
        qo_offset_y = offset_y + start_m * self.BLOCK_M

        return AttentionProgram(self, start_m, off_hz, offset_y, qo_offset_y)


@gl.aggregate
class ProgramScheduler:
    config: AttentionConfig
    start_pid: gl.tensor
    num_pid_n: gl.tensor
    num_pid_in_group: gl.tensor
    num_tiles: gl.tensor

    def __init__(self, config):
        self.config = config
        self.start_pid = gl.program_id(0)
        num_pid_m = gl.cdiv(config.N_CTX, config.BLOCK_M)
        self.num_pid_n = config.Z * config.H
        self.num_pid_in_group = num_pid_m * config.GROUP_SIZE_N
        self.num_tiles = num_pid_m * self.num_pid_n

    def get_tile_pid(self, tile_id):
        group_id = tile_id // self.num_pid_in_group
        first_pid_n = group_id * self.config.GROUP_SIZE_N
        group_size_n = min(self.num_pid_n - first_pid_n, self.config.GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_m = (tile_id % self.num_pid_in_group) // group_size_n
        return pid_m, pid_n


@gl.aggregate
class AttentionProgram:
    config: AttentionConfig
    start_m: gl.tensor
    off_hz: gl.tensor
    offset_y: gl.tensor
    qo_offset_y: gl.tensor

    def __init__(self, config, start_m, off_hz, offset_y, qo_offset_y):
        self.config = config
        self.start_m = start_m
        self.off_hz = off_hz
        self.offset_y = offset_y
        self.qo_offset_y = qo_offset_y

    def get_loop_bounds(self, STAGE: gl.constexpr):
        BLOCK_M: gl.constexpr = self.config.BLOCK_M
        if STAGE == 1:
            lo, hi = 0, self.start_m * BLOCK_M
        elif STAGE == 2:
            lo, hi = self.start_m * BLOCK_M, (self.start_m + 1) * BLOCK_M
        else:
            lo, hi = 0, self.config.N_CTX
        return lo, hi


# ===-----------------------------------------------------------------------===#
# _gluon_attn
# ===-----------------------------------------------------------------------===#


@gl.aggregate
class InnerLoopInfo:
    qk_mma_ctx: MMAContext
    o_mma_ctx: MMAContext
    p_chnl: TensorMemoryChannel
    mi_chnl: Union[TensorMemoryChannel, SharedMemoryChannel]
    epilogue_chnl: SharedMemoryChannel

    def __init__(self, config):
        self.qk_mma_ctx = MMAContext(config.qk_shape, num_buffers=1)
        self.qk_mma_ctx.channel.initialize_for_producer()

        self.o_mma_ctx = MMAContext(config.o_shape, num_buffers=1)
        self.o_mma_ctx.channel.initialize_for_consumer()

        # QK and PV MMAs are serialized, which enables borrowing QK's memory.
        borrow_tmem = self.qk_mma_ctx.channel.mem.index(0)
        p_tmem = borrow_tmem.slice(0, config.BLOCK_N // 2)
        mi_tmem = borrow_tmem.slice(config.BLOCK_N // 2, 1)
        mi_layout: gl.constexpr = TensorMemoryLayout([config.SPLIT_M, 1], unpacked=False)

        self.p_chnl = TensorMemoryChannel._borrow(p_tmem, config.qk_shape, config.dtype, config.p_tmem_layout,
                                                  num_buffers=1, num_consumers=1)
        self.p_chnl.initialize_for_producer()

        if config.mi_use_tmem:
            self.mi_chnl = TensorMemoryChannel._borrow(mi_tmem, [config.SPLIT_M, 1], gl.float32, mi_layout,
                                                       num_buffers=1)
        else:
            self.mi_chnl = SharedMemoryChannel.alloc([config.SPLIT_M], gl.float32,
                                                     gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1)
        self.mi_chnl.initialize_for_producer()

        self.epilogue_chnl = SharedMemoryChannel.alloc([1], gl.int8, gl.constexpr(mbarrier.MBarrierLayout()),
                                                       num_buffers=1)
        self.epilogue_chnl.initialize_for_producer()

    def release(self):
        self.qk_mma_ctx.release()
        self.o_mma_ctx.release()
        self.p_chnl.release()
        self.mi_chnl.release()


@gluon.jit
def _load_inner_loop(config, prog, load_k, load_v, STAGE: gl.constexpr):
    lo, hi = prog.get_loop_bounds(STAGE)
    num_loads = (hi - lo) // config.BLOCK_N
    if num_loads != 0:
        offsetkv_y = prog.offset_y + lo
        load_k.offset = offsetkv_y
        load_v.offset = offsetkv_y

        load_k.wait_and_issue_next()
        for _ in range(num_loads - 1):
            load_k.wait_and_issue_next()
            load_v.wait_and_issue_next()
        load_v.wait_and_issue_next()


@gluon.jit
def _attn_fwd_load(config, info0, info1,  #
                   q0_load_ctx, q1_load_ctx, k_load_ctx, v_load_ctx, M, desc_o,  #
                   STAGE: gl.constexpr):
    load_q0 = PipelinedLoadProducer(q0_load_ctx, 0, 0)
    load_q1 = PipelinedLoadProducer(q1_load_ctx, 0, 0)
    load_k = PipelinedLoadProducer(k_load_ctx, 0, config.BLOCK_N)
    load_v = PipelinedLoadProducer(v_load_ctx, 0, config.BLOCK_N)

    scheduler = ProgramScheduler(config)
    for tile_id in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = config.get_program(*scheduler.get_tile_pid(tile_id))

        load_q0.offset = prog.qo_offset_y + config.SPLIT_M * 0
        load_q1.offset = prog.qo_offset_y + config.SPLIT_M * 1
        load_q0.wait_and_issue_next()
        load_q1.wait_and_issue_next()

        if STAGE & 1:
            _load_inner_loop(config, prog, load_k, load_v, STAGE=4 - STAGE)
        if STAGE & 2:
            _load_inner_loop(config, prog, load_k, load_v, STAGE=2)


@gluon.jit
def _mma_inner_loop(config, prog, q0_smem, q1_smem,  #
                    k_consumer, v_consumer, p0_consumer, p1_consumer,  #
                    qk0_producer, qk1_producer, o0_producer, o1_producer,  #
                    STAGE: gl.constexpr):
    lo, hi = prog.get_loop_bounds(STAGE)
    num_mmas = (hi - lo) // config.BLOCK_N
    if num_mmas != 0:
        qk_p_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        qk_p_phase = 0
        mbarrier.init(qk_p_bar, count=1)

        k_smem, k_bar = k_consumer.acquire()
        qk0_producer.wait_and_issue_next(q0_smem, k_smem.permute((1, 0)), [k_bar], use_acc=False)
        qk1_producer.wait_and_issue_next(q1_smem, k_smem.permute((1, 0)), [k_bar], use_acc=False)
        for _ in range(num_mmas - 1):
            v_smem, v_bar = v_consumer.acquire()
            p0_tmem, p0_bar = p0_consumer.acquire()
            o0_producer.wait_and_issue_next(p0_tmem, v_smem, [v_bar, p0_bar, qk_p_bar], use_acc=True)

            k_smem, k_bar = k_consumer.acquire()
            mbarrier.wait(qk_p_bar, qk_p_phase)
            qk_p_phase ^= 1
            qk0_producer.wait_and_issue_next(q0_smem, k_smem.permute((1, 0)), [k_bar], use_acc=False)

            p1_tmem, p1_bar = p1_consumer.acquire()
            o1_producer.wait_and_issue_next(p1_tmem, v_smem, [v_bar, p1_bar, qk_p_bar], use_acc=True)

            mbarrier.wait(qk_p_bar, qk_p_phase)
            qk_p_phase ^= 1
            qk1_producer.wait_and_issue_next(q1_smem, k_smem.permute((1, 0)), [k_bar], use_acc=False)
        v_smem, v_bar = v_consumer.acquire()
        p0_tmem, p0_bar = p0_consumer.acquire()
        o0_producer.wait_and_issue_next(p0_tmem, v_smem, [v_bar, p0_bar], use_acc=True)

        p1_tmem, p1_bar = p1_consumer.acquire()
        o1_producer.wait_and_issue_next(p1_tmem, v_smem, [v_bar, p1_bar], use_acc=True)

        mbarrier.invalidate(qk_p_bar)


@gluon.jit
def _attn_fwd_mma(config, info0, info1,  #
                  q0_load_ctx, q1_load_ctx, k_load_ctx, v_load_ctx, M, desc_o,  #
                  STAGE: gl.constexpr):
    q0_consumer = q0_load_ctx.channel.create_consumer()
    q1_consumer = q1_load_ctx.channel.create_consumer()
    k_consumer = k_load_ctx.channel.create_consumer()
    v_consumer = v_load_ctx.channel.create_consumer()
    p0_consumer = info0.p_chnl.create_consumer()
    p1_consumer = info1.p_chnl.create_consumer()

    qk0_producer = MMAProducer(info0.qk_mma_ctx)
    qk1_producer = MMAProducer(info1.qk_mma_ctx)
    o0_producer = MMAProducer(info0.o_mma_ctx)
    o1_producer = MMAProducer(info1.o_mma_ctx)

    scheduler = ProgramScheduler(config)
    for tile_id in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = config.get_program(*scheduler.get_tile_pid(tile_id))

        q0_smem, q0_bar = q0_consumer.acquire()
        q1_smem, q1_bar = q1_consumer.acquire()
        if STAGE & 1:
            _mma_inner_loop(config, prog, q0_smem, q1_smem,  #
                            k_consumer, v_consumer, p0_consumer, p1_consumer,  #
                            qk0_producer, qk1_producer, o0_producer, o1_producer,  #
                            STAGE=4 - STAGE)
        if STAGE & 2:
            _mma_inner_loop(config, prog, q0_smem, q1_smem,  #
                            k_consumer, v_consumer, p0_consumer, p1_consumer,  #
                            qk0_producer, qk1_producer, o0_producer, o1_producer,  #
                            STAGE=2)
        tcgen05_commit(q0_bar)
        tcgen05_commit(q1_bar)

        _, o0_bar = o0_producer.producer.acquire()
        mbarrier.arrive(o0_bar, count=1)
        _, o1_bar = o1_producer.producer.acquire()
        mbarrier.arrive(o1_bar, count=1)


@gluon.jit
def _add_f32x2(a, b):
    return gl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            add.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


@gluon.jit
def _mul_f32x2(a, b):
    return gl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


@gluon.jit
def _attn_fwd_correction_compute(config, mi_consumer, o_consumer, m_i, zero_init):
    mi_layout: gl.constexpr = gl.SliceLayout(1, config.o_splitn_layout)
    if config.mi_use_tmem:
        m_ij = mi_consumer.get(config.mi_2d_layout)
        m_ij = gl.convert_layout(m_ij.reshape([config.SPLIT_M]), mi_layout)
    else:
        m_ij = mi_consumer.get(mi_layout)
    alpha = gl.exp2(m_i - m_ij)

    o_tmem, o_bar = o_consumer.acquire()
    if config.SPLIT_D_FACTOR == 1:
        if zero_init.value:
            zero_init.value = False
            o = gl.full(config.o_shape, 0, gl.float32, config.o_layout)
        else:
            o = o_tmem.load(config.o_layout)
            o = _mul_f32x2(o, alpha[:, None])
        o_tmem.store(o)
    else:
        for i in tl.static_range(config.SPLIT_D_FACTOR):
            o_ref = o_tmem.slice(i * config.SPLIT_D, config.SPLIT_D)
            if zero_init.value:
                zero_init.value = False
                o = gl.full([config.SPLIT_M, config.SPLIT_D], 0, gl.float32, config.o_splitn_layout)
            else:
                o = o_ref.load(config.o_splitn_layout)
                o = _mul_f32x2(o, alpha[:, None])
            o_ref.store(o)
    mbarrier.arrive(o_bar, count=1)
    return m_ij


@gluon.jit
def _correction_inner_loop(config, prog, m_i0, m_i1, zero_init0, zero_init1,  #
                           mi0_consumer, mi1_consumer, o0_consumer, o1_consumer,  #
                           STAGE: gl.constexpr):
    lo, hi = prog.get_loop_bounds(STAGE)
    for start_n in range(lo, hi, config.BLOCK_N):
        m_i0 = _attn_fwd_correction_compute(config, mi0_consumer, o0_consumer, m_i0, zero_init0)
        m_i1 = _attn_fwd_correction_compute(config, mi1_consumer, o1_consumer, m_i1, zero_init1)
    return m_i0, m_i1


@gluon.jit
def _attn_fwd_correction(config, info0, info1, STAGE: gl.constexpr):
    m_i0 = gl.full([config.SPLIT_M], -float("inf"), gl.float32, gl.SliceLayout(1, config.o_splitn_layout))
    m_i1 = gl.full([config.SPLIT_M], -float("inf"), gl.float32, gl.SliceLayout(1, config.o_splitn_layout))

    o0_consumer = info0.o_mma_ctx.channel.create_consumer()
    o1_consumer = info1.o_mma_ctx.channel.create_consumer()
    mi0_consumer = info0.mi_chnl.create_consumer()
    mi1_consumer = info1.mi_chnl.create_consumer()

    epilogue0_producer = info0.epilogue_chnl.create_producer()
    epilogue1_producer = info1.epilogue_chnl.create_producer()

    _, e0_bar = epilogue0_producer.acquire()
    _, e1_bar = epilogue1_producer.acquire()

    scheduler = ProgramScheduler(config)
    for tile_id in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = config.get_program(*scheduler.get_tile_pid(tile_id))

        zero_init0 = Ref(gl.tensor)(gl.to_tensor(True))
        zero_init1 = Ref(gl.tensor)(gl.to_tensor(True))

        if STAGE & 1:
            m_i0, m_i1 = _correction_inner_loop(  #
                config, prog, m_i0, m_i1, zero_init0, zero_init1,  #
                mi0_consumer, mi1_consumer, o0_consumer, o1_consumer,  #
                STAGE=4 - STAGE)
        if STAGE & 2:
            m_i0, m_i1 = _correction_inner_loop(  #
                config, prog, m_i0, m_i1, zero_init0, zero_init1,  #
                mi0_consumer, mi1_consumer, o0_consumer, o1_consumer,  #
                STAGE=2)

        o0_tmem, o0_bar = o0_consumer.acquire()
        mbarrier.arrive(e0_bar, count=1)
        o1_tmem, o1_bar = o1_consumer.acquire()
        mbarrier.arrive(e1_bar, count=1)

        _, e0_bar = epilogue0_producer.acquire()
        mbarrier.arrive(o0_bar, count=1)
        _, e1_bar = epilogue1_producer.acquire()
        mbarrier.arrive(o1_bar, count=1)


@gluon.jit
def _softmax_inner_loop(tile_id: gl.constexpr, config, prog,  #
                        qk_consumer, p_producer, mi_producer,  #
                        offs_m, offs_n, m_i, l_i,  #
                        STAGE: gl.constexpr):
    lo, hi = prog.get_loop_bounds(STAGE)

    for start_n in range(lo, hi, config.BLOCK_N):
        qk = qk_consumer.get(config.qk_layout)
        if config.HEAD_DIM == 128:
            p_tmem, p_bar = p_producer.acquire()

        if STAGE == 2:
            # Prevent LLVM from hoisting the partial sums, which triggers spilling.
            offs_n = gl.inline_asm_elementwise("mov.b32 $0, $0;", "=r,r", [offs_n], dtype=gl.int32, is_pure=True,
                                               pack=1)
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = gl.where(mask, qk, -1.0e8)
        m_ij = gl.maximum(m_i, gl.max(qk, 1) * config.qk_scale)
        if config.mi_use_tmem:
            mi_producer.emplace(gl.convert_layout(m_ij.expand_dims(1), config.mi_2d_layout))
        else:
            mi_producer.emplace(m_ij)
        qk = qk * config.qk_scale - m_ij[:, None]

        if config.HEAD_DIM == 64:
            p = gl.exp2(qk)
            l_ij = gl.sum(p, 1)
            alpha = gl.exp2(m_i - m_ij)
            p_producer.emplace(p.to(config.dtype))
        else:
            qk0, qk1, = qk.reshape([config.SPLIT_M, 2, config.BLOCK_N // 2]).permute(0, 2, 1).split()
            p0 = gl.exp2(qk0)
            p_tmem.slice(0, config.BLOCK_N // 2).store(p0.to(config.dtype))
            p1 = gl.exp2(qk1)
            p_tmem.slice(config.BLOCK_N // 2, config.BLOCK_N // 2).store(p1.to(config.dtype))
            mbarrier.arrive(p_bar, count=1)
            p = gl.join(p0, p1).permute(0, 2, 1).reshape([config.SPLIT_M, config.BLOCK_N])
            p = gl.convert_layout(p, config.qk_layout)

            l_ij = gl.sum(p, 1)
            alpha = gl.exp2(m_i - m_ij)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return m_i, l_i


@gluon.jit
def _softmax_tile(tile_id: gl.constexpr, config, info, M, desc_o, STAGE: gl.constexpr):
    qk_slice_dim0: gl.constexpr = gl.SliceLayout(0, config.qk_layout)
    qk_slice_dim1: gl.constexpr = gl.SliceLayout(1, config.qk_layout)

    qk_consumer = info.qk_mma_ctx.channel.create_consumer()
    p_producer = info.p_chnl.create_producer()
    mi_producer = info.mi_chnl.create_producer()
    epilogue_consumer = info.epilogue_chnl.create_consumer()

    offs_n = gl.arange(0, config.BLOCK_N, qk_slice_dim0)

    o_smem = gl.allocate_shared_memory(config.dtype, config.o_shape, desc_o.layout)

    scheduler = ProgramScheduler(config)
    for tid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = config.get_program(*scheduler.get_tile_pid(tid))

        offs_m = prog.start_m * config.BLOCK_M
        offs_m += gl.arange(tile_id * config.SPLIT_M, (1 + tile_id) * config.SPLIT_M, qk_slice_dim1)

        m_i = gl.full([config.SPLIT_M], -float("inf"), gl.float32, qk_slice_dim1)
        l_i = gl.full([config.SPLIT_M], 1.0, gl.float32, qk_slice_dim1)

        if STAGE & 1:
            m_i, l_i = _softmax_inner_loop(tile_id, config, prog,  #
                                           qk_consumer, p_producer, mi_producer,  #
                                           offs_m, offs_n, m_i, l_i,  #
                                           STAGE=4 - STAGE)
        if STAGE & 2:
            m_i, l_i = _softmax_inner_loop(tile_id, config, prog,  #
                                           qk_consumer, p_producer, mi_producer,  #
                                           offs_m, offs_n, m_i, l_i,  #
                                           STAGE=2)

        _, e_bar = epilogue_consumer.acquire()
        o_tmem = info.o_mma_ctx.channel.mem.index(0)
        o = o_tmem.load(config.o_layout)
        mbarrier.arrive(e_bar, count=1)

        m_i += gl.log2(l_i)
        o = o / gl.convert_layout(l_i, gl.SliceLayout(1, config.o_layout))[:, None]

        coalesced: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        offs_m = prog.start_m * config.BLOCK_M
        offs_m += gl.arange(tile_id * config.SPLIT_M, (tile_id + 1) * config.SPLIT_M, coalesced)

        m_ptrs = M + prog.off_hz * config.N_CTX + offs_m
        gl.store(m_ptrs, gl.convert_layout(m_i, coalesced))
        o_smem.store(o.to(config.dtype))
        fence_async_shared()
        store_smem_to_tensor_desc(desc_o, [prog.qo_offset_y + config.SPLIT_M * tile_id, 0], o_smem)


@gluon.jit
def _attn_fwd_softmax0(config, info0, info1,  #
                       q0_load_ctx, q1_load_ctx, k_load_ctx, v_load_ctx, M, desc_o,  #
                       STAGE: gl.constexpr):
    _softmax_tile(0, config, info0, M, desc_o, STAGE)


@gluon.jit
def _attn_fwd_softmax1(config, info0, info1,  #
                       q0_load_ctx, q1_load_ctx, k_load_ctx, v_load_ctx, M, desc_o,  #
                       STAGE: gl.constexpr):
    _softmax_tile(1, config, info1, M, desc_o, STAGE)


@gluon.jit
def _attn_fwd_inner(config, info0, info1,  #
                    desc_q, desc_k, desc_v, desc_o, M,  #
                    STAGE: gl.constexpr):
    num_buffers: gl.constexpr = 2 if config.HEAD_DIM == 128 else 3
    q0_load_ctx = LoadContext(desc_q, num_buffers=1)
    q1_load_ctx = LoadContext(desc_q, num_buffers=1)
    k_load_ctx = LoadContext(desc_k, num_buffers=num_buffers, num_consumers=2)
    v_load_ctx = LoadContext(desc_v, num_buffers=num_buffers, num_consumers=2)

    gl.warp_specialize((
        config,
        info0,
        info1,
        STAGE,
    ), _attn_fwd_correction, (
        config,
        info0,
        info1,
        q0_load_ctx,
        q1_load_ctx,
        k_load_ctx,
        v_load_ctx,
        M,
        desc_o,
        STAGE,
    ), [
        _attn_fwd_softmax0,
        _attn_fwd_softmax1,
        _attn_fwd_mma,
        _attn_fwd_load,
    ], [4, 4, 1, 1], [192, 192, 32, 32])

    q0_load_ctx.release()
    q1_load_ctx.release()
    k_load_ctx.release()
    v_load_ctx.release()


@gluon.jit(do_not_specialize=["Z"])
def _gluon_attn(sm_scale, M, Z, H, N_CTX,  #
                desc_q, desc_k, desc_v, desc_o,  #
                BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
                GROUP_SIZE_N: gl.constexpr, NUM_SMS: gl.constexpr,  #
                STAGE: gl.constexpr, dtype: gl.constexpr,  #
                num_warps: gl.constexpr):
    qk_scale = sm_scale
    qk_scale *= 1.44269504
    config = AttentionConfig(qk_scale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, GROUP_SIZE_N, NUM_SMS,  # i
                             dtype, num_warps, SPLIT_D_FACTOR=2)

    info0 = InnerLoopInfo(config)
    info1 = InnerLoopInfo(config)

    _attn_fwd_inner(config, info0, info1, desc_q, desc_k, desc_v, desc_o, M, STAGE)


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


def torch_dtype_to_triton(dtype):
    if dtype == torch.float8_e5m2:
        return gl.float8e5
    return getattr(tl, str(dtype).split('.')[1])


def make_tensor_desc(x, shape, strides, block_shape):
    layout = get_nvmma_layout(block_shape, torch_dtype_to_triton(x.dtype))
    return TensorDescriptor(x, shape=shape, strides=strides, block_shape=block_shape, layout=layout.value)


def attention_forward(q, k, v, causal, sm_scale):
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    stage = 3 if causal else 1

    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    # The kernel will split BLOCK_M into two subtiles.
    BLOCK_M = 256
    BLOCK_N = 128
    SPLIT_M = BLOCK_M // 2
    GROUP_SIZE_N = 8
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    desc_q = make_tensor_desc(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[SPLIT_M, HEAD_DIM_K])
    desc_v = make_tensor_desc(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_k = make_tensor_desc(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_o = make_tensor_desc(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[SPLIT_M, HEAD_DIM_K])

    num_pid_m = triton.cdiv(q.shape[2], BLOCK_M)
    num_pid_n = q.shape[0] * q.shape[1]
    grid = min(NUM_SMS, num_pid_m * num_pid_n)

    _gluon_attn[(grid, )](
        sm_scale, M, q.shape[0], q.shape[1], q.shape[2],  #
        desc_q, desc_k, desc_v, desc_o,  #
        BLOCK_M, BLOCK_N, HEAD_DIM_K, GROUP_SIZE_N, NUM_SMS,  #
        stage, torch_dtype_to_triton(q.dtype),  #
        num_warps=4)

    return o, M


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [2, 48])
@pytest.mark.parametrize("N_CTX", [256, 1024, 4 * 1024])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(not is_blackwell(), reason="Gluon attention is only supported on Blackwell GPUs")
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype):
    device = "cuda"

    torch.manual_seed(42)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5

    M = torch.tril(torch.ones((N_CTX, N_CTX), device=device))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)

    tri_out, _ = attention_forward(q, k, v, causal, sm_scale)
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)


# ===-----------------------------------------------------------------------===#
# Benchmarking
# ===-----------------------------------------------------------------------===#

BATCH = [4]
N_HEADS = [32]
HEAD_DIM = [64, 128]
causal = [False, True]
providers = ["triton-fp16", "triton-fp8"]
N_CTX = [2**i for i in range(10, 17)]

bench_configs = []
for Z, H, D, is_causal in itertools.product(BATCH, N_HEADS, HEAD_DIM, causal):
    config = triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=N_CTX,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")],
        ylabel="TFLOPS",
        plot_name=f"Attention Z={Z} H={H} D={D} causal={is_causal}",
        args={
            "Z": Z,
            "H": H,
            "HEAD_DIM": D,
            "causal": is_causal,
        },
    )
    bench_configs.append(config)


@triton.testing.perf_report(bench_configs)
def bench(Z, H, N_CTX, HEAD_DIM, causal, provider):
    provider, dtype = provider.split("-")
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp8":
        dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    device = "cuda"

    torch.manual_seed(42)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), device=device).normal_(mean=0.0, std=0.5).requires_grad_()).to(dtype)
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), device=device).normal_(mean=0.0, std=0.5).requires_grad_()).to(dtype)
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), device=device).normal_(mean=0.0, std=0.5).requires_grad_()).to(dtype)
    sm_scale = 1.3

    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
        if provider == "triton":
            fn = lambda: attention_forward(q, k, v, causal, sm_scale)
        elif provider == "cudnn":
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        ms = triton.testing.do_bench(fn)
        flops_per_matmul = 2.0 * Z * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench.run(save_path=".", print_data=True)
