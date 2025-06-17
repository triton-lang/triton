import torch
import triton
import triton.language as tl
import pytest
import itertools

from triton.language.core import _aggregate as aggregate

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


def Channel(T, alloc_fn):

    @aggregate
    class ChannelType:
        mem: T
        ready_bars: gl.shared_memory_descriptor
        empty_bars: gl.shared_memory_descriptor
        num_buffers: gl.constexpr
        num_consumers: gl.constexpr

        @gluon.jit
        def create(shape: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr, num_buffers: gl.constexpr,
                   num_consumers: gl.constexpr = 1):
            mem = alloc_fn(dtype, [num_buffers] + shape, layout)
            return ChannelType._borrow(mem, shape, dtype, layout, num_buffers, num_consumers)

        @gluon.jit
        def _borrow(mem, shape: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr, num_buffers: gl.constexpr,
                    num_consumers: gl.constexpr = 1):
            mem = mem._reinterpret(dtype, [num_buffers] + shape, layout)
            ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            for i in tl.static_range(num_buffers):
                mbarrier.init(ready_bars.index(i), count=1)
                mbarrier.init(empty_bars.index(i), count=num_consumers)
            return ChannelType(mem, ready_bars, empty_bars, num_buffers, num_consumers)

        @gluon.jit
        def increment(self, index, phase):
            if self.num_buffers == 1:
                return gl.to_tensor(0), phase ^ 1
            next_index = index + 1
            rollover = next_index == self.num_buffers
            index = gl.where(rollover, 0, next_index)
            phase = gl.where(rollover, phase ^ 1, phase)
            return index, phase

        def __init__(self, mem, ready_bars, empty_bars, num_buffers, num_consumers):
            self.mem = mem
            self.ready_bars = ready_bars
            self.empty_bars = empty_bars
            self.num_buffers = gl.constexpr(num_buffers)
            self.num_consumers = gl.constexpr(num_consumers)

        @gluon.jit
        def initialize_for_consumer(self):
            for i in tl.static_range(self.num_buffers):
                mbarrier.arrive(self.ready_bars.index(i), count=1)

        @gluon.jit
        def initialize_for_producer(self):
            for i in tl.static_range(self.num_buffers):
                mbarrier.arrive(self.empty_bars.index(i), count=self.num_consumers)

        @gluon.jit
        def acquire_producer(self, index, phase):
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)

            mbarrier.wait(empty_bar, phase)
            return mem, ready_bar

        @gluon.jit
        def acquire_consumer(self, index, phase):
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)

            mbarrier.wait(ready_bar, phase)
            return mem, empty_bar

        @gluon.jit
        def create_producer(self):
            return Producer(self, gl.to_tensor(0), gl.to_tensor(0))

        @gluon.jit
        def create_consumer(self):
            return Consumer(self, gl.to_tensor(0), gl.to_tensor(0))

        @gluon.jit
        def release(self):
            if isinstance(self.mem, gl.shared_memory_descriptor):
                self.mem._keep_alive()
            for i in tl.static_range(self.num_buffers):
                mbarrier.invalidate(self.ready_bars.index(i))
                mbarrier.invalidate(self.empty_bars.index(i))

    @aggregate
    class Producer:
        channel: ChannelType
        phase: gl.tensor
        index: gl.tensor

        def __init__(self, channel, phase, index):
            self.channel = channel
            self.phase = phase
            self.index = index

        @gluon.jit
        def acquire(self):
            smem, ready_bar = self.channel.acquire_producer(self.index, self.phase)
            self.index, self.phase = self.channel.increment(self.index, self.phase)
            return smem, ready_bar, self

        @gluon.jit
        def emplace(self, value):
            smem, ready_bar, self = self.acquire()
            smem.store(value)
            mbarrier.arrive(ready_bar, count=1)
            return self

    @aggregate
    class Consumer:
        channel: ChannelType
        phase: gl.tensor
        index: gl.tensor

        def __init__(self, channel, phase, index):
            self.channel = channel
            self.phase = phase
            self.index = index

        @gluon.jit
        def acquire(self):
            smem, empty_bar = self.channel.acquire_consumer(self.index, self.phase)
            self.index, self.phase = self.channel.increment(self.index, self.phase)
            return smem, empty_bar, self

        @gluon.jit
        def get(self, layout: gl.constexpr):
            smem, empty_bar, self = self.acquire()
            value = smem.load(layout)
            mbarrier.arrive(empty_bar, count=1)
            return value, self

    return ChannelType, Producer, Consumer


SharedMemoryChannel, SharedMemoryProducer, SharedMemoryConsumer = Channel(gl.shared_memory_descriptor,
                                                                          gl.allocate_shared_memory)
TensorMemoryChannel, TensorMemoryProducer, TensorMemoryConsumer = Channel(tensor_memory_descriptor,
                                                                          allocate_tensor_memory)


@aggregate
class LoadContext:
    desc: tensor_descriptor
    channel: SharedMemoryChannel

    @gluon.jit
    def create(desc, num_buffers: gl.constexpr, num_consumers: gl.constexpr = 1):
        shape: gl.constexpr = desc.block_type.shape
        channel = SharedMemoryChannel.create(shape, desc.dtype, gl.constexpr(desc.layout), num_buffers, num_consumers)
        channel.initialize_for_producer()
        return LoadContext(desc, channel)

    def __init__(self, desc, channel):
        self.desc = desc
        self.channel = channel

    @gluon.jit
    def release(self):
        self.channel.release()


@aggregate
class PipelinedLoadProducer:
    desc: tensor_descriptor
    impl: SharedMemoryProducer
    offset: gl.tensor
    step: gl.constexpr

    @gluon.jit
    def create(ctx, offset, step: gl.constexpr):
        return PipelinedLoadProducer(ctx.desc, ctx.channel.create_producer(), offset, step)

    def __init__(self, desc, impl, offset, step):
        self.desc = desc
        self.impl = impl
        self.offset = offset
        self.step = gl.constexpr(step)

    @gluon.jit
    def wait_and_issue_next(self):
        smem, ready_bar, self.impl = self.impl.acquire()

        size: gl.constexpr = get_load_size_bytes(self.desc)
        mbarrier.expect(ready_bar, size)
        tma.async_copy_global_to_shared(self.desc, [self.offset, 0], ready_bar, smem)

        self.offset += self.step
        return self


@aggregate
class MMAContext:
    channel: TensorMemoryChannel
    instr_shape: gl.constexpr
    shape: gl.constexpr

    @gluon.jit
    def create(shape: gl.constexpr, num_buffers: gl.constexpr, dtype: gl.constexpr = gl.float32):
        instr_shape: gl.constexpr = get_mma_instr_shape(shape, dtype)
        tmem_layout: gl.constexpr = TensorMemoryLayout((instr_shape[0], instr_shape[1]), unpacked=True)

        channel = TensorMemoryChannel.create(shape, dtype, tmem_layout, num_buffers)
        return MMAContext(channel, instr_shape, shape)

    def __init__(self, channel, instr_shape, shape):
        self.channel = channel
        self.instr_shape = gl.constexpr(instr_shape)
        self.shape = gl.constexpr(shape)

    @gluon.jit
    def release(self):
        self.channel.release()

    @tl.constexpr_function
    def get_reg_layout(self, num_warps):
        return get_tmem_32x32b_reg_layout(self.instr_shape, self.shape, num_warps)


@aggregate
class MMAProducer:
    producer: TensorMemoryProducer

    @gluon.jit
    def create(ctx):
        return MMAProducer(ctx.channel.create_producer())

    def __init__(self, producer):
        self.producer = producer

    @gluon.jit
    def wait_and_issue_next(self, a, b, release_bars, use_acc):
        tmem, bar, self.producer = self.producer.acquire()
        tcgen05_mma(a, b, tmem, use_acc=use_acc, mbarriers=[bar] + release_bars,
                    mbarrier_preds=[True] * (len(release_bars) + 1))
        return self


# ===-----------------------------------------------------------------------===#
# Gluon Attention
# ===-----------------------------------------------------------------------===#


@aggregate
class AttentionConfig:
    qk_scale: gl.tensor
    Z: gl.tensor
    H: gl.tensor
    N_CTX: gl.tensor
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    HEAD_DIM: gl.constexpr
    dtype: gl.constexpr
    num_warps: gl.constexpr

    SPLIT_N_FACTOR: gl.constexpr
    SPLIT_M: gl.constexpr
    SPLIT_N: gl.constexpr

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

    def __init__(self, qk_scale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, dtype, num_warps, SPLIT_M_FACTOR,
                 SPLIT_N_FACTOR):
        self.qk_scale = qk_scale
        self.Z = Z
        self.H = H
        self.N_CTX = N_CTX
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.dtype = gl.constexpr(dtype)
        self.num_warps = gl.constexpr(num_warps)

        self.SPLIT_N_FACTOR = SPLIT_N_FACTOR
        self.SPLIT_M = self.BLOCK_M // SPLIT_M_FACTOR
        self.SPLIT_N = self.BLOCK_N // self.SPLIT_N_FACTOR

        self.q_shape = gl.constexpr([self.SPLIT_M, self.HEAD_DIM])
        self.k_shape = gl.constexpr([self.BLOCK_N, self.HEAD_DIM])
        self.qk_shape = gl.constexpr([self.SPLIT_M, self.BLOCK_N])
        self.v_shape = gl.constexpr([self.BLOCK_N, self.HEAD_DIM])
        self.o_shape = gl.constexpr([self.SPLIT_M, self.HEAD_DIM])

        qk_instr_shape = get_mma_instr_shape(self.qk_shape, gl.float32)
        o_instr_shape = get_mma_instr_shape(self.o_shape, gl.float32)
        self.qk_tmem_layout = gl.constexpr(TensorMemoryLayout((qk_instr_shape[0], qk_instr_shape[1]), unpacked=True))
        self.o_tmem_layout = gl.constexpr(TensorMemoryLayout((o_instr_shape[0], o_instr_shape[1]), unpacked=True))
        self.p_tmem_layout = gl.constexpr(TensorMemoryLayout((qk_instr_shape[0], qk_instr_shape[1]), unpacked=False))

        self.qk_layout = gl.constexpr(get_tmem_32x32b_reg_layout(qk_instr_shape, self.qk_shape, self.num_warps))
        self.o_layout = gl.constexpr(get_tmem_32x32b_reg_layout(o_instr_shape, self.o_shape, self.num_warps))
        self.o_splitn_layout = gl.constexpr(
            get_tmem_32x32b_reg_layout((o_instr_shape[0], o_instr_shape[1] // self.SPLIT_N_FACTOR, o_instr_shape[2]),
                                       (self.o_shape[0], self.o_shape[1] // self.SPLIT_N_FACTOR), self.num_warps))

    @gluon.jit
    def get_program(self):
        start_m = gl.program_id(0)
        off_hz = gl.program_id(1)
        off_z = off_hz // self.H
        off_h = off_hz % self.H

        offset_y = off_z * (self.N_CTX * self.H) + off_h * self.N_CTX
        qo_offset_y = offset_y + start_m * self.BLOCK_M

        return AttentionProgram(self, start_m, off_hz, offset_y, qo_offset_y)


@aggregate
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

    @gluon.jit
    def get_loop_bounds(self, STAGE: gl.constexpr):
        BLOCK_M: gl.constexpr = self.config.BLOCK_M
        if STAGE == 1:
            lo, hi = 0, self.start_m * BLOCK_M
        elif STAGE == 2:
            lo, hi = self.start_m * BLOCK_M, (self.start_m + 1) * BLOCK_M
        else:
            lo, hi = 0, self.config.N_CTX
        return lo, hi


@aggregate
class AttentionTile:
    acc: gl.tensor
    m_i: gl.tensor
    l_i: gl.tensor
    q_smem: gl.shared_memory_descriptor

    @gluon.jit
    def create(config):
        acc = gl.full([config.SPLIT_M, config.HEAD_DIM], 0, gl.float32, config.o_layout)
        m_i = gl.full([config.SPLIT_M], -float("inf"), gl.float32, gl.SliceLayout(1, config.o_splitn_layout))
        l_i = gl.full([config.SPLIT_M], 1.0, gl.float32, gl.SliceLayout(1, config.qk_layout))

        smem_layout: gl.constexpr = get_nvmma_layout((config.SPLIT_M, config.HEAD_DIM), config.dtype)
        q_smem = gl.allocate_shared_memory(config.dtype, (config.SPLIT_M, config.HEAD_DIM), smem_layout)
        return AttentionTile(acc, m_i, l_i, q_smem)

    def __init__(self, acc, m_i, l_i, q_smem):
        self.acc = acc
        self.m_i = m_i
        self.l_i = l_i
        self.q_smem = q_smem

    @gluon.jit
    def do_epilogue(self, tile_id: gl.constexpr, M, o_smem, desc_o, prog, config):
        self.m_i += gl.convert_layout(gl.log2(self.l_i), self.m_i.type.layout)
        o = self.acc / self.l_i[:, None]

        coalesced: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        offs_m = prog.start_m * config.BLOCK_M
        offs_m += gl.arange(tile_id * config.SPLIT_M, (tile_id + 1) * config.SPLIT_M, coalesced)

        m_ptrs = M + prog.off_hz * config.N_CTX + offs_m
        gl.store(m_ptrs, gl.convert_layout(self.m_i, coalesced))
        o_smem.store(o.to(config.dtype))
        fence_async_shared()
        store_smem_to_tensor_desc(desc_o, [prog.qo_offset_y + config.SPLIT_M * tile_id, 0], o_smem)


@gluon.jit
def _attn_fwd_load(config, k_load_ctx, v_load_ctx, STAGE: gl.constexpr):
    prog = config.get_program()
    lo, hi = prog.get_loop_bounds(STAGE)

    offsetkv_y = prog.offset_y + lo
    load_k = PipelinedLoadProducer.create(k_load_ctx, offsetkv_y, config.BLOCK_N)
    load_v = PipelinedLoadProducer.create(v_load_ctx, offsetkv_y, config.BLOCK_N)

    num_loads = (hi - lo) // config.BLOCK_N
    if num_loads > 0:
        load_k = load_k.wait_and_issue_next()
        for _ in range(num_loads - 1):
            load_k = load_k.wait_and_issue_next()
            load_v = load_v.wait_and_issue_next()
        load_v = load_v.wait_and_issue_next()


# ===-----------------------------------------------------------------------===#
# _gluon_attn_d128
# ===-----------------------------------------------------------------------===#


@aggregate
class InnerLoopInfo:
    qk_mma_ctx: MMAContext
    o_mma_ctx: MMAContext
    p_chnl: TensorMemoryChannel
    mi_chnl: SharedMemoryChannel
    li_smem: gl.shared_memory_descriptor
    q_smem: gl.shared_memory_descriptor

    @gluon.jit
    def create(config, tile):
        qk_mma_ctx = MMAContext.create(config.qk_shape, num_buffers=1)
        qk_mma_ctx.channel.initialize_for_producer()

        o_mma_ctx = MMAContext.create(config.o_shape, num_buffers=1)
        o_mma_ctx.channel.initialize_for_consumer()
        o_mma_ctx.channel.mem.index(0).store(tile.acc)

        p_chnl = TensorMemoryChannel._borrow(qk_mma_ctx.channel.mem, config.qk_shape, config.dtype,
                                             config.p_tmem_layout, num_buffers=1, num_consumers=1)
        p_chnl.initialize_for_producer()

        mi_chnl = SharedMemoryChannel.create([config.SPLIT_M], gl.float32, gl.constexpr(mbarrier.MBarrierLayout()),
                                             num_buffers=1)
        mi_chnl.initialize_for_producer()
        mi_chnl.mem.index(0).store(tile.m_i)

        li_smem = gl.allocate_shared_memory(gl.float32, [config.SPLIT_M], gl.constexpr(mbarrier.MBarrierLayout()))
        li_smem.store(tile.l_i)

        return InnerLoopInfo(qk_mma_ctx, o_mma_ctx, p_chnl, mi_chnl, li_smem, tile.q_smem)

    def __init__(self, qk_mma_ctx, o_mma_ctx, p_chnl, mi_chnl, li_smem, q_smem):
        self.qk_mma_ctx = qk_mma_ctx
        self.o_mma_ctx = o_mma_ctx
        self.p_chnl = p_chnl
        self.mi_chnl = mi_chnl
        self.li_smem = li_smem
        self.q_smem = q_smem

    @gluon.jit
    def consume_result(self, tile):
        tile.acc = self.o_mma_ctx.channel.mem.index(0).load(tile.acc.type.layout)
        tile.l_i = self.li_smem.load(tile.l_i.type.layout)

        self.qk_mma_ctx.release()
        self.o_mma_ctx.release()
        self.p_chnl.release()
        self.mi_chnl.release()
        self.li_smem._keep_alive()
        return tile


@gluon.jit
def _attn_fwd_load_d128(config,  #
                        m_is, infos, k_load_ctx, v_load_ctx,  #
                        STAGE: gl.constexpr):
    _attn_fwd_load(config, k_load_ctx, v_load_ctx, STAGE)


@gluon.jit
def _attn_fwd_mma_d128(config,  #
                       m_is, infos, k_load_ctx, v_load_ctx,  #
                       STAGE: gl.constexpr):
    prog = config.get_program()
    lo, hi = prog.get_loop_bounds(STAGE)
    info0 = infos[0]
    info1 = infos[1]

    k_consumer = k_load_ctx.channel.create_consumer()
    v_consumer = v_load_ctx.channel.create_consumer()
    qk0_producer = MMAProducer.create(info0.qk_mma_ctx)
    qk1_producer = MMAProducer.create(info1.qk_mma_ctx)
    o0_producer = MMAProducer.create(info0.o_mma_ctx)
    o1_producer = MMAProducer.create(info1.o_mma_ctx)
    p0_consumer = info0.p_chnl.create_consumer()
    p1_consumer = info1.p_chnl.create_consumer()

    qk_p_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    qk_p_phase = 0
    mbarrier.init(qk_p_bar, count=1)

    num_mmas = (hi - lo) // config.BLOCK_N
    if num_mmas > 0:
        k_smem, k_bar, k_consumer = k_consumer.acquire()
        qk0_producer = qk0_producer.wait_and_issue_next(info0.q_smem, k_smem.permute((1, 0)), [k_bar], use_acc=False)
        qk1_producer = qk1_producer.wait_and_issue_next(info1.q_smem, k_smem.permute((1, 0)), [k_bar], use_acc=False)
        for _ in range(num_mmas - 1):
            v_smem, v_bar, v_consumer = v_consumer.acquire()
            p0_tmem, p0_bar, p0_consumer = p0_consumer.acquire()
            o0_producer = o0_producer.wait_and_issue_next(p0_tmem, v_smem, [v_bar, p0_bar, qk_p_bar], use_acc=True)

            k_smem, k_bar, k_consumer = k_consumer.acquire()
            mbarrier.wait(qk_p_bar, qk_p_phase)
            qk_p_phase ^= 1
            qk0_producer = qk0_producer.wait_and_issue_next(info0.q_smem, k_smem.permute((1, 0)), [k_bar],
                                                            use_acc=False)

            p1_tmem, p1_bar, p1_consumer = p1_consumer.acquire()
            o1_producer = o1_producer.wait_and_issue_next(p1_tmem, v_smem, [v_bar, p1_bar, qk_p_bar], use_acc=True)

            mbarrier.wait(qk_p_bar, qk_p_phase)
            qk_p_phase ^= 1
            qk1_producer = qk1_producer.wait_and_issue_next(info1.q_smem, k_smem.permute((1, 0)), [k_bar],
                                                            use_acc=False)
        v_smem, v_bar, v_consumer = v_consumer.acquire()
        p0_tmem, p0_bar, p0_consumer = p0_consumer.acquire()
        o0_producer = o0_producer.wait_and_issue_next(p0_tmem, v_smem, [v_bar, p0_bar], use_acc=True)

        p1_tmem, p1_bar, p1_consumer = p1_consumer.acquire()
        o1_producer = o1_producer.wait_and_issue_next(p1_tmem, v_smem, [v_bar, p1_bar], use_acc=True)

    mbarrier.invalidate(qk_p_bar)


@gluon.jit
def _attn_fwd_correction_compute(config, mi_consumer, o_consumer, m_i):
    m_ij, mi_consumer = mi_consumer.get(gl.constexpr(gl.SliceLayout(1, config.o_splitn_layout)))
    alpha = gl.exp2(m_i - m_ij)

    o_tmem, o_bar, o_consumer = o_consumer.acquire()
    if config.SPLIT_N_FACTOR == 1:
        o = o_tmem.load(config.o_layout)
        o = o * alpha[:, None]
        o_tmem.store(o)
    else:
        for i in tl.static_range(config.SPLIT_N_FACTOR):
            o_ref = o_tmem.slice(i * config.SPLIT_N, config.SPLIT_N)
            o = o_ref.load(config.o_splitn_layout)
            o = o * alpha[:, None]
            o_ref.store(o)
    mbarrier.arrive(o_bar, count=1)
    return mi_consumer, o_consumer, m_ij


@gluon.jit
def _attn_fwd_correction(config,  #
                         m_is, infos, k_load_ctx, v_load_ctx,  #
                         STAGE: gl.constexpr):
    prog = config.get_program()
    lo, hi = prog.get_loop_bounds(STAGE)

    o0_consumer = infos[0].o_mma_ctx.channel.create_consumer()
    o1_consumer = infos[1].o_mma_ctx.channel.create_consumer()

    mi0_consumer = infos[0].mi_chnl.create_consumer()
    mi1_consumer = infos[1].mi_chnl.create_consumer()

    m_i0 = m_is[0]
    m_i1 = m_is[1]
    for start_n in range(lo, hi, config.BLOCK_N):
        mi0_consumer, o0_consumer, m_i0 = _attn_fwd_correction_compute(config, mi0_consumer, o0_consumer, m_i0)
        mi1_consumer, o1_consumer, m_i1 = _attn_fwd_correction_compute(config, mi1_consumer, o1_consumer, m_i1)

    o0_consumer.acquire()
    o1_consumer.acquire()
    return (m_i0, m_i1)


@gluon.jit
def _softmax_tile(tile_id: gl.constexpr, config, info, STAGE: gl.constexpr):
    prog = config.get_program()

    qk_slice_dim0: gl.constexpr = gl.SliceLayout(0, config.qk_layout)
    qk_slice_dim1: gl.constexpr = gl.SliceLayout(1, config.qk_layout)

    offs_m = prog.start_m * config.BLOCK_M
    offs_m += gl.arange(tile_id * config.SPLIT_M, (1 + tile_id) * config.SPLIT_M, qk_slice_dim1)
    offs_n = gl.arange(0, config.BLOCK_N, qk_slice_dim0)

    lo, hi = prog.get_loop_bounds(STAGE)

    qk_consumer = info.qk_mma_ctx.channel.create_consumer()
    p_producer = info.p_chnl.create_producer()
    mi_producer = info.mi_chnl.create_producer()

    m_i = info.mi_chnl.mem.index(0).load(qk_slice_dim1)
    l_i = info.li_smem.load(qk_slice_dim1)

    for start_n in range(lo, hi, config.BLOCK_N):
        if STAGE == 2:
            # Prevent LLVM from hoisting the partial sums, which triggers spilling.
            offs_n = gl.inline_asm_elementwise("mov.b32 $0, $0;", "=r,r", [offs_n], dtype=gl.int32, is_pure=True,
                                               pack=1)
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk, qk_consumer = qk_consumer.get(config.qk_layout)
            qk = qk * config.qk_scale + gl.where(mask, 0, -1.0e6)
            m_ij = gl.maximum(m_i, gl.max(qk, 1))
            mi_producer = mi_producer.emplace(m_ij)
            qk -= m_ij[:, None]
        else:
            qk, qk_consumer = qk_consumer.get(config.qk_layout)
            m_ij = gl.maximum(m_i, gl.max(qk, 1) * config.qk_scale)
            mi_producer = mi_producer.emplace(m_ij)
            qk = qk * config.qk_scale - m_ij[:, None]

        p = gl.exp2(qk)

        alpha = gl.exp2(m_i - m_ij)
        l_ij = gl.sum(p, 1)

        p_producer = p_producer.emplace(p.to(config.dtype))

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    info.li_smem.store(l_i)


@gluon.jit
def _attn_fwd_softmax0(config,  #
                       m_is, infos, k_load_ctx, v_load_ctx,  #
                       STAGE: gl.constexpr):
    _softmax_tile(0, config, infos[0], STAGE)


@gluon.jit
def _attn_fwd_softmax1(config,  #
                       m_is, infos, k_load_ctx, v_load_ctx,  #
                       STAGE: gl.constexpr):
    _softmax_tile(1, config, infos[1], STAGE)


@gluon.jit
def _attn_fwd_inner_d128(config, info0, info1, m_i0, m_i1,  #
                         desc_k, desc_v,  #
                         STAGE: gl.constexpr):
    k_load_ctx = LoadContext.create(desc_k, num_buffers=2, num_consumers=2)
    v_load_ctx = LoadContext.create(desc_v, num_buffers=2, num_consumers=2)

    m_i0, m_i1 = gl.warp_specialize((
        config,
        (m_i0, m_i1),
        (info0, info1),
        k_load_ctx,
        v_load_ctx,
        STAGE,
    ), _attn_fwd_correction, [
        _attn_fwd_softmax0,
        _attn_fwd_softmax1,
        _attn_fwd_mma_d128,
        _attn_fwd_load_d128,
    ], [4, 4, 1, 1], [192, 200, 32, 32])

    k_load_ctx.release()
    v_load_ctx.release()

    return m_i0, m_i1


@gluon.jit(do_not_specialize=["Z"])
def _gluon_attn_d128(sm_scale, M, Z, H, N_CTX,  #
                     desc_q, desc_k, desc_v, desc_o,  #
                     HEAD_DIM: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,  #
                     STAGE: gl.constexpr, dtype: gl.constexpr,  #
                     num_warps: gl.constexpr):
    gl.static_assert(BLOCK_N <= HEAD_DIM, "BLOCK_N must be less than or equal to HEAD_DIM")

    qk_scale = sm_scale
    qk_scale *= 1.44269504
    config = AttentionConfig(qk_scale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, dtype, num_warps, SPLIT_M_FACTOR=2,
                             SPLIT_N_FACTOR=triton.cdiv(BLOCK_N, 64))

    prog = config.get_program()

    tile0 = AttentionTile.create(config)
    tile1 = AttentionTile.create(config)
    load_tensor_desc_to_smem(desc_q, (prog.qo_offset_y + config.SPLIT_M * 0, 0), tile0.q_smem)
    load_tensor_desc_to_smem(desc_q, (prog.qo_offset_y + config.SPLIT_M * 1, 0), tile1.q_smem)

    info0 = InnerLoopInfo.create(config, tile0)
    info1 = InnerLoopInfo.create(config, tile1)

    if STAGE & 1:
        tile0.m_i, tile1.m_i = _attn_fwd_inner_d128(config, info0, info1, tile0.m_i, tile1.m_i, desc_k, desc_v,
                                                    4 - STAGE)
    if STAGE & 2:
        tile0 = info0.consume_result(tile0)
        info0 = InnerLoopInfo.create(config, tile0)
        tile1 = info1.consume_result(tile1)
        info1 = InnerLoopInfo.create(config, tile1)
        tile0.m_i, tile1.m_i = _attn_fwd_inner_d128(config, info0, info1, tile0.m_i, tile1.m_i, desc_k, desc_v, 2)

    tile0.q_smem._keep_alive()
    tile1.q_smem._keep_alive()

    o_smem = gl.allocate_shared_memory(desc_o.dtype, config.o_shape, gl.constexpr(desc_o.layout))
    info0.consume_result(tile0).do_epilogue(0, M, o_smem, desc_o, prog, config)
    info1.consume_result(tile1).do_epilogue(1, M, o_smem, desc_o, prog, config)
    o_smem._keep_alive()


# ===-----------------------------------------------------------------------===#
# _gluon_attn_d64
# ===-----------------------------------------------------------------------===#


@aggregate
class InnerLoopInfoD64:
    qk_mma_ctx: MMAContext
    o_mma_ctx: MMAContext
    q_smem: gl.shared_memory_descriptor
    p_tmem: tensor_memory_descriptor

    @gluon.jit
    def create(config, tile):
        qk_mma_ctx = MMAContext.create(config.qk_shape, num_buffers=2)
        qk_mma_ctx.channel.initialize_for_producer()

        o_mma_ctx = MMAContext.create(config.o_shape, num_buffers=1)
        o_mma_ctx.channel.initialize_for_consumer()
        o_mma_ctx.channel.mem.index(0).store(tile.acc)

        p_tmem = allocate_tensor_memory(config.dtype, config.q_shape, config.p_tmem_layout)
        return InnerLoopInfoD64(qk_mma_ctx, o_mma_ctx, tile.q_smem, p_tmem)

    def __init__(self, qk_mma_ctx, o_mma_ctx, q_smem, p_tmem):
        self.qk_mma_ctx = qk_mma_ctx
        self.o_mma_ctx = o_mma_ctx
        self.q_smem = q_smem
        self.p_tmem = p_tmem

    @gluon.jit
    def consume_result(self, tile):
        tile.acc = self.o_mma_ctx.channel.mem.index(0).load(tile.acc.type.layout)
        self.qk_mma_ctx.release()
        self.o_mma_ctx.release()
        return tile


@gluon.jit
def _attn_fwd_load_d64(config, info, m_i, l_i, k_load_ctx, v_load_ctx, STAGE: gl.constexpr):
    _attn_fwd_load(config, k_load_ctx, v_load_ctx, STAGE)


@gluon.jit
def _attn_fwd_mma_d64(config, info, m_i, l_i, k_load_ctx, v_load_ctx, STAGE: gl.constexpr):
    prog = config.get_program()
    lo, hi = prog.get_loop_bounds(STAGE)

    k_consumer = k_load_ctx.channel.create_consumer()
    v_consumer = v_load_ctx.channel.create_consumer()
    qk_producer = MMAProducer.create(info.qk_mma_ctx)
    o_producer = MMAProducer.create(info.o_mma_ctx)

    num_mmas = (hi - lo) // config.BLOCK_N
    if num_mmas > 0:
        k_smem, k_bar, k_consumer = k_consumer.acquire()
        qk_producer = qk_producer.wait_and_issue_next(info.q_smem, k_smem.permute((1, 0)), [k_bar], use_acc=False)
        for _ in range(num_mmas - 1):
            k_smem, k_bar, k_consumer = k_consumer.acquire()
            qk_producer = qk_producer.wait_and_issue_next(info.q_smem, k_smem.permute((1, 0)), [k_bar], use_acc=False)
            v_smem, v_bar, v_consumer = v_consumer.acquire()
            o_producer = o_producer.wait_and_issue_next(info.p_tmem, v_smem, [v_bar], use_acc=True)
        v_smem, v_bar, v_consumer = v_consumer.acquire()
        o_producer = o_producer.wait_and_issue_next(info.p_tmem, v_smem, [v_bar], use_acc=True)


@gluon.jit
def _attn_fwd_softmax(config, info, m_i, l_i, k_load_ctx, v_load_ctx, STAGE: gl.constexpr):
    prog = config.get_program()
    lo, hi = prog.get_loop_bounds(STAGE)

    offs_m = prog.start_m * config.BLOCK_M + gl.arange(0, config.SPLIT_M, gl.SliceLayout(1, config.qk_layout))
    offs_n = gl.arange(0, config.BLOCK_N, gl.SliceLayout(0, config.qk_layout))

    qk_consumer = info.qk_mma_ctx.channel.create_consumer()
    o_consumer = info.o_mma_ctx.channel.create_consumer()

    for start_n in range(lo, hi, config.BLOCK_N):
        qk, qk_consumer = qk_consumer.get(config.qk_layout)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * config.qk_scale + gl.where(mask, 0, -1.0e6)
            m_ij = gl.maximum(m_i, gl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = gl.maximum(m_i, gl.max(qk, 1) * config.qk_scale)
            qk = qk * config.qk_scale - m_ij[:, None]

        p = gl.exp2(qk)

        alpha = gl.exp2(m_i - m_ij)
        l_ij = gl.sum(p, 1)

        o_tmem, o_bar, o_consumer = o_consumer.acquire()
        acc = o_tmem.load(config.o_layout)
        acc = acc * alpha[:, None]
        o_tmem.store(acc)

        p = p.to(config.dtype)
        info.p_tmem.store(p)
        mbarrier.arrive(o_bar, count=1)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    o_consumer.acquire()

    return m_i, l_i


@gluon.jit
def _attn_fwd_inner_d64(config, info, m_i, l_i, desc_k, desc_v,  #
                        STAGE: gl.constexpr):
    k_load_ctx = LoadContext.create(desc_k, num_buffers=2, num_consumers=1)
    v_load_ctx = LoadContext.create(desc_v, num_buffers=2, num_consumers=1)

    m_i, l_i = gl.warp_specialize((config, info, m_i, l_i, k_load_ctx, v_load_ctx, STAGE), _attn_fwd_softmax, [
        _attn_fwd_mma_d64,
        _attn_fwd_load_d64,
    ], [1, 1], [24, 24])

    k_load_ctx.release()
    v_load_ctx.release()

    return m_i, l_i


@gluon.jit(do_not_specialize=["Z"])
def _gluon_attn_d64(sm_scale, M, Z, H, N_CTX,  #
                    desc_q, desc_k, desc_v, desc_o,  #
                    HEAD_DIM: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,  #
                    STAGE: gl.constexpr, dtype: gl.constexpr,  #
                    num_warps: gl.constexpr):
    gl.static_assert(BLOCK_N <= HEAD_DIM, "BLOCK_N must be less than or equal to HEAD_DIM")

    qk_scale = sm_scale
    qk_scale *= 1.44269504
    config = AttentionConfig(qk_scale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, dtype, num_warps, SPLIT_M_FACTOR=1,
                             SPLIT_N_FACTOR=1)

    prog = config.get_program()

    tile = AttentionTile.create(config)
    load_tensor_desc_to_smem(desc_q, (prog.qo_offset_y, 0), tile.q_smem)

    info = InnerLoopInfoD64.create(config, tile)

    if STAGE & 1:
        tile.m_i, tile.l_i = _attn_fwd_inner_d64(config, info, tile.m_i, tile.l_i, desc_k, desc_v, 4 - STAGE)
    if STAGE & 2:
        tile = info.consume_result(tile)
        info = InnerLoopInfoD64.create(config, tile)
        tile.m_i, tile.l_i = _attn_fwd_inner_d64(config, info, tile.m_i, tile.l_i, desc_k, desc_v, 2)

    tile.q_smem._keep_alive()

    o_smem = gl.allocate_shared_memory(desc_o.dtype, config.o_shape, gl.constexpr(desc_o.layout))
    info.consume_result(tile).do_epilogue(0, M, o_smem, desc_o, prog, config)
    o_smem._keep_alive()


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


def attention_forward(q, k, v, causal, sm_scale, impl=None):
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    stage = 3 if causal else 1

    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    if impl is None:
        impl = "d128" if HEAD_DIM_K == 128 else "d64"
    split_m = impl == "d128"

    # The D128 kernel will split BLOCK_M into two subtiles.
    BLOCK_M = 256 if split_m else 128
    SPLIT_M = BLOCK_M // (2 if split_m else 1)

    BLOCK_N = min(128, HEAD_DIM_K)

    desc_q = make_tensor_desc(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[SPLIT_M, HEAD_DIM_K])
    desc_v = make_tensor_desc(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_k = make_tensor_desc(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_o = make_tensor_desc(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[SPLIT_M, HEAD_DIM_K])

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    attn = _gluon_attn_d128 if impl == "d128" else _gluon_attn_d64
    maxnreg = None if impl == "d128" else 128
    attn[grid](
        sm_scale, M, q.shape[0], q.shape[1], q.shape[2],  #
        desc_q, desc_k, desc_v, desc_o,  #
        HEAD_DIM_K, BLOCK_M, BLOCK_N,  #
        stage, torch_dtype_to_triton(q.dtype),  #
        num_warps=4, maxnreg=maxnreg)

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
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("impl", ["d128", "d64"])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(not is_blackwell(), reason="Gluon attention is only supported on Blackwell GPUs")
def test_op(Z, H, N_CTX, HEAD_DIM, causal, impl, dtype):
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

    tri_out, _ = attention_forward(q, k, v, causal, sm_scale, impl=impl)
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)


# ===-----------------------------------------------------------------------===#
# Benchmarking
# ===-----------------------------------------------------------------------===#

BATCH = [4]
N_HEADS = [32]
HEAD_DIM = [64, 128]
causal = [False, True]
providers = ["triton-fp16", "cudnn-fp16"]
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
