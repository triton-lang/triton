import functools

import torch
import triton
import triton.language as tl

from triton import knobs
from triton.language.core import builtin
from triton.language.core import _aggregate as aggregate

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import _unwrap_if_constexpr
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper.tma import tensor_descriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tma,
    mbarrier,
    tcgen05_mma,
)

# ===-----------------------------------------------------------------------===#
# Constexpr Utilities
# ===-----------------------------------------------------------------------===#


def unwrap_constexprs(fn):

    def recursively_unwrap(c):
        c = _unwrap_if_constexpr(c)
        if hasattr(c, "__iter__"):
            c = [recursively_unwrap(x) for x in c]
        return c

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = [recursively_unwrap(arg) for arg in args]
        kwargs = {k: recursively_unwrap(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


# ===-----------------------------------------------------------------------===#
# Layout Utilities
# ===-----------------------------------------------------------------------===#


@tl.constexpr_function
def get_tmem_32x32b_layout(instr_shape, shape, num_warps):
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
                mbarrier.init(ready_bars.subslice(i), count=1)
                mbarrier.init(empty_bars.subslice(i), count=num_consumers)
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
                mbarrier.arrive(self.ready_bars.subslice(i), count=1)

        @gluon.jit
        def initialize_for_producer(self):
            for i in tl.static_range(self.num_buffers):
                mbarrier.arrive(self.empty_bars.subslice(i), count=self.num_consumers)

        @gluon.jit
        def acquire_producer(self, index, phase):
            mem = self.mem.subslice(index)
            ready_bar = self.ready_bars.subslice(index)
            empty_bar = self.empty_bars.subslice(index)

            mbarrier.wait(empty_bar, phase)
            return mem, ready_bar

        @gluon.jit
        def acquire_consumer(self, index, phase):
            mem = self.mem.subslice(index)
            ready_bar = self.ready_bars.subslice(index)
            empty_bar = self.empty_bars.subslice(index)

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
                mbarrier.invalidate(self.ready_bars.subslice(i))
                mbarrier.invalidate(self.empty_bars.subslice(i))

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
        smem_layout: gl.constexpr = get_nvmma_layout(shape, desc.dtype)
        channel = SharedMemoryChannel.create(shape, desc.dtype, smem_layout, num_buffers, num_consumers)
        channel.initialize_for_producer()
        return LoadContext(desc, channel)

    def __init__(self, desc, channel):
        self.desc = desc
        self.channel = channel

    @gluon.jit
    def release(self):
        self.channel.release()


@aggregate
class LoadProducer:
    desc: tensor_descriptor
    impl: SharedMemoryProducer

    @gluon.jit
    def create(ctx):
        return LoadProducer(ctx.desc, ctx.channel.create_producer())

    def __init__(self, desc, impl):
        self.desc = desc
        self.impl = impl

    @gluon.jit
    def wait_and_issue_next(self, coord):
        smem, ready_bar, self.impl = self.impl.acquire()

        size: gl.constexpr = get_load_size_bytes(self.desc)
        mbarrier.expect(ready_bar, size)
        tma.async_copy_global_to_shared(self.desc, coord, ready_bar, smem)
        return self


@tl.constexpr_function
def get_mma_reg_layout(shape, num_warps, dtype=gl.float32):
    instr_shape = get_mma_instr_shape(shape, dtype)
    return get_tmem_32x32b_layout(instr_shape, shape, num_warps)


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
        return get_tmem_32x32b_layout(self.instr_shape, self.shape, num_warps)


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


@aggregate
class PipelinedLoadProducer:
    impl: LoadProducer
    offset: gl.tensor
    step: gl.constexpr

    @gluon.jit
    def create(impl, offset, step: gl.constexpr):
        return PipelinedLoadProducer(impl, offset, step)

    def __init__(self, impl, offset, step):
        self.impl = impl
        self.offset = offset
        self.step = gl.constexpr(step)

    @gluon.jit
    def wait_and_issue_next(self):
        self.impl = self.impl.wait_and_issue_next([self.offset, 0])
        self.offset += self.step
        return self


@aggregate
class TensorMemoryVariable:
    channel: TensorMemoryChannel
    reg_layout: gl.constexpr

    @gluon.jit
    def _allocate(num_buffers: gl.constexpr, shape: gl.constexpr, dtype: gl.constexpr, num_warps: gl.constexpr,
                  unpacked: gl.constexpr):
        instr_shape: gl.constexpr = get_mma_instr_shape(shape, dtype).value
        tmem_layout: gl.constexpr = TensorMemoryLayout([instr_shape[0], instr_shape[1]], unpacked=unpacked)
        reg_layout: gl.constexpr = get_tmem_32x32b_layout(instr_shape, shape, num_warps).value

        channel = TensorMemoryChannel.create(shape, dtype, tmem_layout, num_buffers)
        return TensorMemoryVariable(channel, reg_layout)

    @gluon.jit
    def allocate_mma(shape: gl.constexpr, dtype: gl.constexpr, num_warps: gl.constexpr, num_buffers: gl.constexpr = 1):
        return TensorMemoryVariable._allocate(num_buffers, shape, dtype, num_warps, unpacked=True)

    @gluon.jit
    def allocate_lhs(shape: gl.constexpr, dtype: gl.constexpr, num_warps: gl.constexpr, num_buffers: gl.constexpr = 1):
        return TensorMemoryVariable._allocate(num_buffers, shape, dtype, num_warps, unpacked=False)

    def __init__(self, channel, reg_layout):
        self.channel = channel
        self.reg_layout = gl.constexpr(reg_layout)

    @gluon.jit
    def release(self):
        self.channel.release()


# ===-----------------------------------------------------------------------===#
# Missing APIs
# ===-----------------------------------------------------------------------===#


@builtin
def fence_async_shared(_semantic=None):
    _semantic.builder.create_fence_async_shared(False)


# ===-----------------------------------------------------------------------===#
# _attn_fwd
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


@gluon.jit
def _attn_fwd_load(  #
        m_i0, m_i1,  #
        info0, info1, k_load_ctx, v_load_ctx,  #
        qk_scale, Z, H, N_CTX,  #
        dtype: gl.constexpr, num_warps: gl.constexpr,  #
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
        STAGE: gl.constexpr,  #
):
    start_m, off_hz, offset_y, qo_offset_y, = _get_attn_program(Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM)

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo

    k_producer = LoadProducer.create(k_load_ctx)
    v_producer = LoadProducer.create(v_load_ctx)
    load_k = PipelinedLoadProducer.create(k_producer, offsetkv_y, BLOCK_N)
    load_v = PipelinedLoadProducer.create(v_producer, offsetkv_y, BLOCK_N)

    num_loads = (hi - lo) // BLOCK_N
    if num_loads > 0:
        load_k = load_k.wait_and_issue_next()
        for _ in range(num_loads - 1):
            load_k = load_k.wait_and_issue_next()
            load_v = load_v.wait_and_issue_next()
        load_v = load_v.wait_and_issue_next()


@gluon.jit
def _attn_fwd_mma(  #
        m_i0, m_i1,  #
        info0, info1, k_load_ctx, v_load_ctx,  #
        qk_scale, Z, H, N_CTX,  #
        dtype: gl.constexpr, num_warps: gl.constexpr,  #
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
        STAGE: gl.constexpr,  #
):
    start_m, off_hz, offset_y, qo_offset_y = _get_attn_program(Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM)

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX

    k_consumer = k_load_ctx.channel.create_consumer()
    v_consumer = v_load_ctx.channel.create_consumer()
    qk0_producer = MMAProducer.create(info0.qk_mma_ctx)
    qk1_producer = MMAProducer.create(info1.qk_mma_ctx)
    o0_producer = MMAProducer.create(info0.o_mma_ctx)
    o1_producer = MMAProducer.create(info1.o_mma_ctx)
    p0_consumer = info0.p_var.create_consumer()
    p1_consumer = info1.p_var.create_consumer()

    qk_p_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    qk_p_phase = 0
    mbarrier.init(qk_p_bar, count=1)

    num_mmas = (hi - lo) // BLOCK_N
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
def _attn_fwd_correction_compute(  #
        mi_consumer, o_consumer, m_i,  #
        blocked: gl.constexpr, layout: gl.constexpr,  #
        SPLIT_FACTOR: gl.constexpr, SPLIT_N: gl.constexpr,  #
):
    m_ij, mi_consumer = mi_consumer.get(layout)
    alpha = triton.language.math.exp2(m_i - m_ij)

    o_tmem, o_bar, o_consumer = o_consumer.acquire()
    if SPLIT_FACTOR == 1:
        o = o_tmem.load(blocked)
        o = o * alpha[:, None]
        o_tmem.store(o)
    else:
        for i in tl.static_range(SPLIT_FACTOR):
            o_ref = o_tmem.split(i * SPLIT_N, SPLIT_N)
            o = o_ref.load(blocked)
            o = o * alpha[:, None]
            o_ref.store(o)
    mbarrier.arrive(o_bar, count=1)
    return mi_consumer, o_consumer, m_ij


@gluon.jit
def _attn_fwd_correction(  #
        m_i0, m_i1,  #
        info0, info1, k_load_ctx, v_load_ctx,  #
        qk_scale, Z, H, N_CTX,  #
        dtype: gl.constexpr, num_warps: gl.constexpr,  #
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
        STAGE: gl.constexpr,  #
):
    start_m, off_hz, offset_y, qo_offset_y = _get_attn_program(Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM)
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX

    if BLOCK_M // 2 == 128:
        SPLIT_FACTOR: gl.constexpr = triton.cdiv(BLOCK_N, 64)
        SPLIT_N: gl.constexpr = BLOCK_N // SPLIT_FACTOR
    else:
        SPLIT_FACTOR: gl.constexpr = 1
        SPLIT_N: gl.constexpr = BLOCK_N
    o_shape: gl.constexpr = [BLOCK_M // 2, SPLIT_N]

    blocked: gl.constexpr = get_tmem_32x32b_layout((o_shape[0], o_shape[1], None), o_shape, num_warps)
    layout: gl.constexpr = gl.SliceLayout(1, blocked)

    o0_consumer = info0.o_mma_ctx.channel.create_consumer()
    o1_consumer = info1.o_mma_ctx.channel.create_consumer()

    mi0_consumer = info0.mi_chnl.create_consumer()
    mi1_consumer = info1.mi_chnl.create_consumer()

    m_i0, mi0_consumer = mi0_consumer.get(layout)
    m_i1, mi1_consumer = mi1_consumer.get(layout)

    for start_n in range(lo, hi, BLOCK_N):
        mi0_consumer, o0_consumer, m_i0 = _attn_fwd_correction_compute(mi0_consumer, o0_consumer, m_i0, blocked, layout,
                                                                       SPLIT_FACTOR, SPLIT_N)
        mi1_consumer, o1_consumer, m_i1 = _attn_fwd_correction_compute(mi1_consumer, o1_consumer, m_i1, blocked, layout,
                                                                       SPLIT_FACTOR, SPLIT_N)

    o0_consumer.acquire()
    o1_consumer.acquire()


@gluon.jit
def _softmax_tile(  #
        Z, H, N_CTX, qk_scale,  #
        tile_id: gl.constexpr, m_i, l_i,  #
        qk_mma_ctx, o_mma_ctx, p_var, mi_chnl,  #
        STAGE: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, dtype: gl.constexpr,
        num_warps: gl.constexpr,  #
):
    start_m, off_hz, offset_y, qo_offset_y = _get_attn_program(Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM)

    blocked: gl.constexpr = qk_mma_ctx.get_reg_layout(num_warps)
    SPLIT_M: gl.constexpr = BLOCK_M // 2
    offs_m = start_m * BLOCK_M + gl.arange(tile_id * SPLIT_M,
                                           (1 + tile_id) * SPLIT_M, layout=gl.SliceLayout(1, blocked))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked))

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo

    qk_consumer = qk_mma_ctx.channel.create_consumer()
    p_producer = p_var.create_producer()
    mi_producer = mi_chnl.create_producer()

    mi_producer = mi_producer.emplace(m_i)

    for start_n in range(lo, hi, BLOCK_N):
        qk, qk_consumer = qk_consumer.get(qk_mma_ctx.get_reg_layout(num_warps))

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + gl.where(mask, 0, -1.0e6)
            m_ij = gl.maximum(m_i, gl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = gl.maximum(m_i, gl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        mi_producer = mi_producer.emplace(m_ij)
        p = gl.exp2(qk)

        alpha = gl.exp2(m_i - m_ij)
        l_ij = gl.sum(p, 1)

        p_producer = p_producer.emplace(p.to(dtype))

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        offsetkv_y += BLOCK_N

    return m_i, l_i


@gluon.jit
def _attn_fwd_softmax0(  #
        m_i0, m_i1,  #
        info0, info1, k_load_ctx, v_load_ctx,  #
        qk_scale, Z, H, N_CTX,  #
        dtype: gl.constexpr, num_warps: gl.constexpr,  #
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
        STAGE: gl.constexpr,  #
):
    layout: gl.constexpr = gl.SliceLayout(1, info0.qk_mma_ctx.get_reg_layout(num_warps))
    m_i = info0.mi_chnl.mem.subslice(0).load(layout)
    l_i = info0.li_smem.load(layout)

    m_i, l_i = _softmax_tile(  #
        Z, H, N_CTX, qk_scale,  #
        0, m_i, l_i,  #
        info0.qk_mma_ctx, info0.o_mma_ctx, info0.p_var, info0.mi_chnl,  #
        STAGE, BLOCK_M, BLOCK_N, HEAD_DIM, dtype, num_warps,  #
    )

    info0.mi_chnl.mem.subslice(0).store(m_i)
    info0.li_smem.store(l_i)


@gluon.jit
def _attn_fwd_softmax1(  #
        m_i0, m_i1,  #
        info0, info1, k_load_ctx, v_load_ctx,  #
        qk_scale, Z, H, N_CTX,  #
        dtype: gl.constexpr, num_warps: gl.constexpr,  #
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
        STAGE: gl.constexpr,  #
):
    layout: gl.constexpr = gl.SliceLayout(1, info1.qk_mma_ctx.get_reg_layout(num_warps))
    m_i = info1.mi_chnl.mem.subslice(0).load(layout)
    l_i = info1.li_smem.load(layout)

    m_i, l_i = _softmax_tile(  #
        Z, H, N_CTX, qk_scale,  #
        1, m_i, l_i,  #
        info1.qk_mma_ctx, info1.o_mma_ctx, info1.p_var, info1.mi_chnl,  #
        STAGE, BLOCK_M, BLOCK_N, HEAD_DIM, dtype, num_warps,  #
    )

    info1.mi_chnl.mem.subslice(0).store(m_i)
    info1.li_smem.store(l_i)


@gluon.jit
def _borrow_qk_as_p(qk_mma_ctx, shape, dtype, num_warps):
    instr_shape: gl.constexpr = get_mma_instr_shape(shape, dtype).value
    tmem_layout: gl.constexpr = TensorMemoryLayout([instr_shape[0], instr_shape[1]], unpacked=False)

    return TensorMemoryChannel._borrow(qk_mma_ctx.channel.mem, shape, dtype, tmem_layout, num_buffers=1,
                                       num_consumers=1)


@aggregate
class AttentionInnerLoopTileInfo:
    qk_mma_ctx: MMAContext
    o_mma_ctx: MMAContext
    p_var: TensorMemoryChannel
    mi_chnl: SharedMemoryChannel
    li_smem: gl.shared_memory_descriptor
    q_smem: gl.shared_memory_descriptor

    @gluon.jit
    def create(tile, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, dtype: gl.constexpr,
               num_warps: gl.constexpr):
        qk_mma_ctx = MMAContext.create([BLOCK_M // 2, BLOCK_N], num_buffers=1)
        qk_mma_ctx.channel.initialize_for_producer()

        o_mma_ctx = MMAContext.create([BLOCK_M // 2, HEAD_DIM], num_buffers=1)
        o_mma_ctx.channel.initialize_for_consumer()
        o_mma_ctx.channel.mem.subslice(0).store(tile.acc)

        p_var = _borrow_qk_as_p(qk_mma_ctx, [BLOCK_M // 2, BLOCK_N], dtype, num_warps)
        p_var.initialize_for_producer()

        mi_chnl = SharedMemoryChannel.create([BLOCK_M // 2], gl.float32, gl.constexpr(mbarrier.MBarrierLayout()),
                                             num_buffers=1)
        mi_chnl.initialize_for_producer()
        mi_chnl.mem.subslice(0).store(tile.m_i)

        li_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M // 2], gl.constexpr(mbarrier.MBarrierLayout()))
        li_smem.store(tile.l_i)

        return AttentionInnerLoopTileInfo(qk_mma_ctx, o_mma_ctx, p_var, mi_chnl, li_smem, tile.q_smem)

    def __init__(self, qk_mma_ctx, o_mma_ctx, p_var, mi_chnl, li_smem, q_smem):
        self.qk_mma_ctx = qk_mma_ctx
        self.o_mma_ctx = o_mma_ctx
        self.p_var = p_var
        self.mi_chnl = mi_chnl
        self.li_smem = li_smem
        self.q_smem = q_smem

    @gluon.jit
    def consume_result(self, tile):
        tile.acc = self.o_mma_ctx.channel.mem.subslice(0).load(tile.acc.type.layout)
        tile.m_i = self.mi_chnl.mem.subslice(0).load(tile.m_i.type.layout)
        tile.l_i = self.li_smem.load(tile.l_i.type.layout)

        self.qk_mma_ctx.release()
        self.o_mma_ctx.release()
        self.p_var.release()
        self.mi_chnl.release()
        self.li_smem._keep_alive()
        return tile


@gluon.jit
def _attn_fwd_inner(tile0, tile1,  #
                    desc_k, desc_v,  #
                    qk_scale, Z, H, N_CTX,  #
                    dtype: gl.constexpr, num_warps: gl.constexpr,  #
                    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
                    STAGE: gl.constexpr):
    k_load_ctx = LoadContext.create(desc_k, num_buffers=2, num_consumers=2)
    v_load_ctx = LoadContext.create(desc_v, num_buffers=2, num_consumers=2)

    info0 = AttentionInnerLoopTileInfo.create(tile0, BLOCK_M, BLOCK_N, HEAD_DIM, dtype, num_warps)
    info1 = AttentionInnerLoopTileInfo.create(tile1, BLOCK_M, BLOCK_N, HEAD_DIM, dtype, num_warps)

    gl.warp_specialize((
        tile0.m_i,
        tile1.m_i,
        info0,
        info1,
        k_load_ctx,
        v_load_ctx,
        qk_scale,
        Z,
        H,
        N_CTX,
        dtype,
        num_warps,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        STAGE,
    ), _attn_fwd_correction, [
        _attn_fwd_softmax0,
        _attn_fwd_softmax1,
        _attn_fwd_mma,
        _attn_fwd_load,
    ], [4, 4, 1, 1], [192, 192, 24, 24])

    k_load_ctx.release()
    v_load_ctx.release()

    tile0 = info0.consume_result(tile0)
    tile1 = info1.consume_result(tile1)
    return tile0, tile1


@gluon.jit
def _get_attn_program(  #
        Z, H, N_CTX,  #
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
):
    start_m = gl.program_id(0)
    off_hz = gl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    return start_m, off_hz, offset_y, qo_offset_y


@aggregate
class AttentionTile:
    acc: gl.tensor
    m_i: gl.tensor
    l_i: gl.tensor
    q_smem: gl.shared_memory_descriptor

    @gluon.jit
    def create(SPLIT_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, dtype: gl.constexpr,
               num_warps: gl.constexpr):
        qk_layout: gl.constexpr = get_mma_reg_layout([SPLIT_M, BLOCK_N], num_warps)
        o_layout: gl.constexpr = get_mma_reg_layout([SPLIT_M, HEAD_DIM], num_warps)
        slice_dim1: gl.constexpr = gl.SliceLayout(1, qk_layout)
        acc = gl.full([SPLIT_M, HEAD_DIM], 0, gl.float32, o_layout)
        m_i = gl.full([SPLIT_M], -float("inf"), dtype=gl.float32, layout=slice_dim1)
        l_i = gl.full([SPLIT_M], 1.0, dtype=gl.float32, layout=slice_dim1)

        smem_layout: gl.constexpr = get_nvmma_layout((SPLIT_M, HEAD_DIM), dtype)
        q_smem = gl.allocate_shared_memory(dtype, (SPLIT_M, HEAD_DIM), smem_layout)
        return AttentionTile(acc, m_i, l_i, q_smem)

    def __init__(self, acc, m_i, l_i, q_smem):
        self.acc = acc
        self.m_i = m_i
        self.l_i = l_i
        self.q_smem = q_smem

    @gluon.jit
    def do_epilogue(self, index, M, start_m, off_hz, o_smem, desc_o, qo_offset_y, N_CTX, dtype: gl.constexpr,
                    BLOCK_M: gl.constexpr, SPLIT_M: gl.constexpr, layout: gl.constexpr):
        self.m_i += gl.log2(self.l_i)
        o = self.acc / gl.convert_layout(self.l_i, gl.SliceLayout(1, self.acc.type.layout))[:, None]
        offs_m = start_m * BLOCK_M + gl.arange(index * SPLIT_M, (index + 1) * SPLIT_M, layout=layout)
        m_ptrs = M + off_hz * N_CTX + offs_m
        gl.store(m_ptrs, gl.convert_layout(self.m_i, layout))
        o_smem.store(o.to(dtype))
        fence_async_shared()
        store_smem_to_tensor_desc(desc_o, [qo_offset_y + SPLIT_M * index, 0], o_smem)


@gluon.jit
def _attn_fwd(sm_scale, M, Z, H, N_CTX,  #
              desc_q, desc_k, desc_v, desc_o,  #
              HEAD_DIM: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,  #
              STAGE: gl.constexpr, dtype: gl.constexpr,  #
              num_warps: gl.constexpr, threads_per_warp: gl.constexpr):
    gl.static_assert(BLOCK_N <= HEAD_DIM, "BLOCK_N must be less than or equal to HEAD_DIM")

    SPLIT_M: gl.constexpr = BLOCK_M // 2

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    start_m, off_hz, offset_y, qo_offset_y = _get_attn_program(Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM)

    tile0 = AttentionTile.create(SPLIT_M, BLOCK_N, HEAD_DIM, dtype, num_warps)
    tile1 = AttentionTile.create(SPLIT_M, BLOCK_N, HEAD_DIM, dtype, num_warps)
    load_tensor_desc_to_smem(desc_q, (qo_offset_y + SPLIT_M * 0, 0), tile0.q_smem)
    load_tensor_desc_to_smem(desc_q, (qo_offset_y + SPLIT_M * 1, 0), tile1.q_smem)

    if STAGE & 1:
        tile0, tile1 = _attn_fwd_inner(tile0, tile1, desc_k, desc_v,  #
                                       qk_scale, Z, H, N_CTX,  #
                                       dtype, num_warps, BLOCK_M, BLOCK_N, HEAD_DIM, 4 - STAGE)
    if STAGE & 2:
        tile0, tile1 = _attn_fwd_inner(tile0, tile1, desc_k, desc_v,  #
                                       qk_scale, Z, H, N_CTX,  #
                                       dtype, num_warps, BLOCK_M, BLOCK_N, HEAD_DIM, 2)

    tile0.q_smem._keep_alive()
    tile1.q_smem._keep_alive()

    layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    o_smem = gl.allocate_shared_memory(desc_o.dtype, (BLOCK_M // 2, HEAD_DIM),
                                       get_nvmma_layout((BLOCK_M // 2, HEAD_DIM), desc_o.dtype))
    tile0.do_epilogue(0, M, start_m, off_hz, o_smem, desc_o, qo_offset_y, N_CTX, dtype, BLOCK_M, SPLIT_M, layout)
    tile1.do_epilogue(1, M, start_m, off_hz, o_smem, desc_o, qo_offset_y, N_CTX, dtype, BLOCK_M, SPLIT_M, layout)
    o_smem._keep_alive()


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

    stage = 3 if causal else 1
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    BLOCK_M = 128
    BLOCK_N = min(HEAD_DIM_K, 128)

    desc_q = make_tensor_desc(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM_K])
    desc_v = make_tensor_desc(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_k = make_tensor_desc(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_o = make_tensor_desc(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM_K])

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

    knobs.compilation.disable_line_info = True
    kernel = _attn_fwd.warmup(  #
        sm_scale, M, q.shape[0], q.shape[1], q.shape[2],  #
        desc_q, desc_k, desc_v, desc_o,  #
        HEAD_DIM_K, BLOCK_M, BLOCK_N,  #
        stage, torch_dtype_to_triton(q.dtype),  #
        num_warps=4, threads_per_warp=32,  #
        grid=grid,  #
    )
    print(kernel.asm["ttgir"])


if __name__ == "__main__":
    BATCH = 4
    H = 32
    N_CTX = 16 * 1024
    HEAD_DIM = 128
    causal = True

    dtype = torch.float16
    device = "cuda"

    sm_scale = 1.3
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)

    # attention_forward(q, k, v, causal, sm_scale)
