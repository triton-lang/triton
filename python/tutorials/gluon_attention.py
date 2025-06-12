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

        @gluon.jit
        def create(shape: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr, num_buffers: gl.constexpr):
            mem = alloc_fn(dtype, [num_buffers] + shape, layout)
            ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            for i in tl.static_range(num_buffers):
                mbarrier.init(ready_bars.subslice(i), count=1)
                mbarrier.init(empty_bars.subslice(i), count=1)
            return ChannelType(mem, ready_bars, empty_bars, num_buffers)

        @gluon.jit
        def increment(self, index, phase):
            next_index = index + 1
            rollover = next_index == self.num_buffers
            index = gl.where(rollover, 0, next_index)
            phase = gl.where(rollover, phase ^ 1, phase)
            return index, phase

        def __init__(self, mem, ready_bars, empty_bars, num_buffers):
            self.mem = mem
            self.ready_bars = ready_bars
            self.empty_bars = empty_bars
            self.num_buffers = gl.constexpr(num_buffers)

        @gluon.jit
        def initialize_for_consumer(self):
            for i in tl.static_range(self.num_buffers):
                mbarrier.arrive(self.ready_bars.subslice(i), count=1)

        @gluon.jit
        def initialize_for_producer(self):
            for i in tl.static_range(self.num_buffers):
                mbarrier.arrive(self.empty_bars.subslice(i), count=1)

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
    def create(desc, num_buffers: gl.constexpr):
        shape: gl.constexpr = desc.block_type.shape
        smem_layout: gl.constexpr = get_nvmma_layout(shape, desc.dtype)
        channel = SharedMemoryChannel.create(shape, desc.dtype, smem_layout, num_buffers)
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
def _attn_fwd_load(m_i, l_i,  #
                   q_smem, p_var, mi_chnl,  #
                   qk_mma_ctx, o_mma_ctx,  #
                   k_load_ctx, v_load_ctx,  #
                   qk_scale, Z, H, N_CTX,  #
                   dtype: gl.constexpr, num_warps: gl.constexpr,  #
                   BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
                   STAGE: gl.constexpr):
    start_m, off_hz, offset_y, qo_offset_y, offs_m, offs_n = _get_attn_program(Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM,
                                                                               o_mma_ctx.get_reg_layout(num_warps))

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
def _attn_fwd_mma(m_i, l_i,  #
                  q_smem, p_var, mi_chnl,  #
                  qk_mma_ctx, o_mma_ctx,  #
                  k_load_ctx, v_load_ctx,  #
                  qk_scale, Z, H, N_CTX,  #
                  dtype: gl.constexpr, num_warps: gl.constexpr,  #
                  BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
                  STAGE: gl.constexpr):
    start_m, off_hz, offset_y, qo_offset_y, offs_m, offs_n = _get_attn_program(Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM,
                                                                               o_mma_ctx.get_reg_layout(num_warps))

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX

    k_consumer = k_load_ctx.channel.create_consumer()
    v_consumer = v_load_ctx.channel.create_consumer()
    qk_producer = MMAProducer.create(qk_mma_ctx)
    o_producer = MMAProducer.create(o_mma_ctx)
    p_consumer = p_var.channel.create_consumer()

    num_mmas = (hi - lo) // BLOCK_N
    if num_mmas > 0:
        k_smem, k_bar, k_consumer = k_consumer.acquire()
        qk_producer = qk_producer.wait_and_issue_next(q_smem, k_smem.permute((1, 0)), [k_bar], use_acc=False)
        for _ in range(num_mmas - 1):
            k_smem, k_bar, k_consumer = k_consumer.acquire()
            qk_producer = qk_producer.wait_and_issue_next(q_smem, k_smem.permute((1, 0)), [k_bar],
                                                          use_acc=False)
            v_smem, v_bar, v_consumer = v_consumer.acquire()
            p_tmem, p_bar, p_consumer = p_consumer.acquire()
            o_producer = o_producer.wait_and_issue_next(p_tmem, v_smem, [v_bar, p_bar], use_acc=True)
        v_smem, v_bar, v_consumer = v_consumer.acquire()
        p_tmem, p_bar, p_consumer = p_consumer.acquire()
        o_producer = o_producer.wait_and_issue_next(p_tmem, v_smem, [v_bar, p_bar], use_acc=True)


@gluon.jit
def _attn_fwd_correction(m_i, l_i,  #
                         q_smem, p_var, mi_chnl,  #
                         qk_mma_ctx, o_mma_ctx,  #
                         k_load_ctx, v_load_ctx,  #
                         qk_scale, Z, H, N_CTX,  #
                         dtype: gl.constexpr, num_warps: gl.constexpr,  #
                         BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
                         STAGE: gl.constexpr):
    start_m, off_hz, offset_y, qo_offset_y, offs_m, offs_n = _get_attn_program(Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM,
                                                                               o_mma_ctx.get_reg_layout(num_warps))

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX

    o_consumer = o_mma_ctx.channel.create_consumer()

    blocked: gl.constexpr = get_mma_reg_layout([BLOCK_M, BLOCK_N], num_warps)
    layout: gl.constexpr = gl.SliceLayout(1, blocked)
    mi_consumer = mi_chnl.create_consumer()
    m_i, mi_consumer = mi_consumer.get(layout)

    for start_n in range(lo, hi, BLOCK_N):
        m_ij, mi_consumer = mi_consumer.get(layout)
        alpha = triton.language.math.exp2(m_i - m_ij)

        o_tmem, o_bar, o_consumer = o_consumer.acquire()
        o = o_tmem.load(o_mma_ctx.get_reg_layout(num_warps))
        o = o * alpha[:, None]
        o_tmem.store(o)
        mbarrier.arrive(o_bar, count=1)

        m_i = m_ij

    o_consumer.acquire()


@gluon.jit
def _attn_fwd_default(m_i, l_i,  #
                      q_smem, p_var, mi_chnl,  #
                      qk_mma_ctx, o_mma_ctx,  #
                      k_load_ctx, v_load_ctx,  #
                      qk_scale, Z, H, N_CTX,  #
                      dtype: gl.constexpr, num_warps: gl.constexpr,  #
                      BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
                      STAGE: gl.constexpr):
    start_m, off_hz, offset_y, qo_offset_y, offs_m, offs_n = _get_attn_program(Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM,
                                                                               o_mma_ctx.get_reg_layout(num_warps))

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo

    qk_consumer = qk_mma_ctx.channel.create_consumer()

    p_producer = p_var.channel.create_producer()
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
        p = triton.language.math.exp2(qk)

        alpha = triton.language.math.exp2(m_i - m_ij)
        l_ij = gl.sum(p, 1)

        p_producer = p_producer.emplace(p.to(dtype))

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        offsetkv_y += BLOCK_N

    return m_i, l_i


@gluon.jit
def _attn_fwd_inner(m_i, l_i,  #
                    q_smem, acc,  #
                    desc_k, desc_v,  #
                    qk_scale, Z, H, N_CTX,  #
                    dtype: gl.constexpr, num_warps: gl.constexpr,  #
                    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
                    STAGE: gl.constexpr):
    k_load_ctx = LoadContext.create(desc_k, num_buffers=2)
    v_load_ctx = LoadContext.create(desc_v, num_buffers=2)

    qk_mma_ctx = MMAContext.create([BLOCK_M, BLOCK_N], num_buffers=2)
    qk_mma_ctx.channel.initialize_for_producer()

    o_mma_ctx = MMAContext.create(acc.shape, num_buffers=1)
    o_mma_ctx.channel.initialize_for_consumer()
    o_mma_ctx.channel.mem.subslice(0).store(acc)

    p_var = TensorMemoryVariable.allocate_lhs([BLOCK_M, BLOCK_N], dtype, num_warps)
    p_var.channel.initialize_for_producer()
    mi_chnl = SharedMemoryChannel.create([BLOCK_M], gl.float32, gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1)
    mi_chnl.initialize_for_producer()

    m_i, l_i = gl.warp_specialize((
        m_i,
        l_i,
        q_smem,
        p_var,
        mi_chnl,
        qk_mma_ctx,
        o_mma_ctx,
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
    ), _attn_fwd_default, [_attn_fwd_load, _attn_fwd_mma, _attn_fwd_correction], [1, 1, 4], [24, 24, 80])

    acc = o_mma_ctx.channel.mem.subslice(0).load(acc.type.layout)

    k_load_ctx.release()
    v_load_ctx.release()
    qk_mma_ctx.release()
    o_mma_ctx.release()
    mi_chnl.release()

    return m_i, l_i, acc


@gluon.jit
def _get_attn_program(  #
        Z, H, N_CTX,  #
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
        blocked: gl.constexpr,  #
):
    start_m = gl.program_id(0)
    off_hz = gl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked))

    return start_m, off_hz, offset_y, qo_offset_y, offs_m, offs_n


@gluon.jit
def _attn_fwd(sm_scale, M, Z, H, N_CTX,  #
              desc_q, desc_k, desc_v, desc_o,  #
              HEAD_DIM: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,  #
              STAGE: gl.constexpr, dtype: gl.constexpr,  #
              num_warps: gl.constexpr, threads_per_warp: gl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM, "BLOCK_N must be less than or equal to HEAD_DIM")

    blocked: gl.constexpr = get_mma_reg_layout([BLOCK_M, BLOCK_N], num_warps)
    acc = gl.full([BLOCK_M, BLOCK_N], 0, gl.float32, blocked)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    start_m, off_hz, offset_y, qo_offset_y, offs_m, offs_n = _get_attn_program(Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM,
                                                                               blocked)

    m_i = gl.full([BLOCK_M], 0.0, dtype=tl.float32, layout=gl.constexpr(gl.SliceLayout(1, blocked))) - float("inf")
    l_i = gl.full([BLOCK_M], 0.0, dtype=tl.float32, layout=gl.constexpr(gl.SliceLayout(1, blocked))) + 1.0

    q_smem_layout: gl.constexpr = get_nvmma_layout(desc_q.block_type.shape, desc_q.dtype)
    q_smem = gl.allocate_shared_memory(desc_q.dtype, desc_q.block_type.shape, q_smem_layout)
    load_tensor_desc_to_smem(desc_q, (qo_offset_y, 0), q_smem)

    if STAGE & 1:
        m_i, l_i, acc = _attn_fwd_inner(m_i, l_i, q_smem, acc, desc_k, desc_v,  #
                                        qk_scale, Z, H, N_CTX,  #
                                        dtype, num_warps, BLOCK_M, BLOCK_N, HEAD_DIM, 4 - STAGE)
    if STAGE & 2:
        m_i, l_i, acc = _attn_fwd_inner(m_i, l_i, q_smem, acc, desc_k, desc_v,  #
                                        qk_scale, Z, H, N_CTX,  #
                                        dtype, num_warps, BLOCK_M, BLOCK_N, HEAD_DIM, 2)

    q_smem._keep_alive()

    m_i += triton.language.math.log2(l_i)
    o = acc / l_i[:, None]
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=layout)
    m_ptrs = M + off_hz * N_CTX + offs_m
    gl.store(m_ptrs, gl.convert_layout(m_i, layout))
    o_smem = gl.allocate_shared_memory(desc_o.dtype, (BLOCK_M, HEAD_DIM),
                                       get_nvmma_layout((BLOCK_M, HEAD_DIM), desc_o.dtype))
    o_smem.store(o.to(dtype))
    fence_async_shared()
    store_smem_to_tensor_desc(desc_o, [qo_offset_y, 0], o_smem)


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
