import copy
import math
import torch
import triton
import pytest
import itertools
from dataclasses import dataclass, fields

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tensor_memory_descriptor_type,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
    tcgen05_commit,
    float2,
)
from triton.experimental.gluon.language.nvidia.blackwell.float2 import Float2Tensor

# ===-----------------------------------------------------------------------===#
# Layout Utilities
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def get_mma_instr_shape(shape, element_ty):
    m = 128 if shape[0] >= 128 else 64
    n = 256 if shape[1] >= 256 else shape[1]
    k = 256 // element_ty.primitive_bitwidth
    return (m, n, k)


@gluon.constexpr_function
def get_split_dim(cga_layout, dim):
    return 1 << sum(b[dim] != 0 for b in cga_layout)


def get_mma_operand_cga_layout(layout, op_idx):
    assert op_idx in (0, 1)
    if not layout:
        return layout

    # 2CTA performs an outer product so bases are [1, 0] and [0, 1].
    assert layout[0] == (1, 0)
    first = (1, 0) if op_idx == 0 else (0, 1)

    # Broadcast along K (the reduction dimension).
    def broadcast(b):
        return (b[0], 0) if op_idx == 0 else (0, 2 * b[1])

    return (first, *map(broadcast, layout[1:]))


# ===-----------------------------------------------------------------------===#
# Data Abstractions
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class BarrierCounter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.must_use_result
    @gluon.jit
    def increment(self):
        if self.num_barriers == 1:
            return BarrierCounter(gl.to_tensor(0), self.phase ^ 1, self.num_barriers)
        next_index = self.index + 1
        rollover = next_index == self.num_barriers
        index = gl.where(rollover, 0, next_index)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return BarrierCounter(index, phase, self.num_barriers)


def Channel(T, alloc_fn):

    @gluon.aggregate
    class ChannelType:
        mem: T
        ready_bars: gl.shared_memory_descriptor
        empty_bars: gl.shared_memory_descriptor
        num_buffers: gl.constexpr

        @gluon.jit
        def alloc(shape: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr, num_buffers: gl.constexpr,
                  producer_two_ctas: gl.constexpr = False, consumer_two_ctas: gl.constexpr = False):
            mem = alloc_fn(dtype, [num_buffers] + shape, layout)
            ready_bars = mbarrier.allocate_mbarrier(batch=num_buffers, two_ctas=consumer_two_ctas)
            empty_bars = mbarrier.allocate_mbarrier(batch=num_buffers, two_ctas=producer_two_ctas)
            return ChannelType(mem, ready_bars, empty_bars, num_buffers)

        @gluon.jit
        def init(self, num_producers: gl.constexpr = 1, num_consumers: gl.constexpr = 1):
            for i in gl.static_range(self.num_buffers):
                mbarrier.init(self.ready_bars.index(i), count=num_producers)
                mbarrier.init(self.empty_bars.index(i), count=num_consumers)

        @gluon.jit
        def prime(self, num_consumers: gl.constexpr = 1):
            for i in gl.static_range(self.num_buffers):
                mbarrier.arrive(self.empty_bars.index(i), count=num_consumers)

        @gluon.jit
        def acquire_producer(self, counter, mem):
            index, phase = counter.index, counter.phase
            mem = mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)

            mbarrier.wait(empty_bar, phase)
            return mem, ready_bar

        @gluon.jit
        def acquire_consumer(self, counter, mem):
            index, phase = counter.index, counter.phase
            mem = mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)

            mbarrier.wait(ready_bar, phase)
            return mem, empty_bar

        @gluon.jit
        def create_counter(self):
            return BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), self.num_buffers)

        @gluon.jit
        def create_producer(self):
            return Producer(self, self.create_counter())

        @gluon.jit
        def create_consumer(self):
            return Consumer(self, self.create_counter())

    @gluon.aggregate
    class Producer:
        channel: ChannelType
        counter: BarrierCounter

        @gluon.jit
        def acquire(self):
            return self.acquire_from(self.channel.mem)

        @gluon.jit
        def acquire_from(self, mem):
            mem, ready_bar = self.channel.acquire_producer(self.counter, mem)
            next = Producer(self.channel, self.counter.increment())
            return mem, ready_bar, next

    @gluon.aggregate
    class Consumer:
        channel: ChannelType
        counter: BarrierCounter

        @gluon.jit
        def acquire(self):
            return self.acquire_from(self.channel.mem)

        @gluon.jit
        def acquire_from(self, mem):
            mem, empty_bar = self.channel.acquire_consumer(self.counter, mem)
            next = Consumer(self.channel, self.counter.increment())
            return mem, empty_bar, next

    return ChannelType, Producer, Consumer


SharedMemoryChannel, SharedMemoryProducer, SharedMemoryConsumer = Channel(gl.shared_memory_descriptor,
                                                                          gl.allocate_shared_memory)
TensorMemoryChannel, TensorMemoryProducer, TensorMemoryConsumer = Channel(tensor_memory_descriptor,
                                                                          allocate_tensor_memory)


@gluon.jit
def get_desc_channel(desc, num_buffers: gl.constexpr):
    shape: gl.constexpr = desc.block_type.shape
    layout: gl.constexpr = desc.layout
    return SharedMemoryChannel.alloc(shape, desc.dtype, layout, num_buffers, consumer_two_ctas=gl.num_ctas() > 1)


@gluon.jit
def issue_async_tma_load(smem, bar, desc, offset):
    mbarrier.expect(bar, desc.nbytes_per_cta)
    tma.async_load(desc, [offset, 0], bar, smem, multicast=True)


# ===-----------------------------------------------------------------------===#
# Gluon Attention
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionConfig:
    qk_scale: gl.tensor
    Z: gl.tensor
    H: gl.tensor
    N_CTX: gl.tensor
    CGA_LAYOUT: gl.constexpr
    m_cga_layout: gl.constexpr

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    HEAD_DIM: gl.constexpr
    GROUP_SIZE_N: gl.constexpr
    NUM_SMS: gl.constexpr
    dtype: gl.constexpr
    num_warps: gl.constexpr

    SPLIT_D_FACTOR: gl.constexpr
    SPLIT_EXP_FACTOR: gl.constexpr
    SPLIT_QK_LOAD_FACTOR: gl.constexpr
    SPLIT_M: gl.constexpr
    SPLIT_M_PER_CTA: gl.constexpr
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
    o_splitn_layout: gl.constexpr
    alpha_2d_layout: gl.constexpr

    num_kv_buffers: gl.constexpr
    use_exp2_turnstile: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, qk_scale, Z, H, N_CTX, CGA_LAYOUT, BLOCK_M, BLOCK_N, HEAD_DIM, GROUP_SIZE_N, NUM_SMS, STAGE,
                 SPLIT_EXP_FACTOR, dtype, num_warps, NUM_KV_BUFFERS, USE_EXP2_TURNSTILE):
        self.qk_scale = qk_scale
        self.Z = Z
        self.H = H
        self.N_CTX = N_CTX
        self.CGA_LAYOUT = gl.constexpr(CGA_LAYOUT)
        self.m_cga_layout = gl.constexpr(tuple((basis[0], ) for basis in self.CGA_LAYOUT))

        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.GROUP_SIZE_N = gl.constexpr(GROUP_SIZE_N)
        self.NUM_SMS = gl.constexpr(NUM_SMS)
        self.dtype = gl.constexpr(dtype)
        self.num_warps = gl.constexpr(num_warps)

        self.SPLIT_D_FACTOR = gl.constexpr(2)
        self.SPLIT_EXP_FACTOR = gl.constexpr(SPLIT_EXP_FACTOR)
        self.SPLIT_QK_LOAD_FACTOR = gl.constexpr(2 if STAGE == 1 else 1)
        self.SPLIT_M = gl.constexpr(self.BLOCK_M // 2)
        self.SPLIT_M_PER_CTA = gl.constexpr(self.SPLIT_M // get_split_dim(self.CGA_LAYOUT, 0))
        self.SPLIT_D = gl.constexpr(self.HEAD_DIM // self.SPLIT_D_FACTOR)

        self.q_shape = gl.constexpr([self.SPLIT_M, self.HEAD_DIM])
        self.k_shape = gl.constexpr([self.BLOCK_N, self.HEAD_DIM])
        self.qk_shape = gl.constexpr([self.SPLIT_M, self.BLOCK_N])
        self.v_shape = gl.constexpr([self.BLOCK_N, self.HEAD_DIM])
        self.o_shape = gl.constexpr([self.SPLIT_M, self.HEAD_DIM])

        qk_cta_shape = [self.SPLIT_M_PER_CTA, self.BLOCK_N // get_split_dim(self.CGA_LAYOUT, 1)]
        o_cta_shape = [self.SPLIT_M_PER_CTA, self.HEAD_DIM // get_split_dim(self.CGA_LAYOUT, 1)]
        qk_instr_shape = get_mma_instr_shape(qk_cta_shape, gl.float32)
        o_instr_shape = get_mma_instr_shape(o_cta_shape, gl.float32)
        self.qk_tmem_layout = gl.constexpr(
            TensorMemoryLayout((qk_instr_shape[0], qk_instr_shape[1]), col_stride=1, cga_layout=self.CGA_LAYOUT,
                               two_ctas=bool(self.CGA_LAYOUT)))
        self.o_tmem_layout = gl.constexpr(
            TensorMemoryLayout((o_instr_shape[0], o_instr_shape[1]), col_stride=1, cga_layout=self.CGA_LAYOUT,
                               two_ctas=bool(self.CGA_LAYOUT)))
        self.p_tmem_layout = gl.constexpr(
            TensorMemoryLayout((qk_instr_shape[0], qk_instr_shape[1]), col_stride=1, cga_layout=self.CGA_LAYOUT,
                               two_ctas=bool(self.CGA_LAYOUT)))
        o_splitn_tmem_layout: gl.constexpr = TensorMemoryLayout(
            (o_instr_shape[0], o_instr_shape[1] // self.SPLIT_D_FACTOR), col_stride=1, cga_layout=self.CGA_LAYOUT,
            two_ctas=bool(self.CGA_LAYOUT))
        qk_tmem_ty: gl.constexpr = tensor_memory_descriptor_type(gl.float32, self.qk_shape, self.qk_tmem_layout,
                                                                 self.qk_shape)
        o_splitn_tmem_ty: gl.constexpr = tensor_memory_descriptor_type(
            gl.float32,
            [self.o_shape[0], self.o_shape[1] // self.SPLIT_D_FACTOR],
            o_splitn_tmem_layout,
            self.o_shape,
        )

        self.qk_layout = gl.constexpr(qk_tmem_ty.get_reg_layout(num_warps=self.num_warps,
                                                                instr_variant="32x32b_splitn"))
        self.o_splitn_layout = gl.constexpr(o_splitn_tmem_ty.get_reg_layout(num_warps=self.num_warps))
        self.alpha_2d_layout = gl.constexpr(
            gl.BlockedLayout([1, 1], [32, 1], [self.num_warps, 1], [0, 1], cga_layout=self.CGA_LAYOUT))

        self.num_kv_buffers = gl.constexpr(NUM_KV_BUFFERS)
        self.use_exp2_turnstile = gl.constexpr(USE_EXP2_TURNSTILE)

    @gluon.jit
    def get_program(self, pid_m, pid_n):
        start_m = pid_m
        off_hz = pid_n
        off_z = off_hz // self.H
        off_h = off_hz % self.H

        offset_y = off_z * (self.N_CTX * self.H) + off_h * self.N_CTX
        qo_offset_y = offset_y + start_m * self.BLOCK_M

        return AttentionProgram(self, start_m, off_hz, offset_y, qo_offset_y)


@gluon.aggregate
class ProgramScheduler:
    config: AttentionConfig
    start_pid: gl.tensor
    num_pid_n: gl.tensor
    num_pid_in_group: gl.tensor
    num_tiles: gl.tensor

    @gluon.jit
    def create(config):
        start_pid = gl.program_id(0)
        num_pid_m = gl.cdiv(config.N_CTX, config.BLOCK_M)
        num_pid_n = config.Z * config.H
        num_pid_in_group = num_pid_m * config.GROUP_SIZE_N
        num_tiles = num_pid_m * num_pid_n
        return ProgramScheduler(config, start_pid, num_pid_n, num_pid_in_group, num_tiles)

    @gluon.jit
    def get_program(self, tile_id):
        group_id = tile_id // self.num_pid_in_group
        first_pid_n = group_id * self.config.GROUP_SIZE_N
        group_size_n = min(self.num_pid_n - first_pid_n, self.config.GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_m = (tile_id % self.num_pid_in_group) // group_size_n
        return self.config.get_program(pid_m, pid_n)


@gluon.aggregate
class AttentionProgram:
    config: AttentionConfig
    start_m: gl.tensor
    off_hz: gl.tensor
    offset_y: gl.tensor
    qo_offset_y: gl.tensor

    @gluon.jit
    def get_fused_loop_bounds(self, STAGE: gl.constexpr):
        BLOCK_M: gl.constexpr = self.config.BLOCK_M
        if STAGE == 1:
            return 0, self.config.N_CTX
        elif STAGE == 2:
            return self.start_m * BLOCK_M, (self.start_m + 1) * BLOCK_M
        elif STAGE == 3:
            return 0, (self.start_m + 1) * BLOCK_M
        else:
            return 0, 0

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


# ===-----------------------------------------------------------------------===#
# _gluon_attn
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _borrow_s_as_p(config, s_tmem):
    p_tmem = s_tmem._reinterpret(config.dtype, [config.SPLIT_M, 2 * config.BLOCK_N], config.p_tmem_layout)
    return p_tmem.slice(0, config.BLOCK_N)


@gluon.jit
def _borrow_s_as_alpha(config, s_tmem):
    alpha_layout: gl.constexpr = TensorMemoryLayout([config.SPLIT_M_PER_CTA, 1], col_stride=1,
                                                    cga_layout=config.CGA_LAYOUT, two_ctas=gl.num_ctas() > 1)
    alpha_tmem = s_tmem._reinterpret(layout=alpha_layout)
    return alpha_tmem.slice(config.BLOCK_N // 2, 1)


@gluon.jit
def _borrow_s_for_epilogue(config, s_tmem):
    layout: gl.constexpr = TensorMemoryLayout([config.SPLIT_M_PER_CTA, 1], col_stride=1, cga_layout=config.CGA_LAYOUT,
                                              two_ctas=gl.num_ctas() > 1)
    s_tmem = s_tmem._reinterpret(layout=layout)
    m_i_tmem = s_tmem.slice(config.BLOCK_N // 2 + 1, 1)
    l_i_tmem = s_tmem.slice(config.BLOCK_N // 2 + 2, 1)
    return m_i_tmem, l_i_tmem


@gluon.constexpr_function
def _get_split_n_layout(layout: gl.constexpr, SPLIT_FACTOR: gl.constexpr = 2):
    assert isinstance(layout, gl.DistributedLinearLayout), "split_n requires a distributed layout"
    assert SPLIT_FACTOR == 1 or SPLIT_FACTOR == 2, "split_n requires a split factor of 1 or 2"
    if SPLIT_FACTOR == 1:
        return layout
    else:
        target = [0, layout.shape[1] // 2]  # [0, 2^{m-1}]
        last_reg_idx = len(layout.reg_bases) - 1
        reg_last = layout.reg_bases[last_reg_idx]

        if reg_last == target:
            return layout

        ret = copy.deepcopy(layout)

        # Find [0, 2^{m-1}] across lists and swap it with last reg
        for L in (ret.reg_bases, ret.lane_bases, ret.warp_bases, ret.block_bases):
            for i, b in enumerate(L):
                if b == target:
                    L[i], ret.reg_bases[last_reg_idx] = reg_last, target
                    return ret
        assert False, f"split_n requires having a basis {target}. Got\n{layout}"


@gluon.jit
def _split_n(x, SPLIT_FACTOR: gl.constexpr = 2):
    if SPLIT_FACTOR == 1:
        return (x, )
    else:
        layout: gl.constexpr = _get_split_n_layout(x.type.layout)
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        x0 = gl.convert_layout(x0, layout, assert_trivial=True)
        x1 = gl.convert_layout(x1, layout, assert_trivial=True)
        return _split_n(x0, SPLIT_FACTOR // 2) + _split_n(x1, SPLIT_FACTOR // 2)


@gluon.constexpr_function
def _get_join_n_layout(layout, SPLIT_FACTOR: gl.constexpr = 2):
    assert isinstance(layout, gl.DistributedLinearLayout), "join_n requires a Linear layout"
    shape = list(layout.shape)
    regs = [[0, shape[1] * (1 << i)] for i in range(int(math.log2(SPLIT_FACTOR)))]
    shape[1] *= SPLIT_FACTOR
    return gl.DistributedLinearLayout(
        layout.reg_bases + regs,
        layout.lane_bases,
        layout.warp_bases,
        layout.block_bases,
        shape,
    )


@gluon.jit
def _join_n(xs):
    if len(xs) == 1:
        return xs[0]
    else:
        x0 = _join_n(xs[:len(xs) // 2])
        x1 = _join_n(xs[len(xs) // 2:])
        layout: gl.constexpr = _get_join_n_layout(x0.type.layout)
        x = gl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] * 2])
        return gl.convert_layout(x, layout, assert_trivial=True)


@gluon.jit
def _attn_fwd_load(config, chnls, descs, M, STAGE: gl.constexpr):
    q_chnl, kv_chnl, v_mem, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    q_producer = q_chnl.create_producer()
    kv_producer = kv_chnl.create_producer()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_fused_loop_bounds(STAGE)

        q0_offset = prog.qo_offset_y + config.SPLIT_M * 0
        q0_smem, q0_bar, q_producer = q_producer.acquire()
        issue_async_tma_load(q0_smem, q0_bar, desc_q, q0_offset)

        offsetkv_y = prog.offset_y + lo
        k_smem, k_bar, kv_producer = kv_producer.acquire()
        issue_async_tma_load(k_smem, k_bar, desc_k, offsetkv_y)

        q1_offset = prog.qo_offset_y + config.SPLIT_M * 1
        q1_smem, q1_bar, q_producer = q_producer.acquire()
        issue_async_tma_load(q1_smem, q1_bar, desc_q, q1_offset)

        v_smem, v_bar, kv_producer = kv_producer.acquire_from(v_mem)
        issue_async_tma_load(v_smem, v_bar, desc_v, offsetkv_y)

        for start_n in range(lo + config.BLOCK_N, hi, config.BLOCK_N):
            offsetkv_y = prog.offset_y + start_n
            k_smem, k_bar, kv_producer = kv_producer.acquire()
            issue_async_tma_load(k_smem, k_bar, desc_k, offsetkv_y)
            v_smem, v_bar, kv_producer = kv_producer.acquire_from(v_mem)
            issue_async_tma_load(v_smem, v_bar, desc_v, offsetkv_y)


@gluon.jit
def _attn_fwd_mma(config, chnls, descs, M, STAGE: gl.constexpr):
    q_chnl, kv_chnl, v_mem, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    q_consumer = q_chnl.create_consumer()
    kv_consumer = kv_chnl.create_consumer()
    o_producer = o_chnl.create_producer()

    s0_producer = s0_chnl.create_producer()
    s1_producer = s1_chnl.create_producer()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_fused_loop_bounds(STAGE)
        num_mmas = (hi - lo) // config.BLOCK_N

        q0_smem, q0_bar, q_consumer = q_consumer.acquire()
        k_smem, k_bar, kv_consumer = kv_consumer.acquire()
        s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
        tcgen05_mma(q0_smem, k_smem.permute((1, 0)), s0_tmem, use_acc=False, multicast=True, mbarriers=[s0_bar])

        q1_smem, q1_bar, q_consumer = q_consumer.acquire()
        s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
        tcgen05_mma(q1_smem, k_smem.permute((1, 0)), s1_tmem, use_acc=False, multicast=True, mbarriers=[s1_bar, k_bar])

        v_smem, v_bar, kv_consumer = kv_consumer.acquire_from(v_mem)
        o0_tmem, o0_bar, o_producer = o_producer.acquire()
        s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
        p0_tmem = _borrow_s_as_p(config, s0_tmem)
        tcgen05_mma(p0_tmem, v_smem, o0_tmem, use_acc=False, multicast=True, mbarriers=[o0_bar])
        o1_init = False

        for _ in range(num_mmas - 1):
            k_smem, k_bar, kv_consumer = kv_consumer.acquire()
            tcgen05_mma(q0_smem, k_smem.permute((1, 0)), s0_tmem, use_acc=False, multicast=True, mbarriers=[s0_bar])

            o1_tmem, o1_bar, o_producer = o_producer.acquire()
            s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
            p1_tmem = _borrow_s_as_p(config, s1_tmem)
            tcgen05_mma(p1_tmem, v_smem, o1_tmem, use_acc=o1_init, multicast=True, mbarriers=[o1_bar, v_bar])
            o1_init = True

            tcgen05_mma(q1_smem, k_smem.permute((1, 0)), s1_tmem, use_acc=False, multicast=True,
                        mbarriers=[s1_bar, k_bar])

            v_smem, v_bar, kv_consumer = kv_consumer.acquire_from(v_mem)
            o0_tmem, o0_bar, o_producer = o_producer.acquire()
            s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
            p0_tmem = _borrow_s_as_p(config, s0_tmem)
            tcgen05_mma(p0_tmem, v_smem, o0_tmem, multicast=True, mbarriers=[o0_bar])

        tcgen05_commit(q0_bar, descs=[q0_smem, k_smem.permute((1, 0))])
        tcgen05_commit(q1_bar, descs=[q1_smem, k_smem.permute((1, 0))])

        o1_tmem, o1_bar, o_producer = o_producer.acquire()
        s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
        p1_tmem = _borrow_s_as_p(config, s1_tmem)
        tcgen05_mma(p1_tmem, v_smem, o1_tmem, use_acc=o1_init, multicast=True,
                    mbarriers=[o1_bar, v_bar, s0_bar, s1_bar])


@gluon.jit
def _mask_scalar(qk, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s
    col_lim_right_cur = max(col_lim_right_s, 0)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return gl.where(mask_i_bit, qk, -float("inf"))


@gluon.jit
def _apply_causal_mask(qk, col_limit_right):
    # Apply causal mask via a bitmask calculated for each block of 16 elements.
    # This allows the efficient R2P (register to predicate) instruction to be used at the SASS level.
    # Credit to Tri Dao,
    # https://github.com/Dao-AILab/flash-attention/commit/bac1001e4f6caa09d70537495d6746a685a2fa78
    #
    # NOTE: We use map_elementiwse here in order to generate an interleaved sequence of instructions
    # that processes one element of qk at a time. This improves ptxas's resulting SASS.
    offs_n = gl.arange(0, qk.shape[1])[None, :]
    s = offs_n & ~0xf
    i = offs_n & 0xf
    return gl.map_elementwise(_mask_scalar, qk, col_limit_right, s, i)


@gluon.jit
def _compute_and_store_exp2(config, qk, p_tmem):
    SIZE: gl.constexpr = p_tmem.shape[1] // config.SPLIT_EXP_FACTOR
    qks = _split_n(qk, config.SPLIT_EXP_FACTOR)
    ps = ()
    for i in gl.static_range(config.SPLIT_EXP_FACTOR):
        p = gl.exp2(qks[i])
        p_tmem.slice(i * SIZE, SIZE).store(p.to(config.dtype))
        ps = ps + (p, )
    return _join_n(ps)


@gluon.jit
def _subtiled_qk_load(config, s_tmem, use_tmem_red: gl.constexpr):
    SIZE: gl.constexpr = s_tmem.shape[1] // config.SPLIT_QK_LOAD_FACTOR
    qks = ()
    if use_tmem_red:
        red_total = None
        for i in gl.static_range(config.SPLIT_QK_LOAD_FACTOR):
            vals, reds = s_tmem.slice(i * SIZE, SIZE).load_max()
            red_total = reds if red_total is None else gl.maximum(red_total, reds)
            qks = qks + (vals, )
        return _join_n(qks), red_total
    else:
        for i in gl.static_range(config.SPLIT_QK_LOAD_FACTOR):
            qks = qks + (s_tmem.slice(i * SIZE, SIZE).load(), )
        return _join_n(qks), None


@gluon.jit
def _softmax_inner_loop(tile_id: gl.constexpr, config, prog,  #
                        s_consumer, corr_producer, exp_turnstile, corr_bar,  #
                        offs_m, m_i, l_i, STAGE: gl.constexpr, use_tmem_red: gl.constexpr):
    lo, hi = prog.get_loop_bounds(STAGE)

    for start_n in range(lo, hi, config.BLOCK_N):
        s_tmem, s_bar, s_consumer = s_consumer.acquire()
        qk, qk_max = _subtiled_qk_load(config, s_tmem, use_tmem_red)

        if STAGE == 2:
            col_limit_right = (offs_m - start_n + 1)[:, None]
            qk = _apply_causal_mask(qk, col_limit_right)

        if use_tmem_red:
            qk_max = gl.convert_layout(qk_max, m_i.type.layout)
            m_ij = gl.maximum(m_i, qk_max * config.qk_scale)
        else:
            m_ij = gl.maximum(m_i, gl.max(qk, 1) * config.qk_scale)
        alpha = gl.exp2(m_i - m_ij)

        alpha_tmem = _borrow_s_as_alpha(config, s_tmem)
        alpha_tmem.store(gl.convert_layout(alpha.expand_dims(1), config.alpha_2d_layout))
        mbarrier.arrive(corr_bar, count=1)

        rowmax = float2.pack(-m_ij[:, None].broadcast_to(qk.shape), axis=1)
        qk = float2.pack(qk, axis=1)
        qk = float2.fma(qk, float2.full_like(qk, config.qk_scale), rowmax)
        qk = float2.unpack(qk, axis=1)

        # Force the softmax partitions to take turns in the EX2 section. This
        # prevents contention for the EX2 unit and improves utilization.
        if config.use_exp2_turnstile:
            _, exp_bar, exp_turnstile = exp_turnstile.acquire()

        # FIXME: When using FADD2 reductions, ptxas misbehaves and spills far
        # below the register limit in the FADD2, FMUL2, EX2 section. Subtile by
        # 4 to minimize the spilling.
        p_tmem = _borrow_s_as_p(config, s_tmem)
        p = _compute_and_store_exp2(config, qk, p_tmem)

        mbarrier.arrive(s_bar, count=1)
        _, corr_bar, corr_producer = corr_producer.acquire()

        if config.use_exp2_turnstile:
            mbarrier.arrive(exp_bar, count=1)

        l_ij = float2.pack2(*_split_n(p)).sum(axis=1)
        l_ij = Float2Tensor(gl.convert_layout(l_ij.value, l_i.value.type.layout, assert_trivial=True))
        alpha = gl.convert_layout(alpha, l_i.value.type.layout, assert_trivial=True)
        l_i = float2.fma(l_i, float2.pack2(alpha, alpha), l_ij)
        m_i = m_ij

    return m_i, l_i, corr_bar, s_consumer, corr_producer, exp_turnstile


@gluon.jit
def _softmax_tile(tile_id: gl.constexpr, config, M, desc_o, STAGE: gl.constexpr,  #
                  s_chnl, corr_chnl, exp_turnstile, use_tmem_red: gl.constexpr):
    qk_slice_dim1: gl.constexpr = gl.SliceLayout(1, config.qk_layout)
    sum_layout: gl.constexpr = _get_split_n_layout(config.qk_layout)

    s_consumer = s_chnl.create_consumer()
    corr_producer = corr_chnl.create_producer()
    _, corr_bar, corr_producer = corr_producer.acquire()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)

        offs_m = prog.start_m * config.BLOCK_M
        offs_m += gl.arange(tile_id * config.SPLIT_M, (1 + tile_id) * config.SPLIT_M)

        m_i = gl.full([config.SPLIT_M], -float("inf"), gl.float32, qk_slice_dim1)
        # Accumulate into 2 row-sums so the reduction can be performed with FADD2.
        l_i = gl.full([config.SPLIT_M], 0.0, gl.float32, gl.SliceLayout(1, sum_layout))
        l_i = float2.pack2(l_i, l_i)

        if STAGE & 1:
            m_i, l_i, corr_bar, s_consumer, corr_producer, exp_turnstile = _softmax_inner_loop(  #
                tile_id, config, prog, s_consumer, corr_producer, exp_turnstile, corr_bar,  #
                offs_m, m_i, l_i, STAGE=4 - STAGE, use_tmem_red=use_tmem_red)
        if STAGE & 2:
            m_i, l_i, corr_bar, s_consumer, corr_producer, exp_turnstile = _softmax_inner_loop(  #
                tile_id, config, prog, s_consumer, corr_producer, exp_turnstile, corr_bar,  #
                offs_m, m_i, l_i, STAGE=2, use_tmem_red=use_tmem_red)
        l_i0, l_i1 = float2.unpack2(l_i)
        l_i = l_i0 + l_i1

        s_tmem, s_bar, s_consumer = s_consumer.acquire()
        m_i_tmem, l_i_tmem = _borrow_s_for_epilogue(config, s_tmem)
        m_i_tmem.store(gl.convert_layout(m_i.expand_dims(1), config.alpha_2d_layout))
        l_i_tmem.store(gl.convert_layout(l_i.expand_dims(1), config.alpha_2d_layout))

        mbarrier.arrive(corr_bar, count=1)
        _, corr_bar, corr_producer = corr_producer.acquire()

        mbarrier.arrive(s_bar, count=1)


@gluon.jit
def _attn_fwd_softmax0(config, chnls, descs, M, STAGE: gl.constexpr, use_tmem_red: gl.constexpr):
    q_chnl, kv_chnl, v_mem, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls
    desc_q, desc_k, desc_v, desc_o = descs
    _softmax_tile(0, config, M, desc_o, STAGE, s0_chnl, c0_chnl, exp_turnstile.create_producer(), use_tmem_red)


@gluon.jit
def _attn_fwd_softmax1(config, chnls, descs, M, STAGE: gl.constexpr, use_tmem_red: gl.constexpr):
    q_chnl, kv_chnl, v_mem, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls
    desc_q, desc_k, desc_v, desc_o = descs
    _softmax_tile(1, config, M, desc_o, STAGE, s1_chnl, c1_chnl, exp_turnstile.create_consumer(), use_tmem_red)


@gluon.jit
def _attn_fwd_epilogue(config, chnls, descs, M, STAGE: gl.constexpr):
    q_chnl, kv_chnl, v_mem, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    epi_consumer = epi_chnl.create_consumer()
    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)

        o0_smem, o0_bar, epi_consumer = epi_consumer.acquire()
        tma.async_copy_shared_to_global(desc_o, [prog.qo_offset_y + config.SPLIT_M * 0, 0], o0_smem)

        o1_smem, o1_bar, epi_consumer = epi_consumer.acquire()
        tma.async_copy_shared_to_global(desc_o, [prog.qo_offset_y + config.SPLIT_M * 1, 0], o1_smem)

        tma.store_wait(1)
        mbarrier.arrive(o0_bar, count=1)
        tma.store_wait(0)
        mbarrier.arrive(o1_bar, count=1)


@gluon.jit
def _attn_fwd_correction_rescale(config, s_tmem, corr_consumer, o_consumer):
    alpha_layout: gl.constexpr = gl.SliceLayout(1, config.o_splitn_layout)

    o_tmem, o_bar, o_consumer = o_consumer.acquire()

    _, corr_bar, corr_consumer = corr_consumer.acquire()
    alpha = _borrow_s_as_alpha(config, s_tmem).load(config.alpha_2d_layout)
    mbarrier.arrive(corr_bar, count=1)
    alpha = gl.convert_layout(alpha.reshape([config.SPLIT_M]), alpha_layout)

    alpha = float2.pack(alpha[:, None].broadcast_to(config.o_shape[0], config.SPLIT_D), axis=1)
    for i in gl.static_range(config.SPLIT_D_FACTOR):
        o_ref = o_tmem.slice(i * config.SPLIT_D, config.SPLIT_D)
        o = float2.pack(o_ref.load(config.o_splitn_layout), axis=1)
        o = o * alpha
        o_ref.store(float2.unpack(o, axis=1))
    mbarrier.arrive(o_bar, count=1)
    return corr_consumer, o_consumer


@gluon.jit
def _attn_fwd_correction_epilogue(config, prog, s_tmem, M, corr_consumer, epi_producer, o_consumer):
    alpha_layout: gl.constexpr = gl.SliceLayout(1, config.o_splitn_layout)

    _, corr_bar, corr_consumer = corr_consumer.acquire()
    m_i_tmem, l_i_tmem = _borrow_s_for_epilogue(config, s_tmem)
    m_i = m_i_tmem.load(config.alpha_2d_layout).reshape([config.SPLIT_M])
    m_i = gl.convert_layout(m_i, alpha_layout)
    l_i = l_i_tmem.load(config.alpha_2d_layout).reshape([config.SPLIT_M])
    l_i = gl.convert_layout(l_i, alpha_layout)
    mbarrier.arrive(corr_bar, count=1)

    o_smem, epi_bar, epi_producer = epi_producer.acquire()
    o_tmem, o_bar, o_consumer = o_consumer.acquire()

    # Shared memory subtile size is limited by the swizzle byte size.
    contigDimSize: gl.constexpr = o_smem.type.layout.swizzle_byte_width * 8 // o_smem.type.element_ty.primitive_bitwidth
    if o_smem.type.shape[1] // config.SPLIT_D_FACTOR >= contigDimSize:
        SPLIT_N_FACTOR: gl.constexpr = config.SPLIT_D_FACTOR
    else:
        SPLIT_N_FACTOR: gl.constexpr = 1
    gl.static_assert(o_smem.type.shape[1] // SPLIT_N_FACTOR >= contigDimSize,
                     "Block shape is too small for the swizzle byte size in NVMMA Shared Layout")
    SPLIT_N: gl.constexpr = o_smem.type.shape[1] // SPLIT_N_FACTOR

    scale = float2.pack((1 / l_i)[:, None].broadcast_to(config.o_shape[0], SPLIT_N), axis=1)
    for i in gl.static_range(SPLIT_N_FACTOR):
        o_ref = o_tmem.slice(i * SPLIT_N, SPLIT_N)
        o = float2.pack(o_ref.load(config.o_splitn_layout), axis=1)
        o = o * scale
        o_smem.slice(i * SPLIT_N, SPLIT_N, dim=1).store(float2.unpack(o, axis=1).to(config.dtype))

    fence_async_shared()
    mbarrier.arrive(epi_bar, count=1)
    mbarrier.arrive(o_bar, count=1)

    m_i += gl.log2(l_i)
    coalesced: gl.constexpr = gl.BlockedLayout([1], [32], [config.num_warps], [0], cga_layout=config.m_cga_layout)
    offs_m = prog.start_m * config.BLOCK_M
    offs_m += gl.arange(0 * config.SPLIT_M, 1 * config.SPLIT_M, coalesced)
    m_ptrs = M + prog.off_hz * config.N_CTX + offs_m
    gl.store(m_ptrs, gl.convert_layout(m_i, coalesced))

    return corr_consumer, epi_producer, o_consumer


@gluon.jit
def _attn_fwd_correction(config, chnls, descs, M, STAGE: gl.constexpr):
    q_chnl, kv_chnl, v_mem, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls

    s0_tmem = s0_chnl.mem.index(0)
    s1_tmem = s1_chnl.mem.index(0)
    corr0_consumer = c0_chnl.create_consumer()
    corr1_consumer = c1_chnl.create_consumer()
    o_consumer = o_chnl.create_consumer()

    epi_producer = epi_chnl.create_producer()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_fused_loop_bounds(STAGE)
        num_corrections = (hi - lo) // config.BLOCK_N

        _, corr0_bar, corr0_consumer = corr0_consumer.acquire()
        mbarrier.arrive(corr0_bar, count=1)
        _, corr1_bar, corr1_consumer = corr1_consumer.acquire()
        mbarrier.arrive(corr1_bar, count=1)

        for i in range(num_corrections - 1):
            corr0_consumer, o_consumer = _attn_fwd_correction_rescale(config, s0_tmem, corr0_consumer, o_consumer)
            corr1_consumer, o_consumer = _attn_fwd_correction_rescale(config, s1_tmem, corr1_consumer, o_consumer)

        corr0_consumer, epi_producer, o_consumer = _attn_fwd_correction_epilogue(  #
            config, prog, s0_tmem, M, corr0_consumer, epi_producer, o_consumer)
        corr1_consumer, epi_producer, o_consumer = _attn_fwd_correction_epilogue(  #
            config, prog, s1_tmem, M, corr1_consumer, epi_producer, o_consumer)


def attention_repr(specialization):
    name = "gluon_attention"
    # Up to 150 TFLOPS faster for fp8!
    if specialization.constants["dtype"] == gl.float8e5:
        name = "cutlass_" + name
    return name


@gluon.jit(do_not_specialize=["Z", "H", "N_CTX"], repr=attention_repr)
def attention_kernel(  #
        sm_scale, M, Z, H, N_CTX, desc_q, desc_k, desc_v, desc_o,  #
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
        GROUP_SIZE_N: gl.constexpr, NUM_SMS: gl.constexpr, STAGE: gl.constexpr, SPLIT_EXP_FACTOR: gl.constexpr,  #
        dtype: gl.constexpr, num_warps: gl.constexpr, use_tmem_red: gl.constexpr, NUM_KV_BUFFERS: gl.constexpr,
        USE_EXP2_TURNSTILE: gl.constexpr, CGA_LAYOUT: gl.constexpr):
    qk_scale = sm_scale * 1.44269504
    config = AttentionConfig(qk_scale, Z, H, N_CTX, CGA_LAYOUT, BLOCK_M, BLOCK_N, HEAD_DIM, GROUP_SIZE_N, NUM_SMS,
                             STAGE, SPLIT_EXP_FACTOR,  #
                             dtype, num_warps, NUM_KV_BUFFERS, USE_EXP2_TURNSTILE)

    q_chnl = get_desc_channel(desc_q, num_buffers=2)
    kv_chnl = get_desc_channel(desc_k, num_buffers=config.num_kv_buffers)
    v_mem = kv_chnl.mem._reinterpret(layout=desc_v.layout)
    o_chnl = TensorMemoryChannel.alloc(config.o_shape, gl.float32, config.o_tmem_layout, num_buffers=2,
                                       producer_two_ctas=gl.num_ctas() > 1)
    epi_chnl = SharedMemoryChannel.alloc(config.o_shape, config.dtype, gl.constexpr(desc_o.layout), num_buffers=2)
    s0_chnl = TensorMemoryChannel.alloc(config.qk_shape, gl.float32, config.qk_tmem_layout, num_buffers=1,
                                        producer_two_ctas=gl.num_ctas() > 1)
    s1_chnl = TensorMemoryChannel.alloc(config.qk_shape, gl.float32, config.qk_tmem_layout, num_buffers=1,
                                        producer_two_ctas=gl.num_ctas() > 1)
    sync_layout: gl.constexpr = mbarrier.MBarrierLayout.multicta(gl.num_ctas())
    c0_chnl = SharedMemoryChannel.alloc([gl.num_ctas()], gl.int8, sync_layout, num_buffers=1)
    c1_chnl = SharedMemoryChannel.alloc([gl.num_ctas()], gl.int8, sync_layout, num_buffers=1)
    exp_turnstile = SharedMemoryChannel.alloc([gl.num_ctas()], gl.int8, sync_layout, num_buffers=1)

    qk_mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [q_chnl.mem.index(0), kv_chnl.mem.index(0).permute((1, 0))],
        multicast=True,
        two_ctas=s0_chnl.mem.index(0).type.layout.two_ctas,
    )
    pv_mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [v_mem.index(0)],
        multicast=True,
        two_ctas=o_chnl.mem.index(0).type.layout.two_ctas,
    )
    gl.static_assert(qk_mma_barrier_count == pv_mma_barrier_count,
                     "shared KV channel requires matching K and V consumer counts")

    q_chnl.init(num_consumers=qk_mma_barrier_count)
    kv_chnl.init(num_consumers=qk_mma_barrier_count)
    o_chnl.init(num_producers=pv_mma_barrier_count)
    epi_chnl.init()
    s0_chnl.init(num_producers=qk_mma_barrier_count)
    s1_chnl.init(num_producers=qk_mma_barrier_count)
    c0_chnl.init()
    c1_chnl.init()
    exp_turnstile.init()

    q_chnl.prime(qk_mma_barrier_count)
    kv_chnl.prime(qk_mma_barrier_count)
    o_chnl.prime()
    epi_chnl.prime()
    s0_chnl.prime()
    s1_chnl.prime()
    c0_chnl.prime()
    c1_chnl.prime()
    exp_turnstile.prime()

    chnls = (q_chnl, kv_chnl, v_mem, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile)
    descs = (desc_q, desc_k, desc_v, desc_o)
    gl.warp_specialize([
        (_attn_fwd_correction, (config, chnls, descs, M, STAGE)),
        (_attn_fwd_softmax0, (config, chnls, descs, M, STAGE, use_tmem_red)),
        (_attn_fwd_softmax1, (config, chnls, descs, M, STAGE, use_tmem_red)),
        (_attn_fwd_mma, (config, chnls, descs, M, STAGE)),
        (_attn_fwd_load, (config, chnls, descs, M, STAGE)),
        (_attn_fwd_epilogue, (config, chnls, descs, M, STAGE)),
    ], [4, 4, 1, 1, 1], [192, 192, 24, 24, 24])


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_blackwell_ultra():
    return is_cuda() and torch.cuda.get_device_capability()[0:2] == (10, 3)


@dataclass(frozen=True, slots=True)
class KernelConfig:
    BLOCK_M: int = 256
    BLOCK_N: int = 128
    GROUP_SIZE_N: int | None = None
    SPLIT_EXP_FACTOR: int | None = None
    NUM_WARPS: int = 4
    MAXNREG: int = 128
    OCCUPANCY: int = 1
    USE_TMEM_RED: bool = False
    NUM_KV_BUFFERS: int | None = None
    USE_EXP2_TURNSTILE: bool | None = None
    CTA_LAYOUT: tuple[tuple[int, int], ...] | None = None


def _default_split_exp_factor(head_dim: int) -> int:
    return max(1, 256 // head_dim)


def _default_num_kv_buffers(head_dim: int, dtype: torch.dtype) -> int:
    is_fp16 = dtype in [torch.float16, torch.bfloat16]
    if is_fp16:
        return 3 if head_dim == 128 else 6
    return 4 if head_dim == 128 else 8


def select_kernel_config(
    head_dim: int,
    n_ctx: int,
    dtype: torch.dtype,
    causal: bool,
    use_tmem_red: bool,
    override: KernelConfig | None = None,
) -> KernelConfig:
    is_fp8 = dtype == torch.float8_e5m2
    is_bf16 = dtype == torch.bfloat16
    is_bwu = is_blackwell_ultra()

    block_m = 256
    block_n = 128
    group_size_n = 1
    split_exp_factor = _default_split_exp_factor(head_dim)
    num_warps = 4
    maxnreg = 128
    occupancy = 1
    use_selected_tmem_red = (use_tmem_red or (is_bwu and not causal)) and not causal
    num_kv_buffers = _default_num_kv_buffers(head_dim, dtype)
    use_exp2_turnstile = head_dim == 64
    cta_layout = ()
    if not causal and head_dim == 128 and dtype.itemsize == 2:
        cta_layout = ((1, 0), )

    if causal:
        group_size_n = 8 if head_dim == 64 or n_ctx <= 2048 else 4

    if head_dim == 128:
        split_exp_factor = 4
        if not causal and is_bf16 and n_ctx <= 2048:
            group_size_n = 4
    elif not causal and head_dim == 64 and use_selected_tmem_red:
        split_exp_factor = 1
        if n_ctx <= 1024:
            num_kv_buffers = 2
        elif n_ctx >= 8192:
            maxnreg = 112
    elif causal and head_dim == 64:
        num_kv_buffers = 2
        if n_ctx <= 1024:
            split_exp_factor = 2
        else:
            use_exp2_turnstile = False

    if is_fp8:
        if causal and head_dim == 64:
            group_size_n = 8 if n_ctx <= 2048 else 4
            split_exp_factor = 4 if n_ctx <= 2048 else 2
            maxnreg = 112 if n_ctx >= 4096 else 128
            use_selected_tmem_red = False
            num_kv_buffers = 2
            use_exp2_turnstile = n_ctx <= 1024
        elif causal and head_dim == 128:
            group_size_n = 8 if n_ctx <= 2048 else 4
            split_exp_factor = 2 if n_ctx <= 2048 else 8
            maxnreg = 128
            use_selected_tmem_red = False
            num_kv_buffers = 4
            use_exp2_turnstile = False
        elif not causal and head_dim == 64:
            group_size_n = 1
            split_exp_factor = 2
            maxnreg = 128
            use_selected_tmem_red = is_bwu
            num_kv_buffers = 2 if n_ctx <= 1024 else 8
            use_exp2_turnstile = True
        elif not causal and head_dim == 128:
            group_size_n = 1
            split_exp_factor = 4 if n_ctx <= 2048 else 8
            maxnreg = 128
            use_selected_tmem_red = is_bwu
            num_kv_buffers = 4
            use_exp2_turnstile = False
        else:
            group_size_n = 4 if causal else 1
            split_exp_factor = _default_split_exp_factor(head_dim)
            use_selected_tmem_red = use_tmem_red and not causal

    config = KernelConfig(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_SIZE_N=group_size_n,
        SPLIT_EXP_FACTOR=split_exp_factor,
        NUM_WARPS=num_warps,
        MAXNREG=maxnreg,
        OCCUPANCY=occupancy,
        USE_TMEM_RED=use_selected_tmem_red,
        NUM_KV_BUFFERS=num_kv_buffers,
        USE_EXP2_TURNSTILE=use_exp2_turnstile,
        CTA_LAYOUT=cta_layout,
    )
    if override is None:
        return config

    values = {field.name: getattr(override, field.name) for field in fields(KernelConfig)}
    values = {name: getattr(config, name) if value is None else value for name, value in values.items()}
    return KernelConfig(**values)


def torch_dtype_to_triton(dtype):
    if dtype == torch.float8_e5m2:
        return gl.float8e5
    return getattr(gl, str(dtype).split('.')[1])


def make_tensor_desc(x, shape, strides, block_shape, cga_layout=()):
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, torch_dtype_to_triton(x.dtype), cga_layout=cga_layout)
    return TensorDescriptor(x, shape=shape, strides=strides, block_shape=block_shape, layout=layout)


def attention_forward(q, k, v, causal, sm_scale, o=None, M=None, *, use_tmem_red=False, p: KernelConfig | None = None,
                      cta_layout=None):
    if isinstance(o, bool) and M is None and use_tmem_red is False:
        use_tmem_red = o
        o = None

    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    stage = 3 if causal else 1
    p = select_kernel_config(HEAD_DIM_K, q.shape[2], q.dtype, causal, use_tmem_red, override=p)
    if cta_layout is None:
        cta_layout = p.CTA_LAYOUT

    if o is None:
        o = torch.empty_like(q)
    if M is None:
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    # The kernel will split the cluster tile into two subtiles. Keep the
    # configured M tile per CTA and grow the cluster tile with the CTA layout.
    BLOCK_M = p.BLOCK_M * get_split_dim(cta_layout, 0)
    BLOCK_N = p.BLOCK_N
    SPLIT_M = BLOCK_M // 2
    GROUP_SIZE_N = p.GROUP_SIZE_N
    num_ctas = 2**len(cta_layout)
    NUM_SMS = max(1, torch.cuda.get_device_properties("cuda").multi_processor_count * p.OCCUPANCY // num_ctas)

    lhs_cga_layout = get_mma_operand_cga_layout(cta_layout, 0)
    rhs_cga_layout = get_mma_operand_cga_layout(cta_layout, 1)
    k_cga_layout = tuple((basis[1], basis[0]) for basis in rhs_cga_layout)

    desc_q = make_tensor_desc(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[SPLIT_M, HEAD_DIM_K],
                              cga_layout=lhs_cga_layout)
    desc_v = make_tensor_desc(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[BLOCK_N, HEAD_DIM_K],
                              cga_layout=rhs_cga_layout)
    desc_k = make_tensor_desc(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[BLOCK_N, HEAD_DIM_K],
                              cga_layout=k_cga_layout)
    desc_o = make_tensor_desc(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[SPLIT_M, HEAD_DIM_K],
                              cga_layout=cta_layout)

    num_pid_m = triton.cdiv(q.shape[2], BLOCK_M)
    num_pid_n = q.shape[0] * q.shape[1]
    grid = min(NUM_SMS, num_pid_m * num_pid_n)

    attention_kernel[(grid, )](
        sm_scale, M, q.shape[0], q.shape[1], q.shape[2],  #
        desc_q, desc_k, desc_v, desc_o,  #
        BLOCK_M, BLOCK_N, HEAD_DIM_K, GROUP_SIZE_N, NUM_SMS,  #
        SPLIT_EXP_FACTOR=p.SPLIT_EXP_FACTOR, STAGE=stage, dtype=torch_dtype_to_triton(q.dtype),  #
        num_warps=p.NUM_WARPS, maxnreg=p.MAXNREG, use_tmem_red=p.USE_TMEM_RED, NUM_KV_BUFFERS=p.NUM_KV_BUFFERS,
        USE_EXP2_TURNSTILE=p.USE_EXP2_TURNSTILE, CGA_LAYOUT=cta_layout, num_ctas=num_ctas)

    return o, M


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("N_CTX", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_tmem_red", [False, True] if is_blackwell_ultra() else [False])
@pytest.mark.parametrize("cta_layout", [(), ((1, 0), ), ((1, 0), (2, 0))], ids=["1cta", "2ctas", "4ctas"])
@pytest.mark.skipif(not is_blackwell(), reason="Gluon attention is only supported on Blackwell GPUs")
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, use_tmem_red, cta_layout, profile=False):
    device = "cuda"

    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    if use_tmem_red and not is_blackwell_ultra():
        pytest.skip("TMEM reduction is only supported on Blackwell Ultra GPUs")

    torch.manual_seed(42)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5

    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)

    tri_out, _ = attention_forward(q, k, v, causal, sm_scale, use_tmem_red=use_tmem_red, cta_layout=cta_layout)
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
use_tmem_reds = [False, True] if is_blackwell_ultra() else [False]

bench_configs = []
for Z, H, D, is_causal, use_tmem_red in itertools.product(BATCH, N_HEADS, HEAD_DIM, causal, use_tmem_reds):
    config = triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=N_CTX,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")],
        ylabel="TFLOPS",
        plot_name=f"Attention Z={Z} H={H} D={D} causal={is_causal} use_tmem_red={use_tmem_red}",
        args={
            "Z": Z,
            "H": H,
            "HEAD_DIM": D,
            "causal": is_causal,
            "use_tmem_red": use_tmem_red,
        },
    )
    bench_configs.append(config)


@triton.testing.perf_report(bench_configs)
def bench(Z, H, N_CTX, HEAD_DIM, causal, use_tmem_red, provider):
    provider, dtype = provider.split("-")
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
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

    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
        if provider == "triton":
            fn = lambda: attention_forward(q, k, v, causal, sm_scale, o, M, use_tmem_red=use_tmem_red)
        elif provider == "cudnn":
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        try:
            import triton.profiler  # noqa: F401
            bench_fn = triton.testing.do_bench_cudagraph_proton
        except ImportError:
            # Fallback to do_bench as do_bench_cudagraph does not clear the L2 cache.
            bench_fn = triton.testing.do_bench
        ms = bench_fn(fn)
        flops_per_matmul = 2.0 * Z * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench.run(save_path=".", print_data=True)
