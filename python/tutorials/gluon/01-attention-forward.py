import torch
import triton
import triton.language as tl
import pytest
import itertools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tma,
    mbarrier,
    tcgen05_mma as _tcgen05_mma_impl,
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
# Data Abstractions
# ===-----------------------------------------------------------------------===#


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


@gluon.jit
def get_desc_channel(desc, num_buffers: gl.constexpr, num_consumers: gl.constexpr = 1):
    shape: gl.constexpr = desc.block_type.shape
    layout: gl.constexpr = desc.layout
    return SharedMemoryChannel.alloc(shape, desc.dtype, layout, num_buffers, num_consumers)


@tl.constexpr_function
def get_load_size_bytes(desc):
    size = desc.dtype.primitive_bitwidth // 8
    for dim in desc.block_type.shape:
        size *= dim
    return size


@gluon.jit
def issue_async_tma_load(smem, bar, desc, offset):
    size: gl.constexpr = get_load_size_bytes(desc)
    mbarrier.expect(bar, size)
    tma.async_copy_global_to_shared(desc, [offset, 0], bar, smem)


@gluon.jit
def tcgen05_mma(a, b, d, use_acc, mbarriers):
    _tcgen05_mma_impl(a, b, d, use_acc=use_acc, mbarriers=mbarriers, mbarrier_preds=[True] * len(mbarriers))


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
    alpha_2d_layout: gl.constexpr

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
        self.alpha_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])

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

    def get_program(self, tile_id):
        group_id = tile_id // self.num_pid_in_group
        first_pid_n = group_id * self.config.GROUP_SIZE_N
        group_size_n = min(self.num_pid_n - first_pid_n, self.config.GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_m = (tile_id % self.num_pid_in_group) // group_size_n
        return self.config.get_program(pid_m, pid_n)


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

    def get_loop_bounds(self, STAGE: gl.constexpr):
        BLOCK_M: gl.constexpr = self.config.BLOCK_M
        if STAGE == 1:
            lo, hi = 0, self.start_m * BLOCK_M
        elif STAGE == 2:
            lo, hi = self.start_m * BLOCK_M, (self.start_m + 1) * BLOCK_M
        else:
            lo, hi = 0, self.config.N_CTX
        return lo, hi


@gl.aggregate
class InnerLoopInfo:
    s_chnl: TensorMemoryChannel
    corr_chnl: SharedMemoryChannel

    def __init__(self, config):
        self.s_chnl = TensorMemoryChannel.alloc(config.qk_shape, gl.float32, config.qk_tmem_layout, num_buffers=1)
        self.s_chnl.initialize_for_producer()

        self.corr_chnl = SharedMemoryChannel.alloc([1], gl.int8, gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1)
        self.corr_chnl.initialize_for_producer()

    def release(self):
        self.s_chnl.release()
        self.corr_chnl.release()


# ===-----------------------------------------------------------------------===#
# float2
# ===-----------------------------------------------------------------------===#


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
def _reduce_fadd2(p0a, p1a, p0b, p1b):
    return gl.inline_asm_elementwise(
        """
        {
            .reg .b64 rc, ra, rb;
            mov.b64 ra, { $2, $4 };
            mov.b64 rb, { $3, $5 };
            add.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [p0a, p0b, p1a, p1b],
        dtype=[gl.float32, gl.float32],
        is_pure=True,
        pack=1,
    )


# ===-----------------------------------------------------------------------===#
# _gluon_attn
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _borrow_s_as_p(config, s_tmem):
    p_tmem = s_tmem.slice(0, config.BLOCK_N // 2)
    return p_tmem._reinterpret(config.dtype, config.qk_shape, config.p_tmem_layout)


@gluon.jit
def _borrow_s_as_alpha(config, s_tmem):
    alpha_tmem = s_tmem.slice(config.BLOCK_N // 2, 1)
    alpha_layout: gl.constexpr = TensorMemoryLayout([config.SPLIT_M, 1], unpacked=False)
    return alpha_tmem._reinterpret(gl.float32, [config.SPLIT_M, 1], alpha_layout)


@gluon.jit
def _borrow_s_for_epilogue(config, s_tmem):
    m_i_tmem = s_tmem.slice(config.BLOCK_N // 2 + 1, 1)
    l_i_tmem = s_tmem.slice(config.BLOCK_N // 2 + 2, 1)
    layout: gl.constexpr = TensorMemoryLayout([config.SPLIT_M, 1], unpacked=False)
    m_i_tmem = m_i_tmem._reinterpret(gl.float32, [config.SPLIT_M, 1], layout)
    l_i_tmem = l_i_tmem._reinterpret(gl.float32, [config.SPLIT_M, 1], layout)
    return m_i_tmem, l_i_tmem


@gluon.jit
def _attn_fwd_load(  #
        config, info0, info1,  #
        q_chnl, kv_chnl, o_chnl, epi_chnl,  #
        M, desc_q, desc_k, desc_v, desc_o,  #
        STAGE: gl.constexpr):
    q_producer = q_chnl.create_producer()
    kv_producer = kv_chnl.create_producer()

    scheduler = ProgramScheduler(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_fused_loop_bounds(STAGE)

        q0_offset = prog.qo_offset_y + config.SPLIT_M * 0
        q0_smem, q0_bar = q_producer.acquire()
        issue_async_tma_load(q0_smem, q0_bar, desc_q, q0_offset)

        offsetkv_y = prog.offset_y + lo
        k_smem, k_bar = kv_producer.acquire()
        issue_async_tma_load(k_smem, k_bar, desc_k, offsetkv_y)

        q1_offset = prog.qo_offset_y + config.SPLIT_M * 1
        q1_smem, q1_bar = q_producer.acquire()
        issue_async_tma_load(q1_smem, q1_bar, desc_q, q1_offset)

        v_smem, v_bar = kv_producer.acquire()
        issue_async_tma_load(v_smem, v_bar, desc_v, offsetkv_y)

        for start_n in range(lo + config.BLOCK_N, hi, config.BLOCK_N):
            offsetkv_y = prog.offset_y + start_n
            k_smem, k_bar = kv_producer.acquire()
            issue_async_tma_load(k_smem, k_bar, desc_k, offsetkv_y)
            v_smem, v_bar = kv_producer.acquire()
            issue_async_tma_load(v_smem, v_bar, desc_v, offsetkv_y)


@gluon.jit
def _attn_fwd_mma(  #
        config, info0, info1,  #
        q_chnl, kv_chnl, o_chnl, epi_chnl,  #
        M, desc_q, desc_k, desc_v, desc_o,  #
        STAGE: gl.constexpr):
    q_consumer = q_chnl.create_consumer()
    kv_consumer = kv_chnl.create_consumer()
    o_producer = o_chnl.create_producer()

    s0_producer = info0.s_chnl.create_producer()
    s1_producer = info1.s_chnl.create_producer()

    scheduler = ProgramScheduler(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_fused_loop_bounds(STAGE)
        num_mmas = (hi - lo) // config.BLOCK_N

        q0_smem, q0_bar = q_consumer.acquire()
        k_smem, k_bar = kv_consumer.acquire()
        s0_tmem, s0_bar = s0_producer.acquire()
        tcgen05_mma(q0_smem, k_smem.permute((1, 0)), s0_tmem, use_acc=False, mbarriers=[s0_bar])

        q1_smem, q1_bar = q_consumer.acquire()
        s1_tmem, s1_bar = s1_producer.acquire()
        tcgen05_mma(q1_smem, k_smem.permute((1, 0)), s1_tmem, use_acc=False, mbarriers=[s1_bar, k_bar])

        v_smem, v_bar = kv_consumer.acquire()
        o0_tmem, o0_bar = o_producer.acquire()
        s0_tmem, s0_bar = s0_producer.acquire()
        p0_tmem = _borrow_s_as_p(config, s0_tmem)
        tcgen05_mma(p0_tmem, v_smem, o0_tmem, use_acc=False, mbarriers=[o0_bar])
        o_init = False

        for _ in range(num_mmas - 1):
            k_smem, k_bar = kv_consumer.acquire()
            tcgen05_mma(q0_smem, k_smem.permute((1, 0)), s0_tmem, use_acc=False, mbarriers=[s0_bar])

            o1_tmem, o1_bar = o_producer.acquire()
            s1_tmem, s1_bar = s1_producer.acquire()
            p1_tmem = _borrow_s_as_p(config, s1_tmem)
            tcgen05_mma(p1_tmem, v_smem, o1_tmem, use_acc=o_init, mbarriers=[o1_bar, v_bar])
            o_init = True

            tcgen05_mma(q1_smem, k_smem.permute((1, 0)), s1_tmem, use_acc=False, mbarriers=[s1_bar, k_bar])

            v_smem, v_bar = kv_consumer.acquire()
            o0_tmem, o0_bar = o_producer.acquire()
            s0_tmem, s0_bar = s0_producer.acquire()
            p0_tmem = _borrow_s_as_p(config, s0_tmem)
            tcgen05_mma(p0_tmem, v_smem, o0_tmem, use_acc=o_init, mbarriers=[o0_bar])
            o_init = True

        tcgen05_commit(q0_bar)
        tcgen05_commit(q1_bar)

        o1_tmem, o1_bar = o_producer.acquire()
        s1_tmem, s1_bar = s1_producer.acquire()
        p1_tmem = _borrow_s_as_p(config, s1_tmem)
        tcgen05_mma(p1_tmem, v_smem, o1_tmem, use_acc=o_init, mbarriers=[o1_bar, v_bar, s0_bar, s1_bar])


@gluon.jit
def _softmax_inner_loop(tile_id: gl.constexpr, config, prog,  #
                        s_consumer, corr_producer, corr_bar,  #
                        offs_m, offs_n, m_i, l_i,  #
                        STAGE: gl.constexpr):
    lo, hi = prog.get_loop_bounds(STAGE)

    for start_n in range(lo, hi, config.BLOCK_N):
        s_tmem, s_bar = s_consumer.acquire()
        qk = s_tmem.load(config.qk_layout)

        if STAGE == 2:
            # Prevent LLVM from hoisting the partial sums, which triggers spilling.
            offs_n = gl.inline_asm_elementwise("mov.b32 $0, $0;", "=r,r", [offs_n], dtype=gl.int32, is_pure=True,
                                               pack=1)
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = gl.where(mask, qk, -1.0e8)
        m_ij = gl.maximum(m_i, gl.max(qk, 1) * config.qk_scale)
        alpha = gl.exp2(m_i - m_ij)

        alpha_tmem = _borrow_s_as_alpha(config, s_tmem)
        alpha_tmem.store(gl.convert_layout(alpha.expand_dims(1), config.alpha_2d_layout, assert_trivial=True))
        mbarrier.arrive(corr_bar, count=1)

        qk = _mul_f32x2(qk, gl.full(config.qk_shape, config.qk_scale, gl.float32, qk.type.layout))
        qk = _add_f32x2(qk, -m_ij[:, None])

        p_tmem = _borrow_s_as_p(config, s_tmem)
        qk0, qk1, = qk.reshape([config.SPLIT_M, 2, config.BLOCK_N // 2]).permute(0, 2, 1).split()
        p0 = gl.exp2(qk0)
        p_tmem.slice(0, config.BLOCK_N // 2).store(p0.to(config.dtype))
        p1 = gl.exp2(qk1)
        p_tmem.slice(config.BLOCK_N // 2, config.BLOCK_N // 2).store(p1.to(config.dtype))

        _, corr_bar = corr_producer.acquire()
        mbarrier.arrive(s_bar, count=1)

        # FIXME: This code makes ptxas misbehave and spill :(
        # p0 = gl.convert_layout(p0, config.qk_layout, assert_trivial=True)
        # p1 = gl.convert_layout(p1, config.qk_layout, assert_trivial=True)
        # l_ij0, l_ij1 = gl.reduce((p0, p1), axis=1, combine_fn=_reduce_fadd2)
        # l_i0, l_i1 = gl.inline_asm_elementwise(
        #     """
        #     {
        #         .reg .b64 rout, rli, rlij, ralpha;
        #         mov.b64 rli, { $2, $4 };
        #         mov.b64 rlij, { $3, $5 };
        #         mov.b64 ralpha, { $6, $6 };
        #         fma.rn.f32x2 rout, rli, ralpha, rlij;
        #         mov.b64 { $0, $1 }, rout;
        #     }
        #     """,
        #     "=r,=r,r,r,r,r,r",
        #     [l_i0, l_ij0, l_i1, l_ij1, alpha],
        #     dtype=[gl.float32, gl.float32],
        #     is_pure=True,
        #     pack=1,
        # )

        p = gl.join(p0, p1).permute(0, 2, 1).reshape([config.SPLIT_M, config.BLOCK_N])
        p = gl.convert_layout(p, config.qk_layout, assert_trivial=True)
        l_ij = gl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return m_i, l_i, corr_bar


@gluon.jit
def _softmax_tile(  #
        tile_id: gl.constexpr, config, info, M, desc_o, STAGE: gl.constexpr,  #
        s_chnl, corr_chnl,  #
):
    qk_slice_dim0: gl.constexpr = gl.SliceLayout(0, config.qk_layout)
    qk_slice_dim1: gl.constexpr = gl.SliceLayout(1, config.qk_layout)

    offs_n = gl.arange(0, config.BLOCK_N, qk_slice_dim0)

    s_consumer = s_chnl.create_consumer()
    corr_producer = corr_chnl.create_producer()
    _, corr_bar = corr_producer.acquire()

    scheduler = ProgramScheduler(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)

        offs_m = prog.start_m * config.BLOCK_M
        offs_m += gl.arange(tile_id * config.SPLIT_M, (1 + tile_id) * config.SPLIT_M, qk_slice_dim1)

        m_i = gl.full([config.SPLIT_M], -float("inf"), gl.float32, qk_slice_dim1)
        l_i = gl.full([config.SPLIT_M], 1.0, gl.float32, qk_slice_dim1)

        if STAGE & 1:
            m_i, l_i, corr_bar = _softmax_inner_loop(tile_id, config, prog,  #
                                                     s_consumer, corr_producer, corr_bar,  #
                                                     offs_m, offs_n, m_i, l_i,  #
                                                     STAGE=4 - STAGE)
        if STAGE & 2:
            m_i, l_i, corr_bar = _softmax_inner_loop(tile_id, config, prog,  #
                                                     s_consumer, corr_producer, corr_bar,  #
                                                     offs_m, offs_n, m_i, l_i,  #
                                                     STAGE=2)

        s_tmem, s_bar = s_consumer.acquire()
        m_i_tmem, l_i_tmem = _borrow_s_for_epilogue(config, s_tmem)
        m_i_tmem.store(gl.convert_layout(m_i.expand_dims(1), config.alpha_2d_layout, assert_trivial=True))
        l_i_tmem.store(gl.convert_layout(l_i.expand_dims(1), config.alpha_2d_layout, assert_trivial=True))

        mbarrier.arrive(corr_bar, count=1)
        _, corr_bar = corr_producer.acquire()

        mbarrier.arrive(s_bar, count=1)


@gluon.jit
def _attn_fwd_softmax0(  #
        config, info0, info1,  #
        q_chnl, kv_chnl, o_chnl, epi_chnl,  #
        M, desc_q, desc_k, desc_v, desc_o,  #
        STAGE: gl.constexpr):
    _softmax_tile(0, config, info0, M, desc_o, STAGE, info0.s_chnl, info0.corr_chnl)


@gluon.jit
def _attn_fwd_softmax1(  #
        config, info0, info1,  #
        q_chnl, kv_chnl, o_chnl, epi_chnl,  #
        M, desc_q, desc_k, desc_v, desc_o,  #
        STAGE: gl.constexpr):
    _softmax_tile(1, config, info1, M, desc_o, STAGE, info1.s_chnl, info1.corr_chnl)


@gluon.jit
def _attn_fwd_epilogue(  #
        config, info0, info1,  #
        q_chnl, kv_chnl, o_chnl, epi_chnl,  #
        M, desc_q, desc_k, desc_v, desc_o,  #
        STAGE: gl.constexpr):
    epi_consumer = epi_chnl.create_consumer()

    scheduler = ProgramScheduler(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)

        o0_smem, o0_bar = epi_consumer.acquire()
        tma.async_copy_shared_to_global(desc_o, [prog.qo_offset_y + config.SPLIT_M * 0, 0], o0_smem)

        o1_smem, o1_bar = epi_consumer.acquire()
        tma.async_copy_shared_to_global(desc_o, [prog.qo_offset_y + config.SPLIT_M * 1, 0], o1_smem)

        tma.store_wait(1)
        mbarrier.arrive(o0_bar, count=1)
        tma.store_wait(0)
        mbarrier.arrive(o1_bar, count=1)


@gluon.jit
def _attn_fwd_correction_rescale(config, alpha, o_tmem):
    for i in tl.static_range(config.SPLIT_D_FACTOR):
        o_ref = o_tmem.slice(i * config.SPLIT_D, config.SPLIT_D)
        o = o_ref.load(config.o_splitn_layout)
        o = _mul_f32x2(o, alpha[:, None])
        o_ref.store(o)


@gluon.jit
def _attn_fwd_correction_epilogue(config, scale, o_tmem, o_smem):
    contigDimSize: gl.constexpr = o_smem.type.layout.swizzle_byte_width * 8 / o_smem.type.element_ty.primitive_bitwidth
    if o_smem.type.shape[1] // config.SPLIT_D_FACTOR >= contigDimSize:
        SPLIT_N_FACTOR: gl.constexpr = config.SPLIT_D_FACTOR
    else:
        SPLIT_N_FACTOR: gl.constexpr = 1
    gl.static_assert(o_smem.type.shape[1] // SPLIT_N_FACTOR >= contigDimSize,
                     "Block shape is too small for the swizzle byte size in NVMMA Shared Layout")
    SPLIT_N: gl.constexpr = o_smem.type.shape[1] // SPLIT_N_FACTOR

    for i in tl.static_range(SPLIT_N_FACTOR):
        o_ref = o_tmem.slice(i * SPLIT_N, SPLIT_N)
        o = o_ref.load(config.o_splitn_layout)
        o = _mul_f32x2(o, scale[:, None])
        o_smem.slice(i * SPLIT_N, SPLIT_N, dim=1).store(o.to(config.dtype))


@gluon.jit
def _attn_fwd_correction(config, info0, info1, M, o_chnl, epi_chnl, STAGE: gl.constexpr):
    alpha_layout: gl.constexpr = gl.SliceLayout(1, config.o_splitn_layout)

    s0_tmem = info0.s_chnl.mem.index(0)
    s1_tmem = info1.s_chnl.mem.index(0)

    o_consumer = o_chnl.create_consumer()
    corr0_consumer = info0.corr_chnl.create_consumer()
    corr1_consumer = info1.corr_chnl.create_consumer()

    epi_producer = epi_chnl.create_producer()

    scheduler = ProgramScheduler(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_fused_loop_bounds(STAGE)
        num_corrections = (hi - lo) // config.BLOCK_N

        _, corr0_bar = corr0_consumer.acquire()
        mbarrier.arrive(corr0_bar, count=1)

        _, corr1_bar = corr1_consumer.acquire()
        mbarrier.arrive(corr1_bar, count=1)

        for i in range(num_corrections - 1):
            _, corr0_bar = corr0_consumer.acquire()
            alpha0 = _borrow_s_as_alpha(config, s0_tmem).load(config.alpha_2d_layout)
            mbarrier.arrive(corr0_bar, count=1)
            alpha0 = gl.convert_layout(alpha0.reshape([config.SPLIT_M]), alpha_layout, assert_trivial=True)

            o0_tmem, o0_bar = o_consumer.acquire()
            _attn_fwd_correction_rescale(config, alpha0, o0_tmem)
            mbarrier.arrive(o0_bar, count=1)

            _, corr1_bar = corr1_consumer.acquire()
            alpha1 = _borrow_s_as_alpha(config, s1_tmem).load(config.alpha_2d_layout)
            mbarrier.arrive(corr1_bar, count=1)
            alpha1 = gl.convert_layout(alpha1.reshape([config.SPLIT_M]), alpha_layout, assert_trivial=True)

            o1_tmem, o1_bar = o_consumer.acquire()
            _attn_fwd_correction_rescale(config, alpha1, o1_tmem)
            mbarrier.arrive(o1_bar, count=1)

        _, corr0_bar = corr0_consumer.acquire()
        m_i0_tmem, l_i0_tmem = _borrow_s_for_epilogue(config, s0_tmem)
        m_i0 = m_i0_tmem.load(config.alpha_2d_layout).reshape([config.SPLIT_M])
        m_i0 = gl.convert_layout(m_i0, alpha_layout, assert_trivial=True)
        l_i0 = l_i0_tmem.load(config.alpha_2d_layout).reshape([config.SPLIT_M])
        l_i0 = gl.convert_layout(l_i0, alpha_layout, assert_trivial=True)
        mbarrier.arrive(corr0_bar, count=1)

        o0_smem, epi_bar = epi_producer.acquire()
        o0_tmem, o0_bar = o_consumer.acquire()
        _attn_fwd_correction_epilogue(config, 1 / l_i0, o0_tmem, o0_smem)
        fence_async_shared()
        mbarrier.arrive(epi_bar, count=1)
        mbarrier.arrive(o0_bar, count=1)

        m_i0 += gl.log2(l_i0)
        coalesced: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        offs_m = prog.start_m * config.BLOCK_M
        offs_m += gl.arange(0 * config.SPLIT_M, 1 * config.SPLIT_M, coalesced)
        m_ptrs = M + prog.off_hz * config.N_CTX + offs_m
        gl.store(m_ptrs, gl.convert_layout(m_i0, coalesced, assert_trivial=True))

        _, corr1_bar = corr1_consumer.acquire()
        m_i1_tmem, l_i1_tmem = _borrow_s_for_epilogue(config, s1_tmem)
        m_i1 = m_i1_tmem.load(config.alpha_2d_layout).reshape([config.SPLIT_M])
        m_i1 = gl.convert_layout(m_i1, alpha_layout, assert_trivial=True)
        l_i1 = l_i1_tmem.load(config.alpha_2d_layout).reshape([config.SPLIT_M])
        l_i1 = gl.convert_layout(l_i1, alpha_layout, assert_trivial=True)
        mbarrier.arrive(corr1_bar, count=1)

        o1_smem, epi_bar = epi_producer.acquire()
        o1_tmem, o1_bar = o_consumer.acquire()
        _attn_fwd_correction_epilogue(config, 1 / l_i1, o1_tmem, o1_smem)
        fence_async_shared()
        mbarrier.arrive(epi_bar, count=1)
        mbarrier.arrive(o1_bar, count=1)

        m_i1 += gl.log2(l_i1)
        offs_m = prog.start_m * config.BLOCK_M
        offs_m += gl.arange(1 * config.SPLIT_M, 2 * config.SPLIT_M, coalesced)
        m_ptrs = M + prog.off_hz * config.N_CTX + offs_m
        gl.store(m_ptrs, gl.convert_layout(m_i1, coalesced, assert_trivial=True))


@gluon.jit
def _attn_fwd_inner(config, info0, info1,  #
                    desc_q, desc_k, desc_v, desc_o, M,  #
                    STAGE: gl.constexpr):
    gl.static_assert(desc_k.layout == desc_v.layout and desc_k.block_type == desc_v.block_type,
                     "expected K and V to have the same type and shared memory layout")
    if config.dtype == tl.float16:
        num_kv_buffers: gl.constexpr = 3 if config.HEAD_DIM == 128 else 6
    else:
        num_kv_buffers: gl.constexpr = 4 if config.HEAD_DIM == 128 else 8
    kv_chnl = get_desc_channel(desc_k, num_buffers=num_kv_buffers, num_consumers=1)
    kv_chnl.initialize_for_producer()

    o_chnl = TensorMemoryChannel.alloc(config.o_shape, gl.float32, config.o_tmem_layout, num_buffers=2)
    o_chnl.initialize_for_producer()

    q_chnl = get_desc_channel(desc_q, num_buffers=2)
    q_chnl.initialize_for_producer()

    epi_chnl = SharedMemoryChannel.alloc(config.o_shape, config.dtype, gl.constexpr(desc_o.layout), num_buffers=2)
    epi_chnl.initialize_for_producer()

    gl.warp_specialize((
        config,
        info0,
        info1,
        M,
        o_chnl,
        epi_chnl,
        STAGE,
    ), _attn_fwd_correction, (
        config,
        info0,
        info1,
        q_chnl,
        kv_chnl,
        o_chnl,
        epi_chnl,
        M,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        STAGE,
    ), [
        _attn_fwd_softmax0,
        _attn_fwd_softmax1,
        _attn_fwd_mma,
        _attn_fwd_load,
        _attn_fwd_epilogue,
    ], [4, 4, 1, 1, 1], [192, 192, 24, 24, 24])

    q_chnl.release()
    kv_chnl.release()
    o_chnl.release()
    epi_chnl.release()


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
        num_warps=4, maxnreg=128)

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

profile = False

if not profile:
    BATCH = [4]
    N_HEADS = [32]
    HEAD_DIM = [64, 128]
    causal = [False, True]
    providers = ["triton-fp16", "triton-fp8"]
    N_CTX = [2**i for i in range(10, 17)]
else:
    BATCH = [4]
    N_HEADS = [32]
    HEAD_DIM = [128]
    causal = [True]
    providers = ["triton-fp16"]
    N_CTX = [16 * 1024]

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

        if not profile:
            ms = triton.testing.do_bench(fn)
        else:
            ms, _ = 1, fn()
        flops_per_matmul = 2.0 * Z * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench.run(save_path=".", print_data=True)
