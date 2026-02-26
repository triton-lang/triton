import argparse

import pytest
import torch

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    clc,
    get_tmem_reg_layout,
    tcgen05_commit,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
    tensor_memory_descriptor,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


def as_gl_dtype(torch_dtype):
    if torch_dtype == torch.float16:
        return gl.float16
    if torch_dtype == torch.bfloat16:
        return gl.bfloat16
    if torch_dtype == torch.float32:
        return gl.float32
    raise ValueError(f"Unsupported dtype for Gluon layout: {torch_dtype}")


@gluon.constexpr_function
def get_split_dim(cga_layout, dim):
    return 1 << sum(b[dim] != 0 for b in cga_layout)


def matmul_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GRID_MINOR_DIM": minor_dim,
                "GRID_TILE_WIDTH": grid_tile_width,
                "STAGES": stages,
                "ACC_STAGES": acc_stages,
                # Remove after #7777 lands
                "EPILOGUE_SIZE_N": epilogue_size_n if not cga_layout or BM != 64 else BN,
                "CGA_LAYOUT": cga_layout,
            },
            num_warps=4,
            num_ctas=2**len(cga_layout),
            pre_hook=pre_hook,
        )
        for BM in (128, )
        for BN in (128, 256, 512)
        for BK in (64, 128)
        for minor_dim in (0, 1)
        for grid_tile_width in (4, 8, 16)
        for stages in (2, 4, 6)
        for acc_stages in (2, )
        for epilogue_size_n in (32, 64)
        for cga_layout in ((), ((1, 0), ), ((1, 0), (2, 0)))
        if BN // get_split_dim(cga_layout, 1) <= 256
    ]


def matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    epilogue_size_n = nargs["EPILOGUE_SIZE_N"]
    cga_layout = nargs["CGA_LAYOUT"]

    tile_m = block_m * get_split_dim(cga_layout, 0)
    nargs["a_desc"].block_shape = [tile_m, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [tile_m, epilogue_size_n]

    # Janky version of make_2cta_cga_layout
    def get_cga_layout(layout, op_idx):
        assert op_idx in (0, 1)
        if not layout:
            return layout
        ret = list(layout)
        # 2cta
        if op_idx == 1:
            ret[0] = (0, 1)
        inner = 1 - op_idx
        outer = op_idx
        for i in range(1, len(ret)):
            b = list(ret[i])
            b[inner] = 0
            if op_idx == 1:
                b[outer] *= 2
            ret[i] = tuple(b)
        return tuple(ret)

    cga_layout_a = get_cga_layout(cga_layout, 0)
    cga_layout_b = get_cga_layout(cga_layout, 1)
    cga_layout_c = cga_layout
    for desc, cga_layout in zip(("a_desc", "b_desc", "c_desc"), (cga_layout_a, cga_layout_b, cga_layout_c)):
        nargs[desc].layout = gl.NVMMASharedLayout.get_default_for(
            nargs[desc].block_shape,
            as_gl_dtype(nargs[desc].base.dtype),
            cga_layout=cga_layout,
        )


# From Pallas / CUTLASS
@gluon.jit
def _planar_snake(lin_idx, m_tiles, n_tiles, minor_dim: gl.constexpr, tile_width: gl.constexpr):
    major_size = n_tiles if minor_dim == 0 else m_tiles
    minor_size = m_tiles if minor_dim == 0 else n_tiles

    full_minor_tiles = minor_size // tile_width
    full_minor_size = full_minor_tiles * tile_width
    full_elements = full_minor_tiles * tile_width * major_size

    minor_tile_idx = lin_idx // (tile_width * major_size)

    full_minor_within = lin_idx % tile_width
    full_major_within = (lin_idx // tile_width) % major_size
    full_minor = minor_tile_idx * tile_width + full_minor_within
    full_major = gl.where((minor_tile_idx % 2) == 0, full_major_within, major_size - 1 - full_major_within)

    partial_width = minor_size - full_minor_size
    partial_width = gl.where(partial_width > 0, partial_width, 1)
    partial_lin = lin_idx - full_elements
    partial_minor_within = partial_lin % partial_width
    partial_major_within = (partial_lin // partial_width) % major_size
    partial_minor = minor_tile_idx * tile_width + partial_minor_within
    partial_major = gl.where((minor_tile_idx % 2) == 0, partial_major_within, major_size - 1 - partial_major_within)

    in_full_tile = lin_idx < full_elements
    minor = gl.where(in_full_tile, full_minor, partial_minor)
    major = gl.where(in_full_tile, full_major, partial_major)

    if minor_dim == 0:
        return minor, major
    return major, minor


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


@aggregate
class ClcTileSchedulerConsumer:
    has_work: gl.tensor
    tile_id: gl.tensor
    num_pid_m: gl.tensor
    num_pid_n: gl.tensor
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor
    counter: Counter
    consumed_counter: Counter

    @gluon.jit
    def initialize(M, N, TILE_M: gl.constexpr, TILE_N: gl.constexpr, MINOR_DIM: gl.constexpr,
                   GRID_TILE_WIDTH: gl.constexpr, clc_result_buffers, clc_barriers, clc_consumed_bars):
        tile_id = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(M, TILE_M)
        num_pid_n = gl.cdiv(N, TILE_N)
        has_work = gl.to_tensor(True)
        counter = Counter.create(0, clc_barriers.shape[0])
        consumed_counter = Counter.create(0, clc_barriers.shape[0])
        return ClcTileSchedulerConsumer(
            has_work,
            tile_id,
            num_pid_m,
            num_pid_n,
            TILE_M,
            TILE_N,
            MINOR_DIM,
            GRID_TILE_WIDTH,
            clc_result_buffers,
            clc_barriers,
            clc_consumed_bars,
            counter,
            consumed_counter,
        )

    @gluon.jit
    def get_offsets(self):
        pid_m, pid_n = _planar_snake(self.tile_id, self.num_pid_m, self.num_pid_n, self.MINOR_DIM, self.GRID_TILE_WIDTH)
        return pid_m * self.TILE_M, pid_n * self.TILE_N

    @gluon.jit
    def step(self, iteration):
        # The 0-th iteration uses the program_id as the tile_id.
        # At the end of each iteration we prefetch the next tile.
        # As such we must signal the consumed slot at the end of
        # each iteration skipping the first one.
        consumed_counter = self.consumed_counter
        if iteration > 0:
            mbarrier.arrive(self.clc_consumed_bars.index(consumed_counter.index))
            consumed_counter = consumed_counter.next()
        counter = self.counter
        barrier = self.clc_barriers.index(counter.index)
        result = self.clc_result_buffers.index(counter.index)
        mbarrier.wait(barrier, counter.phase)
        clc_res = clc.load_result(result)
        has_work = clc_res.is_canceled()
        tile_id = self.tile_id
        if has_work:
            tile_id = clc_res.program_id(0)
        return ClcTileSchedulerConsumer(
            has_work,
            tile_id,
            self.num_pid_m,
            self.num_pid_n,
            self.TILE_M,
            self.TILE_N,
            self.MINOR_DIM,
            self.GRID_TILE_WIDTH,
            self.clc_result_buffers,
            self.clc_barriers,
            self.clc_consumed_bars,
            counter.next(),
            consumed_counter,
        )


@aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr

    @gluon.jit
    def get_clc_consumer(self):
        return ClcTileSchedulerConsumer.initialize(
            self.c_desc.shape[0],
            self.c_desc.shape[1],
            self.a_desc.block_shape[0],
            self.b_desc.block_shape[1],
            self.MINOR_DIM,
            self.GRID_TILE_WIDTH,
            self.clc_result_buffers,
            self.clc_barriers,
            self.clc_consumed_bars,
        )


@gluon.jit
def matmul_clc_partition(p):
    has_work = gl.to_tensor(True)
    state = Counter.create(0, p.clc_barriers.shape[0])
    consumed_state = Counter.create(1, p.clc_barriers.shape[0])
    ACC_STAGES: gl.constexpr = p.clc_barriers.shape[0]
    i = 0
    while has_work:
        # Reuse the slot only after all consumer partitions signaled consumed.
        mbarrier.wait(p.clc_consumed_bars.index(consumed_state.index), consumed_state.phase, pred=(i >= ACC_STAGES))
        barrier = p.clc_barriers.index(state.index)
        result = p.clc_result_buffers.index(state.index)
        # 16: clc.try_cancel has a `.b128` modifier
        mbarrier.expect(barrier, 16)
        clc.try_cancel(result, barrier, multicast=True)
        mbarrier.wait(barrier, state.phase)
        has_work = clc.load_result(result).is_canceled()
        state = state.next()
        consumed_state = consumed_state.next()
        i += 1


@gluon.jit
def matmul_load_partition(p):
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1]
    K = p.a_desc.shape[1]

    concurrent_loads: gl.constexpr = p.load_ready_bars.shape[0]
    state = Counter.create(1, concurrent_loads)
    scheduler = p.get_clc_consumer()

    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()
        for k in range(0, K, BLOCK_K):
            pred = (i > 0) or (k >= BLOCK_K * concurrent_loads)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase, pred=pred)
            bar = p.load_ready_bars.index(state.index)
            mbarrier.expect(bar, p.a_desc.nbytes_per_cta + p.b_desc.nbytes_per_cta)
            tma.async_copy_global_to_shared(p.a_desc, [off_m, k], bar, p.a_bufs.index(state.index), multicast=True)
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index), multicast=True)
            state = state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def matmul_mma_partition(p):
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1]
    K = p.a_desc.shape[1]
    ACC_STAGES: gl.constexpr = p.acc_empty_bars.shape[0]

    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, ACC_STAGES)
    scheduler = p.get_clc_consumer()

    i = 0
    while scheduler.has_work:
        acc_buf = p.acc_bufs.index(acc_state.index)
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase, pred=(i >= ACC_STAGES))
        use_acc = False
        for k in range(0, K, BLOCK_K):
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
            tcgen05_mma(p.a_bufs.index(load_state.index), p.b_bufs.index(load_state.index), acc_buf, use_acc=use_acc,
                        multicast=True, mbarriers=[p.load_empty_bars.index(load_state.index)])
            load_state = load_state.next()
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index), descs=[p.a_bufs.index(0), p.b_bufs.index(0)])
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def matmul_epilogue_partition(p):
    TILE_M: gl.constexpr = p.a_desc.block_shape[0]
    TILE_N: gl.constexpr = p.b_desc.block_shape[1]
    SPLIT_TILE_N: gl.constexpr = p.c_desc.block_shape[1]
    SUBTILE_FACTOR: gl.constexpr = TILE_N // SPLIT_TILE_N
    ACC_STAGES: gl.constexpr = p.acc_empty_bars.shape[0]
    dtype: gl.constexpr = p.c_desc.dtype

    acc_state = Counter.create(0, ACC_STAGES)
    acc_smems = gl.allocate_shared_memory(dtype, [ACC_STAGES, TILE_M, SPLIT_TILE_N], p.c_desc.layout)
    sub_acc_state = Counter.create(0, ACC_STAGES)
    scheduler = p.get_clc_consumer()

    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()

        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)

        for s in gl.static_range(SUBTILE_FACTOR):
            acc_sub = acc_buf.slice(SPLIT_TILE_N * s, SPLIT_TILE_N)
            acc_smem = acc_smems.index(sub_acc_state.index)
            acc = acc_sub.load(
                get_tmem_reg_layout(
                    gl.float32,
                    (TILE_M, SPLIT_TILE_N),
                    acc_sub.type.layout,
                    gl.num_warps(),
                )).to(dtype)
            tma.store_wait(pendings=1)
            acc_smem.store(acc)
            fence_async_shared()
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + SPLIT_TILE_N * s], acc_smem)
            sub_acc_state = sub_acc_state.next()
        # Signal that the accumulator slot can be reused only after all stores are done.
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def _matmul_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GRID_MINOR_DIM: gl.constexpr,
    GRID_TILE_WIDTH: gl.constexpr,
    STAGES: gl.constexpr,
    ACC_STAGES: gl.constexpr,
    CGA_LAYOUT: gl.constexpr,
    EPILOGUE_SIZE_N: gl.constexpr,
):
    BLOCK_M: gl.constexpr = a_desc.block_shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_shape[1]
    TWO_CTAS: gl.constexpr = gl.num_ctas() > 1
    N_PARTITIONS: gl.constexpr = 4

    dtype: gl.constexpr = a_desc.dtype
    a_bufs = gl.allocate_shared_memory(dtype, [STAGES] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [STAGES] + b_desc.block_shape, b_desc.layout)
    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count([a_bufs.index(0), b_bufs.index(0)], multicast=True)

    # Equiv. consumed_barrier. Barrier TCGEN05 MMA -> Load TMA
    load_empty_bars = mbarrier.allocate_mbarrier(batch=STAGES)
    # Equiv. ab_tma_barrier. Barrier Load TMA -> TCGEN05 MMA
    load_ready_bars = mbarrier.allocate_mbarrier(batch=STAGES, two_ctas=TWO_CTAS)
    for i in gl.static_range(STAGES):
        mbarrier.init(load_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready_bars.index(i), count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout(
        [BLOCK_SIZE_M, BLOCK_N // get_split_dim(CGA_LAYOUT, 1)],
        col_stride=1,
        cga_layout=CGA_LAYOUT,
        two_ctas=TWO_CTAS,
    )
    acc_bufs = allocate_tensor_memory(gl.float32, [ACC_STAGES, BLOCK_M, BLOCK_N], tmem_layout)
    # Equiv. store_done_barrier. Barrier Store TMA -> TCGEN05 MMA
    acc_empty_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES, two_ctas=TWO_CTAS)
    # Equiv. mma_done_barrier. Barrier TCGEN05 MMA -> Store TMA
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES)
    for i in gl.static_range(ACC_STAGES):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=mma_barrier_count)

    clc_barriers = mbarrier.allocate_mbarrier(batch=ACC_STAGES)
    clc_consumed_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES, two_ctas=TWO_CTAS)
    for i in gl.static_range(ACC_STAGES):
        mbarrier.init(clc_barriers.index(i), count=1)
        # Every partition but itself arrives on the barrier
        mbarrier.init(clc_consumed_bars.index(i), count=N_PARTITIONS - 1)

    cga_layout: gl.constexpr = [[0]] * (gl.num_ctas().bit_length() - 1)
    clc_result_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 2],
                                                   gl.SwizzledSharedLayout(1, 1, 1, [0], cga_layout=cga_layout))

    if gl.num_ctas() > 1:
        mbarrier.sync_cluster_init()
    p = PartitionArgs(
        a_desc,
        b_desc,
        c_desc,
        a_bufs,
        b_bufs,
        load_empty_bars,
        load_ready_bars,
        acc_bufs,
        acc_empty_bars,
        acc_ready_bars,
        clc_result_buffers,
        clc_barriers,
        clc_consumed_bars,
        GRID_MINOR_DIM,
        GRID_TILE_WIDTH,
    )

    gl.warp_specialize([
        (matmul_epilogue_partition, (p, )),
        (matmul_load_partition, (p, )),
        (matmul_mma_partition, (p, )),
        (matmul_clc_partition, (p, )),
    ], [1, 1, 1], [24, 24, 24])


matmul_kernel = triton.autotune(
    configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
)(_matmul_kernel)


def matmul_with_config(
    a,
    b,
    out=None,
    *,
    block_size_m,
    block_size_n,
    block_size_k,
    grid_minor_dim,
    grid_tile_width,
    stages,
    acc_stages,
    cga_layout,
    epilogue_size_n,
):
    if block_size_n // get_split_dim(cga_layout, 1) > 256:
        raise ValueError(f"cga_layout={list(cga_layout)} only supports BLOCK_SIZE_N <= 256")
    M, K = a.shape
    K1, N = b.shape
    if K != K1:
        raise ValueError(f"incompatible shapes: {a.shape} and {b.shape}")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("matmul only supports fp16 inputs")

    tile_m = block_size_m * (2 if cga_layout else 1)
    if M % tile_m != 0 or N % block_size_n != 0 or K % block_size_k != 0:
        raise ValueError(f"Shape {(M, N, K)} incompatible with tile {(tile_m, block_size_n, block_size_k)}")

    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        if out.shape != (M, N):
            raise ValueError(f"Output has invalid shape {out.shape}, expected {(M, N)}")
        if out.device != a.device or out.dtype != a.dtype:
            raise ValueError("Output must match input device and dtype")
        c = out
    dummy_block = [1, 1]
    dummy_layout = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, dummy_block, dummy_layout)
    b_desc = TensorDescriptor.from_tensor(b, dummy_block, dummy_layout)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout)

    matmul_tma_set_block_size_hook({
        "a_desc": a_desc,
        "b_desc": b_desc,
        "c_desc": c_desc,
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GRID_MINOR_DIM": grid_minor_dim,
        "GRID_TILE_WIDTH": grid_tile_width,
        "STAGES": stages,
        "ACC_STAGES": acc_stages,
        "CGA_LAYOUT": cga_layout,
        "EPILOGUE_SIZE_N": epilogue_size_n,
    })

    def grid(meta):
        tile_m = meta["BLOCK_SIZE_M"] * (2 if len(meta["CGA_LAYOUT"]) > 0 else 1)
        block_n = meta["BLOCK_SIZE_N"]
        block_k = meta["BLOCK_SIZE_K"]
        if M % tile_m != 0 or N % block_n != 0 or K % block_k != 0:
            raise ValueError(f"Shape {(M, N, K)} incompatible with tile {(tile_m, block_n, block_k)}")
        num_tiles = triton.cdiv(M, tile_m) * triton.cdiv(N, block_n)
        return (num_tiles, )

    _matmul_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        block_size_m,
        block_size_n,
        block_size_k,
        grid_minor_dim,
        grid_tile_width,
        stages,
        acc_stages,
        cga_layout,
        epilogue_size_n,
        num_warps=4,
        num_ctas=2**len(cga_layout),
    )
    return c


def matmul(a, b):
    M, K = a.shape
    K1, N = b.shape
    if K != K1:
        raise ValueError(f"incompatible shapes: {a.shape} and {b.shape}")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("matmul only supports fp16 inputs")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    dummy_block = [1, 1]
    dummy_layout = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, dummy_block, dummy_layout)
    b_desc = TensorDescriptor.from_tensor(b, dummy_block, dummy_layout)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout)

    def grid(meta):
        tile_m = meta["BLOCK_SIZE_M"] * get_split_dim(meta["CGA_LAYOUT"], 0)
        block_n = meta["BLOCK_SIZE_N"]
        block_k = meta["BLOCK_SIZE_K"]
        if M % tile_m != 0 or N % block_n != 0 or K % block_k != 0:
            raise ValueError(f"Shape {(M, N, K)} incompatible with tile {(tile_m, block_n, block_k)}")
        num_tiles = triton.cdiv(M, tile_m) * triton.cdiv(N, block_n)
        return (num_tiles, )

    matmul_kernel[grid](a_desc, b_desc, c_desc, M, N, K)
    return c


########################################################
# Testing
########################################################


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("block_size_m", [64, 128])
@pytest.mark.parametrize("cga_layout", [(), ((1, 0), ), ((1, 0), (2, 0))])
@pytest.mark.parametrize(
    "grid_minor_dim,grid_tile_width,stages",
    [
        (1, 1, 2),
        (1, 4, 4),
        (0, 12, 6),
        (0, 8, 4),
    ],
)
def test_matmul_configs(cga_layout, block_size_m, grid_minor_dim, grid_tile_width, stages):
    M, N, K = 512, 512, 256
    torch.manual_seed(0)
    a = torch.rand((M, K), device=torch.device("cuda"), dtype=torch.float16)
    b = torch.rand((K, N), device=torch.device("cuda"), dtype=torch.float16)
    expected = torch.matmul(a, b)
    # We don't support epilogue splitting for 2CTA MMA until
    # https://github.com/triton-lang/triton/pull/7777 lands
    # To support epilogue splitting we need to be able to split within a CTA
    can_split_n = sum(b[1] != 0 for b in cga_layout) == 0
    block_size_n = 256
    epilogue_size_n = block_size_n if cga_layout and (block_size_m == 64 or not can_split_n) else 32
    try:
        actual = matmul_with_config(
            a,
            b,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            block_size_k=64,
            grid_minor_dim=grid_minor_dim,
            grid_tile_width=grid_tile_width,
            stages=stages,
            acc_stages=2,
            cga_layout=cga_layout,
            epilogue_size_n=epilogue_size_n,
        )
    except triton.runtime.errors.OutOfResources:
        pytest.skip("not enough shmem")
    torch.testing.assert_close(expected, actual, atol=1e-1, rtol=1e-2)


@pytest.mark.skip(reason="Too slow. Remove skip once we trim the config space")
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_matmul_autotuned_matches_torch():
    M, N, K = 512, 256, 256
    torch.manual_seed(0)
    a = torch.rand((M, K), device=torch.device("cuda"), dtype=torch.float16)
    b = torch.rand((K, N), device=torch.device("cuda"), dtype=torch.float16)
    expected = torch.matmul(a, b)
    actual = matmul(a, b)
    torch.testing.assert_close(expected, actual, atol=1e-1, rtol=1e-2)


########################################################
# Benchmarking
########################################################


def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ["tflop16/s", "time/ms"]
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


def print_benchmark_header():
    print("=" * 60)
    print("Gluon Matmul Benchmark")
    print("=" * 60)
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}, SMs: {props.multi_processor_count}")


def create_benchmark_tensors():
    M, N, K = 4096, 8192, 4096
    print(f"Matrix: M={M}, N={N}, K={K}")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c_triton = torch.empty((M, N), device="cuda", dtype=torch.float16)
    c_torch = torch.empty((M, N), device="cuda", dtype=torch.float16)
    expected = torch.matmul(a, b)
    return (M, N, K), a, b, c_triton, c_torch, expected


def get_benchmark_kernel_config():
    return {
        "tile_m": 128,
        "tile_n": 256,
        "tile_k": 64,
        "grid_minor_dim": 0,
        "grid_tile_width": 16,
        "stages": 6,
        "acc_stages": 2,
        "cga_layout": ((1, 0), ),
        "epilogue_tile_n": 32,
    }


def make_gluon_runner(a, b, c_triton, cfg, use_autotuned=False):
    if use_autotuned:

        def run_gluon():
            return matmul(a, b)

        return run_gluon

    def run_gluon():
        return matmul_with_config(
            a,
            b,
            out=c_triton,
            block_size_m=cfg["tile_m"],
            block_size_n=cfg["tile_n"],
            block_size_k=cfg["tile_k"],
            grid_minor_dim=cfg["grid_minor_dim"],
            grid_tile_width=cfg["grid_tile_width"],
            stages=cfg["stages"],
            acc_stages=cfg["acc_stages"],
            cga_layout=cfg["cga_layout"],
            epilogue_size_n=cfg["epilogue_tile_n"],
        )

    return run_gluon


def run_profile(shape, a, b, c_torch, run_gluon):
    import triton.profiler as proton

    M, N, K = shape

    proton.start("matmul", hook="triton")
    proton.deactivate(0)
    l2_cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    def bench_fn(label, reps, fn):
        print(f"Benchmarking {label}: ...", end="")
        proton.deactivate()
        for _ in range(5):
            fn()  # warmup
            triton.runtime.driver.active.clear_cache(l2_cache)
        try:
            for _ in range(reps):
                proton.deactivate()
                triton.runtime.driver.active.clear_cache(l2_cache)
                proton.activate()
                fn()
        finally:
            proton.deactivate()
        print(f"\rBenchmarking {label}: done")

    bytes_per_elem = a.element_size()
    scope_metrics = {
        "bytes": bytes_per_elem * (M * K + N * K + M * N),
        f"flops{bytes_per_elem * 8}": 2.0 * M * N * K,
    }

    def torch_profiled():
        with proton.scope(f"torch [M={M}, N={N}, K={K}]", scope_metrics):
            torch.matmul(a, b, out=c_torch)

    def gluon_profiled():
        with proton.scope(f"gluon [M={M}, N={N}, K={K}]", scope_metrics):
            run_gluon()

    bench_fn("torch", reps=100, fn=torch_profiled)
    bench_fn("gluon", reps=100, fn=gluon_profiled)

    proton.finalize()
    print("Proton profile written to `matmul.hatchet`")
    show_profile("matmul")


def benchmark(*, profile=True, use_autotuned=False):
    if not is_blackwell():
        raise RuntimeError("This benchmark requires a Blackwell CUDA GPU.")

    print_benchmark_header()
    shape, a, b, c_triton, c_torch, expected = create_benchmark_tensors()
    kernel_cfg = get_benchmark_kernel_config()

    runner_name = "matmul (autotuned)" if use_autotuned else "matmul_with_config"
    print(f"Gluon runner: {runner_name}")
    run_gluon = make_gluon_runner(a, b, c_triton, kernel_cfg, use_autotuned=use_autotuned)
    actual = run_gluon()
    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-2)
    if use_autotuned:
        print(f"Autotuned best config: {matmul_kernel.best_config}")

    if not profile:
        print("Skipping profiling (--no-profile).")
        return

    run_profile(shape, a, b, c_torch, run_gluon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gluon matmul benchmark")
    """
    To enable NCU profiling, run the script with the following example command:
    ```
    ncu --target-processes all \
    --set full \
    --import-source yes \
    --kernel-name-base function \
    --kernel-name 'regex:.*_matmul_kernel.*' \
    --launch-count 1 \
    -o ncu_triton_matmul \
    python 02-matmul-multicta.py --no-profile
    ```
    """
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip Proton profiling and exit after validation.",
    )
    parser.add_argument(
        "--use-autotuned",
        action="store_true",
        help="Use autotuned matmul() instead of matmul_with_config() for the Gluon runner.",
    )
    args = parser.parse_args()
    benchmark(profile=not args.no_profile, use_autotuned=args.use_autotuned)
