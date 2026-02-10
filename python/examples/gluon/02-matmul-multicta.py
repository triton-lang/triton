import importlib
import itertools
import statistics

import pytest
import torch

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_commit,
    tcgen05_mma,
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


def _as_gl_dtype(torch_dtype):
    if torch_dtype == torch.float16:
        return gl.float16
    if torch_dtype == torch.bfloat16:
        return gl.bfloat16
    if torch_dtype == torch.float32:
        return gl.float32
    raise ValueError(f"Unsupported dtype for Gluon layout: {torch_dtype}")


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
                "TWO_CTAS": two_cta,
                "EPILOGUE_SIZE_N": 32,
            },
            num_warps=4,
            num_ctas=2 if two_cta else 1,
            pre_hook=pre_hook,
        )
        for BM in (128, )
        for BN in (128, 256)
        for BK in (64, )
        for minor_dim in (0, 1)
        for grid_tile_width in (1, 4, 8, 12, 16)
        for stages in (2, 4, 6)
        for two_cta in (False, True)
        if not (two_cta and BN > 128)
    ]


def matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    epilogue_size_n = nargs["EPILOGUE_SIZE_N"]
    two_ctas = bool(nargs["TWO_CTAS"])

    tile_m = block_m * (2 if two_ctas else 1)
    # Keeping this here because pallas does this, but in reality
    # we should just multiply block_m by 2 if we want to compute
    # the same number of elements per output tile
    tile_n = block_n * (2 if two_ctas else 1)
    nargs["a_desc"].block_shape = [tile_m, block_k]
    nargs["b_desc"].block_shape = [block_k, tile_n]
    nargs["c_desc"].block_shape = [tile_m, epilogue_size_n]

    if two_ctas:
        cga_layouts = [[[1, 0]], [[0, 1]], [[1, 0]]]
        for desc, cga_layout in zip(("a_desc", "b_desc", "c_desc"), cga_layouts):
            nargs[desc].layout = gl.NVMMASharedLayout.get_default_for(
                nargs[desc].block_shape,
                _as_gl_dtype(nargs[desc].base.dtype),
                cga_layout=cga_layout,
            )
    else:
        for desc in ("a_desc", "b_desc", "c_desc"):
            nargs[desc].layout = gl.NVMMASharedLayout.get_default_for(
                nargs[desc].block_shape,
                _as_gl_dtype(nargs[desc].base.dtype),
            )


# From Pallas.
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
class PersistentTileScheduler:
    pid_start: gl.tensor
    num_pid: gl.tensor
    num_pid_m: gl.tensor
    num_pid_n: gl.tensor
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, pid_start, num_pid, num_pid_m, num_pid_n, TILE_M, TILE_N, MINOR_DIM, GRID_TILE_WIDTH):
        self.pid_start = pid_start
        self.num_pid = num_pid
        self.num_pid_m = num_pid_m
        self.num_pid_n = num_pid_n
        self.TILE_M = gl.constexpr(TILE_M)
        self.TILE_N = gl.constexpr(TILE_N)
        self.MINOR_DIM = gl.constexpr(MINOR_DIM)
        self.GRID_TILE_WIDTH = gl.constexpr(GRID_TILE_WIDTH)

    @gluon.jit
    def initialize(M, N, TILE_M: gl.constexpr, TILE_N: gl.constexpr, MINOR_DIM: gl.constexpr,
                   GRID_TILE_WIDTH: gl.constexpr):
        pid_start = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(M, TILE_M)
        num_pid_n = gl.cdiv(N, TILE_N)
        num_pid = num_pid_m * num_pid_n
        return PersistentTileScheduler(pid_start, num_pid, num_pid_m, num_pid_n, TILE_M, TILE_N, MINOR_DIM,
                                       GRID_TILE_WIDTH)

    @gluon.jit
    def get_num_tiles(self):
        return gl.cdiv(self.num_pid - self.pid_start, gl.num_programs(axis=0))

    @gluon.jit
    def get_offsets(self, idx):
        tile_id = self.pid_start + idx * gl.num_programs(axis=0)
        pid_m, pid_n = _planar_snake(tile_id, self.num_pid_m, self.num_pid_n, self.MINOR_DIM, self.GRID_TILE_WIDTH)
        return pid_m * self.TILE_M, pid_n * self.TILE_N


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
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr
    TWO_CTAS: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_bufs,
                 acc_empty_bars, acc_ready_bars, MINOR_DIM, GRID_TILE_WIDTH, TWO_CTAS):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_bufs = a_bufs
        self.b_bufs = b_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.MINOR_DIM = gl.constexpr(MINOR_DIM)
        self.GRID_TILE_WIDTH = gl.constexpr(GRID_TILE_WIDTH)
        self.TWO_CTAS = gl.constexpr(TWO_CTAS)

    @gluon.jit
    def get_scheduler(self):
        return PersistentTileScheduler.initialize(
            self.c_desc.shape[0],
            self.c_desc.shape[1],
            self.a_desc.block_shape[0],
            self.b_desc.block_shape[1],
            self.MINOR_DIM,
            self.GRID_TILE_WIDTH,
        )


@aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)

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


@gluon.jit
def matmul_load_partition(p):
    if p.TWO_CTAS:
        mbarrier.sync_cluster_init()

    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1]
    K = p.a_desc.shape[1]

    concurrent_loads: gl.constexpr = p.load_ready_bars.shape[0]
    state = Counter.create(1, concurrent_loads)
    scheduler = p.get_scheduler()

    for idx in range(scheduler.get_num_tiles()):
        off_m, off_n = scheduler.get_offsets(idx)
        for k in range(0, K, BLOCK_K):
            pred = (idx > 0) or (k >= BLOCK_K * concurrent_loads)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase, pred=pred)
            bar = p.load_ready_bars.index(state.index)
            mbarrier.expect(bar, p.a_desc.nbytes_per_cta + p.b_desc.nbytes_per_cta)
            tma.async_copy_global_to_shared(p.a_desc, [off_m, k], bar, p.a_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()


@gluon.jit
def matmul_mma_partition(p):
    if p.TWO_CTAS:
        mbarrier.sync_cluster_init()

    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1]
    K = p.a_desc.shape[1]

    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.get_scheduler()

    for i in range(scheduler.get_num_tiles()):
        acc_buf = p.acc_bufs.index(acc_state.index)
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase, pred=(i > 1))
        use_acc = False
        for k in range(0, K, BLOCK_K):
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
            tcgen05_mma(p.a_bufs.index(load_state.index), p.b_bufs.index(load_state.index), acc_buf, use_acc=use_acc,
                        mbarriers=[p.load_empty_bars.index(load_state.index)])
            load_state = load_state.next()
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def matmul_epilogue_partition(p):
    if p.TWO_CTAS:
        mbarrier.sync_cluster_init()

    TILE_M: gl.constexpr = p.a_desc.block_shape[0]
    TILE_N: gl.constexpr = p.b_desc.block_shape[1]
    SPLIT_TILE_N: gl.constexpr = p.c_desc.block_shape[1]
    SUBTILE_FACTOR: gl.constexpr = TILE_N // SPLIT_TILE_N
    dtype: gl.constexpr = p.c_desc.dtype

    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    acc_smems = gl.allocate_shared_memory(dtype, [2, TILE_M, SPLIT_TILE_N], p.c_desc.layout)
    sub_acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = p.get_scheduler()

    for idx in range(scheduler.get_num_tiles()):
        off_m, off_n = scheduler.get_offsets(idx)

        # TODO: we are not emitting read=True
        tma.store_wait(pendings=0)
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)

        for i in gl.static_range(SUBTILE_FACTOR):
            acc_sub = acc_buf.slice(SPLIT_TILE_N * i, SPLIT_TILE_N)
            acc_smem = acc_smems.index(sub_acc_state.index)
            acc_smem.store(
                acc_sub.load(
                    get_tmem_reg_layout(
                        gl.float32,
                        (TILE_M, SPLIT_TILE_N),
                        acc_sub.type.layout,
                        gl.num_warps(),
                        cga_layout=p.c_desc.layout.cga_layout,
                    )).to(dtype))
            fence_async_shared()
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + SPLIT_TILE_N * i], acc_smem)
            # TODO: we are not emitting read=True
            tma.store_wait(pendings=1)
            sub_acc_state = sub_acc_state.next()
        # Signal that the accumulator slot can be reused only after all stores are done.
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def matmul_sync_partition(p):
    if p.TWO_CTAS:
        mbarrier.sync_cluster_init()


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
    TWO_CTAS: gl.constexpr,
    EPILOGUE_SIZE_N: gl.constexpr,
):
    gl.static_assert(STAGES >= 2, "Expected at least 2 stages")
    gl.static_assert(gl.num_ctas() == (2 if TWO_CTAS else 1), "num_ctas mismatch with TWO_CTAS")
    BLOCK_M: gl.constexpr = a_desc.block_shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_shape[1]

    dtype: gl.constexpr = a_desc.dtype
    a_bufs = gl.allocate_shared_memory(dtype, [STAGES] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [STAGES] + b_desc.block_shape, b_desc.layout)

    # Equiv. consumed_barrier. Barrier TCGEN05 MMA -> Load TMA
    load_empty_bars = mbarrier.allocate_mbarrier(batch=STAGES)
    # Equiv. ab_tma_barrier. Barrier Load TMA -> TCGEN05 MMA
    load_ready_bars = mbarrier.allocate_mbarrier(batch=STAGES, two_ctas=TWO_CTAS)
    for i in gl.static_range(STAGES):
        # For multicast we could use tcgen05_mma_barrier_count
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout(
        [BLOCK_SIZE_M, BLOCK_N],
        col_stride=1,
        cta_split_num=(2, 1) if TWO_CTAS else None,
        two_ctas=TWO_CTAS,
    )
    acc_bufs = allocate_tensor_memory(gl.float32, [2, BLOCK_M, BLOCK_N], tmem_layout)
    # Equiv. store_done_barrier. Barrier Store TMA -> TCGEN05 MMA
    acc_empty_bars = mbarrier.allocate_mbarrier(batch=2, two_ctas=TWO_CTAS)
    # Equiv. mma_done_barrier. Barrier TCGEN05 MMA -> Store TMA
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=2)
    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        # For multicast we could use tcgen05_mma_barrier_count
        mbarrier.init(acc_ready_bars.index(i), count=1)

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
        GRID_MINOR_DIM,
        GRID_TILE_WIDTH,
        TWO_CTAS,
    )

    gl.warp_specialize([
        (matmul_epilogue_partition, (p, )),
        (matmul_load_partition, (p, )),
        (matmul_mma_partition, (p, )),
        (matmul_sync_partition, (p, )),
    ], [1, 1, 2], [24, 24, 24])


matmul_kernel = triton.autotune(
    configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
)(_matmul_kernel)


def matmul_with_config(
    a,
    b,
    out=None,
    *,
    block_size_m=128,
    block_size_n=128,
    block_size_k=64,
    grid_minor_dim=0,
    grid_tile_width=1,
    stages=4,
    two_ctas=False,
    epilogue_size_n=32,
):
    if two_ctas and block_size_n > 128:
        raise ValueError("two_ctas only supports BLOCK_SIZE_N <= 128")
    M, K = a.shape
    K1, N = b.shape
    if K != K1:
        raise ValueError(f"incompatible shapes: {a.shape} and {b.shape}")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("matmul only supports fp16 inputs")

    tile_m = block_size_m * (2 if two_ctas else 1)
    tile_n = block_size_n * (2 if two_ctas else 1)
    if M % tile_m != 0 or N % tile_n != 0 or K % block_size_k != 0:
        raise ValueError(f"Shape {(M, N, K)} incompatible with tile {(tile_m, tile_n, block_size_k)}")

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
        "TWO_CTAS": two_ctas,
        "EPILOGUE_SIZE_N": epilogue_size_n,
    })

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(meta):
        tile_m = meta["BLOCK_SIZE_M"] * (2 if bool(meta["TWO_CTAS"]) else 1)
        tile_n = meta["BLOCK_SIZE_N"] * (2 if bool(meta["TWO_CTAS"]) else 1)
        block_k = meta["BLOCK_SIZE_K"]
        if M % tile_m != 0 or N % tile_n != 0 or K % block_k != 0:
            raise ValueError(f"Shape {(M, N, K)} incompatible with tile {(tile_m, tile_n, block_k)}")
        num_tiles = triton.cdiv(M, tile_m) * triton.cdiv(N, tile_n)
        return (min(num_sms, num_tiles), )

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
        two_ctas,
        epilogue_size_n,
        num_warps=4,
        num_ctas=2 if two_ctas else 1,
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

    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    def grid(meta):
        tile_m = meta["BLOCK_SIZE_M"] * (2 if bool(meta["TWO_CTAS"]) else 1)
        tile_n = meta["BLOCK_SIZE_N"] * (2 if bool(meta["TWO_CTAS"]) else 1)
        block_k = meta["BLOCK_SIZE_K"]
        if M % tile_m != 0 or N % tile_n != 0 or K % block_k != 0:
            raise ValueError(f"Shape {(M, N, K)} incompatible with tile {(tile_m, tile_n, block_k)}")
        num_tiles = triton.cdiv(M, tile_m) * triton.cdiv(N, tile_n)
        return (min(num_sms, num_tiles), )

    matmul_kernel[grid](a_desc, b_desc, c_desc, M, N, K)
    return c


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "grid_minor_dim,grid_tile_width,stages,block_size_n",
    [
        (0, 1, 2, 128),
        (1, 8, 4, 128),
        (0, 16, 2, 256),
    ],
)
def test_matmul_single_cta_configs(grid_minor_dim, grid_tile_width, stages, block_size_n):
    M, N, K = 512, 512, 256
    torch.manual_seed(0)
    a = torch.rand((M, K), device=torch.device("cuda"), dtype=torch.float16)
    b = torch.rand((K, N), device=torch.device("cuda"), dtype=torch.float16)
    expected = torch.matmul(a, b)
    actual = matmul_with_config(
        a,
        b,
        block_size_m=128,
        block_size_n=block_size_n,
        block_size_k=64,
        grid_minor_dim=grid_minor_dim,
        grid_tile_width=grid_tile_width,
        stages=stages,
        two_ctas=False,
        epilogue_size_n=32,
    )
    torch.testing.assert_close(expected, actual, atol=1e-1, rtol=1e-2)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "grid_minor_dim,grid_tile_width,stages",
    [
        (1, 1, 2),
        (1, 4, 4),
        (0, 12, 6),
        (0, 8, 4),
    ],
)
def test_matmul_two_cta_configs(grid_minor_dim, grid_tile_width, stages):
    M, N, K = 512, 512, 256
    torch.manual_seed(0)
    a = torch.rand((M, K), device=torch.device("cuda"), dtype=torch.float16)
    b = torch.rand((K, N), device=torch.device("cuda"), dtype=torch.float16)
    expected = torch.matmul(a, b)
    actual = matmul_with_config(
        a,
        b,
        block_size_m=128,
        block_size_n=128,
        block_size_k=64,
        grid_minor_dim=grid_minor_dim,
        grid_tile_width=grid_tile_width,
        stages=stages,
        two_ctas=True,
        epilogue_size_n=32,
    )
    torch.testing.assert_close(expected, actual, atol=1e-1, rtol=1e-2)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "M,N,K",
    [
        (256, 256, 128),
        (512, 256, 256),
    ],
)
def test_matmul_autotuned_matches_torch(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device=torch.device("cuda"), dtype=torch.float16)
    b = torch.rand((K, N), device=torch.device("cuda"), dtype=torch.float16)
    expected = torch.matmul(a, b)
    actual = matmul(a, b)
    torch.testing.assert_close(expected, actual, atol=1e-1, rtol=1e-2)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_matmul_with_config_rejects_invalid_collective_config():
    M, N, K = 512, 512, 256
    a = torch.rand((M, K), device=torch.device("cuda"), dtype=torch.float16)
    b = torch.rand((K, N), device=torch.device("cuda"), dtype=torch.float16)
    with pytest.raises(ValueError, match="BLOCK_SIZE_N <= 128"):
        _ = matmul_with_config(a, b, two_ctas=True, block_size_n=256)


def _tflops(ms, M, N, K):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def _find_best_config(label, tuning_configs, run_config, to_torch, expected, M, N, K, optimal_time_us, bench_fn,
                      report_each=False):
    best_cfg = None
    best_ms = None
    best_tflops = -float("inf")
    best_util = -float("inf")

    if report_each:
        print(f"\n{label} search:")

    for cfg in tuning_configs:
        tile_m, tile_n, tile_k, grid_minor_dim, grid_tile_width, max_concurrent_steps, collective, epilogue_tile_n = cfg
        if collective and tile_n > 128:
            continue
        try:
            ms = bench_fn(run_config, cfg)
        except Exception:
            continue

        runtime_us = ms * 1e3
        util = optimal_time_us / runtime_us * 100
        tflops = _tflops(ms, M, N, K)
        if util > best_util:
            out_torch = to_torch(run_config(cfg))
            torch.testing.assert_close(out_torch, expected, atol=1e-1, rtol=1e-2)
            best_cfg = cfg
            best_ms = ms
            best_tflops = tflops
            best_util = util

        if report_each:
            eff_tile_m = tile_m * (2 if collective else 1)
            eff_tile_n = tile_n * (2 if collective else 1)
            print(f"tile_m={eff_tile_m} tile_n={eff_tile_n} tile_k={tile_k} "
                  f"max_concurrent_steps={max_concurrent_steps} "
                  f"grid_minor_dim={grid_minor_dim} grid_tile_width={grid_tile_width} "
                  f"epilogue_tile_n={epilogue_tile_n} collective={collective} : "
                  f"{runtime_us:<7.1f}us = {util:4.1f}% TC utilization ({tflops:7.2f} TFLOPS)")

    assert best_cfg is not None
    return best_cfg, best_ms, best_tflops, best_util


def benchmark():
    if not is_blackwell():
        raise RuntimeError("This benchmark requires a Blackwell CUDA GPU.")

    print("=" * 60)
    print("Gluon Matmul Benchmark")
    print("=" * 60)
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}, SMs: {props.multi_processor_count}")

    M, N, K = 4096, 8192, 4096
    print(f"Matrix: M={M}, N={N}, K={K}")
    peak_flops = 2.25e15  # f16 TensorCore peak = 2250 TFLOPS
    optimal_time_us = (2 * M * N * K) / peak_flops * 1e6

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c_triton = torch.empty((M, N), device="cuda", dtype=torch.float16)
    c_torch = torch.empty((M, N), device="cuda", dtype=torch.float16)
    expected = torch.matmul(a, b)

    tuning_configs = list(
        itertools.product((128, ),  # tile_m
                          (128, 256),  # tile_n
                          (64, ),  # tile_k
                          (0, 1),  # grid_minor_dim
                          (1, 4, 8, 12, 16),  # grid_tile_width
                          (2, 4, 6),  # max_concurrent_steps
                          (False, True),  # collective
                          (32, ),  # epilogue_tile_n
                          ))

    def run_triton(cfg):
        tile_m, tile_n, tile_k, grid_minor_dim, grid_tile_width, max_concurrent_steps, collective, epilogue_tile_n = cfg
        return matmul_with_config(
            a,
            b,
            out=c_triton,
            block_size_m=tile_m,
            block_size_n=tile_n,
            block_size_k=tile_k,
            grid_minor_dim=grid_minor_dim,
            grid_tile_width=grid_tile_width,
            stages=max_concurrent_steps,
            two_ctas=collective,
            epilogue_size_n=epilogue_tile_n,
        )

    best_triton_cfg, best_triton_ms, best_triton_tflops, best_triton_util = _find_best_config(
        "Triton",
        tuning_configs,
        run_triton,
        to_torch=lambda out: out,
        expected=expected,
        M=M,
        N=N,
        K=K,
        optimal_time_us=optimal_time_us,
        bench_fn=lambda run_config, cfg: triton.testing.do_bench_cudagraph(lambda: run_config(cfg)),
    )

    torch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b, out=c_torch))
    torch_tflops = _tflops(torch_ms, M, N, K)
    torch_util = optimal_time_us / (torch_ms * 1e3) * 100

    best_pallas_cfg = None
    best_pallas_ms = None
    best_pallas_tflops = None
    best_pallas_util = None
    pallas_error = None
    try:
        import functools
        import jax
        import jax.numpy as jnp
        from jax.experimental.mosaic.gpu import profiler
        pallas_mod = importlib.import_module("jax.experimental.pallas.ops.gpu.blackwell_matmul_mgpu")
    except Exception as exc:
        pallas_error = exc

    pallas_device = jax.devices("gpu")[0]
    a_jax = jax.device_put(jnp.asarray(a.detach().cpu().numpy()), pallas_device)
    b_jax = jax.device_put(jnp.asarray(b.detach().cpu().numpy()), pallas_device)

    def run_pallas(cfg):
        tile_m, tile_n, tile_k, grid_minor_dim, grid_tile_width, max_concurrent_steps, collective, epilogue_tile_n = cfg
        pallas_cfg = pallas_mod.TuningConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            max_concurrent_steps=max_concurrent_steps,
            collective=collective,
            epilogue_tile_n=epilogue_tile_n,
            grid_minor_dim=pallas_mod.MatmulDimension(grid_minor_dim),
            grid_tile_width=grid_tile_width,
        )
        return pallas_mod.matmul_kernel(a_jax, b_jax, config=pallas_cfg)

    def pallas_bench(run_config, cfg):
        _, runtimes_ms = profiler.measure(functools.partial(run_config, cfg), iterations=10)()
        return statistics.median(runtimes_ms)

    best_pallas_cfg, best_pallas_ms, best_pallas_tflops, best_pallas_util = _find_best_config(
        "Pallas",
        tuning_configs,
        run_pallas,
        to_torch=lambda out: torch.from_numpy(jax.device_get(out).copy()).to(device=expected.device, dtype=expected.
                                                                             dtype),
        expected=expected,
        M=M,
        N=N,
        K=K,
        optimal_time_us=optimal_time_us,
        bench_fn=pallas_bench,
        report_each=True,
    )

    print("\nSummary:")
    print("Backend                     TFLOPS    util(%)     ms")
    if best_triton_ms is not None:
        print(f"Triton (best)          {best_triton_tflops:8.2f}   {best_triton_util:7.2f} {best_triton_ms:7.3f}")
        print(f"  config: {best_triton_cfg}")
    print(f"PyTorch/cuBLAS         {torch_tflops:8.2f}   {torch_util:7.2f} {torch_ms:7.3f}")
    if best_pallas_ms is not None:
        print(f"Pallas (best)          {best_pallas_tflops:8.2f}   {best_pallas_util:7.2f} {best_pallas_ms:7.3f}")
        print(f"  config: {best_pallas_cfg}")
    if pallas_error is not None:
        print(f"Skipping Pallas benchmark: {pallas_error}")


if __name__ == "__main__":
    benchmark()
