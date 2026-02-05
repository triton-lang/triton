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
    tile_n = block_n

    nargs["a_desc"].block_shape = [tile_m, block_k]
    nargs["b_desc"].block_shape = [block_k, tile_n]
    nargs["c_desc"].block_shape = [tile_m, epilogue_size_n]

    if two_ctas:
        cga_layouts = [[1, 0], [0, 1], [1, 0]]
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

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_bufs,
                 acc_empty_bars, acc_ready_bars, MINOR_DIM, GRID_TILE_WIDTH):
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
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1]
    K = p.a_desc.shape[1]

    state = Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = p.get_scheduler()

    for idx in range(scheduler.get_num_tiles()):
        off_m, off_n = scheduler.get_offsets(idx)
        for k in range(0, K, BLOCK_K):
            bar = p.load_ready_bars.index(state.index)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            mbarrier.expect(bar, p.a_desc.block_type.nbytes + p.b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(p.a_desc, [off_m, k], bar, p.a_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()


@gluon.jit
def matmul_mma_partition(p):
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1]
    K = p.a_desc.shape[1]

    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.get_scheduler()

    for _ in range(scheduler.get_num_tiles()):
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for _ in range(0, K, BLOCK_K):
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
            tcgen05_mma(p.a_bufs.index(load_state.index), p.b_bufs.index(load_state.index), acc_buf, use_acc=use_acc,
                        mbarriers=[p.load_empty_bars.index(load_state.index)])
            load_state = load_state.next()
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def _split_n(x, N: gl.constexpr):
    gl.static_assert(N > 0 and (N & (N - 1)) == 0, "n must be positive and a power of two")
    split_count: gl.constexpr = N.bit_length() - 1
    xs = (x, )
    for _ in gl.static_range(split_count):
        next_xs = ()
        for j in gl.static_range(len(xs)):
            x = xs[j]
            next_xs += x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()
        xs = next_xs
    return xs


@gluon.jit
def matmul_epilogue_partition(p):
    TILE_M: gl.constexpr = p.a_desc.block_shape[0]
    TILE_N: gl.constexpr = p.b_desc.block_shape[1]
    SPLIT_TILE_N: gl.constexpr = p.c_desc.block_shape[1]
    SUBTILE_FACTOR: gl.constexpr = TILE_N // SPLIT_TILE_N
    dtype: gl.constexpr = p.c_desc.dtype

    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    acc_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32,
        (TILE_M, TILE_N),
        p.acc_bufs.type.layout,
        gl.num_warps(),
        cga_layout=p.c_desc.layout.cga_layout,
    )
    acc_smem = gl.allocate_shared_memory(dtype, [TILE_M, SPLIT_TILE_N], p.c_desc.layout)
    scheduler = p.get_scheduler()

    for idx in range(scheduler.get_num_tiles()):
        off_m, off_n = scheduler.get_offsets(idx)

        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index).load(acc_layout)
        acc_state = acc_state.next()

        accs = _split_n(acc, SUBTILE_FACTOR)
        for i in gl.static_range(SUBTILE_FACTOR):
            tma.store_wait(pendings=0)
            acc_smem.store(accs[i].to(dtype))
            if i == 0:
                mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
            # TODO Do we need a 2CTA fence here??
            fence_async_shared()
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + SPLIT_TILE_N * i], acc_smem)
    tma.store_wait(pendings=0)


@triton.autotune(
    configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
)
@gluon.jit
def matmul_kernel(
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

    load_empty_bars = mbarrier.allocate_mbarrier(batch=STAGES, two_ctas=TWO_CTAS)
    load_ready_bars = mbarrier.allocate_mbarrier(batch=STAGES, two_ctas=TWO_CTAS)
    for i in gl.static_range(STAGES):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout(
        [BLOCK_SIZE_M, BLOCK_SIZE_N],
        col_stride=1,
        cta_split_num=(2, 1) if TWO_CTAS else None,
        two_ctas=TWO_CTAS,
    )
    acc_bufs = allocate_tensor_memory(gl.float32, [2, BLOCK_M, BLOCK_N], tmem_layout)
    acc_empty_bars = mbarrier.allocate_mbarrier(batch=2, two_ctas=TWO_CTAS)
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=2, two_ctas=TWO_CTAS)
    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    if TWO_CTAS:
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
        GRID_MINOR_DIM,
        GRID_TILE_WIDTH,
    )

    gl.warp_specialize([
        (matmul_epilogue_partition, (p, )),
        (matmul_load_partition, (p, )),
        (matmul_mma_partition, (p, )),
    ], [1, 1], [24, 24])


def matmul(a, b):
    M, K = a.shape
    K1, N = b.shape
    if K != K1:
        raise ValueError(f"incompatible shapes: {a.shape} and {b.shape}")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("matmul only supports fp16 inputs")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

    dummy_block = [1, 1]
    dummy_layout = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, dummy_block, dummy_layout)
    b_desc = TensorDescriptor.from_tensor(b, dummy_block, dummy_layout)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout)

    def grid(meta):
        block_m = meta["BLOCK_SIZE_M"]
        block_n = meta["BLOCK_SIZE_N"]
        block_k = meta["BLOCK_SIZE_K"]
        two_ctas = bool(meta["TWO_CTAS"])
        tile_m = block_m * (2 if two_ctas else 1)
        tile_n = block_n
        if M % tile_m != 0 or N % tile_n != 0 or K % block_k != 0:
            raise ValueError(f"Shape {(M, N, K)} incompatible with tile {(tile_m, tile_n, block_k)}")
        num_tiles = triton.cdiv(M, tile_m) * triton.cdiv(N, tile_n)
        return (min(num_sms, num_tiles), )

    matmul_kernel[grid](a_desc, b_desc, c_desc, M, N, K)
    return c


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "M,N,K",
    [
        (256, 256, 128),
        (512, 256, 256),
    ],
)
def test_matmul_matches_torch(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device=torch.device("cuda"), dtype=torch.float16)
    b = torch.rand((K, N), device=torch.device("cuda"), dtype=torch.float16)
    expected = torch.matmul(a, b)
    actual = matmul(a, b)
    torch.testing.assert_close(expected, actual, atol=1e-1, rtol=1e-2)


def main():
    problem_it = [(4096, 8192, 4096)]
    for M, N, K in problem_it:
        print(f"==== {M=} {N=} {K=} ====")
        a = torch.rand((M, K), device=torch.device("cuda"), dtype=torch.float16)
        b = torch.rand((K, N), device=torch.device("cuda"), dtype=torch.float16)
        expected = torch.matmul(a, b)
        actual = matmul(a, b)
        torch.testing.assert_close(expected, actual, atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    main()
