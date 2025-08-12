"""
Warp Specialization
===================

This tutorial covers warp specialization. In typical GPU kernels, all the warps
in the kernel are performing parallel slices of the same task. Warp
specialization, however, is a technique where different warps in the kernel are
doing completely different tasks.

With warp specialization, we can overlap execution of independent parts of the
kernel by placing the work in different warps. This minimizes the critical path
in each warp, and we rely on the warp scheduler to dynamically schedule the
warps. We can also overlap non-async operations that exercise different parts of
the hardware without relying on precise SASS-level instruction interleaving.

However, warp specialization comes at the cost of additional synchronization
overhead, potentially higher shared memory usage for communicating data, and
higher overall register pressure.

Warp specialization in Gluon is only supported on Hopper and newer GPUs.
"""

import pytest
import torch
import triton
import importlib
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    tensor_memory_descriptor,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None

# Re-use utilities from the previous tutorial.
t3 = importlib.import_module("03-async-copy")
t4 = importlib.import_module("04-tma")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

# %%
# Let's revisit our elementwise add kernel and implement a warp-specialized
# version. In a warp-specialized kernel, groups of warps that perform a specific
# task are called "partitions", and each can have a different number of warps
# and registers.
#
# First, we need to decide what the partitions will be and how many registers
# they will get. One of the benefits of warp specialization is that partitions
# that only use scalar values require only 1 warp and often very few registers.
# For example, we can have one partition that just issues async TMA loads and
# one partition that just issues TMA stores, each with 1 warp and 24 registers,
# the minimum number of registers we can assign to a warp.
#
# Then we have one compute partition, with either 4 or 8 warps, which performs
# the vector addition. Estimating the right register allocation is difficult,
# and often involves trial and error, profiling, and autotuning. We will need to
# use mbarriers to signal between the partitions using producer-consumer pairs.
#
# To write a warp-specialized kernel, we need to write a separate function for
# each partition. One of the partitions must be chosen as the "default"
# partition and it always has the same number of warps as `num_warps` passed to
# the kernel. The other partitions, i.e. the "worker" partitions, can have
# different numbers of warps. The signature of the worker partition functions
# must all be the same. Only the default partition can accept tensor arguments.
#
# To quickly sketch out the partitions: load partition will fetch inputs to smem
# and signal the compute partition. The compute partition will consume the
# operands and send them to the store partition over smem.
#
# Recall that we need fence_async_shared to synchronize the async and generic
# proxies. This also applies if the buffer accesses are initiated in different
# partitions, even when they are sequenced by mbarrier.arrive:
#
# ```python
# smem.store(value)  # in partition A
# fence_async_shared()
# mbarrier.arrive(bar, count=1)
#
# mbarrier.wait(bar, phase=0)  # in partition B
# tma.async_copy_shared_to_global(desc, [0, 0], smem)
# ```
#
# A fence is needed somewhere between the shared memory store and the TMA store.
#
# ```python
# value = smem.load()
# mbarrier.arrive(bar, count=1)
#
# mbarrier.wait(bar, phase=0)
# fence_async_shared()
# tma.async_copy_global_to_shared(desc, [0, 0], bar, smem)
# ```
#
# A fence is needed somewhere between the shared memory load and the TMA load.


@gluon.jit
def load_partition(descs, barriers, buffers, xoff, numel, YBLOCK: gl.constexpr):
    # Unpack the arguments.
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    xnumel, ynumel = numel

    num_buffers: gl.constexpr = a_bufs.type.shape[0]

    # All the partitions need to have the same number of inner loop iterations.
    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        a_buf = a_bufs.index(index)
        b_buf = b_bufs.index(index)
        load_empty_bar = load_empty_bars.index(index)
        load_ready_bar = load_ready_bars.index(index)

        # Wait for the current buffers to be empty. Recall that mbarriers are
        # initialized to phase 1 complete, so we wait starting with phase 1 to
        # allow the producer to begin filling the pipeline.
        mbarrier.wait(load_empty_bar, phase ^ 1)

        # Okay, a_buf and b_buf are empty. Issue the TMA loads, and have them
        # signal the operand buffers as ready when they complete.
        yoff = i * YBLOCK
        mbarrier.expect(load_ready_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [xoff, yoff], load_ready_bar, a_buf)
        tma.async_copy_global_to_shared(b_desc, [xoff, yoff], load_ready_bar, b_buf)


@gluon.jit
def store_partition(descs, barriers, buffers, xoff, numel, YBLOCK: gl.constexpr):
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    xnumel, ynumel = numel

    # This partition consumes the addition result, passed over smem, and stores
    # them to global memory.
    num_buffers: gl.constexpr = c_bufs.type.shape[0]
    # We will keep `num_buffers-1` stores in flight by software pipelining.
    outstanding_stores: gl.constexpr = num_buffers - 1

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        c_buf = c_bufs.index(index)
        c_ready_bar = c_ready_bars.index(index)

        # Wait for the compute partition to produce c.
        mbarrier.wait(c_ready_bar, phase)
        yoff = i * YBLOCK
        tma.async_copy_shared_to_global(c_desc, [xoff, yoff], c_buf)

        tma.store_wait(outstanding_stores)
        c_empty_bar = c_empty_bars.index((i - outstanding_stores) % num_buffers)
        # Signal the compute partition that the buffer `outstanding_stores`
        # iterations ago is consumed, predicated on there having been at least
        # that many outstanding stores.
        mbarrier.arrive(c_empty_bar, count=1, pred=i >= outstanding_stores)

    # Since we waited for the last value of c, all the other partitions have
    # exited by now. We just need to wait the stores to complete.
    tma.store_wait(0)


# The default partition can have a different signature than the worker partition
# functions.
@gluon.jit
def compute_partition(barriers, buffers, ynumel, YBLOCK: gl.constexpr, layout: gl.constexpr):
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers

    num_load_buffers: gl.constexpr = a_bufs.type.shape[0]
    num_store_buffers: gl.constexpr = c_bufs.type.shape[0]

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        load_index = i % num_load_buffers
        load_phase = i // num_load_buffers & 1
        a_buf = a_bufs.index(load_index)
        b_buf = b_bufs.index(load_index)
        load_ready_bar = load_ready_bars.index(load_index)
        load_empty_bar = load_empty_bars.index(load_index)

        # Wait for the operands then consume them.
        mbarrier.wait(load_ready_bar, load_phase)
        a_val = a_buf.load(layout)
        b_val = b_buf.load(layout)
        # Fence before signalling the load partitions so the TMA load is
        # ordered with the shared load.
        fence_async_shared()
        mbarrier.arrive(load_empty_bar, count=1)

        c_val = a_val + b_val

        store_idx = i % num_store_buffers
        store_phase = i // num_store_buffers & 1
        c_buf = c_bufs.index(store_idx)
        c_empty_bar = c_empty_bars.index(store_idx)
        c_ready_bar = c_ready_bars.index(store_idx)

        mbarrier.wait(c_empty_bar, store_phase ^ 1)
        c_buf.store(c_val)
        # Fence to order with TMA store.
        fence_async_shared()
        mbarrier.arrive(c_ready_bar, count=1)


@gluon.jit
def elementwise_add_warp_specialized_kernel(  #
        a_desc, b_desc, c_desc,  #
        xnumel, ynumel, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
        num_load_buffers: gl.constexpr, num_store_buffers: gl.constexpr, num_warps: gl.constexpr):
    # Pick a layout that makes it easy to avoid bank conflicts.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

    # Allocate all the buffers and barriers.
    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_load_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_load_buffers] + b_desc.block_type.shape, b_desc.layout)
    c_bufs = gl.allocate_shared_memory(c_desc.dtype, [num_store_buffers] + c_desc.block_type.shape, c_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_load_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_load_buffers, 1], mbarrier.MBarrierLayout())
    c_empty_bars = gl.allocate_shared_memory(gl.int64, [num_store_buffers, 1], mbarrier.MBarrierLayout())
    c_ready_bars = gl.allocate_shared_memory(gl.int64, [num_store_buffers, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(num_load_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)
    for i in gl.static_range(num_store_buffers):
        mbarrier.init(c_empty_bars.index(i), count=1)
        mbarrier.init(c_ready_bars.index(i), count=1)

    descs = (a_desc, b_desc, c_desc)
    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    numel = (xnumel, ynumel)

    pid = gl.program_id(0)
    xoff = pid * XBLOCK

    # `gl.warp_specialize` declares a warp-specialized section of the kernel.
    # It accepts arguments for the default partition function, which can include
    # tensors, and the default partition function. It takes arguments for all
    # the worker partitions, which cannot include tensors, and takes a list of
    # worker partition functions. The warps and register budget for each
    # partition are passed as lists.
    #
    # Note that warp and register allocation on NVIDIA GPUs is by warpgroup,
    # which are 4 consecutive warps. The number of warps used by a kernel is
    # rounded to the nearest multiple of 4. The compiler tries to organize the
    # warps to reduce the amount of registers allocated. The default partition
    # receives whatever registers are left over, based on `maxnreg` passed to
    # the kernel.
    gl.warp_specialize(
        default_args=(barriers, buffers, ynumel, YBLOCK, layout),
        default_partition=compute_partition,
        worker_args=(descs, barriers, buffers, xoff, numel, YBLOCK),
        worker_partitions=[load_partition, store_partition],
        worker_num_warps=[1, 1],
        # Registers must be allocated in multiples of 8, between [24, 256].
        worker_num_regs=[24, 24],
    )


def elementwise_add_warp_specialized(a, b, c, XBLOCK=32, YBLOCK=64,  #
                                     num_load_buffers=2, num_store_buffers=2, num_warps=4):
    xnumel, ynumel = a.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )

    block_shape = [XBLOCK, YBLOCK]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)

    # By default, a warp-specialized kernel assumes maxnreg=256, the maximum
    # allowed per thread, in order to determine how to reallocate registers.
    # We need to intentionally set the register limit. Since the kernel will
    # have `num_warps+4` warps total, register usage will be
    #
    #     maxnreg * (num_warps+4) * 32
    #
    # Keep this in mind when deciding how much occupancy you want.
    elementwise_add_warp_specialized_kernel[grid](  #
        a_desc, b_desc, c_desc, xnumel, ynumel,  #
        XBLOCK, YBLOCK, num_load_buffers, num_store_buffers,  #
        num_warps=num_warps, maxnreg=128)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 120)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 64)])
@pytest.mark.parametrize("num_load_buffers, num_store_buffers", [(1, 1), (2, 2)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_elementwise_add_warp_specialized(xnumel, ynumel, XBLOCK, YBLOCK, num_load_buffers, num_store_buffers,
                                          num_warps):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add_warp_specialized(a, b, c, XBLOCK, YBLOCK, num_load_buffers, num_store_buffers, num_warps)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


if __name__ == "__main__":
    print("Benchmarking elementwise_add")
    print("============================")
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")

    XBLOCK = 64
    YBLOCK = 128
    num_load_buffers = 3
    num_store_buffers = 1
    num_warps = 4

    ms = triton.testing.do_bench(lambda: t4.elementwise_add_tma(  #
        A, B, C, XBLOCK, YBLOCK, num_load_buffers))
    print(f"elementwise_add_tma: {t3.get_throughput(ms, C):.2f} TB/s")

    ms = triton.testing.do_bench(lambda: elementwise_add_warp_specialized(  #
        A, B, C, XBLOCK, YBLOCK, num_load_buffers, num_store_buffers, num_warps))
    print(f"elementwise_add_warp_specialized: {t3.get_throughput(ms, C):.2f} TB/s")
    print()

# %%
# Results on GB200:
#
# ```
# elementwise_add_tma: 5.89 TB/s
# elementwise_add_warp_specialized: 5.98 TB/s
# ```
#
# The warp specialized implementation ekes out another performance gain over
# the software pipelined kernel from 04-tma.py by relying on the warp scheduler
# to hide latencies. The gains are modest because the kernel is very bandwidth
# bound, but this shows how warp specialization can more efficiently issue
# loads.

# %%
# Recall in previous tutorials we sometimes designed kernels to run with
# occupancy greater than 1. This is typical of kernels that we expect to stall
# or otherwise cannot exhaustively use the SM's resources. In doing so, we
# relied on the warp scheduler to overlap kernel instances and hide latencies.
#
# However, because programs cannot see what other programs on the SM are doing,
# they cannot coordinate usage of SM compute units or share resources. Warp
# specialization is especially powerful when used to build intricate schedules
# that minimize the critical path and maximize hardware utilization. In other
# words, warp specialization allows us to fuse multiple programs into
# one kernel.

# %%
# Since we have unfinished business with Blackwell matmul from the last
# tutorial, let's demonstrate a warp-specialized persistent matmul with tcgen05.
#
# - Use the same block sizes BLOCK_{M,N,K} = (128, 256, 64)
# - Aim for 4 buffers using techniques to reduce epilogue smem.
# - Double-buffer the accumulator to fully overlap the epilogue.
#
# Because the epilogue is overlapped, we can subtile by a factor of 4 to allow
# 4 buffers. However, for tiny K, it might still be better to steal B.


# Helper class for passing arguments around partitions.
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
    SUBTILE_FACTOR: gl.constexpr
    num_warps: gl.constexpr

    def __init__(self, a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_bufs,
                 acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps):
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
        self.SUBTILE_FACTOR = SUBTILE_FACTOR
        self.num_warps = num_warps


# Counter abstraction for tracking barrier index and phase.
@aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = num_barriers

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self):
        incr = self.index + 1
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)


@gluon.jit
def matmul_load_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    empty_bars = p.load_empty_bars
    ready_bars = p.load_ready_bars
    state = Counter.create(1, empty_bars.shape[0])

    # Just loop over all tiles and issue loads.
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        for k in range(0, K, BLOCK_K):
            # Acquire buffers, issue loads, and complete them asynchronously.
            bar = ready_bars.index(state.index)
            mbarrier.wait(empty_bars.index(state.index), state.phase)
            mbarrier.expect(bar, p.a_desc.block_type.nbytes + p.b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(p.a_desc, [off_m, k], bar, p.a_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()


@gluon.jit
def matmul_mma_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    load_empty_bars = p.load_empty_bars
    load_ready_bars = p.load_ready_bars
    load_state = Counter.create(0, load_empty_bars.shape[0])

    acc_empty_bars = p.acc_empty_bars
    acc_ready_bars = p.acc_ready_bars
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])

    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for _ in range(scheduler.get_num_tiles()):
        # Acquire the accumulator for the entire inner loop.
        mbarrier.wait(acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for k in range(0, K, BLOCK_K):
            # Acquire operands, issue MMA, and complete asynchronously.
            mbarrier.wait(load_ready_bars.index(load_state.index), load_state.phase)
            tcgen05_mma(p.a_bufs.index(load_state.index), p.b_bufs.index(load_state.index), acc_buf, use_acc=use_acc)
            tcgen05_commit(load_empty_bars.index(load_state.index))
            load_state = load_state.next()
            use_acc = True
        # Complete the accumulator asynchronously.
        tcgen05_commit(acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


# Helper for splitting a tensor along N. For our kernel, this only works for
# BLOCK_M=128 and num_warps=4, where all BLOCK_N elements are contiguously
# mapped to the same thread.
@gluon.jit
def _split_n(x, SUBTILE_FACTOR: gl.constexpr):
    split_count: gl.constexpr = SUBTILE_FACTOR.bit_length() - 1  # log2
    xs = (x, )
    for _ in gl.static_range(split_count):
        next_xs = ()
        for j in gl.static_range(len(xs)):
            x = xs[j]
            # Reshape to (M, 2, N//2) then permute so that tensor elements
            # remain contiguous along N.
            next_xs += x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()
        xs = next_xs
    return xs


@gluon.jit
def matmul_epilogue_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    dtype: gl.constexpr = p.c_desc.dtype

    acc_empty_bars = p.acc_empty_bars
    acc_ready_bars = p.acc_ready_bars
    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    acc_layout: gl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, [BLOCK_M, BLOCK_N], p.num_warps)
    SPLIT_N: gl.constexpr = BLOCK_N // p.SUBTILE_FACTOR
    acc_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, SPLIT_N], p.c_desc.layout)

    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        # Wait for the accumulator. Since BLOCK_N=256, we need to interleave
        # the TMEM loads with the SMEM stores to avoid spilling.
        mbarrier.wait(acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index).load(acc_layout)
        acc_state = acc_state.next()

        accs = _split_n(acc, p.SUBTILE_FACTOR)
        for i in gl.static_range(len(accs)):
            acc = accs[i].to(dtype)
            tma.store_wait(pendings=0)  # overlap with downcast
            acc_smem.store(acc.to(dtype))
            # Arrive after the first SMEM store and rely on ptxas to interleave.
            if i == 0:
                mbarrier.arrive(acc_empty_bars.index(acc_state.index), count=1)
            fence_async_shared()
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + SPLIT_N * i], acc_smem)
    # Overlap the last store with the wait, then wait for the last store here.
    tma.store_wait(pendings=0)


@gluon.jit
def matmul_warp_specialized_kernel(a_desc, b_desc, c_desc, SchedulerImpl: gl.constexpr, num_buffers: gl.constexpr,
                                   SUBTILE_FACTOR: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype

    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], unpacked=True)
    acc_bufs = allocate_tensor_memory(gl.float32, [2, BLOCK_M, BLOCK_N], tmem_layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = PartitionArgs(a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_bufs,
                      acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps)
    gl.warp_specialize(
        default_args=(p, SchedulerImpl),
        default_partition=matmul_epilogue_partition,
        worker_args=(p, SchedulerImpl),
        worker_partitions=[matmul_load_partition, matmul_mma_partition],
        worker_num_warps=[1, 1],
        worker_num_regs=[24, 24],
    )


def matmul_warp_specialized(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, SUBTILE_FACTOR, num_warps, SchedulerImpl):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    matmul_warp_specialized_kernel[grid](a_desc, b_desc, c_desc, SchedulerImpl, num_buffers, SUBTILE_FACTOR,
                                         num_warps=num_warps)


t7 = importlib.import_module("07-persistence")


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
@pytest.mark.parametrize("SUBTILE_FACTOR", [4])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("SchedulerImpl", t7.schedulers)
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_matmul_warp_specialized(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, SUBTILE_FACTOR, num_warps,
                                 SchedulerImpl):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    matmul_warp_specialized(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, SUBTILE_FACTOR, num_warps, SchedulerImpl)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


if __name__ == "__main__" and is_blackwell():
    print("Benchmarking matmul_warp_specialized")
    print("====================================")
    args = {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "num_buffers": 4,
        "SUBTILE_FACTOR": 4,
        "num_warps": 4,
        "SchedulerImpl": t7.GroupedPersistentTileScheduler(8),
    }

    M, N = 8192, 8192
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    print("    K  warp-specialized    cublas")
    for K in [2**i for i in range(9, 15)]:
        as_flops = partial(t7.get_flops, M=M, N=N, K=K)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        BT = B.T.contiguous()
        r0 = as_flops(triton.testing.do_bench_cudagraph(lambda: matmul_warp_specialized(A, B, C, **args)))
        r1 = as_flops(triton.testing.do_bench(lambda: cublas.matmul(A, BT, C)))
        print(f"{K:>5} {r0:>17.2f} {r1:>9.2f}")

# %%
#     K  warp-specialized    cublas
#   512           1160.28   1130.67
#  1024           1249.69   1148.52
#  2048           1347.18   1261.59
#  4096           1390.95   1299.38
#  8192           1350.01   1401.10
# 16384           1448.14   1508.76
#
# Much better! We are beating cublas on small K, even though there is still lots
# of tuning we can do to improve performance. On Blackwell, warp specialization
# is critical for achieving peak performance.
