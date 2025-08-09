"""
Persistent Kernels
==================

So far, we have defined kernels such that one programs handles one block of work
and we span all the work using the grid dimensions. This creates a large number
of programs, and we rely on the GPU to schedule the work. The primary benefit is
the GPU will dynamically load-balance the work across its SMs.

However, this approach has downsides. The scheduler incurs an overhead, and the
GPU is not aware of the memory access patterns of the kernels. This also
prevents overlapping across blocks of work, as the GPU waits until kernels have
fully exited before issuing more work.

Persistent kernels is a technique where we assign multiple blocks of work to
each program, and the programs "persist" on the GPU until all the work is
complete. The work assignment is typically static, although dynamic scheduling
is still possible with more advanced techniques or hardware features like
cluster launch control.

In this tutorial, we will explore persistent kernels by implementing a
persistent matmul. We will then show how we can pipeline across the persistent
outer loop to achieve greater overlap and more throughput.
"""

import itertools
import pytest
import torch
import triton
import importlib
import sys
from functools import partial
from typing import Union
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
    warpgroup_mma_accumulator,
)
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

t5 = importlib.import_module("05-wgmma")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

profiling_with_ncu = len(sys.argv) > 1 and sys.argv[1] == "profile"


def get_flops(ms, M, N, K):
    flops = 2 * M * N * K
    return flops * 1e-12 / (ms * 1e-3)


# %%
# In the previous two tutorials, we introduced tensor core operations for Hopper
# and Blackwell NVIDIA GPUs. To make this tutorial more accessible, and to
# demonstrate some Gluon features, we will build an abstraction around both sets
# of tensor core operations so that our persistent matmul can be used on both
# Hopper and Blackwell.
#
# We can use @aggregate to define a class that contains the state of the
# matmul. We will define the API of our MMA wrapper to be like WGMMA's, because
# is the more restrictive of the two.


# MMA wrapper for WGMMA, which maps directly to the WGMMA functions.
@aggregate
class WGMMA:
    acc: Union[warpgroup_mma_accumulator, gl.tensor]
    use_acc: gl.tensor

    def __init__(self, acc, use_acc):
        self.acc = acc
        self.use_acc = use_acc

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        mma_layout: gl.constexpr = t5.pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
        return WGMMA(acc, gl.to_tensor(False))

    @gluon.jit
    def issue_async_mma(self, a, b):
        acc = warpgroup_mma(a, b, self.acc, is_async=True, use_acc=self.use_acc)
        # Note that aggregates don't support in-place mutation, so we need to
        # return a new instance and re-assign it at the callsite.
        return WGMMA(acc, gl.to_tensor(True))

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        acc = warpgroup_mma_wait(num_outstanding, (self.acc, ))
        return WGMMA(acc, self.use_acc)

    # Take the result and reset the accumulator.
    @gluon.jit
    def take_result(self):
        return self.acc, WGMMA(self.acc, gl.to_tensor(False))


# MMA wrapper for tcgen05. In order to implement `wait_num_outstanding`, we
# need to allocate barriers and keep track of how many MMAs have been issued.
# State will be tracked with an accumulator.
@aggregate
class MMAv5:
    use_acc: gl.tensor
    acc_tmem: tensor_memory_descriptor
    bar: gl.shared_memory_descriptor
    counter: gl.tensor
    reg_layout: gl.constexpr

    def __init__(self, use_acc, acc_tmem, bar, counter, reg_layout):
        self.use_acc = use_acc
        self.acc_tmem = acc_tmem
        self.bar = bar
        self.counter = counter
        self.reg_layout = reg_layout

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], unpacked=True)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], layout)
        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, [BLOCK_M, BLOCK_N], num_warps)
        return MMAv5(gl.to_tensor(False), acc_tmem, bar, gl.to_tensor(0), reg_layout)

    @gluon.jit
    def issue_async_mma(self, a, b):
        tcgen05_mma(a, b, self.acc_tmem, use_acc=self.use_acc)
        tcgen05_commit(self.bar)
        return MMAv5(gl.to_tensor(True), self.acc_tmem, self.bar, self.counter + 1, self.reg_layout)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        mbarrier.wait(self.bar, (self.counter - 1 - num_outstanding) & 1)
        return self

    @gluon.jit
    def take_result(self):
        next = MMAv5(gl.to_tensor(False), self.acc_tmem, self.bar, self.counter, self.reg_layout)
        return self.acc_tmem.load(self.reg_layout), next


def select_mma_impl():
    if torch.cuda.get_device_capability()[0] == 9:
        return WGMMA
    elif torch.cuda.get_device_capability()[0] == 10:
        return MMAv5
    else:
        return None


# %%
# Let's validate our abstraction by implementing a matmul where we pipeline both
# the MMA and the loads. This achieves async overlap of both the TMA loads and
# the MMAs by requiring at least two operand buffers. This will make the
# persistent kernel more interesting by allowing us to overlap more things.
#
# We will factor our kernel into components we can re-use between
# implementations.


@gluon.jit
def issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers: gl.constexpr, pred=True):
    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index), pred)
    return producer


@gluon.jit
def issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(index))
    return consumer, mma


@gluon.jit
def matmul_pipelined_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, num_buffers: gl.constexpr,
                            num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    gl.static_assert(num_buffers >= 2, "expected at least 2 buffers")
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    # Separate producer and consumer indices, to support more than 2 buffers.
    producer = 0
    consumer = 0

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Use our MMA abstraction!
    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)

    # Prefetch at most num_buffers-2 loads to allow the MMA to overlap.
    for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)

    for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
        producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)
        consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

    for _ in gl.static_range(num_buffers - 2):
        consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

    mma = mma.wait_num_outstanding(0)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c, mma = mma.take_result()
    c_smem.store(c.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps):
    MMAImpl = select_mma_impl()
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, num_buffers, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_pipelined_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


# %%
# The optimal block shapes for our kernel are BLOCK_M=128 and BLOCK_N=256, which
# gives the maximum instruction shape on both Blackwell and Hopper. However, on
# Hopper we need 8 warps to fit the accumulator in registers.

if __name__ == "__main__":
    M, N, K = 8192, 8192, 16 * 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

if __name__ == "__main__" and not profiling_with_ncu:
    BLOCK_M = 128
    BLOCK_N = 256
    is_hopper = torch.cuda.get_device_capability()[0] == 9
    warps = [8] if is_hopper else [4, 8]
    print("Benchmarking pipelined matmul")
    print("=============================")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print("BLOCK_K num_buffers num_warps tflops/s")
    for (BLOCK_K, num_buffers), num_warps in itertools.product([(128, 2), (64, 3), (64, 4)], warps):
        print(f"{BLOCK_K:>7} {num_buffers:>11} {num_warps:>9}", end=" ")
        fn = lambda: matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps)
        ms = triton.testing.do_bench_cudagraph(fn)
        print(f"{get_flops(ms, M, N, K):8.2f}")
    print()

# %%
# BLOCK_K num_buffers num_warps Blackwell  Hopper
#     128           2         4    735.96
#     128           2         8    697.97  489.26
#      64           3         4   1054.00
#      64           3         8    973.94  673.67
#      64           4         4   1175.70
#      64           4         8   1072.83  669.16
#
# Blackwell performance lines up with what we have seen in previous tutorials,
# but on Hopper we see some wins. On Hopper, performance plateaus at 3 buffers,
# but on Blackwell we see benefits of 4 buffers. This suggests the throughput
# ratio has increased in favour of MMAs from Hopper to Blackwell. Noteworthy is
# our kernels are occupancy 1.

# %%
# To make the kernel persistent, all we have to do is put an outer loop around
# the kernel and iterate over the output tiles assigned to that kernel.
#
# Let's define a tile scheduler abstraction that will allow us to change the
# scheduling strategy, starting with a basic row-major tile scheduler.


@aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    def __init__(self, pid_start, pid_end, num_pid_m):
        self.pid_start = pid_start
        self.pid_end = pid_end
        self.num_pid_m = num_pid_m

    @gluon.jit
    def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n
        pid_per_kernel = gl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, num_pid)
        return PersistentTileScheduler(pid_start, pid_end, num_pid_m)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        # Delinearize the tile ID along M.
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n


# %%
# We can make the kernel persistent by literally placing the outer loop around
# the whole kernel, but let's re-use the TMA barrier and MMA state.
# We must scope the operand buffers to the inner loop so the shared memory
# allocator knows their liveranges do not intersect with the TMA store buffer.


@gluon.jit
def persistent_matmul_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, SchedulerImpl: gl.constexpr,
                             num_buffers: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    # Producer and consumer indices.
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
        b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
        for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)

        for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
            producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)
            consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

        for _ in gl.static_range(num_buffers - 2):
            consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

        mma = mma.wait_num_outstanding(0)
        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
        c, mma = mma.take_result()
        c_smem.store(c.to(dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
        tma.store_wait(pendings=0)


def persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    M, N = C.shape
    MMAImpl = select_mma_impl()

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    persistent_matmul_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, SchedulerImpl, num_buffers, num_warps=num_warps)


schedulers = [PersistentTileScheduler]


@pytest.mark.parametrize("M, N, K", [(2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("SchedulerImpl", schedulers)
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_persistent_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


if __name__ == "__main__" and not profiling_with_ncu:
    print("Benchmarking persistent matmul")
    print("==============================")
    print(f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N}")
    print("BLOCK_K num_buffers num_warps tflops/s")
    for (BLOCK_K, num_buffers), num_warps in itertools.product([(128, 2), (64, 3), (64, 4)], warps):
        print(f"{BLOCK_K:>7} {num_buffers:>11} {num_warps:>9}", end=" ")
        fn = lambda: persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps,
                                       PersistentTileScheduler)
        ms = triton.testing.do_bench_cudagraph(fn)
        print(f"{get_flops(ms, M, N, K):8.2f}")
    print()

# %%
# BLOCK_K num_buffers num_warps  Blackwell  Hopper
#     128           2         4     712.25
#     128           2         8     686.64  502.84
#      64           3         4    1032.16
#      64           3         8     938.81  661.11
#      64           4         4    1142.26
#      64           4         8    1071.46  658.84
#
# The Hopper kernel sees a modest improvement, but the Blackwell kernel
# performance is slightly lower. Let's capture a profile of the kernels on
# Blackwell using ncu. Pass `profile` to this script's arguments to run the two
# kernels once.

if __name__ == "__main__" and profiling_with_ncu:
    matmul_pipelined(A, B, C, 128, 256, 64, 4, 4)
    persistent_matmul(A, B, C, 128, 256, 64, 4, 4, PersistentTileScheduler)

# %%
# There are many reasons the persistent kernel can be slower. Load imbalance can
# arise due to inefficient scheduling (work is not evenly distributed). But it
# can also arise from drift at runtime, such as some TMA accesses taking longer
# than others, which a static tile scheduler cannot compensate for.
#
# Another reason we suspect is the global memory access pattern:
#
# ```
# ncu --set full -o pipelined  --kernel-name matmul_pipelined_kernel  python 07-persistence.py profile
# ncu --set full -o persistent --kernel-name persistent_matmul_kernel python 07-persistence.py profile
# ncu --import  pipelined.ncu-rep | grep "L2 Hit Rate"
#     L2 Hit Rate                            %        61.11
# ncu --import persistent.ncu-rep | grep "L2 Hit Rate"
#     L2 Hit Rate                            %        52.93
# ```
#
# The persistent kernel's L2 hit rate is 10% lower. We can improve L2 efficiency
# by "super-grouping" the tiles along columns. See 03-matrix-multiplication.py
# for more details. Let's encode this strategy in a new tile scheduler.


def GroupedPersistentTileScheduler(GROUP_SIZE_M):
    # Bind this as a constexpr so it can be captured.
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

    # Like C++ templates!
    @aggregate
    class GroupedPersistentTileSchedulerImpl:
        start_pid: gl.tensor
        num_pid_m: gl.tensor
        num_pid_in_group: gl.tensor
        num_pid: gl.tensor

        def __init__(self, start_pid, num_pid_m, num_pid_in_group, num_pid):
            self.start_pid = start_pid
            self.num_pid_m = num_pid_m
            self.num_pid_in_group = num_pid_in_group
            self.num_pid = num_pid

        @gluon.jit
        def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
            start_pid = gl.program_id(axis=0)
            num_pid_m = gl.cdiv(M, BLOCK_M)
            num_pid_n = gl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_pid = num_pid_m * num_pid_n
            return GroupedPersistentTileSchedulerImpl(start_pid, num_pid_m, num_pid_in_group, num_pid)

        @gluon.jit
        def get_num_tiles(self):
            return gl.cdiv(self.num_pid - self.start_pid, gl.num_programs(axis=0))

        @gluon.jit
        def get_tile(self, idx):
            tile_id = self.start_pid + idx * gl.num_programs(axis=0)
            group_id = tile_id // self.num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % self.num_pid_in_group) // group_size_m
            return pid_m, pid_n

    GroupedPersistentTileSchedulerImpl.__name__ = f"GroupedPersistentTileScheduler({GROUP_SIZE_M.value})"
    return GroupedPersistentTileSchedulerImpl


# Add this to the testsuite.
schedulers += [GroupedPersistentTileScheduler(1), GroupedPersistentTileScheduler(8)]

if __name__ == "__main__" and not profiling_with_ncu:
    num_warps = 8 if is_hopper else 4
    num_buffers = 3 if is_hopper else 4
    print("Benchmarking grouped scheduler")
    print("=============================")
    print(f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K}")
    print(f"num_buffers={num_buffers} num_warps={num_warps}")
    print("GROUP_SIZE_M tflops/s")
    for GROUP_SIZE_M in [1, 2, 4, 6, 8]:
        print(f"{GROUP_SIZE_M:>12}", end=" ")
        fn = lambda: persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps,
                                       GroupedPersistentTileScheduler(GROUP_SIZE_M))
        ms = triton.testing.do_bench_cudagraph(fn)
        print(f"{get_flops(ms, M, N, K):8.2f}")
    print()

# %%
# GROUP_SIZE_M Blackwell  Hopper
#            1   1025.11  649.09
#            2   1050.43  651.32
#            4   1032.71  655.51
#            6   1057.27  652.39
#            8   1179.94  648.42
#
# At GROUP_SIZE_M=8, we recover performance on Blackwell. In fact, under ncu we
# see the L2 hit rate increases to 70%, which suggests there are other ways to
# improve the scheduling.
#
# Performance decreases on Hopper with this scheduler. The L2 hit rate of the
# persistent kernel is 86% and 89% for the non-persistent kernel. The grouped
# scheduler does not affect the L2 hit rate but it does increase load imbalance.

# %%
# Pipelining across the outer loop benefits smaller K shapes more because a
# larger proportion of time is spent in the epilogue. We can try overlapping the
# TMA store with the next tile by rotating the TMA store wait.
#
# However, this causes the liverange of the TMA store buffer to overlap with the
# operand buffers, decreasing our max num_buffers to 3. While Hopper is fine
# with 3 buffers, on Blackwell performance can suffer. There are 3 remedies:
#
# 1. Use gl.store which does not require shared memory but it cannot be
#    pipelined. However, the layout conversion requires shared memory.
# 2. Break up the TMA store to multiple steps, allowing us to use smaller
#    buffers, we will only be able to pipeline the last step.
#    reduces the amount of overlap.
# 3. Borrow one of the b_bufs.
#
# For BLOCK_{M,N,K} = (128, 256, 64), one B buffer is half the size of the
# accumulator, but we have enough memory to use 5 buffers for B just so that we
# can steal two buffers for the epilogue, even though the inner loop only uses
# 4 at a time.


# Forked versions of issue_loads and issue_mma that support `stealb`.
@gluon.jit
def issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, stealb: gl.constexpr,
                       num_buffers: gl.constexpr, pred=True):
    index = producer % num_buffers
    b_index = producer % (num_buffers + stealb)
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(b_index), pred)
    return producer


@gluon.jit
def issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, stealb: gl.constexpr, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    b_index = consumer % (num_buffers + stealb)
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(b_index))
    return consumer, mma


@gluon.jit
def persistent_matmul_pipelined_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, SchedulerImpl: gl.constexpr,
                                       num_buffers: gl.constexpr, STEALB: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # All buffers share the same liverange.
    gl.static_assert(num_buffers >= 3, "expected at least 3 buffers")
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    # Add an extra B buffer when stealing.
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers + STEALB] + b_desc.block_type.shape, b_desc.layout)
    if not STEALB:
        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    else:
        gl.static_assert(2 * BLOCK_N * BLOCK_K >= BLOCK_M * BLOCK_N, "B tile not large enough to steal")
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    # Peeled inner loop prologue.
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, ki, bars, a_bufs, b_bufs, STEALB,
                                      num_buffers)
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB, num_buffers)

    for _ in range(num_tiles):
        consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)
        if STEALB:
            # Wait for the epilogue before the first TMA load.
            tma.store_wait(pendings=0)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB,
                                          num_buffers)
            consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        # Peel the next prologue and fuse it with the pipeline drain loop.
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # Predicate the peeled prologue instead of using a conditional.
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, ki, bars, a_bufs, b_bufs, STEALB,
                                          num_buffers, pred)
            consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)
        k = BLOCK_K * (num_buffers - 2)
        producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB,
                                      num_buffers)

        mma = mma.wait_num_outstanding(0)
        c, mma = mma.take_result()
        c = c.to(dtype)
        if not STEALB:
            c_buf = c_smem
            tma.store_wait(pendings=0)
        else:
            # Steal the next 2 B buffers for the epilogue.
            c_buf = b_bufs.index(producer % (num_buffers + STEALB))._reinterpret(dtype, c_desc.block_type.shape,
                                                                                 c_desc.layout)
        c_buf.store(c)
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [epilogue_off_m, epilogue_off_n], c_buf)
    tma.store_wait(pendings=0)


def persistent_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    M, N = C.shape
    MMAImpl = select_mma_impl()

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    persistent_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, SchedulerImpl, num_buffers,
                                             STEALB=num_buffers == 4, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("SchedulerImpl", schedulers)
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_persistent_matmul_pipelined(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    persistent_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


if __name__ == "__main__":
    args = {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "num_buffers": 3 if is_hopper else 4,
        "num_warps": 8 if is_hopper else 4,
    }
    scheduler = PersistentTileScheduler if is_hopper else GroupedPersistentTileScheduler(8)
    nonpersistent = partial(matmul_pipelined, **args)
    persistent = partial(persistent_matmul, **args, SchedulerImpl=scheduler)
    persistent_pipelined = partial(persistent_matmul_pipelined, **args, SchedulerImpl=scheduler)

    M, N = 8192, 8192
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    print("Benchmarking pipelined persistent")
    print("=================================")
    print("    K     nonpersistent    persistent   pipelined    cublas")
    for K in [2**i for i in range(9, 15)]:
        as_flops = partial(get_flops, M=M, N=N, K=K)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        BT = B.T.contiguous()
        r0 = as_flops(triton.testing.do_bench_cudagraph(lambda: nonpersistent(A, B, C)))
        r1 = as_flops(triton.testing.do_bench_cudagraph(lambda: persistent(A, B, C)))
        r2 = as_flops(triton.testing.do_bench_cudagraph(lambda: persistent_pipelined(A, B, C)))
        r3 = as_flops(triton.testing.do_bench(lambda: cublas.matmul(A, BT, C)))
        print(f"{K:>5} {r0:>17.2f} {r1:>13.2f} {r2:>11.2f} {r3:>9.2f}")

# %%
# Blackwell results:
#
#     K     nonpersistent    persistent   pipelined    cublas
#   512            615.86        828.70      993.50   1108.11
#  1024            997.16       1077.28     1173.31   1347.44
#  2048           1152.74       1190.55     1133.37   1435.01
#  4096           1164.05       1120.92     1143.47   1563.98
#  8192           1160.93       1074.97     1185.40   1491.84
# 16384           1185.62       1096.34     1296.93   1548.42
#
# Hopper results:
#
#     K     nonpersistent    persistent   pipelined    cublas
#   512            491.74        485.01      539.88    588.15
#  1024            554.24        575.02      602.52    588.32
#  2048            573.87        594.72      625.91    615.58
#  4096            609.36        630.10      640.48    646.30
#  8192            629.44        646.22      661.57    661.11
# 16384            653.79        660.29      670.00    665.49
#
# Persistent matmul, when pipelined, gains more performance relative to
# nonpersistent at lower K, as we would expect. Load balancing can be
# particularly difficult when the number of SMs do not evenly divide the number
# of blocks, and with 8192x8192, we are smack in the middle with ~13.5 and
# ~15.5 blocks per SM for Hopper and Blackwell, respectively.
#
# On Hopper, our pipelined kernel is competitive with cublas, even pulling ahead
# for medium-sized K. However, cublas has a definitive advantage at low K. On
# Blackwell, it's not even close: cublas is significantly faster.
#
# Some matmul performance takes:
#
# - On Hopper, software pipelining is sufficient to reach peak performance for
#   medium and large K.
# - cublas uses 2-CTA matmul, which uses distributed shared memory to allow
#   256x256 instruction shape. 2-CTA support in Gluon is very spotty,
#   but this enables cublas to more efficiently feed the MMA, which matters more
#   on Blackwell due to the relative increase in MMA throughput vs TMA.
# - cublas matmul is warp-specialized which is necessary on Hopper to fully
#   overlap the epilogue at small K.
# - Our Blackwell implementation is limited by the shared API we designed for
#   Hopper and Blackwell: we are not double-buffering the accumulator and
#   leaving 256 columns of TMEM unused.
# - On Blackwell, we can use `clusterlaunchcontrol` to dynamically schedule
#   work in conjunction with the GPU, getting the best of both worlds.
#
# Main takeaways:
#
# - Persistent kernels replace GPU block scheduling with a (typically) static
#   schedule. This allows more resource and compute coordination/overlap between
#   blocks at the cost of losing dynamic scheduling.
# - Persistent kernels tend to benefit smaller problem sizes, but still deliver
#   benefits for large problem sizes.
