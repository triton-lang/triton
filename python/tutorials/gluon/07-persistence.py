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
is still possible with more advanced techniques or hardware features like CLC.

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

t5 = importlib.import_module("05-wgmma")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

# %%
# In the previous two tutorials, we introduced tensor core operations for Hopper
# and Blackwell NVIDIA GPUs. To make this tutorial more accessible, we will
# build an abstraction around both sets of tensor core operations so that our
# persistent matmul can be used on both Hopper and Blackwell.
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
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_buffers: gl.constexpr,
                   num_warps: gl.constexpr):
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
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_buffers: gl.constexpr,
                   num_warps: gl.constexpr):
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
# Let's quickly validate our abstraction by implementing a matmul where we
# pipeline the MMA with the loads. The structure of this kernel should be
# familiar.
#
# We will sanity check correctness and performance.


@gluon.jit
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    a_bufs = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [2] + b_desc.block_type.shape, b_desc.layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Use our MMA abstraction!
    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, 1, num_warps)
    for k in range(0, K, BLOCK_K):
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(phase))
        tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(phase))
        mbarrier.wait(bar, phase)
        mma = mma.wait_num_outstanding(0)
        mma = mma.issue_async_mma(a_bufs.index(phase), b_bufs.index(phase))
        phase ^= 1
    mbarrier.invalidate(bar)
    mma = mma.wait_num_outstanding(0)

    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c, mma = mma.take_result()
    c_smem.store(c.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    MMAImpl = select_mma_impl()
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_pipelined_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


# This mode runs each kernel once so we can capture it in ncu.
def profiling_in_ncu():
    return len(sys.argv) > 1 and sys.argv[1] == "profile"


if __name__ == "__main__":
    BLOCK_M = BLOCK_N = 128
    BLOCK_K = 64
    num_warps = 4

    M, N, K = 8192, 8192, 16 * 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    fn = lambda: blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    if profiling_in_ncu():
        fn()
    else:
        ms = triton.testing.do_bench_cudagraph(fn)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"Pipelined performance: {tflops_per_sec:.2f} tflops/s")

# %%
# On Hopper:     497 TFLOPS
# On Blackwell: 1010 TFLOPS
#
# This performance lines up with what we have seen in previous tutorials. To
# make the kernel persistent, all we have to do is put an outer loop around the
# kernel and iterate over the pids assigned to that kernel. We can query the
# total number of kernels using gl.num_programs.


@aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    def __init__(self, pid_start, pid_end, num_pid_m):
        self.pid_start = pid_start
        self.pid_end = pid_end
        self.num_pid_m = num_pid_m

    @staticmethod
    def get_grid(M, N, BLOCK_M, BLOCK_N, BLOCK_K, occupancy):
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        return (min(occupancy * num_sms, num_pid), )

    # Basic persistent tile scheduler that distribute the tiles evenly across
    # kernels by computing the number of tiles per kernel.
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

    # Note that we can make our kernel non-persistent by passing a tile
    # scheduler that returns 1 for get_num_tiles() and returns program_id for
    # get_tile(0).
    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        pid = self.pid_start + idx
        # Delinearize the pid into pid_m, pid_n along M. We will see later that
        # this is not the optimal schedule.
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n


# %%
# The most naive way to make the kernel persistent would be to place the outer
# loop around the entire non-persistent kernel. But let's re-use the TMA barrier
# and MMA state. We must scope the operand buffers to the inner loop so the
# shared memory allocator knows their liveranges do not intersect with the TMA
# store buffer.


@gluon.jit
def persistent_matmul_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, SchedulerImpl: gl.constexpr,
                             num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    M = c_desc.shape[0]
    N = c_desc.shape[1]
    K = a_desc.shape[1]

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, 1, num_warps)
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)

        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        a_bufs = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
        b_bufs = gl.allocate_shared_memory(dtype, [2] + b_desc.block_type.shape, b_desc.layout)
        for k in range(0, K, BLOCK_K):
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(phase))
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(phase))
            mbarrier.wait(bar, phase)
            mma = mma.wait_num_outstanding(0)
            mma = mma.issue_async_mma(a_bufs.index(phase), b_bufs.index(phase))
            phase ^= 1
        mma = mma.wait_num_outstanding(0)

        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
        c, mma = mma.take_result()
        c_smem.store(c.to(dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
        tma.store_wait(pendings=0)


def persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, SchedulerImpl):
    M, N = C.shape
    MMAImpl = select_mma_impl()

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N], c_layout)

    occupancy = 3 if BLOCK_K == 64 else 1
    grid = SchedulerImpl.get_grid(M, N, BLOCK_M, BLOCK_N, BLOCK_K, occupancy)
    persistent_matmul_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, SchedulerImpl, num_warps=num_warps)


schedulers = [PersistentTileScheduler]


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("SchedulerImpl", schedulers)
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_persistent_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, SchedulerImpl):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, SchedulerImpl)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


if __name__ == "__main__":
    fn = lambda: persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, PersistentTileScheduler)
    if profiling_in_ncu():
        fn()
    else:
        ms = triton.testing.do_bench_cudagraph(fn)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"Persistent performance: {tflops_per_sec:.2f} tflops/s")

# %%
# On Hopper:     497 -> 525 TFLOPS
# On Blackwell: 1010 -> 919 TFLOPS
#
# On Hopper, we get an immediate modest improvement, but on Blackwell,
# performance drops. We can use ncu and the profiling_in_ncu() flag to capture a
# profile of the kernels. Pass `--set full` to capture all performance metrics.
#
# ```
# ncu --set full -o persistent --kernel-name persistent_matmul_kernel        python 07-persistence.py profile
# ncu --set full -o pipelined  --kernel-name blocked_matmul_pipelined_kernel python 07-persistence.py profile
# ncu --import persistent.ncu-rep | grep "L2 Hit Rate"
#     L2 Hit Rate                            %        69.72
# ncu --import  pipelined.ncu-rep | grep "L2 Hit Rate"
#     L2 Hit Rate                            %        80.34
# ```
#
# These were captured on Blackwell. You can see the L2 hit rate has decreased.
# Hopper has a different L2 cache structure, so it's possible we didn't observe
# this problem by chance.
#
# We can improve L2 efficiency by "super-grouping" the tiles along columns. See
# 03-matrix-multiplication.py for more details. We can supply a different tile
# scheduler.


def GroupedPersistentTileScheduler(GROUP_SIZE_M):
    # Bind this as a constexpr so it can be captured.
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

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

        @staticmethod
        def get_grid(M, N, BLOCK_M, BLOCK_N, BLOCK_K, occupancy):
            num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            return (min(occupancy * num_sms, num_pid), )

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
            # tile_id = gl.cdiv(self.num_pid, gl.num_programs(axis=0)) * gl.program_id(axis=0) + idx
            group_id = tile_id // self.num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % self.num_pid_in_group) // group_size_m
            return pid_m, pid_n

    # Like C++ templates!
    GroupedPersistentTileSchedulerImpl.__name__ = f"GroupedPersistentTileScheduler({GROUP_SIZE_M.value})"
    return GroupedPersistentTileSchedulerImpl


# Add this to the testsuite.
schedulers += [GroupedPersistentTileScheduler(1), GroupedPersistentTileScheduler(8)]

if __name__ == "__main__":
    for GROUP_SIZE_M in [1, 2, 4, 8]:
        fn = lambda: persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps,
                                       GroupedPersistentTileScheduler(GROUP_SIZE_M))
        if profiling_in_ncu():
            fn()
        else:
            ms = triton.testing.do_bench_cudagraph(fn)
            flops = 2 * M * N * K
            tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
            print(f"GROUP_SIZE_M={GROUP_SIZE_M}: {tflops_per_sec:.2f} tflops/s")
    print()

# %%
# GROUP_SIZE_M  Hopper (tflops/s)  Blackwell (tflops/s)
#            1             536.29               1036.07
#            2             542.11               1025.44
#            4             540.41               1030.35
#            8             512.90               1036.42
#
# We have recovered performance on Blackwell and even improved performance on
# Hopper. We see L2 hit rate at around 85% in the profiles on Blackwell now.
# Note the same L2 swizzling can be applied to non-persistent kernels, but
# persistent kernels exacerbate the L2 problem since we can't rely on the CTA
# scheduler to hide the problem.

# %%
# The main benefit of a persistent kernel is we can pipeline across the
# outer loop. Recall in 04-tma.py, we pipelined the TMA store by rotating the
# wait. However, in doing so we increase the liverange of the TMA store buffer,
# reducing occupancy to 2 for BLOCK_K=64. Instead, let's aim for occupancy 1.
# Note that pipelining the outer loop benefits small K more than large K, since
# the proportion of time spent in the epilogue is greater.


@gluon.jit
def persistent_matmul_pipelined_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, SchedulerImpl: gl.constexpr,
                                       num_buffers: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    M = c_desc.shape[0]
    N = c_desc.shape[1]
    K = a_desc.shape[1]

    # The circular buffer will cycle across the outer loop iters.
    gl.static_assert(num_buffers >= 2, "expected at least 2 buffers")
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, 1, num_warps)
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            index = producer % num_buffers
            producer += 1
            bar = bars.index(index)
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index))
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index))

        for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
            index = producer % num_buffers
            producer += 1
            bar = bars.index(index)
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index))
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index))

            index = consumer % num_buffers
            phase = consumer // num_buffers & 1
            consumer += 1
            mbarrier.wait(bars.index(index), phase)
            mma = mma.wait_num_outstanding(0)
            mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(index))

        for _ in gl.static_range(num_buffers - 2):
            index = consumer % num_buffers
            phase = consumer // num_buffers & 1
            consumer += 1
            mbarrier.wait(bars.index(index), phase)
            mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(index))
        mma = mma.wait_num_outstanding(0)

        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
        c, mma = mma.take_result()
        # Async launch the TMA store.
        tma.store_wait(pendings=0)
        c_smem.store(c.to(dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def persistent_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    M, N = C.shape
    MMAImpl = select_mma_impl()

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)

    a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N], c_layout)

    grid = SchedulerImpl.get_grid(M, N, BLOCK_M, BLOCK_N, BLOCK_K, occupancy=1)
    persistent_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, SchedulerImpl, num_buffers,
                                             num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [2, 3])
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
    # Exhaustively use SMEM by increasing BLOCK_N instead of BLOCK_M, since this
    # increases our MMA instruction shape.
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    num_buffers = 3
    # We need 8 warps on Hopper to keep the accumulator in registers.
    is_hopper = torch.cuda.get_device_capability()[0] == 9
    num_warps = 8 if is_hopper else 4

    GMs = [None, 1, 2, 4, 6, 8]
    print("GROUP_SIZE_M: ", end="")
    for GROUP_SIZE_M in GMs:
        print(f"{str(GROUP_SIZE_M):>10}", end="")
    print()

    for K in [2**i for i in range(9, 15)]:
        print(f"K={K:<11} ", end="")
        for GROUP_SIZE_M in GMs:
            A = torch.randn(M, K, device="cuda", dtype=torch.float16)
            B = torch.randn(K, N, device="cuda", dtype=torch.float16)
            C = torch.empty(M, N, device="cuda", dtype=torch.float16)
            scheduler = PersistentTileScheduler if GROUP_SIZE_M is None else GroupedPersistentTileScheduler(
                GROUP_SIZE_M)

            fn = lambda: persistent_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps,
                                                     scheduler)
            if profiling_in_ncu():
                fn()
            else:
                ms = triton.testing.do_bench_cudagraph(fn)
                flops = 2 * M * N * K
                tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
                print(f"{tflops_per_sec:>10.2f}", end="")
        print()
    print()

# %%
# Blackwell results:
#
# ```
# GROUP_SIZE_M:       None         1         2         4         6         8
# K=512             876.72    808.20    802.24    804.04    802.39    808.04
# K=1024           1093.85   1015.53   1014.13   1015.34   1011.57   1025.98
# K=2048           1090.87   1161.39   1122.03   1111.42   1110.91   1151.28
# K=4096            949.96   1039.46    992.55   1051.57   1007.20   1020.01
# K=8192            980.18   1049.52   1033.49   1051.80   1015.44    992.78
# K=16384          1025.81   1075.19   1091.52   1074.91   1087.57   1074.81
# ```
#
# Hopper results:
#
# ```
# GROUP_SIZE_M:       None         1         2         4         6         8
# K=512             486.45    490.79    501.87    500.20    501.37    502.47
# K=1024            596.91    579.76    576.73    580.80    574.31    580.95
# K=2048            631.45    598.09    592.02    597.70    594.95    593.62
# K=4096            667.17    641.11    633.88    633.98    622.32    628.68
# K=8192            676.83    634.71    640.53    646.27    634.16    645.46
# K=16384           684.82    636.41    645.93    642.88    644.19    646.69
# ```
#
# At K=16384, which we previously benchmarked at, we gain 140 TFLOPS! Blackwell
# sees more modest gains at 55 TFLOPS. Observations: Hopper performs better with
# the default scheduler, whereas Blackwell best GROUP_SIZE_M depends on K.

# %%
# To improve performance further, especially for small K, we will pipeline the
# outer loop and overlap the epilogue. Fully overlapping the epilogue is
# challenging with only software pipelining.


@gluon.jit
def persistent_matmul_more_pipelined_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, SchedulerImpl: gl.constexpr,
                                            num_buffers: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    M = c_desc.shape[0]
    N = c_desc.shape[1]
    K = a_desc.shape[1]

    # The circular buffer will cycle across the outer loop iters.
    gl.static_assert(num_buffers >= 2, "expected at least 2 buffers")
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, 1, num_warps)
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            index = producer % num_buffers
            producer += 1
            bar = bars.index(index)
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index))
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index))

        for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
            index = producer % num_buffers
            producer += 1
            bar = bars.index(index)
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index))
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index))

            index = consumer % num_buffers
            phase = consumer // num_buffers & 1
            consumer += 1
            mbarrier.wait(bars.index(index), phase)
            mma = mma.wait_num_outstanding(0)
            mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(index))

        for _ in gl.static_range(num_buffers - 2):
            index = consumer % num_buffers
            phase = consumer // num_buffers & 1
            consumer += 1
            mbarrier.wait(bars.index(index), phase)
            mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(index))
        mma = mma.wait_num_outstanding(0)

        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
        c, mma = mma.take_result()
        # Async launch the TMA store.
        tma.store_wait(pendings=0)
        c_smem.store(c.to(dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def persistent_matmul_more_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    M, N = C.shape
    MMAImpl = select_mma_impl()

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)

    a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N], c_layout)

    grid = SchedulerImpl.get_grid(M, N, BLOCK_M, BLOCK_N, BLOCK_K, occupancy=1)
    persistent_matmul_more_pipelined_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, SchedulerImpl, num_buffers,
                                                  num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [2, 3])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("SchedulerImpl", schedulers)
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_persistent_matmul_more_pipelined(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    persistent_matmul_more_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)
