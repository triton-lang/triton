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
"""

import pytest
import torch
import triton
import importlib
from typing import Union
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma_init,
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
    fence_async_shared,
)

t5 = importlib.import_module("05-wgmma")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")


@aggregate
class WGMMA:
    acc: Union[warpgroup_mma_accumulator, gl.tensor]

    def __init__(self, acc):
        self.acc = acc

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_buffers: gl.constexpr,
                   num_warps: gl.constexpr):
        mma_layout: gl.constexpr = t5.pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
        return WGMMA(acc)

    @gluon.jit
    def issue_async_mma(self, a, b):
        acc = warpgroup_mma(a, b, self.acc, is_async=True)
        return WGMMA(acc)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        acc = warpgroup_mma_wait(num_outstanding, (self.acc, ))
        return WGMMA(acc)

    @gluon.jit
    def get_result(self):
        return self.acc


@aggregate
class MMAv5:
    use_acc: gl.tensor
    acc_tmem: tensor_memory_descriptor
    bars: gl.shared_memory_descriptor
    counter: gl.tensor
    num_buffers: gl.constexpr
    reg_layout: gl.constexpr

    def __init__(self, use_acc, acc_tmem, bars, counter, num_buffers, reg_layout):
        self.use_acc = use_acc
        self.acc_tmem = acc_tmem
        self.bars = bars
        self.counter = counter
        self.num_buffers = gl.constexpr(num_buffers)
        self.reg_layout = reg_layout

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_buffers: gl.constexpr,
                   num_warps: gl.constexpr):
        layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], unpacked=True)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], layout)
        bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(num_buffers):
            mbarrier.init(bars.index(i), count=1)
        reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, [BLOCK_M, BLOCK_N], num_warps)
        return MMAv5(gl.to_tensor(False), acc_tmem, bars, gl.to_tensor(0), num_buffers, reg_layout)

    @gluon.jit
    def issue_async_mma(self, a, b):
        tcgen05_mma(a, b, self.acc_tmem, use_acc=self.use_acc)
        bar = self.bars.index(self.counter % self.num_buffers)
        tcgen05_commit(bar)
        return MMAv5(gl.to_tensor(True), self.acc_tmem, self.bars, self.counter + 1, self.num_buffers, self.reg_layout)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        bar = self.bars.index((self.counter - 1 - num_outstanding) % self.num_buffers)
        mbarrier.wait(bar, (self.counter - 1 - num_outstanding) // self.num_buffers & 1)
        return self

    @gluon.jit
    def get_result(self):
        return self.acc_tmem.load(self.reg_layout)


def select_mma_impl():
    if torch.cuda.get_device_capability()[0] == 9:
        return WGMMA
    elif torch.cuda.get_device_capability()[0] == 10:
        return MMAv5
    else:
        return None


@gluon.jit
def matmul_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, 1, num_warps)

    for k in range(0, K, BLOCK_K):
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_smem)
        mbarrier.wait(bar, phase)
        phase ^= 1

        mma = mma.issue_async_mma(a_smem, b_smem)
        mma = mma.wait_num_outstanding(0)

    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(mma.get_result().to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    MMAImpl = select_mma_impl()

    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_blocked_matmul_pipelined(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)
