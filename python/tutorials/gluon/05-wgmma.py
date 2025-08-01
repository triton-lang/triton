"""
Warp-Group MMA
==============

WGMMA (also known as MMAv3) is a Hopper-specific instruction for performing
matrix multiply-accumulate operations using the Tensor Cores. WGMMA instructions
are asynchronous, meaning they can be pipelined.

In this tutorial, we will cover how to use WGMMAs in Gluon. We will build a
simple matmul kernel to demonstrate practical uses of WGMMA, and show an example
where WGMMAs can be pipelined for better performance.
"""

import pytest
import torch
import triton
import importlib
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
)


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper():
    raise RuntimeError("This tutorial requires a Hopper NVIDIA GPU")

# %%
# Let's illustrate WGMMA with a trivial kernel launched with grid size (1, ).
# This kernel performs MMA on a small tensor.


@gluon.jit
def small_mma_kernel(a_desc, b_desc, c_desc, d_desc,  #
                     M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,  #
                     LHS_IN_REG: gl.constexpr, INSTR_SHAPE_N: gl.constexpr, num_warps: gl.constexpr):
    # Load A, B, and C tiles.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # A has shape [M, K].
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    # B has shape [K, N].
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
    # C has shape [M, N].
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
    tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
    tma.async_copy_global_to_shared(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # `warpgroup_mma` performs d = a * b + c. The `a` operand can be passed as
    # registers or through shared memory. The `b` operand must be passed through
    # shared memory, and the `c` operand must be passed through registers.
    #
    # `warpgroup_mma` itself is composed of many smaller `wgmma.mma_async` PTX
    # instructions, which supports a limited set of instruction shapes.
    #
    # The instruction shape is specified as [m, n, k], where
    #
    # - `k` is always 256 / A.dtype.primitive_bitwidth
    # - `m` is always 16
    # - `n` can be can chosen as follows:
    #
    # For floating point dtypes, `n` must be a positive multiple of 8, up to and
    # including 256. WGMMA supports 8-bit integers, but `n` must be chosen from:
    #
    #   224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 24, 16, 8
    #
    # `n` must be chosen such that it evenly divides into `N`, and it must be
    # less than or equal to `maxN`, where `maxN` is computed as:
    #
    #     mReps = ceildiv(M, m)
    #     nReps = ceildiv(num_warps, mReps)
    #     maxN = max(N // nReps, 8)
    #
    # `warpgroup_mma` divides the MMA across warps using `warps_per_cta`, in the
    # same way `BlockedLayout.warps_per_cta` tiles a tensor across warps. The
    # smallest indivisible unit of `warps_per_cta` is `[4, 1]`. Note that this
    # means WGMMA requires at least 4 warps. To choose the right
    # `warps_per_cta`, start from the atom `[4, 1]` and simply double it along
    # any dimension until it matches the number of warps.
    #
    # Note when `num_warps=8`, we can choose `[4, 2]` or `[8, 1]`, but recall
    # from 02-layouts that this can affect the performance of, e.g., reductions.

    # Let's parameterize the kernel over LHS_IN_REG and INSTR_SHAPE_N to see how
    # it can affect performance.
    m: gl.constexpr = 16
    k: gl.constexpr = 256 / a_desc.dtype.primitive_bitwidth
    n: gl.constexpr = INSTR_SHAPE_N
    warps_per_cta: gl.constexpr = [num_warps, 1]

    # The MMA shape is passed through the layout of `c`, which must always have
    # an NVMMADistributedLayout.
    c_layout: gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )
