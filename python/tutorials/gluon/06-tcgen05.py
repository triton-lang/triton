"""
The 5th Generation TensorCore^TM
================================

This tutorial covers the APIs for interacting with Tensor Cores on Blackwell
GPUs. Blackwell Tensor Cores introduce a new memory space called Tensor Memory
that must be used to interact with the async MMA instructions.

In this tutorial, we will cover allocating and interacting with Tensor Memory
and demonstrate how to use the `tcgen05` MMA instructions. We will build a
simple matmul kernel to demonstrate practical uses of the APIs and show an
example of how to pipeline MMA instructions.
"""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    tensor_memory_descriptor,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
)


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_hopper():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")

# %%
# Tensor memory is a 2D memory space organized into 128 rows and 512 columns of
# 32-bit cells per SM. Accessing tensor memory is significantly faster than
# shared memory, but there are additional limitations:
#
# - Each warp can only access 32 rows of tensor memory based on its warp ID,
#   thus a whole warp group is required to collectively access all 128 rows.
# - Tensor memory is allocated by number of columns. The allocation size must be
#   a power of 2 in the range [32, 512].
# - In Gluon, tensor memory load and store operations require 4 or 8 warps.
# - In Gluon, only 2D tensors can be loaded from and stored to tensor memory.
# - Data can be asynchronously copied from shared memory to tensor memory, but
#   this API is not yet exposed in Gluon.
#
# Data stored in tensor memory has layouts, just like shared memory. Due to the
# tensor memory restrictions, the register layout of tensors being stored to or
# loaded from tensor memory is constrained by the tensor memory layout.
#
# A few more notes on tensor memory:
#
# - Tensor memory is essentially an extra register file. You will notice that
#   128 * 512 = 64K 32-bit cells, just like the SM register file.
# - Tensor memory can be used independent of MMA instructions. It can be used
#   in-place of shared memory to transfer data as permitted by the layout
#   restrictions.
# - Tensor memory is dynamically allocated on the SM, so while tensor memory
#   does not directly affect occupancy, the allocation will block if there is
#   not enough tensor memory available.

# %%
# Tensor memory layouts organize data into 2D blocks:
#
# ```python
# TensorMemoryLayout(
#     block=(blockM, blockN),
#     unpacked=True,
# )
#
# The tensor is divided into (blockM, blockN) blocks, where blockM must be 64
# or 128. blockN must be a power of 2 between [1, 256]. For dtypes smaller than
# 32 bits, multiple elements can be packed into each 32-bit cell if
# unpacked=False, however blockN must then be at least `32 // bitwidth`.
#
# Note that when blockM=64, tensors with multiple blocks are packed in TMEM to
# use all 128 rows. This can complicate subslicing TMEM descriptors.
#
# The underlying `tcgen05.st` and `tcgen05.ld` instructions are warp-level
# instructions that access TMEM in specific patterns. Combined with the warp
# row-addressing restrictions, this gives rise to the register layout
# restrictions on tensor memory. Certain tensor memory layouts support multiple
# register layouts, which affect the selected atom. In this tutorial, we will
# only use the `32x32b` atom: each lane rows 1 row of TMEM.


@gluon.jit
def tmem_example_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr, num_warps: gl.constexpr):
    ldgstg_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

    offs_m = gl.arange(0, M, gl.SliceLayout(1, ldgstg_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, ldgstg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]

    input = gl.load(in_ptr + offs)

    # Allocate some tensor memory.
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(64, 64),
        unpacked=True,
    )

    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=tmem_layout,
    )

    # Get the register layout needed to access the tensor memory using a helper.
    tmem_reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(
        M=64,
        N=64,
        shape=[M, N],
        num_warps=num_warps,
    )

    input = gl.convert_layout(input, tmem_reg_layout)
    tmem.store(input)
    output = tmem.load(tmem_reg_layout)
    output = gl.convert_layout(output, ldgstg_layout)

    gl.store(out_ptr + offs, output)


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_tmem_example_kernel(M, N, num_warps):
    input = torch.randn(M, N, dtype=torch.float32, device="cuda")
    output = torch.empty_like(input)

    tmem_example_kernel[(1, )](input, output, M, N, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)
