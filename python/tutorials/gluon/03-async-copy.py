"""
Async Copy in Gluon
===================

Modern GPUs provide asynchronous instructions for long-running operations like
global memory reads and writes. Asynchronous operations allow overlapping memory
transactions with compute, also known as "pipelining".

Asynchronous instructions vary by GPU vendor and architecture, so this tutorial
focuses on NVIDIA GPUs. On NVIDIA GPUs, async copies transfer data between
global memory and shared memory, unlike `ld.global` and `st.global` which
directly write to and read from the register file.
"""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

# %%
# Let's reimplement the 1D memcpy using `cp.async` to demonstrate the basics.
# Shared memory is represented using a descriptor type. Shared memory has a
# layout, like tensors in registers. The layout is selected to reduce bank
# conflicts when reading and writing to shared memory, but it may also be chosen
# to meet the constraints of certain operations.


@gluon.jit
def memcpy_1d_cpasync_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)

    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel

    # For 1D tensor, pick a simple layout.
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)

    # Issue the async copy.
    cp.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
    # `commit_group` puts all previously issued async copies into a group.
    cp.commit_group()

    # Wait until the number of pending commit groups reaches 0. Then we can
    # retrieve the data from shared memory.
    cp.wait_group(0)

    value = smem.load(layout)
    gl.store(out_ptr + offsets, value, mask=mask)


def memcpy_1d_cpasync(input, output, XBLOCK=8192, num_warps=4):
    grid = (triton.cdiv(input.numel(), XBLOCK), )
    memcpy_1d_cpasync_kernel[grid](input, output, input.numel(), XBLOCK)


@pytest.mark.parametrize("xnumel, XBLOCK", [(200, 128), (1000, 256)])
def test_memcpy_1d_cpasync(xnumel, XBLOCK):
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy_1d_cpasync(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)

# %%
# You can see that we can overlap the async copy with compute by issuing the
# copy and performing compute without waiting on it, also known as "pipelining".
#
