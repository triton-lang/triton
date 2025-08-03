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
import importlib
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

# Re-use utilities from the previous tutorial.
t2 = importlib.import_module("02-layouts")


def is_ampere_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 8


if __name__ == "__main__" and not is_ampere_or_newer():
    raise RuntimeError("This tutorial requires Ampere or newer NVIDIA GPU")

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

    # Wait until the number of pending groups reaches 0. Then we can retrieve
    # the data from shared memory.
    cp.wait_group(0)

    value = smem.load(layout)
    gl.store(out_ptr + offsets, value, mask=mask)


def memcpy_1d_cpasync(input, output, XBLOCK=8192, num_warps=4):
    grid = (triton.cdiv(input.numel(), XBLOCK), )
    memcpy_1d_cpasync_kernel[grid](input, output, input.numel(), XBLOCK, num_warps=num_warps)


@pytest.mark.parametrize("xnumel, XBLOCK", [(200, 128), (1000, 256)])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_memcpy_1d_cpasync(xnumel, XBLOCK):
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy_1d_cpasync(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# You can see that we will able to overlap the async copy with compute by
# issuing the copy and performing compute without waiting on it. Let's use an
# elementwise addition kernel to explore pipelining.
#
# First, let's write the kernel such that each program performs additions for
# the whole row, one block at a time. For simplicity, we will assume all inputs
# have the same global memory layout.


@gluon.jit
def elementwise_add_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
):
    pid = gl.program_id(0)

    # Compute the offset to the row this program will process.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))

    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    for yoff in range(0, ynumel, YBLOCK):
        # Offset to the column block.
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        a_val = gl.load(a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        b_val = gl.load(b_ptrs + ystride_b * yoffs[None, :], mask=mask)

        c_val = a_val + b_val

        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add(A, B, C, XBLOCK=32, YBLOCK=64):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_add_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 32), (128, 128)])
def test_elementwise_add(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add(a, b, c, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


# %%
# Let's rewrite the kernel in-place to use async copies without pipelining,
# which will make it more obvious how we will pipeline the inner loop. Let's
# parameterize the kernel over the shared memory layout to see how it can
# affect performance.


@gluon.jit
def elementwise_add_cpasync_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    # New: declare shared memory for the A tile and B tile.
    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        # Issue loads for both A and B tiles.
        cp.async_copy_global_to_shared(a_smem, a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        cp.async_copy_global_to_shared(b_smem, b_ptrs + ystride_b * yoffs[None, :], mask=mask)
        # Commit both loads to the same group.
        cp.commit_group()
        # Wait until both loads are complete!
        cp.wait_group(0)

        a_val = a_smem.load(layout)
        b_val = b_smem.load(layout)

        c_val = a_val + b_val

        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add_cpasync(A, B, C, smem_layout, XBLOCK=32, YBLOCK=64):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_add_cpasync_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, smem_layout)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 32), (128, 128)])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_elementwise_add_cpasync(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    elementwise_add_cpasync(a, b, c, smem_layout, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


if __name__ == "__main__":
    print("Benchmarking elementwise_add")
    print("============================")
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")

    ms = triton.testing.do_bench(lambda: elementwise_add(A, B, C))
    print(f"elementwise_add: {ms:.2f} ms")

    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    ms = triton.testing.do_bench(lambda: elementwise_add_cpasync(A, B, C, smem_layout))
    print(f"elementwise_add_cpasync: {ms:.2f} ms")

# %%
# ```
# elementwise_add: 7.94 ms
# elementwise_add_cpasync: 2.95 ms
# ```
#
# Surprisingly, the cpasync version is already significantly faster. We picked
# a non-swizzled shared memory layout. Shared memory is organized such that
# consecutive 32-bit elements are stored in separate banks, up to 32 banks. On
# newer GPUs, banks are dual-ported, allowing them to service two 32-bit
# requests per cycle per warp. Any more than that causes the bank to serialize
# the shared memory accesses.
#
# Our register layout maps 32 threads per warp to consecutive elements, meaning
# even without swizzling, the shared memory load will not have bank conflicts.
# However, in other cases, picking the right shared memory layout is important
# for performance.

# %%
# We can pipeline our kernel by double or triple buffering the shared memory.
# For example, if we double-buffer, we can read to one buffer and write to the
# other in parallel.


@gluon.jit
def elementwise_add_pipelined_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
        smem_layout: gl.constexpr, num_buffers: gl.constexpr,  #
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    y_idx = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    xmask = (xoffs < xnumel)[:, None]

    # New: declare multi-buffered shared memory by adding a pipelining dimension
    # to the descriptors.
    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    copy_idx = 0
    read_idx = 0

    # Peel the `num_buffers-1` iterations from the inner loop to issue the first
    # set of copies.
    for _ in gl.static_range(num_buffers - 1):
        # Masking the loads by yoffs < ynumel will handle the case where there
        # are fewer blocks to copy than `num_buffers-1`.
        yoffs = copy_idx * YBLOCK + y_idx
        mask = xmask & (yoffs < ynumel)[None, :]
        cp.async_copy_global_to_shared(a_smem.index(copy_idx % num_buffers),  #
                                       a_ptrs + ystride_a * yoffs[None, :], mask)
        cp.async_copy_global_to_shared(b_smem.index(copy_idx % num_buffers),  #
                                       b_ptrs + ystride_b * yoffs[None, :], mask)
        cp.commit_group()
        copy_idx += 1

    # Inner loop iterations with overlapped copies and compute.
    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        # Issue the overlapped copy.
        yoffs = copy_idx * YBLOCK + y_idx
        mask = xmask & (yoffs < ynumel)[None, :]
        cp.async_copy_global_to_shared(a_smem.index(copy_idx % num_buffers),  #
                                       a_ptrs + ystride_a * yoffs[None, :], mask)
        cp.async_copy_global_to_shared(b_smem.index(copy_idx % num_buffers),  #
                                       b_ptrs + ystride_b * yoffs[None, :], mask)
        cp.commit_group()
        copy_idx += 1

        # Wait for `num_buffers-1` copies to complete, which is the last issued
        # copy. We can process that buffer.
        cp.wait_group(num_buffers - 1)

        a_val = a_smem.index(read_idx % num_buffers).load(layout)
        b_val = b_smem.index(read_idx % num_buffers).load(layout)
        c_val = a_val + b_val
        yoffs = read_idx * YBLOCK + y_idx
        mask = xmask & (yoffs < ynumel)[None, :]
        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)
        read_idx += 1

    # Peeled iterations to drain the pipeline.
    for i in gl.static_range(num_buffers - 1):
        cp.wait_group(num_buffers - 2 - i)

        a_val = a_smem.index(read_idx % num_buffers).load(layout)
        b_val = b_smem.index(read_idx % num_buffers).load(layout)
        c_val = a_val + b_val
        yoffs = read_idx * YBLOCK + y_idx
        mask = xmask & (yoffs < ynumel)[None, :]
        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)
        read_idx += 1


def elementwise_add_pipelined(A, B, C, XBLOCK=32, YBLOCK=64, num_buffers=2):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    return elementwise_add_pipelined_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, smem_layout, num_buffers)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 120)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 64)])
@pytest.mark.parametrize("num_buffers", [1, 2, 3])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_elementwise_add_pipelined(xnumel, ynumel, XBLOCK, YBLOCK, num_buffers):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add_pipelined(a, b, c, XBLOCK, YBLOCK, num_buffers)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


if __name__ == "__main__":
    ms = triton.testing.do_bench(lambda: elementwise_add_pipelined(A, B, C, num_buffers=2))
    print(f"elementwise_add_pipelined (double buffer): {ms:.2f} ms")
    ms = triton.testing.do_bench(lambda: elementwise_add_pipelined(A, B, C, num_buffers=3))
    print(f"elementwise_add_pipelined (triple buffer): {ms:.2f} ms")

# %%
# ```
# elementwise_add_pipelined (double buffer): 2.79 ms
# elementwise_add_pipelined (triple buffer): 2.79 ms
# ```
#
# Pipelining with async copy yields a modest speedup. But notice that increasing
# the number of buffers does not change the result, suggesting that there is a
# bottleneck somewhere else. Pipelining becomes more important when the compute
# in the inner loop is more expensive.
#
# One of the major issues getting in the way of more performance is register
# pressure. For each element, we need to store the 32-bit result, compute a
# 64-bit address, and the mask. With two inputs, this results in a lot of
# registers, where the maximum registers per thread is 256. This is why we used
# a small [32, 64] block size for the kernel. In the next tutorial, we will
# convert tensor descriptors and TMAs, and how they can help reduce register
# pressure at the cost of addressing flexibility.
#
# Main takeaways:
#
# - Asynchronous instructions allow overlapping memory operations with compute.
# - Async copies enable asynchronous global memory reads, and are tracked with
#   commit groups.
# - Software pipelining is a loop optimization technique that is used to overlap
#   async operations.
# - Shared memory layouts affect performance just like tensor layouts. It is
#   important to choose a layout that minimizes bank conflicts, which is also a
#   function of the register layout.
