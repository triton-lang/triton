"""
Async Copy in Gluon
===================

Modern GPUs provide asynchronous instructions for long-running operations like
global memory reads and writes. Asynchronous operations allow overlapping memory
transactions with compute, also known as "pipelining".

Asynchronous instructions vary by GPU vendor and architecture, so this tutorial
focuses on NVIDIA GPUs. On NVIDIA GPUs, async copies transfer data between
global memory and shared memory, unlike `gl.load` and `gl.store` which
directly write to and read from the register file.
"""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp


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
# issuing the copy and performing compute before waiting on it. Let's use an
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
# Let's rewrite the kernel to use async copies without pipelining, which will
# make it more obvious how we will pipeline the inner loop. Let's parameterize
# the kernel over the shared memory layout to see how it can affect performance.


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


def get_throughput(ms, C):
    # Because this kernel is memory-bound, we will measure bandwidth.
    tbytes = (3 * C.numel() * C.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


if __name__ == "__main__":
    print("Benchmarking elementwise_add")
    print("============================")
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")

    ms = triton.testing.do_bench(lambda: elementwise_add(A, B, C))
    print(f"elementwise_add: {get_throughput(ms, C):.2f} TB/s")

    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    ms = triton.testing.do_bench(lambda: elementwise_add_cpasync(A, B, C, smem_layout))
    print(f"elementwise_add_cpasync: {get_throughput(ms, C):.2f} TB/s")

# %%
# ```
# elementwise_add: 1.48 TB/s
# elementwise_add_cpasync: 3.97 TB/s
# ```
#
# Surprisingly, the cpasync version is already significantly faster. We picked
# a non-swizzled shared memory layout. Shared memory is organized such that
# consecutive 32-bit elements are stored in separate banks, up to 32 banks. On
# newer GPUs, banks are dual-ported, allowing them to service two 32-bit
# requests per cycle per warp. Any more than that causes the bank to serialize
# the shared memory accesses.
#
# Our register layout maps 32 threads per warp to consecutive 32-bit elements,
# meaning even without swizzling, the shared memory load will not have bank
# conflicts. In other cases, like with 16-bit or 8-bit elements, swizzling and
# vector length is more important to reduce bank conflicts.

# %%
# Software pipelining is an optimization technique for hiding the latencies of
# operations that execute asynchronously with respect to each other. If we
# prefetch the loads of the next operands before the current add, we can overlap
# it with the add and store. This requires multi-buffering shared memory, so it
# can be used by both the load and the add at the same time.
#
# Based on the relative latencies of the operations, we can determine the
# "pipeline depth". This is the number of prefetched loads in-flight. For
# example, if a load takes 3 times as long as the add, we should pipeline with
# depth 3 so each load has time to complete before the operands are needed.


@gluon.jit
def issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    # Masking the loads by yoffs < ynumel will handle the case where there
    # are fewer blocks to copy than `num_buffers-1`.
    yoffs = copy_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    cp.async_copy_global_to_shared(a_smem.index(copy_idx % num_buffers),  #
                                   a_ptrs + ystride_a * yoffs[None, :], mask)
    cp.async_copy_global_to_shared(b_smem.index(copy_idx % num_buffers),  #
                                   b_ptrs + ystride_b * yoffs[None, :], mask)
    cp.commit_group()
    return copy_idx + 1


@gluon.jit
def perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr, layout: gl.constexpr):
    a_val = a_smem.index(read_idx % num_buffers).load(layout)
    b_val = b_smem.index(read_idx % num_buffers).load(layout)
    c_val = a_val + b_val
    yoffs = read_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)
    return read_idx + 1


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

    # Peel the `num_buffers-1` iterations from the inner loop to prefetch the
    # first set of copies, filling our pipeline.
    for _ in gl.static_range(num_buffers - 1):
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                               YBLOCK, num_buffers)

    # Inner loop iterations with overlapped copies and compute. This is the
    # steady state of the pipeline.
    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        # Issue the overlapped copy.
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                               YBLOCK, num_buffers)

        # Wait for `num_buffers-1` copies to complete, which is the last issued
        # copy. We can process that buffer.
        cp.wait_group(num_buffers - 1)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK, num_buffers,
                               layout)

    # Peeled iterations to drain the pipeline.
    for i in gl.static_range(num_buffers - 1):
        cp.wait_group(num_buffers - 2 - i)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK, num_buffers,
                               layout)


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
    print(f"elementwise_add_pipelined (double buffer): {get_throughput(ms, C):.2f} TB/s")
    ms = triton.testing.do_bench(lambda: elementwise_add_pipelined(A, B, C, num_buffers=3))
    print(f"elementwise_add_pipelined (triple buffer): {get_throughput(ms, C):.2f} TB/s")

# %%
# ```
# elementwise_add_pipelined (double buffer): 4.20 TB/s
# elementwise_add_pipelined (triple buffer): 4.20 TB/s
# ```
#
# Pipelining with async copy yields a modest speedup. But notice that increasing
# the number of buffers further does not yield more performance, confirming that
# this kernel is memory-bound.
#
# One of the major issues getting in the way of more performance is register
# pressure. For each element, we need to store the 32-bit result, compute a
# 64-bit address, and the mask. With two inputs, this results in a lot of
# registers, where the maximum registers per thread is 256. This is why we used
# a small [32, 64] block size for the kernel. In the next tutorial, we will
# convert tensor descriptors and TMAs, and see how they can help reduce register
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
