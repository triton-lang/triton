"""
TMA in Gluon
============

The main problem with global memory accesses is register pressure. For each
`LDG.E` or `STG.E`, we need to compute the 64-bit address, compute the mask if
needed, and store the result in registers. Vectorization can reduce register
pressure, but the problem remains.

On Hopper and newer, TMA (Tensor Memory Accelerator) is a hardware feature for
addressing N-dimensional arrays in global memory. TMAs trade the addressing
flexibility of regular global memory instructions for a more concise address
representation -- the "tensor descriptor".

TMAs memory transactions are also handled by a separate hardware path called the
"async proxy". This boosts the performance of global memory accesses, but it
adds an additional layer of synchronization needed.

In this tutorial, we will cover how to use TMAs in Gluon, demonstrate how they
boost performance, and how to pipeline with TMAs.
"""

import pytest
import torch
import triton
import importlib
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared

# Re-use utilities from the previous tutorial.
t3 = importlib.import_module("03-async-copy")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

# %%
# TMA is used through objects called "tensor descriptors". Tensor descriptors
# live in global memory and contain the shape, strides, base pointer, layout,
# and other information about the tensor. TMA reads and writes are fundamentally
# async, and we will need "mbarrier" objects to synchronize them.
#
# Kernels that use TMAs accept descriptors as kernel arguments, which we can use
# to issue async tranfers:


@gluon.jit
def memcpy_1d_tma_kernel(in_desc, out_desc, XBLOCK: gl.constexpr):
    # We don't need to pass the tensor strides because they are stored in the
    # tensor descriptors
    pid = gl.program_id(0)

    # Each tensor descriptor contains a shared memory layout. Data is
    # transferred between global and shared memory according to that layout.
    smem_layout: gl.constexpr = in_desc.layout
    smem = gl.allocate_shared_memory(in_desc.dtype, [XBLOCK], smem_layout)

    # Completion of async TMA reads are tracked by mbarrier objects. These
    # are 64-bit objects that live in shared memory.
    #
    # An mbarrier is initialized with a count. Each time a mbarrier is
    # "arrived" on, the count is decremented. When the count reaches 0, the
    # current phase of the mbarrier is marked as complete and it moves to the
    # next phase. The mbarrier only tracks the state of the current and
    # previous phase. This is important, because if an mbarrier's phase races
    # too far ahead, its waiter will become out of sync.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    # Completion of an async TMA arrives on an mbarrier once. Thus, initialize
    # the mbarrier with a count of 1 so its phase will complete when the TMA is
    # complete.
    mbarrier.init(bar, count=1)

    # Tensor descriptors have an associated block shape. Each TMA request will
    # copy one block of the tensor descriptor. The coordinates of the TMA
    # request are specified as offsets to the beginning of the block. Masking
    # of out-of-bounds reads and writes is handled automatically by TMAs, using
    # the shape specified on the tensor descriptor.
    gl.static_assert(in_desc.block_type == out_desc.block_type)
    gl.static_assert(in_desc.layout == out_desc.layout)

    # Track completion of the TMA read based on the number of bytes copied.
    # mbarrier.expect sets the number of outstanding bytes tracked by the
    # mbarrier. If we pass the barrier to the TMA copy, it will atomically
    # decrement the number of outstanding bytes as transactions complete. When
    # it reaches 0, the mbarrier is arrived on once.
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(in_desc, [pid * XBLOCK], bar, smem)

    # Wait for completion of the read. We query the completion state of the
    # mbarrier using the parity of the phase, i.e. either 0 or 1. mbarriers are
    # initialized to parity 1 complete, so we wait for parity 0.
    mbarrier.wait(bar, phase=0)

    # When we are done using the mbarrier, we need to invalidate it.
    mbarrier.invalidate(bar)

    # Since the TMA store reads from shared memory, we don't even need to load
    # the result into registers. We can just store the result directly.
    tma.async_copy_shared_to_global(out_desc, [pid * XBLOCK], smem)

    # Unlike TMA reads, the completion of TMA stores is tracked by commit
    # groups, just like async copies. Each async TMA store is implicitly
    # committed to an async store group. We can wait until there are at most
    # `pendings` outstanding TMA stores using `store_wait`. Note that the commit
    # groups for async copy and async TMA stores are separate.
    tma.store_wait(pendings=0)


def memcpy_1d_tma(input, output, XBLOCK=8192):
    assert input.shape == output.shape

    # The layout for a tensor descriptor is always an NVMMASharedLayout. We can
    # use this helper to grab the default NVMMASharedLayout, but sometimes you
    # might need a different layout.
    block_shape = [XBLOCK]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    # Wrap the tensors in tensor descriptors.
    in_desc = TensorDescriptor.from_tensor(input, block_shape, layout)
    out_desc = TensorDescriptor.from_tensor(output, block_shape, layout)

    grid = (triton.cdiv(input.numel(), XBLOCK), )
    # Our kernel only uses scalars, so just a single warp is enough.
    memcpy_1d_tma_kernel[grid](in_desc, out_desc, XBLOCK, num_warps=1)


@pytest.mark.parametrize("XBLOCK", [64])
@pytest.mark.parametrize("xnumel", [40, 500])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_memcpy_1d_tma(XBLOCK, xnumel):
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy_1d_tma(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# Let's rewrite the pipelined elementwise add kernel using TMAs. The structure
# of the kernel is almost the same. However, we now need to allocate one
# mbarrier per buffer to track completion of the reads. We will also use TMA for
# the store, meaning we need to allocate more shared memory for it.
#
# TMAs access shared memory through a different hardware called the "async
# proxy". However, reading and writing shared memory from registers accesses it
# through the "generic proxy". Memory operations across proxies are not ordered,
# so we have to use `fence_async_shared` to establish ordering. Here are some
# examples of hazards that require fences:
#
# ```python
# value = smem.load()
# fence_async_shared()
# tma.async_copy_global_to_shared(desc, [0, 0], bar, smem)
# ```
#
# Without the fence, async_copy_global_to_shared can start copying into `smem`
# while the shared memory load is still in progress.
#
# ```python
# smem.store(value)
# fence_async_shared()
# tma.async_copy_shared_to_global(desc, [0, 0], smem)
# ```
#
# Without the fence, async_copy_shared_to_global can start copying from `smem`
# before the shared memory store is complete.
#
# Note that certain cases imply total completion of a memory transaction and
# do not require a fence. For example, waiting on the result of a TMA load:
#
# ```python
# tma.async_copy_global_to_shared(desc, [0, 0], bar, smem)
# mbarrier.wait(bar, phase=0)
# value = smem.load()
# ```
#
# fence_async_shared is not needed because after the mbarrier.wait on the TMA
# read barrier, we know it has finished writing into shared memory via the async
# proxy. Thus the read via the generic proxy will be ordered after. This applies
# specifically to the TMA read barrier, a fence is still needed in this case:
#
# ```python
# smem.store(value)
# mbarrier.arrive(bar, count=1)
# mbarrier.wait(bar, phase=0)
# fence_async_shared()
# tma.async_copy_shared_to_global(desc, [0, 0], smem)
# ```


@gluon.jit
def issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr):
    # Track completion of both TMA reads with the same mbarrier.
    yoff = copy_index * YBLOCK
    bar = bars.index(copy_index % num_buffers)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [xoff, yoff], bar, a_smem.index(copy_index % num_buffers))
    tma.async_copy_global_to_shared(b_desc, [xoff, yoff], bar, b_smem.index(copy_index % num_buffers))
    return copy_index + 1


@gluon.jit
def perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout: gl.constexpr, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr):
    # Wait for the copy from num_buffers-1 iterations ago to complete.
    read_phase = read_index // num_buffers & 1
    mbarrier.wait(bars.index(read_index % num_buffers), read_phase)
    a_val = a_smem.index(read_index % num_buffers).load(layout)
    b_val = b_smem.index(read_index % num_buffers).load(layout)
    c_val = a_val + b_val
    yoff = read_index * YBLOCK
    # Pipeline the store by rotating the store wait.
    tma.store_wait(pendings=0)
    c_smem.store(c_val)
    fence_async_shared()
    # Issue the store without waiting for it.
    tma.async_copy_shared_to_global(c_desc, [xoff, yoff], c_smem)
    return read_index + 1


@gluon.jit
def elementwise_add_tma_kernel(  #
        a_desc, b_desc, c_desc, xnumel, ynumel,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoff = pid * XBLOCK

    dtype: gl.constexpr = a_desc.type.block_type.element_ty
    # Allocate multibuffered shared memory for the input buffers.
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], b_desc.layout)

    # Allocate shared memory for the TMA store.
    c_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], c_desc.layout)

    # Allocate mbarriers to track completion of the TMA reads.
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)

    copy_index = 0
    read_index = 0

    for _ in gl.static_range(num_buffers - 1):
        copy_index = issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK, num_buffers)

    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        copy_index = issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK, num_buffers)
        read_index = perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout, YBLOCK, num_buffers)

    for _ in gl.static_range(num_buffers - 1):
        read_index = perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout, YBLOCK, num_buffers)

    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(bars.index(i))

    # Wait for the last store to complete.
    tma.store_wait(pendings=0)


def elementwise_add_tma(a, b, c, XBLOCK=32, YBLOCK=64, num_buffers=2):
    assert a.shape == b.shape == c.shape
    xnumel, ynumel = a.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )

    block_shape = [XBLOCK, YBLOCK]
    # TMA descriptors require NVMMASharedLayout.
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    # The strides of TMA descriptors must be 16-byte aligned.
    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)
    elementwise_add_tma_kernel[grid](a_desc, b_desc, c_desc, xnumel, ynumel, XBLOCK, YBLOCK, num_buffers)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 120)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 64)])
@pytest.mark.parametrize("num_buffers", [1, 2, 3])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_elementwise_add_pipelined(xnumel, ynumel, XBLOCK, YBLOCK, num_buffers):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add_tma(a, b, c, XBLOCK, YBLOCK, num_buffers)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


# %%
# Let's compare the pipelined TMA kernel against the pipelined async copy kernel
# from the previous tutorial.

if __name__ == "__main__":
    print("Benchmarking elementwise_add")
    print("============================")
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")

    XBLOCK = 32
    YBLOCK = 64
    num_buffers = 2

    ms = triton.testing.do_bench(lambda: t3.elementwise_add_pipelined(A, B, C, XBLOCK, YBLOCK, num_buffers))
    print(f"elementwise_add_pipelined: {t3.get_throughput(ms, C):.2f} TB/s")

    ms = triton.testing.do_bench(lambda: elementwise_add_tma(A, B, C, XBLOCK, YBLOCK, num_buffers))
    print(f"elementwise_add_tma: {t3.get_throughput(ms, C):.2f} TB/s")

# %%
# ```
# elementwise_add_pipelined: 4.20 TB/s
# elementwise_add_tma: 5.50 TB/s
# ```
#
# Switching to TMAs already yields a large performance boost.
#
# Since our kernel has more register room, we can increase the block size. In
# practice, peak register usage will remain low, because the compiler will
# interleave the smem load, add, and smem store in the inner loop. The main
# limitation to block size is the amount of shared memory.
#
# Each SM has 228 KB of shared memory. If we use 128x128xf32 blocks, we don't
# have enough shared memory to double buffer the inputs. If we use 64x128xf32
# triple buffering uses 224 KB, just barely fitting.

if __name__ == "__main__":
    XBLOCK = 64
    YBLOCK = 128
    num_buffers = 3
    ms = triton.testing.do_bench(lambda: elementwise_add_tma(A, B, C, XBLOCK, YBLOCK, num_buffers))
    print(f"elementwise_add_tma (64x128x3): {t3.get_throughput(ms, C):.2f} TB/s")

# %%
# ```
# elementwise_add_tma (64x128x3): 5.90 TB/s
# ```
#
# We get another modest speedup by increasing the block size and pipeline depth.
#
# Main takeaways:
#
# - TMAs use a separate, often faster, hardware path for transferring between
#   shared and global memory.
# - TMA instructions are asynchronous; we use mbarriers to track completion of
#   reads and commit groups to track completion of stores.
# - TMAs reduce register pressure but restrict addressing flexibility. Depending
#   on the layout of global tensors, it may not be possible to use TMAs.
# - TMA instructions can be pipelined, but require explicit synchronization
#   between the async proxy and generic proxy.
