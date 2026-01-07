"""
Native TMA Gather and Scatter
=============================

This tutorial explains how to use the native async TMA gather and scatter
operations available on Blackwell GPUs. Native gather and scatter operations on
Blackwell GPUs are implemented in the `gl.nvidia.blackwell.tma.async_gather` and
`gl.nvidia.blackwell.tma.async_scatter` functions respectively.

TMA gather and scatter operations only support 2D tensor descriptors, where the
first dimension of the block shape must be 1. Gather accepts a 2D tensor
descriptor, a 1D tensor of row offsets, and a scalar column offset. If the block
shape of the 2D tensor descriptor is `[1, BLOCK_Y]`, gather performs the
following operation returning a 2D tensor:

```python
out = tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y]
```

Where `out.shape` is `(x_offsets.shape[0], BLOCK_Y)`. In other words, gather
loads `x_offsets.shape[0]` separately-indexed rows of size `BLOCK_Y` from the
tensor descriptor, starting at `y_offset`.

Scatter accepts a 2D tensor descriptor, a 1D tensor of row offsets, a scalar
column offset, and a 2D source tensor. If the block shape of the 2D tensor
descriptor is `[1, BLOCK_Y]`, scatter performs the following operation:

```python
tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y] = src
```

Where `src.shape` must be `(x_offsets.shape[0], BLOCK_Y)`. In other words,
scatter writes `src` to the tensor descriptor starting at `y_offset` but to
separately-indexed rows of size `BLOCK_Y`.

Like `async_copy_global_to_shared` and `async_copy_shared_to_global`,
`async_gather` and `async_scatter` access shared memory through the async
proxy, so fences need to be inserted as appropriate.
"""

import sys
import pytest
import torch
import triton
import importlib
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton._C.libtriton import ir, gluon_ir

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (tma, mbarrier, fence_async_shared)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")

# Re-use utilities from the previous tutorials.
t7 = importlib.import_module("07-persistence")

# %%
# `async_gather` and `async_scatter` impose constraints on the layout of the 1D
# row offsets tensor.
#
# Specifically, suppose the row offset tensor is divided into chunks of 4
# consecutive elements, then the layout must map each chunk to consecutive
# registers in the same thread. In addition, the chunks must be broadcasted
# across all threads in the same warp, i.e. all threads in the same warp must
# contain the same data.
#
# These constraints arise from the underlying `gather4` and `scatter4` PTX
# instructions used by `async_gather` and `async_scatter`. Each is a warp-level
# instruction that loads to or stores from 4 consecutive rows in shared memory.
#
# For example, the following layout is always valid for any row offsets tensor:
#
# ```python
# gl.SliceLayout(
#     dim=0,
#     parent=gl.BlockedLayout(
#         size_per_thread=[1, 4],
#         threads_per_warp=[num_threads_per_warp, 1],
#         warps_per_cta=[1, num_warps],
#         order=[1, 0],
#     ),
# )
# ```
#
# Recall from `02-layouts` that the parent `BlockedLayout` specified above will
# tile the dim=1 into chunks of 4 consecutive elements mapped to 4 consecutive
# registers in the same thread, and then tile dim=1 along all the warps. dim=0
# is only tiled across the threads in a warp, but when we take the `SliceLayout`
# along dim=0, all threads in a warp will map to the same 4 consecutive
# elements.
#
# Note that transposing the blocked layout and slicing along dim=1 yields an
# identical layout:
#
# ```python
# gl.SliceLayout(
#     dim=1,
#     parent=gl.BlockedLayout(
#         size_per_thread=[4, 1],
#         threads_per_warp=[1, num_threads_per_warp],
#         warps_per_cta=[num_warps, 1],
#         order=[0, 1],
#     ),
# )
# ```
#
# These are not the only valid layouts for the row offsets tensor. For example,
# given a row offset tensor with the shape `(BLOCK_X)`, a valid layout could be:
#
# ```python
# gl.BlockedLayout(
#     size_per_thread=[BLOCK_X]
#     threads_per_warp=[num_threads_per_warp],
#     warps_per_cta=[num_warps],
#     order=[0],
# )
# ```
#
# This layout is valid because all elements are mapped consecutively to the
# registers in all of the threads, but it is less efficient; because all warps
# have the same data, the compiler will pick only warp 0 to emit all the
# instructions. For example, if `BLOCK_X=256`, warp 0 will execute
# `256 // 4 = 64` gather4 instructions while the rest of the warps do nothing,
# whereas the sliced layouts above will spread the work across all warps,
# resulting in `256 // 4 // 4 = 16` gather4 instructions per warp, assuming
# there are 4 warps.
#
# In general, a layout is valid if its linear layout representation satisfies:
# - The first 2 register bases must be [1] and [2]
# - The lane bases must all be [0]

# %%
# Let's write a tool to convert any layout to a linear layout to help illustrate
# this concept.


def to_linear_layout(layout, shape):
    context = ir.context()
    ir.load_dialects(context)
    builder = gluon_ir.GluonOpBuilder(context)
    return builder.to_linear_layout(layout._to_ir(builder), shape)


if __name__ == "__main__":
    num_threads_per_warp = 32
    num_warps = 4
    BLOCK_X = 256

    layout = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[num_threads_per_warp, 1],
            warps_per_cta=[1, num_warps],
            order=[1, 0],
        ),
    )
    # DistributedLinearLayout(
    #     reg_bases=[[1], [2], [16], [32], [64], [128]],
    #     lane_bases=[[0], [0], [0], [0], [0]],
    #     warp_bases=[[4], [8]],
    #     block_bases=[],
    #     shape=[256]
    # )
    print(to_linear_layout(layout, [256]))

    layout = gl.BlockedLayout(
        size_per_thread=[BLOCK_X],
        threads_per_warp=[num_threads_per_warp],
        warps_per_cta=[num_warps],
        order=[0],
    )
    # DistributedLinearLayout(
    #     reg_bases=[[1], [2], [4], [8], [16], [32], [64], [128]],
    #     lane_bases=[[0], [0], [0], [0], [0]],
    #     warp_bases=[[0], [0]],
    #     block_bases=[],
    #     shape=[256]
    # )
    print(to_linear_layout(layout, [256]))

    # Notice how in the two layouts above, the first two register bases are
    # indeed [1] and [2], and all lane bases are [0]. The different is the
    # second layout's warp bases are all [0], which leads to inefficient code
    # generation for `async_gather` and `async_scatter`.

    # Here is an example of an invalid layout:
    layout = gl.BlockedLayout(
        size_per_thread=[4],
        threads_per_warp=[num_threads_per_warp],
        warps_per_cta=[num_warps],
        order=[0],
    )
    # DistributedLinearLayout(
    #     reg_bases=[[1], [2]],
    #     lane_bases=[[4], [8], [16], [32], [64]],
    #     warp_bases=[[128], [0]],
    #     block_bases=[],
    #     shape=[256]
    # )
    print(to_linear_layout(layout, [256]))

    # This layout is invalid because the lane bases are not all [0].

# %%
# Let's demonstrate how to use `async_gather` and `async_scatter` by writing
# simple kernels. Note that both `async_gather` and `async_scatter` have several
# additional constraints. As we already mentioned, the tensor descriptor must be
# 2D with a block shape in the form of `[1, BLOCK_Y]`. Additionally:
#
# - The row offset tensor must have at least 8 elements. I.e. at least 8 rows
#   must be loaded by async gather or stored by async scatter.
#
# - There is a minimum number of columns based on the dtype. Specifically,
#   `BLOCK_Y >= (32 // tensor_desc.dtype.primitive_bitwidth) * 8`. For example,
#   a `float16` tensor descriptor must have `BLOCK_Y >= 16`.
#
# - The `y_offset` must be aligned to 16 bytes. I.e.
#   `y_offset % (16 // (tensor_desc.dtype.primitive_bitwidth // 8)) == 0`.
#   For example, for `float16`, `y_offset` must be a multiple of 8. This is checked
#   at runtime by the hardware, and if `y_offset` is not aligned to 16 bytes, the
#   CUDA driver will emit an illegal instruction error.
#
# - Elements of `x_offsets` may be out-of-bounds, in which case the loaded rows of
#   `async_gather` will be all zeros, and stored rows in `async_scatter` will be ignored.
#
# - `y_offset` can be out-of-bounds. Row elements in `y_offset:y_offset + BLOCK_Y` that
#   are out-of-bounds will be loaded as zeros by `async_gather` and ignored when stored by `async_scatter`.
#
# - `x_offsets` elements and `y_offset` may only be negative for `async_gather`. If `async_scatter`
#   receives negative row of column offsets, the CUDA driver will emit an illegal instruction error.


# The kernel computes `out = tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y]`.
@gluon.jit
def async_gather_kernel(out_ptr, out_stride_x, out_stride_y, tensor_desc, x_offsets_ptr, y_offset,
                        BLOCK_X: gl.constexpr):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]

    # Load the offsets using a coalesced layout for efficient load vectorization.
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))

    # Convert the offsets layout to a slice layout that satisfies the constraints for `async_gather`.
    offsets_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    # `async_gather` loads the rows from a tensor descriptor and writes them into shared memory.
    # The layout of the shared memory descriptor must match the shared memory layout of the tensor descriptor.
    smem_dest = gl.allocate_shared_memory(tensor_desc.dtype, [BLOCK_X, BLOCK_Y], tensor_desc.layout)

    # `async_gather` is an asynchronous operation that uses an mbarrier to track its completion.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    # Invoke `mbarrier.expect` on the mbarrier with the number of bytes to be loaded.
    mbarrier.expect(bar, BLOCK_X * tensor_desc.block_type.nbytes)

    # Issue the async gather and wait.
    tma.async_gather(tensor_desc, x_offsets, y_offset, barrier=bar, result=smem_dest)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Write the result using a coalesced layout.
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    out = smem_dest.load(coalesced_2d_layout)

    indices_x = gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None] * out_stride_x
    indices_y = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :] * out_stride_y
    gl.store(out_ptr + indices_x + indices_y, out)


def async_gather(input, x_offsets, y_offset, BLOCK_X, BLOCK_Y):
    gl_dtype = getattr(gl, str(input.dtype).split('.')[1])
    # When picking the shared memory layout, we use the dimensions of the shared
    # memory descriptor, which will be [BLOCK_X, BLOCK_Y]. But the block shape of the
    # tensor descriptor must still be [1, BLOCK_Y] to be used with async gather.
    layout = gl.NVMMASharedLayout.get_default_for([BLOCK_X, BLOCK_Y], gl_dtype)
    tensor_desc = TensorDescriptor.from_tensor(input, [1, BLOCK_Y], layout)
    out = torch.empty((BLOCK_X, BLOCK_Y), dtype=input.dtype, device="cuda")
    async_gather_kernel[(1, )](out, *out.stride(), tensor_desc, x_offsets, y_offset, BLOCK_X)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("BLOCK_X", [8, 128])
@pytest.mark.parametrize("BLOCK_Y", [16, 128])
@pytest.mark.parametrize("y_offset", [-16, 0, 48, 1000])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_async_gather(BLOCK_X, BLOCK_Y, y_offset, dtype, X_MAX=1024, Y_MAX=1024):
    torch.manual_seed(0)

    input = torch.randn((X_MAX, Y_MAX), dtype=dtype, device="cuda")
    # Span row offsets from negative to out-of-bounds to test the masked load behavior.
    x_offsets = torch.linspace(-X_MAX, 2 * X_MAX, BLOCK_X, dtype=torch.int32, device="cuda")
    # Randomly shuffle the row offsets.
    x_offsets = x_offsets[torch.randperm(BLOCK_X, device="cuda")]

    out = async_gather(input, x_offsets, y_offset, BLOCK_X, BLOCK_Y)

    # Mask out-of-bounds and negative row offsets.
    x_offsets = torch.where(x_offsets >= X_MAX, -1, x_offsets)
    mask = (x_offsets >= 0).unsqueeze(1)

    # Mask out-of-bounds and negative column offsets by padding with zeros.
    y_lo, y_hi = max(0, y_offset), min(y_offset + BLOCK_Y, Y_MAX)
    ref = input[x_offsets, y_lo:y_hi] * mask
    lo_zeros = torch.zeros(BLOCK_X, y_lo - y_offset, dtype=dtype, device="cuda")
    hi_zeros = torch.zeros(BLOCK_X, y_offset + BLOCK_Y - y_hi, dtype=dtype, device="cuda")
    ref = torch.cat((lo_zeros, ref, hi_zeros), dim=1)

    torch.testing.assert_close(out, ref, atol=0, rtol=0)


# %%
# The CUDA driver will emit an illegal instruction error if `y_offset` is not
# aligned to 16 bytes for both `async_gather` and `async_scatter`, or if negative
# row or column offsets are used for `async_scatter`.

if __name__ == "__main__":
    # Note that any illegal instruction errors will corrupt the CUDA context in current Python
    # process, which prevents executing any other code. Guard each of these examples with a
    # flag so that only 1 is executed at a time.
    if len(sys.argv) > 1 and sys.argv[1] == "test_illegal_gather":
        try:
            # y_offset=2 is not 16-byte aligned for bfloat16
            test_async_gather(BLOCK_X=128, BLOCK_Y=128, y_offset=2, dtype=torch.bfloat16)
        except RuntimeError as e:
            assert "an illegal instruction was encountered" in str(e)
            raise

# %%
# Illegal instruction errors can be frustrating to debug. They typically occur
# because an executed instruction does not match some runtime invariants. To
# figure out which instruction is causing the error, you can run the program
# inside the debugger `cuda-gdb`. For example, if we run
#
# ```bash
# cuda-gdb --args python python/tutorials/gluon/09-tma-gather-scatter.py test_illegal_gather
# ```
#
# Send `r` to run the program, and the debugger will break on the instruction
# that triggered the illegal instruction error:
#
# ```
# CUDA Exception: Warp Illegal Instruction
# The exception was triggered at PC 0x628fbe590  async_gather_kernel  (09-tma-gather-scatter.py:245)
#
# Thread 1 "python" received signal CUDA_EXCEPTION_4, Warp Illegal Instruction.
# [Switching focus to CUDA kernel 0, grid 9, block (0,0,0), thread (96,0,0), device 0, sm 148, warp 0, lane 0]
# 0x0000000628fbe700 in async_gather_kernel<<<(1,1,1),(128,1,1)>>> () at /root/code/triton/python/tutorials/gluon/09-tma-gather-scatter.py:245
# 245         tma.async_gather(tensor_desc, x_offsets, y_offset, barrier=bar, result=smem_dest)
# ```

# %%
# This kernel computes `tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y] = src`.


@gluon.jit
def async_scatter_kernel(tensor_desc, x_offsets_ptr, y_offset, src_ptr, src_stride_x, src_stride_y,
                         BLOCK_X: gl.constexpr):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]

    # Load the source using a coalesced layout for efficient load vectorization.
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    indices_x = gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None] * src_stride_x
    indices_y = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :] * src_stride_y
    src = gl.load(src_ptr + indices_x + indices_y)

    # Load the offsets using a coalesced layout for efficient load vectorization.
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))

    # Convert the offsets layout to a slice layout that satisfies the constraints for `async_scatter`.
    offsets_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    # `async_scatter` stores the rows to a tensor descriptor from shared memory.
    smem_src = gl.allocate_shared_memory(tensor_desc.dtype, [BLOCK_X, BLOCK_Y], tensor_desc.layout)
    smem_src.store(src)
    # An async fence is required between the store to shared memory and the async scatter.
    # Recall from `04-tma` that a fence is needed when using different proxies to access shared
    # memory (generic proxy for the store, and async proxy for the `async_scatter`).
    fence_async_shared()
    tma.async_scatter(tensor_desc, x_offsets, y_offset, smem_src)
    # Wait for the completion of the async scatter using `store_wait`.
    tma.store_wait(0)


def async_scatter(input, x_offsets, y_offset, src, BLOCK_X, BLOCK_Y):
    gl_dtype = getattr(gl, str(input.dtype).split('.')[1])
    # When picking the shared memory layout, we use the dimensions of the shared
    # memory descriptor, which will be [BLOCK_X, BLOCK_Y]. But the block shape of the
    # tensor descriptor must still be [1, BLOCK_Y] to be used with async scatter.
    layout = gl.NVMMASharedLayout.get_default_for([BLOCK_X, BLOCK_Y], gl_dtype)
    tensor_desc = TensorDescriptor.from_tensor(input, [1, BLOCK_Y], layout)
    async_scatter_kernel[(1, )](tensor_desc, x_offsets, y_offset, src, *src.stride(), BLOCK_X)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("BLOCK_X", [8, 128])
@pytest.mark.parametrize("BLOCK_Y", [16, 128])
@pytest.mark.parametrize("y_offset", [0, 48, 1000])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_async_scatter(BLOCK_X, BLOCK_Y, y_offset, dtype, X_MAX=1024, Y_MAX=1024):
    torch.manual_seed(0)

    input = torch.randn((X_MAX, Y_MAX), dtype=dtype, device="cuda")
    input_ref = input.clone()

    # Span row offsets from 0 to out-of-bounds to test the masked store behavior.
    x_offsets = torch.linspace(0, 2 * X_MAX, BLOCK_X, dtype=torch.int32, device="cuda")
    # Randomly shuffle the row offsets.
    x_offsets = x_offsets[torch.randperm(BLOCK_X, device="cuda")]

    src = torch.randn((BLOCK_X, BLOCK_Y), dtype=dtype, device="cuda")
    async_scatter(input, x_offsets, y_offset, src, BLOCK_X, BLOCK_Y)

    # Mask out-of-bounds row offsets.
    mask = x_offsets < X_MAX
    x_offsets = x_offsets[mask]
    src = src[mask]

    # Mask out-of-bounds column offsets.
    y_hi = min(y_offset + BLOCK_Y, Y_MAX)

    input_ref[x_offsets, y_offset:y_hi] = src[:, :y_hi - y_offset]
    torch.testing.assert_close(input, input_ref, atol=0, rtol=0)


# %%
# `async_gather` and `async_scatter` can be pipelined just like `async_copy_global_to_shared`
# and `async_copy_shared_to_global`. To demonstrate this, we will write a matmul kernel
# that has a fused gather and fused scatter along the M dimension:
# `out[out_scatter_indx, :] = X[X_gather_indx, :] @ W`.
#
# Recall in `06-tcgen05-mma` that we demonstrated how to write matmul kernels
# with `tcgen05_mma`. This example performs pipelining of the TMA loads, including `async_gather`,
# with `tcgen05_mma` and pipelining of the `async_scatter` with the persistent outer loop.
#
# In our blocked matmul kernrel with fused gather and scatter, for each tile of the output,
# we will load the M dimension offsets for the X tensor tile and the N dimension offsets for the W
# tensor tile via `gl.load` and schedule them sufficiently ahead of their use to account for the
# latency of the global loads.


@gluon.jit
def issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs,
                BLOCK_M: gl.constexpr, num_buffers: gl.constexpr, pred=True):
    # Load the M dimension offsets for the X tensor tile. We expect the load to be small
    # enough (no more than 128 elements) that we don't need to use a coalesced layout. Load directly into the layout
    # required by `async_gather` to avoid the layout conversion.
    gather_indx_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    offs_x_m = gl.load(X_gather_indx_ptr + off_m + gl.arange(0, BLOCK_M, gather_indx_layout))

    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)

    # The W tensor tile is loaded using a regular `async_copy_global_to_shared`.
    mbarrier.expect(bar, W_desc.block_type.nbytes + BLOCK_M * X_desc.block_type.nbytes)
    tma.async_gather(X_desc, offs_x_m, k, bar, x_bufs.index(index), pred)
    tma.async_copy_global_to_shared(W_desc, [k, off_n], bar, w_bufs.index(index), pred)
    return producer


@gluon.jit
def issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    b_index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(x_bufs.index(index), w_bufs.index(b_index))
    return consumer, mma


@gluon.jit
def matmul_fused_gather_scatter_kernel(X_desc, W_desc, out_desc, X_gather_indx_ptr, out_scatter_indx_ptr,
                                       BLOCK_M: gl.constexpr, SchedulerImpl: gl.constexpr, num_buffers: gl.constexpr):
    BLOCK_N: gl.constexpr = W_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = W_desc.block_type.shape[0]
    dtype: gl.constexpr = X_desc.dtype
    M = X_desc.shape[0]
    N = W_desc.shape[1]
    K = X_desc.shape[1]

    # Allocate shared memory for the input tiles.
    x_bufs = gl.allocate_shared_memory(dtype, [num_buffers, BLOCK_M, BLOCK_K], X_desc.layout)
    w_bufs = gl.allocate_shared_memory(dtype, [num_buffers, BLOCK_K, BLOCK_N], W_desc.layout)

    # Allocate shared memory for the output tile.
    out_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], out_desc.layout)

    # Initialize barriers for multibuffering the loads.
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = t7.MMAv5.initialize(dtype, BLOCK_M, BLOCK_N, gl.num_warps())
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    # Peeled inner loop prologue.
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, ki, bars, x_bufs, w_bufs,
                               BLOCK_M, num_buffers)
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs, BLOCK_M,
                           num_buffers)

    for _ in range(num_tiles):
        consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs,
                                   BLOCK_M, num_buffers)
            consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        # Load the M dimension offsets for the output tile. We expect the load to be small
        # enough (no more than 128 elements) that we don't need to use a coalesced layout.
        # Load directly into the layout required by `async_scatter` to avoid the layout conversion.
        scatter_indx_layout: gl.constexpr = gl.SliceLayout(
            0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
        out_offs_m = gl.load(out_scatter_indx_ptr + epilogue_off_m + gl.arange(0, BLOCK_M, scatter_indx_layout))

        # Peel the next prologue and fuse it with the pipeline drain loop.
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        # Predicate the peeled prologue instead of using a conditional.
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, ki, bars, x_bufs, w_bufs,
                                   BLOCK_M, num_buffers, pred)
            consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)
        k = BLOCK_K * (num_buffers - 2)
        producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs,
                               BLOCK_M, num_buffers)

        mma = mma.wait_num_outstanding(0)
        out, mma = mma.take_result()
        out = out.to(dtype)
        # Pipeline the async scatter by waiting for the previous store to complete.
        tma.store_wait(pendings=0)
        out_smem.store(out)
        fence_async_shared()
        tma.async_scatter(out_desc, out_offs_m, epilogue_off_n, out_smem)
    # Wait for the last async scatter to complete.
    tma.store_wait(pendings=0)


# %%
# We will pick reasonable defaults for the block sizes and number of load buffers.
# Tuning and optimizing the performance of this kernel is left as an exercise for the reader,
# as the primary objective of this tutorial is to demonstrate the use of async gather and scatter.
#
# The only alternative way to implement a matmul kernel with fused gather and
# scatter is to use async_copy (recall `03-async-copy`) or `gl.load` to load
# from global memory and `gl.store` to write to the output tensor in the
# epilogue. While these instructions provide more flexible indexing, they are
# much slower than TMA and async gather and scatter.
#
# One extra note: it is of course possible to use async gather and async scatter with
# warp-specialized kernels. Just keep in mind that because the row offsets is a tensor, you may want
# to give the load and epilogue partitions more than 1 warp to increase instruction issue throughput,
# particularly for the loads as they are on the critical path.


def matmul_fused_gather_scatter(X, X_gather_indx, W, out_scatter_indx, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
                                GROUP_SIZE_M=8, num_buffers=3):
    M = X.shape[0]
    N = W.shape[1]
    out = torch.empty((M, N), dtype=X.dtype, device="cuda")

    # Convert torch dtype to gluon dtype.
    dtype = getattr(gl, str(X.dtype).split('.')[1])
    # Setup descriptors for inputs and outputs.
    X_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], dtype)
    W_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], dtype)
    out_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], dtype)

    X_desc = TensorDescriptor.from_tensor(X, [1, BLOCK_K], X_desc_layout)
    W_desc = TensorDescriptor.from_tensor(W, [BLOCK_K, BLOCK_N], W_desc_layout)
    out_desc = TensorDescriptor.from_tensor(out, [1, BLOCK_N], out_desc_layout)

    # Persistent kernel grid.
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    SchedulerImpl = t7.GroupedPersistentTileScheduler(GROUP_SIZE_M)
    matmul_fused_gather_scatter_kernel[grid](X_desc, W_desc, out_desc, X_gather_indx, out_scatter_indx, BLOCK_M,
                                             SchedulerImpl, num_buffers)
    return out


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 2048), (4096, 4096, 4096)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N", [(128, 128), (128, 64)])
@pytest.mark.parametrize("BLOCK_K, num_buffers", [(128, 2), (64, 3)])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_matmul_fused_gather_scatter(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers):
    torch.manual_seed(0)

    # Randomize the gather indices.
    X_gather_indx = torch.arange(0, M, dtype=torch.int32, device="cuda")
    shfl = torch.randperm(M, device="cuda")
    X_gather_indx = X_gather_indx[shfl]

    # Randomize the scatter indices.
    out_scatter_indx = torch.arange(0, M, dtype=torch.int32, device="cuda")
    shfl = torch.randperm(M, device="cuda")
    out_scatter_indx = out_scatter_indx[shfl]

    X = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    out = matmul_fused_gather_scatter(X, X_gather_indx, W, out_scatter_indx, BLOCK_M, BLOCK_N, BLOCK_K,
                                      num_buffers=num_buffers)

    out_ref = torch.empty_like(out)
    out_ref[out_scatter_indx, :] = X[X_gather_indx, :] @ W
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-3)


# %%
# The main takeaway from this tutorial is understanding how to use `async_gather`
# and `async_scatter`. These instructions provide a middle-ground between
# block DMAs like `async_copy_global_to_shared` and `async_copy_shared_to_global`
# and regular global loads and stores (`gl.load` and `gl.store`) by allowing
# separately-indexed columns while maintaining the performance of TMAs.
#
# Keep in mind the following:
# - `async_gather` and `async_scatter` are typically faster than `gl.load` and
#   `gl.store` when they can be used, but this is not always the case. Plus, TMA
#   instructions use shared memory.
# - Sometimes using `async_gather` or `async_scatter` instead of block DMA
#   instructions like `async_copy_global_to_shared` and `async_copy_shared_to_global`
#   is actually faster, but these situations are rare.
#
# In general, you should consider these instructions when writing kernels and
# experiment to see what is the best way to write a kernel.
