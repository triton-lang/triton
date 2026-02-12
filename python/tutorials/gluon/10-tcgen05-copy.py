"""
TCGen05 Copy Instruction
========================

This tutorial will cover the `tcgen05_copy` instruction: how to use it and its
applications.

The `tcgen05_copy` instruction is an asynchronous tensorcore operation that copies
data from shared memory to tensor memory. The completion of `tcgen05_copy` is
tracked with `tcgen05_commit` on an mbarrier just like `tcgen05_mma`. The
completion of a single or multiple `tcgen05_copy` operations can be tracked by a
single `tcgen05_commit`:

```python
tcgen05_copy(lhs_smem, lhs_tmem)
tcgen05_copy(acc_smem, acc_tmem)
tcgen05_commit(bar)
mbarrier.wait(bar, phase=phase)
acc = acc_tmem.load(acc_reg_layout)
lhs = lhs_tmem.load(lhs_reg_layout)
```

`tcgen05_copy` can be used to copy data into tensor memory that is fed into a
`tcgen05_mma` instruction. Because `tcgen05_copy` is implicitly pipelined with
`tcgen05_mma`, even though it is asynchronous, the MMA is guaranteed to start
after the copy is complete:

```python
tcgen05_copy(smem, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
tcgen05_commit(bar)
mbarrier.wait(bar, phase=phase)
```

The implicit pipelining is because the PTX-level `tcgen05.copy` and `tcgen05.mma`
instructions are executed by the tensor core pipe on the SM, which you can think
of as a single thread running tensor core specific instructions on the SM,
asynchronously from the rest of the SM. In other words, all `tcgen05_*` instructions
enqueue a tensor core operation on the tensor pipe, which are executed in order.

The following is also valid.

```python
tcgen05_copy(lhs_smem0, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
tcgen05_commit(bar)

tcgen05_copy(lhs_smem1, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
```

Because the second `tcgen05_copy` will only execute after the preceeding
`tcgen05_mma` is complete. In other words, `tcgen05_copy`, `tcgen05_mma`, and
`tcgen05_commit` are all implicitly pipelined and executed in order.

`tcgen05_copy` accesses shared memory via the async proxy, just like `tcgen05_mma`.
Make sure to insert fences as appropriate:

```python
lhs_smem.store(value1)
fence_async_shared()
tcgen05_copy(lhs_smem, lhs_tmem)
tcgen05_commit(bar)

mbarrier.wait(bar, phase=phase)
lhs_smem.store(value0)
```

Note that a fence is not needed between `tcgen05_copy` and the second write to
`lhs_smem` because waiting on the completion of the `tcgen05_copy` operation
via the mbarrier implicitly fences the generic and async proxies.

What makes using `tcgen05_copy` particularly tricky is selecting the right
shared memory and tensor memory layouts, as `tcgen05_copy` only supports a
limited set of instruction shapes for copy data from shared to tensor memory.
"""

import itertools
import importlib
import pytest
import triton
import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    tensor_memory_descriptor,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    fence_async_shared,
    tcgen05_copy,
    tcgen05_commit,
    tcgen05_mma,
    mbarrier,
    tma,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")

# Re-use utilities from the previous tutorials.
t7 = importlib.import_module("07-persistence")
t8 = importlib.import_module("08-warp-specialization")

# %%
# Let's write an example kernel that uses `tcgen05_copy` and and show what the
# requirements are for the shared and tensor memory layouts.


@gluon.jit
def tcgen05_copy_kernel(in_ptr, in_stride0, in_stride1, out_ptr, out_stride0, out_stride1, M: gl.constexpr,
                        N: gl.constexpr, smem_layout: gl.constexpr, tmem_layout: gl.constexpr):
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    offs_m = gl.arange(0, M, gl.SliceLayout(1, coalesced_2d_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, coalesced_2d_layout))

    input = gl.load(in_ptr + offs_m[:, None] * in_stride0 + offs_n[None, :] * in_stride1)

    # Allocate shared memory and tensor memory with the tile shape [M, N].
    smem = gl.allocate_shared_memory(input.dtype, (M, N), smem_layout)
    tmem = allocate_tensor_memory(input.dtype, (M, N), tmem_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
    mbarrier.init(bar, count=1)

    # Copy data from shared memory to tensor memory.
    smem.store(input)
    # Fence generic and async proxies
    fence_async_shared()
    # Issue the async copy
    tcgen05_copy(smem, tmem)
    # Track completion of the async copy
    tcgen05_commit(bar)
    # Wait for the async copy to complete
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Read the data from tensor memory.
    tmem_reg_layout: gl.constexpr = get_tmem_reg_layout(input.dtype, (M, N), tmem_layout, gl.num_warps())
    output = tmem.load(tmem_reg_layout)

    # Write using a coalesced layout.
    output = gl.convert_layout(output, coalesced_2d_layout)
    gl.store(out_ptr + offs_m[:, None] * out_stride0 + offs_n[None, :] * out_stride1, output)


def tcgen05_copy_example(M, N, smem_layout, tmem_layout, dtype):
    input = torch.randn(M, N, dtype=dtype, device="cuda")
    output = torch.empty_like(input)
    tcgen05_copy_kernel[(1, )](input, *input.stride(), output, *output.stride(), M, N, smem_layout, tmem_layout)
    # Just check that the input and output are equal.
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# Let's first explore the valid shared memory layouts for the source of
# `tcgen05_copy` when the destination tensor memory layout is a
# `TensorMemoryLayout`, which is common when using TMAs and tensor core
# instructions.
#
# Recall that `TensorMemoryLayout` only supports 2D memory descriptors. When the
# destination tensor memory layout is a `TensorMemoryLayout`, the source shared
# memory layout is typically an `NVMMASharedLayout`. Other exotic layouts are
# supported, such as some `SharedLinearLayout`, but we won't cover them in this
# tutorial.
#
# Additional, the current restrictions apply to the `NVMMASharedLayout`:
# - The layout must be swizzled (swizzle_byte_width > 0).
# - The dtype must be 32-bit (e.g. gl.float32).
# - `TensorMemoryLayout` blockM must be 128.
# - The layout cannot be transposed.

configs = []
TMEM_BLOCK_M = 128
for TMEM_BLOCK_N in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    for M, N in itertools.product([128, 256], [16, 32, 64, 128, 256]):
        if M < TMEM_BLOCK_M or N < TMEM_BLOCK_N or M * N * 4 > 228 * 1024:
            continue
        configs.append((M, N, TMEM_BLOCK_N))


@pytest.mark.parametrize("M, N, TMEM_BLOCK_N", configs)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("swizzle", [32, 64, 128])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tcgen05_copy_nvmma_shared(M, N, TMEM_BLOCK_N, dtype, swizzle):
    bitwidth = dtype.itemsize * 8
    # There are still some shared memory layouts for which an implementation does not exist.
    if M == 256 and swizzle // TMEM_BLOCK_N >= 8:
        pytest.skip("no tcgen05.copy atom exists for codegen")
    # NVMMASharedLayout swizzle block shape has a minimum size.
    if N < swizzle // dtype.itemsize:
        pytest.skip("block shape along contiguous dimension is too small for the swizzle byte width")

    bitwidth = dtype.itemsize * 8
    smem_layout = gl.NVMMASharedLayout(swizzle_byte_width=swizzle, element_bitwidth=bitwidth, rank=2)
    tmem_layout = TensorMemoryLayout(block=(TMEM_BLOCK_M, TMEM_BLOCK_N), col_stride=32 // bitwidth)
    tcgen05_copy_example(M, N, smem_layout, tmem_layout, dtype)


# %%
# Although tcgen05_copy into TensorMemoryLayout only supports 32-bit dtypes,
# this is useful for writing matmul accumulate kernels: `D = A @ B + C`.
# Specifically, we can use TMA to load `C`, asynchronously copy it into tensor
# memory with `tcgen05_copy`, and then issue `tcgen05_mma` to perform the matmul
# while accumulating into tensor memory.
#
# We will use `gl.store` to write the output tiles to save shared memory, since
# C will require a large float32 buffer. We will use warp specialization to
# efficiently overlap the epilogue store with the rest of the kernel. Avoiding
# TMA for the epilogue store also reduces contention for the TMA pipe.


@aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    d_ptr: gl.tensor
    d_stride_m: gl.tensor
    d_stride_n: gl.tensor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    c_buf: gl.shared_memory_descriptor
    c_empty_bar: gl.shared_memory_descriptor
    c_ready_bar: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SchedulerImpl: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, d_ptr, d_stride_m, d_stride_n, a_bufs, b_bufs, load_empty_bars,
                 load_ready_bars, c_buf, c_empty_bar, c_ready_bar, acc_bufs, acc_empty_bars, acc_ready_bars,
                 SchedulerImpl):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.d_ptr = d_ptr
        self.d_stride_m = d_stride_m
        self.d_stride_n = d_stride_n
        self.a_bufs = a_bufs
        self.b_bufs = b_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.c_buf = c_buf
        self.c_empty_bar = c_empty_bar
        self.c_ready_bar = c_ready_bar
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.SchedulerImpl = gl.constexpr(SchedulerImpl)


@gluon.jit
def matmul_accumulate_load_partition(p):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    c_phase = 1
    state = t8.Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # Issue the async TMA load for the C tile.
        mbarrier.wait(p.c_empty_bar, c_phase)
        mbarrier.expect(p.c_ready_bar, p.c_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(p.c_desc, [off_m, off_n], p.c_ready_bar, p.c_buf)
        c_phase ^= 1
        # Inner loop loads.
        for k in range(0, K, BLOCK_K):
            bar = p.load_ready_bars.index(state.index)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            mbarrier.expect(bar, p.a_desc.block_type.nbytes + p.b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(p.a_desc, [off_m, k], bar, p.a_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()


@gluon.jit
def matmul_accmulate_mma_partition(p):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    c_phase = 0
    load_state = t8.Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = t8.Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for _ in range(scheduler.get_num_tiles()):
        # We expect the load of C to take longer than the previous epilogue to
        # release the accumulator, so acquire c_buf first.
        mbarrier.wait(p.c_ready_bar, c_phase)
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        tcgen05_copy(p.c_buf, acc_buf)
        # Release c_buf when the copy is complete. We don't need to wait for the
        # copy to complete because it will be implicitly pipelined with the first MMA.
        tcgen05_commit(p.c_empty_bar)
        c_phase ^= 1
        for k in range(0, K, BLOCK_K):
            # Wait for the operands to be ready.
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
            # Issue the MMA and release the load buffers then it completes.
            tcgen05_mma(p.a_bufs.index(load_state.index), p.b_bufs.index(load_state.index), acc_buf, use_acc=True)
            tcgen05_commit(p.load_empty_bars.index(load_state.index))
            load_state = load_state.next()
        # Release the accumulator when the last MMA is complete.
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def matmul_accumulate_epilogue_partition(p):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    dtype: gl.constexpr = p.c_desc.dtype

    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    range_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, coalesced_2d_layout))
    range_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, coalesced_2d_layout))

    acc_layout: gl.constexpr = get_tmem_reg_layout(dtype, (BLOCK_M, BLOCK_N), p.acc_bufs.type.layout, gl.num_warps())
    acc_state = t8.Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # Wait for the accumulator.
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index).load(acc_layout)
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()
        offs_m = (off_m + range_m)
        offs_n = (off_n + range_n)
        # This `convert_layout` is fairly expensive and it uses a lot of shared
        # memory, because `acc_layout` assigns contiguous columns to the same
        # thread, but the coalesced layout assigns contiguous columns to different
        # threads for efficient global writes. We could subtile the store to
        # reduce the shared memory usage.
        acc = gl.convert_layout(acc, coalesced_2d_layout)
        gl.store(p.d_ptr + offs_m[:, None] * p.d_stride_m + offs_n[None, :] * p.d_stride_n, acc)


@gluon.jit(do_not_specialize=["d_stride_m", "d_stride_n"])
def matmul_accumulate_kernel(a_desc, b_desc, c_desc, d_ptr, d_stride_m, d_stride_n, SchedulerImpl: gl.constexpr,
                             num_buffers: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype

    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    c_buf = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    c_empty_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    c_ready_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(c_empty_bar, count=1)
    mbarrier.init(c_ready_bar, count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_bufs = allocate_tensor_memory(gl.float32, [2, BLOCK_M, BLOCK_N], tmem_layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = PartitionArgs(a_desc, b_desc, c_desc, d_ptr, d_stride_m, d_stride_n, a_bufs, b_bufs, load_empty_bars,
                      load_ready_bars, c_buf, c_empty_bar, c_ready_bar, acc_bufs, acc_empty_bars, acc_ready_bars,
                      SchedulerImpl)
    gl.warp_specialize([
        (matmul_accumulate_epilogue_partition, (p, )),
        (matmul_accmulate_mma_partition, (p, )),
        (matmul_accumulate_load_partition, (p, )),
    ], [1, 1], [24, 24])


def matmul_accumulate(A, B, C, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_SIZE_M=8, num_buffers=3):
    SchedulerImpl = t7.GroupedPersistentTileScheduler(GROUP_SIZE_M)
    M, N = C.shape

    dtype = getattr(gl, str(A.dtype).split('.')[1])
    acc_dtype = getattr(gl, str(C.dtype).split('.')[1])
    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], dtype)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], dtype)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], acc_dtype)

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)
    D = torch.empty((M, N), dtype=C.dtype, device="cuda")

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    matmul_accumulate_kernel[grid](a_desc, b_desc, c_desc, D, *D.stride(), SchedulerImpl, num_buffers)
    return D


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 2048), (4096, 4096, 4096)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N", [(128, 128), (128, 64)])
@pytest.mark.parametrize("BLOCK_K, num_buffers", [(64, 3)])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_matmul_accumulate(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, dtype):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")
    C = torch.randn(M, N, dtype=torch.float32, device="cuda")
    D = matmul_accumulate(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers=num_buffers)
    torch.testing.assert_close(A @ B + C, D, atol=5e-3, rtol=1e-2)


# %%
# Another important use case for `tcgen05_copy` is to asynchronously copy tensor
# scales from shared memory to tensor memory for use by `tcgen05_mma_scaled`.
# In the next tutorial, we will cover `tcgen05_mma_scaled` in more detail, but
# for now just know that the tensor scales must be supplied to `tcgen05_mma_scaled`
# via tensor memory, and the layout of the scales tensor memory must be
# `TensorMemoryScalesLayout`. If we load the scales via TMAs into shared memory,
# we can efficiently copy the scales into tensor memory with `tcgen05_copy`
# which can be implicitly pipelined with the `tcgen05_mma_scaled` instruction:
#
# ```python
# tma.async_copy_global_to_shared(a_scale_desc, ..., bar, a_scale_buf)
# tma.async_copy_global_to_shared(b_scale_desc, ..., bar, b_scale_buf)
# mbarrier.wait(bar, phase)
#
# tcgen05_copy(a_scale_buf, a_scale_tmem)
# tcgen05_copy(b_scale_buf, b_scale_tmem)
# tcgen05_mma_scaled(a_buf, b_buf, acc_tmem, a_scale_tmem, b_scale_tmem, ...)
# tcgen05_commit(mma_bar)
# ```
#
# The main takeaway from this tutorial is understanding how to use `tcgen05_copy`
# to asynchronously copy data from shared memory to tensor memory. `tcgen05_copy`
# doesn't support all layouts, but should support typical NVMMASharedLayouts.
# The instruction is useful in specific cases to copy data from shared to tensor
# memory without round-tripping the data through registers, which increases
# register pressure and is slow. It is also asynchronous and can be implicitly
# pipelined with other `tcgen05` instructions.
