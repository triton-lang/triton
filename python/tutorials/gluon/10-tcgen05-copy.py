"""
TCGen05 Copy Instruction
========================

This tutorial will cover the `tcgen05_copy` instruction: how to use it and its
applications.

The `tcgen05_copy` instruction is an asynchronous tensorcore operation that
copies data from shared memory to tensor memory. `tcgen05_copy` is implicitly
pipelined with `tcgen05_mma` and `tcgen05_commit`. The completion of
`tcgen05_copy` is tracked with `tcgen05_commit` on an mbarrier just like
`tcgen05_mma`.

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

The completion of a single or multiple `tcgen05_copy` operations can be tracked
with `tcgen05_commit`:

```python
tcgen05_copy(lhs_smem, lhs_tmem)
tcgen05_copy(acc_smem, acc_tmem)
tcgen05_commit(bar)
mbarrier.wait(bar, phase=phase)
acc = acc_tmem.load(acc_reg_layout)
lhs = lhs_tmem.load(lhs_reg_layout)
```

The implicit pipelining is because the PTX-level `tcgen05.copy` and `tcgen05.mma`
instructions are executed by the tensor core pipe on the SM, which you can think
of as a single thread running tensor core specific instructions on the SM,
asynchronously from the rest of the SM.

The following is also valid.

```python
tcgen05_copy(lhs_smem0, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
tcgen05_commit(bar)

tcgen05_copy(lhs_smem1, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
```

Because the second `tcgen05_copy` will only execute after the preceeding
`tcgen05_mma` is complete,

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

import pytest
import triton
import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    fence_async_shared,
    tcgen05_copy,
    tcgen05_commit,
    mbarrier,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")

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


@pytest.mark.parametrize("M", [128, 256])
@pytest.mark.parametrize("N", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("TMEM_BLOCK_M", [128])
@pytest.mark.parametrize("TMEM_BLOCK_N", [1, 2, 4, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("swizzle", [32, 64, 128])
def test_tcgen05_copy_nvmma_shared(M, N, TMEM_BLOCK_M, TMEM_BLOCK_N, dtype, swizzle):
    if M < TMEM_BLOCK_M or N < TMEM_BLOCK_N:
        pytest.skip("allocation shape (M, N) is smaller than tensor memory block shape (TMEM_BLOCK_M, TMEM_BLOCK_N)")
    if M == 256 and N == 256:
        pytest.skip("not enough shared memory for (M, N) = (256, 256)")
    if M == 256 and swizzle // TMEM_BLOCK_N >= 8:
        pytest.skip("no tcgen05.copy atom exists for codegen")
    bitwidth = dtype.itemsize * 8
    smem_layout = gl.NVMMASharedLayout(swizzle_byte_width=swizzle, element_bitwidth=bitwidth, rank=2)
    tmem_layout = TensorMemoryLayout(block=(TMEM_BLOCK_M, TMEM_BLOCK_N), col_stride=32 // bitwidth)
    tcgen05_copy_example(M, N, smem_layout, tmem_layout, dtype)
