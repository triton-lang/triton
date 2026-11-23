"""
TDM in Gluon
============

CDNA5 GPUs add TDM, a hardware unit that copies tiles of a tensor between
global memory and shared memory (LDS).

With async copies, the warps do all the work of a copy. Each copy instruction
moves 16 bytes per lane. On CDNA5, a warp has 32 lanes, so a warp needs 16
instructions to copy 8 KB, for example an 8x256 fp32 part of a tile. For each
instruction, the warp computes per-lane offsets and masks in registers and
sets the shared memory base address. The register and shared memory layouts
must also match the fixed write pattern of the instruction.

TDM moves this work from the warps to a dedicated unit. A warp issues one TDM
instruction with a tensor descriptor. The TDM unit generates the addresses,
checks the bounds in every dimension, and writes the tile into LDS in the
requested shared memory layout. It also happens in the background like the async
copies, so the warp can overlap the memory access with other operations.
"""

import importlib
import pytest
import sys
import torch
import triton
from pathlib import Path
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
t1 = importlib.import_module("amd.01-buffer-ops")


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not is_cdna5():
    raise RuntimeError("This tutorial requires an AMD CDNA5 GPU")

# A warp on CDNA5 GPUs has 32 lanes.
warp_size = gl.constexpr(32)

# %%
# A kernel creates a tensor descriptor with `tdm.make_tensor_descriptor`. Like a
# TMA descriptor, it holds the base pointer, the shape and strides of the
# tensor, the block shape, and the shared memory layout. The kernel builds the
# descriptor from ordinary kernel arguments.
#
# `tdm.async_load` copies the current block of a descriptor from global memory
# to shared memory, and `tdm.async_store` copies a tile back. Like buffer loads
# and stores, TDM checks the bounds in hardware. Out-of-bounds elements read
# zeros and are not written.
#
# TDM copies have their own wait counter, `tensorcnt`. `tdm.async_wait(n)` is
# used to wait until at most `n` TDM copies of the warp are still in flight.
#
# When we need to copy a tile of the tensor at an offset, we need to update the
# TDM descriptor with `tdm.update_tensor_descriptor`. This function can also set
# a predicate on the descriptor with `pred`. If the predicate is false, a copy
# with this descriptor is still issued, but it copies nothing.


@gluon.jit
def elementwise_tdm_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    xoff = gl.program_id(0) * XBLOCK
    a_desc = cdna5.tdm.make_tensor_descriptor(  #
        base=a_ptr + xoff * xstride_a,  #
        shape=[xnumel - xoff, ynumel],  #
        strides=[xstride_a, ystride_a],  #
        block_shape=[XBLOCK, YBLOCK],  #
        layout=smem_layout)
    b_desc = cdna5.tdm.make_tensor_descriptor(  #
        base=b_ptr + xoff * xstride_b,  #
        shape=[xnumel - xoff, ynumel],  #
        strides=[xstride_b, ystride_b],  #
        block_shape=[XBLOCK, YBLOCK],  #
        layout=smem_layout)
    c_desc = cdna5.tdm.make_tensor_descriptor(  #
        base=c_ptr + xoff * xstride_c,  #
        shape=[xnumel - xoff, ynumel],  #
        strides=[xstride_c, ystride_c],  #
        block_shape=[XBLOCK, YBLOCK],  #
        layout=smem_layout)

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    c_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    for _ in range(0, ynumel, YBLOCK):
        cdna5.tdm.async_load(a_desc, dest=a_smem)
        cdna5.tdm.async_load(b_desc, dest=b_smem)
        # Wait for the loads. This also waits for the store of the previous
        # iteration, so we can overwrite `c_smem`.
        cdna5.tdm.async_wait(0)

        c_val = a_smem.load(layout) + b_smem.load(layout)
        c_smem.store(c_val)
        cdna5.tdm.async_store(c_desc, src=c_smem)

        # Move the descriptors to the next column block. The descriptor stores
        # the address of the block and the extent of the tensor measured from
        # that address. `add_offsets` moves the block but keeps the extent, so
        # we also pass `clamp_bounds=True` to shrink the extent by the same
        # offsets. Then the bounds check stays correct after the move.
        a_desc = cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[0, YBLOCK], clamp_bounds=True)
        b_desc = cdna5.tdm.update_tensor_descriptor(b_desc, add_offsets=[0, YBLOCK], clamp_bounds=True)
        c_desc = cdna5.tdm.update_tensor_descriptor(c_desc, add_offsets=[0, YBLOCK], clamp_bounds=True)

    # Wait for the last store.
    cdna5.tdm.async_wait(0)


def elementwise_tdm(A, B, C, XBLOCK=32, YBLOCK=256):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [4, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_tdm_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, layout, smem_layout)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 160)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 256), (4, 512)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_elementwise_tdm(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise_tdm(a, b, c, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


if __name__ == "__main__":
    A = torch.randn(1024, 2048, device="cuda")
    B = torch.randn_like(A)
    C = torch.empty_like(A)
    kernel = elementwise_tdm(A, B, C)
    t1.print_instructions(kernel, r"tensor_load|s_wait_tensorcnt|s_barrier|tensor_store")

# %%
# ```
#   tensor_load_to_lds s[0:3], s[24:31]
#   ...
#   s_wait_tensorcnt 0x0
#   s_barrier_signal -1
#   s_barrier_wait -1
#   ...
#   tensor_store_from_lds s[40:43], s[8:15]
# ```
#
# The compiler splits the 32x256 tile between the 4 warps, and each warp loads
# its 8x256 part with one `tensor_load_to_lds` instead of 16 async copy
# instructions. The instruction reads the descriptor from 12 SGPRs in two
# groups, here `s[0:3]` and `s[24:31]`. The kernel updates the descriptor with
# scalar instructions on these SGPRs.
#
# `tdm.async_wait` becomes `s_wait_tensorcnt`. Like `vmcnt`, `tensorcnt` belongs
# to the warp, and loads and stores share it and complete in order. Since the
# wait only covers the copies of the current warp, the compiler adds a CTA
# barrier after it. On CDNA5, the barrier is split into two instructions. With
# `s_barrier_signal`, a warp tells the other warps that it has reached the
# barrier. With `s_barrier_wait`, it waits until all warps of the CTA have
# signaled.

# %%
# As mentioned above, the compiler splits a TDM copy between all warps of the
# CTA. The `warp_used_hint` argument of `tdm.async_load` selects a subset of the
# warps instead. The argument is a bitmask, and bit `n` selects warp `n`. The
# selected warps copy the whole block between them. The other warps still issue
# the instruction, but with a false predicate. The number of selected warps must
# be a power of two, and the selection must follow a regular bit pattern.


@gluon.jit
def elementwise_warp_hint_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    a_desc = cdna5.tdm.make_tensor_descriptor(a_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    b_desc = cdna5.tdm.make_tensor_descriptor(b_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    c_desc = cdna5.tdm.make_tensor_descriptor(c_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    xoff = gl.program_id(0) * XBLOCK
    yoff = gl.program_id(1) * YBLOCK

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    c_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    a_desc = cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    b_desc = cdna5.tdm.update_tensor_descriptor(b_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    c_desc = cdna5.tdm.update_tensor_descriptor(c_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    # Warps 0-3 (0b00001111) load A, and warps 4-7 (0b11110000) load B.
    cdna5.tdm.async_load(a_desc, dest=a_smem, warp_used_hint=0b00001111)
    cdna5.tdm.async_load(b_desc, dest=b_smem, warp_used_hint=0b11110000)
    # Each warp waits only for its own copies. The CTA barrier after the wait
    # makes both blocks visible to all warps.
    cdna5.tdm.async_wait(0)

    c_smem.store(a_smem.load(layout) + b_smem.load(layout))
    cdna5.tdm.async_store(c_desc, src=c_smem)
    cdna5.tdm.async_wait(0)


def elementwise_warp_hint(A, B, C, XBLOCK=32, YBLOCK=256):
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [8, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    return elementwise_warp_hint_kernel[grid](A, B, C, xnumel, ynumel, XBLOCK, YBLOCK, layout, smem_layout, num_warps=8)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 160)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 256), (8, 512)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_elementwise_warp_hint(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise_warp_hint(a, b, c, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


# %%
# Async copies also work on CDNA5, and they have their own wait counter,
# `asynccnt`. `async_copy.wait_group` waits on `asynccnt`, and `tdm.async_wait`
# waits on `tensorcnt`. Within one counter, copies complete in order. The two
# counters are independent of each other. A kernel can wait for an async copy
# without waiting for a TDM copy that it issued earlier.
#
# The kernel below loads A with TDM and then B with an async copy. It waits for
# B and reads it while the TDM copy of A can still be in flight.


@gluon.jit
def elementwise_mix_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    a_desc = cdna5.tdm.make_tensor_descriptor(a_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    xoff = gl.program_id(0) * XBLOCK
    yoff = gl.program_id(1) * YBLOCK
    xoffs = xoff + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    offs = xoffs[:, None] * ynumel + yoffs[None, :]
    mask = (xoffs[:, None] < xnumel) & (yoffs[None, :] < ynumel)

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    a_desc = cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    cdna5.tdm.async_load(a_desc, dest=a_smem)
    cdna5.async_copy.global_to_shared(b_smem, b_ptr + offs, mask)
    cdna5.async_copy.commit_group()

    cdna5.async_copy.wait_group(0)
    b_val = b_smem.load(layout)
    cdna5.tdm.async_wait(0)
    a_val = a_smem.load(layout)
    gl.store(c_ptr + offs, a_val + b_val, mask)


def elementwise_mix(A, B, C, XBLOCK=32, YBLOCK=256):
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [4, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    return elementwise_mix_kernel[grid](A, B, C, xnumel, ynumel, XBLOCK, YBLOCK, layout, smem_layout)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 160)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 256), (4, 512)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_elementwise_mix(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise_mix(a, b, c, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


if __name__ == "__main__":
    A = torch.randn(1024, 2048, device="cuda")
    B = torch.randn_like(A)
    C = torch.empty_like(A)
    kernel = elementwise_mix(A, B, C)
    t1.print_instructions(kernel, r"tensor_load|global_load_async|s_wait_asynccnt|s_wait_tensorcnt|ds_load")

# %%
# ```
#   tensor_load_to_lds s[20:23], s[12:19]
#   global_load_async_to_lds_b128 v32, v[0:1], off
#   ...
#   s_wait_asynccnt 0x0
#   ds_load_b128 v[120:123], v156 offset:32768
#   ...
#   s_wait_tensorcnt 0x0
#   ds_load_b128 v[116:119], v156 offset:512
#   ...
# ```
#
# Each warp issues one `tensor_load_to_lds` for A and 16
# `global_load_async_to_lds_b128` for B. `s_wait_asynccnt 0x0` waits only for
# the async copies, and the reads of B start before `s_wait_tensorcnt 0x0`
# waits for the TDM copy of A.

# %%
# TDM copies can also signal an mbarrier, like async copies. `tdm.async_load`
# takes the barrier as the `mbarrier` argument. Each warp arrives once for each
# TDM instruction it issues, after the copy of the instruction is complete. In
# the kernel below, each warp issues one TDM instruction for A and one for B, so
# we set the count to twice the number of warps.


@gluon.jit
def elementwise_tdm_mbarrier_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    a_desc = cdna5.tdm.make_tensor_descriptor(a_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    b_desc = cdna5.tdm.make_tensor_descriptor(b_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    xoff = gl.program_id(0) * XBLOCK
    yoff = gl.program_id(1) * YBLOCK
    xoffs = xoff + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    offs = xoffs[:, None] * ynumel + yoffs[None, :]
    mask = (xoffs[:, None] < xnumel) & (yoffs[None, :] < ynumel)

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], cdna5.mbarrier.MBarrierLayout())
    cdna5.mbarrier.init(bar, count=2 * gl.num_warps())

    a_desc = cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    b_desc = cdna5.tdm.update_tensor_descriptor(b_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    cdna5.tdm.async_load(a_desc, dest=a_smem, mbarrier=bar)
    cdna5.tdm.async_load(b_desc, dest=b_smem, mbarrier=bar)
    cdna5.mbarrier.wait(bar, 0)

    gl.store(c_ptr + offs, a_smem.load(layout) + b_smem.load(layout), mask)


def elementwise_tdm_mbarrier(A, B, C, XBLOCK=32, YBLOCK=256):
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [4, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    return elementwise_tdm_mbarrier_kernel[grid](A, B, C, xnumel, ynumel, XBLOCK, YBLOCK, layout, smem_layout)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 160)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 256), (4, 512)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_elementwise_tdm_mbarrier(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise_tdm_mbarrier(a, b, c, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


# %%
# Lastly, let's build a pipelined kernel with TDM and compare the performance
# across different memory operations.


@gluon.jit
def issue_loads(copy_idx, num_blocks, a_desc, b_desc, a_smem, b_smem, YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    yoff = copy_idx * YBLOCK
    # Turn off the loads past the last block. They still count in `tensorcnt`,
    # so every iteration waits with the same count.
    pred = copy_idx < num_blocks
    a_desc = cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[0, yoff], pred=pred, clamp_bounds=True)
    b_desc = cdna5.tdm.update_tensor_descriptor(b_desc, add_offsets=[0, yoff], pred=pred, clamp_bounds=True)
    cdna5.tdm.async_load(a_desc, dest=a_smem.index(copy_idx % num_buffers))
    cdna5.tdm.async_load(b_desc, dest=b_smem.index(copy_idx % num_buffers))
    return copy_idx + 1


@gluon.jit
def perform_add(read_idx, a_smem, b_smem, c_smem, c_desc, YBLOCK: gl.constexpr, num_buffers: gl.constexpr,
                layout: gl.constexpr):
    a_val = a_smem.index(read_idx % num_buffers).load(layout)
    b_val = b_smem.index(read_idx % num_buffers).load(layout)
    c_buf = c_smem.index(read_idx % num_buffers)
    c_buf.store(a_val + b_val)
    c_desc = cdna5.tdm.update_tensor_descriptor(c_desc, add_offsets=[0, read_idx * YBLOCK], clamp_bounds=True)
    cdna5.tdm.async_store(c_desc, src=c_buf)
    return read_idx + 1


@gluon.jit
def elementwise_tdm_pipelined_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr, num_buffers: gl.constexpr,  #
):
    xoff = gl.program_id(0) * XBLOCK
    a_desc = cdna5.tdm.make_tensor_descriptor(a_ptr + xoff * xstride_a, [xnumel - xoff, ynumel], [xstride_a, ystride_a],
                                              [XBLOCK, YBLOCK], smem_layout)
    b_desc = cdna5.tdm.make_tensor_descriptor(b_ptr + xoff * xstride_b, [xnumel - xoff, ynumel], [xstride_b, ystride_b],
                                              [XBLOCK, YBLOCK], smem_layout)
    c_desc = cdna5.tdm.make_tensor_descriptor(c_ptr + xoff * xstride_c, [xnumel - xoff, ynumel], [xstride_c, ystride_c],
                                              [XBLOCK, YBLOCK], smem_layout)

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    c_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    num_blocks = gl.cdiv(ynumel, YBLOCK)
    copy_idx = 0
    read_idx = 0

    for _ in gl.static_range(num_buffers - 1):
        copy_idx = issue_loads(copy_idx, num_blocks, a_desc, b_desc, a_smem, b_smem, YBLOCK, num_buffers)

    for _ in range(num_blocks):
        copy_idx = issue_loads(copy_idx, num_blocks, a_desc, b_desc, a_smem, b_smem, YBLOCK, num_buffers)
        cdna5.tdm.async_wait((num_buffers - 1) * 2)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_smem, c_desc, YBLOCK, num_buffers, layout)

    cdna5.tdm.async_wait(0)


def elementwise_tdm_pipelined(A, B, C, XBLOCK=32, YBLOCK=256, num_buffers=2):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [4, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_tdm_pipelined_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, layout, smem_layout, num_buffers)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 160)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 256), (4, 512)])
@pytest.mark.parametrize("num_buffers", [1, 2, 3])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_elementwise_tdm_pipelined(xnumel, ynumel, XBLOCK, YBLOCK, num_buffers):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise_tdm_pipelined(a, b, c, XBLOCK, YBLOCK, num_buffers)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


def elementwise(A, B, C, XBLOCK=32, YBLOCK=256):
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [4, 1], [1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return t1.elementwise_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, layout)


if __name__ == "__main__":
    print("Benchmarking elementwise add")
    print("============================")
    xnumel, ynumel = 16 * 1024, 16 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A)

    def bench(name, fn):
        ms = triton.testing.do_bench(fn, warmup=200, rep=1000)
        print(f"{name}: {t1.get_throughput(ms, C):.2f} TB/s")

    XBLOCK, YBLOCK = 4, 512
    bench("global load", lambda: elementwise(A, B, C, XBLOCK, YBLOCK))
    bench("tdm", lambda: elementwise_tdm(A, B, C, XBLOCK, YBLOCK))
    bench("tdm pipelined (double buffer)", lambda: elementwise_tdm_pipelined(A, B, C, XBLOCK, YBLOCK, 2))

# %%
# ```
# global load: 13.49 TB/s
# tdm: 17.67 TB/s
# tdm pipelined (double buffer): 18.22 TB/s
# ```
#
# TDM is 31% faster than global loads, and double buffering adds another 3%.
# The best kernel reaches 18.22 TB/s.

# %%
# TDM can also prefetch a block into the L2 cache with `tdm.prefetch`, which is
# also called L2 prefetch. On CDNA5 GPUs, Each workgroup processor (WGP) has its
# own data cache, which shares the same SRAM with the shared memory. Behind the
# WGP caches is the L2 cache, which sits on the two Fabric and Cache Dies (FCDs)
# on one GPU. When there is data in L2 cache, later loads can cached instead of
# going all the way to the HBM.
#
# Gluon's `tdm.prefetch` takes a descriptor and the offsets of a block, just
# like a TDM load, but it doesn't write anything into shared memory. It can be
# treated a memory access without destination in shared memory. The compiler
# splits the block into chunks of 256 bytes, spreads them over the lanes, warps
# and CTAs, and issues one `global_prefetch_b8` instruction for each piece.
#
# By default, a prefetch is non-speculative. The hardware translates the address
# even if the translation is not cached, and the compiler keeps the addresses
# within the tensor. With `speculative=True`, the compiler skips this bounds
# check, and the hardware will drop the prefetch if the address translation is
# not cached.
#
# Let's show the effect of L2 prefetch with a simple elementwise kernel, where
# each program walks through the column blocks of its rows with a single buffer,
# so the copy of the next block only starts after the current block is done.


@gluon.jit
def elementwise_tdm_prefetch_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr, PREFETCH: gl.constexpr,  #
):
    xoff = gl.program_id(0) * XBLOCK
    a_desc = cdna5.tdm.make_tensor_descriptor(a_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    b_desc = cdna5.tdm.make_tensor_descriptor(b_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    c_desc = cdna5.tdm.make_tensor_descriptor(c_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    c_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    for yoff in range(0, ynumel, YBLOCK):
        cdna5.tdm.async_load(cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[xoff, yoff], clamp_bounds=True),
                             dest=a_smem)
        cdna5.tdm.async_load(cdna5.tdm.update_tensor_descriptor(b_desc, add_offsets=[xoff, yoff], clamp_bounds=True),
                             dest=b_smem)
        if PREFETCH:
            # Bring the next column block into L2 while this one is loaded and added.
            pred = yoff + YBLOCK < ynumel
            cdna5.tdm.prefetch(a_desc, [xoff, yoff + YBLOCK], pred=pred)
            cdna5.tdm.prefetch(b_desc, [xoff, yoff + YBLOCK], pred=pred)
        cdna5.tdm.async_wait(0)

        c_smem.store(a_smem.load(layout) + b_smem.load(layout))
        cdna5.tdm.async_store(cdna5.tdm.update_tensor_descriptor(c_desc, add_offsets=[xoff, yoff], clamp_bounds=True),
                              src=c_smem)
    cdna5.tdm.async_wait(0)


def elementwise_tdm_prefetch(A, B, C, prefetch, XBLOCK=64, YBLOCK=256):
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [4, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_tdm_prefetch_kernel[grid](A, B, C, xnumel, ynumel, XBLOCK, YBLOCK, layout, smem_layout, prefetch)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 160)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(64, 256), (4, 512)])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_elementwise_tdm_prefetch(xnumel, ynumel, XBLOCK, YBLOCK, prefetch):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise_tdm_prefetch(a, b, c, prefetch, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


if __name__ == "__main__":
    print("Benchmarking prefetch")
    print("=====================")
    xnumel, ynumel = 16 * 1024, 16 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A)

    for name, prefetch in [("no prefetch", False), ("prefetch", True)]:
        ms = triton.testing.do_bench(lambda: elementwise_tdm_prefetch(A, B, C, prefetch), warmup=200, rep=1000)
        print(f"{name}: {t1.get_throughput(ms, C):.2f} TB/s")

# %%
# ```
# no prefetch: 15.89 TB/s
# prefetch: 17.64 TB/s
# ```
#
# We can see prefetch makes the kernel 11% faster. In practice, L2 prefetch
# may not always be helpful - we can also do double buffering to hide memory
# latency. Moreover, L2 prefetch may compete with TDM for memory bandwidth.
# But it can still be helpful in cases where we may not be able to use shared
# memory to buffer another future requests.

# %%
# The main takeaway here is to use TDM on CDNA5 whenever a kernel copies tiles
# between global memory and shared memory. TDM takes the address computation and
# the bounds checks off the warps. A warp issues far fewer instructions for a
# copy, and the kernel runs faster.
#
# As on CDNA4, the shared memory budget limits the number of buffers. On CDNA5,
# the compute unit is called a workgroup processor (WGP). A WGP has 320 KB of
# LDS, twice as much as a CDNA4 CU. A kernel can use more buffers and a deeper
# pipeline. The CTAs on a WGP still share the LDS, and each buffer reduces the
# number of CTAs that fit on a WGP.
