"""
TDM in Gluon
============

CDNA5 GPUs add TDM, a hardware unit that copies tiles of a tensor between
global memory and shared memory (LDS).

With async copies, the warps do all the work of a copy. Each copy instruction
moves 16 bytes per lane. On CDNA5, a warp has 32 lanes, so a warp needs 16
instructions to copy 8 KB, for example an 8x256 fp32 part of a tile. For each
instruction, the warp computes per-lane offsets and masks in VGPRs and sets the
shared memory base address. The register and shared memory layouts must also
match the fixed write pattern of the instruction.

TDM moves this work from the warps to a dedicated unit. A warp issues one TDM
instruction with a tensor descriptor. The TDM unit generates the addresses,
checks the bounds in every dimension, and writes the tile into LDS in the
requested shared memory layout. It also happens in the background like the async
copies, so the warp can overlap the memory access with other operations.
"""

import importlib
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna5 import tdm
from triton.experimental.gluon.language.amd.cdna5 import async_copy as cp
from triton.experimental.gluon.amd.cdna5 import TensorDescriptor

t1 = importlib.import_module("01-buffer-ops")


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not is_cdna5():
    raise RuntimeError("This tutorial requires an AMD CDNA5 GPU")

# A warp on CDNA5 GPUs has 32 lanes.
warp_size = 32

# %%
# A kernel creates a tensor descriptor with `tdm.make_tensor_descriptor`. Like a
# TMA descriptor, it holds the base pointer, the shape and strides of the
# tensor, the block shape, and the shared memory layout. The kernel builds the
# descriptor from ordinary kernel arguments.
#
# `tdm.async_load` copies the current block of a descriptor from global memory
# to shared memory, and `tdm.async_store` copies a tile back. Like buffer loads
# and stores, TDM checks the bounds in hardware. Out-of-bounds elements read
# zeros and are not written. TDM copies have their own wait counter,
# `tensorcnt`. `tdm.async_wait(n)` waits until at most `n` TDM copies of the
# warp are still in flight.
#
# A TDM copy takes no block coordinates. The descriptor holds the position of
# the current block, and the kernel moves it with
# `tdm.update_tensor_descriptor`. This function can also set a predicate on the
# descriptor with `pred`. If the predicate is false, a copy with this descriptor
# is still issued, but it copies nothing.


@gluon.jit
def elementwise_tdm_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    xoff = gl.program_id(0) * XBLOCK
    a_desc = tdm.make_tensor_descriptor(  #
        base=a_ptr + xoff * xstride_a,  #
        shape=[xnumel - xoff, ynumel],  #
        strides=[xstride_a, ystride_a],  #
        block_shape=[XBLOCK, YBLOCK],  #
        layout=smem_layout)
    b_desc = tdm.make_tensor_descriptor(  #
        base=b_ptr + xoff * xstride_b,  #
        shape=[xnumel - xoff, ynumel],  #
        strides=[xstride_b, ystride_b],  #
        block_shape=[XBLOCK, YBLOCK],  #
        layout=smem_layout)
    c_desc = tdm.make_tensor_descriptor(  #
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
        tdm.async_load(a_desc, dest=a_smem)
        tdm.async_load(b_desc, dest=b_smem)
        # Wait for the loads. This also waits for the store of the previous
        # iteration, so we can overwrite `c_smem`.
        tdm.async_wait(0)

        c_val = a_smem.load(layout) + b_smem.load(layout)
        c_smem.store(c_val)
        tdm.async_store(c_desc, src=c_smem)

        # Move the descriptors to the next column block. The descriptor stores
        # the address of the block and the extent of the tensor measured from
        # that address. `add_offsets` moves the block but keeps the extent, so
        # we also pass `clamp_bounds=True` to shrink the extent by the same
        # offsets. Then the bounds check stays correct after the move.
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, YBLOCK], clamp_bounds=True)
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, YBLOCK], clamp_bounds=True)
        c_desc = tdm.update_tensor_descriptor(c_desc, add_offsets=[0, YBLOCK], clamp_bounds=True)

    # Wait for the last store.
    tdm.async_wait(0)


def elementwise_tdm(A, B, C, XBLOCK=32, YBLOCK=256, num_warps=4):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [num_warps, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_tdm_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, layout, smem_layout, num_warps=num_warps)


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
# The host can also create a tensor descriptor with
# `TensorDescriptor.from_tensor` and pass it to a kernel as an argument. The
# kernel then reads the element type, the block shape and the shared memory
# layout from the descriptor.


@gluon.jit
def copy_kernel(in_desc, out_desc):
    smem = gl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape, in_desc.layout)
    xoff = gl.program_id(0) * in_desc.block_shape[0]
    yoff = gl.program_id(1) * in_desc.block_shape[1]
    in_desc = tdm.update_tensor_descriptor(in_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    out_desc = tdm.update_tensor_descriptor(out_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    tdm.async_load(in_desc, dest=smem)
    tdm.async_wait(0)
    tdm.async_store(out_desc, src=smem)
    tdm.async_wait(0)


def copy(A, B, XBLOCK=32, YBLOCK=256, num_warps=4):
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    in_desc = TensorDescriptor.from_tensor(A, [XBLOCK, YBLOCK], smem_layout)
    out_desc = TensorDescriptor.from_tensor(B, [XBLOCK, YBLOCK], smem_layout)
    grid = (triton.cdiv(A.shape[0], XBLOCK), triton.cdiv(A.shape[1], YBLOCK))
    return copy_kernel[grid](in_desc, out_desc, num_warps=num_warps)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 160)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 256), (4, 512)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_copy(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.empty_like(a)
    copy(a, b, XBLOCK, YBLOCK)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


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
    a_desc = tdm.make_tensor_descriptor(a_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    b_desc = tdm.make_tensor_descriptor(b_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    c_desc = tdm.make_tensor_descriptor(c_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    xoff = gl.program_id(0) * XBLOCK
    yoff = gl.program_id(1) * YBLOCK

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    c_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    c_desc = tdm.update_tensor_descriptor(c_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    # Warps 0-3 (0b00001111) load A, and warps 4-7 (0b11110000) load B.
    tdm.async_load(a_desc, dest=a_smem, warp_used_hint=0b00001111)
    tdm.async_load(b_desc, dest=b_smem, warp_used_hint=0b11110000)
    # Each warp waits only for its own copies. The CTA barrier after the wait
    # makes both blocks visible to all warps.
    tdm.async_wait(0)

    c_smem.store(a_smem.load(layout) + b_smem.load(layout))
    tdm.async_store(c_desc, src=c_smem)
    tdm.async_wait(0)


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
# `asynccnt`. `cp.wait_group` waits on `asynccnt`, and `tdm.async_wait` waits on
# `tensorcnt`. Within one counter, copies complete in order. The two counters
# are independent of each other. A kernel can wait for an async copy without
# waiting for a TDM copy that it issued earlier.
#
# The kernel below loads A with TDM and then B with an async copy. It waits for
# B and reads it while the TDM copy of A can still be in flight.


@gluon.jit
def elementwise_mix_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    a_desc = tdm.make_tensor_descriptor(a_ptr, [xnumel, ynumel], [ynumel, 1], [XBLOCK, YBLOCK], smem_layout)
    xoff = gl.program_id(0) * XBLOCK
    yoff = gl.program_id(1) * YBLOCK
    xoffs = xoff + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    offs = xoffs[:, None] * ynumel + yoffs[None, :]
    mask = (xoffs[:, None] < xnumel) & (yoffs[None, :] < ynumel)

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[xoff, yoff], clamp_bounds=True)
    tdm.async_load(a_desc, dest=a_smem)
    cp.global_to_shared(b_smem, b_ptr + offs, mask)
    cp.commit_group()

    cp.wait_group(0)
    b_val = b_smem.load(layout)
    tdm.async_wait(0)
    a_val = a_smem.load(layout)
    gl.store(c_ptr + offs, a_val + b_val, mask)


def elementwise_mix(A, B, C, XBLOCK=32, YBLOCK=256, num_warps=4):
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [num_warps, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    return elementwise_mix_kernel[grid](A, B, C, xnumel, ynumel, XBLOCK, YBLOCK, layout, smem_layout,
                                        num_warps=num_warps)


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
# Lastly, let's build a pipelined kernel with TDM and compare the performance
# across different memory operations.


@gluon.jit
def issue_loads(copy_idx, num_blocks, a_desc, b_desc, a_smem, b_smem, YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    yoff = copy_idx * YBLOCK
    # Turn off the loads past the last block. They still count in `tensorcnt`,
    # so every iteration waits with the same count.
    pred = copy_idx < num_blocks
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, yoff], pred=pred, clamp_bounds=True)
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, yoff], pred=pred, clamp_bounds=True)
    tdm.async_load(a_desc, dest=a_smem.index(copy_idx % num_buffers))
    tdm.async_load(b_desc, dest=b_smem.index(copy_idx % num_buffers))
    return copy_idx + 1


@gluon.jit
def perform_add(read_idx, a_smem, b_smem, c_smem, c_desc, YBLOCK: gl.constexpr, num_buffers: gl.constexpr,
                layout: gl.constexpr):
    a_val = a_smem.index(read_idx % num_buffers).load(layout)
    b_val = b_smem.index(read_idx % num_buffers).load(layout)
    c_buf = c_smem.index(read_idx % num_buffers)
    c_buf.store(a_val + b_val)
    c_desc = tdm.update_tensor_descriptor(c_desc, add_offsets=[0, read_idx * YBLOCK], clamp_bounds=True)
    tdm.async_store(c_desc, src=c_buf)
    return read_idx + 1


@gluon.jit
def elementwise_tdm_pipelined_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr, num_buffers: gl.constexpr,  #
):
    xoff = gl.program_id(0) * XBLOCK
    a_desc = tdm.make_tensor_descriptor(a_ptr + xoff * xstride_a, [xnumel - xoff, ynumel], [xstride_a, ystride_a],
                                        [XBLOCK, YBLOCK], smem_layout)
    b_desc = tdm.make_tensor_descriptor(b_ptr + xoff * xstride_b, [xnumel - xoff, ynumel], [xstride_b, ystride_b],
                                        [XBLOCK, YBLOCK], smem_layout)
    c_desc = tdm.make_tensor_descriptor(c_ptr + xoff * xstride_c, [xnumel - xoff, ynumel], [xstride_c, ystride_c],
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
        tdm.async_wait((num_buffers - 1) * 2)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_smem, c_desc, YBLOCK, num_buffers, layout)

    tdm.async_wait(0)


def elementwise_tdm_pipelined(A, B, C, XBLOCK=32, YBLOCK=256, num_buffers=2, num_warps=4):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [num_warps, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_tdm_pipelined_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, layout, smem_layout, num_buffers, num_warps=num_warps)


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


# %%
# On an MI455X GPU, we compare the TDM kernels with the global load kernel on
# 16K x 16K fp32 tensors with 4x512 tiles.


def elementwise(A, B, C, XBLOCK=32, YBLOCK=256, num_warps=4):
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [num_warps, 1], [1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return t1.elementwise_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, layout, num_warps=num_warps)


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
# global load: 13.54 TB/s
# tdm: 17.90 TB/s
# tdm pipelined (double buffer): 18.38 TB/s
# ```
#
# TDM is 32% faster than global loads, and double buffering adds another 3%.
# The best kernel reaches 18.38 TB/s, 94% of the 19.6 TB/s peak HBM bandwidth
# of MI455X.

# %%
# The main takeaway here is to use TDM on CDNA5 whenever a kernel copies tiles
# between global memory and shared memory. TDM takes the address computation and
# the bounds checks off the warps. A warp issues far fewer instructions for a
# copy, and the kernel runs faster.
#
# As on CDNA4, the shared memory budget limits the number of buffers. On CDNA5,
# each CU has 320 KB of LDS, twice as much as on CDNA4. A kernel can use more
# buffers and a deeper pipeline. The CTAs on a CU still share the LDS, and each
# buffer reduces the number of CTAs that fit on a CU.
