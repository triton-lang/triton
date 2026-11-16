"""
Async Copy
==========

This tutorial covers async copies on AMD GPUs. Async copies is an important
memory operation allows the kernel to overlap memory access with compute for
software pipelining. Despite the name, async copies on AMD GPUs behave quite
differently from NVIDIA's async copies - Warps wait for the completion of memory
instructions with wait counters instead of using the mbarrier mechanism.

Different types of memory instructions have separate counters. This
counter-based mechanism applies to most memory transfers, including the
transfers among global memory, shared memory and registers. We can treat all of
these memory instructions as "async" operations, i.e., the memory accesses
happen in the background. However, only some of the counters are
directly visible in Gluon, while others are managed by the compiler.

In this tutorial, we only discuss the memory instructions between global memory
and shared memory on CDNA4 and CDNA5, and focus on these Gluon can control
explicitly.
"""

import contextlib
import importlib
import os
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import buffer
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cp
from triton.experimental.gluon.language.amd import cdna5

t1 = importlib.import_module("amd.01-buffer-ops")


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx950"


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not (is_cdna4() or is_cdna5()):
    raise RuntimeError("This tutorial requires an AMD CDNA4 or CDNA5 GPU")

# A warp has 64 lanes on CDNA4 GPUs and 32 lanes on CDNA5 GPUs.
warp_size = gl.constexpr(64 if is_cdna4() else 32)

# %%
# CDNA4 async copy
# ----------------
#
# Let's first look at some important wait counters:
#
# - `vmcnt` counts global memory accesses.
# - `lgkmcnt` counts shared memory accesses.
#
# To wait for a result, the warp runs an `s_waitcnt` instruction. For example,
# `s_waitcnt vmcnt(N)` waits until at most `N` of its memory instructions are
# still in flight. The global load kernel from the previous tutorial shows how
# these waits work.

if __name__ == "__main__" and is_cdna4():
    A = torch.randn(1024, 2048, device="cuda")
    B = torch.randn_like(A)
    C = torch.empty_like(A)
    kernel = t1.elementwise(A, B, C, t1.elementwise_kernel, XBLOCK=16)
    t1.print_instructions(kernel, r"global_load|vmcnt|v_pk_add")

# %%
# ```
#   global_load_dwordx4 v[4:7], v[14:15], off
#   ...
#   s_waitcnt vmcnt(4)
#   v_pk_add_f32 v[10:11], v[10:11], v[18:19]
#   v_pk_add_f32 v[8:9], v[8:9], v[16:17]
#   s_waitcnt vmcnt(2)
# ```
#
# Each warp issues 8 loads, 4 for A and 4 for B. Before the first adds, the
# warp waits with `vmcnt(4)` until only 4 loads are still in flight. Note that
# instructions of the same kind complete in the order they were issued, so
# waiting for one instruction also waits for all earlier ones. After
# `vmcnt(4)`, the first 4 loads are complete, and they hold the data of the
# first adds. The next adds wait with `vmcnt(2)`. This in-order completion is an
# important difference from NVIDIA.
#
# For loads into registers, as in this kernel, the compiler manages the wait
# counters. It inserts each wait right before the consumer instruction. A kernel
# can also load data from global memory directly to shared memory. In this case
# the kernel needs to wait for these copies itself. Gluon exposes this operation
# as `global_load_to_shared`. `commit_group` groups the copies issued since the
# last commit, and `wait_group(n)` waits until at most `n` groups are still in
# flight.


@gluon.jit
def elementwise_async_copy_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    pid = gl.program_id(0)
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]
        cp.global_load_to_shared(a_smem, a_ptrs + ystride_a * yoffs[None, :], mask)
        cp.global_load_to_shared(b_smem, b_ptrs + ystride_b * yoffs[None, :], mask)
        cp.commit_group()
        cp.wait_group(0)

        a_val = a_smem.load(layout)
        b_val = b_smem.load(layout)
        gl.store(c_ptrs + ystride_c * yoffs[None, :], a_val + b_val, mask=mask)


# %%
# Buffer operations can also load directly to shared memory. `load_to_shared`
# takes the same descriptor and offsets as a buffer load. It writes zeros for
# masked-off and out-of-bounds elements.


@gluon.jit
def elementwise_async_copy_w_buffer_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    pid = gl.program_id(0)
    xoff = pid * XBLOCK
    a_desc = t1.make_buffer_descriptor(a_ptr, xoff, xnumel, xstride_a, XBLOCK)
    b_desc = t1.make_buffer_descriptor(b_ptr, xoff, xnumel, xstride_b, XBLOCK)
    c_desc = t1.make_buffer_descriptor(c_ptr, xoff, xnumel, xstride_c, XBLOCK)

    xidx = gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    yidx = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    a_offs = xstride_a * xidx[:, None] + ystride_a * yidx[None, :]
    b_offs = xstride_b * xidx[:, None] + ystride_b * yidx[None, :]
    c_offs = xstride_c * xidx[:, None] + ystride_c * yidx[None, :]

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    for yoff in range(0, ynumel, YBLOCK):
        mask = (yoff + yidx < ynumel)[None, :]
        buffer.load_to_shared(a_smem, a_desc, a_offs, mask, soffset=yoff * ystride_a)
        buffer.load_to_shared(b_smem, b_desc, b_offs, mask, soffset=yoff * ystride_b)
        cp.commit_group()
        cp.wait_group(0)

        a_val = a_smem.load(layout)
        b_val = b_smem.load(layout)
        buffer.store(a_val + b_val, c_desc, c_offs, mask, soffset=yoff * ystride_c)


def elementwise_async_copy(A, B, C, kernel, XBLOCK=32, YBLOCK=256, num_warps=4):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [num_warps, 1], [1, 0])
    # With these parameters, `SwizzledSharedLayout` is a plain shared memory
    # layout without padding or swizzling.
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, layout, smem_layout, num_warps=num_warps)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 160)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 256), (4, 512)])
@pytest.mark.parametrize("kernel", [elementwise_async_copy_kernel, elementwise_async_copy_w_buffer_kernel])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_elementwise_async_copy(xnumel, ynumel, XBLOCK, YBLOCK, kernel):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise_async_copy(a, b, c, kernel, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


if __name__ == "__main__" and is_cdna4():
    for kernel in [elementwise_async_copy_kernel, elementwise_async_copy_w_buffer_kernel]:
        print(f"{kernel.fn.__name__}:")
        t1.print_asm_summary(elementwise_async_copy(A, B, C, kernel, XBLOCK=16))

# %%
# ```
# elementwise_async_copy_kernel:
#   227 instructions, 24 branches, 36 VGPRs
#   global_load_lds_dwordx4 v[0:1], off
# elementwise_async_copy_w_buffer_kernel:
#   146 instructions, 2 branches, 50 VGPRs
#   buffer_load_dwordx4 v14, s[16:19], s5 offen lds
# ```
#
# The two copies compare like the loads in the previous tutorial. The global
# async copy, `global_load_lds_dwordx4`, takes a 64-bit address in `v[0:1]`. The
# buffer async copy is `buffer_load_dwordx4` with the `lds` flag. It takes a
# 32-bit offset in `v14`, the descriptor in `s[16:19]`, and `soffset` in `s5`.

# %%
# You can also see that neither instruction names a shared memory address
# in its operands. The address actually comes from `M0`, a special SGPR that
# these instructions read implicitly. The compiler sets `M0` with
# `s_mov_b32 m0, ...` before each copy.
#
# This puts a strict constraint on where the data goes. In
# `buffer_load_dwordx4 ... lds`, each lane loads 16 bytes, and lane `i` can only
# write its 16 bytes at shared memory offset `16 * i` from the address in `M0`.
# So one instruction always writes 1 KB of consecutive shared memory, without
# any gap in between.


@contextlib.contextmanager
def suppress_stderr():
    # The compiler prints errors from C++, which `contextlib.redirect_stderr`
    # can't catch. So we redirect the file descriptor itself.
    stderr = os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(stderr, 2)
        os.close(stderr)


def try_compile_elementwise(layout, XBLOCK, YBLOCK):
    A = torch.randn(XBLOCK, YBLOCK, device="cuda")
    B = torch.randn_like(A)
    C = torch.empty_like(A)
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    try:
        with suppress_stderr():
            elementwise_async_copy_w_buffer_kernel[(1, )](A, B, C, XBLOCK, YBLOCK, *A.stride(), *B.stride(),
                                                          *C.stride(), XBLOCK, YBLOCK, layout, smem_layout, num_warps=4)
        return True
    except RuntimeError:
        return False


if __name__ == "__main__" and is_cdna4():
    layout_r1 = gl.BlockedLayout([1, 4], [1, warp_size], [4, 1], [1, 0])
    layout_r8 = gl.BlockedLayout([1, 4], [8, warp_size // 8], [4, 1], [1, 0])

    # The 64 lanes of a warp cover one contiguous row of 256 values.
    assert try_compile_elementwise(layout_r1, 32, 256)
    # The 8 rows of a warp are 256 values apart.
    assert not try_compile_elementwise(layout_r8, 32, 256)
    # The 8 rows of a warp are adjacent.
    assert try_compile_elementwise(layout_r8, 32, 32)

    # Inspect the generated kernel instructions.
    kernel = elementwise_async_copy(A, B, C, elementwise_async_copy_w_buffer_kernel, XBLOCK=16)
    t1.print_instructions(kernel, r"m0|lds|vmcnt|s_barrier|ds_read")

# %%
# ```
#   s_mov_b32 m0, s4
#   s_barrier
#   buffer_load_dwordx4 v14, s[16:19], s5 offen lds
#   ...
#   s_waitcnt vmcnt(0)
#   s_barrier
#   ds_read_b128 v[14:17], v13
#   ...
# ```
#
# As you can see, the compiler sets `M0` before every copy, here with
# `s_mov_b32 m0, s4`. Each copy writes a different 1 KB of the tile, so each
# copy needs its own `M0` value.
#
# Besides `M0`, the assembly code also shows how the warps synchronize around
# the copies, with a wait and two barriers. The wait only covers the copies of
# the current warp, and other warps copy other parts of the tile. So the warps
# need to meet at a CTA barrier, `s_barrier`, before they read the tile. Another
# barrier keeps a copy from overwriting a buffer that other warps may still
# read. These barriers are inserted by the compiler analysis.

# %%
# Next, we build a software-pipelined version of the kernel.


@gluon.jit
def issue_loads(copy_idx, a_smem, b_smem, a_desc, b_desc, a_offs, b_offs, yidx, ynumel, ystride_a, ystride_b,
                YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    yoff = copy_idx * YBLOCK
    mask = (yoff + yidx < ynumel)[None, :]
    buffer.load_to_shared(a_smem.index(copy_idx % num_buffers), a_desc, a_offs, mask, soffset=yoff * ystride_a)
    buffer.load_to_shared(b_smem.index(copy_idx % num_buffers), b_desc, b_offs, mask, soffset=yoff * ystride_b)
    cp.commit_group()
    return copy_idx + 1


@gluon.jit
def perform_add(read_idx, a_smem, b_smem, c_desc, c_offs, yidx, ynumel, ystride_c, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr, layout: gl.constexpr):
    a_val = a_smem.index(read_idx % num_buffers).load(layout)
    b_val = b_smem.index(read_idx % num_buffers).load(layout)
    yoff = read_idx * YBLOCK
    mask = (yoff + yidx < ynumel)[None, :]
    buffer.store(a_val + b_val, c_desc, c_offs, mask, soffset=yoff * ystride_c)
    return read_idx + 1


@gluon.jit
def elementwise_pipelined_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr, num_buffers: gl.constexpr,  #
):
    pid = gl.program_id(0)
    xoff = pid * XBLOCK
    a_desc = t1.make_buffer_descriptor(a_ptr, xoff, xnumel, xstride_a, XBLOCK)
    b_desc = t1.make_buffer_descriptor(b_ptr, xoff, xnumel, xstride_b, XBLOCK)
    c_desc = t1.make_buffer_descriptor(c_ptr, xoff, xnumel, xstride_c, XBLOCK)

    xidx = gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    yidx = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    a_offs = xstride_a * xidx[:, None] + ystride_a * yidx[None, :]
    b_offs = xstride_b * xidx[:, None] + ystride_b * yidx[None, :]
    c_offs = xstride_c * xidx[:, None] + ystride_c * yidx[None, :]

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    copy_idx = 0
    read_idx = 0

    for _ in gl.static_range(num_buffers - 1):
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_desc, b_desc, a_offs, b_offs, yidx, ynumel, ystride_a,
                               ystride_b, YBLOCK, num_buffers)

    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_desc, b_desc, a_offs, b_offs, yidx, ynumel, ystride_a,
                               ystride_b, YBLOCK, num_buffers)
        cp.wait_group(num_buffers - 1)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_desc, c_offs, yidx, ynumel, ystride_c, YBLOCK, num_buffers,
                               layout)

    for i in gl.static_range(num_buffers - 1):
        cp.wait_group(num_buffers - 2 - i)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_desc, c_offs, yidx, ynumel, ystride_c, YBLOCK, num_buffers,
                               layout)


def elementwise_pipelined(A, B, C, XBLOCK=32, YBLOCK=256, num_buffers=2, num_warps=4):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [num_warps, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_pipelined_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, layout, smem_layout, num_buffers, num_warps=num_warps)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 160)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 256), (4, 512)])
@pytest.mark.parametrize("num_buffers", [1, 2])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_elementwise_pipelined(xnumel, ynumel, XBLOCK, YBLOCK, num_buffers):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise_pipelined(a, b, c, XBLOCK, YBLOCK, num_buffers)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


# %%
# In the pipelined loop, a store from the previous iteration sits between the
# copies. Since stores share the `vmcnt` counter, the compiler counts them too.

if __name__ == "__main__" and is_cdna4():
    kernel = elementwise_pipelined(A, B, C, XBLOCK=16, num_buffers=2)
    t1.print_instructions(kernel, r"vmcnt")

# %%
# ```
#   s_waitcnt vmcnt(8)
#   s_waitcnt vmcnt(12)
#   s_waitcnt vmcnt(8)
# ```
#
# Each commit group has 8 instructions per warp. `vmcnt(12)` keeps the 8 copies
# of the next group and the 4 stores that were issued after the group we wait
# for. Where the compiler can't count the stores, for example across loop
# iterations, it waits for them too (`vmcnt(8)`).

if __name__ == "__main__" and is_cdna4():
    print("Benchmarking elementwise add")
    print("============================")
    xnumel, ynumel = 16 * 1024, 16 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A)

    def bench(name, fn):
        ms = triton.testing.do_bench(fn)
        print(f"{name}: {t1.get_throughput(ms, C):.2f} TB/s")

    XBLOCK, YBLOCK = 4, 512
    bench("global load", lambda: t1.elementwise(A, B, C, t1.elementwise_kernel, XBLOCK, YBLOCK))
    bench("global async copy", lambda: elementwise_async_copy(A, B, C, elementwise_async_copy_kernel, XBLOCK, YBLOCK))
    bench("buffer async copy",
          lambda: elementwise_async_copy(A, B, C, elementwise_async_copy_w_buffer_kernel, XBLOCK, YBLOCK))
    bench("pipelined (double buffer)", lambda: elementwise_pipelined(A, B, C, XBLOCK, YBLOCK, 2))

# %%
# ```
# global load: 4.74 TB/s
# global async copy: 5.17 TB/s
# buffer async copy: 5.20 TB/s
# pipelined (double buffer): 5.24 TB/s
# ```
#
# Async copies are 9% faster than global loads, and both kinds of async copies
# reach about the same bandwidth.

# %%
# The double buffer makes the kernel only 1% faster than the buffer async copy.
# More buffers keep more copies in flight, but every buffer of a pipelined
# kernel also takes shared memory. Keep this budget in mind when you choose the
# number of buffers.
#
# A compute unit (CU) is the AMD equivalent of an SM. On CDNA4, each CU has
# 160 KB of shared memory (LDS) and runs several CTAs at the same time if their
# resources fit. The CTAs on a CU also hide the memory latency of each other. So
# more buffers are not always better. Each buffer keeps more copies in flight
# within a CTA, but it also takes shared memory, and fewer CTAs fit on a CU.
#
# For example, with 4x512 fp32 tiles, two buffers for A and B take 32 KB, and up
# to 5 CTAs fit on a CU. With 32x256 tiles, two buffers take 128 KB, and only
# one CTA fits, while two CTAs fit without pipelining. For memory-bound kernels
# with small tiles, one or two buffers are usually enough. Add more buffers only
# when a CTA has compute to overlap.

# %%
# CDNA5 async copy
# ----------------
#
# CDNA5 also provides async copies from global memory to shared memory. In
# Gluon, they are available as `global_to_shared` in `cdna5.async_copy`, and
# they use the same `commit_group` and `wait_group` as on CDNA4.
#
# Another thing changed in CDNA5 changes are  wait counters. CDNA5 offers the
# following counters:
#
# - `loadcnt` counts global loads into registers.
# - `storecnt` counts global stores.
# - `asynccnt` counts async copies between global memory and shared memory,
# - `tensorcnt` counts TDM copies, which we introduce in the next tutorial.
# - `dscnt` counts shared memory accesses.
#
# Each counter also has its own wait instruction, such as `s_wait_loadcnt` and
# `s_wait_asynccnt`, instead of one `s_waitcnt` for all counters. So a wait for
# one kind of instruction no longer waits for the other kinds.
#
# Let's use the same elementwise kernel as before to demonstrate async copies.


@gluon.jit
def elementwise_async_copy_cdna5_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    pid = gl.program_id(0)
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]
        cdna5.async_copy.global_to_shared(a_smem, a_ptrs + ystride_a * yoffs[None, :], mask)
        cdna5.async_copy.global_to_shared(b_smem, b_ptrs + ystride_b * yoffs[None, :], mask)
        cdna5.async_copy.commit_group()
        cdna5.async_copy.wait_group(0)

        a_val = a_smem.load(layout)
        b_val = b_smem.load(layout)
        gl.store(c_ptrs + ystride_c * yoffs[None, :], a_val + b_val, mask=mask)


def elementwise_async_copy_cdna5(A, B, C, XBLOCK=32, YBLOCK=256, num_warps=4):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [num_warps, 1], [1, 0])
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_async_copy_cdna5_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, layout, smem_layout, num_warps=num_warps)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 160)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 256), (4, 512)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_elementwise_async_copy_cdna5(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise_async_copy_cdna5(a, b, c, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


if __name__ == "__main__" and is_cdna5():
    A = torch.randn(1024, 2048, device="cuda")
    B = torch.randn_like(A)
    C = torch.empty_like(A)
    kernel = elementwise_async_copy_cdna5(A, B, C, XBLOCK=16)
    t1.print_instructions(kernel, r"m0|lds|_wait_asynccnt")

# %%
# ```
#   global_load_async_to_lds_b128 v24, v[2:3], off
#   ...
#   s_wait_asynccnt 0x0
# ```
#
# The copy instruction is `global_load_async_to_lds_b128`. Like a global load,
# it takes a 64-bit address in `v[2:3]`. Unlike CDNA4, it also takes a shared
# memory address for each lane, here in `v24`, and it doesn't use `M0`. The
# kernel doesn't set `M0` at all.
#
# The wait also differs. `wait_group` becomes `s_wait_asynccnt`, which only
# waits for async copies, and never for stores or loads into registers.
#
# Since each lane has its own shared memory address, one copy instruction
# doesn't have to fill consecutive shared memory. On CDNA4, the layout that
# spreads the lanes of a warp over 8 rows fails to compile on a tile with 256
# columns. On CDNA5, it works.

# %%
# Next, we measure the bandwidth on an MI455X GPU.

if __name__ == "__main__" and is_cdna5():
    print("Benchmarking elementwise add")
    print("============================")
    xnumel, ynumel = 16 * 1024, 16 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A)

    def bench(name, fn):
        ms = triton.testing.do_bench(fn)
        print(f"{name}: {t1.get_throughput(ms, C):.2f} TB/s")

    def global_load(XBLOCK, YBLOCK):
        layout = gl.BlockedLayout([1, 4], [1, warp_size], [4, 1], [1, 0])
        t1.elementwise_kernel[(triton.cdiv(xnumel, XBLOCK), )](A, B, C, xnumel, ynumel, *A.stride(), *B.stride(),
                                                               *C.stride(), XBLOCK, YBLOCK, layout, num_warps=4)

    XBLOCK, YBLOCK = 4, 512
    bench("global load", lambda: global_load(XBLOCK, YBLOCK))
    bench("async copy", lambda: elementwise_async_copy_cdna5(A, B, C, XBLOCK, YBLOCK))

# %%
# ```
# global load: 13.13 TB/s
# async copy: 17.69 TB/s
# ```
#
# Async copies are 35% faster than global loads.
