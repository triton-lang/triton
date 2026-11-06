"""
Buffer Operations
=================

The Gluon tutorials in `python/tutorials/gluon` target NVIDIA GPUs. This series
covers AMD GPUs. We assume that you already know the basic Gluon concepts,
especially layouts. Many ideas from the NVIDIA tutorials carry over to AMD GPUs,
so this series focuses on the Gluon features that are specific to AMD GPUs.

We start with buffer operations, the AMD-specific way to access global memory. A
buffer operation addresses memory with a buffer resource descriptor and offsets.
The descriptor holds a base address and a size. The hardware checks each access
against the size. Out-of-bounds loads return zero, and out-of-bounds stores are
dropped. So a kernel often doesn't need a mask.

The limitation of buffer operations is that the offsets must fit in 32 bits. So
one descriptor covers at most 4 GB. To access a larger buffer, the kernel must
create a new descriptor with an updated base address and size. Each program can
make its own descriptors with a different base address, so a kernel as a whole
can access much more than 4 GB.
"""

import re
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import buffer


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx950"


if __name__ == "__main__" and not is_cdna4():
    raise RuntimeError("This tutorial requires an AMD CDNA4 GPU")

# A warp on CDNA4 GPUs has 64 lanes.
warp_size = 64

# %%
# Buffer operations build the address from three parts:
#
# ```
# address = base + soffset + voffset
# ```
#
# - `base` comes from a buffer resource descriptor. The descriptor is 128 bits
#   wide and lives in 4 scalar registers (SGPRs). It holds the base address and
#   the size of the buffer in bytes, called `num_records`. The hardware checks
#   `soffset + voffset` against `num_records`.
# - `soffset` is a 32-bit offset in one SGPR. It is the same for all lanes.
# - `voffset` is a 32-bit offset in one vector register (VGPR). Each lane has
#   its own.
#
# Let's start with an elementwise add kernel using standard global load
# and store.


@gluon.jit
def elementwise_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
):
    pid = gl.program_id(0)
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]
        a_val = gl.load(a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        b_val = gl.load(b_ptrs + ystride_b * yoffs[None, :], mask=mask)
        gl.store(c_ptrs + ystride_c * yoffs[None, :], a_val + b_val, mask=mask)


# %%
# To use buffer operations, each program makes its own descriptors. The base
# pointer points to the first element of the program, and the size covers the
# bytes of all elements that the program owns.
#
# The hardware only checks that an offset is below `num_records`. An offset can
# be wrong for the kernel and still be in range. For example, when `ynumel` is
# not a multiple of `YBLOCK`, the last loop iteration has lanes whose offsets
# point into the data of the next row. So we still need to mask for certain
# dimensions. The compiler applies the mask without branches. It replaces the
# offsets of masked-off lanes with an out-of-bounds offset, and the hardware
# drops these accesses.
#
# The per-lane offsets are computed once, before the loop. In the loop, we only
# move `soffset`, which is a single scalar.


@gluon.jit
def make_buffer_descriptor(ptr, xoff, xnumel, xstride, XBLOCK: gl.constexpr):
    nrows = gl.minimum(xnumel - xoff, XBLOCK)
    nbytes = nrows * xstride * ptr.dtype.element_ty.primitive_bitwidth // 8
    return buffer.make_buffer_descriptor(ptr + xoff * xstride, nbytes)


@gluon.jit
def elementwise_w_buffer_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr,  #
):
    pid = gl.program_id(0)
    xoff = pid * XBLOCK
    a_desc = make_buffer_descriptor(a_ptr, xoff, xnumel, xstride_a, XBLOCK)
    b_desc = make_buffer_descriptor(b_ptr, xoff, xnumel, xstride_b, XBLOCK)
    c_desc = make_buffer_descriptor(c_ptr, xoff, xnumel, xstride_c, XBLOCK)

    xidx = gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    yidx = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    a_offs = xstride_a * xidx[:, None] + ystride_a * yidx[None, :]
    b_offs = xstride_b * xidx[:, None] + ystride_b * yidx[None, :]
    c_offs = xstride_c * xidx[:, None] + ystride_c * yidx[None, :]

    for yoff in range(0, ynumel, YBLOCK):
        mask = (yoff + yidx < ynumel)[None, :]
        a_val = buffer.load(a_desc, a_offs, mask, soffset=yoff * ystride_a)
        b_val = buffer.load(b_desc, b_offs, mask, soffset=yoff * ystride_b)
        buffer.store(a_val + b_val, c_desc, c_offs, mask, soffset=yoff * ystride_c)


def elementwise(A, B, C, kernel, XBLOCK=32, YBLOCK=256, num_warps=4):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape

    # In this layout, each lane accesses 4 consecutive fp32 values, which is one
    # 128-bit access.
    layout = gl.BlockedLayout([1, 4], [1, warp_size], [num_warps, 1], [1, 0])
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, layout, num_warps=num_warps)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 120)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 256), (64, 128)])
@pytest.mark.parametrize("kernel", [elementwise_kernel, elementwise_w_buffer_kernel])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_elementwise(xnumel, ynumel, XBLOCK, YBLOCK, kernel):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise(a, b, c, kernel, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


# %%
# The compiled kernel keeps its assembly code in `kernel.asm["amdgcn"]`. We
# count the instructions and the branches, and read the number of VGPRs from the
# kernel metadata.


def get_vgpr_count(kernel):
    return int(re.search(r"\.vgpr_count:\s+(\d+)", kernel.asm["amdgcn"]).group(1))


def get_instructions(kernel):
    asm = kernel.asm["amdgcn"]
    body = asm[asm.index(".text"):asm.index(".Lfunc_end")]
    lines = [line.strip() for line in body.splitlines()]
    return [line for line in lines if re.match(r"[a-z]", line) and not line.endswith(":")]


def print_asm_summary(kernel):
    instrs = get_instructions(kernel)
    num_branches = sum(instr.startswith("s_cbranch") for instr in instrs)
    print(f"  {len(instrs)} instructions, {num_branches} branches, {get_vgpr_count(kernel)} VGPRs")
    print(f"  {next(instr for instr in instrs if re.match('(global|buffer)_load', instr))}")


if __name__ == "__main__":
    A = torch.randn(1024, 1024, device="cuda")
    B = torch.randn_like(A)
    C = torch.empty_like(A)
    for kernel in [elementwise_kernel, elementwise_w_buffer_kernel]:
        print(f"{kernel.fn.__name__}:")
        print_asm_summary(elementwise(A, B, C, kernel))

# %%
# ```
# elementwise_kernel:
#   351 instructions, 22 branches, 67 VGPRs
#   global_load_dwordx4 v[4:7], v[2:3], off
# elementwise_w_buffer_kernel:
#   180 instructions, 2 branches, 63 VGPRs
#   buffer_load_dwordx4 v[26:29], v25, s[16:19], s4 offen
# ```
#
# A global load uses the `global_load_dwordx4` instruction. Each lane passes a
# 64-bit address in two VGPRs, here `v[2:3]`. The `buffer_load_dwordx4`
# instruction takes a 32-bit offset in one VGPR, `v25`, the descriptor in
# `s[16:19]`, and `soffset` in `s4`.
#
# For masked global load, the compiler has to generate branches for each lane
# to check if the address is in bounds. That's why many branches appear in the
# generated assembly code. However in the buffer version, the bounds check is
# done natively by the hardware so all most of the branches are eliminated.
#

# %%
# Next, we measure the bandwidth of the two kernels on an MI355X GPU.


def get_throughput(ms, C):
    tbytes = 3 * C.numel() * C.element_size() / 1e12
    return tbytes / (ms * 1e-3)


if __name__ == "__main__":
    print("Benchmarking elementwise add")
    print("============================")
    xnumel, ynumel = 16 * 1024, 16 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A)

    for kernel in [elementwise_kernel, elementwise_w_buffer_kernel]:
        ms = triton.testing.do_bench(lambda: elementwise(A, B, C, kernel))
        print(f"{kernel.fn.__name__}: {get_throughput(ms, C):.2f} TB/s")

# %%
# ```
# elementwise_kernel: 4.36 TB/s
# elementwise_w_buffer_kernel: 4.90 TB/s
# ```
#
# The buffer kernel is 12% faster. The branches around the masked global loads
# cost about 5%. The rest comes from the smaller addresses. Each lane passes a
# 32-bit offset instead of a 64-bit address, so the hardware spends less time
# processing the addresses of each memory instruction.

# %%
# The hardware checks the bounds at a 4-byte granularity. For 32-bit data, the
# check is exact for each element. However, for 16-bit or 8-bit data, if the
# size is not a multiple of 4 bytes, the last valid elements are also out of
# bounds and read zero. To avoid this, round `num_records` up to a multiple of
# 4, and mask the elements past the end. The following example shows both cases.


@gluon.jit
def copy_kernel(src_ptr, dst_ptr, numel, nbytes, BLOCK: gl.constexpr, layout: gl.constexpr):
    desc = buffer.make_buffer_descriptor(src_ptr, nbytes)
    offs = gl.arange(0, BLOCK, layout)
    gl.store(dst_ptr + offs, buffer.load(desc, offs), mask=offs < numel)


if __name__ == "__main__":
    numel = 1001
    x = torch.arange(1, numel + 1, device="cuda", dtype=torch.float16)
    y = torch.zeros_like(x)
    layout = gl.BlockedLayout([8], [warp_size], [1], [0])
    for nbytes in [2 * numel, triton.cdiv(2 * numel, 4) * 4]:
        copy_kernel[(1, )](x, y, numel, nbytes, 1024, layout, num_warps=1)
        print(f"num_records={nbytes}: last values {y[-3:].tolist()}")

# %%
# ```
# num_records=2002: last values [999.0, 1000.0, 0.0]
# num_records=2004: last values [999.0, 1000.0, 1001.0]
# ```
#
# We copy 1001 fp16 values, once with a descriptor of exactly 2002 bytes and
# once with the size rounded up to 2004 bytes. Each lane loads 8 values
# (128 bits) at once, and the store masks the elements past the end. With 2002
# bytes, the last value reads zero.
