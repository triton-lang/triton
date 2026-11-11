"""
MFMA and WMMA
=============

AMD GPUs run matrix multiplications on dedicated matrix cores. CDNA4 GPUs use
matrix fused-multiply-add (MFMA) instructions, and CDNA5 GPUs use wave matrix
multiply-accumulate (WMMA) instructions. Both read their operands from registers
and accumulate in registers. They can't consume shared memory (LDS) directly,
so the kernel needs extra instructions to load every operand from LDS into
registers, and it must load them fast enough to keep the matrix cores busy.

This tutorial first introduces MFMA and its layouts with the bf16 data type on
CDNA4, and the ways to load the operands from LDS. Then it covers WMMA on CDNA5,
where the LDS adds a new kind of conflict.
"""

import importlib
import re
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4, cdna5
from triton.experimental.gluon.language.amd.cdna4 import async_copy, buffer
from triton.experimental.gluon.language.amd.cdna4 import compute_async_copy_layout, compute_efficient_padded_shared_layout
from triton.experimental.gluon.language.amd.cdna5 import tdm

t1 = importlib.import_module("01-buffer-ops")


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx950"


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not (is_cdna4() or is_cdna5()):
    raise RuntimeError("This tutorial requires an AMD CDNA4 or CDNA5 GPU")

# %%
# CDNA4 MFMA
# ----------
#
# A warp on CDNA4 GPUs has 64 lanes.
#
# Gluon exposes MFMA as `mfma(a, b, acc)`. This operation has layout
# requirements for its operands. The accumulator `acc` must use an
# `AMDMFMALayout`, and the operands `a` and `b` must use a `DotOperandLayout`
# with an MFMA layout as parent. These layouts control which MFMA instructions
# the compiler generates.
#
# The operand data type and `instr_shape` select the MFMA instruction of each
# warp. `instr_shape` is the shape of the instruction in [M, N, K] format. With
# multiple warps, `warps_per_cta` distributes them over the accumulator along
# the M and N dimensions. These need to be chosen based on the block size.


@gluon.jit
def mfma_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,  #
                INSTR_SHAPE: gl.constexpr, WARPS_PER_CTA: gl.constexpr):
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=INSTR_SHAPE, transposed=True, warps_per_cta=WARPS_PER_CTA)
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

    am = gl.arange(0, M, gl.SliceLayout(1, a_layout))
    ak = gl.arange(0, K, gl.SliceLayout(0, a_layout))
    a = gl.load(a_ptr + am[:, None] * K + ak[None, :])
    bk = gl.arange(0, K, gl.SliceLayout(1, b_layout))
    bn = gl.arange(0, N, gl.SliceLayout(0, b_layout))
    b = gl.load(b_ptr + bk[:, None] * N + bn[None, :])

    acc = gl.zeros([M, N], gl.float32, mfma_layout)
    acc = cdna4.mfma(a, b, acc)

    cm = gl.arange(0, M, gl.SliceLayout(1, mfma_layout))
    cn = gl.arange(0, N, gl.SliceLayout(0, mfma_layout))
    gl.store(c_ptr + cm[:, None] * N + cn[None, :], acc)


def mfma(A, B, instr_shape, warps_per_cta):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    num_warps = warps_per_cta[0] * warps_per_cta[1]
    mfma_kernel[(1, )](A, B, C, M, N, K, tuple(instr_shape), tuple(warps_per_cta), num_warps=num_warps)
    return C


@pytest.mark.parametrize("M, N, K", [(32, 32, 32), (64, 128, 64)])
# For bf16, CDNA4 has two MFMA instructions shapes:
# - `v_mfma_f32_16x16x32_bf16`
# - `v_mfma_f32_32x32x16_bf16`.
@pytest.mark.parametrize("instr_shape", [[16, 16, 32], [32, 32, 16]])
@pytest.mark.parametrize("warps_per_cta", [[2, 2], [4, 1]])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_mfma(M, N, K, instr_shape, warps_per_cta):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = mfma(A, B, instr_shape, warps_per_cta)
    torch.testing.assert_close(A.float() @ B.float(), C, atol=1e-3, rtol=1e-3)


# %%
# Let's examine what the MFMA layouts look like.

if __name__ == "__main__" and is_cdna4():
    mfma_layout = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, 1])
    print(gl.DotOperandLayout(0, mfma_layout, 8).format_tensor_view([16, 32]))
    print(mfma_layout.format_tensor_view([16, 16]))

# %%
# ```
# [[ T0:0,  T0:1, ...,  T0:7, T16:0, ..., T16:7, T32:0, ..., T48:7]
# [  T1:0,  T1:1, ...,  T1:7, T17:0, ..., T17:7, T33:0, ..., T49:7]
# ...
# [ T15:0, T15:1, ..., T15:7, T31:0, ..., T31:7, T47:0, ..., T63:7]]
# ```
#
# For operand A of one warp, lane `i` holds row `i % 16`, and each lane holds 8
# consecutive elements along K. Lanes 0-15 hold K 0-7, lanes 16-31 hold K 8-15,
# and so on. Operand B uses the same pattern, with lane `i` holding column
# `i % 16`.
#
# ```
# [[ T0:0,  T1:0,  T2:0, ..., T15:0]
# [  T0:1,  T1:1,  T2:1, ..., T15:1]
# [  T0:2,  T1:2,  T2:2, ..., T15:2]
# [  T0:3,  T1:3,  T2:3, ..., T15:3]
# [ T16:0, T17:0, T18:0, ..., T31:0]
# ...
# [ T48:3, T49:3, T50:3, ..., T63:3]]
# ```
#
# In the accumulator, lane `i` holds column `i % 16`. Lanes 0-15 hold rows 0-3,
# lanes 16-31 hold rows 4-7, and so on. Each lane holds 4 consecutive elements
# of a column in 4 registers.

# %%
# Understanding the MFMA layouts can help us do some optimization tricks.
# One typical case is for layout conversion. In kernel like attention, there are
# 2 chained MFMAs, where the output of the first is used as an operand of the
# second MFMA (with softmax in between). From the layout view shown above, the
# operand and output layouts are different. We can use the `transposed` flag
# and `k_width` to adjust them and avoid this conversion.

# %%
# With `transposed=True`, each lane of the accumulator holds 4 consecutive
# elements of a row instead of a column. This makes it closer to the layout
# of operand A in the next MFMA.

if __name__ == "__main__" and is_cdna4():
    transposed_layout = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
    print(transposed_layout.format_tensor_view([16, 16]))

# %%
# ```
# [[ T0:0,  T0:1,  T0:2,  T0:3, T16:0, ..., T16:3, T32:0, ..., T48:3]
# [  T1:0,  T1:1,  T1:2,  T1:3, T17:0, ..., T17:3, T33:0, ..., T49:3]
# ...
# [ T15:0, T15:1, T15:2, T15:3, T31:0, ..., T31:3, T47:0, ..., T63:3]]
# ```

# %%
# To make the two layouts identical, we also need `k_width`. `k_width` means the
# number of consecutive elements along K that a lane holds in an MFMA operand.
# The 16x16x32 MFMA instruction requires 8 values of A and 8 values of B per
# lane, which means the `k_width=8`. However, we can use a different `k_width`
# unless it is the same for both operands. This will still produce the correct
# results - essentially, we are swapping the order of elements along the K
# dimension. We can use `k_width=4`, then each lane holds 4 consecutive elements
# which exactly match the transposed accumulator layout.

if __name__ == "__main__" and is_cdna4():
    a_view = gl.DotOperandLayout(0, transposed_layout, 4).format_tensor_view([16, 32])
    print(a_view)
    assert a_view == transposed_layout.format_tensor_view([16, 32])

# %%
# ```
# [[ T0:0,  T0:1,  T0:2,  T0:3, T16:0, ..., T48:3,  T0:4, ..., T48:7]
# [  T1:0,  T1:1,  T1:2,  T1:3, T17:0, ..., T49:3,  T1:4, ..., T49:7]
# ...
# [ T15:0, T15:1, T15:2, T15:3, T31:0, ..., T63:3, T15:4, ..., T63:7]]
# ```

# %%
# Now we can chain two MFMAs together without any layout conversion.


@gluon.jit
def offs_2d(rows: gl.constexpr, cols: gl.constexpr, stride, layout: gl.constexpr):
    r = gl.arange(0, rows, gl.SliceLayout(1, layout))
    c = gl.arange(0, cols, gl.SliceLayout(0, layout))
    return r[:, None] * stride + c[None, :]


@gluon.jit
def mfma_chain_kernel(a_ptr, b_ptr, c_ptr, out_ptr, BLOCK: gl.constexpr):
    warps: gl.constexpr = gl.num_warps()
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[warps, 1])
    a = gl.load(a_ptr + offs_2d(BLOCK, BLOCK, BLOCK, gl.DotOperandLayout(0, mfma_layout, 8)))
    b = gl.load(b_ptr + offs_2d(BLOCK, BLOCK, BLOCK, gl.DotOperandLayout(1, mfma_layout, 8)))
    s = cdna4.mfma(a, b, gl.zeros([BLOCK, BLOCK], gl.float32, mfma_layout))

    # `assert_trivial=True` makes the compilation fail if the conversion happens
    p = gl.convert_layout(s.to(gl.bfloat16), gl.DotOperandLayout(0, mfma_layout, 4), assert_trivial=True)
    c = gl.load(c_ptr + offs_2d(BLOCK, BLOCK, BLOCK, gl.DotOperandLayout(1, mfma_layout, 4)))
    out = cdna4.mfma(p, c, gl.zeros([BLOCK, BLOCK], gl.float32, mfma_layout))
    gl.store(out_ptr + offs_2d(BLOCK, BLOCK, BLOCK, mfma_layout), out)


@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_mfma_chain():
    A, B, C = (torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = torch.empty(64, 64, device="cuda", dtype=torch.float32)
    mfma_chain_kernel[(1, )](A, B, C, out, 64, num_warps=4)
    ref = (A.float() @ B.float()).to(torch.bfloat16).float() @ C.float()
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)


# %%
# The layouts above are for registers. Real matrix multiplication kernels
# usually load the data into shared memory with async copies first, and then
# copy it from shared memory to registers. This needs extra layouts for shared
# memory. These layouts determine which LDS instructions the compiler uses and
# whether there are bank conflicts.
#
# On CDNA4, there are two kinds of LDS reads for MFMA operands. A regular read,
# such as `ds_read_b128`, loads consecutive bytes for each lane. It fits an
# operand whose K dimension is contiguous in shared memory. A transposed read,
# such as `ds_read_b64_tr_b16`, handles an operand stored with M or N
# contiguous, for example a row-major B with shape [K, N]. Each lane reads 64
# contiguous bits, and the hardware exchanges the data between lanes, so that
# each lane receives the values it needs along K. The compiler picks which one
# to use based on contiguity and layout.


@gluon.jit
def mfma_smem_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                     B_K_CONTIGUOUS: gl.constexpr, a_smem_layout: gl.constexpr, b_smem_layout: gl.constexpr):
    warps: gl.constexpr = gl.num_warps()
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, warps // 2])
    # Layout for global memory.
    gmem_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [warps, 1], [1, 0])

    gl.static_assert(a_ptr.dtype.element_ty == gl.bfloat16 and b_ptr.dtype.element_ty == gl.bfloat16)
    a_desc = buffer.make_buffer_descriptor(a_ptr, M * K * 2)
    b_desc = buffer.make_buffer_descriptor(b_ptr, N * K * 2)

    a_smem = gl.allocate_shared_memory(gl.bfloat16, [M, K], a_smem_layout)
    buffer.load_to_shared(a_smem, a_desc, offs_2d(M, K, K, gmem_layout))
    if B_K_CONTIGUOUS:
        b_smem = gl.allocate_shared_memory(gl.bfloat16, [N, K], b_smem_layout)
        buffer.load_to_shared(b_smem, b_desc, offs_2d(N, K, K, gmem_layout))
    else:
        b_smem = gl.allocate_shared_memory(gl.bfloat16, [K, N], b_smem_layout)
        buffer.load_to_shared(b_smem, b_desc, offs_2d(K, N, N, gmem_layout))
    async_copy.commit_group()
    async_copy.wait_group(0)

    a = a_smem.load(gl.DotOperandLayout(0, mfma_layout, 8))
    if B_K_CONTIGUOUS:
        # Permute the view of B from [N, K] to [K, N].
        b = b_smem.permute([1, 0]).load(gl.DotOperandLayout(1, mfma_layout, 8))
    else:
        b = b_smem.load(gl.DotOperandLayout(1, mfma_layout, 8))

    acc = cdna4.mfma(a, b, gl.zeros([M, N], gl.float32, mfma_layout))
    gl.store(c_ptr + offs_2d(M, N, N, mfma_layout), acc)


def mfma_smem(A, B, b_k_contiguous):
    M, K = A.shape
    N = B.shape[0] if b_k_contiguous else B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    compiled = mfma_smem_kernel[(1, )](A, B, C, M, N, K, b_k_contiguous, smem_layout, smem_layout, num_warps=4)
    return C, compiled


@pytest.mark.parametrize("b_k_contiguous", [True, False])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_mfma_smem(b_k_contiguous):
    A = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    C, _ = mfma_smem(A, B.T.contiguous() if b_k_contiguous else B, b_k_contiguous)
    torch.testing.assert_close(A.float() @ B.float(), C, atol=1e-3, rtol=1e-3)


def count_lds_reads(kernel):
    counts = {}
    for instr in t1.get_instructions(kernel):
        if instr.startswith("ds_read"):
            name = instr.split()[0]
            counts[name] = counts.get(name, 0) + 1
    return counts


if __name__ == "__main__" and is_cdna4():
    A = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    for b_k_contiguous in [True, False]:
        B = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        _, compiled = mfma_smem(A, B if b_k_contiguous else B.T.contiguous(), b_k_contiguous)
        print(f"B K-contiguous={b_k_contiguous}: {count_lds_reads(compiled)}")

# %%
# ```
# B K-contiguous=True: {'ds_read_b128': 8}
# B K-contiguous=False: {'ds_read_b64_tr_b16': 8, 'ds_read_b128': 4}
# ```
#
# With `k_width=8`, each lane holds 8 consecutive bf16 values along K for an
# MFMA, which is 16 bytes. When K is contiguous in shared memory, these 16 bytes
# are adjacent, and one `ds_read_b128` loads them. Operand A takes 4
# `ds_read_b128` in both cases.
#
# For operand B, when it is K-contiguous, the compiler uses the same regular
# reads as for A. When it is N-contiguous, the compiler switches to the
# transposed read `ds_read_b64_tr_b16`, and the data still arrives in the
# operand layout without extra shuffles.

# %%
# The kernel above uses a plain shared memory layout, which can have bank
# conflicts. Bank conflicts works sightly differently from instruction to
# instruction, but we can briefly explain it here: The LDS of a CDNA4 CU has 64
# banks, each 4 bytes wide, so it can serve 256 bytes per cycle. These 256 bytes
# don't cover all 64 lanes of a wide access, for example, a `ds_read_b128`
# can read 1KB. So one LDS instruction will be split into multiple cycle, and
# certain lanes will be selected together to form a group. Bank conflict only
# happens within a group. For `ds_read_b128`, each group consists of 16 lanes
# and they are not consecutive - the first group has lanes 0-3, 12-15, 20-23 and
# 24-27, and the second group has lanes 4-7, 8-11, 16-19 and 28-31, so on.
#
# We will not explore all these bank conflicts details. In practice, Gluon
# provided the utility - `gl.bank_conflicts` to help check bank conflict. It
# counts the number of extra accesses that bank conflicts cause between a
# register layout and a shared memory layout.


@gluon.jit
def bank_conflicts_kernel(reg_layout: gl.constexpr, smem_layout: gl.constexpr, NUM_CONFLICTS: gl.constexpr):
    smem = gl.allocate_shared_memory(gl.bfloat16, [256, 64], smem_layout)
    a = smem.load(reg_layout)
    gl.static_assert(gl.bank_conflicts(a.type, smem.type) == NUM_CONFLICTS)


if __name__ == "__main__" and is_cdna4():
    mfma_layout = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
    a_layout = gl.DotOperandLayout(0, mfma_layout, 8)
    layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, layout, NUM_CONFLICTS=3)  # 3 conflicts means a 4-way bank conflict

# %%
# There are two ways to avoid bank conflicts, swizzling and padding, both can
# be implemented using Gluon's shared memory layout.

if __name__ == "__main__" and is_cdna4():
    swizzled_layout = gl.SwizzledSharedLayout(  #
        vec=8, per_phase=1, max_phase=8, order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, swizzled_layout, 0)

    padded_layout = gl.PaddedSharedLayout.with_identity_for(  #
        # Pad 16 elements after every 64 elements
        interval_padding_pairs=[[64, 16]],  #
        shape=[256, 64],  #
        order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, padded_layout, 0)

# %%
# Choosing the optimal layout to avoid bank conflicts requires some heuristics.
# Gluon provides the `compute_efficient_padded_shared_layout` function to help
# compute an ideal shared memory layout for MFMA operands.

if __name__ == "__main__" and is_cdna4():
    mfma_layout = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
    a_layout = gl.DotOperandLayout(0, mfma_layout, 8)
    efficient_layout = compute_efficient_padded_shared_layout(a_layout, [256, 64], gl.bfloat16)
    bank_conflicts_kernel[(1, )](a_layout, efficient_layout, 0)

# %%
# With async copies, the layout of the copies in global memory must match the
# shared memory layout. `compute_async_copy_layout` can help to construct the
# corresponding load layout from a padded shared memory layout.

# %%
# On CDNA4, there are 2 types of vector registers: VGPRs, which is the regular
# vector registers, and AGPRs, for accumulation. There are 256 VGPRs and 256
# AGPRs for warp to use. All vector instructions can use VGPRs, but a few can
# use AGPRs. MFMA instructions can read their operands and accumulator from
# AGPRs and write their result to AGPRs. Some memory instructions can also
# load data into AGPRs like `ds_read_b128`.
#
# The compiler decides which type to use. By default, it uses VGPRs, and only
# moves values to AGPRs when run out of VGPRs. The `mfma` operation inGluon
# provides an experimental argument `cd_regclass` which specifies the register
# class for the accumulator (C) and output (D) . Setting `cd_regclass="a"`
# will force them in AGPRs.


@gluon.jit
def mfma_w_agpr_kernel(a_ptr, b_ptr, c_ptr, K, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                       CD_REGCLASS: gl.constexpr):
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
    a_offs = offs_2d(BLOCK_M, BLOCK_K, K, gl.DotOperandLayout(0, mfma_layout, 8))
    b_offs = offs_2d(BLOCK_K, BLOCK_N, BLOCK_N, gl.DotOperandLayout(1, mfma_layout, 8))
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mfma_layout)
    for k in range(0, K, BLOCK_K):
        a = gl.load(a_ptr + k + a_offs)
        b = gl.load(b_ptr + k * BLOCK_N + b_offs)
        acc = cdna4.mfma(a, b, acc, cd_regclass=CD_REGCLASS)
    gl.store(c_ptr + offs_2d(BLOCK_M, BLOCK_N, BLOCK_N, mfma_layout), acc)


if __name__ == "__main__" and is_cdna4():
    for BLOCK_N, cd_regclass in [(64, None), (64, "a"), (128, None), (128, "a")]:
        A = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(256, BLOCK_N, device="cuda", dtype=torch.bfloat16)
        C = torch.empty(128, BLOCK_N, device="cuda", dtype=torch.float32)
        kernel = mfma_w_agpr_kernel[(1, )](A, B, C, 256, 128, BLOCK_N, 32, cd_regclass, num_warps=1)
        torch.testing.assert_close(A.float() @ B.float(), C, atol=1e-2, rtol=1e-2)
        asm = kernel.asm["amdgcn"]
        num_vgprs = re.search(r"; NumVgprs: (\d+)", asm).group(1)
        num_agprs = re.search(r"; NumAgprs: (\d+)", asm).group(1)
        print(f"128x{BLOCK_N}, cd_regclass={cd_regclass}: {num_vgprs} VGPRs, {num_agprs} AGPRs")

# %%
# ```
# 128x64, cd_regclass=None: 222 VGPRs, 0 AGPRs
# 128x64, cd_regclass=a: 100 VGPRs, 128 AGPRs
# 128x128, cd_regclass=None: 256 VGPRs, 70 AGPRs
# 128x128, cd_regclass=a: 144 VGPRs, 256 AGPRs
# ```
#
# With a single warp, the fp32 accumulator of a 128x64 tile takes
# 128 * 64 / 64 = 128 registers per lane, and a 128x128 tile takes 256. The
# 128x64 tile fits in the VGPRs, so the compiler uses no AGPRs by default. For
# the 128x128 tile, the compiler uses all 256 VGPRs and puts the remaining
# values in AGPRs. With `cd_regclass="a"`, the accumulator takes 128 AGPRs for
# the 128x64 tile and 256 AGPRs for the 128x128 tile, and the VGPR count drops.
#
# The rule of thumb here is to keep the default when the kernel needs at most
# 256 VGPRs, and only use AGPRs when it exceeds this limit. What matters is
# which values go to AGPRs. If a kernel needs more than 256 VGPRs, the compiler
# by default moves some values to AGPRs. Ideally, these values should be the
# accumulator, because the MFMA instructions read and write it in AGPRs
# directly, without copies. Sometimes they are other values that other
# instructions need in VGPRs, so the compiler copies them back and forth, which
# impacts the performance of the MFMA instructions.

# %%
# After features introduced above, we can combine them into a pipelined matmul
# kernel. To simplify the search space, we will focus on the 256x256x64 block
# size and 4 warps setting.


@gluon.aggregate
class MfmaLayouts:
    # Operand A's global memory, shared memory, and register layouts
    a_gmem_layout: gl.constexpr
    a_smem_layout: gl.constexpr
    a_reg_layout: gl.constexpr
    # Operand B's global memory, shared memory, and register layouts
    b_gmem_layout: gl.constexpr
    b_smem_layout: gl.constexpr
    b_reg_layout: gl.constexpr
    # Accumulator's register layout
    acc_layout: gl.constexpr


@gluon.constexpr_function
def get_mfma_layouts():
    # Assume the 256x256x64 block size and 4 warps per CTA
    # Operand A has shape [256, 64]
    # Operand B has shape [256, 64] as well, so transposed

    acc_layout = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
    a_reg_layout = gl.DotOperandLayout(0, acc_layout, 8)
    b_reg_layout = gl.DotOperandLayout(1, acc_layout, 8)

    a_smem_layout = compute_efficient_padded_shared_layout(a_reg_layout, [256, 64], gl.bfloat16)
    b_smem_layout = compute_efficient_padded_shared_layout(a_reg_layout, [256, 64], gl.bfloat16)

    gmem_layout = compute_async_copy_layout(a_smem_layout, gl.bfloat16, num_warps=4)

    return MfmaLayouts(gmem_layout, a_smem_layout, a_reg_layout,  #
                       gmem_layout, b_smem_layout, b_reg_layout,  #
                       acc_layout)


@gluon.jit
def issue_async_copy(k, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem, stride_ak, stride_bk, BLOCK_K: gl.constexpr,
                     NUM_BUFFERS: gl.constexpr):
    buffer.load_to_shared(a_smem.index(k % NUM_BUFFERS), a_desc, a_offs, soffset=k * BLOCK_K * stride_ak)
    buffer.load_to_shared(b_smem.index(k % NUM_BUFFERS), b_desc, b_offs, soffset=k * BLOCK_K * stride_bk)
    async_copy.commit_group()


@gluon.jit
def load_mfma_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS: gl.constexpr):
    a_buf = a_smem.index(k % NUM_BUFFERS)
    b_buf = b_smem.index(k % NUM_BUFFERS)
    a = a_buf.load(layouts.a_reg_layout)
    b = b_buf.permute([1, 0]).load(layouts.b_reg_layout)
    return a, b


@gluon.jit
def buffer_store_output(acc, c_ptr, M, N, stride_cm, off_m, off_n, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    mfma_layout: gl.constexpr = acc.type.layout
    c_desc = t1.make_buffer_descriptor(c_ptr + off_n, off_m, M, stride_cm, BLOCK_M)
    rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma_layout))
    cols = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma_layout))
    mask = (off_n + cols < N)[None, :]
    buffer.store(acc.to(c_ptr.dtype.element_ty), c_desc, rows[:, None] * stride_cm + cols[None, :], mask)


@gluon.jit
def mfma_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                       stride_am, stride_ak, stride_bn, stride_bk, stride_cm,  #
                       NUM_BUFFERS: gl.constexpr = 2):
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = 256
    BLOCK_K: gl.constexpr = 64
    layouts = get_mfma_layouts()

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    off_m = pid // num_pid_n * BLOCK_M
    off_n = pid % num_pid_n * BLOCK_N

    a_desc = t1.make_buffer_descriptor(a_ptr, off_m, M, stride_am, BLOCK_M)
    b_desc = t1.make_buffer_descriptor(b_ptr, off_n, N, stride_bn, BLOCK_N)
    a_offs = offs_2d(BLOCK_M, BLOCK_K, stride_am, layouts.a_gmem_layout)
    b_offs = offs_2d(BLOCK_N, BLOCK_K, stride_bn, layouts.b_gmem_layout)

    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K], layouts.b_smem_layout)

    num_k = gl.cdiv(K, BLOCK_K)
    for i in gl.static_range(NUM_BUFFERS - 1):
        issue_async_copy(i, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem, stride_ak, stride_bk, BLOCK_K, NUM_BUFFERS)

    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k - (NUM_BUFFERS - 1)):
        issue_async_copy(k + NUM_BUFFERS - 1, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem, stride_ak, stride_bk,
                         BLOCK_K, NUM_BUFFERS)
        async_copy.wait_group(NUM_BUFFERS - 1)
        a, b = load_mfma_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS)
        acc = cdna4.mfma(a, b, acc)

    for i in gl.static_range(NUM_BUFFERS - 1):
        async_copy.wait_group(NUM_BUFFERS - 2 - i)
        a, b = load_mfma_operands(num_k - (NUM_BUFFERS - 1) + i, a_smem, b_smem, layouts, NUM_BUFFERS)
        acc = cdna4.mfma(a, b, acc)

    buffer_store_output(acc, c_ptr, M, N, stride_cm, off_m, off_n, BLOCK_M, BLOCK_N)


def mfma_matmul(A, B, kernel, **kwargs):
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256), )
    compiled = kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                            num_warps=4, **kwargs)
    return C, compiled


@pytest.mark.parametrize("M, N, K", [(256, 256, 128), (500, 300, 256), (1024, 512, 512)])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_mfma_matmul(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C, _ = mfma_matmul(A, B, mfma_matmul_kernel)
    torch.testing.assert_close(A.float() @ B.float().T, C.float(), atol=1e-1, rtol=1e-2)


# %%
# MFMAs can only start after their operands are read from LDS. This means the
# MFMAs have to wait until all LDS reads finish, which wastes time. One
# technique we can use in Gluon is subtiling. We split the K dimension of a tile
# into sub-tiles, so that the LDS reads of one sub-tile overlap with the MFMAs
# of another.


@gluon.jit
def load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS: gl.constexpr, k_start: gl.constexpr,
                           k_size: gl.constexpr):
    a_buf = a_smem.index(k % NUM_BUFFERS).slice(k_start, k_size, 1)
    b_buf = b_smem.index(k % NUM_BUFFERS).slice(k_start, k_size, 1)
    a = a_buf.load(layouts.a_reg_layout)
    b = b_buf.permute([1, 0]).load(layouts.b_reg_layout)
    return a, b


@gluon.jit
def mfma_matmul_subtiled_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                                stride_am, stride_ak, stride_bn, stride_bk, stride_cm):
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = 256
    BLOCK_K: gl.constexpr = 64
    NUM_BUFFERS: gl.constexpr = 2
    layouts = get_mfma_layouts()

    SUB_K: gl.constexpr = BLOCK_K // 2
    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    off_m = pid // num_pid_n * BLOCK_M
    off_n = pid % num_pid_n * BLOCK_N

    a_desc = t1.make_buffer_descriptor(a_ptr, off_m, M, stride_am, BLOCK_M)
    b_desc = t1.make_buffer_descriptor(b_ptr, off_n, N, stride_bn, BLOCK_N)
    a_offs = offs_2d(BLOCK_M, BLOCK_K, stride_am, layouts.a_gmem_layout)
    b_offs = offs_2d(BLOCK_N, BLOCK_K, stride_bn, layouts.b_gmem_layout)

    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K], layouts.b_smem_layout)

    num_k = gl.cdiv(K, BLOCK_K)
    issue_async_copy(0, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem, stride_ak, stride_bk, BLOCK_K, NUM_BUFFERS)
    issue_async_copy(1, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem, stride_ak, stride_bk, BLOCK_K, NUM_BUFFERS)
    async_copy.wait_group(1)

    a0, b0 = load_subtiled_operands(0, a_smem, b_smem, layouts, NUM_BUFFERS, 0, SUB_K)
    a1, b1 = load_subtiled_operands(0, a_smem, b_smem, layouts, NUM_BUFFERS, SUB_K, SUB_K)
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    acc = cdna4.mfma(a0, b0, acc)

    for k in range(1, num_k - 1):
        async_copy.wait_group(0)
        a0, b0 = load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS, 0, SUB_K)
        acc = cdna4.mfma(a1, b1, acc)
        issue_async_copy(k + 1, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem, stride_ak, stride_bk, BLOCK_K,
                         NUM_BUFFERS)
        a1, b1 = load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS, SUB_K, SUB_K)
        acc = cdna4.mfma(a0, b0, acc)

    async_copy.wait_group(0)
    a0, b0 = load_subtiled_operands(num_k - 1, a_smem, b_smem, layouts, NUM_BUFFERS, 0, SUB_K)
    acc = cdna4.mfma(a1, b1, acc)
    a1, b1 = load_subtiled_operands(num_k - 1, a_smem, b_smem, layouts, NUM_BUFFERS, SUB_K, SUB_K)
    acc = cdna4.mfma(a0, b0, acc)
    acc = cdna4.mfma(a1, b1, acc)

    buffer_store_output(acc, c_ptr, M, N, stride_cm, off_m, off_n, BLOCK_M, BLOCK_N)


@pytest.mark.parametrize("M, N, K", [(256, 256, 128), (500, 300, 256), (1024, 512, 512)])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_mfma_matmul_subtile(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C, _ = mfma_matmul(A, B, mfma_matmul_subtiled_kernel)
    torch.testing.assert_close(A.float() @ B.float().T, C.float(), atol=1e-1, rtol=1e-2)


# %%
# The benchmark compares the two kernels on an MI355X GPU with
# M = N = K = 8192.

if __name__ == "__main__" and is_cdna4():
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 8192, 8192, 8192
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    def bench(name, fn):
        ms = triton.testing.do_bench(fn, warmup=200, rep=1000)
        print(f"{name}: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

    bench("matmul", lambda: mfma_matmul(A, B, mfma_matmul_kernel))
    bench("matmul_subtile", lambda: mfma_matmul(A, B, mfma_matmul_subtiled_kernel))

# %%
# ```
# matmul: 1153 TFLOPS
# matmul_subtile: 1198 TFLOPS
# ```

# %%
# This tutorial shows the importance of layouts are in Gluon for AMD GPUs. Many
# optimizations come from choosing the right layout and letting the compiler
# generate the expected instructions. Layouts control how data flows from global
# memory to shared memory, from shared memory to registers, and between
# registers. We can see that the data path must be carefully managed in order to
# keep the compute unit busy. Later tutorials will introduce more advanced
# layouts-based optimizations.

# %%
# CDNA5 WMMA
# ----------
#
# A warp on CDNA5 GPUs has 32 lanes.
#
# In Gluon, a WMMA is exposed as `cdna5.wmma(a, b, acc)`. Just like `mfma`, the
# accumulator carries the instruction layout, here an `AMDWMMALayout`, and both
# operands use a `DotOperandLayout` whose parent is that layout. CDNA5 has one
# bf16 shape, `v_wmma_f32_16x16x32_bf16`, a 16x32 tile times a 32x16 tile. So
# `instr_shape` is `[16, 16, 32]`, and each operand needs `k_width=8`.
#
# `AMDWMMALayout` describes the placement of the warps in a different way.
# Instead of `warps_per_cta`, `warp_bases` gives the tile offset of each bit of
# the warp ID, in units of instruction tiles. With `[[0, 1], [1, 0]]`, warp 1
# takes the next tile along N and warp 2 the next tile along M, which is the
# same 2x2 warp grid as `warps_per_cta=[2, 2]`.

if __name__ == "__main__" and is_cdna5():
    wmma_layout = gl.amd.AMDWMMALayout(  #
        version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[])
    print(wmma_layout.format_tensor_view([32, 32]))

# %%
# ```
# [[  T0:0, ...,  T0:7,  T16:0, ...,  T16:7,  T32:0, ...,  T48:0, ...,  T48:7]
# [   T1:0, ...,  T1:7,  T17:0, ...,  T17:7,  T33:0, ...,  T49:0, ...,  T49:7]
# ...
# [  T15:0, ..., T15:7,  T31:0, ...,  T31:7,  T47:0, ...,  T63:0, ...,  T63:7]
# [  T64:0, ..., T64:7,  T80:0, ...,  T80:7,  T96:0, ..., T112:0, ..., T112:7]
# ...
# [  T79:0, ..., T79:7,  T95:0, ...,  T95:7, T111:0, ..., T127:0, ..., T127:7]]
# ```
#
# Each warp has 32 lanes and owns one 16x16 tile of the 32x32 block. Warp 0
# (lanes T0-T31) has the top left tile, warp 1 (T32-T63) the top right tile,
# warp 2 (T64-T95) the bottom left tile, and warp 3 (T96-T127) the bottom right
# tile.
#
# `wmma_kernel` loads A and B into registers and multiplies them with `wmma`.


@gluon.jit
def wmma_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, WARP_BASES: gl.constexpr):
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(  #
        version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=WARP_BASES, reg_bases=[])
    a = gl.load(a_ptr + offs_2d(M, K, K, gl.DotOperandLayout(0, wmma_layout, 8)))
    b = gl.load(b_ptr + offs_2d(K, N, N, gl.DotOperandLayout(1, wmma_layout, 8)))
    acc = cdna5.wmma(a, b, gl.zeros([M, N], gl.float32, wmma_layout))
    gl.store(c_ptr + offs_2d(M, N, N, wmma_layout), acc)


def wmma(A, B, warp_bases):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    wmma_kernel[(1, )](A, B, C, M, N, K, warp_bases, num_warps=2**len(warp_bases))
    return C


@pytest.mark.parametrize("M, N, K", [(16, 16, 32), (64, 128, 64)])
@pytest.mark.parametrize("warp_bases", [(), ((0, 1), (1, 0))])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_wmma(M, N, K, warp_bases):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = wmma(A, B, warp_bases)
    torch.testing.assert_close(A.float() @ B.float(), C, atol=1e-3, rtol=1e-3)


# %%
# The LDS also changes from CDNA4 to CDNA5. On CDNA5, a CTA runs on a workgroup
# processor (WGP), which takes the place of the compute unit (CU) of CDNA4. A
# WGP has 320 KB of LDS, twice the 160 KB of a CDNA4 CU, so a kernel can keep
# more and larger tiles in flight. The banks stay the same, 64 banks of 4 bytes.
# With 32 lanes per warp, `ds_load_b128` serves the lanes of a warp in two
# groups of 16 consecutive lanes, while `ds_read_b128` on CDNA4 serves four
# groups of 16 lanes that are not consecutive.
#
# The tiles also reach the LDS in a different way. On CDNA4, an async copy
# writes the data of each lane to consecutive LDS addresses, so a kernel shapes
# the shared memory layout through the global addresses that it loads. On
# CDNA5, TDM copies a whole tile, and writes it into LDS in the shared memory
# layout, including any padding.
#
# `wmma_smem_kernel` copies A and B into LDS with TDM, and the warps read the
# operands from there. `bank_conflicts_kernel` then counts the bank conflicts of
# these reads with `gl.bank_conflicts`, for operand A of a 256x64 tile.


@gluon.jit
def wmma_smem_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, smem_layout: gl.constexpr):
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(  #
        version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[])
    gl.static_assert(a_ptr.dtype.element_ty == gl.bfloat16 and b_ptr.dtype.element_ty == gl.bfloat16)
    a_desc = tdm.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [M, K], smem_layout)
    b_desc = tdm.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [N, K], smem_layout)
    a_smem = gl.allocate_shared_memory(gl.bfloat16, [M, K], smem_layout)
    b_smem = gl.allocate_shared_memory(gl.bfloat16, [N, K], smem_layout)
    tdm.async_load(a_desc, [0, 0], a_smem)
    tdm.async_load(b_desc, [0, 0], b_smem)
    tdm.async_wait(0)

    a = a_smem.load(gl.DotOperandLayout(0, wmma_layout, 8))
    # View the [N, K] buffer as [K, N].
    b = b_smem.permute([1, 0]).load(gl.DotOperandLayout(1, wmma_layout, 8))
    acc = cdna5.wmma(a, b, gl.zeros([M, N], gl.float32, wmma_layout))
    gl.store(c_ptr + offs_2d(M, N, N, wmma_layout), acc)


def wmma_smem(A, B):
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    wmma_smem_kernel[(1, )](A, B, C, M, N, K, smem_layout, num_warps=4)
    return C


@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_wmma_smem():
    A = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    C = wmma_smem(A, B)
    torch.testing.assert_close(A.float() @ B.float().T, C, atol=1e-3, rtol=1e-3)


if __name__ == "__main__" and is_cdna5():
    wmma_layout = gl.amd.AMDWMMALayout(  #
        version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[])
    a_layout = gl.DotOperandLayout(0, wmma_layout, 8)
    plain_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, plain_layout, 7)

# %%
# Without swizzling or padding, the read costs 7 extra accesses, an 8-way
# conflict. One row of 64 bf16 values is 128 bytes, so every row covers half of
# the banks, and the lanes that read different rows collide.
#
# Both fixes from MFMA apply. For swizzling, note that two rows now fill all 64
# banks, so the swizzle pattern advances every second row (`per_phase=2`). For
# padding, 8 extra values (16 bytes) after each row shift the next row by 4
# banks. Padding is the natural choice on CDNA5, because TDM adds it on the fly
# while it writes the tile.

if __name__ == "__main__" and is_cdna5():
    swizzled_layout = gl.SwizzledSharedLayout(  #
        vec=8, per_phase=2, max_phase=8, order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, swizzled_layout, 0)

    padded_layout = gl.PaddedSharedLayout.with_identity_for(  #
        # Pad 8 elements after every 64 elements
        interval_padding_pairs=[[64, 8]],  #
        shape=[256, 64],  #
        order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, padded_layout, 0)

# %%
# Padding keeps the lanes of one read on different banks. But on CDNA5, reads
# of different warps can still slow each other down, because of how the WGP
# connects its SIMDs to the LDS. Warp `i` runs on SIMD `i % 4`, and the 4 SIMDs
# work in two pairs, SIMDs 0 and 2, and SIMDs 1 and 3.
# Each pair has its own port into the LDS. Behind the ports, the 320 KB of LDS
# consists of 5 physical partitions of 64 KB. Both pairs can read in the same
# cycle only if they access different partitions. Otherwise their reads take
# turns, and each pair gets half the bandwidth. This is a partition conflict.
#
# The simple `warp_bases=[[0, 1], [1, 0]]` runs into it. Warps 0 and 1 sit on
# different SIMD pairs but compute tiles in the same rows, so both read the
# same rows of A, which live in the same partition.
#
# To avoid it, the two SIMD pairs must read different rows of A and B, and these
# rows must sit in different partitions. `make_partitioned_dot_layouts` returns
# a WMMA layout that places the warps this way, and a `PartitionedSharedLayout`
# that splits each buffer into pieces in different partitions. Each piece keeps
# the padded layout from above.

if __name__ == "__main__" and is_cdna5():
    a_smem_layout, b_smem_layout, wmma_layout = gl.amd.cdna5.make_partitioned_dot_layouts(  #
        256, 256, padded_layout, padded_layout, 4, [16, 16, 32], b_transposed=True)
    print(wmma_layout)
    print(a_smem_layout)

# %%
# ```
# AMDWMMALayout(version=3, transposed=True, warp_bases=[[8, 8], [4, 0]],
#               reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [8, 0]], ...)
# PartitionedSharedLayout(
#     num_partitions=2, num_groups=1, partition_dim=0,
#     partition_layout=PaddedSharedLayout(
#         interval_padding_pairs=[[64, 8]], ...))
# ```
#
# Warp 1 moved 8 tiles (128 elements) down and 8 tiles right of warp 0. Now one
# SIMD pair works on the upper half of the rows of A and B, and the other pair
# on the lower half. The shared layout cuts A into these two halves along
# dimension 0 and stores them in different partitions, and B is cut the same
# way. TDM follows the cut, so each warp copies into one half.

# %%
# The matrix multiplication kernel computes a 256x256 tile per CTA with 4 warps
# and BLOCK_K = 64, and TDM copies the tiles into a ring of LDS buffers. The
# accumulator takes 256 * 256 / 128 = 512 fp32 values per lane, which fits in
# the 1024 VGPRs of a CDNA5 lane.
#
# Similar to the MFMA kernel, each K block is split into two sub-tiles of 32.
# While the WMMAs work on one sub-tile, the LDS reads for the other are in
# flight. The output goes back through LDS and a TDM store.


@gluon.aggregate
class WmmaLayouts:
    # Accumulator and operand register layouts
    acc_layout: gl.constexpr
    a_reg_layout: gl.constexpr
    b_reg_layout: gl.constexpr
    # Shared memory layouts of A, B and C
    a_smem_layout: gl.constexpr
    b_smem_layout: gl.constexpr
    c_smem_layout: gl.constexpr


@gluon.constexpr_function
def get_wmma_layouts(partitioned):
    # Assume the 256x256x64 block size and 4 warps per CTA
    # Operand A has shape [256, 64]
    # Operand B has shape [256, 64] as well, so transposed
    a_smem_layout = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [256, 64], [1, 0])
    b_smem_layout = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [256, 64], [1, 0])
    if partitioned:
        a_smem_layout, b_smem_layout, acc_layout = gl.amd.cdna5.make_partitioned_dot_layouts(  #
            256, 256, a_smem_layout, b_smem_layout, 4, [16, 16, 32], b_transposed=True)
    else:
        acc_layout = gl.amd.AMDWMMALayout(  #
            version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[])
    a_reg_layout = gl.DotOperandLayout(0, acc_layout, 8)
    b_reg_layout = gl.DotOperandLayout(1, acc_layout, 8)
    c_smem_layout = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [256, 256], [1, 0])
    return WmmaLayouts(acc_layout, a_reg_layout, b_reg_layout, a_smem_layout, b_smem_layout, c_smem_layout)


@gluon.jit
def issue_tdm_copy(k, a_desc, b_desc, a_smem, b_smem, off_m, off_n, BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr):
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, k * BLOCK_K], clamp_bounds=True)
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[off_n, k * BLOCK_K], clamp_bounds=True)
    tdm.async_load(a_desc, dest=a_smem.index(k % NUM_BUFFERS))
    tdm.async_load(b_desc, dest=b_smem.index(k % NUM_BUFFERS))


@gluon.jit
def store_output(acc, c_ptr, M, N, stride_cm, off_m, off_n, c_smem_layout: gl.constexpr):
    BLOCK_M: gl.constexpr = acc.shape[0]
    BLOCK_N: gl.constexpr = acc.shape[1]
    c_smem = gl.allocate_shared_memory(c_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], c_smem_layout)
    c_smem.store(acc.to(c_ptr.dtype.element_ty))
    c_desc = tdm.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [BLOCK_M, BLOCK_N], c_smem_layout)
    c_desc = tdm.update_tensor_descriptor(c_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
    tdm.async_store(c_desc, src=c_smem)
    tdm.async_wait(0)


@gluon.jit
def wmma_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                       stride_am, stride_ak, stride_bn, stride_bk, stride_cm,  #
                       PARTITIONED: gl.constexpr, NUM_BUFFERS: gl.constexpr = 2):
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = 256
    BLOCK_K: gl.constexpr = 64
    layouts = get_wmma_layouts(PARTITIONED)

    SUB_K: gl.constexpr = BLOCK_K // 2
    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    off_m = pid // num_pid_n * BLOCK_M
    off_n = pid % num_pid_n * BLOCK_N

    a_desc = tdm.make_tensor_descriptor(a_ptr, [M, K], [stride_am, stride_ak], [BLOCK_M, BLOCK_K],
                                        layouts.a_smem_layout)
    b_desc = tdm.make_tensor_descriptor(b_ptr, [N, K], [stride_bn, stride_bk], [BLOCK_N, BLOCK_K],
                                        layouts.b_smem_layout)
    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K], layouts.b_smem_layout)

    num_k = gl.cdiv(K, BLOCK_K)
    for i in gl.static_range(NUM_BUFFERS):
        issue_tdm_copy(i, a_desc, b_desc, a_smem, b_smem, off_m, off_n, BLOCK_K, NUM_BUFFERS)
    tdm.async_wait((NUM_BUFFERS - 1) * 2)

    a0, b0 = load_subtiled_operands(0, a_smem, b_smem, layouts, NUM_BUFFERS, 0, SUB_K)
    a1, b1 = load_subtiled_operands(0, a_smem, b_smem, layouts, NUM_BUFFERS, SUB_K, SUB_K)
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    acc = cdna5.wmma(a0, b0, acc)

    for k in range(1, num_k - (NUM_BUFFERS - 1)):
        tdm.async_wait((NUM_BUFFERS - 2) * 2)
        a0, b0 = load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS, 0, SUB_K)
        acc = cdna5.wmma(a1, b1, acc)
        # K block `k - 1` is done, so its slot can take the next copy.
        issue_tdm_copy(k + NUM_BUFFERS - 1, a_desc, b_desc, a_smem, b_smem, off_m, off_n, BLOCK_K, NUM_BUFFERS)
        a1, b1 = load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS, SUB_K, SUB_K)
        acc = cdna5.wmma(a0, b0, acc)

    for i in gl.static_range(NUM_BUFFERS - 1):
        k = num_k - (NUM_BUFFERS - 1) + i
        tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        a0, b0 = load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS, 0, SUB_K)
        acc = cdna5.wmma(a1, b1, acc)
        a1, b1 = load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS, SUB_K, SUB_K)
        acc = cdna5.wmma(a0, b0, acc)
    acc = cdna5.wmma(a1, b1, acc)

    store_output(acc, c_ptr, M, N, stride_cm, off_m, off_n, layouts.c_smem_layout)


def wmma_matmul(A, B, partitioned, num_buffers=2):
    M, K = A.shape
    N = B.shape[0]
    assert triton.cdiv(K, 64) >= num_buffers
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256), )
    compiled = wmma_matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                                        C.stride(0), partitioned, num_buffers, num_warps=4)
    return C, compiled


@pytest.mark.parametrize("M, N, K", [(256, 256, 256), (500, 300, 512), (1024, 512, 1024)])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("num_buffers", [2, 3])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_wmma_matmul(M, N, K, partitioned, num_buffers):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C, _ = wmma_matmul(A, B, partitioned, num_buffers)
    torch.testing.assert_close(A.float() @ B.float().T, C.float(), atol=1e-1, rtol=1e-2)


# %%
# We run the kernel with the partitioned layouts and 2, 3 and 4 buffers on an
# MI455X GPU with M = N = K = 8192.

if __name__ == "__main__" and is_cdna5():
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 8192, 8192, 8192
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    for num_buffers in [2, 3, 4]:
        ms = triton.testing.do_bench(lambda: wmma_matmul(A, B, True, num_buffers), warmup=200, rep=1000)
        print(f"{num_buffers} buffers: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

# %%
# ```
# 2 buffers: 2318 TFLOPS
# 3 buffers: 2711 TFLOPS
# 4 buffers: 2736 TFLOPS
# ```
