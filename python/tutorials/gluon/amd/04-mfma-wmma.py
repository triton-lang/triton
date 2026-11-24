"""
MFMA and WMMA
=============

AMD GPUs run matrix multiplications on dedicated matrix cores. CDNA4 GPUs use
matrix fused-multiply-add (MFMA) instructions, and CDNA5 GPUs use wave matrix
multiply-accumulate (WMMA) instructions. Both read their operands from registers
and accumulate in registers. They can't consume shared memory (LDS) directly,
so the kernel needs extra instructions to load every operand from LDS into
registers.

This tutorial introduces MFMA/WMMA and their layouts with the bf16 data type.
We will also cover different ways to load the operands from LDS and demonstrate
their importance for performance.
"""

import importlib
import pytest
import sys
import torch
import triton
from pathlib import Path
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4, cdna5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
t1 = importlib.import_module("amd.01-buffer-ops")
t2 = importlib.import_module("amd.02-async-copy")


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
# CDNA4 MFMA
# ----------
#
# Gluon exposes MFMA as `mfma(a, b, acc)`. The operation has layout
# requirements for its operands, where the accumulator `acc` must use an
# `AMDMFMALayout`, and the operands `a` and `b` must use a `DotOperandLayout`
# with an MFMA layout as parent. These layouts control which MFMA instructions
# the compiler generates.
#
# The operand data type and `instr_shape` of `AMDMFMALayout` decide the MFMA
# instruction to be used. `instr_shape` is the shape of the instruction in
# [M, N, K] format for one warp. With multiple warps, the `warps_per_cta` of
# `AMDMFMALayout` describes how to distribute them over the accumulator along
# the M and N dimensions.
#
# The instruction shape and warp distribution decide a basic MFMA tile. When the
# accumulator is larger than the MFMA tile, there will be multiple MFMA tiles to
# cover the entire tensor.


@gluon.jit
def offs_2d(rows: gl.constexpr, cols: gl.constexpr, stride, layout: gl.constexpr):
    r = gl.arange(0, rows, gl.SliceLayout(1, layout))
    c = gl.arange(0, cols, gl.SliceLayout(0, layout))
    return r[:, None] * stride + c[None, :]


@gluon.jit
def mfma_kernel(a_ptr, b_ptr, c_ptr,  #
                M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,  #
                INSTR_SHAPE: gl.constexpr, WARPS_PER_CTA: gl.constexpr):
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=INSTR_SHAPE, transposed=True, warps_per_cta=WARPS_PER_CTA)
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

    a = gl.load(a_ptr + offs_2d(M, K, K, a_layout))
    b = gl.load(b_ptr + offs_2d(K, N, N, b_layout))

    acc = gl.zeros([M, N], gl.float32, mfma_layout)
    acc = cdna4.mfma(a, b, acc)

    gl.store(c_ptr + offs_2d(M, N, N, mfma_layout), acc)


def mfma(A, B, instr_shape, warps_per_cta):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    mfma_kernel[(1, )](A, B, C, M, N, K, tuple(instr_shape), tuple(warps_per_cta))
    return C


@pytest.mark.parametrize("M, N, K", [(32, 32, 32), (64, 64, 64)])
# For bf16, CDNA4 has two MFMA instruction shapes:
# - `v_mfma_f32_16x16x32_bf16`
# - `v_mfma_f32_32x32x16_bf16`
@pytest.mark.parametrize("instr_shape", [[16, 16, 32], [32, 32, 16]])
# [2,2] = 2 warps along M and 2 warps along N
# [4,1] = 4 warps along M and 1 warp along N
@pytest.mark.parametrize("warps_per_cta", [[2, 2], [4, 1], [1, 4]])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_mfma(M, N, K, instr_shape, warps_per_cta):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = mfma(A, B, instr_shape, warps_per_cta)
    torch.testing.assert_close(A.float() @ B.float(), C, atol=1e-3, rtol=1e-3)


# %%
# We can visualize the MFMA layouts using the `format_tensor_view` method.

if __name__ == "__main__" and is_cdna4():
    mfma_layout = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, 1])
    # the operand A layout
    print(gl.DotOperandLayout(0, mfma_layout, 8).format_tensor_view([16, 32]))
    # the accumulator layout
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
# Knowing the MFMA layouts can help us implement some optimizations, like
# reducing layout conversions. Layout conversion can happen in kernels like
# attention, where there is a chain of two MFMAs, and the output of the first
# one becomes an operand of the second one. As shown above, the accumulator
# layout differs from the operand layout, so the compiler has to convert between
# them. We can use some fields on the `AMDMFMALayout` to match these two
# layouts.

# %%
# The first one is `transposed=True`. With it, each lane of the accumulator
# holds 4 consecutive elements of a row instead of a column. This makes it
# closer to the layout of operand A in the next MFMA, as both become row-major.

if __name__ == "__main__" and is_cdna4():
    acc_layout = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
    print(acc_layout.format_tensor_view([16, 16]))

# %%
# ```
# [[ T0:0,  T0:1,  T0:2,  T0:3, T16:0, ..., T16:3, T32:0, ..., T48:3]
# [  T1:0,  T1:1,  T1:2,  T1:3, T17:0, ..., T17:3, T33:0, ..., T49:3]
# ...
# [ T15:0, T15:1, T15:2, T15:3, T31:0, ..., T31:3, T47:0, ..., T63:3]]
# ```

# %%
# To make the two layouts identical, we need to set a different `k_width`.
# `k_width` is the number of consecutive elements along K that a lane holds in
# an MFMA operand. The 16x16x32 MFMA instruction takes 8 values of A and 8
# values of B per lane, so the natural choice is `k_width=8` to match what
# hardware expects. But we can use a different `k_width` as long as both
# operands use the same one. The results still stay correct, because this only
# changes the order of the elements along K, in the same way for A and B. With
# `k_width=4`, each lane holds 4 consecutive elements, which exactly
# matches the transposed accumulator layout.

if __name__ == "__main__" and is_cdna4():
    a_layout = gl.DotOperandLayout(0, acc_layout, 4)
    print(a_layout.format_tensor_view([16, 32]))

# %%
# ```
# [[ T0:0,  T0:1,  T0:2,  T0:3, T16:0, ..., T48:3,  T0:4, ..., T48:7]
# [  T1:0,  T1:1,  T1:2,  T1:3, T17:0, ..., T49:3,  T1:4, ..., T49:7]
# ...
# [ T15:0, T15:1, T15:2, T15:3, T31:0, ..., T63:3, T15:4, ..., T63:7]]
# ```
#
# Let's put this in practice with 2 chained MFMA operations.


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
    mfma_chain_kernel[(1, )](A, B, C, out, 64)
    ref = (A.float() @ B.float()).to(torch.bfloat16).float() @ C.float()
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)


# %%
# All layouts shown above are for registers. Real kernels usually load the data
# into shared memory with async copies first, and then copy it from shared
# memory to registers. In Triton, this process is all handled by compiler
# optimization. In Gluon, the kernel can control how shared memory / LDS is
# used.
#
# Similarly, Gluon needs extra layouts for LDS on AMD. These layouts determine
# which LDS instructions the compiler uses. They also determine whether there
# will be bank conflicts.
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
def mfma_smem_kernel(a_ptr, b_ptr, c_ptr,  #
                     M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, B_K_CONTIGUOUS: gl.constexpr,  #
                     a_smem_layout: gl.constexpr, b_smem_layout: gl.constexpr):
    warps: gl.constexpr = gl.num_warps()
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, warps // 2])
    gmem_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [warps, 1], [1, 0])

    gl.static_assert(a_ptr.dtype.element_ty == gl.bfloat16 and b_ptr.dtype.element_ty == gl.bfloat16)
    a_desc = cdna4.buffer.make_buffer_descriptor(a_ptr, M * K * 2)
    b_desc = cdna4.buffer.make_buffer_descriptor(b_ptr, N * K * 2)

    # Load data from global memory to shared memory
    a_smem = gl.allocate_shared_memory(gl.bfloat16, [M, K], a_smem_layout)
    cdna4.buffer.load_to_shared(a_smem, a_desc, offs_2d(M, K, K, gmem_layout))
    if B_K_CONTIGUOUS:
        b_smem = gl.allocate_shared_memory(gl.bfloat16, [N, K], b_smem_layout)
        cdna4.buffer.load_to_shared(b_smem, b_desc, offs_2d(N, K, K, gmem_layout))
    else:
        b_smem = gl.allocate_shared_memory(gl.bfloat16, [K, N], b_smem_layout)
        cdna4.buffer.load_to_shared(b_smem, b_desc, offs_2d(K, N, N, gmem_layout))
    cdna4.async_copy.commit_group()
    cdna4.async_copy.wait_group(0)

    # Load data from shared memory to registers
    a = a_smem.load(gl.DotOperandLayout(0, mfma_layout, 8))
    if B_K_CONTIGUOUS:
        # When B is K-contiguous in shared memory, we need to permute
        # from [N, K] to [K, N] - this is a requirement for the `mfma` op.
        # The permutation itself is free. It only generates the ds_read
        # instructions without extra shuffles.
        b = b_smem.permute([1, 0]).load(gl.DotOperandLayout(1, mfma_layout, 8))
    else:
        b = b_smem.load(gl.DotOperandLayout(1, mfma_layout, 8))

    acc = gl.zeros([M, N], gl.float32, mfma_layout)
    acc = cdna4.mfma(a, b, acc)
    gl.store(c_ptr + offs_2d(M, N, N, mfma_layout), acc)


def mfma_smem(A, B, b_k_contiguous):
    M, K = A.shape
    N = B.shape[0] if b_k_contiguous else B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    compiled = mfma_smem_kernel[(1, )](A, B, C, M, N, K, b_k_contiguous, smem_layout, smem_layout)
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
        t1.print_instructions(compiled, r"ds_read")

# %%
# ```
# B K-contiguous=True: {'ds_read_b128': 8}
#   ds_read_b128 v[10:13], v4 offset:8192
#   ds_read_b128 v[30:33], v4 offset:12288
#   ...
#   ds_read_b128 v[22:25], v8
#   ds_read_b128 v[26:29], v8 offset:64
#   ...
# B K-contiguous=False: {'ds_read_b64_tr_b16': 8, 'ds_read_b128': 4}
#   ds_read_b64_tr_b16 v[10:11], v4 offset:8192
#   ds_read_b64_tr_b16 v[12:13], v4 offset:8704
#   ...
#   ds_read_b128 v[18:21], v8
#   ds_read_b128 v[22:25], v8 offset:64
#   ...
# ```
#
# With `k_width=8`, each lane holds 8 consecutive bf16 values along K for an
# MFMA, which is 16 bytes. When K is contiguous in shared memory, these 16 bytes
# are adjacent, and one `ds_read_b128` can load all of them. This is what we see
# here for operand A, in both cases.
#
# For operand B, when it is K-contiguous, the compiler uses the same regular
# reads as for A. But when it is N-contiguous, the compiler switches to the
# transposed read `ds_read_b64_tr_b16`, and the data still arrives in the
# operand layout without extra shuffles.

# %%
# The kernel above uses a simple shared memory layout, which can have bank
# conflicts. We can check whether bank conflicts occur with
# `gl.bank_conflicts`. It counts the number of extra accesses that bank
# conflicts cause between a register layout and a shared memory layout.


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
# There are two ways to avoid bank conflicts, swizzling and padding, and both
# can be implemented using a special shared memory layout in Gluon. A swizzled
# layout keeps every row in place, but moves the groups of `vec` values inside a
# row with an XOR of the row index. A padded layout keeps the values in order,
# but adds unused values after every `interval` values, so that the next rows
# start at other banks.
#
# Selecting the layout parameters needs some understanding of how LDS serves
# data. This differs from instruction to instruction, but we can still briefly
# explain the common parts here. The LDS of a CDNA4 CU has 64 banks, each 4
# bytes wide, so it can serve 256 bytes per cycle. Some instructions like
# `ds_read_b128` cannot be served in a single cycle for all lanes. In this
# case, the LDS splits the instruction into multiple cycles, serving a group of
# lanes in each cycle. Bank conflicts only happen within a group. For
# `ds_read_b128`, each group has 16 lanes. The first group has lanes 0-3, 12-15,
# 20-23 and 24-27, the second group has lanes 4-7, 8-11, 16-19 and 28-31, and so
# on.
#
# There are some heuristics for picking the shared memory layout. For a swizzled
# layout, we can let `vec` be `k_width`, and let `per_phase` be the
# number of rows in one 256-byte line of banks, and `max_phase` be the
# number of rows that a group reads divided by `per_phase`. For example, in our
# 256x64 bf16 case, a row has 128 bytes, which is 32 banks. So 2 rows can occupy
# the full 64 banks. Therefore, we can set `per_phase=2`. And as we know
# `ds_read_b128` has 16 lanes per group, and these lanes will read 16 different
# rows (see the MFMA operand layout above), so `max_phase` can be
# `16 / per_phase` = 8.
#
# For a padded layout, the heuristic is simpler. We just pad a fixed amount
# after every row. The pad amount is the number of different columns that a
# group reads. In our case, there are 64 elements per row, so we pad every
# 64 elements. And one lane loads 8 elements at once, so that's 8 different
# columns. Recall the group and operand layout, we have lanes 0-3 and 12-15 on
# columns 0-7, and lanes 20-23 and 24-27 on columns 8-15, so that's 16 different
# columns in total.

if __name__ == "__main__" and is_cdna4():
    swizzled_layout = gl.SwizzledSharedLayout(  #
        vec=8, per_phase=2, max_phase=8, order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, swizzled_layout, 0)

    padded_layout = gl.PaddedSharedLayout.with_identity_for(  #
        interval_padding_pairs=[[64, 16]], shape=[256, 64], order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, padded_layout, 0)

# %%
# So far we only considered the data path from shared memory to registers.
# Another factor is the data path from global memory to shared memory. As we saw
# in the async copy tutorial, a load instruction has no shared memory address
# for each lane, so the data it writes into shared memory needs to be
# consecutive. This means we cannot add padding between the rows. You can see
# that the padded layout above fails to compile with async copies.


@gluon.jit
def async_copy_tile_kernel(a_ptr, gmem_layout: gl.constexpr, smem_layout: gl.constexpr):
    smem = gl.allocate_shared_memory(gl.bfloat16, [256, 64], smem_layout)
    a_desc = cdna4.buffer.make_buffer_descriptor(a_ptr, 256 * 64 * 2)
    cdna4.buffer.load_to_shared(smem, a_desc, offs_2d(256, 64, 64, gmem_layout))
    cdna4.async_copy.commit_group()
    cdna4.async_copy.wait_group(0)


def try_compile_async_copy(gmem_layout, smem_layout):
    A = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    try:
        with t2.suppress_stderr():
            async_copy_tile_kernel[(1, )](A, gmem_layout, smem_layout)
        return True
    except RuntimeError:
        return False


if __name__ == "__main__" and is_cdna4():
    gmem_layout = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    padded_layout = gl.PaddedSharedLayout.with_identity_for(  #
        interval_padding_pairs=[[64, 16]], shape=[256, 64], order=[1, 0])
    assert not try_compile_async_copy(gmem_layout, padded_layout)

# %%
# There is a trick to work around this. Instead of adding gaps within one load
# instruction, we add them between instructions or across warps, since a tile is
# composed of multiple instructions. And we can carefully shuffle elements
# across instructions to avoid bank conflicts.
#
# We still use our 256x64 tile and `k_width=8` example, with `ds_read_b128` for
# shared memory and `buffer_load_dwordx4 ... lds` for global memory. As we can't
# pad inside one load instruction, we have to set the interval to 512 values,
# which is 1 KB, exactly the amount one `buffer_load_dwordx4 ... lds` writes. As
# before, one group of lanes touches 16 columns, so we can still use a padding
# amount of 16. As the tile has 64 columns, with an interval of 512 values, we
# pad after every 8 rows. It still has bank conflicts, because lanes 0-3 and
# 20-23 still fall into these 8 rows together.
#
# Next we need to reorder rows. As in our heuristics for padded layouts, we add
# a padding after every row. The goal of shuffling the rows is to achieve the
# exact same pattern, by placing each row 1 KB apart, so that each of them is
# put into a new load instruction. And we will fill the rest of the rows in a
# similar manner. After reordering, the rows look like this. We still pad every
# 8 rows, across the 1 KB boundary.
#
# ```
# row 000
# row 016
# row 032
# row 048
# row 064
# row 080
# row 096
# row 112
# [padding]
# row 001
# row 017
# row 033
# row 049
# row 065
# row 081
# row 097
# row 113
# [padding]
# ...
# ```
#
# `PaddedSharedLayout` describes this reordering with `offset_bases`. Its format
# is just like the bases of a linear layout.


@gluon.constexpr_function
def get_smem_layout_for_async_copies(shape, dtype):
    rows, cols = shape
    elem_bytes = dtype.primitive_bitwidth // 8
    bytes_per_copy = 1024
    rows_per_copy = bytes_per_copy // (cols * elem_bytes)

    # Number of elements per row: [0, 1], [0, 2], [0, 4], ... [0, 32]
    col_bases = [[0, 2**i] for i in range(cols.bit_length() - 1)]
    # Distribute 1 row from every 16-row slot: [16, 0], [32, 0], [64, 0]
    row_bases = [[16 * 2**i, 0] for i in range(rows_per_copy.bit_length() - 1)]
    # Fill remaining rows in the 16-row slot
    row_bases += [[1, 0], [2, 0], [4, 0], [8, 0]]
    # Fill the rest of rows
    row_bases += [[2**i, 0] for i in range((16 * rows_per_copy).bit_length() - 1, rows.bit_length() - 1)]

    return gl.PaddedSharedLayout(  #
        interval_padding_pairs=[[1024 // elem_bytes, 32 // elem_bytes]],  #
        offset_bases=col_bases + row_bases,  #
        cga_layout=[], shape=shape)


# %%
# The async copy has to write the rows in the same order. As `BlockedLayout`
# can't describe this reordering, we also build a linear layout for it:


@gluon.constexpr_function
def get_gmem_layout_for_async_copies(shape, dtype):
    bases = get_smem_layout_for_async_copies(shape, dtype).offset_bases

    # split offset_bases into register, lane, warp
    # [0, reg_end)         -> register
    # [reg_end, lane_end)  -> lane
    # [lane_end, warp_end) -> warp
    reg_end = (128 // dtype.primitive_bitwidth).bit_length() - 1
    lane_end = reg_end + 6  # assume 64 lanes per warp
    warp_end = lane_end + 2  # assume 4 warps per CTA

    return gl.DistributedLinearLayout(  #
        reg_bases=bases[:reg_end] + bases[warp_end:],  #
        lane_bases=bases[reg_end:lane_end],  #
        warp_bases=bases[lane_end:warp_end],  #
        block_bases=[], shape=shape)


if __name__ == "__main__" and is_cdna4():
    gmem_layout = get_gmem_layout_for_async_copies([256, 64], gl.bfloat16)
    smem_layout = get_smem_layout_for_async_copies([256, 64], gl.bfloat16)
    assert try_compile_async_copy(gmem_layout, smem_layout)
    bank_conflicts_kernel[(1, )](a_layout, smem_layout, 0)

# %%
# Now we combine these features into a pipelined matmul kernel. To reuse all
# optimization ideas we discussed so far, we only use 256x256x64 blocks and 4
# warps. But ideas here can be generalized to other block sizes.


@gluon.aggregate
class MatmulConfig:
    # Tensor shapes
    M: gl.tensor | gl.constexpr
    N: gl.tensor | gl.constexpr
    K: gl.tensor | gl.constexpr
    # Operand dtype
    dtype: gl.constexpr
    # Tile size
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    # Kernel config
    NUM_BUFFERS: gl.constexpr


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

    # Both operands use the shared layout and the global layout from above.
    gmem_layout = get_gmem_layout_for_async_copies([256, 64], gl.bfloat16)
    smem_layout = get_smem_layout_for_async_copies([256, 64], gl.bfloat16)
    return MfmaLayouts(gmem_layout, smem_layout, a_reg_layout,  #
                       gmem_layout, smem_layout, b_reg_layout,  #
                       acc_layout)


@gluon.jit
def matmul_cdna4_ld_gmem(cfg, k, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem):
    i = k % cfg.NUM_BUFFERS
    cdna4.buffer.load_to_shared(a_smem.index(i), a_desc, a_offs, soffset=k * cfg.BLOCK_K)
    cdna4.buffer.load_to_shared(b_smem.index(i), b_desc, b_offs, soffset=k * cfg.BLOCK_K)
    cdna4.async_copy.commit_group()


@gluon.jit
def matmul_cdna4_ld_smem(cfg, layouts, k, a_smem, b_smem):
    a_buf = a_smem.index(k % cfg.NUM_BUFFERS)
    b_buf = b_smem.index(k % cfg.NUM_BUFFERS)
    a = a_buf.load(layouts.a_reg_layout)
    b = b_buf.permute([1, 0]).load(layouts.b_reg_layout)
    return a, b


@gluon.jit
def matmul_cdna4_kernel(a_ptr, b_ptr, c_ptr, M, N, K, NUM_BUFFERS: gl.constexpr = 2):
    cfg = MatmulConfig(M, N, K, a_ptr.dtype.element_ty,  #
                       BLOCK_M=256, BLOCK_N=256, BLOCK_K=64,  #
                       NUM_BUFFERS=NUM_BUFFERS)
    layouts = get_mfma_layouts()

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(cfg.N, cfg.BLOCK_N)
    off_m = pid // num_pid_n * cfg.BLOCK_M
    off_n = pid % num_pid_n * cfg.BLOCK_N

    a_desc = t1.make_buffer_descriptor(a_ptr, off_m, cfg.M, cfg.K, cfg.BLOCK_M)
    b_desc = t1.make_buffer_descriptor(b_ptr, off_n, cfg.N, cfg.K, cfg.BLOCK_N)
    a_offs = offs_2d(cfg.BLOCK_M, cfg.BLOCK_K, cfg.K, layouts.a_gmem_layout)
    b_offs = offs_2d(cfg.BLOCK_N, cfg.BLOCK_K, cfg.K, layouts.b_gmem_layout)

    a_smem = gl.allocate_shared_memory(cfg.dtype, [cfg.NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(cfg.dtype, [cfg.NUM_BUFFERS, cfg.BLOCK_N, cfg.BLOCK_K], layouts.b_smem_layout)

    num_k = gl.cdiv(cfg.K, cfg.BLOCK_K)
    for i in gl.static_range(cfg.NUM_BUFFERS - 1):
        matmul_cdna4_ld_gmem(cfg, i, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem)

    acc = gl.zeros([cfg.BLOCK_M, cfg.BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k - (cfg.NUM_BUFFERS - 1)):
        matmul_cdna4_ld_gmem(cfg, k + cfg.NUM_BUFFERS - 1, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem)
        cdna4.async_copy.wait_group(cfg.NUM_BUFFERS - 1)
        a, b = matmul_cdna4_ld_smem(cfg, layouts, k, a_smem, b_smem)
        acc = cdna4.mfma(a, b, acc)

    for i in gl.static_range(cfg.NUM_BUFFERS - 1):
        cdna4.async_copy.wait_group(cfg.NUM_BUFFERS - 2 - i)
        a, b = matmul_cdna4_ld_smem(cfg, layouts, num_k - (cfg.NUM_BUFFERS - 1) + i, a_smem, b_smem)
        acc = cdna4.mfma(a, b, acc)

    c_desc = t1.make_buffer_descriptor(c_ptr + off_n, off_m, cfg.M, cfg.N, cfg.BLOCK_M)
    rows = gl.arange(0, cfg.BLOCK_M, gl.SliceLayout(1, layouts.acc_layout))
    cols = gl.arange(0, cfg.BLOCK_N, gl.SliceLayout(0, layouts.acc_layout))
    mask = (off_n + cols < cfg.N)[None, :]
    cdna4.buffer.store(acc.to(c_ptr.dtype.element_ty), c_desc, rows[:, None] * cfg.N + cols[None, :], mask)


def matmul_cdna4(A, B, kernel):
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256), )
    compiled = kernel[grid](A, B, C, M, N, K)
    return C, compiled


@pytest.mark.parametrize("M, N, K", [(256, 256, 128), (500, 300, 256), (1024, 512, 512)])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_matmul_cdna4(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C, _ = matmul_cdna4(A, B, matmul_cdna4_kernel)
    torch.testing.assert_close(A.float() @ B.float().T, C.float(), atol=1e-1, rtol=1e-2)


# %%
# MFMAs can only start after their operands are read from LDS. When a group of
# LDS reads is issued, there is no need to wait for all of it to complete. One
# technique we can use in Gluon is subtiling, where we split these LDS reads
# into smaller groups and start MFMAs earlier. There are many ways to split the
# LDS reads along the M, N or K dimension. Here we will demonstrate how to split
# it along the K dimension.


@gluon.jit
def matmul_cdna4_ld_smem_subtiled(cfg, layouts, k, a_smem, b_smem, k_start, k_size):
    a_buf = a_smem.index(k % cfg.NUM_BUFFERS).slice(k_start, k_size, 1)
    b_buf = b_smem.index(k % cfg.NUM_BUFFERS).slice(k_start, k_size, 1)
    a = a_buf.load(layouts.a_reg_layout)
    b = b_buf.permute([1, 0]).load(layouts.b_reg_layout)
    return a, b


@gluon.jit
def matmul_cdna4_subtiled_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                                 NUM_BUFFERS: gl.constexpr = 2):
    cfg = MatmulConfig(M, N, K, a_ptr.dtype.element_ty,  #
                       BLOCK_M=256, BLOCK_N=256, BLOCK_K=64,  #
                       NUM_BUFFERS=NUM_BUFFERS)
    layouts = get_mfma_layouts()

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(cfg.N, cfg.BLOCK_N)
    off_m = pid // num_pid_n * cfg.BLOCK_M
    off_n = pid % num_pid_n * cfg.BLOCK_N

    a_desc = t1.make_buffer_descriptor(a_ptr, off_m, cfg.M, cfg.K, cfg.BLOCK_M)
    b_desc = t1.make_buffer_descriptor(b_ptr, off_n, cfg.N, cfg.K, cfg.BLOCK_N)
    a_offs = offs_2d(cfg.BLOCK_M, cfg.BLOCK_K, cfg.K, layouts.a_gmem_layout)
    b_offs = offs_2d(cfg.BLOCK_N, cfg.BLOCK_K, cfg.K, layouts.b_gmem_layout)

    a_smem = gl.allocate_shared_memory(cfg.dtype, [cfg.NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(cfg.dtype, [cfg.NUM_BUFFERS, cfg.BLOCK_N, cfg.BLOCK_K], layouts.b_smem_layout)

    num_k = gl.cdiv(cfg.K, cfg.BLOCK_K)
    for i in gl.static_range(cfg.NUM_BUFFERS - 1):
        matmul_cdna4_ld_gmem(cfg, i, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem)

    acc = gl.zeros([cfg.BLOCK_M, cfg.BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k - (cfg.NUM_BUFFERS - 1)):
        matmul_cdna4_ld_gmem(cfg, k + cfg.NUM_BUFFERS - 1, a_desc, b_desc, a_offs, b_offs, a_smem, b_smem)
        cdna4.async_copy.wait_group(cfg.NUM_BUFFERS - 1)
        a0, b0 = matmul_cdna4_ld_smem_subtiled(cfg, layouts, k, a_smem, b_smem, 0, cfg.BLOCK_K // 2)
        a1, b1 = matmul_cdna4_ld_smem_subtiled(cfg, layouts, k, a_smem, b_smem, cfg.BLOCK_K // 2, cfg.BLOCK_K // 2)
        acc = cdna4.mfma(a0, b0, acc)
        acc = cdna4.mfma(a1, b1, acc)

    for i in gl.static_range(cfg.NUM_BUFFERS - 1):
        k = num_k - (cfg.NUM_BUFFERS - 1) + i
        cdna4.async_copy.wait_group(cfg.NUM_BUFFERS - 2 - i)
        a0, b0 = matmul_cdna4_ld_smem_subtiled(cfg, layouts, k, a_smem, b_smem, 0, cfg.BLOCK_K // 2)
        a1, b1 = matmul_cdna4_ld_smem_subtiled(cfg, layouts, k, a_smem, b_smem, cfg.BLOCK_K // 2, cfg.BLOCK_K // 2)
        acc = cdna4.mfma(a0, b0, acc)
        acc = cdna4.mfma(a1, b1, acc)

    c_desc = t1.make_buffer_descriptor(c_ptr + off_n, off_m, cfg.M, cfg.N, cfg.BLOCK_M)
    rows = gl.arange(0, cfg.BLOCK_M, gl.SliceLayout(1, layouts.acc_layout))
    cols = gl.arange(0, cfg.BLOCK_N, gl.SliceLayout(0, layouts.acc_layout))
    mask = (off_n + cols < cfg.N)[None, :]
    cdna4.buffer.store(acc.to(c_ptr.dtype.element_ty), c_desc, rows[:, None] * cfg.N + cols[None, :], mask)


@pytest.mark.parametrize("M, N, K", [(256, 256, 128), (500, 300, 256), (1024, 512, 512)])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_matmul_cdna4_subtiled(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C, _ = matmul_cdna4(A, B, matmul_cdna4_subtiled_kernel)
    torch.testing.assert_close(A.float() @ B.float().T, C.float(), atol=1e-1, rtol=1e-2)


if __name__ == "__main__" and is_cdna4():
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 8192, 8192, 8192
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    def bench(name, fn):
        ms = triton.testing.do_bench(fn, warmup=200, rep=1000)
        print(f"{name}: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

    bench("matmul", lambda: matmul_cdna4(A, B, matmul_cdna4_kernel))
    bench("matmul_subtile", lambda: matmul_cdna4(A, B, matmul_cdna4_subtiled_kernel))

# %%
# ```
# matmul: 1039 TFLOPS
# matmul_subtile: 1075 TFLOPS
# ```

# %%
# CDNA5 WMMA
# ----------
#
# CDNA5 replaces MFMA with WMMA. Most of the concepts and optimizations we
# introduced above apply to WMMA, but with slight differences, especially since
# CDNA5 uses a warp size of 32.
#
# Similarly, WMMA is exposed in Gluon as `wmma(a, b, acc)`. The accumulator and
# the operands need specific layouts based on `AMDWMMALayout`. `AMDWMMALayout`
# describes the placement of the warps in a different way - instead of
# `warps_per_cta`, it uses linear-layout-style bases, `warp_bases`. For example
# `warp_bases=[[0, 1], [1, 0]]` is equivalent to `warps_per_cta=[2, 2]` in
# `AMDMFMALayout`.

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
# We can see here each warp has 32 lanes and owns one 16x16 tile of the 32x32
# block. Warp 0 (lanes T0-T31) has the top left tile, warp 1 (T32-T63) the top
# right tile, and so on.


@gluon.jit
def wmma_kernel(a_ptr, b_ptr, c_ptr,  #
                M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,  #
                WARP_BASES: gl.constexpr):
    # CDNA5's BF16 WMMA only has shape 16x16x32
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(  #
        version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=WARP_BASES, reg_bases=[])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)

    a = gl.load(a_ptr + offs_2d(M, K, K, a_layout))
    b = gl.load(b_ptr + offs_2d(K, N, N, b_layout))

    acc = gl.zeros([M, N], gl.float32, wmma_layout)
    acc = cdna5.wmma(a, b, acc)

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
# Another big change on CDNA5 is LDS. On CDNA5, a CTA runs on a workgroup
# processor (WGP), which takes the place of the compute unit (CU) of CDNA4. A
# WGP has 320 KB of LDS, twice the 160 KB of a CDNA4 CU, so a kernel can keep
# more and larger tiles in flight. CDNA5 still uses 64 banks of 4 bytes. With 32
# lanes per warp, `ds_load_b128` can be served in 2 groups instead of 4 now.
#
# Moreover, as mentioned before, async copies also changed on CDNA5. On CDNA4,
# an async copy can only write to consecutive LDS addresses for lanes in a warp.
# On CDNA5, an async copy takes a shared memory address for each lane, so it no
# longer has this requirement. This makes it easier to use the padded layouts.
# CDNA5 also adds TDM, which makes copies a lot easier. There is no need to
# consider the layout details for global memory anymore.


@gluon.jit
def wmma_smem_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                     B_K_CONTIGUOUS: gl.constexpr, smem_layout: gl.constexpr):
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(  #
        version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[])

    a_desc = cdna5.tdm.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [M, K], smem_layout)
    a_smem = gl.allocate_shared_memory(gl.bfloat16, [M, K], smem_layout)
    cdna5.tdm.async_load(a_desc, dest=a_smem)
    if B_K_CONTIGUOUS:
        b_desc = cdna5.tdm.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [N, K], smem_layout)
        b_smem = gl.allocate_shared_memory(gl.bfloat16, [N, K], smem_layout)
    else:
        b_desc = cdna5.tdm.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [K, N], smem_layout)
        b_smem = gl.allocate_shared_memory(gl.bfloat16, [K, N], smem_layout)
    cdna5.tdm.async_load(b_desc, dest=b_smem)
    cdna5.tdm.async_wait(0)

    a = a_smem.load(gl.DotOperandLayout(0, wmma_layout, 8))
    if B_K_CONTIGUOUS:
        b = b_smem.permute([1, 0]).load(gl.DotOperandLayout(1, wmma_layout, 8))
    else:
        b = b_smem.load(gl.DotOperandLayout(1, wmma_layout, 8))
    acc = cdna5.wmma(a, b, gl.zeros([M, N], gl.float32, wmma_layout))
    gl.store(c_ptr + offs_2d(M, N, N, wmma_layout), acc)


def wmma_smem(A, B, b_k_contiguous):
    M, K = A.shape
    N = B.shape[0] if b_k_contiguous else B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    compiled = wmma_smem_kernel[(1, )](A, B, C, M, N, K, b_k_contiguous, smem_layout)
    return C, compiled


@pytest.mark.parametrize("b_k_contiguous", [True, False])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_wmma_smem(b_k_contiguous):
    A = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    C, _ = wmma_smem(A, B.T.contiguous() if b_k_contiguous else B, b_k_contiguous)
    torch.testing.assert_close(A.float() @ B.float(), C, atol=1e-3, rtol=1e-3)


if __name__ == "__main__" and is_cdna5():
    wmma_layout = gl.amd.AMDWMMALayout(  #
        version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[])
    a_layout = gl.DotOperandLayout(0, wmma_layout, 8)
    plain_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, plain_layout, 7)

# %%
# When using TDM, we still need to consider bank conflicts. TDM only supports
# padding but not swizzling on CDNA5, and it adds the padding on the fly while
# it writes the tile. We can use similar heuristics as on CDNA4 to decide the
# padded layout. The interval stays one row of 64 values. For the amount, we can
# count the different columns that a group reads as well. On CDNA5, a group has
# 16 consecutive lanes, which all read columns 0-7, so the amount can be 8.

if __name__ == "__main__" and is_cdna5():
    padded_layout = gl.PaddedSharedLayout.with_identity_for(  #
        interval_padding_pairs=[[64, 8]],  #
        shape=[256, 64],  #
        order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, padded_layout, 0)

# %%
# Only considering bank conflicts on CDNA5 is not enough. Another new aspect of
# conflicts on LDS is the "partition conflict". On CDNA5, a WGP has 4 SIMDs,
# which are the units that run the warps. Warp `i` runs on SIMD `i % 4`, and the
# 4 SIMDs work in two pairs, SIMDs 0 and 2, and SIMDs 1 and 3. Each pair has its
# own port into the LDS.
#
# Behind the ports, the 320 KB of LDS consists of 5 physical partitions of
# 64 KB. Both pairs can read in the same cycle only if they access different
# partitions. Otherwise their reads are serialized, and each pair gets half of
# the LDS bandwidth.
#
# The simple `warp_bases=[[0, 1], [1, 0]]` layout runs into this problem, where
# warps 0 and 1 sit on different SIMD pairs but load tiles of the same row,
# which live in the same partition. Let's say we split A by rows A0 and A1. So
# we can see here W0 and W1 both access the same A0 from 2 SIMD pairs. B has no
# such issue, because warps 0 and 2, which read B0, are on the same pair.
#
# ```
#     B0  B1
# A0  W0  W1
# A1  W2  W3
# ```
#
# To avoid it, the two SIMD pairs must read different rows of A and B, and these
# rows must also sit in different partitions. We can further split A0 into two
# halves, A00 and A01, and A1 into A10 and A11. Here we have 4 splits of A, and
# we can arrange them as shown below, where in each step, the two SIMD pairs
# access different halves of A. Apart from this splitting, A0 and A1 still need
# to be placed in different partitions, which is handled by the compiler's
# shared memory allocator.
#
# ```
#      step 1    step 2
#      ------    ------
#      B0  B1    B0  B1
# A00  W0  .     .   W1
# A01  W2  .     .   W3
# A10  .   W1    W0  .
# A11  .   W3    W2  .
# ```
#
# Gluon describes this optimization with another shared memory layout,
# `PartitionedSharedLayout`. It cuts a shared memory buffer into pieces along
# `partition_dim`, places the pieces in different partitions, and lays out each
# piece with `partition_layout`, here the padded layout from above.
#
# To help generate such a layout for WMMA, we can use
# `make_partitioned_dot_layouts`. It takes the padded layouts of A and B, and
# returns a `PartitionedSharedLayout` for each of them, together with a WMMA
# layout that places the warps of each SIMD pair on their own pieces.

if __name__ == "__main__" and is_cdna5():
    a_smem_layout, b_smem_layout, wmma_layout = cdna5.make_partitioned_dot_layouts(  #
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

# %%
# Now let's combine these CDNA5 features together to write a matmul kernel. TDM
# copies the tiles into two LDS buffers, and the warps read them with the padded
# or the partitioned layouts.
#
# Similar to the MFMA kernel, each K block is split into two sub-tiles of 32.
# While the WMMAs work on one sub-tile, the LDS reads for the other are in
# flight. The output goes back through LDS and a TDM store.


@gluon.aggregate
class WmmaLayouts:
    # Operand A's shared memory and register layouts
    a_smem_layout: gl.constexpr
    a_reg_layout: gl.constexpr
    # Operand B's shared memory and register layouts
    b_smem_layout: gl.constexpr
    b_reg_layout: gl.constexpr
    # Accumulator's register layout
    acc_layout: gl.constexpr


@gluon.constexpr_function
def get_wmma_layouts(partitioned):
    # Assume the 256x256x64 block size and 4 warps per CTA
    # Operand A has shape [256, 64]
    # Operand B has shape [256, 64] as well, so transposed
    a_smem_layout = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [256, 64], [1, 0])
    b_smem_layout = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [256, 64], [1, 0])
    if partitioned:
        a_smem_layout, b_smem_layout, acc_layout = cdna5.make_partitioned_dot_layouts(  #
            256, 256, a_smem_layout, b_smem_layout, 4, [16, 16, 32], b_transposed=True)
    else:
        acc_layout = gl.amd.AMDWMMALayout(  #
            version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[])
    a_reg_layout = gl.DotOperandLayout(0, acc_layout, 8)
    b_reg_layout = gl.DotOperandLayout(1, acc_layout, 8)
    return WmmaLayouts(a_smem_layout, a_reg_layout,  #
                       b_smem_layout, b_reg_layout,  #
                       acc_layout)


@gluon.jit
def matmul_cdna5_ld_smem_subtiled(cfg, layouts, k, a_smem, b_smem, k_start, k_size):
    a_buf = a_smem.index(k % cfg.NUM_BUFFERS).slice(k_start, k_size, 1)
    b_buf = b_smem.index(k % cfg.NUM_BUFFERS).slice(k_start, k_size, 1)
    a = a_buf.load(layouts.a_reg_layout)
    b = b_buf.permute([1, 0]).load(layouts.b_reg_layout)
    return a, b


@gluon.jit
def matmul_cdna5_ld_gmem(cfg, k, pred, off_m, off_n, a_desc, b_desc, a_smem, b_smem):
    a_desc = cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, k * cfg.BLOCK_K], pred=pred,
                                                clamp_bounds=True)
    b_desc = cdna5.tdm.update_tensor_descriptor(b_desc, add_offsets=[off_n, k * cfg.BLOCK_K], pred=pred,
                                                clamp_bounds=True)
    cdna5.tdm.async_load(a_desc, dest=a_smem.index(k % cfg.NUM_BUFFERS))
    cdna5.tdm.async_load(b_desc, dest=b_smem.index(k % cfg.NUM_BUFFERS))


@gluon.jit
def matmul_cdna5_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                        PARTITIONED: gl.constexpr, NUM_BUFFERS: gl.constexpr = 2):
    cfg = MatmulConfig(M, N, K, a_ptr.dtype.element_ty,  #
                       BLOCK_M=256, BLOCK_N=256, BLOCK_K=64,  #
                       NUM_BUFFERS=NUM_BUFFERS)
    layouts = get_wmma_layouts(PARTITIONED)

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(cfg.N, cfg.BLOCK_N)
    off_m = pid // num_pid_n * cfg.BLOCK_M
    off_n = pid % num_pid_n * cfg.BLOCK_N

    a_desc = cdna5.tdm.make_tensor_descriptor(a_ptr, [cfg.M, cfg.K], [cfg.K, 1], [cfg.BLOCK_M, cfg.BLOCK_K],
                                              layouts.a_smem_layout)
    b_desc = cdna5.tdm.make_tensor_descriptor(b_ptr, [cfg.N, cfg.K], [cfg.K, 1], [cfg.BLOCK_N, cfg.BLOCK_K],
                                              layouts.b_smem_layout)
    a_smem = gl.allocate_shared_memory(cfg.dtype, [cfg.NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(cfg.dtype, [cfg.NUM_BUFFERS, cfg.BLOCK_N, cfg.BLOCK_K], layouts.b_smem_layout)

    num_k = gl.cdiv(cfg.K, cfg.BLOCK_K)
    for k in gl.static_range(cfg.NUM_BUFFERS - 1):
        matmul_cdna5_ld_gmem(cfg, k, k < num_k, off_m, off_n, a_desc, b_desc, a_smem, b_smem)

    acc = gl.zeros([cfg.BLOCK_M, cfg.BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k):
        next_k = k + cfg.NUM_BUFFERS - 1
        matmul_cdna5_ld_gmem(cfg, next_k, next_k < num_k, off_m, off_n, a_desc, b_desc, a_smem, b_smem)
        cdna5.tdm.async_wait((cfg.NUM_BUFFERS - 1) * 2)
        a0, b0 = matmul_cdna5_ld_smem_subtiled(cfg, layouts, k, a_smem, b_smem, 0, cfg.BLOCK_K // 2)
        a1, b1 = matmul_cdna5_ld_smem_subtiled(cfg, layouts, k, a_smem, b_smem, cfg.BLOCK_K // 2, cfg.BLOCK_K // 2)
        acc = cdna5.wmma(a0, b0, acc)
        acc = cdna5.wmma(a1, b1, acc)

    c_smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(  #
        [[cfg.BLOCK_N, 8]], [cfg.BLOCK_M, cfg.BLOCK_N], [1, 0])
    c_smem = gl.allocate_shared_memory(c_ptr.dtype.element_ty, [cfg.BLOCK_M, cfg.BLOCK_N], c_smem_layout)
    c_smem.store(acc.to(c_ptr.dtype.element_ty))
    c_desc = cdna5.tdm.make_tensor_descriptor(c_ptr, [cfg.M, cfg.N], [cfg.N, 1], [cfg.BLOCK_M, cfg.BLOCK_N],
                                              c_smem_layout)
    c_desc = cdna5.tdm.update_tensor_descriptor(c_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
    cdna5.tdm.async_store(c_desc, src=c_smem)
    cdna5.tdm.async_wait(0)


def matmul_cdna5(A, B, partitioned):
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256), )
    compiled = matmul_cdna5_kernel[grid](A, B, C, M, N, K, partitioned)
    return C, compiled


@pytest.mark.parametrize("M, N, K", [(256, 256, 256), (500, 300, 512), (1024, 512, 1024)])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul_cdna5(M, N, K, partitioned):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C, _ = matmul_cdna5(A, B, partitioned)
    torch.testing.assert_close(A.float() @ B.float().T, C.float(), atol=1e-1, rtol=1e-2)


if __name__ == "__main__" and is_cdna5():
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 8192, 8192, 8192
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    for name, partitioned in [("padded", False), ("partitioned", True)]:
        ms = triton.testing.do_bench(lambda: matmul_cdna5(A, B, partitioned), warmup=200, rep=1000)
        print(f"{name}: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

# %%
# ```
# padded: 2247 TFLOPS
# partitioned: 2300 TFLOPS
# ```

# %%
# This tutorial shows how important layouts are in Gluon for AMD GPUs. Many
# optimizations come from choosing the right layout and letting the compiler
# generate the expected instructions. Layouts control how data flows from global
# memory to shared memory, from shared memory to registers, and between
# registers. The data path must be managed carefully to keep the compute unit
# busy.
