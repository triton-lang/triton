"""
MFMA and WMMA
=============

AMD GPUs run matrix multiplications on dedicated matrix cores. CDNA4 GPUs use
matrix fused-multiply-add (MFMA) instructions, and CDNA5 GPUs use wave matrix
multiply-accumulate (WMMA) instructions. Both read their operands from registers
and accumulate in registers. As they can't consume shared memory (LDS) directly,
so the kernel needs extra instructions to load every operand from LDS into
registers.

This tutorial will introduces MFMA/WMMA and its layouts with the bf16 data type.
We will also cover different ways to load the operands from LDS and demonstrate
its importance for performance.
"""

import importlib
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4, cdna5
from triton.experimental.gluon.language.amd.cdna4 import async_copy, buffer
from triton.experimental.gluon.language.amd.cdna5 import tdm

t1 = importlib.import_module("01-buffer-ops")
t2 = importlib.import_module("02-async-copy")


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
# Gluon exposes MFMA as `mfma(a, b, acc)`. This operation has layout
# requirements for its operands. The accumulator `acc` must use an
# `AMDMFMALayout`, and the operands `a` and `b` must use a `DotOperandLayout`
# with an MFMA layout as parent. These layouts control which MFMA instructions
# the compiler generates.
#
# The operand data type and `instr_shape` select the MFMA instruction of each
# warp. `instr_shape` is the shape of the instruction in [M, N, K] format for
# one warp. With multiple warps, `warps_per_cta` distributes them over the
# accumulator along the M and N dimensions. When you write a kernel, you choose
# them based on the block size.


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
# For bf16, CDNA4 has two MFMA instruction shapes:
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
# Knowing the MFMA layouts helps us avoid layout conversions. Kernels like
# attention chain two MFMAs, and the output of the first one, after a softmax,
# becomes an operand of the second one. As shown above, the accumulator layout
# differs from the operand layout, so the compiler has to convert between them.
# We can use the `transposed` flag and `k_width` to make the two layouts match
# and avoid this conversion.

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
# To make the two layouts identical, we also need `k_width`. `k_width` is the
# number of consecutive elements along K that a lane holds in an MFMA operand.
# The 16x16x32 MFMA instruction takes 8 values of A and 8 values of B per lane,
# so the natural choice is `k_width=8`. But we can use a different `k_width` as
# long as both operands use the same one. The results stay correct, because
# this only changes the order of the elements along K, in the same way for A
# and B. With `k_width=4`, each lane holds 4 consecutive elements, which exactly
# matches the transposed accumulator layout.

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
# All layouts shown above are for registers. Real matrix multiplication kernels
# usually load the data into shared memory with async copies first, and then
# copy it from shared memory to registers. Gluon needs extra layouts for shared
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
        # When B is K contiguous in shared memory, we need to permute
        # from [N, K] to [K, N] - this is a requirement for the `mfma` op.
        # The permutation itself is free. It only generates the ds_read
        # instructions without other things.
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
#   ds_read_b128 v[18:21], v8
#   ds_read_b128 v[22:25], v8 offset:64
#   ...
# ```
#
# With `k_width=8`, each lane holds 8 consecutive bf16 values along K for an
# MFMA, which is 16 bytes. When K is contiguous in shared memory, these 16 bytes
# are adjacent, and one `ds_read_b128` loads them. Operand A takes 4
# `ds_read_b128` in both cases.
#
# Take `ds_read_b128 v[26:29], v8 offset:64` as an example. It reads 128 bits,
# or 16 bytes, for each lane. Each lane passes its own shared memory address in
# a VGPR, here `v8`, and the instruction adds the immediate `offset` in bytes.
# The 16 bytes land in 4 VGPRs, `v[26:29]`, which hold the 8 bf16 values of the
# lane. The compiler computes the address once and reaches other parts of the
# tile with the offset. Here `offset:64` moves 32 bf16 values along K, to the
# operand of the next MFMA.
#
# For operand B, when it is K-contiguous, the compiler uses the same regular
# reads as for A. When it is N-contiguous, the compiler switches to the
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
# When picking the shared memory layout, we only need to consider the lanes in a
# group. For a swizzled layout, `vec` can be `k_width`, `per_phase` can be the
# number of rows in one 256-byte line of banks, and `max_phase` can be the
# number of rows that a group reads divided by `per_phase`. For example, in our
# 256x64 bf16 case, a row has 128 bytes which is 32 banks. So 2 rows can occupy
# the full 64 banks. Therefore, we can set `per_phase=2`. And as we know
# `ds_read_b128` has 16 lanes per group, and these lanes will read 16 different
# rows, so `max_phase` can be 16 / `per_phase` = 8.
#
# For a padded layout, the heuristic is simpler. We just pad a fixed amount
# after every row. The pad amount is the number of different columns that a
# group reads. In our case, there are 64 elements per row, and a lane loads 8
# elements at once. A group reads columns 0-7 of rows 0-3 and 12-15, and columns
# 8-15 of rows 4-7 and 8-11. These are 16 columns, so we set the
# interval-padding pair to [64, 16].

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
# in the async copy tutorial, a copy instruction has no shared memory address
# for each lane, so the data it writes into shared memory needs to be
# consecutive. So we cannot add padding between the rows, and the padded layout
# above fails to compile with async copies.


@gluon.jit
def async_copy_tile_kernel(a_ptr, gmem_layout: gl.constexpr, smem_layout: gl.constexpr):
    smem = gl.allocate_shared_memory(gl.bfloat16, [256, 64], smem_layout)
    a_desc = buffer.make_buffer_descriptor(a_ptr, 256 * 64 * 2)
    buffer.load_to_shared(smem, a_desc, offs_2d(256, 64, 64, gmem_layout))
    async_copy.commit_group()
    async_copy.wait_group(0)


def try_compile_async_copy(gmem_layout, smem_layout):
    A = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    try:
        with t2.suppress_stderr():
            async_copy_tile_kernel[(1, )](A, gmem_layout, smem_layout, num_warps=4)
        return True
    except RuntimeError:
        return False


if __name__ == "__main__" and is_cdna4():
    gmem_layout = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    padded_layout = gl.PaddedSharedLayout.with_identity_for(  #
        interval_padding_pairs=[[64, 16]], shape=[256, 64], order=[1, 0])
    # The `gmem_layout` can not fill the whole row in one load which creates
    # non-consecutive shared memory. So the compile will fail.
    assert not try_compile_async_copy(gmem_layout, padded_layout)

# %%
# There is a trick to work around this. Instead of adding gaps within one copy
# instruction, we add them between instructions or across warps, given a tile
# is composed of multiple instructions. And we can carefully shuffle elements
# across instructions to avoid bank conflicts.
#
# Still use our 256x64 tile and `k_width=8` example. We will use `ds_read_b128`
# for shared memory and `buffer_load_dwordx4 ... lds` for global memory. As we
# can pad inside one shared memory load, we have to set the interval to be 512
# which is 1KB, exactly the amount one `ds_read_b128` will read. Similar as
# before, consider one group of lanes will touch 16 columns, we can still use
# padding amount to be 16.
#
# But with padding after every 512 values, the 8 rows written by one copy
# instruction still have no padding between them. So we also reorder the rows
# in shared memory. We place 16 consecutive rows 1 KB apart, so that each of
# them starts a new copy instruction and gets one more padding than the previous
# one. Then we fill the holes in between with the rows that are a multiple of 16
# apart. In shared memory, the rows are arranged like this:
#
# ```
# r0,   r16,  r32,  r48,  r64,  r80,  r96,  r112, padding,
# r1,   r17,  r33,  r49,  r65,  r81,  r97,  r113, padding,
# ...
# r15,  r31,  r47,  r63,  r79,  r95,  r111, r127, padding,
# r128, r144, r160, r176, r192, r208, r224, r240, padding,
# ...
# ```
#
# Rows 0 to 15 now start 1056 bytes apart, which is 32 bytes apart within the
# 256 bytes of all banks, the same as with per-row padding. The rows in the
# same 1 KB are 16 apart, and one group of lanes never reads them together.
#
# `PaddedSharedLayout` describes this reordering with `offset_bases`. Like the
# bases of a linear layout, they map each bit of the offset in shared memory to
# a row and a column of the tile. The first 6 bases cover the 64 values of a
# row. The next 3 bases fill the rest of one 1 KB with rows 16, 32 and 64. The
# next 4 bases move to rows 1, 2, 4 and 8 in the next 1 KB, and the last base
# moves to row 128.


@gluon.constexpr_function
def get_smem_layout_for_async_copies(shape, dtype):
    # One copy instruction writes 1 KB, and we pad 32 bytes after it.
    rows, cols = shape
    elem_bytes = dtype.primitive_bitwidth // 8
    rows_per_copy = 1024 // (cols * elem_bytes)
    # The values of a row
    col_bases = [[0, 2**i] for i in range(cols.bit_length() - 1)]
    # Rows 16 apart within one copy instruction
    row_bases = [[16 * 2**i, 0] for i in range(rows_per_copy.bit_length() - 1)]
    # The next copy instructions
    row_bases += [[1, 0], [2, 0], [4, 0], [8, 0]]
    # The rest of the tile
    row_bases += [[2**i, 0] for i in range((16 * rows_per_copy).bit_length() - 1, rows.bit_length() - 1)]
    return gl.PaddedSharedLayout(  #
        interval_padding_pairs=[[1024 // elem_bytes, 32 // elem_bytes]],  #
        offset_bases=col_bases + row_bases,  #
        cga_layout=[],  #
        shape=shape)


# %%
# The async copy has to write the rows in the same order. In one
# `buffer_load_dwordx4 ... lds`, the lanes write their 16 bytes one after
# another into the 1 KB, so lanes 0-7 have to load row 0, lanes 8-15 row 16,
# and so on. `BlockedLayout` can't describe this, so we build the layout for the
# global memory side as a `DistributedLinearLayout` from the same bases. The
# first 3 register bases and the 6 lane bases are the first 9 bases of the
# shared memory layout, which cover the 1 KB of one copy instruction. The warp
# bases and the other register bases are the remaining bases, which pick the
# 1 KB to write. Each copy instruction still loads 8 full rows from global
# memory.


@gluon.constexpr_function
def get_gmem_layout_for_async_copies(shape, dtype):
    # Each lane copies 16 bytes, in the order of the shared memory layout. The
    # bases go to the registers of one lane, the 64 lanes, and the 4 warps.
    bases = get_smem_layout_for_async_copies(shape, dtype).offset_bases
    vec_end = (128 // dtype.primitive_bitwidth).bit_length() - 1
    lane_end = vec_end + 6
    warp_end = lane_end + 2
    return gl.DistributedLinearLayout(  #
        reg_bases=bases[:vec_end] + bases[warp_end:],  #
        lane_bases=bases[vec_end:lane_end],  #
        warp_bases=bases[lane_end:warp_end],  #
        block_bases=[],  #
        shape=shape)


if __name__ == "__main__" and is_cdna4():
    gmem_layout = get_gmem_layout_for_async_copies([256, 64], gl.bfloat16)
    smem_layout = get_smem_layout_for_async_copies([256, 64], gl.bfloat16)
    assert try_compile_async_copy(gmem_layout, smem_layout)
    bank_conflicts_kernel[(1, )](a_layout, smem_layout, 0)

# %%
# Now we combine these features into a pipelined matmul kernel. To keep the
# search space small, we only use 256x256x64 blocks and 4 warps.


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

    # Both operands use the padded layout and the copy layout from above.
    gmem_layout = get_gmem_layout_for_async_copies([256, 64], gl.bfloat16)
    smem_layout = get_smem_layout_for_async_copies([256, 64], gl.bfloat16)
    return MfmaLayouts(gmem_layout, smem_layout, a_reg_layout,  #
                       gmem_layout, smem_layout, b_reg_layout,  #
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
# This tutorial shows how important layouts are in Gluon for AMD GPUs. Many
# optimizations come from choosing the right layout and letting the compiler
# generate the expected instructions. Layouts control how data flows from global
# memory to shared memory, from shared memory to registers, and between
# registers. The data path must be managed carefully to keep the compute unit
# busy.

# %%
# CDNA5 WMMA
# ----------
#
# CDNA5 replaces MFMA with WMMA, the wave matrix multiply-accumulate
# instructions. The idea stays the same, but the instructions and their layouts
# change slightly, given CDNA5 has a warp size of 32. WMMA is exposed in Gluon
# as `wmma(a, b, acc)`. Same as `mfma`, the accumulator and the operands need
# specific layouts, here based on `AMDWMMALayout` instead of `AMDMFMALayout`.
#
# `AMDWMMALayout` describes the placement of the warps in a different way.
# Instead of `warps_per_cta`, it uses bases in the style of a linear layout,
# for both the warps and the registers. Each basis is an offset in units of
# instruction tiles. `warp_bases` gives the offset of each bit of the warp ID,
# and `reg_bases` gives the offset of each bit of the index of the instruction
# tiles that one warp repeats in its registers. With
# `warp_bases=[[0, 1], [1, 0]]`, warp 1 takes the next tile along N and warp 2
# the next tile along M, which is the same 2x2 warp grid as
# `warps_per_cta=[2, 2]`.

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
# (lanes T0-T31) has the top left tile, warp 1 (T32-T63) the top right tile, and
# so on.


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
# more and larger tiles in flight. CDNA5 still uses 64 banks of 4 bytes. With 32
# lanes per warp, `ds_load_b128` can be served in 2 groups of lanes instead of
# 4.
#
# Another big change, as mentioned in `02-async-copy.py`, is async copies. On
# CDNA4, an async copy writes the data of each lane to consecutive LDS
# addresses. On CDNA5, an async copy takes a shared memory address for each
# lane, so it no longer has this requirement, and the padded layouts work
# without any trick. Moreover, CDNA5 also adds TDM, as shown in `03-tdm.py`. TDM
# makes copies a lot easier, without much consideration of the layout details,
# so TDM is what we prefer to use.


@gluon.jit
def wmma_smem_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                     B_K_CONTIGUOUS: gl.constexpr, smem_layout: gl.constexpr):
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(  #
        version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[])
    a_desc = tdm.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [M, K], smem_layout)
    a_smem = gl.allocate_shared_memory(gl.bfloat16, [M, K], smem_layout)
    tdm.async_load(a_desc, [0, 0], a_smem)
    if B_K_CONTIGUOUS:
        b_desc = tdm.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [N, K], smem_layout)
        b_smem = gl.allocate_shared_memory(gl.bfloat16, [N, K], smem_layout)
    else:
        b_desc = tdm.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [K, N], smem_layout)
        b_smem = gl.allocate_shared_memory(gl.bfloat16, [K, N], smem_layout)
    tdm.async_load(b_desc, [0, 0], b_smem)
    tdm.async_wait(0)

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
    compiled = wmma_smem_kernel[(1, )](A, B, C, M, N, K, b_k_contiguous, smem_layout, num_warps=4)
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
# it writes the tile. Without padding, the read has 8-way conflicts. We can use
# the similar heuristics as on CDNA4 to decide the padded layout. The interval
# stays one row of 64 values, so that each row can start at different banks.
# For the amount, we again count the different columns that a group reads. On
# CDNA5, a group has 16 consecutive lanes, and lanes 0-15 all read columns 0-7,
# each from a different row. This is only 8 columns, so the interval-padding
# pair becomes [64, 8].

if __name__ == "__main__" and is_cdna5():
    padded_layout = gl.PaddedSharedLayout.with_identity_for(  #
        interval_padding_pairs=[[64, 8]],  #
        shape=[256, 64],  #
        order=[1, 0])
    bank_conflicts_kernel[(1, )](a_layout, padded_layout, 0)

# %%
# Only considering bank conflicts on CDNA5 is not enough. Reads of different
# warps can still slow each other down, which we call a partition conflict. A
# WGP has 4 SIMDs, which are the units that run the warps. Warp `i` runs on SIMD
# `i % 4`, and the 4 SIMDs work in two pairs, SIMDs 0 and 2, and SIMDs 1 and 3.
# Each pair has its own port into the LDS. Behind the ports, the 320 KB of LDS
# consists of 5 physical partitions of 64 KB. Both pairs can read in the same
# cycle only if they access different partitions. Otherwise their reads take
# turns, and each pair gets half the bandwidth.
#
# The simple `warp_bases=[[0, 1], [1, 0]]` runs into it. Warps 0 and 1 sit on
# different SIMD pairs but compute tiles in the same rows, so both read the
# same rows of A, which live in the same partition.
#
# To avoid it, the two SIMD pairs must read different rows of A and B, and these
# rows must sit in different partitions. Gluon describes this placement with
# another shared memory layout, `PartitionedSharedLayout`. It cuts a buffer
# into pieces along `partition_dim`, places the pieces in `num_partitions`
# different partitions, and lays out each piece with `partition_layout`, here
# the padded layout from above.
#
# To help generate such a layout, we can use `make_partitioned_dot_layouts`. It
# takes the padded layouts of A and B, and returns a `PartitionedSharedLayout`
# for each of them, together with a WMMA layout that places the warps of each
# SIMD pair on their own pieces.

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
# The bases are in units of 16x16 instruction tiles. Warp 1 starts 8 tiles
# (128 values) down and 8 tiles right of warp 0, and the last register basis
# moves each warp 8 tiles down. The 256x256 output tile is split among the warps
# like this, where "1st" and "2nd" mark which half of its rows a warp reads
# first:
#
# ```
#              N 0-127       N 128-255
# M   0-63     warp 0, 1st   warp 1, 2nd
# M  64-127    warp 2, 1st   warp 3, 2nd
# M 128-191    warp 0, 2nd   warp 1, 1st
# M 192-255    warp 2, 2nd   warp 3, 1st
# ```
#
# Warps 0 and 2 form one SIMD pair, and warps 1 and 3 the other. For B, the
# first pair reads rows 0-127, and the second pair reads rows 128-255. For A,
# the first pair reads rows 0-127 while the second pair reads rows 128-255, and
# then they swap. The shared layout cuts A and B into these two halves along
# dimension 0 and stores them in different partitions, so the two pairs always
# read from different partitions. TDM follows the cut, so each warp copies into
# one half.

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
    # Accumulator's register layout and output's shared memory layout
    acc_layout: gl.constexpr
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
    return WmmaLayouts(a_smem_layout, a_reg_layout,  #
                       b_smem_layout, b_reg_layout,  #
                       acc_layout, c_smem_layout)


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
                       PARTITIONED: gl.constexpr):
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = 256
    BLOCK_K: gl.constexpr = 64
    NUM_BUFFERS: gl.constexpr = 2
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


def wmma_matmul(A, B, partitioned):
    M, K = A.shape
    N = B.shape[0]
    assert triton.cdiv(K, 64) >= 2
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256), )
    compiled = wmma_matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                                        C.stride(0), partitioned, num_warps=4)
    return C, compiled


@pytest.mark.parametrize("M, N, K", [(256, 256, 256), (500, 300, 512), (1024, 512, 1024)])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_wmma_matmul(M, N, K, partitioned):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C, _ = wmma_matmul(A, B, partitioned)
    torch.testing.assert_close(A.float() @ B.float().T, C.float(), atol=1e-1, rtol=1e-2)


# %%
# We can compare the padded and the partitioned layouts on an MI455X GPU with
# M = N = K = 8192.

if __name__ == "__main__" and is_cdna5():
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 8192, 8192, 8192
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    for name, partitioned in [("padded", False), ("partitioned", True)]:
        ms = triton.testing.do_bench(lambda: wmma_matmul(A, B, partitioned), warmup=200, rep=1000)
        print(f"{name}: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

# %%
# ```
# padded: 2206 TFLOPS
# partitioned: 2304 TFLOPS
# ```
