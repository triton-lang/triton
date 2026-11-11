"""
WMMA in Gluon
=============

On CDNA5, the matrix cores take wave matrix multiply-accumulate (WMMA)
instructions instead of MFMA. Similar to MFMA, WMMA reads its operands from
registers. So the same question decides the performance: can the operands move
from shared memory (LDS) into registers as fast as the matrix cores consume
them?

We start with a single WMMA, using bf16. Then we feed WMMA from LDS and remove
bank conflicts. CDNA5 also brings a second kind of LDS conflict, the partition
conflict. Finally, we put everything together in a matrix multiplication
kernel.
"""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna5 import tdm, wmma


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not is_cdna5():
    raise RuntimeError("This tutorial requires an AMD CDNA5 GPU")

# A warp on CDNA5 GPUs has 32 lanes.
warp_size = 32

# %%
# In Gluon, a WMMA is exposed as `wmma(a, b, acc)`. Just like `mfma`, the
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

if __name__ == "__main__":
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
def offs_2d(rows: gl.constexpr, cols: gl.constexpr, stride, layout: gl.constexpr):
    r = gl.arange(0, rows, gl.SliceLayout(1, layout))
    c = gl.arange(0, cols, gl.SliceLayout(0, layout))
    return r[:, None] * stride + c[None, :]


@gluon.jit
def wmma_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, WARP_BASES: gl.constexpr):
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(  #
        version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=WARP_BASES, reg_bases=[])
    a = gl.load(a_ptr + offs_2d(M, K, K, gl.DotOperandLayout(0, wmma_layout, 8)))
    b = gl.load(b_ptr + offs_2d(K, N, N, gl.DotOperandLayout(1, wmma_layout, 8)))
    acc = wmma(a, b, gl.zeros([M, N], gl.float32, wmma_layout))
    gl.store(c_ptr + offs_2d(M, N, N, wmma_layout), acc)


def wmma_matmul(A, B, warp_bases):
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
    C = wmma_matmul(A, B, warp_bases)
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
    acc = wmma(a, b, gl.zeros([M, N], gl.float32, wmma_layout))
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


@gluon.jit
def bank_conflicts_kernel(smem_layout: gl.constexpr, NUM_CONFLICTS: gl.constexpr):
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(  #
        version=3, instr_shape=[16, 16, 32], transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[])
    a_smem = gl.allocate_shared_memory(gl.bfloat16, [256, 64], smem_layout)
    a = a_smem.load(gl.DotOperandLayout(0, wmma_layout, 8))
    gl.static_assert(gl.bank_conflicts(a.type, a_smem.type) == NUM_CONFLICTS)


if __name__ == "__main__":
    plain_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    bank_conflicts_kernel[(1, )](plain_layout, 7)

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

if __name__ == "__main__":
    swizzled_layout = gl.SwizzledSharedLayout(  #
        vec=8, per_phase=2, max_phase=8, order=[1, 0])
    bank_conflicts_kernel[(1, )](swizzled_layout, 0)

    padded_layout = gl.PaddedSharedLayout.with_identity_for(  #
        # Pad 8 elements after every 64 elements
        interval_padding_pairs=[[64, 8]],  #
        shape=[256, 64],  #
        order=[1, 0])
    bank_conflicts_kernel[(1, )](padded_layout, 0)

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

if __name__ == "__main__":
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
class MatmulLayouts:
    # Accumulator and operand register layouts
    acc_layout: gl.constexpr
    a_reg_layout: gl.constexpr
    b_reg_layout: gl.constexpr
    # Shared memory layouts of A, B and C
    a_smem_layout: gl.constexpr
    b_smem_layout: gl.constexpr
    c_smem_layout: gl.constexpr


@gluon.constexpr_function
def get_layouts(partitioned):
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
    return MatmulLayouts(acc_layout, a_reg_layout, b_reg_layout, a_smem_layout, b_smem_layout, c_smem_layout)


@gluon.jit
def issue_copy(k, a_desc, b_desc, a_smem, b_smem, off_m, off_n, BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr):
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, k * BLOCK_K], clamp_bounds=True)
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[off_n, k * BLOCK_K], clamp_bounds=True)
    tdm.async_load(a_desc, dest=a_smem.index(k % NUM_BUFFERS))
    tdm.async_load(b_desc, dest=b_smem.index(k % NUM_BUFFERS))


@gluon.jit
def load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS: gl.constexpr, k_start: gl.constexpr,
                           k_size: gl.constexpr):
    a_buf = a_smem.index(k % NUM_BUFFERS).slice(k_start, k_size, 1)
    b_buf = b_smem.index(k % NUM_BUFFERS).slice(k_start, k_size, 1)
    a = a_buf.load(layouts.a_reg_layout)
    b = b_buf.permute([1, 0]).load(layouts.b_reg_layout)
    return a, b


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
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                  stride_am, stride_ak, stride_bn, stride_bk, stride_cm,  #
                  PARTITIONED: gl.constexpr, NUM_BUFFERS: gl.constexpr = 2):
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = 256
    BLOCK_K: gl.constexpr = 64
    layouts = get_layouts(PARTITIONED)

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
        issue_copy(i, a_desc, b_desc, a_smem, b_smem, off_m, off_n, BLOCK_K, NUM_BUFFERS)
    tdm.async_wait((NUM_BUFFERS - 1) * 2)

    a0, b0 = load_subtiled_operands(0, a_smem, b_smem, layouts, NUM_BUFFERS, 0, SUB_K)
    a1, b1 = load_subtiled_operands(0, a_smem, b_smem, layouts, NUM_BUFFERS, SUB_K, SUB_K)
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    acc = wmma(a0, b0, acc)

    for k in range(1, num_k - (NUM_BUFFERS - 1)):
        tdm.async_wait((NUM_BUFFERS - 2) * 2)
        a0, b0 = load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS, 0, SUB_K)
        acc = wmma(a1, b1, acc)
        # K block `k - 1` is done, so its slot can take the next copy.
        issue_copy(k + NUM_BUFFERS - 1, a_desc, b_desc, a_smem, b_smem, off_m, off_n, BLOCK_K, NUM_BUFFERS)
        a1, b1 = load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS, SUB_K, SUB_K)
        acc = wmma(a0, b0, acc)

    for i in gl.static_range(NUM_BUFFERS - 1):
        k = num_k - (NUM_BUFFERS - 1) + i
        tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        a0, b0 = load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS, 0, SUB_K)
        acc = wmma(a1, b1, acc)
        a1, b1 = load_subtiled_operands(k, a_smem, b_smem, layouts, NUM_BUFFERS, SUB_K, SUB_K)
        acc = wmma(a0, b0, acc)
    acc = wmma(a1, b1, acc)

    store_output(acc, c_ptr, M, N, stride_cm, off_m, off_n, layouts.c_smem_layout)


def matmul(A, B, partitioned, num_buffers=2):
    M, K = A.shape
    N = B.shape[0]
    assert triton.cdiv(K, 64) >= num_buffers
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256), )
    compiled = matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                                   partitioned, num_buffers, num_warps=4)
    return C, compiled


@pytest.mark.parametrize("M, N, K", [(256, 256, 256), (500, 300, 512), (1024, 512, 1024)])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("num_buffers", [2, 3])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul(M, N, K, partitioned, num_buffers):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C, _ = matmul(A, B, partitioned, num_buffers)
    torch.testing.assert_close(A.float() @ B.float().T, C.float(), atol=1e-1, rtol=1e-2)


# %%
# We run the kernel with the partitioned layouts and 2, 3 and 4 buffers on an
# MI455X GPU with M = N = K = 8192.

if __name__ == "__main__":
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 8192, 8192, 8192
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    for num_buffers in [2, 3, 4]:
        ms = triton.testing.do_bench(lambda: matmul(A, B, True, num_buffers), warmup=200, rep=1000)
        print(f"{num_buffers} buffers: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

# %%
# ```
# 2 buffers: 2318 TFLOPS
# 3 buffers: 2711 TFLOPS
# 4 buffers: 2736 TFLOPS
# ```
