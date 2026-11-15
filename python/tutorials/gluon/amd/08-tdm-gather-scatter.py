"""
TDM Gather and Scatter
======================

TDM moves a tile of a tensor, where each dimension of the tile covers a range
of consecutive indices. It can also copy a tile whose indices are not
consecutive in one dimension, which is called gather and scatter. Gather and
scatter only work for 2D tensors. A gather loads rows of a 2D tensor based on
row indices. A scatter is for storing rows.

In this tutorial, we will explain how TDM gather and scatter work. In the end,
we will use gather and scatter to implement an MoE-style grouped matmul.
"""

import importlib
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna5 import tdm, wmma_scaled

t1 = importlib.import_module("01-buffer-ops")
t7 = importlib.import_module("07-mfma-wmma-scaled")


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not is_cdna5():
    raise RuntimeError("This tutorial requires an AMD CDNA5 GPU")

# A warp on CDNA5 GPUs has 32 lanes.
warp_size = gl.constexpr(32)

# %%
# In Gluon, TDM gather and scatter are exposed as `tdm.async_gather` and
# `tdm.async_scatter`. They take the TDM descriptor and a 1D tensor of row
# indices to specify which rows to load or store.
#
# Similar to normal TDM loads and stores, we can add an offset, but only for the
# columns in gather and scatter. The offset is added by updating the TDM
# descriptor with `update_tensor_descriptor`.
#
# TDM gather and scatter use the same `tensor_load_to_lds` and
# `tensor_store_from_lds` instructions, and the same counter `tensorcnt`. The
# kernel also waits for them with `tdm.async_wait`.
#
# In a typical kernel using gather or scatter, we first load the row indices
# from global memory, and then use them for the gather/scatter, which makes it
# an indirect access. Different from TMA gather/scatter, the only requirement on
# the index layout is that all lanes of a warp hold the same indices.
#
# We can build such a layout by slicing a 2D blocked layout:
#
# ```python
# gl.SliceLayout(
#     dim=1,
#     parent=gl.BlockedLayout(
#         size_per_thread=[BLOCK_M // num_warps, 1],
#         threads_per_warp=[1, warp_size],
#         warps_per_cta=[num_warps, 1],
#         order=[1, 0],
#     ),
# )
# ```
#
# The parent layout puts the indices along dim 0 and the lanes along dim 1.
# Along dim 0, each warp takes the indices of its own consecutive rows, held in
# the registers of a lane. Along dim 1, the lanes of a warp are spread over the
# columns. Slicing away dim 1 removes the only difference between the lanes, so
# all lanes of a warp hold the same indices.
#
# Different warps still hold different rows, so the TDM instructions are spread
# over the warps. If every warp held all the indices instead, for example with
# `size_per_thread=[BLOCK_M, 1]` and `warps_per_cta=[1, num_warps]`, only warp 0
# would issue the instructions, and the other warps would stay idle.


@gluon.jit
def gather_kernel(x_ptr, idx_ptr, out_ptr, M, K, BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr,
                  INDEX_DTYPE: gl.constexpr):
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    x_desc = tdm.make_tensor_descriptor(x_ptr, [M, K], [K, 1], [BLOCK_M, BLOCK_K], smem_layout)
    num_warps: gl.constexpr = gl.num_warps()
    idx_layout: gl.constexpr = gl.SliceLayout(
        dim=1, parent=gl.BlockedLayout(size_per_thread=[BLOCK_M // num_warps, 1], threads_per_warp=[1, warp_size],
                                       warps_per_cta=[num_warps, 1], order=[1, 0]))
    idx = gl.load(idx_ptr + gl.arange(0, BLOCK_M, idx_layout)).to(INDEX_DTYPE)

    smem = gl.allocate_shared_memory(x_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], smem_layout)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    cols = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))
    for k in range(0, K, BLOCK_K):
        tdm.async_gather(x_desc, idx, smem)
        tdm.async_wait(0)
        mask = (k + cols < K)[None, :]
        gl.store(out_ptr + rows[:, None] * K + k + cols[None, :], smem.load(layout), mask=mask)
        x_desc = tdm.update_tensor_descriptor(x_desc, add_offsets=[0, BLOCK_K], clamp_bounds=True)


@gluon.jit
def scatter_kernel(src_ptr, idx_ptr, y_ptr, M, K, BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr,
                   INDEX_DTYPE: gl.constexpr):
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    y_desc = tdm.make_tensor_descriptor(y_ptr, [M, K], [K, 1], [BLOCK_M, BLOCK_K], smem_layout)
    num_warps: gl.constexpr = gl.num_warps()
    idx_layout: gl.constexpr = gl.SliceLayout(
        dim=1, parent=gl.BlockedLayout(size_per_thread=[BLOCK_M // num_warps, 1], threads_per_warp=[1, warp_size],
                                       warps_per_cta=[num_warps, 1], order=[1, 0]))
    idx = gl.load(idx_ptr + gl.arange(0, BLOCK_M, idx_layout)).to(INDEX_DTYPE)

    smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], smem_layout)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    cols = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))
    for k in range(0, K, BLOCK_K):
        mask = (k + cols < K)[None, :]
        smem.store(gl.load(src_ptr + rows[:, None] * K + k + cols[None, :], mask=mask, other=0))
        tdm.async_scatter(y_desc, idx, smem)
        tdm.async_wait(0)
        y_desc = tdm.update_tensor_descriptor(y_desc, add_offsets=[0, BLOCK_K], clamp_bounds=True)


@pytest.mark.parametrize("BLOCK_M, BLOCK_K", [(16, 64), (64, 128)])
@pytest.mark.parametrize("index_dtype", [gl.int16, gl.int32])
@pytest.mark.parametrize("M", [1024, 512])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_gather(BLOCK_M, BLOCK_K, index_dtype, M):
    K = 1000
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, 1024, (BLOCK_M, ), device="cuda", dtype=torch.int32).sort().values
    out = torch.empty(BLOCK_M, K, device="cuda", dtype=x.dtype)
    gather_kernel[(1, )](x, idx, out, M, K, BLOCK_M, BLOCK_K, index_dtype)
    ref = torch.zeros_like(out)
    valid = idx < M
    ref[valid] = x[idx[valid].long()]
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("BLOCK_M, BLOCK_K", [(16, 64), (64, 128)])
@pytest.mark.parametrize("index_dtype", [gl.int16, gl.int32])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_scatter(BLOCK_M, BLOCK_K, index_dtype):
    M, K = 1024, 1000
    y = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    idx = torch.randperm(M, device="cuda", dtype=torch.int32)[:BLOCK_M]
    src = torch.randn(BLOCK_M, K, device="cuda", dtype=y.dtype)
    ref = y.clone()
    ref[idx.long()] = src
    scatter_kernel[(1, )](src, idx, y, M, K, BLOCK_M, BLOCK_K, index_dtype)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


# %%
# A gather uses the same `tensor_load_to_lds` as a normal TDM load, with gather
# mode set by a bit in its descriptor. The instruction takes 4 groups of SGPRs,
# 20 SGPRs in total. The first 2 groups hold the addresses and the shape, as in
# a 2D copy. The last 2 groups hold row indices. They have only 8 SGPRs, so one
# instruction holds at most 16 int16 indices, or 16 rows. This means that in a
# kernel with 4 warps, one TDM gather instruction per warp can load at most 64
# rows. If we need to load more rows, each warp needs more than one TDM
# instruction.
#
# Apart from int16 indices, TDM gather can also use int32 indices. An SGPR holds
# 2 int16 indices or 1 int32 index, so one instruction takes 16 int16 indices or
# 8 int32 indices. So more instructions are needed when using int32 indices. In
# exchange, int32 indices can address a larger tensor.

if __name__ == "__main__":
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, 1024, (128, ), device="cuda", dtype=torch.int32)
    for num_rows in [64, 128]:
        out = torch.empty(num_rows, 1024, device="cuda", dtype=x.dtype)
        for index_dtype in [gl.int16, gl.int32]:
            print(f"{num_rows} rows, {index_dtype}:")
            kernel = gather_kernel[(1, )](x, idx, out, 1024, 1024, num_rows, 256, index_dtype)
            t1.print_instructions(kernel, r"tensor_load_to_lds")

# %%
# ```
# 64 rows, int16:
#   tensor_load_to_lds s[20:23], s[4:11], s[12:15], s[16:19]
# 64 rows, int32:
#   tensor_load_to_lds s[12:15], s[4:11], s[16:19], s[20:23]
#   tensor_load_to_lds s[12:15], s[4:11], s[24:27], s[28:31]
# 128 rows, int16:
#   tensor_load_to_lds s[28:31], s[4:11], s[12:15], s[16:19]
#   tensor_load_to_lds s[28:31], s[4:11], s[20:23], s[24:27]
# 128 rows, int32:
#   tensor_load_to_lds s[28:31], s[4:11], s[12:15], s[16:19]
#   tensor_load_to_lds s[28:31], s[4:11], s[20:23], s[24:27]
#   tensor_load_to_lds s[28:31], s[4:11], s[36:39], s[40:43]
#   tensor_load_to_lds s[28:31], s[4:11], s[44:47], s[48:51]
# Benchmarking gather
# ```
#
# As we can see, with int16 indices, a tile of 64 rows needs only 1
# `tensor_load_to_lds` per warp, while a tile of 128 rows needs 2, each with the
# next 16 indices of the warp. int32 indices double the instructions, to 2 and
# 4.

# %%
# TDM gather and scatter also support out-of-bounds accesses. As in other TDM
# copies, a gather reads zeros for rows past the end of the tensor, and a
# scatter drops them. However, the hardware only guarantees this when the
# indices do not decrease. Indices that are all in bounds can come in any order.
#
# Gather and scatter support padded shared memory layouts as well, to avoid bank
# conflicts. But they also have some limitations on shared memory. A gather
# supports only one padding interval, which must divide the width of a row, and
# a scatter only supports a padding interval of exactly one row. The number of
# indices must also be a power of 2.

# %%
# We can use standard global load to achieve similar gather-style load but TDM
# gather provides a much better performance. Let's compare them for a large
# tensor copy.


@gluon.jit
def copy_kernel(x_ptr, idx_ptr, out_ptr, M, K, BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr, INDEX_DTYPE: gl.constexpr,
                USE_TDM: gl.constexpr):
    off_m = gl.program_id(0) * BLOCK_M
    num_warps: gl.constexpr = gl.num_warps()
    if USE_TDM:
        smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        x_desc = tdm.make_tensor_descriptor(x_ptr, [M, K], [K, 1], [BLOCK_M, BLOCK_K], smem_layout)
        out_desc = tdm.make_tensor_descriptor(out_ptr, [M, K], [K, 1], [BLOCK_M, BLOCK_K], smem_layout)
        out_desc = tdm.update_tensor_descriptor(out_desc, add_offsets=[off_m, 0], clamp_bounds=True)
        idx_layout: gl.constexpr = gl.SliceLayout(
            dim=1, parent=gl.BlockedLayout(size_per_thread=[BLOCK_M // num_warps, 1], threads_per_warp=[1, warp_size],
                                           warps_per_cta=[num_warps, 1], order=[1, 0]))
        idx = gl.load(idx_ptr + off_m + gl.arange(0, BLOCK_M, idx_layout)).to(INDEX_DTYPE)
        smem = gl.allocate_shared_memory(x_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], smem_layout)
        for k in range(0, K, BLOCK_K):
            tdm.async_gather(x_desc, idx, smem)
            tdm.async_wait(0)
            tdm.async_store(out_desc, src=smem)
            tdm.async_wait(0)
            x_desc = tdm.update_tensor_descriptor(x_desc, add_offsets=[0, BLOCK_K], clamp_bounds=True)
            out_desc = tdm.update_tensor_descriptor(out_desc, add_offsets=[0, BLOCK_K], clamp_bounds=True)
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, warp_size], [num_warps, 1], [1, 0])
        rows = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
        cols = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))
        idx = gl.load(idx_ptr + rows)
        for k in range(0, K, BLOCK_K):
            x = gl.load(x_ptr + idx[:, None].to(gl.int64) * K + k + cols[None, :])
            gl.store(out_ptr + rows[:, None].to(gl.int64) * K + k + cols[None, :], x)


def copy(x, idx, use_tdm, BLOCK_M=16, BLOCK_K=2048, index_dtype=gl.int16):
    M, K = x.shape
    out = torch.empty(idx.shape[0], K, device=x.device, dtype=x.dtype)
    grid = (idx.shape[0] // BLOCK_M, )
    copy_kernel[grid](x, idx, out, M, K, BLOCK_M, BLOCK_K, index_dtype, use_tdm)
    return out


@pytest.mark.parametrize("use_tdm", [True, False])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_copy(use_tdm):
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, 4096, (2048, ), device="cuda", dtype=torch.int32)
    out = copy(x, idx, use_tdm)
    torch.testing.assert_close(out, x[idx.long()], atol=0, rtol=0)


if __name__ == "__main__":
    print("Benchmarking gather")
    print("===================")
    M, K = 16384, 8192
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, M, (M, ), device="cuda", dtype=torch.int32)

    def bench(name, fn):
        ms = triton.testing.do_bench(fn, warmup=200, rep=1000)
        print(f"{name}: {2 * x.numel() * x.element_size() / ms * 1e-9:.1f} TB/s")

    bench("global load", lambda: copy(x, idx, False))
    bench("tdm gather", lambda: copy(x, idx, True))

# %%
# ```
# global load: 12.7 TB/s
# tdm gather: 18.9 TB/s
# ```
#
# The TDM gather is 49% faster than the global load kernel.

# %%
# Now let's use gather and scatter to implement an MoE-style grouped matmul. An
# MoE layer has `E` experts, and each expert has its own weight matrix. Every
# token picks a few experts. We sort the rows by expert, so the rows of one
# expert are contiguous, and each output tile belongs to one expert. We assume
# balanced experts, with the same number of rows per expert. The weights of all
# experts are stacked into one `[E * N, K]` matrix.
#
# The two matmuls of the layer use gather and scatter. mlp1 gathers the rows of
# `A`, `Y = A[gather_idx] @ W[e].T`, and a row of `A` appears once for each
# expert that its token picks. mlp2 scatters the rows of `Y`,
# `Y[scatter_idx] = A @ W[e].T`, where `scatter_idx` is a permutation of the
# rows.
#
# Each output tile runs the K loop of the matmul in `07-mfma-wmma-scaled.py`.
# The expert of the tile decides the offset of the `W` descriptors in the
# stacked weights. With `GATHER`, the row indices of a tile are loaded once, and
# gather the `A` tile of every K block. With `SCATTER`, the output tile is
# scattered to its rows.
#
# TDM gather doesn't support partitioned layouts, so with `GATHER`, `A` uses the
# padded layout without partitions, while `W` keeps its partitioned layout. The
# scales of all experts are preshuffled as one tensor. `N` is a multiple of the
# preshuffle group, so a group never crosses two experts.
#
# The row indices use int16, so each TDM instruction gathers 16 rows, and a
# warp issues 4 TDM instructions for its part of a 256-row tile. A warp can have
# at most 11 TDM copies in flight, and the compiler adds a wait before the next
# copy when there might be more. So with `GATHER`, one fused instruction copies
# the tile of `W` with warps 0 and 1, and its scales with warps 2 and 3. This
# keeps the copies of two K blocks within the limit.


def random_experts(E, N, K):
    weights = [t7.random_mxfp4(N, K) for _ in range(E)]
    W = torch.stack([w[0] for w in weights])
    W_scale = t7.preshuffle_scales_cdna5(torch.cat([w[1] for w in weights]))
    return W, W_scale, [w[2] for w in weights]


@gluon.jit
def matmul_ld_gmem(k, pred, off_m, off_n, gather_idx, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem,
                   BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr, GATHER: gl.constexpr):
    i = k % NUM_BUFFERS
    if GATHER:
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, k * BLOCK_K], pred=pred, clamp_bounds=True)
        tdm.async_gather(a_desc, gather_idx, a_smem.index(i))
    else:
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, k * BLOCK_K], pred=pred, clamp_bounds=True)
        tdm.async_load(a_desc, dest=a_smem.index(i))
    w_desc = tdm.update_tensor_descriptor(w_desc, add_offsets=[off_n, k * BLOCK_K // 2], pred=pred, clamp_bounds=True)
    ws_desc = tdm.update_tensor_descriptor(ws_desc, add_offsets=[off_n // 128, k * BLOCK_K // 32 * 128], pred=pred,
                                           clamp_bounds=True)
    if GATHER:
        tdm.async_load_fused([(w_desc, w_smem.index(i), 0b0011), (ws_desc, ws_smem.index(i), 0b1100)])
    else:
        tdm.async_load(w_desc, dest=w_smem.index(i))
        tdm.async_load(ws_desc, dest=ws_smem.index(i))


@gluon.jit
def matmul_ld_smem(k, a_smem, w_smem, ws_smem, layouts: gl.constexpr, NUM_BUFFERS: gl.constexpr):
    a = a_smem.index(k % NUM_BUFFERS).load(layouts.a_reg_layout)
    w = w_smem.index(k % NUM_BUFFERS).permute([1, 0]).load(layouts.w_reg_layout)
    w_scale = t7.unshuffle_scales_cdna5(ws_smem.index(k % NUM_BUFFERS)).load(layouts.ws_reg_layout)
    return a, w, w_scale


@gluon.jit
def matmul_st_gmem(acc, y_ptr, scatter_idx_ptr, rows, M, N: gl.constexpr, stride_ym: gl.constexpr, off_m, off_n,
                   y_smem_layout: gl.constexpr, SCATTER: gl.constexpr):
    BLOCK_M: gl.constexpr = acc.shape[0]
    BLOCK_N: gl.constexpr = acc.shape[1]
    y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], y_smem_layout)
    y_smem.store(acc.to(y_ptr.dtype.element_ty))
    y_desc = tdm.make_tensor_descriptor(y_ptr, [M, N], [stride_ym, 1], [BLOCK_M, BLOCK_N], y_smem_layout)
    if SCATTER:
        scatter_idx = gl.load(scatter_idx_ptr + rows).to(gl.int16)
        y_desc = tdm.update_tensor_descriptor(y_desc, add_offsets=[0, off_n], clamp_bounds=True)
        tdm.async_scatter(y_desc, scatter_idx, y_smem)
    else:
        y_desc = tdm.update_tensor_descriptor(y_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
        tdm.async_store(y_desc, src=y_smem)
    tdm.async_wait(0)


@gluon.jit
def matmul_kernel(a_ptr, w_ptr, w_scale_ptr, y_ptr, gather_idx_ptr, scatter_idx_ptr, M, NUM_EXPERTS, N: gl.constexpr,
                  K: gl.constexpr, stride_am: gl.constexpr, stride_wn: gl.constexpr, stride_scale: gl.constexpr,
                  stride_ym: gl.constexpr, ROWS_PER_EXPERT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                  BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr, GATHER: gl.constexpr, SCATTER: gl.constexpr):
    gl.static_assert(ROWS_PER_EXPERT % BLOCK_M == 0)
    num_warps: gl.constexpr = gl.num_warps()
    layouts: gl.constexpr = t7.get_mx_wmma_layouts(BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    if GATHER:
        a_smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BLOCK_M, BLOCK_K], [1, 0])
    else:
        a_smem_layout: gl.constexpr = layouts.a_smem_layout
    a_desc = tdm.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BLOCK_M, BLOCK_K], a_smem_layout)
    w_desc = tdm.make_tensor_descriptor(w_ptr, [NUM_EXPERTS * N, K // 2], [stride_wn, 1], [BLOCK_N, BLOCK_K // 2],
                                        layouts.w_smem_layout)
    ws_desc = tdm.make_tensor_descriptor(w_scale_ptr, [NUM_EXPERTS * N // 128, K // 32 * 128], [stride_scale, 1],
                                         [BLOCK_N // 128, BLOCK_K // 32 * 128], layouts.ws_smem_layout)
    a_smem = gl.allocate_shared_memory(a_desc.dtype, [NUM_BUFFERS, BLOCK_M, BLOCK_K], a_desc.layout)
    w_smem = gl.allocate_shared_memory(w_desc.dtype, [NUM_BUFFERS, BLOCK_N, BLOCK_K // 2], w_desc.layout)
    ws_smem = gl.allocate_shared_memory(gl.uint8, [NUM_BUFFERS, BLOCK_N // 128, BLOCK_K // 32 * 128], ws_desc.layout)

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    off_m = pid // num_pid_n * BLOCK_M
    off_n = pid % num_pid_n * BLOCK_N
    expert = off_m // ROWS_PER_EXPERT
    off_wn = expert * N + off_n
    idx_layout: gl.constexpr = gl.SliceLayout(
        dim=1, parent=gl.BlockedLayout(size_per_thread=[BLOCK_M // num_warps, 1], threads_per_warp=[1, warp_size],
                                       warps_per_cta=[num_warps, 1], order=[1, 0]))
    rows = off_m + gl.arange(0, BLOCK_M, idx_layout)
    if GATHER:
        gather_idx = gl.load(gather_idx_ptr + rows).to(gl.int16)
    else:
        gather_idx = None

    num_k = gl.cdiv(K, BLOCK_K)
    for k in gl.static_range(NUM_BUFFERS - 1):
        matmul_ld_gmem(k, True, off_m, off_wn, gather_idx, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem, BLOCK_K,
                       NUM_BUFFERS, GATHER)
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k):
        next_k = k + NUM_BUFFERS - 1
        matmul_ld_gmem(next_k, next_k < num_k, off_m, off_wn, gather_idx, a_desc, w_desc, ws_desc, a_smem, w_smem,
                       ws_smem, BLOCK_K, NUM_BUFFERS, GATHER)
        tdm.async_wait((NUM_BUFFERS - 1) * (2 if GATHER else 3))
        a, w, w_scale = matmul_ld_smem(k, a_smem, w_smem, ws_smem, layouts, NUM_BUFFERS)
        acc = wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)

    y_smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0])
    matmul_st_gmem(acc, y_ptr, scatter_idx_ptr, rows, M, N, stride_ym, off_m, off_n, y_smem_layout, SCATTER)


def matmul(A, W, W_scale, rows_per_expert, gather_idx=None, scatter_idx=None, BLOCK_M=256, BLOCK_N=256, BLOCK_K=256,
           NUM_BUFFERS=3):
    E, N, K = W.shape[0], W.shape[1], A.shape[1]
    M = E * rows_per_expert
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (M // BLOCK_M * triton.cdiv(N, BLOCK_N), )
    matmul_kernel[grid](A, W, W_scale, Y, gather_idx, scatter_idx, M, E, N, K, A.stride(0), W.stride(1),
                        W_scale.stride(0), Y.stride(0), rows_per_expert, BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS,
                        gather_idx is not None, scatter_idx is not None)
    return Y


@pytest.mark.parametrize("E, R, N, K", [(4, 256, 512, 1024), (8, 512, 768, 512)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul_gather(E, R, N, K):
    T = 300
    A = t7.random_fp8(T, K)
    W, W_scale, W_ref = random_experts(E, N, K)
    gather_idx = torch.randint(0, T, (E, R), device="cuda", dtype=torch.int32).sort(dim=1).values.flatten()
    Y = matmul(A, W, W_scale, R, gather_idx=gather_idx)
    ref = torch.cat([A[gather_idx[e * R:(e + 1) * R].long()].float() @ W_ref[e].T for e in range(E)])
    torch.testing.assert_close(ref, Y.float(), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("E, R, N, K", [(4, 256, 512, 1024), (8, 512, 768, 512)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul_scatter(E, R, N, K):
    A = t7.random_fp8(E * R, K)
    W, W_scale, W_ref = random_experts(E, N, K)
    scatter_idx = torch.randperm(E * R, device="cuda", dtype=torch.int32)
    Y = matmul(A, W, W_scale, R, scatter_idx=scatter_idx)
    ref = torch.empty(E * R, N, device="cuda")
    ref[scatter_idx.long()] = torch.cat([A[e * R:(e + 1) * R].float() @ W_ref[e].T for e in range(E)])
    torch.testing.assert_close(ref, Y.float(), atol=1e-1, rtol=1e-2)


# %%
# We will use the following spec for the benchmark: 128 experts, 8192 tokens
# that each pick 8 experts, and a hidden dimension `H` of 7168. mlp1 multiplies
# by an `[H, 2 * H]` weight, and a SwiGLU activation would bring its output back
# to `H` columns. mlp2 multiplies by an `[H, H]` weight. So mlp1 has twice the
# work of mlp2. All row indices fit into int16.

if __name__ == "__main__":
    print("Benchmarking grouped matmul")
    print("===========================")
    E, T, TOPK, H = 128, 8192, 8, 7168
    R = T * TOPK // E
    M = E * R
    for name, (N, K) in [("mlp1", (2 * H, H)), ("mlp2", (H, H))]:
        W = torch.randint(0, 256, (E, N, K // 2), device="cuda", dtype=torch.uint8)
        W_scale = t7.preshuffle_scales_cdna5(torch.randint(120, 128, (E * N, K // 32), device="cuda",
                                                           dtype=torch.uint8))

        if name == "mlp1":
            A = t7.random_fp8(T, K)
            gather_idx = torch.arange(T, device="cuda", dtype=torch.int32).repeat_interleave(TOPK)
            gather_idx = gather_idx[torch.randperm(M, device="cuda")].view(E, R).sort(dim=1).values.flatten()
            kwargs = dict(gather_idx=gather_idx)
        else:
            A = t7.random_fp8(M, K)
            kwargs = dict(scatter_idx=torch.randperm(M, device="cuda", dtype=torch.int32))
        ms = triton.testing.do_bench(lambda: matmul(A, W, W_scale, R, **kwargs), warmup=100, rep=500)
        print(f"{name}: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

# %%
# ```
# mlp1: 6506 TFLOPS
# mlp2: 6060 TFLOPS
# ```
