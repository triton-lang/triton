"""
Multi-CTA
=========

CDNA5 GPUs can group CTAs into a cluster. A cluster has up to 16 CTAs, which run
at the same time on the WGPs of one shader engine. In Triton, a cluster is
called a CGA. The CTAs of a cluster can synchronize with a cluster barrier, and
share loads with multicast, where one copy from global memory writes the LDS of
several CTAs. Different from NVIDIA GPUs, a CTA can't access the LDS of another
CTA, so the CTAs exchange data through global memory.

In this tutorial, we will explain how layouts describe the CTAs of a cluster,
how to use the cluster barrier, and how copies and multicast work in a cluster.
In the end, we will apply clusters to the warp pipelined matmul of
`07-warp-pipeline.py`.
"""

import importlib
import math
import pytest
import torch
import triton
from triton._C.libtriton import ir, gluon_ir
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna5 import async_copy, buffer, cluster, tdm, wmma_scaled

t5 = importlib.import_module("05-mfma-wmma-scaled")
t7 = importlib.import_module("07-warp-pipeline")


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not is_cdna5():
    raise RuntimeError("This tutorial requires an AMD CDNA5 GPU")

# A warp on CDNA5 GPUs has 32 lanes.
warp_size = gl.constexpr(32)

# %%
# A layout maps the registers, lanes and warps of a CTA to the elements of a
# tensor, with one basis per bit of each ID (see `02-layouts.py`). A multi-CTA
# layout scales this structure up by one level. It adds the `block` dimension
# for the ID of the CTA in the cluster, and layouts take its bases as the
# `cga_layout` argument.
#
# Let's convert two blocked layouts for 2 CTAs to linear layouts and compare
# their bases.


def to_linear_layout(layout, shape):
    context = ir.context()
    ir.load_dialects(context)
    builder = gluon_ir.GluonOpBuilder(context)
    return builder.to_linear_layout(layout._to_ir(builder), shape)


if __name__ == "__main__":
    for cga_layout in [[[1, 0]], [[0, 0]]]:
        layout = gl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[4, 8], warps_per_cta=[2, 1], order=[1, 0],
                                  cga_layout=cga_layout)
        print(to_linear_layout(layout, [32, 32]))

# %%
# ```
# DistributedLinearLayout(
#     reg_bases=[[0, 1], [0, 2], [8, 0]],
#     lane_bases=[[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]],
#     warp_bases=[[4, 0]],
#     block_bases=[[16, 0]],
#     shape=[32, 32])
# DistributedLinearLayout(
#     reg_bases=[[0, 1], [0, 2], [8, 0], [16, 0]],
#     lane_bases=[[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]],
#     warp_bases=[[4, 0]],
#     block_bases=[[0, 0]],
#     shape=[32, 32])
# ```
#
# In the first layout, the block basis `[16, 0]` moves CTA 1 down by 16 rows.
# Each CTA holds one half of the tensor, just like the warp basis `[4, 0]`
# splits the rows of a CTA between the 2 warps. In the second layout, the block
# basis is zero. A zero basis means that the CTAs that differ in that bit hold
# the same data, so both CTAs hold the whole tensor, and each thread gets one
# more register basis.
#
# We launch a multi-CTA kernel with `num_ctas`, which must be a power of 2 and
# at most 16. The grid then counts clusters, and `gl.program_id` gives the index
# of the cluster.

# %%
# Because a CTA can't read the LDS of another CTA, the CTAs of a cluster
# exchange data through global memory. A CTA writes its data, all CTAs
# synchronize, and then each CTA reads the data of the others. The cluster
# barrier does the synchronization. Each CTA calls `cluster.arrive` and then
# `cluster.wait`, which blocks until every CTA of the cluster has arrived.
#
# Only one warp of each CTA signals the cluster barrier, so the warps of a CTA
# first synchronize with `gl.barrier`, which also waits for their stores. The
# data must also be visible to other WGPs, so the stores and the loads use the
# `"gpu"` scope.
#
# The kernel below sums the columns of a tensor in a cluster. Each CTA adds up
# its part of the columns for all rows and writes the partial sums. After the
# cluster barrier, each CTA adds up the partial sums of its part of the rows.
# The attention example `mxfp_fa_cdna5.py` follows the same pattern to combine
# the results of a split-K attention.


@gluon.jit
def cluster_sum_kernel(x_ptr, partial_ptr, y_ptr, M: gl.constexpr, K: gl.constexpr, X_CGA_LAYOUT: gl.constexpr,
                       Y_CGA_LAYOUT: gl.constexpr, USE_CLUSTER: gl.constexpr):
    num_ctas: gl.constexpr = gl.num_ctas()
    num_warps: gl.constexpr = gl.num_warps()
    x_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 1, 4], threads_per_warp=[1, 4, 8],
                                              warps_per_cta=[1, num_warps, 1], order=[2, 1, 0], cga_layout=X_CGA_LAYOUT)
    c = gl.arange(0, num_ctas, gl.SliceLayout(1, gl.SliceLayout(2, x_layout)))
    m = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(2, x_layout)))
    k = gl.arange(0, K // num_ctas, gl.SliceLayout(0, gl.SliceLayout(1, x_layout)))
    x = gl.load(x_ptr + m[None, :, None] * K + c[:, None, None] * (K // num_ctas) + k[None, None, :])
    partial = gl.sum(x, axis=2)

    partial_desc = buffer.make_buffer_descriptor(partial_ptr, num_ctas * M * 4)
    buffer.store(partial, partial_desc, c[:, None] * M + m[None, :], scope="gpu")
    if USE_CLUSTER:
        gl.barrier()
        cluster.arrive()
        cluster.wait()

    y_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, warp_size],
                                              warps_per_cta=[1, num_warps], order=[1, 0], cga_layout=Y_CGA_LAYOUT)
    c = gl.arange(0, num_ctas, gl.SliceLayout(1, y_layout))
    m = gl.arange(0, M, gl.SliceLayout(0, y_layout))
    partial = buffer.load(partial_desc, c[:, None] * M + m[None, :], scope="gpu")
    gl.store(y_ptr + m, gl.sum(partial, axis=0))


@pytest.mark.parametrize("num_ctas", [2, 4, 8])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_cluster_sum(num_ctas):
    M, K = 128, 256
    x = torch.randn(M, K, device="cuda")
    partial = torch.empty(num_ctas, M, device="cuda")
    y = torch.empty(M, device="cuda")
    x_cga_layout = tuple((1 << i, 0, 0) for i in range(int(math.log2(num_ctas))))
    y_cga_layout = tuple((0, 1 << i) for i in range(int(math.log2(num_ctas))))
    cluster_sum_kernel[(1, )](x, partial, y, M, K, x_cga_layout, y_cga_layout, True, num_ctas=num_ctas)
    torch.testing.assert_close(y, x.sum(1), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    M, K = 128, 256
    x = torch.randn(M, K, device="cuda")
    partial = torch.empty(4, M, device="cuda")
    y = torch.empty(M, device="cuda")
    compiled = cluster_sum_kernel[(1, )](x, partial, y, M, K, ((1, 0, 0), (2, 0, 0)), ((0, 1), (0, 2)), True,
                                         num_ctas=4)
    lines = [line.split(";")[0].strip() for line in compiled.asm["amdgcn"].splitlines()]
    lines = [line for line in lines if line and not line.startswith(".")]
    j = next(i for i, line in enumerate(lines) if "s_barrier_wait -3" in line)
    i = max(i for i, line in enumerate(lines[:j]) if "scope:SCOPE_DEV" in line)
    print("\n".join(lines[i:j + 1]))

    num_wrong = 0
    for _ in range(50):
        partial.zero_()
        cluster_sum_kernel[(1, )](x, partial, y, M, K, ((1, 0, 0), (2, 0, 0)), ((0, 1), (0, 2)), False, num_ctas=4)
        num_wrong += not torch.allclose(y, x.sum(1), atol=1e-3, rtol=1e-3)
    print(f"wrong results without the cluster barrier: {num_wrong} / 50")

# %%
# ```
# buffer_store_b32 v6, v0, s[4:7], null offen scope:SCOPE_DEV
# s_wait_storecnt 0x0
# s_barrier_signal -1
# s_barrier_wait -1
# s_cbranch_scc1 .LBB0_2
# s_barrier_signal -3
# s_barrier_wait -3
# wrong results without the cluster barrier: 49 / 50
# ```
#
# The store uses the device scope. `gl.barrier` waits for the stores and
# synchronizes the warps of the CTA with the workgroup barrier, which has ID -1.
# Then warp 0 signals the cluster barrier, which has ID -3, and all warps wait
# on it. Without the cluster barrier, a CTA can read the partial sums before the
# other CTAs have written them, and most results are wrong.

# %%
# Async copies and TDM copies also work in a cluster. The shared memory layout
# takes a `cga_layout` too, and it tells each CTA which part of the tile to
# copy. If the bases are not zero, the CTAs copy different parts.
#
# The kernel below copies tiles with 4 CTAs per cluster, either with TDM or with
# an async copy. Each CTA copies one quarter of the rows of a tile into its LDS,
# and then stores its rows back to global memory.


@gluon.jit
def copy_kernel(x_ptr, y_ptr, M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, X_CGA_LAYOUT: gl.constexpr,
                Y_CGA_LAYOUT: gl.constexpr, USE_TDM: gl.constexpr):
    num_warps: gl.constexpr = gl.num_warps()
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0],
                                                        cga_layout=X_CGA_LAYOUT)
    off_m = gl.program_id(0) * BLOCK_M
    smem = gl.allocate_shared_memory(x_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], smem_layout)
    if USE_TDM:
        x_desc = tdm.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [BLOCK_M, BLOCK_N], smem_layout)
        tdm.async_load(x_desc, [off_m, 0], smem)
        tdm.async_wait(0)
    else:
        layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8],
                                                warps_per_cta=[num_warps, 1], order=[1, 0], cga_layout=X_CGA_LAYOUT)
        m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
        n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
        async_copy.global_to_shared(smem, x_ptr + m[:, None] * N + n[None, :])
        async_copy.commit_group()
        async_copy.wait_group(0)

    y_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8],
                                              warps_per_cta=[num_warps, 1], order=[1, 0], cga_layout=Y_CGA_LAYOUT)
    m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, y_layout))
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, y_layout))
    gl.store(y_ptr + m[:, None] * N + n[None, :], smem.load(y_layout))


@pytest.mark.parametrize("x_cga_layout", [((1, 0), ), ((1, 0), (2, 0)), ((0, 0), ), ((0, 0), (0, 0))])
@pytest.mark.parametrize("use_tdm", [True, False])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_copy(x_cga_layout, use_tdm):
    M, N = 1024, 64
    x = torch.randn(M, N, device="cuda")
    y = torch.empty_like(x)
    y_cga_layout = tuple((1 << i, 0) for i in range(len(x_cga_layout)))
    copy_kernel[(M // 128, )](x, y, M, N, 128, N, x_cga_layout, y_cga_layout, use_tdm, num_ctas=2**len(x_cga_layout))
    torch.testing.assert_close(y, x, atol=0, rtol=0)


if __name__ == "__main__":
    M, N = 1024, 64
    x = torch.randn(M, N, device="cuda")
    y = torch.empty_like(x)
    for use_tdm in [True, False]:
        compiled = copy_kernel[(M // 128, )](x, y, M, N, 128, N, ((1, 0), (2, 0)), ((1, 0), (2, 0)), use_tdm,
                                             num_ctas=4)
        assert torch.equal(x, y)
        asm = compiled.asm["amdgcn"]
        ops = ["tensor_load_to_lds", "global_load_async", "cluster_load_async"]
        print(f"use_tdm={use_tdm}: {({op: asm.count(op) for op in ops if op in asm})}")

# %%
# ```
# use_tdm=True: {'tensor_load_to_lds': 1}
# use_tdm=False: {'global_load_async': 4}
# ```
#
# With TDM, each CTA issues one copy of its 32 rows. With the async copy, each
# thread issues 4 copies for its part of the 32 rows.

# %%
# When several CTAs need the same tile, they can share one load with multicast.
# Each CTA issues the same load with a mask of the CTAs in the cluster that need
# the data. When all of them have issued the load, the data is read once and
# written to the LDS of each CTA. Multicast loads bypass the cache of the WGP.
#
# We ask for multicast with a zero basis in the `cga_layout`, which means that
# the CTAs that differ in that bit hold the same data. The compiler finds these
# CTAs and sets the mask, so there is no multicast argument in the code. The
# async copy then uses `cluster_load_async` instead of `global_load_async`.
# TDM keeps the same instruction, and it sets the mask in the descriptor.
#
# A mask can name at most 4 CTAs. If more CTAs share a tile, the compiler
# splits them into groups of 4, and each group loads the tile once.
#
# The kernel above shows multicast with zero bases. Each CTA still stores one
# quarter of the rows, but now the rows come from the tile it received with
# multicast.

if __name__ == "__main__":
    for use_tdm in [True, False]:
        y = torch.empty_like(x)
        compiled = copy_kernel[(M // 128, )](x, y, M, N, 128, N, ((0, 0), (0, 0)), ((1, 0), (2, 0)), use_tdm,
                                             num_ctas=4)
        assert torch.equal(x, y)
        asm = compiled.asm["amdgcn"]
        ops = ["tensor_load_to_lds", "global_load_async", "cluster_load_async"]
        print(f"use_tdm={use_tdm}: {({op: asm.count(op) for op in ops if op in asm})}")

# %%
# ```
# use_tdm=True: {'tensor_load_to_lds': 1}
# use_tdm=False: {'cluster_load_async': 16}
# ```
#
# Now each CTA loads the whole tile of 128 rows. The async copy needs 16
# multicast copies per thread, 4 times as many as before.

# %%
# Now let's apply multicast to the warp pipelined matmul of
# `07-warp-pipeline.py`. The output tiles
# in the same row load the same tiles of `A`, and the tiles in the same column
# load the same tiles of `W`. We group `CTAS_M x CTAS_N` CTAs into a cluster.
# The cluster computes one tile of `Y`, and each CTA computes its own 256 x 256
# part of it.
#
# For a 2x2 cluster, the accumulator uses:
#
# ```
# cga_y = [[1, 0], [0, 1]]
# ```
#
# CTA ID bit 0 selects the row of the part, and bit 1 selects the column.
# `DotOperandLayout` derives the layouts of the operands. For `A`, it sets the
# column basis to zero, which gives `[[1, 0], [0, 0]]`. So the 2 CTAs of a row
# share their `A` tiles with multicast.
#
# The shared memory layouts use the same `cga_layout` as the operands. `W` is
# stored as `[N, K]`, so we swap the dimensions of its bases. The scales of `W`
# use the `cga_layout` of `W`, so the CTAs that share a tile of `W` also share
# its scales. We take the partitioned layouts of one CTA from
# `05-mfma-wmma-scaled.py` and add the `cga_layout`.


@gluon.aggregate
class ClusterLayouts:
    # Operand A's shared memory and register layouts
    a_smem_layout: gl.constexpr
    a_reg_layout: gl.constexpr
    # Operand W's shared memory and register layouts
    w_smem_layout: gl.constexpr
    w_reg_layout: gl.constexpr
    # The scales' shared memory and register layouts
    ws_smem_layout: gl.constexpr
    ws_reg_layout: gl.constexpr
    # Accumulator's register layout and output's shared memory layout
    acc_layout: gl.constexpr
    y_smem_layout: gl.constexpr


@gluon.constexpr_function
def get_cluster_layouts(CTAS_M, CTAS_N, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    CTA_M = BLOCK_M // CTAS_M
    CTA_N = BLOCK_N // CTAS_N
    cga_y = [[1 << i, 0] for i in range(int(math.log2(CTAS_M)))]
    cga_y += [[0, 1 << i] for i in range(int(math.log2(CTAS_N)))]
    cta_acc = t5.get_mx_wmma_layouts(CTA_M, CTA_N, BLOCK_K, num_warps).acc_layout.value
    acc_layout = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=cta_acc.warp_bases,
                                      reg_bases=cta_acc.reg_bases, instr_shape=[16, 16, 128], cga_layout=cga_y)
    w_parent_layout = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=cta_acc.warp_bases,
                                           reg_bases=cta_acc.reg_bases, instr_shape=[16, 16, 64], cga_layout=cga_y)
    a_reg_layout = gl.DotOperandLayout(operand_index=0, parent=acc_layout, k_width=16)
    w_reg_layout = gl.DotOperandLayout(operand_index=1, parent=w_parent_layout, k_width=16)
    cga_a = a_reg_layout.cga_layout
    cga_w = [[n, k] for k, n in w_reg_layout.cga_layout]
    padded_a = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BLOCK_M, BLOCK_K], [1, 0], cga_a)
    padded_w = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BLOCK_N, BLOCK_K // 2], [1, 0], cga_w)
    a_smem_layout, w_smem_layout, _ = gl.amd.cdna5.make_partitioned_dot_layouts(  #
        CTA_M, CTA_N, padded_a, padded_w, num_warps, [16, 16, 128], b_transposed=True)
    ws_smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0], cga_layout=cga_w)
    ws_reg_layout = gl.amd.cdna5.get_wmma_scale_layout(w_reg_layout, [BLOCK_N, BLOCK_K // 32])
    y_smem_layout = gl.PaddedSharedLayout.with_identity_for([[CTA_N, 8]], [BLOCK_M, BLOCK_N], [1, 0], cga_y)
    return ClusterLayouts(a_smem_layout, a_reg_layout,  #
                          w_smem_layout, w_reg_layout,  #
                          ws_smem_layout, ws_reg_layout,  #
                          acc_layout, y_smem_layout)


# %%
# The kernel is the warp pipelined matmul of `07-warp-pipeline.py` with the
# cluster layouts. The grid counts clusters, and each cluster computes a tile of
# `CTAS_M * 256 x CTAS_N * 256`.


@gluon.jit
def matmul_ld_gmem(k, pred, slot: gl.constexpr, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem,
                   BLOCK_K: gl.constexpr):
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, k * BLOCK_K], pred=pred, clamp_bounds=True)
    w_desc = tdm.update_tensor_descriptor(w_desc, add_offsets=[off_n, k * BLOCK_K // 2], pred=pred, clamp_bounds=True)
    ws_desc = tdm.update_tensor_descriptor(ws_desc, add_offsets=[off_n // 128, k * BLOCK_K // 32 * 128], pred=pred,
                                           clamp_bounds=True)
    tdm.async_load(a_desc, dest=a_smem.index(slot), warp_used_hint=0b1111)
    tdm.async_load(w_desc, dest=w_smem.index(slot), warp_used_hint=0b1111)
    tdm.async_load(ws_desc, dest=ws_smem.index(slot), warp_used_hint=0b1111)


@gluon.jit
def matmul_ld_smem(slot: gl.constexpr, half: gl.constexpr, a_smem, w_smem, ws_smem, layouts: gl.constexpr):
    HALF_K: gl.constexpr = a_smem.shape[2] // 2
    ws_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(layouts.w_reg_layout, [w_smem.shape[1], HALF_K // 32])
    a = a_smem.index(slot).slice(half * HALF_K, HALF_K, 1).load(layouts.a_reg_layout)
    w = w_smem.index(slot).slice(half * HALF_K // 2, HALF_K // 2, 1).permute([1, 0]).load(layouts.w_reg_layout)
    w_scale = t5.unshuffle_scales_cdna5(ws_smem.index(slot)).slice(half * HALF_K // 32, HALF_K // 32, 1).load(ws_layout)
    return a, w, w_scale


@gluon.jit
def matmul_kernel(a_ptr, w_ptr, w_scale_ptr, y_ptr,  #
                  M, N, K: gl.constexpr,  #
                  stride_am, stride_wn, stride_scale, stride_ym,  #
                  CTAS_M: gl.constexpr, CTAS_N: gl.constexpr, BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr):
    gl.static_assert(K % BLOCK_K == 0)
    BLOCK_M: gl.constexpr = 256 * CTAS_M
    BLOCK_N: gl.constexpr = 256 * CTAS_N
    layouts: gl.constexpr = get_cluster_layouts(CTAS_M, CTAS_N, BLOCK_M, BLOCK_N, BLOCK_K, gl.num_warps())

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    off_m = pid // num_pid_n * BLOCK_M
    off_n = pid % num_pid_n * BLOCK_N

    a_desc = tdm.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    w_desc = tdm.make_tensor_descriptor(w_ptr, [N, K // 2], [stride_wn, 1], [BLOCK_N, BLOCK_K // 2],
                                        layouts.w_smem_layout)
    ws_desc = tdm.make_tensor_descriptor(w_scale_ptr, [gl.cdiv(N, 128), K // 32 * 128], [stride_scale, 1],
                                         [BLOCK_N // 128, BLOCK_K // 32 * 128], layouts.ws_smem_layout)
    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    w_smem = gl.allocate_shared_memory(w_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K // 2],
                                       layouts.w_smem_layout)
    ws_smem = gl.allocate_shared_memory(gl.uint8, [NUM_BUFFERS, BLOCK_N // 128, BLOCK_K // 32 * 128],
                                        layouts.ws_smem_layout)

    num_k: gl.constexpr = K // BLOCK_K
    TAIL: gl.constexpr = num_k % NUM_BUFFERS
    for i in gl.static_range(NUM_BUFFERS):
        matmul_ld_gmem(i, i < num_k, i, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem, BLOCK_K)
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(0, num_k - TAIL, NUM_BUFFERS):
        for i in gl.static_range(NUM_BUFFERS):
            tdm.async_wait((NUM_BUFFERS - 1) * 3)
            with gl.amd.warp_pipeline_stage("lds_low", priority=1):
                a, w, w_scale = matmul_ld_smem(i, 0, a_smem, w_smem, ws_smem, layouts)
            with gl.amd.warp_pipeline_stage("wmma_low", priority=0):
                acc = wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)
            with gl.amd.warp_pipeline_stage("lds_high", priority=1):
                a, w, w_scale = matmul_ld_smem(i, 1, a_smem, w_smem, ws_smem, layouts)
            with gl.amd.warp_pipeline_stage("wmma_high", priority=0):
                acc = wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)
            with gl.amd.warp_pipeline_stage("refill", priority=1):
                next_k = k + i + NUM_BUFFERS
                matmul_ld_gmem(next_k, next_k < num_k, i, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem,
                               ws_smem, BLOCK_K)
    for i in gl.static_range(TAIL):
        tdm.async_wait((TAIL - 1 - i) * 3)
        for half in gl.static_range(2):
            with gl.amd.warp_pipeline_stage("lds", priority=1):
                a, w, w_scale = matmul_ld_smem(i, half, a_smem, w_smem, ws_smem, layouts)
            with gl.amd.warp_pipeline_stage("wmma", priority=0):
                acc = wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)

    y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], layouts.y_smem_layout)
    y_smem.store(acc.to(y_ptr.dtype.element_ty))
    y_desc = tdm.make_tensor_descriptor(y_ptr, [M, N], [stride_ym, 1], [BLOCK_M, BLOCK_N], layouts.y_smem_layout)
    y_desc = tdm.update_tensor_descriptor(y_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
    tdm.async_store(y_desc, src=y_smem)
    tdm.async_wait(0)


def matmul(A, W, W_scale, CTAS_M, CTAS_N, BLOCK_K=256, NUM_BUFFERS=3):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 256 * CTAS_M) * triton.cdiv(N, 256 * CTAS_N), )
    matmul_kernel[grid](
        A, W, W_scale, Y,  #
        M, N, K,  #
        A.stride(0), W.stride(0), W_scale.stride(0), Y.stride(0),  #
        CTAS_M, CTAS_N, BLOCK_K, NUM_BUFFERS, num_warps=8, num_ctas=CTAS_M * CTAS_N)
    return Y


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 1024), (2048, 1024, 768)])
@pytest.mark.parametrize("CTAS_M, CTAS_N", [(1, 1), (2, 1), (1, 2), (2, 2), (2, 4)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul(M, N, K, CTAS_M, CTAS_N):
    A = t5.random_fp8(M, K)
    W, W_scale, W_ref = t5.random_mxfp4(N, K)
    Y = matmul(A, W, t5.preshuffle_scales_cdna5(W_scale), CTAS_M, CTAS_N)
    torch.testing.assert_close(A.float() @ W_ref.T, Y.float(), atol=1e-1, rtol=1e-2)


# %%
# We compare the kernel of `07-warp-pipeline.py` with clusters of different
# shapes on an MI455X GPU with M = N = K = 8192.

if __name__ == "__main__":
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 8192, 8192, 8192
    A = t5.random_fp8(M, K)
    W, W_scale, _ = t5.random_mxfp4(N, K)
    W_scale = t5.preshuffle_scales_cdna5(W_scale)

    def bench(name, fn):
        ms = triton.testing.do_bench(fn, warmup=100, rep=500)
        print(f"{name}: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

    bench("one CTA per tile", lambda: t7.matmul(A, W, W_scale))
    for CTAS_M, CTAS_N in [(2, 1), (1, 2), (2, 2), (2, 4), (4, 4)]:
        bench(f"{CTAS_M}x{CTAS_N} cluster", lambda: matmul(A, W, W_scale, CTAS_M, CTAS_N))

# %%
# ```
# one CTA per tile: 7770 TFLOPS
# 2x1 cluster: 7986 TFLOPS
# 1x2 cluster: 8159 TFLOPS
# 2x2 cluster: 8257 TFLOPS
# 2x4 cluster: 8358 TFLOPS
# 4x4 cluster: 8447 TFLOPS
# ```
