"""
Multi-CTA
=========

CDNA5 GPUs can group multiple CTAs (workgroups) into a cluster (workgroup
cluster), also called a CGA in Gluon's terms. CTAs within a cluster can use a
cluster-level barrier to synchronize. Gluon also allows triggering multicast
when multiple CTAs access the same data. But different from NVIDIA's multi-CTA
support, multiple CTAs on AMD GPUs still need to share data via global memory.

In this tutorial, we will introduce the multi-CTA support of AMD CDNA5. We will
show how to use the CGA layout on top of the WMMA layout and the shared memory
layout to describe how the workload is assigned to multiple CTAs.
"""

import importlib
import math
import pytest
import sys
import torch
import triton
from pathlib import Path
from triton._C.libtriton import ir, gluon_ir
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
t5 = importlib.import_module("amd.05-mfma-wmma-scaled")


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not is_cdna5():
    raise RuntimeError("This tutorial requires an AMD CDNA5 GPU")

# A warp on CDNA5 GPUs has 32 lanes.
warp_size = gl.constexpr(32)

# %%
# In a multi-CTA kernel, the programming model levels up from the CTA to CGA.
# When we launch the kernel and ask for the program ID, Gluon gives the ID of
# the current CGA instead of the CTA. Within a CGA, we can use layouts to
# distribute a tensor over the CTAs, just like we did for warps. And similarly,
# these CTAs work on the tensor collectively.


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
# basis is zero, so both CTAs hold the whole tensor, and each thread gets one
# more register basis.
#
# We can launch a multi-CTA kernel with `num_ctas`, just like `num_warps`.
# On CDNA5, a CGA can have up to 16 CTAs, so `num_ctas` should be at most 16.
# Also considering linear layouts, it can only be a power of two.

# %%
# AMD GPUs don't have distributed shared memory like NVIDIA GPUs, so a CTA can't
# access the shared memory of another CTA. But CTAs in a cluster can still
# exchange data through global memory. To do so, these CTAs first need to write
# out their own data partition, synchronize, and then load the data owned by the
# other CTAs. This process can be implemented using different layouts for write
# and read. And the synchronization primitive is exposed as barrier primitives:
# `cluster.arrive` which signals the arrival, and `cluster.wait` which blocks
# until every CTA of the cluster has arrived.
#
# Another thing to notice is the stores and the loads will need the `"gpu"`
# scope. The scope tells the hardware which cache level must handle the access.
# With the default scope, a store completes when it reaches the cache of a WGP,
# The other CTAs of the cluster run on other WGPs, so they might not see the
# store, or they might read old data from their own cache. The barrier doesn't
# flush any cache. So we store and load with the `"gpu"` scope, which
# becomes visible to all WGPs.
#
# Let's use a sum reduction to show it. The input `x` has shape
# `[M * num_ctas, N]`, and each CTA owns one private `[M, N]` tile of it. We
# want to sum these tiles into `y` of shape `[M, N]`. Each CTA first writes out
# its tile. After the cluster barrier, each CTA reads one `[M, N / num_ctas]`
# column block of every tile, sums them, and writes this block of `y`. The write
# layout splits the rows of `x` between the CTAs, and the read layout splits the
# columns of `y`.


@gluon.jit
def sum_kernel(x_ptr, xp_ptr, y_ptr, M: gl.constexpr, N: gl.constexpr,  #
               WRITE_CGA_LAYOUT: gl.constexpr, READ_CGA_LAYOUT: gl.constexpr):
    num_ctas: gl.constexpr = gl.num_ctas()
    num_warps: gl.constexpr = gl.num_warps()

    # Each CTA holds its own [M, N] tile of x.
    write_layout: gl.constexpr = gl.BlockedLayout(  #
        size_per_thread=[1, 4], threads_per_warp=[1, warp_size], warps_per_cta=[num_warps, 1], order=[1, 0],  #
        cga_layout=WRITE_CGA_LAYOUT)
    m = gl.arange(0, M * num_ctas, gl.SliceLayout(1, write_layout))
    n = gl.arange(0, N, gl.SliceLayout(0, write_layout))
    x = gl.load(x_ptr + m[:, None] * N + n[None, :])
    xp_desc = cdna5.buffer.make_buffer_descriptor(xp_ptr, M * num_ctas * N * 4)
    cdna5.buffer.store(x, xp_desc, m[:, None] * N + n[None, :], scope="gpu")

    # Wait until every CTA has written its tile.
    gl.barrier()
    cdna5.cluster.arrive()
    cdna5.cluster.wait()

    # Each CTA holds an [M, N / num_ctas] column block of y.
    read_layout: gl.constexpr = gl.BlockedLayout(  #
        size_per_thread=[1, 4], threads_per_warp=[1, warp_size], warps_per_cta=[num_warps, 1], order=[1, 0],  #
        cga_layout=READ_CGA_LAYOUT)
    m = gl.arange(0, M, gl.SliceLayout(1, read_layout))
    n = gl.arange(0, N, gl.SliceLayout(0, read_layout))
    offs = m[:, None] * N + n[None, :]
    y = gl.zeros([M, N], gl.float32, read_layout)
    for c in gl.static_range(num_ctas):
        y += cdna5.buffer.load(xp_desc, c * M * N + offs, scope="gpu")
    gl.store(y_ptr + offs, y)


@pytest.mark.parametrize("num_ctas", [2, 4, 8])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_sum(num_ctas):
    M, N = 64, 256
    x = torch.randn(M * num_ctas, N, device="cuda")
    xp = torch.empty_like(x)
    y = torch.empty(M, N, device="cuda")

    write_cga_layout = tuple((1 << i, 0) for i in range(int(math.log2(num_ctas))))
    read_cga_layout = tuple((0, 1 << i) for i in range(int(math.log2(num_ctas))))
    sum_kernel[(1, )](x, xp, y, M, N, write_cga_layout, read_cga_layout, num_ctas=num_ctas)

    torch.testing.assert_close(y, x.view(num_ctas, M, N).sum(0), atol=1e-4, rtol=1e-4)


# %%
# Async copies and TDM copies also work in a cluster. For TDM, we can specify
# how we can distribute CTAs in the shared memory layout


@gluon.jit
def copy_kernel(x_ptr, y_ptr, M, N,  #
                BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,  #
                X_CGA_LAYOUT: gl.constexpr, Y_CGA_LAYOUT: gl.constexpr,  #
                USE_TDM: gl.constexpr):
    num_warps: gl.constexpr = gl.num_warps()
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(  #
        vec=1, per_phase=1, max_phase=1, order=[1, 0], cga_layout=X_CGA_LAYOUT)
    off_m = gl.program_id(0) * BLOCK_M
    smem = gl.allocate_shared_memory(x_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], smem_layout)

    if USE_TDM:
        x_desc = cdna5.tdm.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [BLOCK_M, BLOCK_N], smem_layout)
        x_desc = cdna5.tdm.update_tensor_descriptor(x_desc, add_offsets=[off_m, 0], clamp_bounds=True)
        cdna5.tdm.async_load(x_desc, dest=smem)
        cdna5.tdm.async_wait(0)
    else:
        # An async copy only writes into the shared memory of its own CTA, so the
        # cga_layout of the pointers must match the cga_layout of shared memory.
        layout: gl.constexpr = gl.BlockedLayout(  #
            size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[num_warps, 1], order=[1, 0],  #
            cga_layout=X_CGA_LAYOUT)

        m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
        n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
        cdna5.async_copy.global_to_shared(smem, x_ptr + m[:, None] * N + n[None, :])
        cdna5.async_copy.commit_group()
        cdna5.async_copy.wait_group(0)

    y_layout: gl.constexpr = gl.BlockedLayout(  #
        size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[num_warps, 1], order=[1, 0],  #
        cga_layout=Y_CGA_LAYOUT)
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


# %%
# When several CTAs read the same tile, they can share one load using a
# hardware feature called "multicast". Multicast works at the L2 cache level.
# When multiple CTAs request the same data, there is only one request to L2, and
# the other CTAs share the data without reading it from L2 again. So multicast
# saves the bandwidth between the shared memory of the WGPs and L2, but not the
# HBM bandwidth, because the data is cached in L2 anyway.
#
# Moreover, these requests must arrive close to each other in time. The cache
# waits for the CTAs only until a timeout, and a CTA that comes later still gets
# its own request.
#
# In Gluon, multicast is triggered when the `cga_layout` gives several CTAs the
# same data. The compiler emits multicast loads for these CTAs:
# `cluster_load_async_to_lds` for async copies, or sets the multicast bits in
# the tensor descriptor for TDM.
#
# For example, in this layout, 4 CTAs compute a WMMA as a 2x2 grid of tiles. Bit
# 0 of the CTA ID selects the row of the tile, and bit 1 selects the column:
# ```python
# acc_layout = gl.amd.AMDWMMALayout(
#     version=3, instr_shape=[16, 16, 32], transposed=True,
#     warp_bases=[], reg_bases=[], cga_layout=[[1, 0], [0, 1]])
# a_layout = gl.DotOperandLayout(0, acc_layout, 8)  # CGA: [[1, 0], [0, 0]]
# b_layout = gl.DotOperandLayout(1, acc_layout, 8)  # CGA: [[0, 0], [0, 1]]
# ```
#
# CTAs 0 and 2 share the same rows of A, and CTAs 1 and 3 share the same rows of
# A. Similarly, CTAs 0 and 1 share the same columns of B, and CTAs 2 and 3 share
# the same columns of B. So when these CTAs request A, each pair of CTAs that
# shares the same rows gets them with one load, and the same holds for B.

# %%
# Now let's apply multicast to the matmul of `05-mfma-wmma-scaled.py`. We group
# `CTAS_M x CTAS_N` CTAs into a CGA. The CTAs in the same row load the same tiles
# of `A`, and the CTAs in the same column load the same tiles of `W`.


@gluon.aggregate
class MatmulConfig:
    # Tensor shapes
    M: gl.tensor | gl.constexpr
    N: gl.tensor | gl.constexpr
    K: gl.constexpr
    # Operand formats
    A_FORMAT: gl.constexpr
    W_FORMAT: gl.constexpr
    # Tile size
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    CTA_M: gl.constexpr
    CTA_N: gl.constexpr
    # Kernel config
    NUM_BUFFERS: gl.constexpr


@gluon.aggregate
class WmmaLayouts:
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
    # Each CTA computes a CTA_M x CTA_N part of the tile of the CGA.
    CTA_M = BLOCK_M // CTAS_M
    CTA_N = BLOCK_N // CTAS_N

    cga_y = [[1 << i, 0] for i in range(int(math.log2(CTAS_M)))]
    cga_y += [[0, 1 << i] for i in range(int(math.log2(CTAS_N)))]

    # Take the layouts of one CTA from 05-mfma-wmma-scaled.py and add the cga_layout.
    cta_acc = t5.get_mx_wmma_layouts(CTA_M, CTA_N, BLOCK_K, num_warps).acc_layout.value
    acc_layout = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=cta_acc.warp_bases,
                                      reg_bases=cta_acc.reg_bases, instr_shape=[16, 16, 128], cga_layout=cga_y)
    w_parent_layout = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=cta_acc.warp_bases,
                                           reg_bases=cta_acc.reg_bases, instr_shape=[16, 16, 64], cga_layout=cga_y)
    a_reg_layout = gl.DotOperandLayout(operand_index=0, parent=acc_layout, k_width=16)
    w_reg_layout = gl.DotOperandLayout(operand_index=1, parent=w_parent_layout, k_width=16)
    cga_a = a_reg_layout.cga_layout

    # W is stored as [N, K], so we swap the dimensions of its bases.
    cga_w = [[n, k] for k, n in w_reg_layout.cga_layout]
    padded_a = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BLOCK_M, BLOCK_K], [1, 0], cga_a)
    padded_w = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BLOCK_N, BLOCK_K // 2], [1, 0], cga_w)
    a_smem_layout, w_smem_layout, _ = cdna5.make_partitioned_dot_layouts(  #
        CTA_M, CTA_N, padded_a, padded_w, num_warps, [16, 16, 128], b_transposed=True)

    ws_smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0], cga_layout=cga_w)
    ws_reg_layout = cdna5.get_wmma_scale_layout(w_reg_layout, [BLOCK_N, BLOCK_K // 32])
    y_smem_layout = gl.PaddedSharedLayout.with_identity_for([[CTA_N, 8]], [BLOCK_M, BLOCK_N], [1, 0], cga_y)

    return WmmaLayouts(a_smem_layout, a_reg_layout,  #
                       w_smem_layout, w_reg_layout,  #
                       ws_smem_layout, ws_reg_layout,  #
                       acc_layout, y_smem_layout)


@gluon.jit
def matmul_ld_gmem(cfg, k, pred, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem):
    i = k % cfg.NUM_BUFFERS
    a_desc = cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, k * cfg.BLOCK_K], pred=pred,
                                                clamp_bounds=True)
    w_desc = cdna5.tdm.update_tensor_descriptor(w_desc, add_offsets=[off_n, k * cfg.BLOCK_K // 2], pred=pred,
                                                clamp_bounds=True)
    ws_desc = cdna5.tdm.update_tensor_descriptor(ws_desc, add_offsets=[off_n // 128, k * cfg.BLOCK_K // 32 * 128],
                                                 pred=pred, clamp_bounds=True)
    cdna5.tdm.async_load(a_desc, dest=a_smem.index(i))
    cdna5.tdm.async_load(w_desc, dest=w_smem.index(i))
    cdna5.tdm.async_load(ws_desc, dest=ws_smem.index(i))


@gluon.jit
def matmul_ld_smem(cfg, layouts, k, a_smem, w_smem, ws_smem):
    a = a_smem.index(k % cfg.NUM_BUFFERS).load(layouts.a_reg_layout)
    w = w_smem.index(k % cfg.NUM_BUFFERS).permute([1, 0]).load(layouts.w_reg_layout)
    w_scale = t5.unshuffle_scales_cdna5(ws_smem.index(k % cfg.NUM_BUFFERS)).load(layouts.ws_reg_layout)
    return a, w, w_scale


@gluon.jit
def matmul_kernel(a_ptr, w_ptr, w_scale_ptr, y_ptr,  #
                  M, N, K: gl.constexpr,  #
                  CTAS_M: gl.constexpr, CTAS_N: gl.constexpr, NUM_BUFFERS: gl.constexpr = 3):
    cfg = MatmulConfig(M, N, K, A_FORMAT="e4m3", W_FORMAT="e2m1",  #
                       BLOCK_M=256 * CTAS_M, BLOCK_N=256 * CTAS_N, BLOCK_K=256, CTA_M=CTAS_M, CTA_N=CTAS_N,  #
                       NUM_BUFFERS=NUM_BUFFERS)
    layouts: gl.constexpr = get_cluster_layouts(cfg.CTA_M, cfg.CTA_N, cfg.BLOCK_M, cfg.BLOCK_N, cfg.BLOCK_K,
                                                gl.num_warps())

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(cfg.N, cfg.BLOCK_N)
    off_m = pid // num_pid_n * cfg.BLOCK_M
    off_n = pid % num_pid_n * cfg.BLOCK_N

    a_desc = cdna5.tdm.make_tensor_descriptor(a_ptr, [cfg.M, cfg.K], [cfg.K, 1], [cfg.BLOCK_M, cfg.BLOCK_K],
                                              layouts.a_smem_layout)
    w_desc = cdna5.tdm.make_tensor_descriptor(w_ptr, [cfg.N, cfg.K // 2], [cfg.K // 2, 1],
                                              [cfg.BLOCK_N, cfg.BLOCK_K // 2], layouts.w_smem_layout)
    ws_desc = cdna5.tdm.make_tensor_descriptor(w_scale_ptr, [gl.cdiv(cfg.N, 128), cfg.K // 32 * 128],
                                               [cfg.K // 32 * 128, 1], [cfg.BLOCK_N // 128, cfg.BLOCK_K // 32 * 128],
                                               layouts.ws_smem_layout)
    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [cfg.NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K],
                                       layouts.a_smem_layout)
    w_smem = gl.allocate_shared_memory(w_ptr.dtype.element_ty, [cfg.NUM_BUFFERS, cfg.BLOCK_N, cfg.BLOCK_K // 2],
                                       layouts.w_smem_layout)
    ws_smem = gl.allocate_shared_memory(gl.uint8, [cfg.NUM_BUFFERS, cfg.BLOCK_N // 128, cfg.BLOCK_K // 32 * 128],
                                        layouts.ws_smem_layout)

    num_k = gl.cdiv(cfg.K, cfg.BLOCK_K)
    for k in gl.static_range(cfg.NUM_BUFFERS - 1):
        matmul_ld_gmem(cfg, k, k < num_k, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem)

    acc = gl.zeros([cfg.BLOCK_M, cfg.BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k):
        next_k = k + cfg.NUM_BUFFERS - 1
        matmul_ld_gmem(cfg, next_k, next_k < num_k, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem)
        cdna5.tdm.async_wait((cfg.NUM_BUFFERS - 1) * 3)
        a, w, w_scale = matmul_ld_smem(cfg, layouts, k, a_smem, w_smem, ws_smem)
        acc = cdna5.wmma_scaled(a, None, cfg.A_FORMAT, w, w_scale, cfg.W_FORMAT, acc)

    y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [cfg.BLOCK_M, cfg.BLOCK_N], layouts.y_smem_layout)
    y_smem.store(acc.to(y_ptr.dtype.element_ty))
    y_desc = cdna5.tdm.make_tensor_descriptor(y_ptr, [cfg.M, cfg.N], [cfg.N, 1], [cfg.BLOCK_M, cfg.BLOCK_N],
                                              layouts.y_smem_layout)
    y_desc = cdna5.tdm.update_tensor_descriptor(y_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
    cdna5.tdm.async_store(y_desc, src=y_smem)
    cdna5.tdm.async_wait(0)


def matmul(A, W, W_scale, CTAS_M, CTAS_N):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 256 * CTAS_M) * triton.cdiv(N, 256 * CTAS_N), )
    matmul_kernel[grid](
        A, W, W_scale, Y,  #
        M, N, K,  #
        CTAS_M, CTAS_N, num_ctas=CTAS_M * CTAS_N)
    return Y


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 1024), (2048, 1024, 768)])
@pytest.mark.parametrize("CTAS_M, CTAS_N", [(1, 1), (2, 1), (1, 2), (2, 2), (2, 4), (4, 4)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul(M, N, K, CTAS_M, CTAS_N):
    A = t5.random_fp8(M, K)
    W, W_scale, W_ref = t5.random_mxfp4(N, K)
    Y = matmul(A, W, t5.preshuffle_scales_cdna5(W_scale), CTAS_M, CTAS_N)
    torch.testing.assert_close(A.float() @ W_ref.T, Y.float(), atol=1e-1, rtol=1e-2)


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

    bench("no cluster", lambda: t5.matmul_cdna5(A, W, W_scale))
    for CTAS_M, CTAS_N in [(2, 1), (1, 2), (2, 2), (2, 4), (4, 4)]:
        bench(f"{CTAS_M}x{CTAS_N} cluster", lambda: matmul(A, W, W_scale, CTAS_M, CTAS_N))

# %%
# ```
# no cluster: 7720 TFLOPS
# 2x1 cluster: 7928 TFLOPS
# 1x2 cluster: 8025 TFLOPS
# 2x2 cluster: 8155 TFLOPS
# 2x4 cluster: 8325 TFLOPS
# 4x4 cluster: 8370 TFLOPS
# ```
#
# As we can see, multicast makes the kernel faster, and the more CTAs share a
# tile, the faster the kernel gets. With 4x4 clusters, the kernel is 8% faster
# than without a cluster.
