"""
Multi-CTA
=========

CDNA5 GPUs can group multiple CTAs (workgroups) into a cluster (workgroup
cluster), also called a CGA in Gluon's terms. CTAs within a cluster can use a
cluster-level barrier to synchronize. Gluon also allows triggering multicast
when multiple CTAs access the same data. But different from NVIDIA's multi-CTA
support, multiple CTAs on AMD GPUs still need to share data via global memory.

In this tutorial, we will showcase the multi-CTA support of AMD CDNA5. We will
show how to use the CGA layout on top of the WMMA layout and the shared memory
layout to describe how the workload is assigned to multiple CTAs.
"""

import importlib
import math
import pytest
import torch
import triton
from triton._C.libtriton import ir, gluon_ir
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna5

t5 = importlib.import_module("amd.05-mfma-wmma-scaled")


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not is_cdna5():
    raise RuntimeError("This tutorial requires an AMD CDNA5 GPU")

# A warp on CDNA5 GPUs has 32 lanes.
warp_size = gl.constexpr(32)

# %%
# In a multi-CTA kernel, the programming level goes up from the CTA to the CGA.
# When we launch the kernel and ask for the program ID, Gluon gives the ID of
# the current CGA instead of the CTA. Within a CGA, we can use layouts to
# distribute a tensor over the CTAs, just like we did for warps. And similarly,
# the CTAs work on the tensor collectively.
#
# A layout maps the registers, lanes and warps of a CTA to the elements of a
# tensor, with one basis per bit of each ID (see `02-layouts.py`). A multi-CTA
# layout adds the `block` dimension for the ID of the CTA in the CGA, and
# layouts take its bases as the `cga_layout` argument.
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
# basis is zero, so both CTAs hold the whole tensor, and each thread gets one
# more register basis.
#
# A CGA has up to 16 CTAs. We launch a multi-CTA kernel with `num_ctas`, which
# must be a power of 2 and at most 16. The grid then counts CGAs, and
# `gl.program_id` gives the ID of the CGA.

# %%
# AMD GPUs don't have distributed shared memory like NVIDIA GPUs, so a CTA can't
# access the shared memory of another CTA. The CTAs of a cluster exchange data
# through global memory instead. To do so, the CTAs first write out their own
# data, synchronize, and then load the data owned by the other CTAs. This
# process can be implemented using different layouts for the write and the read.
# The synchronization primitive is exposed as a split barrier: `cluster.arrive`
# signals the arrival, and `cluster.wait` blocks until every CTA of the cluster
# has arrived.
#
# `cluster.arrive` lowers to `s_barrier_signal -3`, and `cluster.wait` lowers to
# `s_barrier_wait -3`. Barrier -3 is the cluster barrier, and barrier -1 is the
# workgroup barrier of a CTA. The cluster barrier counts CTAs, not warps. It
# completes when every CTA of the cluster has signaled once, and then it
# releases all warps of the cluster. So only warp 0 of each CTA signals the
# cluster barrier, and all warps wait on it. Before that, the warps of a CTA
# meet at the workgroup barrier with `gl.barrier`, so that warp 0 signals only
# after all warps of the CTA have written their data. `gl.barrier` also waits
# for the stores of each warp.
#
# The stores and the loads also need the `"gpu"` scope. The scope tells the
# hardware which cache level must handle the access. With the default scope, a
# store completes when it reaches the cache of the WGP, and a load can hit in
# that cache. The other CTAs of the cluster run on other WGPs, so they might not
# see the store, or they might read old data from their own cache. The barrier
# doesn't flush any cache. So we store and load with the `"gpu"` scope, which
# goes to a cache that all WGPs share.
#
# Let's use a simple sum reduction to show it. Each CTA sums one part of a
# vector and writes its partial sum to global memory. After the cluster barrier,
# each CTA loads all partial sums and adds them up. The write layout gives each
# CTA one partial sum, and the read layout gives every CTA all of them.


@gluon.jit
def sum_kernel(x_ptr, partial_ptr, y_ptr, N: gl.constexpr, WRITE_CGA_LAYOUT: gl.constexpr,
               READ_CGA_LAYOUT: gl.constexpr):
    num_ctas: gl.constexpr = gl.num_ctas()
    # Each CTA holds one row of x and sums it.
    x_layout: gl.constexpr = gl.BlockedLayout(  #
        size_per_thread=[1, 4], threads_per_warp=[1, warp_size], warps_per_cta=[1, gl.num_warps()], order=[1, 0],
        cga_layout=WRITE_CGA_LAYOUT)
    c = gl.arange(0, num_ctas, gl.SliceLayout(1, x_layout))
    n = gl.arange(0, N // num_ctas, gl.SliceLayout(0, x_layout))
    x = gl.load(x_ptr + c[:, None] * (N // num_ctas) + n[None, :])
    partial = gl.sum(x, axis=1)
    partial_desc = cdna5.buffer.make_buffer_descriptor(partial_ptr, num_ctas * 4)
    cdna5.buffer.store(partial, partial_desc, c, scope="gpu")

    # Wait until every CTA has written its partial sum.
    gl.barrier()
    cdna5.cluster.arrive()
    cdna5.cluster.wait()

    # Every CTA holds all partial sums.
    read_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[warp_size],
                                                 warps_per_cta=[gl.num_warps()], order=[0], cga_layout=READ_CGA_LAYOUT)
    partial = cdna5.buffer.load(partial_desc, gl.arange(0, num_ctas, read_layout), scope="gpu")
    gl.store(y_ptr, gl.sum(partial, axis=0))


@pytest.mark.parametrize("num_ctas", [2, 4, 8])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_sum(num_ctas):
    N = 4096
    x = torch.randn(N, device="cuda")
    partial = torch.empty(num_ctas, device="cuda")
    y = torch.empty(1, device="cuda")
    write_cga_layout = tuple((1 << i, 0) for i in range(int(math.log2(num_ctas))))
    read_cga_layout = tuple((0, ) for i in range(int(math.log2(num_ctas))))
    sum_kernel[(1, )](x, partial, y, N, write_cga_layout, read_cga_layout, num_ctas=num_ctas)
    torch.testing.assert_close(y, x.sum(0, keepdim=True), atol=1e-3, rtol=1e-3)


# %%
# Async copies and TDM copies also work in a cluster. The shared memory layout
# takes a `cga_layout` too, and it tells each CTA which part of the tile to
# copy.
#
# The kernel below copies tiles with 4 CTAs per cluster, either with TDM or with
# an async copy. Each CTA copies one quarter of the rows of a tile into its
# shared memory, and then stores its rows back to global memory.


@gluon.jit
def copy_kernel(x_ptr, y_ptr, M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, X_CGA_LAYOUT: gl.constexpr,
                Y_CGA_LAYOUT: gl.constexpr, USE_TDM: gl.constexpr):
    num_warps: gl.constexpr = gl.num_warps()
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0],
                                                        cga_layout=X_CGA_LAYOUT)
    off_m = gl.program_id(0) * BLOCK_M
    smem = gl.allocate_shared_memory(x_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], smem_layout)
    if USE_TDM:
        x_desc = cdna5.tdm.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [BLOCK_M, BLOCK_N], smem_layout)
        x_desc = cdna5.tdm.update_tensor_descriptor(x_desc, add_offsets=[off_m, 0], clamp_bounds=True)
        cdna5.tdm.async_load(x_desc, dest=smem)
        cdna5.tdm.async_wait(0)
    else:
        layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8],
                                                warps_per_cta=[num_warps, 1], order=[1, 0], cga_layout=X_CGA_LAYOUT)
        m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
        n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
        cdna5.async_copy.global_to_shared(smem, x_ptr + m[:, None] * N + n[None, :])
        cdna5.async_copy.commit_group()
        cdna5.async_copy.wait_group(0)

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


# %%
# When several CTAs need the same tile, they can share one load with multicast.
# Each CTA issues the same load with a mask of the CTAs in the cluster that need
# the data. When all of them have issued the load, the data is read once and
# written to the shared memory of each CTA. A mask can name at most 4 CTAs. If
# more CTAs share a tile, the compiler splits them into groups of 4, and each
# group loads the tile once.
#
# We can implement multicast using a specific `cga_layout`. When the layout
# gives several CTAs the same data, the compiler finds these CTAs and sets the
# mask. TDM and async copies work the same way, while the async copy is lowered
# to `cluster_load_async` instead of `global_load_async`, and in TDM, a field of
# the tensor descriptor holds the mask.

# %%
# Now let's apply multicast to the matmul of `05-mfma-wmma-scaled.py`. We group
# `CTAS_M x CTAS_N` CTAs into a CGA, and the CGA computes one tile of `Y`. The
# CTAs in the same row load the same tiles of `A`, and the CTAs in the same
# column load the same tiles of `W`. With multicast, the CGA reads each of these
# tiles once, so it reads fewer bytes from global memory for the same work.


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
    # Each CTA computes a CTA_M x CTA_N part of the tile of the CGA.
    CTA_M = BLOCK_M // CTAS_M
    CTA_N = BLOCK_N // CTAS_N
    # The low bits of the CTA ID select the row of the part, and the high bits
    # select the column.
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
    return ClusterLayouts(a_smem_layout, a_reg_layout,  #
                          w_smem_layout, w_reg_layout,  #
                          ws_smem_layout, ws_reg_layout,  #
                          acc_layout, y_smem_layout)


@gluon.jit
def matmul_kernel(a_ptr, w_ptr, w_scale_ptr, y_ptr,  #
                  M, N, K,  #
                  stride_am, stride_wn, stride_scale, stride_ym,  #
                  CTAS_M: gl.constexpr, CTAS_N: gl.constexpr, NUM_BUFFERS: gl.constexpr):
    BLOCK_M: gl.constexpr = 256 * CTAS_M
    BLOCK_N: gl.constexpr = 256 * CTAS_N
    BLOCK_K: gl.constexpr = 256
    layouts: gl.constexpr = get_cluster_layouts(CTAS_M, CTAS_N, BLOCK_M, BLOCK_N, BLOCK_K, gl.num_warps())

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    off_m = pid // num_pid_n * BLOCK_M
    off_n = pid % num_pid_n * BLOCK_N

    a_desc = cdna5.tdm.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    w_desc = cdna5.tdm.make_tensor_descriptor(w_ptr, [N, K // 2], [stride_wn, 1], [BLOCK_N, BLOCK_K // 2],
                                              layouts.w_smem_layout)
    ws_desc = cdna5.tdm.make_tensor_descriptor(w_scale_ptr, [gl.cdiv(N, 128), K // 32 * 128], [stride_scale, 1],
                                               [BLOCK_N // 128, BLOCK_K // 32 * 128], layouts.ws_smem_layout)
    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    w_smem = gl.allocate_shared_memory(w_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K // 2],
                                       layouts.w_smem_layout)
    ws_smem = gl.allocate_shared_memory(gl.uint8, [NUM_BUFFERS, BLOCK_N // 128, BLOCK_K // 32 * 128],
                                        layouts.ws_smem_layout)

    # The K loop of `05-mfma-wmma-scaled.py`, with the cluster layouts.
    num_k = gl.cdiv(K, BLOCK_K)
    for k in gl.static_range(NUM_BUFFERS - 1):
        t5.matmul_cdna5_ld_gmem(k, k < num_k, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem, BLOCK_K,
                                NUM_BUFFERS, 128)
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k):
        next_k = k + NUM_BUFFERS - 1
        t5.matmul_cdna5_ld_gmem(next_k, next_k < num_k, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem,
                                BLOCK_K, NUM_BUFFERS, 128)
        cdna5.tdm.async_wait((NUM_BUFFERS - 1) * 3)
        a, w, w_scale = t5.matmul_cdna5_ld_smem(k, a_smem, w_smem, ws_smem, layouts, NUM_BUFFERS, True)
        acc = cdna5.wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)

    y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], layouts.y_smem_layout)
    y_smem.store(acc.to(y_ptr.dtype.element_ty))
    y_desc = cdna5.tdm.make_tensor_descriptor(y_ptr, [M, N], [stride_ym, 1], [BLOCK_M, BLOCK_N], layouts.y_smem_layout)
    y_desc = cdna5.tdm.update_tensor_descriptor(y_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
    cdna5.tdm.async_store(y_desc, src=y_smem)
    cdna5.tdm.async_wait(0)


def matmul(A, W, W_scale, CTAS_M, CTAS_N, NUM_BUFFERS=3):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 256 * CTAS_M) * triton.cdiv(N, 256 * CTAS_N), )
    matmul_kernel[grid](
        A, W, W_scale, Y,  #
        M, N, K,  #
        A.stride(0), W.stride(0), W_scale.stride(0), Y.stride(0),  #
        CTAS_M, CTAS_N, NUM_BUFFERS, num_ctas=CTAS_M * CTAS_N)
    return Y


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 1024), (2048, 1024, 768)])
@pytest.mark.parametrize("CTAS_M, CTAS_N", [(1, 1), (2, 1), (1, 2), (2, 2), (2, 4), (4, 4)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul(M, N, K, CTAS_M, CTAS_N):
    A = t5.random_fp8(M, K)
    W, W_scale, W_ref = t5.random_mxfp4(N, K)
    Y = matmul(A, W, t5.preshuffle_scales_cdna5(W_scale), CTAS_M, CTAS_N)
    torch.testing.assert_close(A.float() @ W_ref.T, Y.float(), atol=1e-1, rtol=1e-2)


# %%
# We can compare the kernel of `05-mfma-wmma-scaled.py` with clusters of
# different shapes with M = N = K = 8192.

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
# no cluster: 7727 TFLOPS
# 2x1 cluster: 7954 TFLOPS
# 1x2 cluster: 8020 TFLOPS
# 2x2 cluster: 8180 TFLOPS
# 2x4 cluster: 8248 TFLOPS
# 4x4 cluster: 8337 TFLOPS
# ```
