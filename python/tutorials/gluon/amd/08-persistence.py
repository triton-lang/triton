"""
Persistence
===========

So far, our kernels launch one CTA for each output tile. A persistent kernel
launches only as many CTAs as the GPU can run at the same time, and each CTA
loops over many tiles. This saves the setup of a new CTA for every tile, and the
kernel decides which tiles each CTA computes and in which order.

In this tutorial, we will explain how a persistent kernel schedules its tiles,
and how to order the tiles for the chiplets of a CDNA5 GPU. In the end, we will
make the warp pipelined matmul of `07-warp-pipeline.py` persistent.
"""

import importlib
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna5 import tdm, wmma_scaled

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
# The matmul of `07-warp-pipeline.py` uses almost all of the LDS of a WGP, so
# only one CTA fits on a WGP. An MI455X GPU has 256 WGPs, so only 256 CTAs run
# at the same time, out of 1024 for an 8192 x 8192 output.
#
# A persistent version launches 256 CTAs. Each CTA loops over the output tiles
# and takes every `P`-th tile, where `P` is the number of CTAs, so the outer
# loop takes over the role of the grid. The following kernel records which CTA
# computes each of 64 tiles, when 16 CTAs run at the same time.


@gluon.jit
def schedule_kernel(cta_ptr, NUM_TILES: gl.constexpr, XCD_REMAP: gl.constexpr):
    cta = gl.program_id(0)
    num_programs = gl.num_programs(0)
    pid = cta
    if XCD_REMAP:
        pid = cta % 8 * (num_programs // 8) + cta // 8
    for t in range(pid, NUM_TILES, num_programs):
        gl.store(cta_ptr + t, cta)


if __name__ == "__main__":
    cta = torch.empty(64, device="cuda", dtype=torch.int32)
    schedule_kernel[(16, )](cta, 64, False)
    for step in cta.view(-1, 16).tolist():
        print(" ".join(f"{x:2}" for x in step))

# %%
# ```
#  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
#  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
#  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
#  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
# ```
#
# Each line shows the CTAs of 16 tiles that run at the same time. CTA 0
# computes tiles 0, 16, 32 and 48, one after another.

# %%
# Besides the number of CTAs, the order of the tiles matters too. A CDNA5 GPU is
# built from several chiplets, the accelerator complex dies (XCDs). An MI455X
# GPU has 8 XCDs, and each XCD has its own L2 cache and 32 WGPs. Each WGP has
# its own LDS and 4 SIMDs.
#
# The hardware sends the CTAs to the XCDs round-robin, so CTA `i` runs on XCD `i
# % 8`. Tiles next to each other share rows of `A` or columns of `W`, so they
# should run on the same XCD, where they share the L2 cache. But the round-robin
# order puts neighbors on different XCDs. So we remap the CTA index such that
# each XCD gets a contiguous range of tiles. The remap needs the number of CTAs
# to be a multiple of 8, which is true when we launch one CTA per WGP. Let's
# print the XCD of the CTA that computes each tile, without and with the remap.

if __name__ == "__main__":
    for xcd_remap in [False, True]:
        schedule_kernel[(16, )](cta, 64, xcd_remap)
        print(f"XCD_REMAP={xcd_remap}")
        for step in (cta % 8).view(-1, 16).tolist():
            print(" ".join(str(x) for x in step))

# %%
# ```
# XCD_REMAP=False
# 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
# 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
# 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
# 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
# XCD_REMAP=True
# 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7
# 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7
# 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7
# 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7
# ```
#
# With the remap, neighboring tiles run on the same XCD.

# %%
# The persistent kernel wraps the body of the kernel of `07-warp-pipeline.py`
# in the loop over the tiles. Each tile allocates its own shared memory, so the
# output tile can reuse the LDS of the input buffers. The stages of the K loop
# stay the same, and the two groups of warps realign after the K loop of every
# tile. The K blocks that are left after the loop run without stages, because
# stages outside the K loop would turn the loop over the tiles into a pipelined
# loop too. For the same reason, the K loop needs at least two iterations, or
# the compiler removes it.


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
    # The scales of half a K block need their own layout.
    ws_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(layouts.w_reg_layout, [w_smem.shape[1], HALF_K // 32])
    a = a_smem.index(slot).slice(half * HALF_K, HALF_K, 1).load(layouts.a_reg_layout)
    w = w_smem.index(slot).slice(half * HALF_K // 2, HALF_K // 2, 1).permute([1, 0]).load(layouts.w_reg_layout)
    w_scale = t5.unshuffle_scales_cdna5(ws_smem.index(slot)).slice(half * HALF_K // 32, HALF_K // 32, 1).load(ws_layout)
    return a, w, w_scale


@gluon.jit
def matmul_kernel(a_ptr, w_ptr, w_scale_ptr, y_ptr,  #
                  M, N, K: gl.constexpr,  #
                  stride_am, stride_wn, stride_scale, stride_ym,  #
                  BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                  XCD_REMAP: gl.constexpr):
    gl.static_assert(K % BLOCK_K == 0)
    gl.static_assert(K // BLOCK_K >= 2 * NUM_BUFFERS)
    layouts: gl.constexpr = t5.get_mx_wmma_layouts(BLOCK_M, BLOCK_N, BLOCK_K, gl.num_warps())
    a_desc = tdm.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    w_desc = tdm.make_tensor_descriptor(w_ptr, [N, K // 2], [stride_wn, 1], [BLOCK_N, BLOCK_K // 2],
                                        layouts.w_smem_layout)
    ws_desc = tdm.make_tensor_descriptor(w_scale_ptr, [gl.cdiv(N, 128), K // 32 * 128], [stride_scale, 1],
                                         [BLOCK_N // 128, BLOCK_K // 32 * 128], layouts.ws_smem_layout)
    y_smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0])

    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_tiles = gl.cdiv(M, BLOCK_M) * num_pid_n
    num_k: gl.constexpr = K // BLOCK_K
    TAIL: gl.constexpr = num_k % NUM_BUFFERS
    num_programs = gl.num_programs(0)
    pid = gl.program_id(0)
    if XCD_REMAP:
        pid = pid % 8 * (num_programs // 8) + pid // 8
    for t in range(pid, num_tiles, num_programs):
        a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K],
                                           layouts.a_smem_layout)
        w_smem = gl.allocate_shared_memory(w_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K // 2],
                                           layouts.w_smem_layout)
        ws_smem = gl.allocate_shared_memory(gl.uint8, [NUM_BUFFERS, BLOCK_N // 128, BLOCK_K // 32 * 128],
                                            layouts.ws_smem_layout)
        off_m = t // num_pid_n * BLOCK_M
        off_n = t % num_pid_n * BLOCK_N

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
                a, w, w_scale = matmul_ld_smem(i, half, a_smem, w_smem, ws_smem, layouts)
                acc = wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)

        y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], y_smem_layout)
        y_smem.store(acc.to(y_ptr.dtype.element_ty))
        y_desc = tdm.make_tensor_descriptor(y_ptr, [M, N], [stride_ym, 1], [BLOCK_M, BLOCK_N], y_smem_layout)
        y_desc = tdm.update_tensor_descriptor(y_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
        tdm.async_store(y_desc, src=y_smem)
        tdm.async_wait(0)


def matmul(A, W, W_scale, xcd_remap=True, BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, NUM_BUFFERS=3):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (torch.cuda.get_device_properties(A.device).multi_processor_count, )
    matmul_kernel[grid](
        A, W, W_scale, Y,  #
        M, N, K,  #
        A.stride(0), W.stride(0), W_scale.stride(0), Y.stride(0),  #
        BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, xcd_remap, num_warps=8)
    return Y


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 2048), (2048, 1024, 1792), (500, 300, 2560)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul(M, N, K):
    A = t5.random_fp8(M, K)
    W, W_scale, W_ref = t5.random_mxfp4(N, K)
    Y = matmul(A, W, t5.preshuffle_scales_cdna5(W_scale))
    torch.testing.assert_close(A.float() @ W_ref.T, Y.float(), atol=1e-1, rtol=1e-2)


# %%
# We compare the warp pipelined kernel of `07-warp-pipeline.py` with its
# persistent version on an MI455X GPU with M = N = K = 8192.

if __name__ == "__main__":
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 8192, 8192, 8192
    A = t5.random_fp8(M, K)
    W, W_scale, _ = t5.random_mxfp4(N, K)
    W_scale = t5.preshuffle_scales_cdna5(W_scale)

    def tflops(fn):
        return 2 * M * N * K / triton.testing.do_bench(fn, warmup=100, rep=500) * 1e-9

    r0 = tflops(lambda: t7.matmul(A, W, W_scale))
    r1 = tflops(lambda: matmul(A, W, W_scale, xcd_remap=False))
    r2 = tflops(lambda: matmul(A, W, W_scale))
    print("TFLOPS  non-persistent  persistent  persistent + XCD remap")
    print(f"8192    {r0:>14.0f}  {r1:>10.0f}  {r2:>22.0f}")

# %%
# ```
# TFLOPS  non-persistent  persistent  persistent + XCD remap
# 8192              7722        7801                    7870
# ```
