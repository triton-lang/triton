"""
Warp Pipeline
=============

Many units on AMD GPUs can execute instructions in parallel. What we have built
already shows this when we overlap global memory accesses with MFMA/WMMA. But
these previous examples overlap instructions within a warp. AMD GPUs also allow
overlapping instructions across warps. This includes 2 different forms of
overlapping: hardware-controlled overlap, where we increase the occupancy of
warps without any control and let the hardware decide how to schedule
instructions across warps, and software-controlled overlap, which we will
discuss in this tutorial.

The way we achieve software-controlled overlap across warps is called "block
pingpong", or "warp pipeline" in Gluon. It is similar to the concept of "warp
specialization", but implemented differently. At a high level, we split the
kernel program into stages and let the stages run at a shifted pace for
different warps. Therefore, instructions from different stages can be executed
in parallel. Typically, we want to overlap different types of stages, for
example memory vs. compute, or different types of compute.

Gluon exposes warp pipeline via the API `warp_pipeline_stage` to declare a
stage, and it automatically does the overlap for us. This tutorial will explain
how warp pipeline works and showcase it in the matmul kernel we built in the
previous chapters.
"""

import importlib
import re
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna5 import tdm, wmma_scaled

t5 = importlib.import_module("amd.05-mfma-wmma-scaled")


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not is_cdna5():
    raise RuntimeError("This tutorial requires an AMD CDNA5 GPU")

# A warp on CDNA5 GPUs has 32 lanes.
warp_size = gl.constexpr(32)

# %%
# All our previous tutorials assume 4 warps. Now we switch to the 8-warp
# configuration, where 2 warps share a SIMD. Note that the two warps of a SIMD
# can only run at the same time if they meet the occupancy requirements: the
# VGPRs of both warps must fit in the 1024 VGPRs of the SIMD, and the shared
# memory of the CTA must fit in the 320 KB of the WGP.
#
# In Gluon, we use `gl.amd.warp_pipeline_stage` to split a loop into stages for
# 8 warps. Each `with gl.amd.warp_pipeline_stage(...)` groups a region of Gluon
# ops as a stage. The compiler will interleave 2 different stages from the 2
# warps on one SIMD. This interleaving is achieved by using conditional
# barriers, where the stages of one warp are executed earlier, leaving a gap of
# one stage. Specifically, let's call warps 0-3 group0 and warps 4-7 group1, and
# let group0 go first. Before the loop, a conditional barrier holds back group1,
# so group0 starts one stage earlier. Then, at the end of each stage, group0 and
# group1 wait for each other at a barrier, so group0 always stays one stage
# ahead of group1. After the loop, another conditional barrier lets group1 catch
# up. Let's say each warp repeats stage 0 and stage 1. The timeline of each
# group looks like below, where `|` means a barrier.
#
# ```
# group0: stage 0 | stage 1 | stage 0 | stage 1 |
# group1:         | stage 0 | stage 1 | stage 0 | stage 1
# ```
#
# Warp pipeline can help to overlap compute and memory accesses, especially the
# memory accesses that have no explicit wait in Gluon, such as shared memory
# (LDS) reads. Async copies and TDM copies have explicit waits, so with several
# buffers, a single warp already hides their latency. Shared memory reads don't.
# The compiler puts their wait right before the first use of the data. Recall
# from the previous tutorials that we can use subtiling to overlap them with the
# WMMAs. But a single warp can still only overlap them partly, because it issues
# its instructions in order and stops at each wait.
#
# Warp pipeline can achieve an even better interleaving of shared memory reads
# and WMMAs, and keep both units busy. While one warp waits for its shared
# memory reads, the SIMD can run the WMMAs of the other warp. But the two warps
# run the same loop, so without coordination, they tend to read shared memory at
# the same time, and then run WMMAs at the same time. While both read, the
# matrix core is idle, and while both run WMMAs, the shared memory is idle. Warp
# pipeline keeps the two groups one stage apart, so that one group reads shared
# memory while the other group runs WMMAs, and both stay busy.
#
# We can build a simple memory + WMMA loop. It keeps a block of `A` and `B` in
# shared memory, and each iteration runs the WMMAs of the block without any
# access to global memory. As in `04-mfma-wmma.py`, the loop splits the K
# dimension of the block into two sub-tiles with `slice`. It reads each sub-tile
# from shared memory and runs one WMMA per sub-tile, so the compiler can issue
# the reads of the second sub-tile while the WMMA of the first one runs. First,
# we run it without stages, with 4 and 8 warps.


@gluon.jit
def wmma_kernel(y_ptr, k):
    BLOCK_K: gl.constexpr = 256
    layouts: gl.constexpr = t5.get_mx_wmma_layouts(256, 256, BLOCK_K, gl.num_warps())
    bs_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(layouts.w_reg_layout, [256, 4])
    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [256, BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(gl.uint8, [256, BLOCK_K // 2], layouts.w_smem_layout)
    a_smem.store(gl.full([256, BLOCK_K], 0.0, gl.float8e4nv, layouts.a_reg_layout))
    b_smem.permute([1, 0]).store(gl.full([BLOCK_K // 2, 256], 0, gl.uint8, layouts.w_reg_layout))
    b_scale = gl.full([256, 4], 127, gl.uint8, bs_layout)
    acc = gl.zeros([256, 256], gl.float32, layouts.acc_layout)
    for _ in range(0, k, BLOCK_K):
        a = a_smem.slice(0, BLOCK_K // 2, 1).load(layouts.a_reg_layout)
        b = b_smem.slice(0, BLOCK_K // 4, 1).permute([1, 0]).load(layouts.w_reg_layout)
        acc = wmma_scaled(a, None, "e4m3", b, b_scale, "e2m1", acc)
        a = a_smem.slice(BLOCK_K // 2, BLOCK_K // 2, 1).load(layouts.a_reg_layout)
        b = b_smem.slice(BLOCK_K // 4, BLOCK_K // 4, 1).permute([1, 0]).load(layouts.w_reg_layout)
        acc = wmma_scaled(a, None, "e4m3", b, b_scale, "e2m1", acc)
    m = gl.arange(0, 256, gl.SliceLayout(1, layouts.acc_layout))
    n = gl.arange(0, 256, gl.SliceLayout(0, layouts.acc_layout))
    gl.store(y_ptr + m[:, None] * 256 + n[None, :], acc)


# %%
# Next, let's split the loop into stages. Here we show a 2-stage version and a
# 4-stage version. The 2-stage version has a "memory" stage that reads both
# halves and a "compute" stage that runs both WMMAs. The 4-stage version has one
# "memory" stage and one "compute" stage per half.


@gluon.jit
def wmma_2stage_kernel(y_ptr, k):
    BLOCK_K: gl.constexpr = 256
    layouts: gl.constexpr = t5.get_mx_wmma_layouts(256, 256, BLOCK_K, gl.num_warps())
    bs_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(layouts.w_reg_layout, [256, 4])
    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [256, BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(gl.uint8, [256, BLOCK_K // 2], layouts.w_smem_layout)
    a_smem.store(gl.full([256, BLOCK_K], 0.0, gl.float8e4nv, layouts.a_reg_layout))
    b_smem.permute([1, 0]).store(gl.full([BLOCK_K // 2, 256], 0, gl.uint8, layouts.w_reg_layout))
    b_scale = gl.full([256, 4], 127, gl.uint8, bs_layout)
    acc = gl.zeros([256, 256], gl.float32, layouts.acc_layout)
    for _ in range(0, k, BLOCK_K):
        with gl.amd.warp_pipeline_stage("memory", priority=1):
            a0 = a_smem.slice(0, BLOCK_K // 2, 1).load(layouts.a_reg_layout)
            b0 = b_smem.slice(0, BLOCK_K // 4, 1).permute([1, 0]).load(layouts.w_reg_layout)
            a1 = a_smem.slice(BLOCK_K // 2, BLOCK_K // 2, 1).load(layouts.a_reg_layout)
            b1 = b_smem.slice(BLOCK_K // 4, BLOCK_K // 4, 1).permute([1, 0]).load(layouts.w_reg_layout)
        with gl.amd.warp_pipeline_stage("compute", priority=0):
            acc = wmma_scaled(a0, None, "e4m3", b0, b_scale, "e2m1", acc)
            acc = wmma_scaled(a1, None, "e4m3", b1, b_scale, "e2m1", acc)
    m = gl.arange(0, 256, gl.SliceLayout(1, layouts.acc_layout))
    n = gl.arange(0, 256, gl.SliceLayout(0, layouts.acc_layout))
    gl.store(y_ptr + m[:, None] * 256 + n[None, :], acc)


@gluon.jit
def wmma_4stage_kernel(y_ptr, k):
    BLOCK_K: gl.constexpr = 256
    layouts: gl.constexpr = t5.get_mx_wmma_layouts(256, 256, BLOCK_K, gl.num_warps())
    bs_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(layouts.w_reg_layout, [256, 4])
    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [256, BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(gl.uint8, [256, BLOCK_K // 2], layouts.w_smem_layout)
    a_smem.store(gl.full([256, BLOCK_K], 0.0, gl.float8e4nv, layouts.a_reg_layout))
    b_smem.permute([1, 0]).store(gl.full([BLOCK_K // 2, 256], 0, gl.uint8, layouts.w_reg_layout))
    b_scale = gl.full([256, 4], 127, gl.uint8, bs_layout)
    acc = gl.zeros([256, 256], gl.float32, layouts.acc_layout)
    for _ in range(0, k, BLOCK_K):
        with gl.amd.warp_pipeline_stage("memory0", priority=1):
            a = a_smem.slice(0, BLOCK_K // 2, 1).load(layouts.a_reg_layout)
            b = b_smem.slice(0, BLOCK_K // 4, 1).permute([1, 0]).load(layouts.w_reg_layout)
        with gl.amd.warp_pipeline_stage("compute0", priority=0):
            acc = wmma_scaled(a, None, "e4m3", b, b_scale, "e2m1", acc)
        with gl.amd.warp_pipeline_stage("memory1", priority=1):
            a = a_smem.slice(BLOCK_K // 2, BLOCK_K // 2, 1).load(layouts.a_reg_layout)
            b = b_smem.slice(BLOCK_K // 4, BLOCK_K // 4, 1).permute([1, 0]).load(layouts.w_reg_layout)
        with gl.amd.warp_pipeline_stage("compute1", priority=0):
            acc = wmma_scaled(a, None, "e4m3", b, b_scale, "e2m1", acc)
    m = gl.arange(0, 256, gl.SliceLayout(1, layouts.acc_layout))
    n = gl.arange(0, 256, gl.SliceLayout(0, layouts.acc_layout))
    gl.store(y_ptr + m[:, None] * 256 + n[None, :], acc)


# %%
# Let's look at the TTGIR of `wmma_2stage_kernel`.

if __name__ == "__main__":
    y = torch.empty(256 * 256, device="cuda")
    compiled = wmma_2stage_kernel[(1, )](y, 1024, num_warps=8)
    for line in compiled.asm["ttgir"].splitlines():
        if re.search(r"scf\.for|scf\.execute_region|local_load|dot_scaled|warp_pipeline", line):
            # Drop the types, the locations and the prefix of the attributes.
            line = re.split(r" : tensor| : !ttg| -> | lhs = | iter_args| loc\(", line)[0][4:]
            print(line.replace(" : i32", "").replace("triton.warp_pipeline.", ""))

# %%
# ```
# %acc = scf.for %_ = %c0_i32 to %k step %c256_i32
#   %acc_6:4 = scf.execute_region
#     %a0_8 = ttg.local_load %a0
#     %b0_10 = ttg.local_load %b0_9
#     %a1_11 = ttg.local_load %a1
#     %b1_13 = ttg.local_load %b1_12
#   } {priority = 1, stage = "memory"}
#   %acc_7 = scf.execute_region
#     %acc_8 = tt.dot_scaled %acc_6#0 scale %cst, %acc_6#1 scale %cst_2, %acc_5
#     %acc_9 = tt.dot_scaled %acc_6#2 scale %cst, %acc_6#3 scale %cst_2, %acc_8
#   } {priority = 0, stage = "compute"}
# } {pipelined_for}
# ```
#
# Each pipeline stage becomes an `scf.execute_region` with its name and
# priority. The loop of the stages is marked as `pipelined_for`, so that the
# compiler knows to pipeline this loop across group0 and group1. When the
# compiler lowers the TTGIR, the `pipelined_for` loop gets a conditional barrier
# before the loop to hold back group1, and another conditional barrier after the
# loop to let group1 catch up. Each `scf.execute_region` is inlined into the
# loop and ends with a barrier at the stage boundary, and its priority becomes
# an `s_setprio` before the stage, which sets the scheduling priority of the
# warps that run the stage. The generated assembly looks like below:
#
# ```
# ; before the loop: hold back group1 by one stage
# v_cmpx_lt_u32 0x7f, v0          ; turn off group0
# s_barrier_signal -1             ; group1 waits until group0 finishes "memory"
# s_barrier_wait -1
# s_setprio 1
#
# ; loop
# ds_load_b128 ... (x64)          ; "memory" stage, without a wait
# s_setprio 0
# s_barrier_signal -1             ; stage boundary
# s_barrier_wait -1
# s_wait_dscnt 0x2e               ; "compute" stage, waits for the first WMMA
# v_wmma_scale_f32_16x16x128_f8f6f4 ...
# s_wait_dscnt 0x2c               ; waits for the second WMMA
# v_wmma_scale_f32_16x16x128_f8f6f4 ...
# ...
# s_setprio 1
# s_barrier_signal -1             ; stage boundary
# s_barrier_wait -1
#
# ; after the loop: let group1 catch up
# s_setprio 0
# s_and_saveexec_b32 s0, vcc_lo   ; turn on only group0
# s_barrier_signal -1             ; group0 waits until group1 finishes "compute"
# s_barrier_wait -1
# ```
#
# Note that the "memory" stage doesn't wait for its shared memory reads, and the
# waits are in the "compute" stage. The compiler puts each `s_wait_dscnt` right
# before the first WMMA that uses the data. This helps in two ways. First, the
# "memory" stage only issues the reads, so it ends early. The stage boundary is
# a barrier for both groups, so a wait for all reads there would also hold the
# other group, even after it finished its WMMAs. Second, the "compute" stage
# waits for the operands of one WMMA at a time, so its first WMMAs run while its
# later reads still arrive.
#
# The compiler only adds a wait at a stage boundary when the two groups depend
# on each other through shared memory. For example, when one group writes shared
# memory that the other group reads in its next stage, or when a buffer is
# refilled while the other group may still read it. This loop only reads shared
# memory, so there is no such dependency, and no wait at the stage boundary.

# %%
# Now let's run the kernels and compare their time.

if __name__ == "__main__":
    k = 2048 * 256
    for name, kernel, num_warps in [("wmma_w4", wmma_kernel, 4), ("wmma_w8", wmma_kernel, 8),
                                    ("wmma_w8_s2", wmma_2stage_kernel, 8), ("wmma_w8_s4", wmma_4stage_kernel, 8)]:
        ms = triton.testing.do_bench(lambda: kernel[(1024, )](y, k, num_warps=num_warps))
        print(f"{name}: {ms:.2f} ms")

# %%
# ```
# wmma_w4: 5.02 ms
# wmma_w8: 6.34 ms
# wmma_w8_s2: 11.88 ms
# wmma_w8_s4: 4.62 ms
# ```
#
# With 8 warps and no stages, each warp computes a 64 x 128 part of the block
# instead of 128 x 128, so the CTA reads 256 KB from shared memory per iteration
# instead of 192 KB, and the two warps of a SIMD often read shared memory at the
# same time. `wmma_w8_s2` holds the operands of both halves in registers at the
# same time. It needs more than the 512 VGPRs that a warp can use with 8 warps,
# so the compiler spills some registers to scratch memory inside the loop.
# `wmma_w8_s4` holds the operands of only one half and uses 424 VGPRs. Each
# "compute" stage of one group runs next to a "memory" stage of the other group,
# so the matrix core stays busy while the shared memory reads of the other group
# arrive.
#
# So the size of the stages matters. A stage should be large enough to overlap
# with the stage of the other group, but small enough that the registers of the
# two warps of a SIMD still fit.
#
# Here we only show the overlap of memory accesses and compute. In practice, we
# can also overlap different types of compute. For example, in attention, we can
# overlap the vector instructions of the softmax with the WMMAs of the QK and PV
# matmuls.

# %%
# Now we apply warp pipeline to the fp8 x mxfp4 matmul of
# `05-mfma-wmma-scaled.py`. The kernel uses 8 warps, and each K block runs in
# four stages. The first stage reads the first half of the K block from shared
# memory, and the second stage runs its WMMAs. The third and fourth stages do
# the same for the second half. The first stage also refills the buffer slot of
# the previous K block, which both groups have finished reading. So while one
# group runs a "wmma" stage, the other group reads shared memory or copies.
#
# We give the stages that access memory priority 1, so their instructions issue
# next to the WMMAs of the other group.


@gluon.jit
def matmul_ld_gmem(k, pred, slot: gl.constexpr, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem,
                   BLOCK_K: gl.constexpr):
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, k * BLOCK_K], pred=pred, clamp_bounds=True)
    w_desc = tdm.update_tensor_descriptor(w_desc, add_offsets=[off_n, k * BLOCK_K // 2], pred=pred, clamp_bounds=True)
    ws_desc = tdm.update_tensor_descriptor(ws_desc, add_offsets=[off_n // 128, k * BLOCK_K // 32 * 128], pred=pred,
                                           clamp_bounds=True)
    # Only group0 copies. tdm.async_wait only waits for the copies of the current
    # warp, and group1 runs one stage behind, so group0 could read a part copied
    # by group1 before the copy completes.
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
                  NUM_BUFFERS: gl.constexpr):
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = 256
    BLOCK_K: gl.constexpr = 256
    gl.static_assert(K % BLOCK_K == 0)
    layouts: gl.constexpr = t5.get_mx_wmma_layouts(BLOCK_M, BLOCK_N, BLOCK_K, gl.num_warps())

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
    for i in gl.static_range(NUM_BUFFERS - 1):
        matmul_ld_gmem(i, i < num_k, i, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem, BLOCK_K)
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(0, num_k - TAIL, NUM_BUFFERS):
        for i in gl.static_range(NUM_BUFFERS):
            # A wait can't be inside a stage, so we wait before the first stage.
            tdm.async_wait((NUM_BUFFERS - 2) * 3)
            with gl.amd.warp_pipeline_stage("lds_low", priority=1):
                a, w, w_scale = matmul_ld_smem(i, 0, a_smem, w_smem, ws_smem, layouts)
                next_k = k + i + NUM_BUFFERS - 1
                matmul_ld_gmem(next_k, next_k < num_k, (i + NUM_BUFFERS - 1) % NUM_BUFFERS, off_m, off_n, a_desc,
                               w_desc, ws_desc, a_smem, w_smem, ws_smem, BLOCK_K)
            with gl.amd.warp_pipeline_stage("wmma_low", priority=0):
                acc = wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)
            with gl.amd.warp_pipeline_stage("lds_high", priority=1):
                a, w, w_scale = matmul_ld_smem(i, 1, a_smem, w_smem, ws_smem, layouts)
            with gl.amd.warp_pipeline_stage("wmma_high", priority=0):
                acc = wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)
    for i in gl.static_range(TAIL):
        tdm.async_wait((TAIL - 1 - i) * 3)
        for half in gl.static_range(2):
            with gl.amd.warp_pipeline_stage("lds", priority=1):
                a, w, w_scale = matmul_ld_smem(i, half, a_smem, w_smem, ws_smem, layouts)
            with gl.amd.warp_pipeline_stage("wmma", priority=0):
                acc = wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)

    y_smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0])
    y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], y_smem_layout)
    y_smem.store(acc.to(y_ptr.dtype.element_ty))
    y_desc = tdm.make_tensor_descriptor(y_ptr, [M, N], [stride_ym, 1], [BLOCK_M, BLOCK_N], y_smem_layout)
    y_desc = tdm.update_tensor_descriptor(y_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
    tdm.async_store(y_desc, src=y_smem)
    tdm.async_wait(0)


def matmul(A, W, W_scale, NUM_BUFFERS=3):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256), )
    matmul_kernel[grid](
        A, W, W_scale, Y,  #
        M, N, K,  #
        A.stride(0), W.stride(0), W_scale.stride(0), Y.stride(0),  #
        NUM_BUFFERS, num_warps=8)
    return Y


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 1024), (2048, 1024, 768), (500, 300, 1280)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul(M, N, K):
    A = t5.random_fp8(M, K)
    W, W_scale, W_ref = t5.random_mxfp4(N, K)
    Y = matmul(A, W, t5.preshuffle_scales_cdna5(W_scale))
    torch.testing.assert_close(A.float() @ W_ref.T, Y.float(), atol=1e-1, rtol=1e-2)


# %%
# We can compare this kernel with the kernel of `05-mfma-wmma-scaled.py`.

if __name__ == "__main__":
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 2048, 2048, 16384
    A = t5.random_fp8(M, K)
    W, W_scale, _ = t5.random_mxfp4(N, K)
    W_scale = t5.preshuffle_scales_cdna5(W_scale)

    def bench(name, fn):
        ms = triton.testing.do_bench(fn, warmup=100, rep=500)
        print(f"{name}: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

    bench("4 warp", lambda: t5.matmul_cdna5(A, W, W_scale))
    bench("8 warp", lambda: matmul(A, W, W_scale))

# %%
# ```
# 4 warp: 2750 TFLOPS
# 8 warp: 3101 TFLOPS
# ```
