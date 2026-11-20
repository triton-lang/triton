"""
Warp Pipeline
=============

Many units on AMD GPUs can execute instructions in parallel. This includes
overlapping compute and memory accesses, or different types of compute. What
we have seen in previous tutorials already demonstrates this idea of overlapping
memory and compute within a warp. We can extend this idea to overlap
instructions across warps as well. This includes 2 different forms of
overlapping: hardware-controlled overlap, where we increase the occupancy of
warps without any control and let the hardware decide how to schedule
instructions across warps, and software-controlled overlap, where we have
fine-grained control of what instructions will be overlapped.

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
from triton.experimental.gluon.language.amd import cdna5

t5 = importlib.import_module("amd.05-mfma-wmma-scaled")
t7 = importlib.import_module("amd.07-multicta")


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not is_cdna5():
    raise RuntimeError("This tutorial requires an AMD CDNA5 GPU")

# A warp on CDNA5 GPUs has 32 lanes.
warp_size = gl.constexpr(32)

# %%
# All our previous tutorials assume 4 warps. Now we will switch to use 8 warps.
# On CDNA5 GPUs, each workgroup processor (WGP) is partitioned into 4 SIMDs,
# each with its own set of 1024 vector registers (VGPRs). When the kernel is
# launched with 8 warps, 2 warps will be assigned to each SIMD, and each can use
# up to 512 VGPRs. The assignment is done in round-robin fashion, so warps 0
# and 4 will be assigned to the same SIMD. The SIMD can issue instructions for
# both warps, which gives an opportunity to overlap the execution of different
# warps.
#
# To control what can be overlapped, we can use `warp_pipeline_stage` in Gluon
# to split a loop into stages. Operations within the scope of a stage will be
# grouped together. The compiler then interleaves 2 different stages from the 2
# warps on one SIMD.
#
# This interleaving is achieved by using conditional barriers, where the stages
# of one warp are executed earlier, leaving a gap of one stage. Before the loop,
# a conditional barrier holds back warps 4-7, so warps 0-3 can start one stage
# earlier. Then, at the end of each stage, all warps sync at a barrier, so warps
# 0-3 always stay one stage ahead. After the loop, another conditional barrier
# lets warps 4-7 catch up. Let's say each warp repeats stage 0 and stage 1. The
# timeline of each warp looks like below. Here `|` means a barrier.
#
# ```
# warps 0-3: stage 0 | stage 1 | stage 0 | stage 1 |
# warps 4-7:         | stage 0 | stage 1 | stage 0 | stage 1
# ```
#
# Warp pipeline can help to overlap compute and memory accesses, especially the
# memory accesses that have no explicit wait in Gluon, such as shared memory
# (LDS) reads. For the operations that have explicit waits, like async copies
# and TDM, the kernel can use multiple buffers and prefetch data ahead of time.
# The kernel can control when it issues the instructions and when it waits, so
# it can fit enough compute in between to hide the latency. But for LDS, the
# wait is implicit and managed by the compiler. The actual logic varies, but in
# general, the compiler will insert a wait count right before the instruction
# that uses the data. So it could happen that there isn't enough compute in
# between, causing stalls on memory access. There are different ways to
# mitigate this issue, for example "subtiling", which we introduced in tutorial
# 04, and which basically controls how many LDS reads will be issued together.
# But the overlap still happens within a single warp and can be limited.
#
# Warp pipeline may be able to achieve a better overlap of compute and memory
# accesses by using multiple warps. While one warp waits for its LDS reads, the
# SIMD can run the WMMAs of the other warp. We can build a simple kernel which
# only has LDS reads and WMMAs in the loop to demonstrate this idea.


@gluon.aggregate
class WmmaLayouts:
    a_smem_layout: gl.constexpr
    a_reg_layout: gl.constexpr
    b_smem_layout: gl.constexpr
    b_reg_layout: gl.constexpr
    acc_layout: gl.constexpr


@gluon.constexpr_function
def get_wmma_layouts(BLOCK_K, num_warps):
    smem_layout = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [256, BLOCK_K], [1, 0])
    a_smem_layout, b_smem_layout, acc_layout = cdna5.make_partitioned_dot_layouts(  #
        256, 256, smem_layout, smem_layout, num_warps, [16, 16, 128], b_transposed=True)
    return WmmaLayouts(a_smem_layout, gl.DotOperandLayout(0, acc_layout, 16),  #
                       b_smem_layout, gl.DotOperandLayout(1, acc_layout, 16),  #
                       acc_layout)


@gluon.jit
def wmma_kernel(a_ptr, b_ptr, y_ptr, k):
    BLOCK_K: gl.constexpr = 256
    layouts: gl.constexpr = get_wmma_layouts(BLOCK_K, gl.num_warps())
    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [256, BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(gl.float8e4nv, [256, BLOCK_K], layouts.b_smem_layout)

    # Load the tiles of A and B into shared memory once. The loop reads them
    # again and again, so it has no global memory accesses.
    a_offs = gl.arange(0, 256, gl.SliceLayout(1, layouts.a_reg_layout))[:, None] * BLOCK_K + \
             gl.arange(0, BLOCK_K, gl.SliceLayout(0, layouts.a_reg_layout))[None, :]
    a_tile = gl.load(a_ptr + a_offs)
    a_smem.store(a_tile)
    b_offs = gl.arange(0, BLOCK_K, gl.SliceLayout(1, layouts.b_reg_layout))[:, None] + \
             gl.arange(0, 256, gl.SliceLayout(0, layouts.b_reg_layout))[None, :] * BLOCK_K
    b_tile = gl.load(b_ptr + b_offs)
    b_smem.permute([1, 0]).store(b_tile)

    # The loop only does LDS reads and WMMAs.
    acc = gl.zeros([256, 256], gl.float32, layouts.acc_layout)
    for _ in range(0, k, BLOCK_K):
        # Subtile the block along the K dimension.
        a = a_smem.slice(0, BLOCK_K // 2, 1).load(layouts.a_reg_layout)
        b = b_smem.slice(0, BLOCK_K // 2, 1).permute([1, 0]).load(layouts.b_reg_layout)
        acc = cdna5.wmma_scaled(a, None, "e4m3", b, None, "e4m3", acc)
        a = a_smem.slice(BLOCK_K // 2, BLOCK_K // 2, 1).load(layouts.a_reg_layout)
        b = b_smem.slice(BLOCK_K // 2, BLOCK_K // 2, 1).permute([1, 0]).load(layouts.b_reg_layout)
        acc = cdna5.wmma_scaled(a, None, "e4m3", b, None, "e4m3", acc)

    m = gl.arange(0, 256, gl.SliceLayout(1, layouts.acc_layout))
    n = gl.arange(0, 256, gl.SliceLayout(0, layouts.acc_layout))
    gl.store(y_ptr + m[:, None] * 256 + n[None, :], acc)


# %%
# Next, let's split the loop into stages. Here we use 4 stages: one "memory"
# stage and one "compute" stage for each half of the K block.


@gluon.jit
def wmma_staged_kernel(a_ptr, b_ptr, y_ptr, k):
    BLOCK_K: gl.constexpr = 256
    layouts: gl.constexpr = get_wmma_layouts(BLOCK_K, gl.num_warps())
    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [256, BLOCK_K], layouts.a_smem_layout)
    b_smem = gl.allocate_shared_memory(gl.float8e4nv, [256, BLOCK_K], layouts.b_smem_layout)

    a_offs = gl.arange(0, 256, gl.SliceLayout(1, layouts.a_reg_layout))[:, None] * BLOCK_K + \
             gl.arange(0, BLOCK_K, gl.SliceLayout(0, layouts.a_reg_layout))[None, :]
    a_tile = gl.load(a_ptr + a_offs)
    a_smem.store(a_tile)
    b_offs = gl.arange(0, BLOCK_K, gl.SliceLayout(1, layouts.b_reg_layout))[:, None] + \
             gl.arange(0, 256, gl.SliceLayout(0, layouts.b_reg_layout))[None, :] * BLOCK_K
    b_tile = gl.load(b_ptr + b_offs)
    b_smem.permute([1, 0]).store(b_tile)

    acc = gl.zeros([256, 256], gl.float32, layouts.acc_layout)
    for _ in range(0, k, BLOCK_K):
        with gl.amd.warp_pipeline_stage("memory0", priority=1):
            a = a_smem.slice(0, BLOCK_K // 2, 1).load(layouts.a_reg_layout)
            b = b_smem.slice(0, BLOCK_K // 2, 1).permute([1, 0]).load(layouts.b_reg_layout)
        with gl.amd.warp_pipeline_stage("compute0", priority=0):
            acc = cdna5.wmma_scaled(a, None, "e4m3", b, None, "e4m3", acc)
        with gl.amd.warp_pipeline_stage("memory1", priority=1):
            a = a_smem.slice(BLOCK_K // 2, BLOCK_K // 2, 1).load(layouts.a_reg_layout)
            b = b_smem.slice(BLOCK_K // 2, BLOCK_K // 2, 1).permute([1, 0]).load(layouts.b_reg_layout)
        with gl.amd.warp_pipeline_stage("compute1", priority=0):
            acc = cdna5.wmma_scaled(a, None, "e4m3", b, None, "e4m3", acc)

    m = gl.arange(0, 256, gl.SliceLayout(1, layouts.acc_layout))
    n = gl.arange(0, 256, gl.SliceLayout(0, layouts.acc_layout))
    gl.store(y_ptr + m[:, None] * 256 + n[None, :], acc)


# %%
# Let's look at the IR of `wmma_staged_kernel`.

if __name__ == "__main__":
    A = t5.random_fp8(256, 256)
    B = t5.random_fp8(256, 256)
    y = torch.empty(256 * 256, device="cuda")
    compiled = wmma_staged_kernel[(1, )](A, B, y, 1024, num_warps=8)
    for line in compiled.asm["ttgir"].splitlines():
        if re.search(r"scf\.for|scf\.execute_region|local_load|dot_scaled|warp_pipeline", line):
            # Drop the types, the locations and the prefix of the attributes.
            line = re.split(r" : tensor| : !ttg| -> | lhs = | iter_args| loc\(", line)[0][4:]
            print(line.replace(" : i32", "").replace("triton.warp_pipeline.", ""))

# %%
# ```
# %acc = scf.for %_ = %c0_i32 to %k step %c256_i32
#   %acc_24:2 = scf.execute_region
#     %a_28 = ttg.local_load %a
#     %b_30 = ttg.local_load %b_29
#   } {priority = 1, stage = "memory0"}
#   %acc_25 = scf.execute_region
#     %acc_28 = tt.dot_scaled %acc_24#0 scale %cst_0, %acc_24#1 scale %cst, %acc_23
#   } {priority = 0, stage = "compute0"}
#   %acc_26:2 = scf.execute_region
#     %a_28 = ttg.local_load %a
#     %b_30 = ttg.local_load %b_29
#   } {priority = 1, stage = "memory1"}
#   %acc_27 = scf.execute_region
#     %acc_28 = tt.dot_scaled %acc_26#0 scale %cst_0, %acc_26#1 scale %cst, %acc_25
#   } {priority = 0, stage = "compute1"}
# } {pipelined_for}
# ```
#
# Each pipeline stage becomes an `scf.execute_region` with its name and
# priority. The loop of the stages is marked as `pipelined_for`, so that the
# compiler knows to pipeline this loop across warps 0-3 and warps 4-7. When the
# compiler lowers the TTGIR, the `pipelined_for` loop gets a conditional barrier
# before the loop to hold back warps 4-7, and another conditional barrier after
# the loop to let warps 4-7 catch up. Each `scf.execute_region` is inlined into
# the loop and ends with a barrier at the stage boundary, and its priority
# becomes an `s_setprio` before the stage, which sets the scheduling priority of
# the warps that run the stage. The generated assembly looks like below:
#
# ```
# ; before the loop: hold back warps 4-7 by one stage
# v_cmpx_lt_u32 0x7f, v0          ; turn off warps 0-3
# s_barrier_signal -1             ; warps 4-7 wait for "memory0" of warps 0-3
# s_barrier_wait -1
# s_setprio 1
#
# ; loop
# ds_load_b128 ... (x48)          ; "memory0" stage, without a wait
# s_setprio 0
# s_barrier_signal -1             ; stage boundary
# s_barrier_wait -1
# s_wait_dscnt 0x1c               ; "compute0" stage, waits for the first WMMA
# v_wmma_scale_f32_16x16x128_f8f6f4 ...
# s_wait_dscnt 0x18               ; waits for the next WMMA
# v_wmma_scale_f32_16x16x128_f8f6f4 ...
# ...
# s_setprio 1
# s_barrier_signal -1             ; stage boundary
# s_barrier_wait -1
# ...                             ; "memory1" and "compute1" stages
#
# ; after the loop: let warps 4-7 catch up
# s_setprio 0
# s_and_saveexec_b32 s0, vcc_lo   ; turn on only warps 0-3
# s_barrier_signal -1             ; warps 0-3 wait for "compute1" of warps 4-7
# s_barrier_wait -1
# ```
#
# Note that a "memory" stage doesn't wait for its LDS reads to finish, and the
# waits are in the next "compute" stage, because the compiler still puts each
# `s_wait_dscnt` right before the first WMMA that uses the data. Also, given
# that issuing is relatively short, when a warp has issued all its LDS reads in
# the "memory" stage, it has to wait for the "compute" stage of the other warp,
# which is the period when these LDS reads can happen.
#
# One more thing is that in the "compute" stage, the compiler will not wait for
# all operands at once. Instead, it waits for the operands of one WMMA at a
# time, which also helps to hide some latency within a warp.

if __name__ == "__main__":
    k = 2048 * 256
    for name, kernel, num_warps in [("wmma_w4", wmma_kernel, 4), ("wmma_w8", wmma_kernel, 8),
                                    ("wmma_w8_s4", wmma_staged_kernel, 8)]:
        ms = triton.testing.do_bench(lambda: kernel[(1024, )](A, B, y, k, num_warps=num_warps))
        print(f"{name}: {ms:.2f} ms")

# %%
# ```
# wmma_w4: 5.32 ms
# wmma_w8: 6.75 ms
# wmma_w8_s4: 4.69 ms
# ```
#
# We can see here that with pipeline stages, 8 warps can finish even quicker
# than the 4-warp version, while naively using 8 warps without pipeline stages
# can be slower.
#
# In practice, this observation may not always hold, as it depends on the stage
# splits and the workload of each stage. So it requires some tuning to find the
# optimal configuration for a given problem size.
#
# Moreover, here we only show the overlap of memory accesses and compute. We
# can also overlap different types of compute. For example, in attention, we can
# overlap the vector instructions of the softmax with the WMMAs of the QK and PV
# dots.

# %%
# Now we can apply warp pipeline to the final matmul kernel from the previous
# tutorial, with the same cluster layouts, and see how it performs.


@gluon.jit
def matmul_ld_gmem(k, pred, slot, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem,
                   BLOCK_K: gl.constexpr):
    a_desc = cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, k * BLOCK_K], pred=pred, clamp_bounds=True)
    w_desc = cdna5.tdm.update_tensor_descriptor(w_desc, add_offsets=[off_n, k * BLOCK_K // 2], pred=pred,
                                                clamp_bounds=True)
    ws_desc = cdna5.tdm.update_tensor_descriptor(ws_desc, add_offsets=[off_n // 128, k * BLOCK_K // 32 * 128],
                                                 pred=pred, clamp_bounds=True)
    # Only warps 0-3 copy. tdm.async_wait only waits for the copies of the current
    # warp, and warps 4-7 run one stage behind, so warps 0-3 could read a part
    # copied by warps 4-7 before the copy completes.
    cdna5.tdm.async_load(a_desc, dest=a_smem.index(slot), warp_used_hint=0b1111)
    cdna5.tdm.async_load(w_desc, dest=w_smem.index(slot), warp_used_hint=0b1111)
    cdna5.tdm.async_load(ws_desc, dest=ws_smem.index(slot), warp_used_hint=0b1111)


@gluon.jit
def matmul_ld_smem(slot, half: gl.constexpr, a_smem, w_smem, ws_smem, layouts: gl.constexpr):
    HALF_K: gl.constexpr = a_smem.shape[2] // 2
    ws_layout: gl.constexpr = cdna5.get_wmma_scale_layout(layouts.w_reg_layout, [w_smem.shape[1], HALF_K // 32])
    a = a_smem.index(slot).slice(half * HALF_K, HALF_K, 1).load(layouts.a_reg_layout)
    w = w_smem.index(slot).slice(half * HALF_K // 2, HALF_K // 2, 1).permute([1, 0]).load(layouts.w_reg_layout)
    w_scale = t5.unshuffle_scales_cdna5(ws_smem.index(slot)).slice(half * HALF_K // 32, HALF_K // 32, 1).load(ws_layout)
    return a, w, w_scale


@gluon.jit
def matmul_kernel(a_ptr, w_ptr, w_scale_ptr, y_ptr,  #
                  M, N, K,  #
                  stride_am, stride_wn, stride_scale, stride_ym,  #
                  CTAS_M: gl.constexpr, CTAS_N: gl.constexpr, WARP_PIPELINE: gl.constexpr,  #
                  NUM_BUFFERS: gl.constexpr = 3):
    BLOCK_M: gl.constexpr = 256 * CTAS_M
    BLOCK_N: gl.constexpr = 256 * CTAS_N
    BLOCK_K: gl.constexpr = 256
    layouts: gl.constexpr = t7.get_cluster_layouts(CTAS_M, CTAS_N, BLOCK_M, BLOCK_N, BLOCK_K, gl.num_warps())

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

    num_k = gl.cdiv(K, BLOCK_K)
    for i in gl.static_range(NUM_BUFFERS - 1):
        matmul_ld_gmem(i, i < num_k, i, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem, BLOCK_K)
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k):
        cdna5.tdm.async_wait((NUM_BUFFERS - 2) * 3)
        next_k = k + NUM_BUFFERS - 1
        if WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("memory0", priority=1):
                a, w, w_scale = matmul_ld_smem(k % NUM_BUFFERS, 0, a_smem, w_smem, ws_smem, layouts)
                matmul_ld_gmem(next_k, next_k < num_k, next_k % NUM_BUFFERS, off_m, off_n, a_desc, w_desc, ws_desc,
                               a_smem, w_smem, ws_smem, BLOCK_K)
            with gl.amd.warp_pipeline_stage("compute0", priority=0):
                acc = cdna5.wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)
            with gl.amd.warp_pipeline_stage("memory1", priority=1):
                a, w, w_scale = matmul_ld_smem(k % NUM_BUFFERS, 1, a_smem, w_smem, ws_smem, layouts)
            with gl.amd.warp_pipeline_stage("compute1", priority=0):
                acc = cdna5.wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)
        else:
            # The same loop without stages.
            a, w, w_scale = matmul_ld_smem(k % NUM_BUFFERS, 0, a_smem, w_smem, ws_smem, layouts)
            matmul_ld_gmem(next_k, next_k < num_k, next_k % NUM_BUFFERS, off_m, off_n, a_desc, w_desc, ws_desc, a_smem,
                           w_smem, ws_smem, BLOCK_K)
            acc = cdna5.wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)
            a, w, w_scale = matmul_ld_smem(k % NUM_BUFFERS, 1, a_smem, w_smem, ws_smem, layouts)
            acc = cdna5.wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)

    y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], layouts.y_smem_layout)
    y_smem.store(acc.to(y_ptr.dtype.element_ty))
    y_desc = cdna5.tdm.make_tensor_descriptor(y_ptr, [M, N], [stride_ym, 1], [BLOCK_M, BLOCK_N], layouts.y_smem_layout)
    y_desc = cdna5.tdm.update_tensor_descriptor(y_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
    cdna5.tdm.async_store(y_desc, src=y_smem)
    cdna5.tdm.async_wait(0)


def matmul(A, W, W_scale, CTAS_M=4, CTAS_N=4, warp_pipeline=True):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 256 * CTAS_M) * triton.cdiv(N, 256 * CTAS_N), )
    matmul_kernel[grid](
        A, W, W_scale, Y,  #
        M, N, K,  #
        A.stride(0), W.stride(0), W_scale.stride(0), Y.stride(0),  #
        CTAS_M, CTAS_N, warp_pipeline, num_warps=8, num_ctas=CTAS_M * CTAS_N)
    return Y


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 1024), (2048, 1024, 768)])
@pytest.mark.parametrize("CTAS_M, CTAS_N", [(1, 1), (2, 2), (4, 4)])
@pytest.mark.parametrize("warp_pipeline", [False, True])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul(M, N, K, CTAS_M, CTAS_N, warp_pipeline):
    A = t5.random_fp8(M, K)
    W, W_scale, W_ref = t5.random_mxfp4(N, K)
    Y = matmul(A, W, t5.preshuffle_scales_cdna5(W_scale), CTAS_M, CTAS_N, warp_pipeline=warp_pipeline)
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

    bench("4 warps", lambda: t7.matmul(A, W, W_scale, 4, 4))
    bench("8 warps", lambda: matmul(A, W, W_scale, warp_pipeline=False))
    bench("8 warps, warp pipeline", lambda: matmul(A, W, W_scale))

# %%
# ```
# 4 warps: 8424 TFLOPS
# 8 warps: 7860 TFLOPS
# 8 warps, warp pipeline: 8842 TFLOPS
# ```
#
# We can see the 8-warp version with warp pipeline is much better than the
# version without, and it is also faster than the 4-warp version.
