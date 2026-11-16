"""
Warp Pipelining
===============

A SIMD holds several warps at the same time, and in each cycle it issues
instructions from the warps that are ready. Matrix, vector, LDS and memory
instructions use different units, so instructions from different warps can
execute at the same time. On CDNA5, a warp that issues a WMMA can't issue its
next instruction until the WMMA finishes, so one warp alone can't overlap its
WMMAs with its other work. Warp pipelining splits the warps of a CTA into two
groups that run the same loop one stage apart, so the WMMAs of one group overlap
with the memory accesses of the other. It is similar to warp specialization on
NVIDIA GPUs, but all warps run the same code.

In this tutorial, we will explain how warp pipelining works and how to split a
loop into stages. In the end, we will apply it to the fp8 x mxfp4 matmul of
`05-mfma-wmma-scaled.py`.
"""

import importlib
import re
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna5
from triton.experimental.gluon.language.amd.cdna5 import tdm, wmma_scaled

t5 = importlib.import_module("05-mfma-wmma-scaled")


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not is_cdna5():
    raise RuntimeError("This tutorial requires an AMD CDNA5 GPU")

# A warp on CDNA5 GPUs has 32 lanes.
warp_size = gl.constexpr(32)

# %%
# In Gluon, warp pipelining is exposed as `gl.amd.warp_pipeline_stage`. Each
# `with` block of it in a loop becomes a stage. The compiler splits the warps
# into two groups, warps 0-3 and warps 4-7, so each SIMD has one warp of each
# group. Before the loop, a conditional barrier holds back the second group by
# one stage. At every stage boundary, the two groups meet at a barrier, so they
# stay one stage apart. After the loop, another conditional barrier aligns them
# again. The `priority` of a stage becomes an `s_setprio` for the warps that run
# the stage.
#
# Let's look at the barriers of a small loop, with a stage that loads from
# global memory and a stage that computes.


@gluon.jit
def two_stage_kernel(x_ptr, y_ptr, N):
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[warp_size], warps_per_cta=[8],
                                            order=[0])
    offs = gl.arange(0, 256, layout)
    acc = gl.zeros([256], gl.float32, layout)
    for i in range(N):
        with gl.amd.warp_pipeline_stage("load", priority=1):
            x = gl.load(x_ptr + i * 256 + offs)
        with gl.amd.warp_pipeline_stage("compute", priority=0):
            acc = acc * x + 1.0
    gl.store(y_ptr + offs, acc)


if __name__ == "__main__":
    x = torch.randn(64 * 256, device="cuda")
    y = torch.empty(256, device="cuda")
    compiled = two_stage_kernel[(1, )](x, y, 64, num_warps=8)
    asm = compiled.asm["amdgcn"]
    lines = [line.strip() for line in asm[asm.index(".text"):asm.index(".Lfunc_end")].splitlines()]
    pattern = r"(s_barrier|s_setprio|v_cmp|s_and_saveexec|global_load|v_add_f32)"
    head = next(i for i, line in enumerate(lines) if "Loop Header" in line)
    tail = max(i for i, line in enumerate(lines) if line.startswith("s_cbranch_scc1"))
    print("before the loop:")
    for line in lines[:head]:
        if re.match(pattern, line):
            print(f"  {line}")
    print("loop:")
    barriers = 0
    for line in lines[head:]:
        if re.match(pattern, line):
            print(f"  {line}")
        barriers += line.startswith("s_barrier_wait")
        if barriers == 2:
            break
    print("after the loop:")
    for line in lines[tail:]:
        if re.match(pattern, line):
            print(f"  {line}")
        if line.startswith("s_barrier_wait"):
            break

# %%
# ```
# before the loop:
#   s_barrier_signal -1
#   v_cmp_gt_u32_e32 vcc_lo, 0x80, v0
#   s_barrier_wait -1
#   v_cmpx_lt_u32_e32 0x7f, v0
#   s_barrier_signal -1
#   s_barrier_wait -1
#   s_setprio 1
# loop:
#   global_load_b32 v2, v[6:7], off
#   s_setprio 0
#   s_barrier_signal -1
#   s_barrier_wait -1
#   v_add_f32_e32 v2, 1.0, v2
#   s_setprio 1
#   s_barrier_signal -1
#   s_barrier_wait -1
# after the loop:
#   s_setprio 0
#   s_and_saveexec_b32 s0, vcc_lo
#   s_barrier_signal -1
#   s_barrier_wait -1
# ```
#
# Before the loop, all warps meet at a barrier, and `v_cmp_gt_u32` marks the
# threads of warps 0-3 in `vcc_lo`. Then `v_cmpx_lt_u32` turns off warps 0-3,
# so only warps 4-7 run one more barrier. Warps 0-3 go ahead and run the first
# stage, and warps 4-7 pass their barrier only at the end of that stage, so they
# start the loop one stage later. In the loop, each stage ends with the
# `s_setprio` of the next stage and a barrier. After the loop,
# `s_and_saveexec_b32` turns warps 0-3 back on, and they run one more barrier
# while warps 4-7 finish their last stage.

# %%
# The stages decide what overlaps. While one group runs a stage, the other group
# runs the stage before it. A loop with two stages overlaps the two with each
# other, and there are several useful ways to split a loop. A split between
# memory accesses and compute hides the latency of the loads behind the WMMAs.
# A split between LDS reads and WMMAs keeps the matrix core busy while the
# operands arrive. A split between WMMAs and vector instructions fits attention
# kernels, where the softmax runs many VALU instructions.
#
# The following small WMMA loops show each split. They only show the stages, and
# their performance doesn't matter here. In the printout, `G` is a global load,
# `T` is a TDM copy, `R` is an LDS read, `W` is a WMMA, `V` is another vector
# instruction, and `|` is a barrier.


@gluon.jit
def offs_2d(rows: gl.constexpr, cols: gl.constexpr, layout: gl.constexpr):
    r = gl.arange(0, rows, gl.SliceLayout(1, layout))
    c = gl.arange(0, cols, gl.SliceLayout(0, layout))
    return r[:, None] * cols + c[None, :]


@gluon.jit
def wmma_w_gmem_kernel(a_ptr, b_ptr, c_ptr, N: gl.constexpr):
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=[[1, 0], [2, 0], [4, 0]],
                                                     reg_bases=[], instr_shape=[16, 16, 32])
    a_layout: gl.constexpr = gl.DotOperandLayout(0, wmma_layout, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, wmma_layout, 8)
    acc = gl.zeros([128, 16], gl.float32, wmma_layout)
    for i in range(N):
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a = gl.load(a_ptr + i * 128 * 32 + offs_2d(128, 32, a_layout))
            b = gl.load(b_ptr + i * 32 * 16 + offs_2d(32, 16, b_layout))
        with gl.amd.warp_pipeline_stage("wmma", priority=0):
            acc = cdna5.wmma(a, b, acc)
    gl.store(c_ptr + offs_2d(128, 16, wmma_layout), acc)


@gluon.jit
def wmma_w_smem_kernel(a_ptr, b_ptr, c_ptr, N: gl.constexpr):
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=[[1, 0], [2, 0], [4, 0]],
                                                     reg_bases=[], instr_shape=[16, 16, 32])
    a_layout: gl.constexpr = gl.DotOperandLayout(0, wmma_layout, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, wmma_layout, 8)
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    a_blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[8, 4], warps_per_cta=[8, 1],
                                               order=[1, 0])
    b_blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 2], threads_per_warp=[4, 8], warps_per_cta=[8, 1],
                                               order=[1, 0])
    a_smem = gl.allocate_shared_memory(gl.bfloat16, [128, 32], smem_layout,
                                       gl.load(a_ptr + offs_2d(128, 32, a_blocked)))
    b_smem = gl.allocate_shared_memory(gl.bfloat16, [32, 16], smem_layout, gl.load(b_ptr + offs_2d(32, 16, b_blocked)))
    acc = gl.zeros([128, 16], gl.float32, wmma_layout)
    for i in range(N):
        with gl.amd.warp_pipeline_stage("lds", priority=1):
            a = a_smem.load(a_layout)
            b = b_smem.load(b_layout)
        with gl.amd.warp_pipeline_stage("wmma", priority=0):
            acc = cdna5.wmma(a, b, acc)
    gl.store(c_ptr + offs_2d(128, 16, wmma_layout), acc)


@gluon.jit
def wmma_w_valu_kernel(a_ptr, b_ptr, c_ptr, N: gl.constexpr):
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=[[1, 0], [2, 0], [4, 0]],
                                                     reg_bases=[], instr_shape=[16, 16, 32])
    a_layout: gl.constexpr = gl.DotOperandLayout(0, wmma_layout, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, wmma_layout, 8)
    a = gl.load(a_ptr + offs_2d(128, 32, a_layout))
    b = gl.load(b_ptr + offs_2d(32, 16, b_layout))
    s = gl.zeros([128, 16], gl.float32, wmma_layout)
    acc = gl.zeros([128, 16], gl.float32, wmma_layout)
    for i in range(N):
        with gl.amd.warp_pipeline_stage("wmma", priority=0):
            s = cdna5.wmma(a, b, s)
        with gl.amd.warp_pipeline_stage("valu", priority=1):
            acc = acc + gl.exp2(s * 0.5 - gl.max(s, axis=1)[:, None])
    gl.store(c_ptr + offs_2d(128, 16, wmma_layout), acc)


if __name__ == "__main__":
    A = torch.randn(64 * 128 * 32, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(64 * 32 * 16, device="cuda", dtype=torch.bfloat16)
    C = torch.empty(128 * 16, device="cuda")
    for kernel in [wmma_w_gmem_kernel, wmma_w_smem_kernel, wmma_w_valu_kernel]:
        print(f"{kernel.fn.__name__}:")
        compiled = kernel[(1, )](A, B, C, 64, num_warps=8)
        asm = compiled.asm["amdgcn"]
        # The loop of interest is the innermost loop with WMMAs.
        for body in asm.split("Loop Header")[1:]:
            body = body[:body.index("s_cbranch")]
            if "v_wmma" in body:
                break
        order = ""
        for line in body.splitlines():
            line = line.strip()
            for prefix, kind in [("global_load", "G"), ("tensor_load", "T"), ("ds_load", "R"), ("v_wmma", "W"),
                                 ("s_barrier_wait", "|"), ("v_", "V")]:
                if line.startswith(prefix):
                    order += kind
                    break
        # Count the repeated letters, for example "RRR" becomes "3R".
        runs = []
        for kind in order:
            if runs and runs[-1][1] == kind:
                runs[-1][0] += 1
            else:
                runs.append([1, kind])
        stages = " ".join(kind if kind == "|" else f"{n}{kind}" for n, kind in runs).split("| ")[1:]
        for i in range(0, len(stages), 2):
            print("  | " + "| ".join(stages[i:i + 2]).strip())

# %%
# ```
# wmma_w_gmem_kernel:
#   | 8V 1W | 18G
#   | 8V 1W | 1V
# wmma_w_smem_kernel:
#   | 1W | 4R
#   | 1W | 4R
#   ...
# wmma_w_valu_kernel:
#   | 34V | 1W
#   | 34V |
# ```
#
# In the first kernel, a "mem" stage issues 18 global loads, while a "wmma"
# stage runs the WMMA after 8 vector instructions that pack the loaded
# operands. In the second kernel, an "lds" stage reads the operands of the next
# WMMA from the LDS, and the compiler unrolls the loop, so the pattern repeats.
# In the third kernel, the "valu" stage runs 34 vector instructions for the
# softmax-like update, while the other group runs the WMMA.

# %%
# Now we apply warp pipelining to the fp8 x mxfp4 matmul of
# `05-mfma-wmma-scaled.py`. The kernel uses 8 warps, and each K block runs in
# five stages. The first stage reads the first half of the K block from LDS, and
# the second stage runs its WMMAs. The third and fourth stages do the same for
# the second half. The last stage refills the buffer slot that the first stage
# read. So while one group runs a "wmma" stage, the other group reads the LDS or
# copies.
#
# A few rules keep the stages correct. First, waits go between stages. We call
# `tdm.async_wait` before the first stage, outside of any stage. Second, only
# the leading group copies. `tdm.async_wait` only waits for the copies of the
# current warp, and the second group runs one stage behind. If it copied part of
# a tile, the first group could read that part before the copy completes, so all
# copies take `warp_used_hint=0b1111`, and only warps 0-3 copy. Third, the
# buffer slots are constants. We unroll the loop over the 3 buffers, so the
# compiler can tell which slot each stage accesses, and it doesn't add extra
# waits and barriers. The K blocks that are left after the loop run in a tail
# with the same stages, but without the refill. Last, the refill goes into the
# slot that the first stage read. When the leading group refills it, the second
# group runs the fourth stage, so both groups have finished reading the slot.
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
                  BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr):
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

    y_smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0])
    y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], y_smem_layout)
    y_smem.store(acc.to(y_ptr.dtype.element_ty))
    y_desc = tdm.make_tensor_descriptor(y_ptr, [M, N], [stride_ym, 1], [BLOCK_M, BLOCK_N], y_smem_layout)
    y_desc = tdm.update_tensor_descriptor(y_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
    tdm.async_store(y_desc, src=y_smem)
    tdm.async_wait(0)


def matmul(A, W, W_scale, BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, NUM_BUFFERS=3):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    matmul_kernel[grid](
        A, W, W_scale, Y,  #
        M, N, K,  #
        A.stride(0), W.stride(0), W_scale.stride(0), Y.stride(0),  #
        BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, num_warps=8)
    return Y


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 1024), (2048, 1024, 768), (500, 300, 1280)])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul(M, N, K):
    A = t5.random_fp8(M, K)
    W, W_scale, W_ref = t5.random_mxfp4(N, K)
    Y = matmul(A, W, t5.preshuffle_scales_cdna5(W_scale))
    torch.testing.assert_close(A.float() @ W_ref.T, Y.float(), atol=1e-1, rtol=1e-2)


# %%
# We compare the warp pipelined kernel with the kernel of
# `05-mfma-wmma-scaled.py` on an MI455X GPU with M = N = K = 8192.

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

    bench("4 warp", lambda: t5.matmul_cdna5(A, W, W_scale))
    bench("8 warp", lambda: matmul(A, W, W_scale))

# %%
# ```
# 4 warp: 7860 TFLOPS
# 8 warp: 7765 TFLOPS
# ```
