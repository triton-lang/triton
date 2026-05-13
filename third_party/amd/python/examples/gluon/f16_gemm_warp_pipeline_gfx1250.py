import pytest
import torch

import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

# Handle imports for both pytest (module context) and direct execution
try:
    from .gfx1250_utils import static_profile
    from .f16_gemm_common_gfx1250 import (
        create_shared_layouts,
        create_tensor_descriptors,
        issue_loads,
        issue_wmma,
        lds_load,
        issue_wmma_compute,
    )
except ImportError:
    from gfx1250_utils import static_profile
    from f16_gemm_common_gfx1250 import (
        create_shared_layouts,
        create_tensor_descriptors,
        issue_loads,
        issue_wmma,
        lds_load,
        issue_wmma_compute,
    )


@gluon.jit
def gemm_tdm_pipelined_warp_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                                             M, N, K,  #
                                             stride_am, stride_ak,  #
                                             stride_bk, stride_bn,  #
                                             stride_cm, stride_cn,  #
                                             BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                             BLOCK_K: ttgl.constexpr,  #
                                             NUM_BUFFERS: ttgl.constexpr,  #
                                             TRANSPOSE_B: ttgl.constexpr,  #
                                             WARP_BASES: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])

    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, pid_m * BLOCK_M * stride_am, pid_n * BLOCK_N * stride_bn,
                                               stride_am, stride_ak, stride_bn, stride_bk, SHARED_LAYOUT_A,
                                               SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    producer = 0
    consumer = 0
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

    # Prefetch NUM_BUFFERS - 1 tiles; the main loop produces one tile for
    # each tile it consumes, and the epilogue drains the prefetched tail.
    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)

    # Wait for the first prefetch
    ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        with ttgl.amd.warp_pipeline_stage("stage0", priority=1):
            consumer, a, b = lds_load(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                      TRANSPOSE_B)
        # Wait for the last one, either before the loop or a load below.
        ttgl.amd.gfx1250.tdm.async_wait(0)
        with ttgl.amd.warp_pipeline_stage("stage1", priority=0):
            producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)
            accumulator = issue_wmma_compute(a, b, accumulator)

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        with ttgl.amd.warp_pipeline_stage("stage0_epilogue", priority=1):
            consumer, a, b = lds_load(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                      TRANSPOSE_B)
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1 - i) * 2)
        with ttgl.amd.warp_pipeline_stage("stage1_epilogue", priority=0):
            accumulator = issue_wmma_compute(a, b, accumulator)

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask_c)


# ---------------------------------------------------------------------------
# Partial TDM copy variant: only a subset of warps issue TDM copies.
# Cleared warps get pred=0 (hardware no-op), freeing TDM bandwidth.
# ---------------------------------------------------------------------------


@gluon.jit
def issue_loads_predicated(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K: ttgl.constexpr,
                           NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr,
                           TDM_WARP_USED_HINT: ttgl.constexpr):
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, producer * BLOCK_K], a_buffer.index(producer % NUM_BUFFERS),
                                    warp_used_hint=TDM_WARP_USED_HINT)
    if not TRANSPOSE_B:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [producer * BLOCK_K, off_bn], b_buffer.index(producer % NUM_BUFFERS),
                                        warp_used_hint=TDM_WARP_USED_HINT)
    else:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, producer * BLOCK_K], b_buffer.index(producer % NUM_BUFFERS),
                                        warp_used_hint=TDM_WARP_USED_HINT)
    producer += 1
    return producer


@gluon.jit
def gemm_tdm_predicated_pipelined_warp_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                                                        M, N, K,  #
                                                        stride_am, stride_ak,  #
                                                        stride_bk, stride_bn,  #
                                                        stride_cm, stride_cn,  #
                                                        BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                        BLOCK_K: ttgl.constexpr,  #
                                                        NUM_BUFFERS: ttgl.constexpr,  #
                                                        TRANSPOSE_B: ttgl.constexpr,  #
                                                        WARP_BASES: ttgl.constexpr,  #
                                                        TDM_WARP_USED_HINT: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])

    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, pid_m * BLOCK_M * stride_am, pid_n * BLOCK_N * stride_bn,
                                               stride_am, stride_ak, stride_bn, stride_bk, SHARED_LAYOUT_A,
                                               SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    producer = 0
    consumer = 0
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

    for _ in ttgl.static_range(2):
        producer = issue_loads_predicated(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                          TRANSPOSE_B, TDM_WARP_USED_HINT)

    ttgl.amd.gfx1250.tdm.async_wait(1 * 2)
    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        with ttgl.amd.warp_pipeline_stage("stage0", priority=1):
            consumer, a, b = lds_load(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                      TRANSPOSE_B)
            producer = issue_loads_predicated(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                              TRANSPOSE_B, TDM_WARP_USED_HINT)
        with ttgl.amd.warp_pipeline_stage("stage1", priority=0):
            accumulator = issue_wmma_compute(a, b, accumulator)
        ttgl.amd.gfx1250.tdm.async_wait(2)

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1 - i) * 2)
        consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                           accumulator, (NUM_BUFFERS - 2 - i) * 2, NUM_BUFFERS, TRANSPOSE_B)

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask_c)


# ---------------------------------------------------------------------------
# KernelB variant: loads+TDM in stage0, dot-only in stage1 (num_buffers>=3).
#
# Schedule (num_buffers=4):
#          w0      w1               w2             w3
# g0 g1 g2 / l0 g3 / d0    / l1 g4 / d1    / l2 g5 / d2    / l3
# g0 g1 g2         / l0 g3 / d0    / l1 g4 / d1    / l2 g5 / d2    /
#          w0              w1              w2              w3
#
# The write index is derived from `phase` (the read counter) so that the write
# and the read share the same SSA base, letting the membar dynamic-index
# disjointness analysis prove the two slots never overlap within an iteration.
# ---------------------------------------------------------------------------


@gluon.jit
def gemm_tdm_pipelined_warp_pipelined_kernelB(a_ptr, b_ptr, c_ptr,  #
                                              M, N, K,  #
                                              stride_am, stride_ak,  #
                                              stride_bk, stride_bn,  #
                                              stride_cm, stride_cn,  #
                                              BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                              BLOCK_K: ttgl.constexpr,  #
                                              NUM_BUFFERS: ttgl.constexpr,  #
                                              TRANSPOSE_B: ttgl.constexpr,  #
                                              WARP_BASES: ttgl.constexpr,  #
                                              PRIO_SWAP: ttgl.constexpr = False):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 3, "kernelB requires NUM_BUFFERS >= 3")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])

    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, pid_m * BLOCK_M * stride_am, pid_n * BLOCK_N * stride_bn,
                                               stride_am, stride_ak, stride_bn, stride_bk, SHARED_LAYOUT_A,
                                               SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    phase = 0
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

    producer = 0
    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)

    ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3) * 2)
    _p0: ttgl.constexpr = 1 if PRIO_SWAP else 0
    _p1: ttgl.constexpr = 0 if PRIO_SWAP else 1

    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        with ttgl.amd.warp_pipeline_stage("stage0", priority=_p0):
            write_phase = phase + (NUM_BUFFERS - 1)
            phase, a, b = lds_load(phase, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                   TRANSPOSE_B)
            issue_loads(write_phase, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3) * 2)
        with ttgl.amd.warp_pipeline_stage("stage1", priority=_p1):
            accumulator = issue_wmma_compute(a, b, accumulator)

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        with ttgl.amd.warp_pipeline_stage("stage0", priority=_p0):
            phase, a, b = lds_load(phase, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                   TRANSPOSE_B)
        ttgl.amd.gfx1250.tdm.async_wait(0)
        with ttgl.amd.warp_pipeline_stage("stage1", priority=_p1):
            accumulator = issue_wmma_compute(a, b, accumulator)

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask_c)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(256, 256, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [3])
@pytest.mark.parametrize("TRANSPOSE_B", [True])
@pytest.mark.parametrize("M,N,K", [(2048, 2048, 2048)])
@pytest.mark.parametrize("DUMP", [False])
@pytest.mark.parametrize("USE_KERNEL_B", [False])
def test_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, DUMP, USE_KERNEL_B):
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    torch.manual_seed(42)

    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    if TRANSPOSE_B:
        b = b.T.contiguous()
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = c.cuda()

    # warpsPerCTA = [4, 2]
    num_warps = 8
    WARP_BASES = [(0, 1), (1, 0), (2, 0)]

    warp_bases = tuple(WARP_BASES)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    kernel_fn = (gemm_tdm_pipelined_warp_pipelined_kernelB
                 if USE_KERNEL_B else gemm_tdm_pipelined_warp_pipelined_kernel)
    kernel = kernel_fn[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, WARP_BASES=warp_bases,  #
        num_warps=num_warps, waves_per_eu=num_warps // 4)
    static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ (b.to(torch.float32) if not TRANSPOSE_B else b.T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)
    if DUMP:
        print("triton")
        print(c_triton)
        print("torch")
        print(c_torch)
        print("Done.")


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(256, 256, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [3])
@pytest.mark.parametrize("TRANSPOSE_B", [True])
@pytest.mark.parametrize("M,N,K", [(2048, 2048, 2048)])
@pytest.mark.parametrize("DUMP", [False])
def test_runtime_gemm_tdm_predicated_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, DUMP):
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    torch.manual_seed(42)

    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    if TRANSPOSE_B:
        b = b.T.contiguous()
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = c.cuda()

    # warpsPerCTA = [4, 2]
    num_warps = 8
    WARP_BASES = [(0, 1), (1, 0), (2, 0)]

    # 4-warp partial TDM copy: warps 0-3 issue, warps 4-7 are no-ops.
    tdm_warp_used_hint = 0b00001111

    warp_bases = tuple(WARP_BASES)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    kernel = gemm_tdm_predicated_pipelined_warp_pipelined_kernel[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, WARP_BASES=warp_bases,  #
        TDM_WARP_USED_HINT=tdm_warp_used_hint, num_warps=num_warps, waves_per_eu=num_warps // 4)
    static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ (b.to(torch.float32) if not TRANSPOSE_B else b.T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)
    if DUMP:
        print("triton")
        print(c_triton)
        print("torch")
        print(c_torch)
        print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=256, help='problem M size')
    parser.add_argument("-N", type=int, default=256, help='problem N size')
    parser.add_argument("-K", type=int, default=1024, help='problem K size')
    parser.add_argument("--num-buffers", type=int, choices=[2, 3, 4], default=3, help='num shared memory buffers')
    parser.add_argument("--4warp-tdm", action="store_true", dest="four_warp_tdm",
                        help="Use 4-warp partial TDM copy (warps 4-7 skip TDM copies)")
    parser.add_argument("--dump", action="store_true", help="Print out result/golden tensors")
    parser.add_argument("--kernelB", action="store_true", help="Use the kernelB variant")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    NUM_BUFFERS = args.num_buffers
    NUM_WARPS = 8
    TRANSPOSE_B = True
    DUMP = args.dump
    USE_KERNEL_B = args.kernelB
    print(
        f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {TRANSPOSE_B=}, {NUM_WARPS=}, {NUM_BUFFERS=}, {USE_KERNEL_B=}"
    )

    if args.four_warp_tdm:
        test_runtime_gemm_tdm_predicated_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, DUMP)
    else:
        test_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, DUMP,
                                        USE_KERNEL_B)
