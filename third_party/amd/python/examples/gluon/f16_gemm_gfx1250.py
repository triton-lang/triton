# ruff: noqa: E402
import hip

# Needed for internal dev flow for now; will remove later
hip.hip.hipInit(0)

import pytest
import torch

import triton
import math
from triton.experimental import gluon
from triton.language.core import _aggregate as aggregate
import triton.experimental.gluon.language as ttgl
from triton._C.libtriton.gluon_ir import make_cga_layout

# Handle imports for both pytest (module context) and direct execution
try:
    from .gfx1250_utils import static_profile
    from .f16_gemm_common_gfx1250 import (
        create_shared_layouts,
        create_tensor_descriptors,
        issue_loads,
        issue_l2_prefetches,
        issue_l2_prefetches_prologue,
        issue_wmma,
        lds_subtile_load,
        TileScheduler,
    )
except ImportError:
    from gfx1250_utils import static_profile
    from f16_gemm_common_gfx1250 import (
        create_shared_layouts,
        create_tensor_descriptors,
        issue_loads,
        issue_l2_prefetches,
        issue_l2_prefetches_prologue,
        issue_wmma,
        lds_subtile_load,
        TileScheduler,
    )


@gluon.jit
def persistent_gemm_tdm_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                                         M, N, K,  #
                                         stride_am, stride_ak,  #
                                         stride_bk, stride_bn,  #
                                         stride_cm, stride_cn,  #
                                         BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,  #
                                         NUM_BUFFERS: ttgl.constexpr,  #
                                         TRANSPOSE_B: ttgl.constexpr,  #
                                         SHARED_LAYOUT_A: ttgl.constexpr,  #
                                         SHARED_LAYOUT_B: ttgl.constexpr,  #
                                         ACCUMULATOR_LAYOUT: ttgl.constexpr,  #
                                         OPERAND_LAYOUT_A: ttgl.constexpr,  #
                                         OPERAND_LAYOUT_B: ttgl.constexpr, L2_PREFETCH_DISTANCE: ttgl.constexpr):

    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, 0, 0, stride_am, stride_ak, stride_bn, stride_bk,
                                               SHARED_LAYOUT_A, SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                               TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES=0)
    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()

    pid = scheduler.apply_chiplet_transform_chunked(pid, num_sms, num_xcds=8, chunk_size=2)

    for tile_idx in range(pid, scheduler.get_num_tiles(), num_sms):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M=8)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        producer = 0
        consumer = 0
        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=ACCUMULATOR_LAYOUT)

        issue_l2_prefetches_prologue(L2_PREFETCH_DISTANCE, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                                     NUM_BUFFERS, TRANSPOSE_B)

        for _ in ttgl.static_range(NUM_BUFFERS - 1):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)

        for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)
            # We prefetch distance - 1 iterations ahead because producer is already incremented by 1
            issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                                TRANSPOSE_B)
            consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                               accumulator, (NUM_BUFFERS - 1) * 2, NUM_BUFFERS, TRANSPOSE_B)

        for i in ttgl.static_range(NUM_BUFFERS - 1):
            consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                               accumulator, (NUM_BUFFERS - 2 - i) * 2, NUM_BUFFERS, TRANSPOSE_B)

        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, ACCUMULATOR_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, ACCUMULATOR_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


@gluon.jit
def persistent_gemm_tdm_pipelined_lds_prefetch_kernel(a_ptr, b_ptr, c_ptr,  #
                                                      M, N, K,  #
                                                      stride_am, stride_ak,  #
                                                      stride_bk, stride_bn,  #
                                                      stride_cm, stride_cn,  #
                                                      BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                      BLOCK_K: ttgl.constexpr,  #
                                                      NUM_BUFFERS: ttgl.constexpr,  #
                                                      TRANSPOSE_B: ttgl.constexpr,  #
                                                      SHARED_LAYOUT_A: ttgl.constexpr,  #
                                                      SHARED_LAYOUT_B: ttgl.constexpr,  #
                                                      ACCUMULATOR_LAYOUT: ttgl.constexpr,  #
                                                      OPERAND_LAYOUT_A: ttgl.constexpr,  #
                                                      OPERAND_LAYOUT_B: ttgl.constexpr,
                                                      L2_PREFETCH_DISTANCE: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, 0, 0, stride_am, stride_ak, stride_bn, stride_bk,
                                               SHARED_LAYOUT_A, SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                               TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES=0)
    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_tiles = scheduler.get_num_tiles()

    for tile_idx in range(pid, num_tiles, num_sms):
        # Calculate current tile coordinates
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M=8)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        # Calculate next tile coordinates for prefetching
        pid_m_next, pid_n_next = scheduler.get_swizzled_tile_coords(tile_idx + num_sms, GROUP_SIZE_M=8)
        off_am_next = pid_m_next * BLOCK_M
        off_bn_next = pid_n_next * BLOCK_N

        producer = 0

        issue_l2_prefetches_prologue(L2_PREFETCH_DISTANCE, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                                     NUM_BUFFERS, TRANSPOSE_B)
        # Initial prefetch for first NUM_BUFFERS - 1 tiles
        for i in ttgl.static_range(NUM_BUFFERS - 1):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)

        consumer = 0
        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=ACCUMULATOR_LAYOUT)

        for _ in range(0, ttgl.cdiv(K, BLOCK_K) - NUM_BUFFERS):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)
            # We prefetch distance - 1 iterations ahead because producer is already incremented by 1
            issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                                TRANSPOSE_B)
            consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                               accumulator, (NUM_BUFFERS - 1) * 2, NUM_BUFFERS, TRANSPOSE_B)

        # Last load for current tile and L2 prefetch for next tile
        producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                               TRANSPOSE_B)
        issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc, off_am_next, off_bn_next, BLOCK_K,
                            TRANSPOSE_B)

        for i in ttgl.static_range(NUM_BUFFERS):
            consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                               accumulator, (NUM_BUFFERS - 1 - i) * 2, NUM_BUFFERS, TRANSPOSE_B)

        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, ACCUMULATOR_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, ACCUMULATOR_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


def _build_gemm_layouts(BLOCK_M, BLOCK_N, BLOCK_K, cga_layout_a, cga_layout_b, cga_layout_c, WARP_BASES, TRANSPOSE_B):
    # If TRANSPOSE_B we need to transpose each basis vector of the CGALayout for the
    # shared allocation because the permute will transpose the basis vectors before we
    # load them for wmmas.
    if TRANSPOSE_B:
        # Transpose each basis vector: [a, b] -> [b, a]
        cga_layout_b_transposed = tuple([tuple([row[1], row[0]]) for row in cga_layout_b])
    else:
        cga_layout_b_transposed = cga_layout_b

    SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                [1, 0], cga_layout_a)
    if not TRANSPOSE_B:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K, BLOCK_N],
                                                                                    [1, 0], cga_layout_b_transposed)
    else:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K],
                                                                                    [1, 0], cga_layout_b_transposed)

    WMMA_LAYOUT_A = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32], cga_layout_a)
    WMMA_LAYOUT_B = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32], cga_layout_b)
    ACCUMULATOR_LAYOUT = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32], cga_layout_c)
    OPERAND_LAYOUT_A = ttgl.DotOperandLayout(0, WMMA_LAYOUT_A, 8)
    OPERAND_LAYOUT_B = ttgl.DotOperandLayout(1, WMMA_LAYOUT_B, 8)

    return SHARED_LAYOUT_A, SHARED_LAYOUT_B, ACCUMULATOR_LAYOUT, OPERAND_LAYOUT_A, OPERAND_LAYOUT_B


@gluon.jit
def gemm_tdm_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                              M, N, K,  #
                              stride_am, stride_ak,  #
                              stride_bk, stride_bn,  #
                              stride_cm, stride_cn,  #
                              BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,  #
                              NUM_BUFFERS: ttgl.constexpr,  #
                              TRANSPOSE_B: ttgl.constexpr,  #
                              SHARED_LAYOUT_A: ttgl.constexpr, SHARED_LAYOUT_B: ttgl.constexpr,
                              ACCUMULATOR_LAYOUT: ttgl.constexpr, OPERAND_LAYOUT_A: ttgl.constexpr,
                              OPERAND_LAYOUT_B: ttgl.constexpr, L2_PREFETCH_DISTANCE: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

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
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=ACCUMULATOR_LAYOUT)

    # Prefetching buffers we load in the prologue does not make sense
    issue_l2_prefetches_prologue(L2_PREFETCH_DISTANCE, producer, a_desc, b_desc, 0, 0, BLOCK_K, NUM_BUFFERS,
                                 TRANSPOSE_B)

    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)

    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)
        # We prefetch distance - 1 iterations ahead because producer is already incremented by 1
        issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc, 0, 0, BLOCK_K, TRANSPOSE_B)
        consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                           accumulator, (NUM_BUFFERS - 1) * 2, NUM_BUFFERS, TRANSPOSE_B)

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                           accumulator, (NUM_BUFFERS - 2 - i) * 2, NUM_BUFFERS, TRANSPOSE_B)

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, ACCUMULATOR_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, ACCUMULATOR_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


@gluon.jit
def gemm_tdm_pipelined_single_warp_per_simd_schedule_kernel(a_ptr, b_ptr, c_ptr,  #
                                                            M, N, K,  #
                                                            stride_am, stride_ak,  #
                                                            stride_bk, stride_bn,  #
                                                            stride_cm, stride_cn,  #
                                                            BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                            BLOCK_K: ttgl.constexpr,  #
                                                            NUM_BUFFERS: ttgl.constexpr,  #
                                                            TRANSPOSE_B: ttgl.constexpr,  #
                                                            WARP_BASES: ttgl.constexpr,
                                                            L2_PREFETCH_DISTANCE: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    NUM_SUBTILES: ttgl.constexpr = 4
    SUBTILE_LEN: ttgl.constexpr = BLOCK_K // NUM_SUBTILES
    ttgl.static_assert(SUBTILE_LEN == 32, "Subtile length must match the kdim of the wmma instruction")

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

    issue_l2_prefetches_prologue(L2_PREFETCH_DISTANCE, producer, a_desc, b_desc, 0, 0, BLOCK_K, NUM_BUFFERS,
                                 TRANSPOSE_B)

    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)

    ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
    # LDS load SubIteration0
    a0, b0 = lds_subtile_load(consumer, 0, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                              TRANSPOSE_B, SUBTILE_LEN)

    loop_ub = ttgl.cdiv(K, BLOCK_K)
    epilogue_lb = loop_ub - (NUM_BUFFERS - 1)
    for i in range(0, loop_ub):
        # SubIteration0
        # LDS load SubIteration1
        a1, b1 = lds_subtile_load(consumer, SUBTILE_LEN, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                  NUM_BUFFERS, TRANSPOSE_B, SUBTILE_LEN)
        # WMMA Subtile0
        accumulator = ttgl.amd.gfx1250.wmma(a0, b0, accumulator)

        # SubIteration1
        # TDM load for next tile
        # If we are in epilogue, we have already issued our tile loads
        producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B,
                               pred=i < epilogue_lb)

        # We prefetch distance - 1 iterations ahead because producer is already incremented by 1
        issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc, 0, 0, BLOCK_K, TRANSPOSE_B)

        # LDS load SubIteration2
        a2, b2 = lds_subtile_load(consumer, 2 * SUBTILE_LEN, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                  NUM_BUFFERS, TRANSPOSE_B, SUBTILE_LEN)
        # WMMA Subtile1
        accumulator = ttgl.amd.gfx1250.wmma(a1, b1, accumulator)

        # SubIteration2
        # LDS load SubIteration3
        a3, b3 = lds_subtile_load(consumer, 3 * SUBTILE_LEN, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                  NUM_BUFFERS, TRANSPOSE_B, SUBTILE_LEN)
        # WMMA Subtile2
        accumulator = ttgl.amd.gfx1250.wmma(a2, b2, accumulator)

        # SubIteration3
        consumer += 1
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
        # LDS load SubIteration0 for next tile
        a0, b0 = lds_subtile_load(consumer, 0, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                  TRANSPOSE_B, SUBTILE_LEN)
        accumulator = ttgl.amd.gfx1250.wmma(a3, b3, accumulator)

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


def _run_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, PERSISTENT, PREFETCH,
                                    L2_PREFETCH_DISTANCE, M, N, K, num_warps, ctas_per_cga):
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    num_ctas = ctas_per_cga[0] * ctas_per_cga[1]
    if num_ctas > 1 and PERSISTENT:
        pytest.skip("Skip tests with multiple CTAs and persistent or prefetch")

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

    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    cga_layout_a = make_cga_layout(ctas_per_cga, [ctas_per_cga[0], 1], [0, 1])
    cga_layout_b = make_cga_layout(ctas_per_cga, [1, ctas_per_cga[1]], [0, 1])
    cga_layout_c = make_cga_layout(ctas_per_cga, [ctas_per_cga[0], ctas_per_cga[1]], [0, 1])

    # Build all layouts in wrapper function (outside kernel)
    SHARED_LAYOUT_A, SHARED_LAYOUT_B, ACCUMULATOR_LAYOUT, OPERAND_LAYOUT_A, OPERAND_LAYOUT_B = _build_gemm_layouts(
        BLOCK_M, BLOCK_N, BLOCK_K, cga_layout_a, cga_layout_b, cga_layout_c, warp_bases, TRANSPOSE_B)

    if not PERSISTENT:

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
        kernel = gemm_tdm_pipelined_kernel[grid](
            a_device, b_device, c_device,  #
            M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
            NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B,  #
            SHARED_LAYOUT_A=SHARED_LAYOUT_A, SHARED_LAYOUT_B=SHARED_LAYOUT_B, ACCUMULATOR_LAYOUT=ACCUMULATOR_LAYOUT,
            OPERAND_LAYOUT_A=OPERAND_LAYOUT_A, OPERAND_LAYOUT_B=OPERAND_LAYOUT_B, num_warps=num_warps,
            waves_per_eu=num_warps // 4, num_ctas=num_ctas, L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE)
        static_profile(kernel)
    else:
        # num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        # NOTE: Explicitly set num_sms to small number to ensure that each CU will compute multiple tiles.
        num_sms = 8
        grid = (min(num_sms, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), 1)
        if PREFETCH:
            kernel = persistent_gemm_tdm_pipelined_lds_prefetch_kernel[grid](
                a_device, b_device, c_device,  #
                M, N, K,  #
                stride_am, stride_ak,  #
                stride_bk, stride_bn,  #
                stride_cm, stride_cn,  #
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
                NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B,  #
                SHARED_LAYOUT_A=SHARED_LAYOUT_A, SHARED_LAYOUT_B=SHARED_LAYOUT_B, ACCUMULATOR_LAYOUT=ACCUMULATOR_LAYOUT,
                OPERAND_LAYOUT_A=OPERAND_LAYOUT_A, OPERAND_LAYOUT_B=OPERAND_LAYOUT_B,  #
                num_warps=num_warps, num_ctas=num_ctas, waves_per_eu=num_warps // 4,
                L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE)
            static_profile(kernel)
        else:
            kernel = persistent_gemm_tdm_pipelined_kernel[grid](
                a_device, b_device, c_device,  #
                M, N, K,  #
                stride_am, stride_ak,  #
                stride_bk, stride_bn,  #
                stride_cm, stride_cn,  #
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
                NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B,  #
                SHARED_LAYOUT_A=SHARED_LAYOUT_A, SHARED_LAYOUT_B=SHARED_LAYOUT_B, ACCUMULATOR_LAYOUT=ACCUMULATOR_LAYOUT,
                OPERAND_LAYOUT_A=OPERAND_LAYOUT_A, OPERAND_LAYOUT_B=OPERAND_LAYOUT_B,  #
                num_warps=num_warps, num_ctas=num_ctas, waves_per_eu=num_warps // 4,
                L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE)
            static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ (b.to(torch.float32) if not TRANSPOSE_B else b.T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)


def _build_multi_cta_gemm_cases():
    """
    Build multi-CTA GEMM configs, it scales the problem size and blocks dims by
    ctas_per_cga so each CTA works on BLOCK_M/BLOCK_N sized tile.
    """

    base_shapes = [(256, 256, 512), (250, 250, 510)]
    base_blocks = [(32, 32, 64)]
    ctas_per_cga_list = [[2, 1], [4, 2], [4, 4]]
    configs = []
    for M, N, K in base_shapes:
        for BLOCK_M, BLOCK_N, BLOCK_K in base_blocks:
            for ctas in ctas_per_cga_list:
                configs.append((M * ctas[0], N * ctas[1], K, BLOCK_M * ctas[0], BLOCK_N * ctas[1], BLOCK_K, ctas))
    return configs


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 3, 4])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("PERSISTENT", [False, True])
@pytest.mark.parametrize("PREFETCH", [False, True])
@pytest.mark.parametrize("L2_PREFETCH_DISTANCE", [0, 2])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (250, 250, 510)])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_runtime_gemm_tdm_pipelined_single_cta(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, PERSISTENT,
                                               PREFETCH, L2_PREFETCH_DISTANCE, M, N, K, num_warps):
    _run_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, PERSISTENT, PREFETCH,
                                    L2_PREFETCH_DISTANCE, M, N, K, num_warps, [1, 1])


@pytest.mark.parametrize("NUM_BUFFERS", [2, 3, 4])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("PERSISTENT", [False])
@pytest.mark.parametrize("PREFETCH", [False, True])
@pytest.mark.parametrize("L2_PREFETCH_DISTANCE", [0, 2])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("M,N,K,BLOCK_M,BLOCK_N,BLOCK_K,ctas_per_cga", _build_multi_cta_gemm_cases())
def test_runtime_gemm_tdm_pipelined_multi_cta(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, ctas_per_cga, NUM_BUFFERS,
                                              TRANSPOSE_B, PERSISTENT, PREFETCH, L2_PREFETCH_DISTANCE, num_warps):
    _run_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, PERSISTENT, PREFETCH,
                                    L2_PREFETCH_DISTANCE, M, N, K, num_warps, ctas_per_cga)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32)])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("L2_PREFETCH_DISTANCE", [0, 2])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (250, 250, 510)])
def test_runtime_gemm_tdm_pipelined_single_warp_per_simd_schedule(BLOCK_M, BLOCK_N, NUM_BUFFERS, TRANSPOSE_B,
                                                                  L2_PREFETCH_DISTANCE, M, N, K):
    num_warps = 4
    BLOCK_K = 128  # 4 subtiles * 32 (wmma kdim)

    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

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
    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    kernel = gemm_tdm_pipelined_single_warp_per_simd_schedule_kernel[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B,  #
        num_warps=num_warps, WARP_BASES=tuple(warp_bases), waves_per_eu=num_warps // 4,
        L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE)
    static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ (b.to(torch.float32) if not TRANSPOSE_B else b.T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)


# Helper class for passing arguments around partitions.
@aggregate
class PartitionArgs:
    a_desc: ttgl.amd.gfx1250.tdm.tensor_descriptor
    b_desc: ttgl.amd.gfx1250.tdm.tensor_descriptor
    a_buffer: ttgl.shared_memory_descriptor
    b_buffer: ttgl.shared_memory_descriptor
    empty_bars: ttgl.shared_memory_descriptor
    ready_bars: ttgl.shared_memory_descriptor
    BLOCK_K: ttgl.constexpr
    NUM_BUFFERS: ttgl.constexpr
    TRANSPOSE_B: ttgl.constexpr
    WMMA_LAYOUT: ttgl.constexpr
    c_dtype: ttgl.constexpr  # TODO: Should be able to get this from c_ptr.type.element_ty in consumer_partition

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, a_buffer, b_buffer, empty_bars, ready_bars, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B,
                 WMMA_LAYOUT, c_dtype):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.a_buffer = a_buffer
        self.b_buffer = b_buffer
        self.empty_bars = empty_bars
        self.ready_bars = ready_bars
        self.BLOCK_K = ttgl.constexpr(BLOCK_K)
        self.NUM_BUFFERS = ttgl.constexpr(NUM_BUFFERS)
        self.TRANSPOSE_B = ttgl.constexpr(TRANSPOSE_B)
        self.WMMA_LAYOUT = ttgl.constexpr(WMMA_LAYOUT)
        self.c_dtype = ttgl.constexpr(c_dtype)


# Helper class for passing arguments around persistent warp-specialization partitions.
@aggregate
class PersistentPartitionArgs:
    a_desc: ttgl.amd.gfx1250.tdm.tensor_descriptor
    b_desc: ttgl.amd.gfx1250.tdm.tensor_descriptor
    c_desc: ttgl.amd.gfx1250.tdm.tensor_descriptor
    a_buffer: ttgl.shared_memory_descriptor
    b_buffer: ttgl.shared_memory_descriptor
    acc_buffer: ttgl.shared_memory_descriptor
    load_empty_bars: ttgl.shared_memory_descriptor
    load_ready_bars: ttgl.shared_memory_descriptor
    acc_empty_bars: ttgl.shared_memory_descriptor
    acc_ready_bars: ttgl.shared_memory_descriptor
    BLOCK_K: ttgl.constexpr
    NUM_BUFFERS: ttgl.constexpr
    NUM_ACC_BUFFERS: ttgl.constexpr
    TRANSPOSE_B: ttgl.constexpr
    WMMA_LAYOUT: ttgl.constexpr
    c_dtype: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, a_buffer, b_buffer, acc_buffer, load_empty_bars, load_ready_bars,
                 acc_empty_bars, acc_ready_bars, BLOCK_K, NUM_BUFFERS, NUM_ACC_BUFFERS, TRANSPOSE_B, WMMA_LAYOUT,
                 c_dtype):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_buffer = a_buffer
        self.b_buffer = b_buffer
        self.acc_buffer = acc_buffer
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.BLOCK_K = ttgl.constexpr(BLOCK_K)
        self.NUM_BUFFERS = ttgl.constexpr(NUM_BUFFERS)
        self.NUM_ACC_BUFFERS = ttgl.constexpr(NUM_ACC_BUFFERS)
        self.TRANSPOSE_B = ttgl.constexpr(TRANSPOSE_B)
        self.WMMA_LAYOUT = ttgl.constexpr(WMMA_LAYOUT)
        self.c_dtype = ttgl.constexpr(c_dtype)


# Helper class for passing arguments around persistent warp-specialization partitions (subtiled variant).
@aggregate
class PersistentPartitionSubtiledArgs:
    a_desc: ttgl.amd.gfx1250.tdm.tensor_descriptor
    b_desc: ttgl.amd.gfx1250.tdm.tensor_descriptor
    c_desc: ttgl.amd.gfx1250.tdm.tensor_descriptor
    a_buffer: ttgl.shared_memory_descriptor
    b_buffer: ttgl.shared_memory_descriptor
    acc_buffer: ttgl.shared_memory_descriptor
    load_empty_bars: ttgl.shared_memory_descriptor
    load_ready_bars: ttgl.shared_memory_descriptor
    acc_empty_bars: ttgl.shared_memory_descriptor
    acc_ready_bars: ttgl.shared_memory_descriptor
    BLOCK_K: ttgl.constexpr
    NUM_BUFFERS: ttgl.constexpr
    NUM_ACC_BUFFERS: ttgl.constexpr
    TRANSPOSE_B: ttgl.constexpr
    WMMA_LAYOUT: ttgl.constexpr
    c_dtype: ttgl.constexpr
    NUM_QUADS: ttgl.constexpr
    NUM_QUADS_M: ttgl.constexpr
    NUM_QUADS_N: ttgl.constexpr
    QUADRANT_M: ttgl.constexpr
    QUADRANT_N: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, a_buffer, b_buffer, acc_buffer, load_empty_bars, load_ready_bars,
                 acc_empty_bars, acc_ready_bars, BLOCK_K, NUM_BUFFERS, NUM_ACC_BUFFERS, TRANSPOSE_B, WMMA_LAYOUT,
                 c_dtype, NUM_QUADS, NUM_QUADS_M, NUM_QUADS_N, QUADRANT_M, QUADRANT_N):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_buffer = a_buffer
        self.b_buffer = b_buffer
        self.acc_buffer = acc_buffer
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.BLOCK_K = ttgl.constexpr(BLOCK_K)
        self.NUM_BUFFERS = ttgl.constexpr(NUM_BUFFERS)
        self.NUM_ACC_BUFFERS = ttgl.constexpr(NUM_ACC_BUFFERS)
        self.TRANSPOSE_B = ttgl.constexpr(TRANSPOSE_B)
        self.WMMA_LAYOUT = ttgl.constexpr(WMMA_LAYOUT)
        self.c_dtype = ttgl.constexpr(c_dtype)
        self.NUM_QUADS = ttgl.constexpr(NUM_QUADS)
        self.NUM_QUADS_M = ttgl.constexpr(NUM_QUADS_M)
        self.NUM_QUADS_N = ttgl.constexpr(NUM_QUADS_N)
        self.QUADRANT_M = ttgl.constexpr(QUADRANT_M)
        self.QUADRANT_N = ttgl.constexpr(QUADRANT_N)


@aggregate
class PhaseCounter:
    """Tracks iteration count and computes phase."""
    iteration: ttgl.tensor
    num_barriers: ttgl.constexpr

    @gluon.constexpr_function
    def __init__(self, iteration, num_barriers):
        self.iteration = iteration
        self.num_barriers = ttgl.constexpr(num_barriers)

    @gluon.jit
    def create(iteration, num_barriers: ttgl.constexpr):
        """Creates a counter starting at a specific iteration."""
        return PhaseCounter(ttgl.to_tensor(iteration), num_barriers)

    @gluon.jit
    def phase(self):
        """Computes phase parity (0 for even, 1 for odd)."""
        return (self.iteration // self.num_barriers) & 1

    @gluon.must_use_result
    @gluon.jit
    def next(self):
        """Advances to next iteration."""
        return PhaseCounter(self.iteration + 1, self.num_barriers)


@gluon.jit
def producer_partition(args):
    """Producer partition: Issues TDM async loads for A and B matrices."""
    K = args.a_desc.shape[1]

    num_k_tiles = ttgl.cdiv(K, args.BLOCK_K)

    off_am = 0
    off_bn = 0

    # Assume phase 0 is already completed as the buffers are initially empty; start from phase 1
    empty_phase_counter = PhaseCounter.create(args.NUM_BUFFERS, args.NUM_BUFFERS)

    for k_tile_idx in range(num_k_tiles):
        k_offset = k_tile_idx * args.BLOCK_K
        buffer_idx = k_tile_idx % args.NUM_BUFFERS

        empty_bar = args.empty_bars.index(buffer_idx)
        ready_bar = args.ready_bars.index(buffer_idx)
        # Wait for the buffers to be consumed before loading
        ttgl.amd.gfx1250.mbarrier.wait(empty_bar, empty_phase_counter.phase())

        # Only attach mbarrier to the last load so we signal once after both loads complete
        ttgl.amd.gfx1250.tdm.async_load(args.a_desc, [off_am, k_offset], args.a_buffer.index(buffer_idx))
        if args.TRANSPOSE_B:
            ttgl.amd.gfx1250.tdm.async_load(args.b_desc, [off_bn, k_offset], args.b_buffer.index(buffer_idx),
                                            mbarrier=ready_bar)
        else:
            ttgl.amd.gfx1250.tdm.async_load(args.b_desc, [k_offset, off_bn], args.b_buffer.index(buffer_idx),
                                            mbarrier=ready_bar)

        empty_phase_counter = empty_phase_counter.next()


@gluon.jit
def consumer_partition(args, c_ptr, M, N, stride_cm, stride_cn, pid_m, pid_n):
    """Consumer partition: Waits for loaded data, performs WMMA operations, and stores results."""
    K = args.a_desc.shape[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, args.WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, args.WMMA_LAYOUT, 8)

    BLOCK_M: ttgl.constexpr = args.a_desc.block_shape[0]
    BLOCK_N: ttgl.constexpr = args.b_desc.block_shape[0] if args.TRANSPOSE_B else args.b_desc.block_shape[1]

    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=args.c_dtype, layout=args.WMMA_LAYOUT)

    num_k_tiles = ttgl.cdiv(K, args.BLOCK_K)

    ready_phase_counter = PhaseCounter.create(0, args.NUM_BUFFERS)

    for k_tile_idx in range(num_k_tiles):
        buffer_idx = k_tile_idx % args.NUM_BUFFERS
        ready_bar = args.ready_bars.index(buffer_idx)
        empty_bar = args.empty_bars.index(buffer_idx)

        # Wait for the buffers to be filled by the producer
        ttgl.amd.gfx1250.mbarrier.wait(ready_bar, ready_phase_counter.phase())

        a = args.a_buffer.index(buffer_idx).load(layout=OPERAND_LAYOUT_A)
        if args.TRANSPOSE_B:
            b = args.b_buffer.index(buffer_idx).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)
        else:
            b = args.b_buffer.index(buffer_idx).load(layout=OPERAND_LAYOUT_B)

        accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)

        # Signal that we're done with these buffers (producer can reuse them)
        ttgl.amd.gfx1250.mbarrier.arrive(empty_bar, count=1)

        ready_phase_counter = ready_phase_counter.next()

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, args.WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, args.WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask_c)


@gluon.jit
def gemm_tdm_warp_specialized_kernel(a_ptr, b_ptr, c_ptr,  #
                                     M, N, K,  #
                                     stride_am, stride_ak,  #
                                     stride_bk, stride_bn,  #
                                     stride_cm, stride_cn,  #
                                     BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,  #
                                     NUM_BUFFERS: ttgl.constexpr,  #
                                     TRANSPOSE_B: ttgl.constexpr,  #
                                     NUM_WARPS: ttgl.constexpr,  #
                                     WARP_BASES: ttgl.constexpr):
    """Warp specialized GEMM kernel with TDM pipelining."""
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    PRODUCER_WARPS: ttgl.constexpr = NUM_WARPS // 2
    CONSUMER_WARPS: ttgl.constexpr = NUM_WARPS // 2
    WARP_SIZE: ttgl.constexpr = 32

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])
    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, pid_m * BLOCK_M * stride_am, pid_n * BLOCK_N * stride_bn,
                                               stride_am, stride_ak, stride_bn, stride_bk, SHARED_LAYOUT_A,
                                               SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    empty_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_BUFFERS, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
    ready_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_BUFFERS, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())

    # Initialize mbarriers
    # empty_bars: signals when consumer is done with buffers
    # ready_bars: signals when producer has filled buffers
    for i in ttgl.static_range(NUM_BUFFERS):
        # empty_bars: arrive on barrier once per thread, so use consumer thread count
        ttgl.amd.gfx1250.mbarrier.init(empty_bars.index(i), count=CONSUMER_WARPS * WARP_SIZE)
        # ready_bars: TDM arrives on barrier once per warp, so use producer warp count
        ttgl.amd.gfx1250.mbarrier.init(ready_bars.index(i), count=PRODUCER_WARPS)

    args = PartitionArgs(a_desc, b_desc, a_buffer, b_buffer, empty_bars, ready_bars, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B,
                         WMMA_LAYOUT, c_ptr.type.element_ty)

    ttgl.warp_specialize([
        (consumer_partition, (args, c_ptr, M, N, stride_cm, stride_cn, pid_m, pid_n)),
        (producer_partition, (args, )),
    ], [PRODUCER_WARPS])


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("PERSISTENT", [False, True])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (250, 250, 510)])
@pytest.mark.parametrize("NUM_TOTAL_WARPS", [8, 12, 16])
def test_runtime_gemm_tdm_warp_specialized(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, PERSISTENT, M, N, K,
                                           NUM_TOTAL_WARPS):
    """Test warp specialized GEMM kernel."""
    if PERSISTENT and NUM_TOTAL_WARPS != 12:
        pytest.skip("Persistent WS kernel uses 12 total warps")
    if not PERSISTENT and NUM_TOTAL_WARPS == 12:
        pytest.skip("Non-persistent WS kernel uses 8 or 16 total warps")

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

    if not PERSISTENT:
        warp_bases = [(0, 1)]
        for i in range(int(math.log2(NUM_TOTAL_WARPS // 4))):
            warp_bases.append((1 << i, 0))
        warp_bases = tuple(warp_bases)

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
        kernel = gemm_tdm_warp_specialized_kernel[grid](
            a_device, b_device, c_device,  #
            M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
            NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B,  #
            NUM_WARPS=NUM_TOTAL_WARPS,  #
            WARP_BASES=tuple(warp_bases),  #
            num_warps=NUM_TOTAL_WARPS // 2)
    else:
        warp_bases = [(0, 1)]
        compute_warps = 4
        for i in range(int(math.log2(compute_warps // 2))):
            warp_bases.append((1 << i, 0))
        warp_bases = tuple(warp_bases)

        num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        # num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        # NOTE: Explicitly set num_sms to small number to ensure that each CU will compute multiple tiles.
        num_sms = 8
        grid = (min(num_sms, num_tiles), 1)

        kernel = persistent_gemm_tdm_warp_specialized_kernel[grid](
            a_device, b_device, c_device,  #
            M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
            NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, NUM_WARPS=NUM_TOTAL_WARPS, COMPUTE_WARPS=compute_warps,
            WARP_BASES=tuple(warp_bases),  #
            num_warps=NUM_TOTAL_WARPS // 3)

    static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ (b.to(torch.float32) if not TRANSPOSE_B else b.T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(128, 128, 256), (256, 256, 128)])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("TRANSPOSE_B", [True])
@pytest.mark.parametrize("PERSISTENT", [True])
@pytest.mark.parametrize("M,N,K", [(1024, 1024, 512)])
@pytest.mark.parametrize("NUM_TOTAL_WARPS", [12])
def test_runtime_gemm_tdm_warp_specialized_subtiled(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, PERSISTENT, M,
                                                    N, K, NUM_TOTAL_WARPS):
    """Test warp specialized GEMM kernel (subtiled variant for large blocks)."""
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

    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    # num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    # NOTE: Explicitly set num_sms to small number to ensure that each CU will compute multiple tiles.
    num_sms = 8
    grid = (min(num_sms, num_tiles), 1)

    warp_bases = [(0, 1)]
    compute_warps = 4
    for i in range(int(math.log2(compute_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    kernel = persistent_gemm_tdm_warp_specialized_subtiled_kernel[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, NUM_WARPS=NUM_TOTAL_WARPS, COMPUTE_WARPS=compute_warps,
        WARP_BASES=warp_bases,  #
        num_warps=NUM_TOTAL_WARPS // 3)

    static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ (b.to(torch.float32) if not TRANSPOSE_B else b.T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-3, atol=1e-3)


@gluon.jit
def split_accumulator_quadrant(acc):
    """Split an accumulator into 4 subtiles.

    Returns a tuple of 4 subtiles in row-major order: (top-left, top-right, bottom-left, bottom-right)
    """
    BLOCK_M: ttgl.constexpr = acc.shape[0]
    BLOCK_N: ttgl.constexpr = acc.shape[1]
    SUBTILE_M: ttgl.constexpr = BLOCK_M // 2
    SUBTILE_N: ttgl.constexpr = BLOCK_N // 2

    # Reshape [BLOCK_M, BLOCK_N] -> [2, SUBTILE_M, 2, SUBTILE_N]
    acc_4d = acc.reshape([2, SUBTILE_M, 2, SUBTILE_N])

    # Permute to [SUBTILE_M, SUBTILE_N, 2, 2] so split dimensions are at the end
    acc_4d = acc_4d.permute(1, 3, 0, 2)

    # Split along last dimension (split_n = 2) -> two tensors of [SUBTILE_M, SUBTILE_N, 2]
    acc_n0, acc_n1 = acc_4d.split()

    # Split each along last dimension (split_m = 2) -> four tensors of [SUBTILE_M, SUBTILE_N]
    acc_00, acc_10 = acc_n0.split()
    acc_01, acc_11 = acc_n1.split()

    return acc_00, acc_01, acc_10, acc_11


@gluon.jit
def persistent_producer_partition(args, scheduler):
    """Persistent Producer partition: Issues TDM async loads for A and B matrices."""
    K = args.a_desc.shape[1]
    BLOCK_M: ttgl.constexpr = args.a_desc.block_shape[0]
    BLOCK_N: ttgl.constexpr = args.b_desc.block_shape[0] if args.TRANSPOSE_B else args.b_desc.block_shape[1]

    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_k_tiles = ttgl.cdiv(K, args.BLOCK_K)
    num_tiles = scheduler.get_num_tiles()

    # Assume phase 0 is already completed as the buffers are initially empty; start from phase 1
    load_empty_phase_counter = PhaseCounter.create(args.NUM_BUFFERS, args.NUM_BUFFERS)

    for tile_idx in range(pid, num_tiles, num_sms):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M=8)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        for k_tile_idx in range(num_k_tiles):
            k_offset = k_tile_idx * args.BLOCK_K
            buffer_idx = k_tile_idx % args.NUM_BUFFERS

            empty_bar = args.load_empty_bars.index(buffer_idx)
            ready_bar = args.load_ready_bars.index(buffer_idx)

            # Wait for the buffers to be consumed before loading
            ttgl.amd.gfx1250.mbarrier.wait(empty_bar, load_empty_phase_counter.phase())

            # Only attach mbarrier to the last load so we signal once after both loads complete
            ttgl.amd.gfx1250.tdm.async_load(args.a_desc, [off_am, k_offset], args.a_buffer.index(buffer_idx))
            if args.TRANSPOSE_B:
                ttgl.amd.gfx1250.tdm.async_load(args.b_desc, [off_bn, k_offset], args.b_buffer.index(buffer_idx),
                                                mbarrier=ready_bar)
            else:
                ttgl.amd.gfx1250.tdm.async_load(args.b_desc, [k_offset, off_bn], args.b_buffer.index(buffer_idx),
                                                mbarrier=ready_bar)

            load_empty_phase_counter = load_empty_phase_counter.next()


@gluon.jit
def persistent_compute_partition(args, scheduler):
    """Persistent Compute partition: Waits for loaded data, performs WMMA operations, and writes accumulator to shared memory."""
    K = args.a_desc.shape[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, args.WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, args.WMMA_LAYOUT, 8)

    BLOCK_M: ttgl.constexpr = args.a_desc.block_shape[0]
    BLOCK_N: ttgl.constexpr = args.b_desc.block_shape[0] if args.TRANSPOSE_B else args.b_desc.block_shape[1]

    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_k_tiles = ttgl.cdiv(K, args.BLOCK_K)
    num_tiles = scheduler.get_num_tiles()

    load_ready_phase_counter = PhaseCounter.create(0, args.NUM_BUFFERS)
    # Assume phase 0 is already completed as the buffers are initially empty; start from phase 1
    acc_empty_phase_counter = PhaseCounter.create(args.NUM_ACC_BUFFERS, args.NUM_ACC_BUFFERS)

    local_tile_counter = 0
    for tile_idx in range(pid, num_tiles, num_sms):
        acc_buffer_idx = local_tile_counter % args.NUM_ACC_BUFFERS
        acc_empty_bar = args.acc_empty_bars.index(acc_buffer_idx)
        acc_ready_bar = args.acc_ready_bars.index(acc_buffer_idx)

        # Wait for the accumulator buffer to be empty (consumed by epilogue partition)
        ttgl.amd.gfx1250.mbarrier.wait(acc_empty_bar, acc_empty_phase_counter.phase())

        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=args.c_dtype, layout=args.WMMA_LAYOUT)

        for k_tile_idx in range(num_k_tiles):
            buffer_idx = k_tile_idx % args.NUM_BUFFERS
            ready_bar = args.load_ready_bars.index(buffer_idx)
            empty_bar = args.load_empty_bars.index(buffer_idx)

            # Wait for the buffers to be filled by the producer
            ttgl.amd.gfx1250.mbarrier.wait(ready_bar, load_ready_phase_counter.phase())

            a = args.a_buffer.index(buffer_idx).load(layout=OPERAND_LAYOUT_A)
            if args.TRANSPOSE_B:
                b = args.b_buffer.index(buffer_idx).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)
            else:
                b = args.b_buffer.index(buffer_idx).load(layout=OPERAND_LAYOUT_B)

            accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)

            # Signal that we're done with these buffers (producer can reuse them)
            ttgl.amd.gfx1250.mbarrier.arrive(empty_bar, count=1)

            load_ready_phase_counter = load_ready_phase_counter.next()

        # Store accumulator to shared memory for epilogue partition
        args.acc_buffer.index(acc_buffer_idx).store(accumulator)

        # Signal epilogue partition that accumulator is ready to be consumed
        ttgl.amd.gfx1250.mbarrier.arrive(acc_ready_bar, count=1)
        acc_empty_phase_counter = acc_empty_phase_counter.next()
        local_tile_counter += 1


@gluon.jit
def persistent_epilogue_partition(args, c_ptr, M, N, stride_cm, stride_cn, scheduler):
    """Epilogue partition: Waits for accumulator, issues TDM async store from shared to global memory."""
    BLOCK_M: ttgl.constexpr = args.a_desc.block_shape[0]
    BLOCK_N: ttgl.constexpr = args.b_desc.block_shape[0] if args.TRANSPOSE_B else args.b_desc.block_shape[1]

    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_tiles = scheduler.get_num_tiles()

    acc_ready_phase_counter = PhaseCounter.create(0, args.NUM_ACC_BUFFERS)

    local_tile_counter = 0
    for tile_idx in range(pid, num_tiles, num_sms):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M=8)
        acc_buffer_idx = local_tile_counter % args.NUM_ACC_BUFFERS
        acc_ready_bar = args.acc_ready_bars.index(acc_buffer_idx)
        acc_empty_bar = args.acc_empty_bars.index(acc_buffer_idx)

        # Wait for the accumulator to be filled by the compute partition
        ttgl.amd.gfx1250.mbarrier.wait(acc_ready_bar, acc_ready_phase_counter.phase())

        ttgl.amd.gfx1250.tdm.async_store(args.c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N],
                                         args.acc_buffer.index(acc_buffer_idx), mbarrier=acc_empty_bar)

        acc_ready_phase_counter = acc_ready_phase_counter.next()
        local_tile_counter += 1

    ttgl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit
def persistent_producer_subtiled_partition(args, scheduler):
    """Persistent Producer partition: Issues TDM async loads for A and B matrices."""
    K = args.a_desc.shape[1]
    QUADRANT_M: ttgl.constexpr = args.QUADRANT_M
    QUADRANT_N: ttgl.constexpr = args.QUADRANT_N
    BLOCK_M: ttgl.constexpr = args.QUADRANT_M * args.NUM_QUADS_M
    BLOCK_N: ttgl.constexpr = args.QUADRANT_N * args.NUM_QUADS_N
    NUM_QUADS: ttgl.constexpr = args.NUM_QUADS
    NUM_QUADS_N: ttgl.constexpr = args.NUM_QUADS_N

    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_k_tiles = ttgl.cdiv(K, args.BLOCK_K)
    num_tiles = scheduler.get_num_tiles()

    # Assume phase 0 is already completed as the buffers are initially empty; start from phase 1
    load_empty_phase_counter = PhaseCounter.create(args.NUM_BUFFERS, args.NUM_BUFFERS)

    for tile_idx in range(pid, num_tiles, num_sms):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M=8)

        for quad_idx in ttgl.static_range(NUM_QUADS):
            quad_m = quad_idx // NUM_QUADS_N
            quad_n = quad_idx % NUM_QUADS_N

            off_am = pid_m * BLOCK_M + quad_m * QUADRANT_M
            off_bn = pid_n * BLOCK_N + quad_n * QUADRANT_N

            for k_tile_idx in range(num_k_tiles):
                k_offset = k_tile_idx * args.BLOCK_K
                buffer_idx = k_tile_idx % args.NUM_BUFFERS

                empty_bar = args.load_empty_bars.index(buffer_idx)
                ready_bar = args.load_ready_bars.index(buffer_idx)

                # Wait for the buffers to be consumed before loading
                ttgl.amd.gfx1250.mbarrier.wait(empty_bar, load_empty_phase_counter.phase())

                # Only attach mbarrier to the last load so we signal once after both loads complete
                ttgl.amd.gfx1250.tdm.async_load(args.a_desc, [off_am, k_offset], args.a_buffer.index(buffer_idx))
                if args.TRANSPOSE_B:
                    ttgl.amd.gfx1250.tdm.async_load(args.b_desc, [off_bn, k_offset], args.b_buffer.index(buffer_idx),
                                                    mbarrier=ready_bar)
                else:
                    ttgl.amd.gfx1250.tdm.async_load(args.b_desc, [k_offset, off_bn], args.b_buffer.index(buffer_idx),
                                                    mbarrier=ready_bar)

                load_empty_phase_counter = load_empty_phase_counter.next()


@gluon.jit
def persistent_compute_subtiled_partition(args, scheduler):
    """Persistent Compute partition: Waits for loaded data, performs WMMA operations, and writes accumulator to shared memory."""
    K = args.a_desc.shape[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, args.WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, args.WMMA_LAYOUT, 8)

    QUADRANT_M: ttgl.constexpr = args.QUADRANT_M
    QUADRANT_N: ttgl.constexpr = args.QUADRANT_N
    NUM_QUADS: ttgl.constexpr = args.NUM_QUADS
    SUBTILES_PER_ACC: ttgl.constexpr = 4

    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_k_tiles = ttgl.cdiv(K, args.BLOCK_K)
    num_tiles = scheduler.get_num_tiles()

    load_ready_phase_counter = PhaseCounter.create(0, args.NUM_BUFFERS)
    # Assume phase 0 is already completed as the buffers are initially empty; start from phase 1
    acc_empty_phase_counter = PhaseCounter.create(args.NUM_ACC_BUFFERS, args.NUM_ACC_BUFFERS)

    for tile_idx in range(pid, num_tiles, num_sms):
        for quad_idx in ttgl.static_range(NUM_QUADS):
            # Process accumulator quadrants (1/4 of full accumulator tile) to avoid register spilling
            accumulator = ttgl.zeros((QUADRANT_M, QUADRANT_N), dtype=args.c_dtype, layout=args.WMMA_LAYOUT)

            for k_tile_idx in range(num_k_tiles):
                buffer_idx = k_tile_idx % args.NUM_BUFFERS
                ready_bar = args.load_ready_bars.index(buffer_idx)
                empty_bar = args.load_empty_bars.index(buffer_idx)

                # Wait for the buffers to be filled by the producer
                ttgl.amd.gfx1250.mbarrier.wait(ready_bar, load_ready_phase_counter.phase())

                a = args.a_buffer.index(buffer_idx).load(layout=OPERAND_LAYOUT_A)
                if args.TRANSPOSE_B:
                    b = args.b_buffer.index(buffer_idx).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)
                else:
                    b = args.b_buffer.index(buffer_idx).load(layout=OPERAND_LAYOUT_B)

                accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)

                # Signal that we're done with these buffers (producer can reuse them)
                ttgl.amd.gfx1250.mbarrier.arrive(empty_bar, count=1)

                load_ready_phase_counter = load_ready_phase_counter.next()

            # Split accumulator quadrant into subtiles to reduce shared memory usage
            subtiles = split_accumulator_quadrant(accumulator)

            for subtile_idx in ttgl.static_range(SUBTILES_PER_ACC):
                subtile = subtiles[subtile_idx]
                acc_buffer_idx = subtile_idx % args.NUM_ACC_BUFFERS
                acc_empty_bar = args.acc_empty_bars.index(acc_buffer_idx)
                acc_ready_bar = args.acc_ready_bars.index(acc_buffer_idx)
                # Wait for the accumulator subtile buffer to be empty (consumed by epilogue partition)
                ttgl.amd.gfx1250.mbarrier.wait(acc_empty_bar, acc_empty_phase_counter.phase())
                # Store buffer to shared memory for epilogue partition
                args.acc_buffer.index(acc_buffer_idx).store(subtile)
                # Signal epilogue partition that accumulator subtile is ready to be consumed
                ttgl.amd.gfx1250.mbarrier.arrive(acc_ready_bar, count=1)
                acc_empty_phase_counter = acc_empty_phase_counter.next()


@gluon.jit
def persistent_epilogue_subtiled_partition(args, scheduler):
    """Epilogue partition: Waits for accumulator, issues TDM async store from shared to global memory."""
    QUADRANT_M: ttgl.constexpr = args.QUADRANT_M
    QUADRANT_N: ttgl.constexpr = args.QUADRANT_N
    BLOCK_M: ttgl.constexpr = args.QUADRANT_M * args.NUM_QUADS_M
    BLOCK_N: ttgl.constexpr = args.QUADRANT_N * args.NUM_QUADS_N
    NUM_QUADS: ttgl.constexpr = args.NUM_QUADS
    NUM_QUADS_N: ttgl.constexpr = args.NUM_QUADS_N
    ACC_SUBTILE: ttgl.constexpr = 64  # Each subtile is 64x64
    SUBTILES_PER_QUAD: ttgl.constexpr = 4

    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_tiles = scheduler.get_num_tiles()

    acc_ready_phase_counter = PhaseCounter.create(0, args.NUM_ACC_BUFFERS)

    for tile_idx in range(pid, num_tiles, num_sms):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M=8)

        for quad_idx in ttgl.static_range(NUM_QUADS):
            quad_m = quad_idx // NUM_QUADS_N
            quad_n = quad_idx % NUM_QUADS_N

            quad_m_offset = quad_m * QUADRANT_M
            quad_n_offset = quad_n * QUADRANT_N

            for subtile_idx in ttgl.static_range(SUBTILES_PER_QUAD):
                local_subtile_m = subtile_idx // 2
                local_subtile_n = subtile_idx % 2

                acc_buffer_idx = subtile_idx % args.NUM_ACC_BUFFERS
                acc_ready_bar = args.acc_ready_bars.index(acc_buffer_idx)
                acc_empty_bar = args.acc_empty_bars.index(acc_buffer_idx)

                ttgl.amd.gfx1250.mbarrier.wait(acc_ready_bar, acc_ready_phase_counter.phase())

                offs_m = pid_m * BLOCK_M + quad_m_offset + local_subtile_m * ACC_SUBTILE
                offs_n = pid_n * BLOCK_N + quad_n_offset + local_subtile_n * ACC_SUBTILE

                ttgl.amd.gfx1250.tdm.async_store(args.c_desc, [offs_m, offs_n], args.acc_buffer.index(acc_buffer_idx),
                                                 mbarrier=acc_empty_bar)
                acc_ready_phase_counter = acc_ready_phase_counter.next()

    ttgl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit
def persistent_gemm_tdm_warp_specialized_kernel(a_ptr, b_ptr, c_ptr,  #
                                                M, N, K,  #
                                                stride_am, stride_ak,  #
                                                stride_bk, stride_bn,  #
                                                stride_cm, stride_cn,  #
                                                BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                BLOCK_K: ttgl.constexpr,  #
                                                NUM_BUFFERS: ttgl.constexpr,  #
                                                TRANSPOSE_B: ttgl.constexpr,  #
                                                NUM_WARPS: ttgl.constexpr, COMPUTE_WARPS: ttgl.constexpr,
                                                WARP_BASES: ttgl.constexpr):
    """Persistent warp specialized GEMM kernel with three partitions (producer, compute, epilogue)."""
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    ttgl.static_assert(NUM_WARPS == 12, "NUM_WARPS must be 12 for this kernel")

    # WS kernels require num_warps to be a multiple of 4; default partition (epilogue) must have multiple of 4 warps.
    PRODUCER_WARPS: ttgl.constexpr = 4
    EPILOGUE_WARPS: ttgl.constexpr = 4
    WARP_SIZE: ttgl.constexpr = 32

    # accumulator buffers used for double-buffering to overlap epilogue with load of the next tile
    NUM_ACC_BUFFERS: ttgl.constexpr = 2

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])
    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]

    SHARED_LAYOUT_ACC: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, 0, 0, stride_am, stride_ak, stride_bn, stride_bk,
                                               SHARED_LAYOUT_A, SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                               TRANSPOSE_B)

    c_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT_ACC)

    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES=0)

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)
    acc_buffer = ttgl.allocate_shared_memory(c_ptr.type.element_ty, shape=[NUM_ACC_BUFFERS, BLOCK_M, BLOCK_N],
                                             layout=SHARED_LAYOUT_ACC)

    load_empty_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_BUFFERS, 1],
                                                  ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
    load_ready_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_BUFFERS, 1],
                                                  ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
    acc_empty_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_ACC_BUFFERS, 1],
                                                 ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
    acc_ready_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_ACC_BUFFERS, 1],
                                                 ttgl.amd.gfx1250.mbarrier.MBarrierLayout())

    # Initialize mbarriers
    # load_empty_bars: signals when compute partition has consumed the shared memory buffers for matrices A and B
    # load_ready_bars: signals when producer partition has filled the shared memory buffer for matrices A and B
    # acc_empty_bars: signals when epilogue partition has stored the accumulator provided by the compute partition
    # acc_ready_bars: signals when compute partition has filled the accuumulator to be consumed by the epilogue partition
    for i in ttgl.static_range(NUM_BUFFERS):
        # load_empty_bars: arrive on barrier once per thread, so use compute thread count
        ttgl.amd.gfx1250.mbarrier.init(load_empty_bars.index(i), count=COMPUTE_WARPS * WARP_SIZE)
        # load_ready_bars: TDM arrives on barrier once per warp, so use producer warp count
        ttgl.amd.gfx1250.mbarrier.init(load_ready_bars.index(i), count=PRODUCER_WARPS)

    for i in ttgl.static_range(NUM_ACC_BUFFERS):
        # acc_empty_bars: TDM arrives on barrier once per warp, so use epilogue warp count
        ttgl.amd.gfx1250.mbarrier.init(acc_empty_bars.index(i), count=EPILOGUE_WARPS)
        # acc_ready_bars: arrive on barrier once per thread, so use compute thread count
        ttgl.amd.gfx1250.mbarrier.init(acc_ready_bars.index(i), count=COMPUTE_WARPS * WARP_SIZE)

    args = PersistentPartitionArgs(a_desc, b_desc, c_desc, a_buffer, b_buffer, acc_buffer, load_empty_bars,
                                   load_ready_bars, acc_empty_bars, acc_ready_bars, BLOCK_K, NUM_BUFFERS,
                                   NUM_ACC_BUFFERS, TRANSPOSE_B, WMMA_LAYOUT, c_ptr.type.element_ty)

    ttgl.warp_specialize([
        (persistent_epilogue_partition, (args, c_ptr, M, N, stride_cm, stride_cn, scheduler)),
        (persistent_compute_partition, (args, scheduler)),
        (persistent_producer_partition, (args, scheduler)),
    ], [COMPUTE_WARPS, PRODUCER_WARPS])


@gluon.jit
def persistent_gemm_tdm_warp_specialized_subtiled_kernel(a_ptr, b_ptr, c_ptr,  #
                                                         M, N, K,  #
                                                         stride_am, stride_ak,  #
                                                         stride_bk, stride_bn,  #
                                                         stride_cm, stride_cn,  #
                                                         BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                         BLOCK_K: ttgl.constexpr,  #
                                                         NUM_BUFFERS: ttgl.constexpr,  #
                                                         TRANSPOSE_B: ttgl.constexpr,  #
                                                         NUM_WARPS: ttgl.constexpr,  #
                                                         COMPUTE_WARPS: ttgl.constexpr,  #
                                                         WARP_BASES: ttgl.constexpr):
    """Persistent warp specialized GEMM kernel with quadrant-based subtiling (three partitions: producer, compute, epilogue)."""
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    ttgl.static_assert(NUM_WARPS == 12, "NUM_WARPS must be 12 for this kernel")

    # WS kernels require num_warps to be a multiple of 4; default partition (epilogue) must have multiple of 4 warps.
    PRODUCER_WARPS: ttgl.constexpr = 4
    EPILOGUE_WARPS: ttgl.constexpr = 4
    WARP_SIZE: ttgl.constexpr = 32

    # accumulator buffers used for double-buffering to overlap epilogue with load of the next tile
    NUM_ACC_BUFFERS: ttgl.constexpr = 2

    # Accumulator subtile size for shared memory (fixed at 64x64)
    ACC_SUBTILE_M: ttgl.constexpr = 64
    ACC_SUBTILE_N: ttgl.constexpr = 64

    QUADRANT_M: ttgl.constexpr = 128
    QUADRANT_N: ttgl.constexpr = 128
    NUM_QUADS_M: ttgl.constexpr = BLOCK_M // QUADRANT_M
    NUM_QUADS_N: ttgl.constexpr = BLOCK_N // QUADRANT_N
    NUM_QUADS: ttgl.constexpr = NUM_QUADS_M * NUM_QUADS_N

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])
    shared_layouts: ttgl.constexpr = create_shared_layouts(QUADRANT_M, QUADRANT_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]

    SHARED_LAYOUT_ACC: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, 0, 0, stride_am, stride_ak, stride_bn, stride_bk,
                                               SHARED_LAYOUT_A, SHARED_LAYOUT_B, M, N, K, QUADRANT_M, QUADRANT_N,
                                               BLOCK_K, TRANSPOSE_B)

    c_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                                         block_shape=(ACC_SUBTILE_M, ACC_SUBTILE_N),
                                                         layout=SHARED_LAYOUT_ACC)

    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES=0)

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)
    acc_buffer = ttgl.allocate_shared_memory(c_ptr.type.element_ty,
                                             shape=[NUM_ACC_BUFFERS, ACC_SUBTILE_M,
                                                    ACC_SUBTILE_N], layout=SHARED_LAYOUT_ACC)

    load_empty_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_BUFFERS, 1],
                                                  ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
    load_ready_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_BUFFERS, 1],
                                                  ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
    acc_empty_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_ACC_BUFFERS, 1],
                                                 ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
    acc_ready_bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_ACC_BUFFERS, 1],
                                                 ttgl.amd.gfx1250.mbarrier.MBarrierLayout())

    # Initialize mbarriers
    # load_empty_bars: signals when compute partition has consumed the shared memory buffers for matrices A and B
    # load_ready_bars: signals when producer partition has filled the shared memory buffer for matrices A and B
    # acc_empty_bars: signals when epilogue partition has stored the accumulator provided by the compute partition
    # acc_ready_bars: signals when compute partition has filled the accuumulator to be consumed by the epilogue partition
    for i in ttgl.static_range(NUM_BUFFERS):
        # load_empty_bars: arrive on barrier once per thread, so use compute thread count
        ttgl.amd.gfx1250.mbarrier.init(load_empty_bars.index(i), count=COMPUTE_WARPS * WARP_SIZE)
        # load_ready_bars: TDM arrives on barrier once per warp, so use producer warp count
        ttgl.amd.gfx1250.mbarrier.init(load_ready_bars.index(i), count=PRODUCER_WARPS)

    for i in ttgl.static_range(NUM_ACC_BUFFERS):
        # acc_empty_bars: TDM arrives on barrier once per warp, so use epilogue warp count
        ttgl.amd.gfx1250.mbarrier.init(acc_empty_bars.index(i), count=EPILOGUE_WARPS)
        # acc_ready_bars: arrive on barrier once per thread, so use compute thread count
        ttgl.amd.gfx1250.mbarrier.init(acc_ready_bars.index(i), count=COMPUTE_WARPS * WARP_SIZE)

    args = PersistentPartitionSubtiledArgs(a_desc, b_desc, c_desc, a_buffer, b_buffer, acc_buffer, load_empty_bars,
                                           load_ready_bars, acc_empty_bars, acc_ready_bars, BLOCK_K, NUM_BUFFERS,
                                           NUM_ACC_BUFFERS, TRANSPOSE_B, WMMA_LAYOUT, c_ptr.type.element_ty, NUM_QUADS,
                                           NUM_QUADS_M, NUM_QUADS_N, QUADRANT_M, QUADRANT_N)

    ttgl.warp_specialize([
        (persistent_epilogue_subtiled_partition, (args, scheduler)),
        (persistent_compute_subtiled_partition, (args, scheduler)),
        (persistent_producer_subtiled_partition, (args, scheduler)),
    ], [COMPUTE_WARPS, PRODUCER_WARPS])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=256, help='problem M size')
    parser.add_argument("-N", type=int, default=256, help='problem N size')
    parser.add_argument("-K", type=int, default=1024, help='problem K size')
    parser.add_argument("--block_m", type=int, default=256, help='Block M size')
    parser.add_argument("--block_n", type=int, default=256, help='Block N size')
    parser.add_argument("--block_k", type=int, default=128, help='Block K size')
    parser.add_argument("--num-warps", type=int, choices=[4, 8, 12, 16], default=4,
                        help='num warps (for warp specialized, this is num total warps)')
    parser.add_argument("--ctas-per-cga", type=int, nargs=2, default=[1, 1],
                        help='CTA arrangement per CGA as [M, N]. Defaults to [1, 1]')
    parser.add_argument("--num-buffers", type=int, choices=[1, 2, 3, 4], default=2, help='num shared memory buffers')
    parser.add_argument("--persistent", action="store_true", help="Use persistent variant")
    parser.add_argument("--prefetch-lds", action="store_true", help="Enable prefetch LDS")
    parser.add_argument(
        "--prefetch-l2-distance", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help=
        "Prefetch distance (in iterations) for operands into L2 before issuing preloads (TDM). 0 disables L2 prefetch.")
    parser.add_argument("--single-warp-schedule", action="store_true", help="Use single warp per SIMD schedule variant")
    parser.add_argument("--warp-specialized", action="store_true", help="Use warp specialized variant")
    parser.add_argument("--subtiled", action="store_true", help="Use subtiled quadrant processing")
    args = parser.parse_args()

    assert not (args.persistent and args.single_warp_schedule)
    assert not (args.warp_specialized and args.single_warp_schedule)
    if args.single_warp_schedule:
        assert args.num_warps == 4
        assert args.prefetch_lds
    if args.warp_specialized and not args.persistent:
        assert args.num_warps in [8, 16]
    elif args.warp_specialized and args.persistent:
        assert args.num_warps == 12
    elif not args.persistent:
        assert args.num_warps != 16

    M, N, K = args.M, args.N, args.K
    BLOCK_M, BLOCK_N, BLOCK_K = args.block_m, args.block_n, args.block_k
    NUM_BUFFERS = args.num_buffers
    NUM_WARPS = args.num_warps
    CTAS_PER_CGA = args.ctas_per_cga
    NUM_CTAS = CTAS_PER_CGA[0] * CTAS_PER_CGA[1]
    TRANSPOSE_B = True
    PERSISTENT = args.persistent
    PREFETCH = args.prefetch_lds
    L2_PREFETCH_DISTANCE = args.prefetch_l2_distance

    if NUM_CTAS not in [1, 2, 4, 8, 16]:
        raise ValueError(f"NUM_CTAS (product of CTAS_PER_CGA) {NUM_CTAS} not supported")

    if args.warp_specialized:
        assert NUM_CTAS == 1, "NUM_CTAS > 1 not supported for warp specialized gemm"
        assert L2_PREFETCH_DISTANCE == 0, "L2 prefetch no support in warp specialized kernel"
        # For warp specialized, allow larger blocks with subtiled variant
        if args.subtiled:
            BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 128
            print(f"Limited block size support; resetting to {BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}")
            kernel_type = "persistent" if PERSISTENT else "non-persistent"
            print(f"Running {kernel_type} warp specialized GEMM kernel (subtiled):")
            print(
                f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {TRANSPOSE_B=}, NUM_TOTAL_WARPS={NUM_WARPS}, {NUM_BUFFERS=}, {PERSISTENT=}"
            )
            test_runtime_gemm_tdm_warp_specialized_subtiled(BLOCK_M, BLOCK_N, BLOCK_K,  #
                                                            NUM_BUFFERS, TRANSPOSE_B, PERSISTENT,  #
                                                            M, N, K, NUM_WARPS)
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
            print(f"Limited block size support; resetting to {BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}")
            kernel_type = "persistent" if PERSISTENT else "non-persistent"
            print(f"Running {kernel_type} warp specialized GEMM kernel:")
            print(
                f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {TRANSPOSE_B=}, NUM_TOTAL_WARPS={NUM_WARPS}, {NUM_BUFFERS=}, {PERSISTENT=}"
            )
            test_runtime_gemm_tdm_warp_specialized(BLOCK_M, BLOCK_N, BLOCK_K,  #
                                                   NUM_BUFFERS, TRANSPOSE_B, PERSISTENT,  #
                                                   M, N, K, NUM_WARPS)
    elif args.single_warp_schedule:
        assert NUM_CTAS == 1, "NUM_CTAS > 1 not supported for single warp schedule"
        print(
            f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {TRANSPOSE_B=}, {NUM_WARPS=}, {NUM_BUFFERS=}, {PERSISTENT=}, {PREFETCH=}, {L2_PREFETCH_DISTANCE=}"
        )
        test_runtime_gemm_tdm_pipelined_single_warp_per_simd_schedule(BLOCK_M, BLOCK_N,  #
                                                                      NUM_BUFFERS, TRANSPOSE_B, L2_PREFETCH_DISTANCE,  #
                                                                      M, N, K)
    else:
        print(
            f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {TRANSPOSE_B=}, {NUM_WARPS=}, {NUM_BUFFERS=}, {PERSISTENT=}, {PREFETCH=}, {L2_PREFETCH_DISTANCE=}, {CTAS_PER_CGA=}"
        )
        _run_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K,  #
                                        NUM_BUFFERS, TRANSPOSE_B, PERSISTENT, PREFETCH, L2_PREFETCH_DISTANCE,  #
                                        M, N, K, NUM_WARPS, CTAS_PER_CGA)
