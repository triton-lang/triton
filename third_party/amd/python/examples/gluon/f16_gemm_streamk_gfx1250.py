# ruff: noqa: E402
import hip

# Needed for internal dev flow for now; will remove later
hip.hip.hipInit(0)

import pytest
import torch
import math
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
        TileScheduler,
    )
except ImportError:
    from gfx1250_utils import static_profile
    from f16_gemm_common_gfx1250 import (
        create_shared_layouts,
        create_tensor_descriptors,
        issue_loads,
        issue_wmma,
        TileScheduler,
    )


@gluon.jit
def split_accumulator_quadrant(
    accumulator,
    HALF_M: ttgl.constexpr,
    HALF_N: ttgl.constexpr,
    qm: ttgl.constexpr,
    qn: ttgl.constexpr,
):
    """
    Extract a single quadrant from the accumulator to avoid holding all 4 at once.
    qm, qn are constexpr in {0,1} selecting M/N quadrant.
    """
    acc_4d = accumulator.reshape([2, HALF_M, 2, HALF_N])
    acc_4d = acc_4d.permute(1, 3, 0, 2)
    acc_n0, acc_n1 = acc_4d.split()
    if qn == 0:
        acc_m0, acc_m1 = acc_n0.split()
    else:
        acc_m0, acc_m1 = acc_n1.split()
    if qm == 0:
        return acc_m0
    return acc_m1


@gluon.jit
def store_quadrant_to_p_buffer(
    acc_q,
    p_ptr,
    pid,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    HALF_M: ttgl.constexpr,
    HALF_N: ttgl.constexpr,
    rm_q,
    rn_q,
    qm: ttgl.constexpr,
    qn: ttgl.constexpr,
):
    """
    Store a single quadrant to P buffer using buffer_store.
    Expects acc_q already converted to QUAD_WMMA layout.
    """
    P_base_offs = pid * BLOCK_M * BLOCK_N
    row_offs = rm_q + (qm * HALF_M)
    col_offs = rn_q + (qn * HALF_N)
    p_offs = P_base_offs + row_offs[:, None] * BLOCK_N + col_offs[None, :]
    ttgl.amd.gfx1250.buffer_store(acc_q, p_ptr, p_offs)


@gluon.jit
def load_quadrant_from_p_buffer(p_ptr, next_pid, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                HALF_M: ttgl.constexpr, HALF_N: ttgl.constexpr, rm_q, rn_q, qm: ttgl.constexpr,
                                qn: ttgl.constexpr):
    """
    Load a single quadrant from P buffer using buffer_load.
    """
    P_base_offs = next_pid * BLOCK_M * BLOCK_N
    row_offs = rm_q + (qm * HALF_M)
    col_offs = rn_q + (qn * HALF_N)
    p_offs = P_base_offs + row_offs[:, None] * BLOCK_N + col_offs[None, :]
    return ttgl.amd.gfx1250.buffer_load(p_ptr, p_offs)


@gluon.jit
def store_quadrant_to_c_buffer(
    acc_q,
    c_ptr,
    pid_m,
    pid_n,
    M,
    N,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    HALF_M: ttgl.constexpr,
    HALF_N: ttgl.constexpr,
    rm_q,
    rn_q,
    stride_cm,
    stride_cn,
    qm: ttgl.constexpr,
    qn: ttgl.constexpr,
):
    """
    Store a single quadrant to output C buffer using buffer_store with masking.
    Expects acc_q already converted to QUAD_WMMA layout.
    """
    rm = pid_m * BLOCK_M + rm_q + (qm * HALF_M)
    rn = pid_n * BLOCK_N + rn_q + (qn * HALF_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    offs = stride_cm * rm[:, None] + stride_cn * rn[None, :]
    ttgl.amd.gfx1250.buffer_store(acc_q, c_ptr, offs, mask=mask)


@gluon.jit
def process_streamk_tiles(
    a_desc,
    b_desc,
    a_buffer,
    b_buffer,
    c_ptr,
    p_ptr,
    locks_ptr,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    pid,
    num_sms,
    num_full_tiles,
    scheduler,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    TRANSPOSE_B: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    WMMA_LAYOUT: ttgl.constexpr,
    OPERAND_LAYOUT_A: ttgl.constexpr,
    OPERAND_LAYOUT_B: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
):
    """
    Phase 2: Process StreamK tiles for 4 warps
    """
    if STREAMK_TILES == 0:
        return

    # Initialize P buffer and locks
    rm = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    rn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    p_offset = pid * BLOCK_M * BLOCK_N + rm[:, None] * BLOCK_N + rn[None, :]
    ttgl.store(
        p_ptr + p_offset,
        ttgl.zeros((BLOCK_M, BLOCK_N), dtype=p_ptr.type.element_ty, layout=WMMA_LAYOUT),
    )
    ttgl.store(locks_ptr + pid, 0)

    # Compute StreamK params
    iters_per_tile = ttgl.cdiv(K, BLOCK_K)
    num_streamk_tiles = scheduler.get_num_streamk_tiles()
    total_streamk_iters = num_streamk_tiles * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // num_sms
    streamk_remainder_iters = total_streamk_iters % num_sms

    # Compute iteration range
    base_offset = num_full_tiles * iters_per_tile
    start_iter = (base_offset + pid * streamk_iters_pcu + ttgl.minimum(pid, streamk_remainder_iters))
    last_iter = (base_offset + (pid + 1) * streamk_iters_pcu + ttgl.minimum(pid + 1, streamk_remainder_iters))

    current_start_iter = start_iter
    while current_start_iter < last_iter:
        remainder = current_start_iter % iters_per_tile
        end_iter = ttgl.minimum(current_start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = current_start_iter // iters_per_tile
        tile_iter = tile_id * iters_per_tile

        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_id, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)
        num_k_iters = end_iter - current_start_iter

        for k_idx in range(num_k_iters):
            k_offset = (remainder + k_idx) * BLOCK_K

            ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, k_offset], a_buffer.index(0))
            if not TRANSPOSE_B:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_offset, off_bn], b_buffer.index(0))
            else:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, k_offset], b_buffer.index(0))

            ttgl.amd.gfx1250.tdm.async_wait(0)

            a_operand = a_buffer.index(0).load(layout=OPERAND_LAYOUT_A)
            if not TRANSPOSE_B:
                b_operand = b_buffer.index(0).load(layout=OPERAND_LAYOUT_B)
            else:
                b_operand = (b_buffer.index(0).permute([1, 0]).load(layout=OPERAND_LAYOUT_B))

            accumulator = ttgl.amd.gfx1250.wmma(a_operand, b_operand, accumulator)

        # Shared quadrant setup for 256x256 tiles
        HALF_M: ttgl.constexpr = BLOCK_M // 2
        HALF_N: ttgl.constexpr = BLOCK_N // 2

        # Contributor or Owner logic
        if current_start_iter != tile_iter:
            # Contributor: Store accumulator to P buffer
            if BLOCK_M == 256 and BLOCK_N == 256:
                # Store quadrants separately for 256x256 tiles using buffer_store
                acc_4d = accumulator.reshape([2, HALF_M, 2, HALF_N])
                acc_4d = acc_4d.permute(1, 3, 0, 2)
                acc_n0, acc_n1 = acc_4d.split()
                acc_00, acc_10 = acc_n0.split()
                acc_01, acc_11 = acc_n1.split()

                # Use WMMA layout offsets for buffer_store
                rm_q = ttgl.arange(0, HALF_M)
                rn_q = ttgl.arange(0, HALF_N)
                P_base_offs = pid * BLOCK_M * BLOCK_N

                # Store each quadrant with buffer_store
                p00_offs = P_base_offs + rm_q[:, None] * BLOCK_N + rn_q[None, :]
                ttgl.amd.gfx1250.buffer_store(acc_00, p_ptr, p00_offs)

                p01_offs = P_base_offs + rm_q[:, None] * BLOCK_N + (rn_q[None, :] + HALF_N)
                ttgl.amd.gfx1250.buffer_store(acc_01, p_ptr, p01_offs)

                p10_offs = P_base_offs + (rm_q[:, None] + HALF_M) * BLOCK_N + rn_q[None, :]
                ttgl.amd.gfx1250.buffer_store(acc_10, p_ptr, p10_offs)

                p11_offs = P_base_offs + (rm_q[:, None] + HALF_M) * BLOCK_N + (rn_q[None, :] + HALF_N)
                ttgl.amd.gfx1250.buffer_store(acc_11, p_ptr, p11_offs)
            else:
                # Full tile store for smaller tiles
                rm1 = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
                rn1 = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
                p_offset = pid * BLOCK_M * BLOCK_N + rm1[:, None] * BLOCK_N + rn1[None, :]
                ttgl.amd.gfx1250.buffer_store(accumulator, p_ptr, p_offset)

            ttgl.barrier()
            ttgl.atomic_xchg(locks_ptr + pid, 1)

        else:
            # Owner: Aggregate contributors and store result
            next_pid = pid + 1
            end = end_iter

            if BLOCK_M == 256 and BLOCK_N == 256:
                acc_4d = accumulator.reshape([2, HALF_M, 2, HALF_N])
                acc_4d = acc_4d.permute(1, 3, 0, 2)
                acc_n0, acc_n1 = acc_4d.split()
                acc_00, acc_10 = acc_n0.split()
                acc_01, acc_11 = acc_n1.split()

                # Use WMMA layout offsets for buffer_load
                rm_q = ttgl.arange(0, HALF_M)
                rn_q = ttgl.arange(0, HALF_N)

                while end < tile_iter + iters_per_tile and next_pid < num_sms:
                    while ttgl.atomic_cas(locks_ptr + next_pid, 1, 1) != 1:
                        pass

                    P_base_offs = next_pid * BLOCK_M * BLOCK_N

                    # Load and accumulate quadrants with buffer_load
                    p00_offs = P_base_offs + rm_q[:, None] * BLOCK_N + rn_q[None, :]
                    acc_00 += ttgl.amd.gfx1250.buffer_load(p_ptr, p00_offs)

                    p01_offs = P_base_offs + rm_q[:, None] * BLOCK_N + (rn_q[None, :] + HALF_N)
                    acc_01 += ttgl.amd.gfx1250.buffer_load(p_ptr, p01_offs)

                    p10_offs = P_base_offs + (rm_q[:, None] + HALF_M) * BLOCK_N + rn_q[None, :]
                    acc_10 += ttgl.amd.gfx1250.buffer_load(p_ptr, p10_offs)

                    p11_offs = P_base_offs + (rm_q[:, None] + HALF_M) * BLOCK_N + (rn_q[None, :] + HALF_N)
                    acc_11 += ttgl.amd.gfx1250.buffer_load(p_ptr, p11_offs)

                    end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                    next_pid += 1

                # Store quadrants to output C with buffer_store
                rm_top = pid_m * BLOCK_M + rm_q
                rm_bottom = pid_m * BLOCK_M + (rm_q + HALF_M)
                rn_left = pid_n * BLOCK_N + rn_q
                rn_right = pid_n * BLOCK_N + (rn_q + HALF_N)

                mask00 = (rm_top[:, None] < M) & (rn_left[None, :] < N)
                offs00 = stride_cm * rm_top[:, None] + stride_cn * rn_left[None, :]
                ttgl.amd.gfx1250.buffer_store(acc_00, c_ptr, offs00, mask=mask00)

                mask01 = (rm_top[:, None] < M) & (rn_right[None, :] < N)
                offs01 = stride_cm * rm_top[:, None] + stride_cn * rn_right[None, :]
                ttgl.amd.gfx1250.buffer_store(acc_01, c_ptr, offs01, mask=mask01)

                mask10 = (rm_bottom[:, None] < M) & (rn_left[None, :] < N)
                offs10 = stride_cm * rm_bottom[:, None] + stride_cn * rn_left[None, :]
                ttgl.amd.gfx1250.buffer_store(acc_10, c_ptr, offs10, mask=mask10)

                mask11 = (rm_bottom[:, None] < M) & (rn_right[None, :] < N)
                offs11 = stride_cm * rm_bottom[:, None] + stride_cn * rn_right[None, :]
                ttgl.amd.gfx1250.buffer_store(acc_11, c_ptr, offs11, mask=mask11)
            else:
                # Full accumulator for smaller tiles
                rm = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
                rn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
                offs_m = pid_m * BLOCK_M + rm
                offs_n = pid_n * BLOCK_N + rn

                while end < tile_iter + iters_per_tile and next_pid < num_sms:
                    while ttgl.atomic_cas(locks_ptr + next_pid, 1, 1) != 1:
                        pass

                    p_offset_load = next_pid * BLOCK_M * BLOCK_N + rm[:, None] * BLOCK_N + rn[None, :]
                    contrib_acc = ttgl.amd.gfx1250.buffer_load(p_ptr, p_offset_load)
                    accumulator += contrib_acc

                    end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                    next_pid += 1

                mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                offs_c = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
                ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask)

        current_start_iter = end_iter


@gluon.jit
def process_streamk_tiles_8warps(
    a_desc,
    b_desc,
    a_buffer,
    b_buffer,
    c_ptr,
    p_ptr,
    locks_ptr,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    pid,
    num_sms,
    num_full_tiles,
    scheduler,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    TRANSPOSE_B: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    WMMA_LAYOUT: ttgl.constexpr,
    OPERAND_LAYOUT_A: ttgl.constexpr,
    OPERAND_LAYOUT_B: ttgl.constexpr,
    WARP_BASES: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
):
    """
    Phase 2: StreamK tiles for 8-warp kernels.
    """
    if STREAMK_TILES == 0:
        return

    # Initialize P buffer and locks
    rm = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    rn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    p_offset = pid * BLOCK_M * BLOCK_N + rm[:, None] * BLOCK_N + rn[None, :]
    ttgl.store(
        p_ptr + p_offset,
        ttgl.zeros((BLOCK_M, BLOCK_N), dtype=p_ptr.type.element_ty, layout=WMMA_LAYOUT),
    )
    ttgl.store(locks_ptr + pid, 0)

    # Compute StreamK params
    iters_per_tile = ttgl.cdiv(K, BLOCK_K)
    num_streamk_tiles = scheduler.get_num_streamk_tiles()
    total_streamk_iters = num_streamk_tiles * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // num_sms
    streamk_remainder_iters = total_streamk_iters % num_sms

    # Compute iteration range
    base_offset = num_full_tiles * iters_per_tile
    start_iter = (base_offset + pid * streamk_iters_pcu + ttgl.minimum(pid, streamk_remainder_iters))
    last_iter = (base_offset + (pid + 1) * streamk_iters_pcu + ttgl.minimum(pid + 1, streamk_remainder_iters))

    current_start_iter = start_iter
    while current_start_iter < last_iter:
        remainder = current_start_iter % iters_per_tile
        end_iter = ttgl.minimum(current_start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = current_start_iter // iters_per_tile
        tile_iter = tile_id * iters_per_tile

        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_id, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)
        num_k_iters = end_iter - current_start_iter

        for k_idx in range(num_k_iters):
            k_offset = (remainder + k_idx) * BLOCK_K

            ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, k_offset], a_buffer.index(0))
            if not TRANSPOSE_B:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_offset, off_bn], b_buffer.index(0))
            else:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, k_offset], b_buffer.index(0))

            ttgl.amd.gfx1250.tdm.async_wait(0)

            a_operand = a_buffer.index(0).load(layout=OPERAND_LAYOUT_A)
            if not TRANSPOSE_B:
                b_operand = b_buffer.index(0).load(layout=OPERAND_LAYOUT_B)
            else:
                b_operand = (b_buffer.index(0).permute([1, 0]).load(layout=OPERAND_LAYOUT_B))

            accumulator = ttgl.amd.gfx1250.wmma(a_operand, b_operand, accumulator)

        # Quadrant handling for 256x256 tiles to reduce VGPR pressure
        if BLOCK_M == 256 and BLOCK_N == 256:
            HALF_M: ttgl.constexpr = BLOCK_M // 2
            HALF_N: ttgl.constexpr = BLOCK_N // 2
            rm_q = ttgl.arange(0, HALF_M)
            rn_q = ttgl.arange(0, HALF_N)

            if current_start_iter != tile_iter:
                # Contributor: store quadrants to P buffer sequentially
                for qm in ttgl.static_range(2):
                    for qn in ttgl.static_range(2):
                        acc_q = split_accumulator_quadrant(accumulator, HALF_M, HALF_N, qm, qn)
                        store_quadrant_to_p_buffer(
                            acc_q,
                            p_ptr,
                            pid,
                            BLOCK_M,
                            BLOCK_N,
                            HALF_M,
                            HALF_N,
                            rm_q,
                            rn_q,
                            qm,
                            qn,
                        )
                ttgl.barrier()
                ttgl.atomic_xchg(locks_ptr + pid, 1)
            else:
                # Owner: aggregate contributors and store result sequentially
                for qm in ttgl.static_range(2):
                    for qn in ttgl.static_range(2):
                        acc_q = split_accumulator_quadrant(accumulator, HALF_M, HALF_N, qm, qn)

                        next_pid = pid + 1
                        end = end_iter
                        while end < tile_iter + iters_per_tile and next_pid < num_sms:
                            while ttgl.atomic_cas(locks_ptr + next_pid, 1, 1) != 1:
                                pass
                            contrib_q = load_quadrant_from_p_buffer(
                                p_ptr,
                                next_pid,
                                BLOCK_M,
                                BLOCK_N,
                                HALF_M,
                                HALF_N,
                                rm_q,
                                rn_q,
                                qm,
                                qn,
                            )
                            acc_q = acc_q + contrib_q
                            end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                            next_pid += 1

                        store_quadrant_to_c_buffer(
                            acc_q,
                            c_ptr,
                            pid_m,
                            pid_n,
                            M,
                            N,
                            BLOCK_M,
                            BLOCK_N,
                            HALF_M,
                            HALF_N,
                            rm_q,
                            rn_q,
                            stride_cm,
                            stride_cn,
                            qm,
                            qn,
                        )
        else:
            # Full accumulator for smaller tiles
            rm_full = ttgl.arange(0, BLOCK_M)
            rn_full = ttgl.arange(0, BLOCK_N)
            if current_start_iter != tile_iter:
                p_off = (pid * BLOCK_M * BLOCK_N + rm_full[:, None] * BLOCK_N + rn_full[None, :])
                ttgl.amd.gfx1250.buffer_store(accumulator, p_ptr, p_off)
                ttgl.barrier()
                ttgl.atomic_xchg(locks_ptr + pid, 1)
            else:
                next_pid = pid + 1
                end = end_iter
                offs_m = pid_m * BLOCK_M + rm_full
                offs_n = pid_n * BLOCK_N + rn_full

                while end < tile_iter + iters_per_tile and next_pid < num_sms:
                    while ttgl.atomic_cas(locks_ptr + next_pid, 1, 1) != 1:
                        pass
                    p_offset_load = (next_pid * BLOCK_M * BLOCK_N + rm_full[:, None] * BLOCK_N + rn_full[None, :])
                    contrib_acc = ttgl.amd.gfx1250.buffer_load(p_ptr, p_offset_load)
                    accumulator += contrib_acc
                    end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                    next_pid += 1

                mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                offs_c = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
                ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask)

        current_start_iter = end_iter


@gluon.jit
def streamk_gemm_tdm_pipelined_kernel_4warps(
    a_ptr,
    b_ptr,
    c_ptr,
    p_ptr,
    locks_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    NUM_BUFFERS: ttgl.constexpr,
    TRANSPOSE_B: ttgl.constexpr,
    NUM_WARPS: ttgl.constexpr,
    WARP_BASES: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
):
    """
    StreamK GEMM kernel for 4 warps only.
    Phase 1: Standard pipelined persistent loop (no warp specialization).
    Phase 2: Uses process_streamk_tiles (4-warp StreamK remainder path).
    """
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    ttgl.static_assert(NUM_WARPS == 4, "This kernel is only valid for NUM_WARPS == 4")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])
    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc, b_desc = create_tensor_descriptors(
        a_ptr,
        b_ptr,
        0,
        0,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        SHARED_LAYOUT_A,
        SHARED_LAYOUT_B,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TRANSPOSE_B,
    )
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    # Initialize scheduler
    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES)

    # Phase 1: Process full tiles (persistent scheduling) - 4-warp pipelined loop
    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_full_tiles = scheduler.get_num_full_tiles()

    pid = scheduler.apply_chiplet_transform_chunked(pid, num_sms, num_xcds=8, chunk_size=2)

    for tile_idx in range(pid, num_full_tiles, num_sms):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        producer = 0
        consumer = 0
        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

        for i in ttgl.static_range(NUM_BUFFERS - 1):
            producer = issue_loads(
                producer,
                a_desc,
                b_desc,
                off_am,
                off_bn,
                a_buffer,
                b_buffer,
                BLOCK_K,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )

        for k_iter in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
            producer = issue_loads(
                producer,
                a_desc,
                b_desc,
                off_am,
                off_bn,
                a_buffer,
                b_buffer,
                BLOCK_K,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )
            consumer, accumulator = issue_wmma(
                consumer,
                a_buffer,
                OPERAND_LAYOUT_A,
                b_buffer,
                OPERAND_LAYOUT_B,
                accumulator,
                (NUM_BUFFERS - 1) * 2,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )

        for i in ttgl.static_range(NUM_BUFFERS - 1):
            consumer, accumulator = issue_wmma(
                consumer,
                a_buffer,
                OPERAND_LAYOUT_A,
                b_buffer,
                OPERAND_LAYOUT_B,
                accumulator,
                (NUM_BUFFERS - 2 - i) * 2,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )

        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)

    # Phase 2: Process StreamK tiles - 4-warp path
    process_streamk_tiles(
        a_desc,
        b_desc,
        a_buffer,
        b_buffer,
        c_ptr,
        p_ptr,
        locks_ptr,
        M,
        N,
        K,
        stride_cm,
        stride_cn,
        pid,
        num_sms,
        num_full_tiles,
        scheduler,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TRANSPOSE_B,
        STREAMK_TILES,
        WMMA_LAYOUT,
        OPERAND_LAYOUT_A,
        OPERAND_LAYOUT_B,
        GROUP_SIZE_M,
    )


@gluon.jit
def streamk_gemm_tdm_pipelined_kernel_8warps(
    a_ptr,
    b_ptr,
    c_ptr,
    p_ptr,
    locks_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    NUM_BUFFERS: ttgl.constexpr,
    TRANSPOSE_B: ttgl.constexpr,
    NUM_WARPS: ttgl.constexpr,
    WARP_BASES: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
):
    """
    StreamK GEMM kernel for 8 warps only.
    Phase 1: Warp-pipelined persistent loop (ping-pong pattern).
    Phase 2: Uses process_streamk_tiles_8warps (8-warp StreamK remainder path with quadrant splitting).
    """
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be 3")
    ttgl.static_assert(NUM_WARPS == 8, "This kernel is only valid for NUM_WARPS == 8")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])
    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc, b_desc = create_tensor_descriptors(
        a_ptr,
        b_ptr,
        0,
        0,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        SHARED_LAYOUT_A,
        SHARED_LAYOUT_B,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TRANSPOSE_B,
    )
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    # Initialize scheduler
    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES)

    # Phase 1: Process full tiles (persistent scheduling)
    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_full_tiles = scheduler.get_num_full_tiles()

    pid = scheduler.apply_chiplet_transform_chunked(pid, num_sms, num_xcds=8, chunk_size=2)

    for tile_idx in range(pid, num_full_tiles, num_sms):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        producer = 0
        consumer = 0
        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

        for i in ttgl.static_range(NUM_BUFFERS - 1):
            producer = issue_loads(
                producer,
                a_desc,
                b_desc,
                off_am,
                off_bn,
                a_buffer,
                b_buffer,
                BLOCK_K,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )

        for k_iter in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
            producer = issue_loads(
                producer,
                a_desc,
                b_desc,
                off_am,
                off_bn,
                a_buffer,
                b_buffer,
                BLOCK_K,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )
            consumer, accumulator = issue_wmma(
                consumer,
                a_buffer,
                OPERAND_LAYOUT_A,
                b_buffer,
                OPERAND_LAYOUT_B,
                accumulator,
                (NUM_BUFFERS - 1) * 2,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )

        for i in ttgl.static_range(NUM_BUFFERS - 1):
            consumer, accumulator = issue_wmma(
                consumer,
                a_buffer,
                OPERAND_LAYOUT_A,
                b_buffer,
                OPERAND_LAYOUT_B,
                accumulator,
                (NUM_BUFFERS - 2 - i) * 2,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )

        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)

    # Phase 2: Process StreamK tiles - 8-warp path with quadrant splitting
    process_streamk_tiles_8warps(
        a_desc,
        b_desc,
        a_buffer,
        b_buffer,
        c_ptr,
        p_ptr,
        locks_ptr,
        M,
        N,
        K,
        stride_cm,
        stride_cn,
        pid,
        num_sms,
        num_full_tiles,
        scheduler,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TRANSPOSE_B,
        STREAMK_TILES,
        WMMA_LAYOUT,
        OPERAND_LAYOUT_A,
        OPERAND_LAYOUT_B,
        WARP_BASES,
        GROUP_SIZE_M,
    )


@gluon.jit
def streamk_gemm_tdm_prefetch_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    p_ptr,
    locks_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    NUM_BUFFERS: ttgl.constexpr,
    TRANSPOSE_B: ttgl.constexpr,
    NUM_WARPS: ttgl.constexpr,
    WARP_BASES: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
):
    """
    StreamK GEMM kernel for 4 warps with TDM, software pipelining, and prologue-epilogue overlap.
    This variant prefetches data for the NEXT tile during the epilogue of the current tile.
    Phase 1: Prefetch-optimized persistent loop.
    Phase 2: Uses process_streamk_tiles (4-warp StreamK remainder path).
    """
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    ttgl.static_assert(NUM_WARPS == 4, "This kernel is only valid for NUM_WARPS == 4")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])
    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc, b_desc = create_tensor_descriptors(
        a_ptr,
        b_ptr,
        0,
        0,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        SHARED_LAYOUT_A,
        SHARED_LAYOUT_B,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TRANSPOSE_B,
    )
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    # Initialize scheduler
    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES)

    # Phase 1: Process full tiles with TRUE prologue-epilogue overlap
    # Uses separate k_counter and buffer_slot tracking to enable cross-tile prefetch
    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_full_tiles = scheduler.get_num_full_tiles()

    pid = scheduler.apply_chiplet_transform_chunked(pid, num_sms, num_xcds=8, chunk_size=2)

    num_k_iters = ttgl.cdiv(K, BLOCK_K)

    # ===== Initial prologue: Prefetch for FIRST tile =====
    # buffer_slot tracks which shared memory buffer to use (mod NUM_BUFFERS)
    # k_counter tracks which k iteration we're prefetching for current tile
    buffer_slot = 0
    first_tile_idx = pid
    if first_tile_idx < num_full_tiles:
        pid_m_first, pid_n_first = scheduler.get_swizzled_tile_coords(first_tile_idx, GROUP_SIZE_M)
        off_am_first = pid_m_first * BLOCK_M
        off_bn_first = pid_n_first * BLOCK_N

        # Prefill first NUM_BUFFERS-1 slots with first tile's data
        for i in ttgl.static_range(NUM_BUFFERS - 1):
            k_offset = i * BLOCK_K
            slot = buffer_slot % NUM_BUFFERS
            ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am_first, k_offset], a_buffer.index(slot))
            if not TRANSPOSE_B:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_offset, off_bn_first], b_buffer.index(slot))
            else:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn_first, k_offset], b_buffer.index(slot))
            buffer_slot += 1

    # k_counter for current tile (starts at NUM_BUFFERS-1 because we prefilled)
    k_counter = NUM_BUFFERS - 1
    consumer_slot = 0

    for tile_idx in range(pid, num_full_tiles, num_sms):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

        for _ in range(0, num_k_iters - (NUM_BUFFERS - 1)):
            # Issue load for current tile
            k_offset = k_counter * BLOCK_K
            slot = buffer_slot % NUM_BUFFERS
            ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, k_offset], a_buffer.index(slot))
            if not TRANSPOSE_B:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_offset, off_bn], b_buffer.index(slot))
            else:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, k_offset], b_buffer.index(slot))
            buffer_slot += 1
            k_counter += 1

            # Wait and consume
            ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
            cons_slot = consumer_slot % NUM_BUFFERS
            a_operand = a_buffer.index(cons_slot).load(layout=OPERAND_LAYOUT_A)
            if not TRANSPOSE_B:
                b_operand = b_buffer.index(cons_slot).load(layout=OPERAND_LAYOUT_B)
            else:
                b_operand = (b_buffer.index(cons_slot).permute([1, 0]).load(layout=OPERAND_LAYOUT_B))
            accumulator = ttgl.amd.gfx1250.wmma(a_operand, b_operand, accumulator)
            consumer_slot += 1

        next_tile_idx = tile_idx + num_sms
        has_next_tile = next_tile_idx < num_full_tiles
        if has_next_tile:
            pid_m_next, pid_n_next = scheduler.get_swizzled_tile_coords(next_tile_idx, GROUP_SIZE_M)
            off_am_next = pid_m_next * BLOCK_M
            off_bn_next = pid_n_next * BLOCK_N
        else:
            off_am_next = off_am
            off_bn_next = off_bn

        for i in ttgl.static_range(NUM_BUFFERS - 1):
            ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
            cons_slot = consumer_slot % NUM_BUFFERS
            a_operand = a_buffer.index(cons_slot).load(layout=OPERAND_LAYOUT_A)
            if not TRANSPOSE_B:
                b_operand = b_buffer.index(cons_slot).load(layout=OPERAND_LAYOUT_B)
            else:
                b_operand = (b_buffer.index(cons_slot).permute([1, 0]).load(layout=OPERAND_LAYOUT_B))
            accumulator = ttgl.amd.gfx1250.wmma(a_operand, b_operand, accumulator)
            consumer_slot += 1

            # Prefetch for next tile (overlapped with current drain)
            if has_next_tile:
                k_offset_next = i * BLOCK_K
                slot = buffer_slot % NUM_BUFFERS
                ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am_next, k_offset_next], a_buffer.index(slot))
                if not TRANSPOSE_B:
                    ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_offset_next, off_bn_next], b_buffer.index(slot))
                else:
                    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn_next, k_offset_next], b_buffer.index(slot))
                buffer_slot += 1

        # Reset k_counter for next tile (it starts at NUM_BUFFERS-1 because we just prefilled)
        k_counter = NUM_BUFFERS - 1

        # Store result
        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)

    # Phase 2: Process StreamK tiles
    process_streamk_tiles(
        a_desc,
        b_desc,
        a_buffer,
        b_buffer,
        c_ptr,
        p_ptr,
        locks_ptr,
        M,
        N,
        K,
        stride_cm,
        stride_cn,
        pid,
        num_sms,
        num_full_tiles,
        scheduler,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TRANSPOSE_B,
        STREAMK_TILES,
        WMMA_LAYOUT,
        OPERAND_LAYOUT_A,
        OPERAND_LAYOUT_B,
        GROUP_SIZE_M,
    )


def run_streamk_gemm_tdm_pipelined(
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    NUM_BUFFERS,
    TRANSPOSE_B,
    M,
    N,
    K,
    num_warps,
    use_prefetch=False,
    disable_streamk=False,
):
    """
      Helper function for StreamK GEMM kernel testing.
    """
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        print(f"Skipping: K/BLOCK_K ({triton.cdiv(K, BLOCK_K)}) < NUM_BUFFERS ({NUM_BUFFERS})")
        return

    # num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    # NOTE: Explicitly set num_sms to small number to ensure that each CU will compute multiple tiles.
    num_sms = 8
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    STREAMK_TILES = 0 if disable_streamk else (total_tiles % num_sms)

    torch.manual_seed(42)

    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    if TRANSPOSE_B:
        b = b.T.contiguous()
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = ((b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0)))
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    # Use persistent grid (StreamK uses persistent kernel infrastructure)
    num_sms = 8
    grid = (min(num_sms, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), 1)

    # Allocate StreamK buffers
    p = torch.empty(num_sms * BLOCK_M * BLOCK_N, dtype=torch.float32)
    locks = torch.empty(num_sms, dtype=torch.int32)

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = c.cuda()
    p_device = p.cuda()
    locks_device = locks.cuda()

    if num_warps == 8:
        kernel_name = "pipelined-8warps"
    elif use_prefetch:
        kernel_name = "prefetch-4warps"
    else:
        kernel_name = "pipelined-4warps"
    print(f"\nTesting StreamK {kernel_name} kernel with STREAMK_TILES={STREAMK_TILES}")
    print(f"Grid: {grid}, Total tiles: {total_tiles}")

    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    # Select kernel based on num_warps and use_prefetch flag
    if num_warps == 8:
        kernel_fn = streamk_gemm_tdm_pipelined_kernel_8warps
    elif use_prefetch:
        kernel_fn = streamk_gemm_tdm_prefetch_kernel
    else:
        kernel_fn = streamk_gemm_tdm_pipelined_kernel_4warps
    kernel = kernel_fn[grid](
        a_device,
        b_device,
        c_device,
        p_device,
        locks_device,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        TRANSPOSE_B=TRANSPOSE_B,
        NUM_WARPS=num_warps,
        WARP_BASES=warp_bases,  #
        STREAMK_TILES=STREAMK_TILES,
        num_warps=num_warps,
        waves_per_eu=num_warps // 4,
    )
    static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ (b.to(torch.float32) if not TRANSPOSE_B else b.T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)
    print(f"✓ StreamK {kernel_name} kernel test passed!")


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (258, 258, 510)])
@pytest.mark.parametrize("use_prefetch", [False, True])
def test_streamk_gemm_tdm_4warps(
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    NUM_BUFFERS,
    TRANSPOSE_B,
    M,
    N,
    K,
    use_prefetch,
):
    """Test 4-warp StreamK GEMM kernel (both pipelined and prefetch variants)."""
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    run_streamk_gemm_tdm_pipelined(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BUFFERS,
        TRANSPOSE_B,
        M,
        N,
        K,
        num_warps=4,
        use_prefetch=use_prefetch,
    )


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [3])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (258, 258, 510)])
def test_streamk_gemm_tdm_8warps(
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    NUM_BUFFERS,
    TRANSPOSE_B,
    M,
    N,
    K,
):
    """Test 8-warp StreamK GEMM kernel."""
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    run_streamk_gemm_tdm_pipelined(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BUFFERS,
        TRANSPOSE_B,
        M,
        N,
        K,
        num_warps=8,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="StreamK GEMM kernel test - automatically calculates StreamK tiles for load balancing",
        epilog="Example: python3 f16_sk_gemm_gfx1250.py -M 258 -N 258 -K 510",
    )
    parser.add_argument("-M", type=int, default=1028, help="problem M size (default: 258)")
    parser.add_argument("-N", type=int, default=1028, help="problem N size (default: 258)")
    parser.add_argument("-K", type=int, default=1024, help="problem K size (default: 510)")
    parser.add_argument("--block-m", type=int, default=256, help="BLOCK_M tile size (default: 32)")
    parser.add_argument("--block-n", type=int, default=256, help="BLOCK_N tile size (default: 32)")
    parser.add_argument("--block-k", type=int, default=128, help="BLOCK_K tile size (default: 128)")
    parser.add_argument(
        "--num-warps",
        type=int,
        choices=[4, 8],
        default=4,
        help="num warps (default: 4)",
    )
    parser.add_argument(
        "--num-buffers",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="num shared memory buffers (default: 2, use 3 for 8-warp warp-pipelining)",
    )
    parser.add_argument("--num-sms", type=int, default=8, help="number of SMs to use (default: 8)")
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Use prefetch kernel variant with prologue-epilogue overlap",
    )
    parser.add_argument(
        "--disable-streamk",
        action="store_true",
        help="Disable StreamK (STREAMK_TILES=0), use pure persistent mode",
    )
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    BLOCK_M, BLOCK_N, BLOCK_K = args.block_m, args.block_n, args.block_k
    NUM_BUFFERS = args.num_buffers
    NUM_WARPS = args.num_warps
    TRANSPOSE_B = True
    NUM_SMS = args.num_sms

    # Calculate tile dimensions for display
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    streamk_tiles = 0 if args.disable_streamk else (total_tiles % NUM_SMS)
    mode = ("Persistent (StreamK disabled)"
            if args.disable_streamk else f"StreamK with {streamk_tiles} remainder tiles")

    STREAMK = not args.disable_streamk
    print(f"Mode: {mode} (out of {total_tiles} total tiles)")
    print(
        f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {TRANSPOSE_B=}, {NUM_WARPS=}, {NUM_BUFFERS=}, PERSISTENT=True, {STREAMK=}, PREFETCH={args.prefetch}"
    )

    # Run StreamK kernel test
    run_streamk_gemm_tdm_pipelined(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BUFFERS,
        TRANSPOSE_B,
        M,
        N,
        K,
        NUM_WARPS,
        use_prefetch=args.prefetch,
        disable_streamk=args.disable_streamk,
    )
