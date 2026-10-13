r"""Clustered warp-pipeline GEMMs for AMD CDNA5.

Each kernel uses eight warps per CTA, FP32 accumulation and BF16 output.
BF16, MXFP4 and FP8 x MXFP4 use a 4x4 CTA cluster (1024x1024 output tile).
MXFP8 supports 2x2/4x4 clusters and 64/128/256-square CTA tiles. TDM multicasts
operands into LDS.

* BF16: BK128, two data slots and an unroll-2 load/compute/refill schedule.
* MXFP8: BK256 with CTA256 (two data/three scale slots), or BK512 with
  CTA64/128 (two data/scale slots); five stages and six-tile unroll.
* MXFP4: BK512, two data slots and three scale slots, six-tile unroll.
* FP8 x MXFP4: BK256, three slots each for A, B and B scales, six-tile unroll.

The three scaled kernels use five stages with phase gap one:
load low -> compute low -> load high/arrive/advance -> compute high
-> wait/refill.
Leading refill overlaps trailing compute after its LDS reads have completed.
TDM readiness waits and warp-pipeline boundaries protect buffer lifetimes;
cluster arrive/wait are scheduling hints. B reuse is not enabled here.

Run from this directory on CDNA5, for example::

    python gemm_warp_pipeline_cdna5.py --dtype all --check \
        --mxfp8-tile 256x256x256 --mxfp8-cluster 4x4
    python gemm_warp_pipeline_cdna5.py --dtype mxfp8 --check --benchmark \
        --mxfp8-tile 128x128x512 --mxfp8-cluster 4x4

Rotating buffers are opt-in and imply --benchmark::

    python gemm_warp_pipeline_cdna5.py --dtype mxfp8 \
        --rotate-inputs --repeats 100 \
        --mxfp8-tile 256x256x256 --mxfp8-cluster 4x4

Rotation clones A, B, scales and output with identical input values at distinct
addresses. It times one graph replay containing --repeats launches (default 100,
maximum 200), per dtype, excluding warmup. Samples and replays must both be one.
All copies must fit GPU memory; there is no fallback to a smaller pool.
Without rotation, timing defaults remain 4 samples x 20 replays x 50 launches.

MXFP8 requires explicit --mxfp8-tile and --mxfp8-cluster selections, including
when --dtype all is used. Unsupported configurations are rejected.
M/N must be divisible by CTA size times cluster width.
The other dtypes require M/N multiples of 1024. K constraints are checked
before launch.
MXFP8 --mxfp8-output-buffers 2 overlaps the BK256 N-half output stores using
two LDS buffers; the default is 1. BK512 supports only 1.
--flat-pipeline selects separate BK256/BK512 bodies with compile-time K and
fully expanded stages, including tails. It is opt-in and requires
--dtype mxfp8; all supported MXFP8 tiles are accepted; large K increases
compilation time and code size.
--input positive_mxfp8 generates positive FP8 values from uniform [0, 0.1),
with random scales rounded up to E8M0: K128 for A and N128xK128 for B, expanded
to block-32 scales. It requires --dtype mxfp8. --input-mode is an alias for
--input.
The default --input auto selects trig for BF16/MXFP8/MXFP4 and random for
FP8 x MXFP4. With --dtype all, this selection is made separately for each dtype.
--seed (default 42) controls random and positive_mxfp8 inputs, as well as the
random scales for FP8 x MXFP4 trig inputs. MXFP8/MXFP4 trig inputs always use
the fixed hipBLASLt seed 1713573849; BF16 trig inputs are deterministic without
an RNG. These three trig modes ignore --seed.
Timing excludes input generation, compilation and correctness checking.
No workspace-specific imports are needed.
"""

import argparse
import gc
import statistics
import numpy as np
import torch
import triton
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.cdna5 import PartitionedSharedLayout, tdm
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

try:
    from .f16_gemm_common_cdna5 import chiplet_transform
    from .mxfp_gemm_cdna5 import init_data, pack_scale, torch_gemm_mxfp
except ImportError:
    from f16_gemm_common_cdna5 import chiplet_transform
    from mxfp_gemm_cdna5 import init_data, pack_scale, torch_gemm_mxfp

BF16_BLOCK_K = 128
NUM_WARPS = 8
TRIG_SEED = 1713573849


@gluon.jit
def _get_xcd_swizzled_pids(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, GRID_MN: gl.constexpr,
                           NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pid = chiplet_transform(pid, GRID_MN, NUM_XCDS)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    return (pid_m, pid_n)


@gluon.jit
def _cluster_sync():
    gl.amd.cdna5.cluster.arrive()
    gl.amd.cdna5.cluster.wait()


@gluon.jit
def _tdm_store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, STORE_LAYOUT: gl.constexpr,
                         BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    c_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N // CTA_N, 8]], [BLOCK_M, BLOCK_N],
                                                                            [1, 0], STORE_LAYOUT.cga_layout)
    c_shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_shared_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        block_shape=(BLOCK_M, BLOCK_N), layout=c_shared_layout)
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


@gluon.jit
def _tdm_store_split_n2(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1, OUTPUT_CGA_LAYOUT: gl.constexpr,
                        CTA_M: gl.constexpr, CTA_N: gl.constexpr, TWO_BUFFERS: gl.constexpr = False):
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[half_n, 8]], [CTA_M, CTA_N, cta_m, half_n],
                                                                          [3, 2, 1, 0], OUTPUT_CGA_LAYOUT)
    shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout)
    if TWO_BUFFERS:
        shared1 = gl.allocate_shared_memory(c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout)
    output0 = acc0.reshape((CTA_M, cta_m, CTA_N, half_n)).permute((0, 2, 1, 3))
    output1 = acc1.reshape((CTA_M, cta_m, CTA_N, half_n)).permute((0, 2, 1, 3))
    desc = tdm.make_tensor_descriptor(base=c_ptr, shape=(M // cta_m, N // cta_n, cta_m, cta_n),
                                      strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
                                      block_shape=(CTA_M, CTA_N, cta_m, half_n), layout=shared_layout)
    base_m = pid_m * CTA_M
    base_n = pid_n * CTA_N
    shared.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared)
    if TWO_BUFFERS:
        shared1.store(output1.to(c_ptr.type.element_ty))
        tdm.async_store(desc, [base_m, base_n, 0, half_n], shared1)
    else:
        tdm.async_wait(0)
        shared.store(output1.to(c_ptr.type.element_ty))
        tdm.async_store(desc, [base_m, base_n, 0, half_n], shared)
    tdm.async_wait(0)


@gluon.jit
def _bf16_cluster_consume_and_refill(a_buf, b_buf, slot, refill_slot, a_desc, b_desc, acc, DOT_A: gl.constexpr,
                                     DOT_B: gl.constexpr, BLOCK_K: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    tdm.async_wait(0)
    with gl.amd.warp_pipeline_stage('stage0'):
        tdm.async_load(a_desc, [0, 0], a_buf.index(refill_slot), warp_used_hint=15)
        tdm.async_load(b_desc, [0, 0], b_buf.index(refill_slot), warp_used_hint=15)
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage('stage1'):
        acc = gl.amd.cdna5.wmma(a, b, acc)
    with gl.amd.warp_pipeline_stage('stage0'):
        a = a_buf.index(slot).slice(half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K])
        gl.amd.cdna5.cluster.arrive()
    with gl.amd.warp_pipeline_stage('stage1'):
        acc = gl.amd.cdna5.wmma(a, b, acc)
        gl.amd.cdna5.cluster.wait()
    return (a_desc, b_desc, acc)


@gluon.jit
def _split_b_n_halves(b, DOT_B: gl.constexpr, BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    b_grid = b.reshape((64, CTA_N, cta_n))
    b0 = gl.convert_layout(
        gl.amd.slice(b_grid, [64, CTA_N, half_n], [0, 0, 0]).reshape((64, packed_n)), DOT_B, assert_trivial=True)
    b1 = gl.convert_layout(
        gl.amd.slice(b_grid, [64, CTA_N, half_n], [0, 0, half_n]).reshape((64, packed_n)), DOT_B, assert_trivial=True)
    return (b0, b1)


@gluon.jit
def bf16_gemm_warp_pipeline(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                            stride_cn, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
                            SHARED_LAYOUT_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
                            BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    block_m: gl.constexpr = BLOCK_M
    block_n: gl.constexpr = BLOCK_N
    block_k: gl.constexpr = 128
    gl.static_assert(a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16())
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M == CTA_M * 256)
    gl.static_assert(BLOCK_N == CTA_N * 256)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, block_m, block_n, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 8)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + pid_m * block_m * stride_am, shape=(M, K),
                                        strides=(stride_am, stride_ak), block_shape=(block_m, block_k),
                                        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + pid_n * block_n * stride_bn, shape=(N, K),
                                        strides=(stride_bn, stride_bk), block_shape=(block_n, block_k),
                                        layout=SHARED_LAYOUT_B)
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, block_m, block_k], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, block_n, block_k], SHARED_LAYOUT_B)
    tdm.async_load(a_desc, [0, 0], a_buf.index(0), warp_used_hint=15)
    tdm.async_load(b_desc, [0, 0], b_buf.index(0), warp_used_hint=15)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, block_k])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, block_k])
    acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, block_k)
    gl.assume(iter_max >= 2)
    gl.assume(iter_max % 2 == 0)
    for _ in range(0, (iter_max - 2) // 2):
        a_desc, b_desc, acc = _bf16_cluster_consume_and_refill(a_buf, b_buf, 0, 1, a_desc, b_desc, acc, dot_a, dot_b,
                                                               block_k)
        a_desc, b_desc, acc = _bf16_cluster_consume_and_refill(a_buf, b_buf, 1, 0, a_desc, b_desc, acc, dot_a, dot_b,
                                                               block_k)
    a_desc, b_desc, acc = _bf16_cluster_consume_and_refill(a_buf, b_buf, 0, 1, a_desc, b_desc, acc, dot_a, dot_b,
                                                           block_k)
    tdm.async_wait(0)
    half_k: gl.constexpr = block_k // 2
    with gl.amd.warp_pipeline_stage('stage0'):
        a = a_buf.index(1).slice(0, half_k, 1).load(layout=dot_a)
        b = b_buf.index(1).slice(0, half_k, 1).permute([1, 0]).load(layout=dot_b)
    with gl.amd.warp_pipeline_stage('stage1'):
        acc = gl.amd.cdna5.wmma(a, b, acc)
    with gl.amd.warp_pipeline_stage('stage0'):
        a = a_buf.index(1).slice(half_k, half_k, 1).load(layout=dot_a)
        b = b_buf.index(1).slice(half_k, half_k, 1).permute([1, 0]).load(layout=dot_b)
    with gl.amd.warp_pipeline_stage('stage1'):
        acc = gl.amd.cdna5.wmma(a, b, acc)
    _tdm_store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, WMMA_LAYOUT, block_m, block_n, CTA_N)


@gluon.jit
def _mxfp8_load_scale(scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr, BLOCK_NONK: gl.constexpr,
                      BK_SCALE: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape((BLOCK_NONK // 128, BK_SCALE // 4, 32, 4, 4)).permute(
        (0, 3, 2, 1, 4)).reshape((BLOCK_NONK, BK_SCALE))
    return scale_slice.slice(0, BLOCK_NONK, 0).slice(start_k, SUBTILE_SCALE_K, 1).load(layout=LAYOUT)


@gluon.jit
def _mxfp8_issue_b2_parent_data_without_update(a_desc, b_desc, a_buf, b_buf, slot):
    tdm.async_load(a_desc, dest=a_buf.index(slot), warp_used_hint=15)
    tdm.async_load(b_desc, dest=b_buf.index(slot), warp_used_hint=15)


@gluon.jit
def _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, slot):
    _mxfp8_issue_b2_parent_data_without_update(a_desc, b_desc, a_buf, b_buf, slot)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 0, 256])
    return (a_desc, b_desc)


@gluon.jit
def _mxfp8_load_b2_parent(buffer, slot, start_n: gl.constexpr, start_k: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr,
                          DOT_B: gl.constexpr):
    rank3 = buffer.index(slot).slice(start_n, 128, 1).slice(start_k, 128, 2).permute([2, 0,
                                                                                      1]).load(layout=B2_LOAD_LAYOUT)
    rank2 = rank3.reshape((128, buffer.type.shape[1] * 128))
    return gl.convert_layout(rank2, DOT_B)


@gluon.jit
def _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, slot):
    tdm.async_load_fused([(as_desc, as_buf.index(slot), 3), (bs_desc, bs_buf.index(slot), 12)])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 1024])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])
    return (as_desc, bs_desc)


@gluon.jit
def _mxfp8_bk256_b2_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, refill_scale_slot, a_desc, b_desc, as_desc, bs_desc, acc0,
        acc1, DOT_A: gl.constexpr, DOT_B: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_A_LOAD: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr, REFILL_SCALE: gl.constexpr, DATA_SCALE_WAITCNT: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    tdm.async_wait(DATA_SCALE_WAITCNT)
    next_a_desc = a_desc
    next_b_desc = b_desc
    next_as_desc = as_desc
    next_bs_desc = bs_desc
    with gl.amd.warp_pipeline_stage('mxfp8_stage0_load_low', phase_gap=1):
        as0 = _mxfp8_load_scale(as_buf, scale_slot, 0, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as0 = gl.convert_layout(as0, SCALE_A_LAYOUT, assert_trivial=True)
        bs0 = _mxfp8_load_scale(bs_buf, scale_slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        bs0_hi = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(0, 128, 1).load(layout=DOT_A)
        b0_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 0, B2_LOAD_LAYOUT, DOT_B)
        b0_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 0, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage('mxfp8_stage1_compute_low'):
        acc0 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_lo, bs0_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_hi, bs0_hi, 'e4m3', acc1)
    with gl.amd.warp_pipeline_stage('mxfp8_stage2_load_high_advance'):
        a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(128, 128, 1).load(layout=DOT_A)
        as1 = _mxfp8_load_scale(as_buf, scale_slot, 4, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as1 = gl.convert_layout(as1, SCALE_A_LAYOUT, assert_trivial=True)
        bs1 = _mxfp8_load_scale(bs_buf, scale_slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        bs1_hi = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        b1_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 128, B2_LOAD_LAYOUT, DOT_B)
        b1_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 128, B2_LOAD_LAYOUT, DOT_B)
        gl.amd.cdna5.cluster.arrive()
        next_a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
        next_b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 0, 256])
        if REFILL_SCALE:
            next_as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 1024])
            next_bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])
    with gl.amd.warp_pipeline_stage('mxfp8_stage3_compute_high'):
        acc0 = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1_lo, bs1_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1_hi, bs1_hi, 'e4m3', acc1)
    with gl.amd.warp_pipeline_stage('mxfp8_stage4_wait_refill'):
        gl.amd.cdna5.cluster.wait()
        _mxfp8_issue_b2_parent_data_without_update(a_desc, b_desc, a_buf, b_buf, data_slot)
        if REFILL_SCALE:
            tdm.async_load_fused([(as_desc, as_buf.index(refill_scale_slot), 3),
                                  (bs_desc, bs_buf.index(refill_scale_slot), 12)])
        a_desc = next_a_desc
        b_desc = next_b_desc
        as_desc = next_as_desc
        bs_desc = next_bs_desc
    return (a_desc, b_desc, as_desc, bs_desc, acc0, acc1)


@gluon.jit
def _mxfp8_bk256_b2_consume_tail(a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc0, acc1, DOT_A: gl.constexpr,
                                 DOT_B: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
                                 SCALE_A_LOAD: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
                                 BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    with gl.amd.warp_pipeline_stage('b2_tail_load_low'):
        a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(0, 128, 1).load(layout=DOT_A)
        as0 = _mxfp8_load_scale(as_buf, scale_slot, 0, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as0 = gl.convert_layout(as0, SCALE_A_LAYOUT, assert_trivial=True)
        bs0 = _mxfp8_load_scale(bs_buf, scale_slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        bs0_hi = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        b0_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 0, B2_LOAD_LAYOUT, DOT_B)
        b0_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 0, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage('b2_tail_compute_low'):
        acc0 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_lo, bs0_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_hi, bs0_hi, 'e4m3', acc1)
    with gl.amd.warp_pipeline_stage('b2_tail_load_high'):
        a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(128, 128, 1).load(layout=DOT_A)
        as1 = _mxfp8_load_scale(as_buf, scale_slot, 4, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as1 = gl.convert_layout(as1, SCALE_A_LAYOUT, assert_trivial=True)
        bs1 = _mxfp8_load_scale(bs_buf, scale_slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        bs1_hi = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        b1_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 128, B2_LOAD_LAYOUT, DOT_B)
        b1_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 128, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage('b2_tail_compute_high'):
        acc0 = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1_lo, bs1_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1_hi, bs1_hi, 'e4m3', acc1)
    return (acc0, acc1)


@gluon.jit
def mxfp8_gemm_warp_pipeline(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                             stride_bn, stride_cm, stride_cn, stride_scale_a, stride_scale_b, GRID_MN: gl.constexpr,
                             SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr,
                             SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
                             LOAD_WMMA_LAYOUT: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr,
                             OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                             CTA_M: gl.constexpr, CTA_N: gl.constexpr, TWO_BUFFER_OUTPUT: gl.constexpr = False):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(1, LOAD_WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK_N // 2, 4])
    scale_b_load: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b_load, [BLOCK_N, 4])
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, CTA_N, 256, 256], SHARED_LAYOUT_B2)
    scale_slots: gl.constexpr = 3
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [scale_slots, BLOCK_M // 128, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [scale_slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
                                        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
                                        layout=SHARED_LAYOUT_A)
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    b_desc = tdm.make_tensor_descriptor(base=b_base, shape=(CTA_N, N // CTA_N, K),
                                        strides=(256 * stride_bn, stride_bn, stride_bk), block_shape=(CTA_N, 256, 256),
                                        layout=SHARED_LAYOUT_B2)
    as_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + pid_m * BLOCK_M // 128 * stride_scale_a,
                                         shape=(M // 128, K // 32 * 128), strides=(stride_scale_a, 1),
                                         block_shape=(BLOCK_M // 128, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * BLOCK_N // 128 * stride_scale_b,
                                         shape=(N // 128, K // 32 * 128), strides=(stride_scale_b, 1),
                                         block_shape=(BLOCK_N // 128, 1024), layout=SHARED_SCALE_B)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 0)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 0)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 1)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 1)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 2)
    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 4)
    gl.assume(iter_max % 2 == 0)
    data_scale_waitcnt: gl.constexpr = 4
    for _ in range(0, (iter_max - 4) // 6):
        for inner in gl.static_range(6):
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, inner % 2, inner % 3, inner % 3, a_desc, b_desc, as_desc, bs_desc, acc0,
                acc1, dot_a, dot_b, B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load,
                BLOCK_M, BLOCK_N, CTA_N, True, data_scale_waitcnt)
    for cleanup_pair in range(0, (iter_max - 4) % 6 // 2):
        cleanup_tile = cleanup_pair * 2
        s0 = cleanup_tile % 3
        s1 = (cleanup_tile + 1) % 3
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, 0, s0, s0, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
            B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
            data_scale_waitcnt)
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, 1, s1, s1, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
            B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
            data_scale_waitcnt)
    residual = iter_max - 4
    s0 = residual % 3
    s1 = (residual + 1) % 3
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, 0, s0, s0, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
        B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
        data_scale_waitcnt)
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, 1, s1, s1, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
        B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, False,
        data_scale_waitcnt)
    tdm.async_wait(3)
    acc0, acc1 = _mxfp8_bk256_b2_consume_tail(a_buf, b_buf, as_buf, bs_buf, 0, (iter_max - 2) % 3, acc0, acc1, dot_a,
                                              dot_b, B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout,
                                              scale_b_load, BLOCK_M, BLOCK_N, CTA_N)
    tdm.async_wait(0)
    acc0, acc1 = _mxfp8_bk256_b2_consume_tail(a_buf, b_buf, as_buf, bs_buf, 1, (iter_max - 1) % 3, acc0, acc1, dot_a,
                                              dot_b, B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout,
                                              scale_b_load, BLOCK_M, BLOCK_N, CTA_N)
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    _tdm_store_split_n2(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1, OUTPUT_CGA_LAYOUT, CTA_M, CTA_N,
                        TWO_BUFFER_OUTPUT)


@gluon.jit
def mxfp8_gemm_warp_pipeline_flat(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K: gl.constexpr, stride_am,
                                  stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scale_a, stride_scale_b,
                                  GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr,
                                  SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
                                  LOAD_WMMA_LAYOUT: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr,
                                  OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                  CTA_M: gl.constexpr, CTA_N: gl.constexpr, TWO_BUFFER_OUTPUT: gl.constexpr = False):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(1, LOAD_WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK_N // 2, 4])
    scale_b_load: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b_load, [BLOCK_N, 4])
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, CTA_N, 256, 256], SHARED_LAYOUT_B2)
    scale_slots: gl.constexpr = 3
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [scale_slots, BLOCK_M // 128, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [scale_slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
                                        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
                                        layout=SHARED_LAYOUT_A)
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    b_desc = tdm.make_tensor_descriptor(base=b_base, shape=(CTA_N, N // CTA_N, K),
                                        strides=(256 * stride_bn, stride_bn, stride_bk), block_shape=(CTA_N, 256, 256),
                                        layout=SHARED_LAYOUT_B2)
    as_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + pid_m * BLOCK_M // 128 * stride_scale_a,
                                         shape=(M // 128, K // 32 * 128), strides=(stride_scale_a, 1),
                                         block_shape=(BLOCK_M // 128, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * BLOCK_N // 128 * stride_scale_b,
                                         shape=(N // 128, K // 32 * 128), strides=(stride_scale_b, 1),
                                         block_shape=(BLOCK_N // 128, 1024), layout=SHARED_SCALE_B)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 0)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 0)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 1)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 1)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 2)
    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    # Expand every K tile before warp-pipeline lowering, including the drain below.
    iter_max: gl.constexpr = K // 256
    gl.static_assert(iter_max >= 4 and iter_max % 2 == 0)
    data_scale_waitcnt: gl.constexpr = 4
    for tile in gl.static_range(iter_max - 4):
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, tile % 2, tile % 3, tile % 3, a_desc, b_desc, as_desc, bs_desc, acc0, acc1,
            dot_a, dot_b, B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M,
            BLOCK_N, CTA_N, True, data_scale_waitcnt)
    residual = iter_max - 4
    s0 = residual % 3
    s1 = (residual + 1) % 3
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, 0, s0, s0, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
        B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
        data_scale_waitcnt)
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, 1, s1, s1, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
        B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, False,
        data_scale_waitcnt)
    tdm.async_wait(3)
    acc0, acc1 = _mxfp8_bk256_b2_consume_tail(a_buf, b_buf, as_buf, bs_buf, 0, (iter_max - 2) % 3, acc0, acc1, dot_a,
                                              dot_b, B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout,
                                              scale_b_load, BLOCK_M, BLOCK_N, CTA_N)
    tdm.async_wait(0)
    acc0, acc1 = _mxfp8_bk256_b2_consume_tail(a_buf, b_buf, as_buf, bs_buf, 1, (iter_max - 1) % 3, acc0, acc1, dot_a,
                                              dot_b, B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout,
                                              scale_b_load, BLOCK_M, BLOCK_N, CTA_N)
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    _tdm_store_split_n2(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1, OUTPUT_CGA_LAYOUT, CTA_M, CTA_N,
                        TWO_BUFFER_OUTPUT)


@gluon.jit
def _mxfp8_tiled_issue_refill(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, slot,
                              BLOCK_K: gl.constexpr, SCALE_STEP: gl.constexpr):
    tdm.async_load(a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, SCALE_STEP])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, SCALE_STEP])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def _mxfp8_tiled_load_scale(scale_buf, slot, start_k: gl.constexpr, layout: gl.constexpr):
    return scale_buf.index(slot).slice(start_k, 4, 1).load(layout=layout)


@gluon.jit
def _mxfp8_tiled_consume_tail(a_buf, b_buf, as_buf, bs_buf, slot, acc, dot_a: gl.constexpr, dot_b: gl.constexpr,
                              scale_a_layout: gl.constexpr, scale_b_layout: gl.constexpr, block_k: gl.constexpr):
    for k in gl.static_range(0, block_k, 128):
        with gl.amd.warp_pipeline_stage('tiled_tail_load'):
            a = a_buf.index(slot).slice(k, 128, 1).load(layout=dot_a)
            a_scale = _mxfp8_tiled_load_scale(as_buf, slot, k // 32, scale_a_layout)
            b = b_buf.index(slot).slice(k, 128, 1).permute([1, 0]).load(layout=dot_b)
            b_scale = _mxfp8_tiled_load_scale(bs_buf, slot, k // 32, scale_b_layout)
        with gl.amd.warp_pipeline_stage('tiled_tail_compute'):
            acc = gl.amd.cdna5.wmma_scaled(a, a_scale, 'e4m3', b, b_scale, 'e4m3', acc)
    return acc


@gluon.jit
def _mxfp8_tiled_bk512_five(a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc, bs_desc, acc,
                            dot_a: gl.constexpr, dot_b: gl.constexpr, scale_a_layout: gl.constexpr,
                            scale_b_layout: gl.constexpr):
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage('tiled_stage0_load_k0', phase_gap=1):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=dot_a)
        as0 = _mxfp8_tiled_load_scale(as_buf, slot, 0, scale_a_layout)
        bs0 = _mxfp8_tiled_load_scale(bs_buf, slot, 0, scale_b_layout)
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=dot_a)
        as1 = _mxfp8_tiled_load_scale(as_buf, slot, 4, scale_a_layout)
        bs1 = _mxfp8_tiled_load_scale(bs_buf, slot, 4, scale_b_layout)
        b0 = b_buf.index(slot).slice(0, 128, 1).permute([1, 0]).load(layout=dot_b)
        b1 = b_buf.index(slot).slice(128, 128, 1).permute([1, 0]).load(layout=dot_b)
    with gl.amd.warp_pipeline_stage('tiled_stage1_compute_k0'):
        acc = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0, bs0, 'e4m3', acc)
        acc = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1, bs1, 'e4m3', acc)
    with gl.amd.warp_pipeline_stage('tiled_stage2_load_k2'):
        a2 = a_buf.index(slot).slice(256, 128, 1).load(layout=dot_a)
        as2 = _mxfp8_tiled_load_scale(as_buf, slot, 8, scale_a_layout)
        b2 = b_buf.index(slot).slice(256, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs2 = _mxfp8_tiled_load_scale(bs_buf, slot, 8, scale_b_layout)
        a3 = a_buf.index(slot).slice(384, 128, 1).load(layout=dot_a)
        as3 = _mxfp8_tiled_load_scale(as_buf, slot, 12, scale_a_layout)
        b3 = b_buf.index(slot).slice(384, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs3 = _mxfp8_tiled_load_scale(bs_buf, slot, 12, scale_b_layout)
        gl.amd.cdna5.cluster.arrive()
    with gl.amd.warp_pipeline_stage('tiled_stage3_compute_k2'):
        acc = gl.amd.cdna5.wmma_scaled(a2, as2, 'e4m3', b2, bs2, 'e4m3', acc)
        acc = gl.amd.cdna5.wmma_scaled(a3, as3, 'e4m3', b3, bs3, 'e4m3', acc)
    with gl.amd.warp_pipeline_stage('tiled_stage4_wait_refill'):
        gl.amd.cdna5.cluster.wait()
        a_desc, b_desc, as_desc, bs_desc = _mxfp8_tiled_issue_refill(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                                                                     as_buf, bs_buf, slot, 512, 16)
    return (a_desc, b_desc, as_desc, bs_desc, acc)


@gluon.jit
def mxfp8_gemm_warp_pipeline_bk512(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak,
                                   stride_bk, stride_bn, stride_cm, stride_cn, stride_as, stride_bs,
                                   GRID_MN: gl.constexpr, SHARED_A: gl.constexpr, SHARED_B: gl.constexpr,
                                   SHARED_AS: gl.constexpr, SHARED_BS: gl.constexpr, WMMA: gl.constexpr,
                                   BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, CLUSTER: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CLUSTER * CLUSTER)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK_N, 4])
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, BLOCK_M, 512], SHARED_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, BLOCK_N, 512], SHARED_B)
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [2, BLOCK_M, 16], SHARED_AS)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [2, BLOCK_N, 16], SHARED_BS)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
                                        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 512), layout=SHARED_A)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
                                        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 512), layout=SHARED_B)
    as_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + pid_m * BLOCK_M * stride_as, shape=(M, K // 32),
                                         strides=(stride_as, 1), block_shape=(BLOCK_M, 16), layout=SHARED_AS)
    bs_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * BLOCK_N * stride_bs, shape=(N, K // 32),
                                         strides=(stride_bs, 1), block_shape=(BLOCK_N, 16), layout=SHARED_BS)
    for slot in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = _mxfp8_tiled_issue_refill(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                                                                     as_buf, bs_buf, slot, 512, 16)
    iter_max = gl.cdiv(K, 512)
    gl.assume(iter_max >= 3)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    for _ in range((iter_max - 2) // 6):
        for inner in gl.static_range(6):
            a_desc, b_desc, as_desc, bs_desc, acc = _mxfp8_tiled_bk512_five(a_buf, b_buf, as_buf, bs_buf, inner % 2,
                                                                            a_desc, b_desc, as_desc, bs_desc, acc,
                                                                            dot_a, dot_b, scale_a_layout,
                                                                            scale_b_layout)
    for tile in range((iter_max - 2) // 6 * 6, iter_max - 2):
        a_desc, b_desc, as_desc, bs_desc, acc = _mxfp8_tiled_bk512_five(a_buf, b_buf, as_buf, bs_buf, tile % 2, a_desc,
                                                                        b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
                                                                        scale_a_layout, scale_b_layout)
    tdm.async_wait(3)
    acc = _mxfp8_tiled_consume_tail(a_buf, b_buf, as_buf, bs_buf, (iter_max - 2) % 2, acc, dot_a, dot_b, scale_a_layout,
                                    scale_b_layout, 512)
    tdm.async_wait(0)
    acc = _mxfp8_tiled_consume_tail(a_buf, b_buf, as_buf, bs_buf, (iter_max - 1) % 2, acc, dot_a, dot_b, scale_a_layout,
                                    scale_b_layout, 512)
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    _tdm_store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, WMMA, BLOCK_M, BLOCK_N, CLUSTER)


@gluon.jit
def mxfp8_gemm_warp_pipeline_bk512_flat(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K: gl.constexpr, stride_am,
                                        stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_as, stride_bs,
                                        GRID_MN: gl.constexpr, SHARED_A: gl.constexpr, SHARED_B: gl.constexpr,
                                        SHARED_AS: gl.constexpr, SHARED_BS: gl.constexpr, WMMA: gl.constexpr,
                                        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, CLUSTER: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CLUSTER * CLUSTER)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK_N, 4])
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, BLOCK_M, 512], SHARED_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, BLOCK_N, 512], SHARED_B)
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [2, BLOCK_M, 16], SHARED_AS)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [2, BLOCK_N, 16], SHARED_BS)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
                                        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 512), layout=SHARED_A)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
                                        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 512), layout=SHARED_B)
    as_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + pid_m * BLOCK_M * stride_as, shape=(M, K // 32),
                                         strides=(stride_as, 1), block_shape=(BLOCK_M, 16), layout=SHARED_AS)
    bs_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * BLOCK_N * stride_bs, shape=(N, K // 32),
                                         strides=(stride_bs, 1), block_shape=(BLOCK_N, 16), layout=SHARED_BS)
    for slot in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = _mxfp8_tiled_issue_refill(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                                                                     as_buf, bs_buf, slot, 512, 16)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    iter_max: gl.constexpr = K // 512
    gl.static_assert(K % 512 == 0 and iter_max >= 3)
    for tile in gl.static_range(iter_max - 2):
        a_desc, b_desc, as_desc, bs_desc, acc = _mxfp8_tiled_bk512_five(a_buf, b_buf, as_buf, bs_buf, tile % 2, a_desc,
                                                                        b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
                                                                        scale_a_layout, scale_b_layout)
    tdm.async_wait(3)
    acc = _mxfp8_tiled_consume_tail(a_buf, b_buf, as_buf, bs_buf, (iter_max - 2) % 2, acc, dot_a, dot_b, scale_a_layout,
                                    scale_b_layout, 512)
    tdm.async_wait(0)
    acc = _mxfp8_tiled_consume_tail(a_buf, b_buf, as_buf, bs_buf, (iter_max - 1) % 2, acc, dot_a, dot_b, scale_a_layout,
                                    scale_b_layout, 512)
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    _tdm_store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, WMMA, BLOCK_M, BLOCK_N, CLUSTER)


@gluon.jit
def _fp8_mxfp4_load_scale(scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr, BLOCK_N: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape((BLOCK_N // 128, 2, 32, 4, 4)).permute((0, 3, 2, 1, 4)).reshape(
        (BLOCK_N, 8))
    return scale_slice.slice(0, BLOCK_N, 0).slice(start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def _fp8_mxfp4_split_scale(bs, SCALE_B_LAYOUT: gl.constexpr, BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    bs_grid = bs.reshape((CTA_N, cta_n, 4))
    bs_lo = gl.convert_layout(
        gl.amd.slice(bs_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
        assert_trivial=True)
    bs_hi = gl.convert_layout(
        gl.amd.slice(bs_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
        assert_trivial=True)
    return (bs_lo, bs_hi)


@gluon.jit
def _fp8_mxfp4_five_consume_and_refill(a_buf, b_buf, bs_buf, data_slot, scale_slot, a_desc, b_desc, bs_desc, acc0, acc1,
                                       DOT_A: gl.constexpr, DOT_B: gl.constexpr, DOT_B_LOAD: gl.constexpr,
                                       SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
                                       B_WARP_HINT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                       CTA_N: gl.constexpr):
    tdm.async_wait(4)
    with gl.amd.warp_pipeline_stage('fp8mxfp4_five0_load_low', phase_gap=1):
        a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(0, 128, 1).load(layout=DOT_A)
        bs0_full = _fp8_mxfp4_load_scale(bs_buf, scale_slot, 0, SCALE_B_LOAD, BLOCK_N)
        bs0, bs1 = _fp8_mxfp4_split_scale(bs0_full, SCALE_B_LAYOUT, BLOCK_N, CTA_N)
        b0_full = b_buf.index(data_slot).slice(0, BLOCK_N, 0).slice(0, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b0, b1 = _split_b_n_halves(b0_full, DOT_B, BLOCK_N, CTA_N)
    with gl.amd.warp_pipeline_stage('fp8mxfp4_five1_compute_low'):
        acc0 = gl.amd.cdna5.wmma_scaled(a0, None, 'e4m3', b0, bs0, 'e2m1', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a0, None, 'e4m3', b1, bs1, 'e2m1', acc1)
    with gl.amd.warp_pipeline_stage('fp8mxfp4_five2_load_high_advance'):
        a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(128, 128, 1).load(layout=DOT_A)
        bs1_full = _fp8_mxfp4_load_scale(bs_buf, scale_slot, 4, SCALE_B_LOAD, BLOCK_N)
        bs2, bs3 = _fp8_mxfp4_split_scale(bs1_full, SCALE_B_LAYOUT, BLOCK_N, CTA_N)
        b1_full = b_buf.index(data_slot).slice(0, BLOCK_N, 0).slice(64, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b2, b3 = _split_b_n_halves(b1_full, DOT_B, BLOCK_N, CTA_N)
        gl.amd.cdna5.cluster.arrive()
        next_a = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
        next_b = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 128])
        next_bs = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])
    with gl.amd.warp_pipeline_stage('fp8mxfp4_five3_compute_high'):
        acc0 = gl.amd.cdna5.wmma_scaled(a1, None, 'e4m3', b2, bs2, 'e2m1', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a1, None, 'e4m3', b3, bs3, 'e2m1', acc1)
    with gl.amd.warp_pipeline_stage('fp8mxfp4_five4_wait_refill'):
        gl.amd.cdna5.cluster.wait()
        tdm.async_load_fused([(a_desc, a_buf.index(data_slot), 3), (b_desc, b_buf.index(data_slot), B_WARP_HINT)])
        tdm.async_load(bs_desc, dest=bs_buf.index(scale_slot), warp_used_hint=15)
    return (next_a, next_b, next_bs, acc0, acc1)


@gluon.jit
def fp8_mxfp4_gemm_warp_pipeline(a_ptr, b_ptr, c_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                                 stride_cm, stride_cn, stride_scale, GRID_MN: gl.constexpr,
                                 SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
                                 SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
                                 LOAD_WMMA_LAYOUT: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
                                 BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr,
                                 GROUP_ALONG_N: gl.constexpr, GROUP_SIZE: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    # Group neighboring N tiles to favor A cache locality.
    if GROUP_ALONG_N:
        pid = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        pid = chiplet_transform(pid, GRID_MN, 8)
        num_pid_in_group = GROUP_SIZE * num_pid_m
        group_id = pid // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE
        group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE)
        pid_n = first_pid_n + pid % num_pid_in_group % group_size_n
        pid_m = pid % num_pid_in_group // group_size_n
    else:
        pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, GROUP_SIZE)
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(3, WMMA_LAYOUT.transposed, WMMA_LAYOUT.warp_bases,
                                                     WMMA_LAYOUT.reg_bases, [16, 16, 64], WMMA_LAYOUT.cga_layout)
    load_wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(3, LOAD_WMMA_LAYOUT.transposed, LOAD_WMMA_LAYOUT.warp_bases,
                                                          LOAD_WMMA_LAYOUT.reg_bases, [16, 16, 64],
                                                          LOAD_WMMA_LAYOUT.cga_layout)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(1, load_wmma_packed, 16)
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK_N // 2, 4])
    scale_b_load: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b_load, [BLOCK_N, 4])
    b_warp_hint: gl.constexpr = 12
    slots: gl.constexpr = 3
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [slots, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [slots, BLOCK_N, 128], SHARED_LAYOUT_B)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
                                        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
                                        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K // 2),
                                        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 128),
                                        layout=SHARED_LAYOUT_B)
    bs_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * BLOCK_N // 128 * stride_scale,
                                         shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
                                         block_shape=(BLOCK_N // 128, 1024), layout=SHARED_SCALE_B)
    for prefetch_idx in gl.static_range(slots):
        tdm.async_load_fused([(a_desc, a_buf.index(prefetch_idx), 3), (b_desc, b_buf.index(prefetch_idx), b_warp_hint)])
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 128])
        tdm.async_load(bs_desc, dest=bs_buf.index(prefetch_idx), warp_used_hint=15)
        bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])
    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= slots + 1)
    for _ in range(0, (iter_max - 3) // 6):
        for offset in gl.static_range(6):
            a_desc, b_desc, bs_desc, acc0, acc1 = _fp8_mxfp4_five_consume_and_refill(
                a_buf, b_buf, bs_buf, offset % 3, offset % 3, a_desc, b_desc, bs_desc, acc0, acc1, dot_a, dot_b,
                dot_b_load, scale_b_layout, scale_b_load, b_warp_hint, BLOCK_M, BLOCK_N, CTA_N)
    for tile_idx in range((iter_max - 3) // 6 * 6, iter_max - 3):
        a_desc, b_desc, bs_desc, acc0, acc1 = _fp8_mxfp4_five_consume_and_refill(a_buf, b_buf, bs_buf, tile_idx % 3,
                                                                                 tile_idx % 3, a_desc, b_desc, bs_desc,
                                                                                 acc0, acc1, dot_a, dot_b, dot_b_load,
                                                                                 scale_b_layout, scale_b_load,
                                                                                 b_warp_hint, BLOCK_M, BLOCK_N, CTA_N)
    for drain_idx in gl.static_range(slots):
        tdm.async_wait(2 * (slots - 1 - drain_idx))
        tile_idx = iter_max - slots + drain_idx
        with gl.amd.warp_pipeline_stage('fp8mxfp4_tail_load_k0'):
            a0 = a_buf.index(tile_idx % 3).slice(0, BLOCK_M, 0).slice(0, 128, 1).load(layout=dot_a)
            b0_full = b_buf.index(tile_idx % 3).slice(0, BLOCK_N, 0).slice(0, 64,
                                                                           1).permute([1, 0]).load(layout=dot_b_load)
            bs0_full = _fp8_mxfp4_load_scale(bs_buf, tile_idx % 3, 0, scale_b_load, BLOCK_N)
            b0, b1 = _split_b_n_halves(b0_full, dot_b, BLOCK_N, CTA_N)
            bs0, bs1 = _fp8_mxfp4_split_scale(bs0_full, scale_b_layout, BLOCK_N, CTA_N)
        with gl.amd.warp_pipeline_stage('fp8mxfp4_tail_compute_k0'):
            acc0 = gl.amd.cdna5.wmma_scaled(a0, None, 'e4m3', b0, bs0, 'e2m1', acc0)
            acc1 = gl.amd.cdna5.wmma_scaled(a0, None, 'e4m3', b1, bs1, 'e2m1', acc1)
        with gl.amd.warp_pipeline_stage('fp8mxfp4_tail_load_k1'):
            a1 = a_buf.index(tile_idx % 3).slice(0, BLOCK_M, 0).slice(128, 128, 1).load(layout=dot_a)
            b1_full = b_buf.index(tile_idx % 3).slice(0, BLOCK_N, 0).slice(64, 64,
                                                                           1).permute([1, 0]).load(layout=dot_b_load)
            bs1_full = _fp8_mxfp4_load_scale(bs_buf, tile_idx % 3, 4, scale_b_load, BLOCK_N)
            b2, b3 = _split_b_n_halves(b1_full, dot_b, BLOCK_N, CTA_N)
            bs2, bs3 = _fp8_mxfp4_split_scale(bs1_full, scale_b_layout, BLOCK_N, CTA_N)
        with gl.amd.warp_pipeline_stage('fp8mxfp4_tail_compute_k1'):
            acc0 = gl.amd.cdna5.wmma_scaled(a1, None, 'e4m3', b2, bs2, 'e2m1', acc0)
            acc1 = gl.amd.cdna5.wmma_scaled(a1, None, 'e4m3', b3, bs3, 'e2m1', acc1)
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    bs_buf._keep_alive()
    _tdm_store_split_n2(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1, OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)


@gluon.jit
def _mxfp4_load_scale(scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape((8, 4, 32, 4, 4)).permute((0, 3, 2, 1, 4)).reshape((1024, 16))
    return scale_slice.slice(0, 1024, 0).slice(start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def _mxfp4_issue_data_refill(a_desc, b_desc, a_buf, b_buf, slot):
    tdm.async_load_fused([(a_desc, a_buf.index(slot), 3), (b_desc, b_buf.index(slot), 12)])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 256])
    return (a_desc, b_desc)


@gluon.jit
def _mxfp4_issue_scale_refill(as_desc, bs_desc, as_buf, bs_buf, slot):
    tdm.async_load_fused([(as_desc, as_buf.index(slot), 3), (bs_desc, bs_buf.index(slot), 12)])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 2048])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 2048])
    return (as_desc, bs_desc)


@gluon.jit
def _mxfp4_consume_k256_split_slots(a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc,
                                    START_K_PACKED: gl.constexpr, START_SCALE_K: gl.constexpr, DOT_A: gl.constexpr,
                                    DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    with gl.amd.warp_pipeline_stage('mxfp4_u2_tail_load'):
        a0 = a_buf.index(data_slot).slice(0, 1024, 0).slice(START_K_PACKED, 64, 1).load(layout=DOT_A)
        b0 = b_buf.index(data_slot).slice(0, 1024, 0).slice(START_K_PACKED, 64, 1).permute([1, 0]).load(layout=DOT_B)
        as0 = _mxfp4_load_scale(as_buf, scale_slot, START_SCALE_K, SCALE_A_LAYOUT)
        bs0 = _mxfp4_load_scale(bs_buf, scale_slot, START_SCALE_K, SCALE_B_LAYOUT)
        a1 = a_buf.index(data_slot).slice(0, 1024, 0).slice(START_K_PACKED + 64, 64, 1).load(layout=DOT_A)
        b1 = b_buf.index(data_slot).slice(0, 1024, 0).slice(START_K_PACKED + 64, 64, 1).permute([1,
                                                                                                 0]).load(layout=DOT_B)
        as1 = _mxfp4_load_scale(as_buf, scale_slot, START_SCALE_K + 4, SCALE_A_LAYOUT)
        bs1 = _mxfp4_load_scale(bs_buf, scale_slot, START_SCALE_K + 4, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage('mxfp4_u2_tail_compute'):
        acc = gl.amd.cdna5.wmma_scaled(a0, as0, 'e2m1', b0, bs0, 'e2m1', acc)
        acc = gl.amd.cdna5.wmma_scaled(a1, as1, 'e2m1', b1, bs1, 'e2m1', acc)
    return acc


@gluon.jit
def _mxfp4_consume_tile_split_slots(a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc, DOT_A: gl.constexpr,
                                    DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    acc = _mxfp4_consume_k256_split_slots(a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc, 0, 0, DOT_A, DOT_B,
                                          SCALE_A_LAYOUT, SCALE_B_LAYOUT)
    return _mxfp4_consume_k256_split_slots(a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc, 128, 8, DOT_A,
                                           DOT_B, SCALE_A_LAYOUT, SCALE_B_LAYOUT)


@gluon.jit
def _mxfp4_five_stage_consume_and_refill(a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, a_desc, b_desc, as_desc,
                                         bs_desc, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
                                         SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
                                         REFILL_SCALE: gl.constexpr):
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage('mxfp4_five_stage0_load_low', phase_gap=1):
        a0 = a_buf.index(data_slot).slice(0, 1024, 0).slice(0, 64, 1).load(layout=DOT_A)
        as0 = _mxfp4_load_scale(as_buf, scale_slot, 0, SCALE_A_LAYOUT)
        bs0 = _mxfp4_load_scale(bs_buf, scale_slot, 0, SCALE_B_LAYOUT)
        a1 = a_buf.index(data_slot).slice(0, 1024, 0).slice(64, 64, 1).load(layout=DOT_A)
        as1 = _mxfp4_load_scale(as_buf, scale_slot, 4, SCALE_A_LAYOUT)
        bs1 = _mxfp4_load_scale(bs_buf, scale_slot, 4, SCALE_B_LAYOUT)
        b0 = b_buf.index(data_slot).slice(0, 1024, 0).slice(0, 64, 1).permute([1, 0]).load(layout=DOT_B)
        b1 = b_buf.index(data_slot).slice(0, 1024, 0).slice(64, 64, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage('mxfp4_five_stage1_compute_low'):
        acc = gl.amd.cdna5.wmma_scaled(a0, as0, 'e2m1', b0, bs0, 'e2m1', acc)
        acc = gl.amd.cdna5.wmma_scaled(a1, as1, 'e2m1', b1, bs1, 'e2m1', acc)
    with gl.amd.warp_pipeline_stage('mxfp4_five_stage2_load_high_advance'):
        a2 = a_buf.index(data_slot).slice(0, 1024, 0).slice(128, 64, 1).load(layout=DOT_A)
        b2 = b_buf.index(data_slot).slice(0, 1024, 0).slice(128, 64, 1).permute([1, 0]).load(layout=DOT_B)
        as2 = _mxfp4_load_scale(as_buf, scale_slot, 8, SCALE_A_LAYOUT)
        bs2 = _mxfp4_load_scale(bs_buf, scale_slot, 8, SCALE_B_LAYOUT)
        a3 = a_buf.index(data_slot).slice(0, 1024, 0).slice(192, 64, 1).load(layout=DOT_A)
        b3 = b_buf.index(data_slot).slice(0, 1024, 0).slice(192, 64, 1).permute([1, 0]).load(layout=DOT_B)
        as3 = _mxfp4_load_scale(as_buf, scale_slot, 12, SCALE_A_LAYOUT)
        bs3 = _mxfp4_load_scale(bs_buf, scale_slot, 12, SCALE_B_LAYOUT)
        gl.amd.cdna5.cluster.arrive()
        next_a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
        next_b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 256])
        next_as_desc = as_desc
        next_bs_desc = bs_desc
        if REFILL_SCALE:
            next_as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 2048])
            next_bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 2048])
    with gl.amd.warp_pipeline_stage('mxfp4_five_stage3_compute_high'):
        acc = gl.amd.cdna5.wmma_scaled(a2, as2, 'e2m1', b2, bs2, 'e2m1', acc)
        acc = gl.amd.cdna5.wmma_scaled(a3, as3, 'e2m1', b3, bs3, 'e2m1', acc)
    with gl.amd.warp_pipeline_stage('mxfp4_five_stage4_wait_refill'):
        gl.amd.cdna5.cluster.wait()
        tdm.async_load_fused([(a_desc, a_buf.index(data_slot), 3), (b_desc, b_buf.index(data_slot), 12)])
        if REFILL_SCALE:
            tdm.async_load_fused([(as_desc, as_buf.index(scale_slot), 3), (bs_desc, bs_buf.index(scale_slot), 12)])
    return (next_a_desc, next_b_desc, next_as_desc, next_bs_desc, acc)


@gluon.jit
def mxfp4_gemm_warp_pipeline(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                             stride_bn, stride_cm, stride_cn, stride_scale, GRID_MN: gl.constexpr,
                             SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
                             SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, 1024, 1024, GRID_MN, 8, 4)
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(3, WMMA_LAYOUT.transposed, WMMA_LAYOUT.warp_bases,
                                                     WMMA_LAYOUT.reg_bases, [32, 16, 64], WMMA_LAYOUT.cga_layout)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_packed, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [1024, 4])
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [1024, 4])
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [3, 8, 2048], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [3, 8, 2048], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + pid_m * 1024 * stride_am, shape=(M, K // 2),
                                        strides=(stride_am, stride_ak), block_shape=(1024, 256), layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + pid_n * 1024 * stride_bn, shape=(N, K // 2),
                                        strides=(stride_bn, stride_bk), block_shape=(1024, 256), layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + pid_m * 1024 // 128 * stride_scale,
                                         shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
                                         block_shape=(8, 2048), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * 1024 // 128 * stride_scale,
                                         shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
                                         block_shape=(8, 2048), layout=SHARED_SCALE_B)
    a_desc, b_desc = _mxfp4_issue_data_refill(a_desc, b_desc, a_buf, b_buf, 0)
    as_desc, bs_desc = _mxfp4_issue_scale_refill(as_desc, bs_desc, as_buf, bs_buf, 0)
    a_desc, b_desc = _mxfp4_issue_data_refill(a_desc, b_desc, a_buf, b_buf, 1)
    as_desc, bs_desc = _mxfp4_issue_scale_refill(as_desc, bs_desc, as_buf, bs_buf, 1)
    as_desc, bs_desc = _mxfp4_issue_scale_refill(as_desc, bs_desc, as_buf, bs_buf, 2)
    acc = gl.zeros((1024, 1024), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 512)
    gl.assume(iter_max >= 4)
    gl.assume(iter_max % 2 == 0)
    for _ in range(0, (iter_max - 4) // 6):
        for offset in gl.static_range(6):
            a_desc, b_desc, as_desc, bs_desc, acc = _mxfp4_five_stage_consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, offset % 2, offset % 3, a_desc, b_desc, as_desc, bs_desc, acc, dot_a,
                dot_b, scale_a_layout, scale_b_layout, True)
    for tile_idx in range((iter_max - 4) // 6 * 6, iter_max - 4):
        a_desc, b_desc, as_desc, bs_desc, acc = _mxfp4_five_stage_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, tile_idx % 2, tile_idx % 3, a_desc, b_desc, as_desc, bs_desc, acc, dot_a,
            dot_b, scale_a_layout, scale_b_layout, True)
    residual_tile = iter_max - 4
    residual_scale0 = residual_tile % 3
    residual_scale1 = (residual_tile + 1) % 3
    a_desc, b_desc, as_desc, bs_desc, acc = _mxfp4_five_stage_consume_and_refill(a_buf, b_buf, as_buf, bs_buf, 0,
                                                                                 residual_scale0, a_desc, b_desc,
                                                                                 as_desc, bs_desc, acc, dot_a, dot_b,
                                                                                 scale_a_layout, scale_b_layout, True)
    a_desc, b_desc, as_desc, bs_desc, acc = _mxfp4_five_stage_consume_and_refill(a_buf, b_buf, as_buf, bs_buf, 1,
                                                                                 residual_scale1, a_desc, b_desc,
                                                                                 as_desc, bs_desc, acc, dot_a, dot_b,
                                                                                 scale_a_layout, scale_b_layout, False)
    tdm.async_wait(2)
    acc = _mxfp4_consume_tile_split_slots(a_buf, b_buf, as_buf, bs_buf, 0, (iter_max - 2) % 3, acc, dot_a, dot_b,
                                          scale_a_layout, scale_b_layout)
    tdm.async_wait(0)
    acc = _mxfp4_consume_tile_split_slots(a_buf, b_buf, as_buf, bs_buf, 1, (iter_max - 1) % 3, acc, dot_a, dot_b,
                                          scale_a_layout, scale_b_layout)
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    _tdm_store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, WMMA_LAYOUT, 1024, 1024, 4)


def _trig_scale_chunks(rows, k):
    """Yield value bits and block-32 scales matching hipBLASLt's trig generator.

    Preserve its 32 independently seeded MT19937 streams, contiguous block
    ranges, and libstdc++ double generation order. Chunking bounds memory use.
    """
    if k % 32:
        raise ValueError('MXFP trig generation requires K divisible by 32')
    block_size = 32
    num_threads = 32
    blocks = rows * k // block_size
    if blocks % num_threads:
        raise ValueError('MXFP trig generation requires scale blocks divisible by 32')
    blocks_per_thread = blocks // num_threads
    blocks_per_chunk = 32 * 1024
    output_block = 0
    for thread in range(num_threads):
        rng = np.random.RandomState(TRIG_SEED + thread)
        remaining = blocks_per_thread
        while remaining:
            chunk_blocks = min(remaining, blocks_per_chunk)
            count = chunk_blocks * block_size
            # Match libstdc++ generate_canonical<double, 53>: low word first.
            raw = rng.randint(0, 2**32, size=(count, 2), dtype=np.uint32)
            uniform = (raw[:, 0].astype(np.float64) + raw[:, 1].astype(np.float64) * 2.0**32) * 2.0**(-64)
            bits = np.cos(uniform * (2.0 * np.pi)).view(np.uint64).reshape(chunk_blocks, block_size)
            exponent = (bits >> 52 & 2047).astype(np.int32) - 1023
            element_scales = np.maximum(exponent, -127) + 127
            block_scales = np.floor(element_scales.mean(axis=1) + 0.5).astype(np.int32)
            adjusted = element_scales - block_scales[:, None]
            yield output_block, bits, adjusted, block_scales.astype(np.uint8)
            output_block += chunk_blocks
            remaining -= chunk_blocks


def _make_mxfp8_trig_matrix(rows, k):
    """Encode trig inputs as E4M3 values with E8M0 block scales."""
    data = torch.empty((rows, k), dtype=torch.uint8)
    scales = torch.empty((rows, k // 32), dtype=torch.uint8)
    data_flat = data.numpy().reshape(-1)
    scale_flat = scales.numpy().reshape(-1)
    for output_block, bits, adjusted, block_scales in _trig_scale_chunks(rows, k):
        sign = (bits >> 63 << 7).astype(np.uint8)
        mantissa = ((bits & (1 << 52) - 1) >> 49).astype(np.uint8)
        encoded = np.zeros(bits.shape, dtype=np.uint8)
        normal = adjusted >= -6
        encoded[normal] = sign[normal] | (adjusted[normal] + 7).astype(np.uint8) << 3 | mantissa[normal]
        subnormal = (adjusted >= -9) & (adjusted < -6)
        subnormal_mantissa = mantissa[subnormal] >> 1 | 4
        subnormal_mantissa >>= (-7 - adjusted[subnormal]).astype(np.uint8)
        encoded[subnormal] = sign[subnormal] | subnormal_mantissa
        data_begin = output_block * 32
        data_flat[data_begin:data_begin + encoded.size] = encoded.reshape(-1)
        scale_flat[output_block:output_block + block_scales.size] = block_scales
    return (data, scales)


def _make_mxfp4_trig_matrix(rows, k):
    """Encode trig inputs as E2M1 values with E8M0 block scales."""
    data = torch.empty((rows, k), dtype=torch.uint8)
    scales = torch.empty((rows, k // 32), dtype=torch.uint8)
    data_flat = data.numpy().reshape(-1)
    scale_flat = scales.numpy().reshape(-1)
    for output_block, bits, adjusted, block_scales in _trig_scale_chunks(rows, k):
        sign = (bits >> 63 << 3).astype(np.uint8)
        mantissa = ((bits & (1 << 52) - 1) >> 51).astype(np.uint8)
        encoded = np.zeros(bits.shape, dtype=np.uint8)
        normal = adjusted >= 0
        encoded[normal] = sign[normal] | np.minimum((adjusted[normal] + 1) * 2 + mantissa[normal], 7).astype(np.uint8)
        subnormal = adjusted == -1
        encoded[subnormal] = sign[subnormal] | 1
        data_begin = output_block * 32
        data_flat[data_begin:data_begin + encoded.size] = encoded.reshape(-1)
        scale_flat[output_block:output_block + block_scales.size] = block_scales
    return (data, scales)


def validate_gemm_shape(dtype, M, N, K, tile=None, cluster=None):
    """Apply the same dimension constraints to CLI and Python callers."""
    if dtype == 'mxfp8':
        return mxfp8_config(M, N, K, tile, cluster)
    constraints = {'bf16': (256, 256), 'mxfp4': (1024, 2048), 'fp8_mxfp4': (256, 1024)}
    if dtype not in constraints:
        raise ValueError(f'unsupported GEMM dtype: {dtype}')
    if M <= 0 or N <= 0 or M % 1024 or N % 1024:
        raise ValueError('M and N must be positive multiples of 1024')
    multiple, minimum = constraints[dtype]
    if K < minimum or K % multiple:
        raise ValueError(f'{dtype} requires K >= {minimum} and divisible by {multiple}')


def make_bf16_case(args):
    validate_gemm_shape('bf16', args.M, args.N, args.K)
    torch.manual_seed(args.seed)
    if args.input_mode == 'random':
        a = torch.randn((args.M, args.K), dtype=torch.bfloat16, device='cuda')
        b = torch.randn((args.N, args.K), dtype=torch.bfloat16, device='cuda')
    else:
        a = torch.empty((args.M, args.K), dtype=torch.bfloat16, device='cuda')
        b = torch.empty((args.N, args.K), dtype=torch.bfloat16, device='cuda')
        chunk = 8 * 1024 * 1024
        for output, cosine in ((a, False), (b, True)):
            flat = output.view(-1)
            for begin in range(0, flat.numel(), chunk):
                end = min(begin + chunk, flat.numel())
                angle = torch.arange(begin, end, dtype=torch.float64, device='cuda')
                value = torch.cos(angle) if cosine else torch.sin(angle)
                flat[begin:end].copy_(value.float())
    c = torch.zeros((args.M, args.N), dtype=torch.bfloat16, device='cuda')
    block_m = block_n = 1024
    cga_layout_c = make_cga_layout([4, 4], [4, 4], [0, 1])
    slice_m = block_m // 4
    slice_n = block_n // 4
    local_a = gl.PaddedSharedLayout.with_identity_for([[BF16_BLOCK_K, 8]], [block_m, BF16_BLOCK_K], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for([[BF16_BLOCK_K, 8]], [block_n, BF16_BLOCK_K], [1, 0])
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(block_m, block_n, local_a, local_b, NUM_WARPS,
                                                                 [16, 16, 32], a_transposed=False, b_transposed=True,
                                                                 slice_m=slice_m, slice_n=slice_n, transposed=True)
    wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
                                local_wmma.instr_shape, cga_layout_c)
    dot_a = gl.DotOperandLayout(0, wmma, 8)
    dot_b = gl.DotOperandLayout(1, wmma, 8)
    cga_a = dot_a.cga_layout
    cga_b = tuple((tuple([basis[1], basis[0]]) for basis in dot_b.cga_layout))
    padded_a = gl.PaddedSharedLayout.with_identity_for([[BF16_BLOCK_K, 8]], [block_m, BF16_BLOCK_K], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for([[BF16_BLOCK_K, 8]], [block_n, BF16_BLOCK_K], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(slice_m, slice_n, padded_a, padded_b, NUM_WARPS,
                                                                      [16, 16, 32], a_transposed=False,
                                                                      b_transposed=True, slice_m=slice_m,
                                                                      slice_n=slice_n, transposed=True)
    grid = (triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch(a=a, b=b, c=c):
        return bf16_gemm_warp_pipeline[grid](a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1), b.stride(1),
                                             b.stride(0), c.stride(0), c.stride(1), GRID_MN=grid[0],
                                             SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b, WMMA_LAYOUT=wmma,
                                             BLOCK_M=1024, BLOCK_N=1024, CTA_M=4, CTA_N=4, num_warps=8, waves_per_eu=2,
                                             num_ctas=16)

    def check():
        c.zero_()
        launch()
        torch.cuda.synchronize()
        reference = (a.cpu().float() @ b.cpu().T.float()).to(torch.bfloat16)
        torch.testing.assert_close(c.cpu(), reference, rtol=0.01, atol=0.01)
        print('result verified', flush=True)

    return (launch, check, (a, b, c))


def make_fp8_mxfp4_case(args):
    validate_gemm_shape('fp8_mxfp4', args.M, args.N, args.K)
    block_m = block_n = 1024
    torch.manual_seed(args.seed)
    if args.input_mode == 'trig':
        a = torch.empty((args.M, args.K), dtype=torch.float8_e4m3fn)
        flat = a.view(-1)
        chunk = 8 * 1024 * 1024
        for begin in range(0, flat.numel(), chunk):
            end = min(begin + chunk, flat.numel())
            angle = torch.arange(begin, end, dtype=torch.float64)
            values = torch.sin(angle)
            flat[begin:end].copy_(values.float())
        b = MXFP4Tensor(size=(args.K, args.N))
        encoded = torch.empty(args.K * args.N, dtype=torch.uint8)
        chunk = 8 * 1024 * 1024
        for begin in range(0, encoded.numel(), chunk):
            end = min(begin + chunk, encoded.numel())
            angle = torch.arange(begin, end, dtype=torch.float64)
            encoded[begin:end].copy_(MXFP4Tensor(data=torch.cos(angle).float()).data)
        b.data = encoded.view(args.K, args.N)
    else:
        a = init_data('float8_e4m3', args.M, args.K)
        b = init_data('float4', args.K, args.N)
    scale_k = args.K // 32
    b_scale_obj = MXScaleTensor(size=(args.N, scale_k)).random(low=1.0, high=32.0)
    reference = None
    if args.check:
        a_scale_obj = MXScaleTensor(data=torch.ones((args.M, scale_k), dtype=torch.float32))
        reference = torch_gemm_mxfp(a, b, a_scale_obj, b_scale_obj, 32, args.M, args.N, args.K).to(torch.bfloat16)
    a_d = a.contiguous().cuda()
    b_d = b.to_packed_tensor(dim=0).data.T.contiguous().cuda()
    bs_d = pack_scale(b_scale_obj.data).cuda()
    c_d = torch.empty((args.M, args.N), dtype=torch.bfloat16, device='cuda')
    packed_n = 512
    cga = make_cga_layout([4, 4], [4, 4], [0, 1])
    output_cga = tuple((tuple(basis) + (0, 0) for basis in cga))
    local_a = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [block_m, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for([[128, 16]], [packed_n, 128], [1, 0])
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(block_m, packed_n, local_a, local_b, 8, [16, 16, 128],
                                                                 a_transposed=False, b_transposed=True, slice_m=256,
                                                                 slice_n=128)
    wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
                                local_wmma.instr_shape, cga)
    load_wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases,
                                     tuple(local_wmma.reg_bases) + ((0, 8), ), local_wmma.instr_shape, cga)
    load_wmma_packed = gl.amd.AMDWMMALayout(3, load_wmma.transposed, load_wmma.warp_bases, load_wmma.reg_bases,
                                            [16, 16, 64], cga)
    dot_a = gl.DotOperandLayout(0, load_wmma, 16)
    dot_b = gl.DotOperandLayout(1, load_wmma_packed, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple((tuple([basis[1], basis[0]]) for basis in dot_b.cga_layout))
    padded_a = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [block_m, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for([[128, 16]], [block_n, 128], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(256, 256, padded_a, padded_b, 8, [16, 16, 128],
                                                                      a_transposed=False, b_transposed=True,
                                                                      slice_m=256, slice_n=256)
    shared_bs = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [block_n // 128, 1024], [1, 0], cga_b)
    grid = (triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)
    group_size = 8 if args.K <= 8192 else 4

    def launch(a_d=a_d, b_d=b_d, c_d=c_d, bs_d=bs_d):
        return fp8_mxfp4_gemm_warp_pipeline[grid](
            a_d, b_d, c_d, bs_d, args.M, args.N, args.K, a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0), GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma, LOAD_WMMA_LAYOUT=load_wmma,
            OUTPUT_CGA_LAYOUT=output_cga, BLOCK_M=block_m, BLOCK_N=block_n, CTA_M=4, CTA_N=4, GROUP_ALONG_N=True,
            GROUP_SIZE=group_size, num_warps=8, waves_per_eu=2, num_ctas=16)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(c_d.cpu(), reference, rtol=0.01, atol=0.5)
        print('result verified', flush=True)

    return (launch, check, (a_d, b_d, c_d, bs_d))


MXFP8_TILES = {'256x256x256': (256, 256), '128x128x512': (128, 512), '64x64x512': (64, 512)}


def mxfp8_config(M, N, K, tile, cluster):
    """Validate the user-selected CTA tile, cluster and matrix dimensions."""
    if tile is None or cluster is None:
        raise ValueError('MXFP8 requires explicit --mxfp8-tile and --mxfp8-cluster')
    if tile not in MXFP8_TILES:
        raise ValueError(f'unsupported MXFP8 tile: {tile}; choose from {", ".join(MXFP8_TILES)}')
    if cluster not in ('2x2', '4x4'):
        raise ValueError(f'unsupported MXFP8 cluster: {cluster}; choose 2x2 or 4x4')
    cta_tile, block_k = MXFP8_TILES[tile]
    width = int(cluster[0])
    block = cta_tile * width
    minimum = 1024 if block_k == 256 else 1536
    if M <= 0 or N <= 0 or M % block or N % block:
        raise ValueError(f'MXFP8 {tile} / {width}x{width} requires positive M/N divisible by {block}')
    if K < minimum or K % 512:
        raise ValueError(f'MXFP8 {tile} requires K >= {minimum} and divisible by 512')
    return cta_tile, block_k, width


def make_mxfp8_case(args):
    tile, block_k, cluster = validate_gemm_shape('mxfp8', args.M, args.N, args.K, args.mxfp8_tile, args.mxfp8_cluster)
    flat_pipeline = getattr(args, 'flat_pipeline', False)
    bk256_kernel = mxfp8_gemm_warp_pipeline_flat if flat_pipeline else mxfp8_gemm_warp_pipeline
    bk512_kernel = mxfp8_gemm_warp_pipeline_bk512_flat if flat_pipeline else mxfp8_gemm_warp_pipeline_bk512
    if args.mxfp8_output_buffers == 2 and block_k != 256:
        raise ValueError('--mxfp8-output-buffers 2 requires the 256x256x256 tile')
    block_m = block_n = tile * cluster
    print(f'CTA tile={tile}x{tile}x{block_k}, cluster={cluster}x{cluster}', flush=True)
    torch.manual_seed(args.seed)
    if args.input_mode == 'positive_mxfp8':
        # Positive E4M3 values with independent K128 scales for A and N128xK128 scales for B.
        a = (torch.rand((args.M, args.K), device='cuda', dtype=torch.float32) / 10).to(torch.float8_e4m3fn)
        b = (torch.rand((args.N, args.K), device='cuda', dtype=torch.float32) / 10).to(torch.float8_e4m3fn).T
        scale_values = [
            torch.rand((args.M, args.K // 128), device='cuda', dtype=torch.float32),
            torch.rand((args.N // 128, args.K // 128), device='cuda', dtype=torch.float32)
        ]
        scale_codes = []
        for values in scale_values:
            # Match FP8 E8M0 RoundUp conversion, including the normalization arithmetic.
            bits = ((values * 448.0) / 448.0).view(torch.int32)
            exponent = (bits >> 23) & 255
            scale_codes.append((exponent + (((bits & 0x7fffff) != 0) & (exponent < 255))).to(torch.uint8))
        a_scale_obj = MXScaleTensor(size=(args.M, args.K // 32))
        b_scale_obj = MXScaleTensor(size=(args.N, args.K // 32))
        a_scale_obj.data = scale_codes[0].repeat_interleave(4, dim=1)
        b_scale_obj.data = scale_codes[1].repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
    elif args.input_mode == 'trig':
        a_bits, a_scale_bits = _make_mxfp8_trig_matrix(args.M, args.K)
        b_bits, b_scale_bits = _make_mxfp8_trig_matrix(args.N, args.K)
        a_scale_obj = MXScaleTensor(size=a_scale_bits.shape)
        b_scale_obj = MXScaleTensor(size=b_scale_bits.shape)
        a_scale_obj.data = a_scale_bits
        b_scale_obj.data = b_scale_bits
        a = a_bits.view(torch.float8_e4m3fn)
        b = b_bits.view(torch.float8_e4m3fn).T
    else:
        a = init_data('float8_e4m3', args.M, args.K)
        b = init_data('float8_e4m3', args.K, args.N)
        scale_k = args.K // 32
        a_scale_obj = MXScaleTensor(size=(args.M, scale_k)).random(low=1.0, high=32.0)
        b_scale_obj = MXScaleTensor(size=(args.N, scale_k)).random(low=1.0, high=32.0)
    reference = None
    if args.check and args.input_mode != 'positive_mxfp8':
        reference = torch_gemm_mxfp(a, b, a_scale_obj, b_scale_obj, 32, args.M, args.N, args.K).to(torch.bfloat16)
    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    as_d = (pack_scale(a_scale_obj.data) if block_k == 256 else a_scale_obj.data.contiguous()).cuda()
    bs_d = (pack_scale(b_scale_obj.data) if block_k == 256 else b_scale_obj.data.contiguous()).cuda()
    c_d = torch.empty((args.M, args.N), dtype=torch.bfloat16, device='cuda')
    # First derive the per-CTA WMMA ownership, then attach the cluster CGA bases.
    cga = make_cga_layout([cluster, cluster], [cluster, cluster], [0, 1])
    output_cga = tuple(tuple(basis) + (0, 0) for basis in cga)
    slice_n = tile // 2 if block_k == 256 else tile
    local_a = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_m, block_k], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_n, block_k], [1, 0])
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(block_m, block_n, local_a, local_b, 8, [16, 16, 128],
                                                                 a_transposed=False, b_transposed=True, slice_m=tile,
                                                                 slice_n=slice_n)
    load_wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
                                     local_wmma.instr_shape, cga)
    dot_a = gl.DotOperandLayout(0, load_wmma, 16)
    dot_b = gl.DotOperandLayout(1, load_wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple((basis[1], basis[0]) for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_m, block_k], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_n, block_k], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(tile, tile, padded_a, padded_b, 8, [16, 16, 128],
                                                                      a_transposed=False, b_transposed=True,
                                                                      slice_m=tile, slice_n=slice_n)
    if block_k == 256:
        # BK256 splits N into two accumulators and uses packed block-32 scales.
        packed_n = block_n // 2
        local_b = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [packed_n, 256], [1, 0])
        _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(block_m, packed_n, local_a, local_b, 8,
                                                                     [16, 16, 128], a_transposed=False,
                                                                     b_transposed=True, slice_m=tile, slice_n=slice_n)
        wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
                                    local_wmma.instr_shape, cga)
        shared_as = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [block_m // 128, 1024], [1, 0], cga_a)
        shared_bs = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [block_n // 128, 1024], [1, 0], cga_b)
        # One parent B allocation contains both N halves for every CTA column.
        inner = shared_b.partition_layout
        parent_cga = tuple((basis[0], 0, basis[1]) for basis in inner.cga_layout)
        padded = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [cluster, 64, 256], [2, 1, 0], parent_cga)
        b2_shared = PartitionedSharedLayout(2, 2, 1, padded)
        b2_load = gl.DistributedLinearLayout(
            reg_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [32, 0, 0],
                       [64, 0, 0], [0, 0, 16], [0, 0, 32]], lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8],
                                                                        [16, 0, 0]], warp_bases=[[0, 0, 64], [0, 0, 0],
                                                                                                 [0, 0, 0]],
            block_bases=[[0, 0, 0]] * (cluster.bit_length() - 1) + [[0, 1 << i, 0]
                                                                    for i in range(cluster.bit_length() - 1)],
            shape=[128, cluster, 128])
    else:
        # BK512 uses a full accumulator and unshuffled scales for CTA64/128.
        wmma = load_wmma
        shared_as = gl.PaddedSharedLayout.with_identity_for([[16, 8]], [block_m, 16], [1, 0], cga_a)
        shared_bs = gl.PaddedSharedLayout.with_identity_for([[16, 8]], [block_n, 16], [1, 0], cga_b)
    grid = (triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)

    def launch(a_d=a_d, b_d=b_d, c_d=c_d, as_d=as_d, bs_d=bs_d):
        if block_k == 512:
            return bk512_kernel[grid](a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K, a_d.stride(0), a_d.stride(1),
                                      b_d.stride(1), b_d.stride(0), c_d.stride(0), c_d.stride(1), as_d.stride(0),
                                      bs_d.stride(0), GRID_MN=grid[0], SHARED_A=shared_a, SHARED_B=shared_b,
                                      SHARED_AS=shared_as, SHARED_BS=shared_bs, WMMA=wmma, BLOCK_M=block_m,
                                      BLOCK_N=block_n, CLUSTER=cluster, num_warps=8, num_ctas=cluster * cluster,
                                      waves_per_eu=2)
        return bk256_kernel[grid](a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K, a_d.stride(0), a_d.stride(1),
                                  b_d.stride(1), b_d.stride(0), c_d.stride(0), c_d.stride(1), as_d.stride(0),
                                  bs_d.stride(0), GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B2=b2_shared,
                                  SHARED_SCALE_A=shared_as, SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
                                  LOAD_WMMA_LAYOUT=load_wmma, B2_LOAD_LAYOUT=b2_load, OUTPUT_CGA_LAYOUT=output_cga,
                                  BLOCK_M=block_m, BLOCK_N=block_n, CTA_M=cluster, CTA_N=cluster, num_warps=8,
                                  waves_per_eu=2, num_ctas=cluster * cluster,
                                  TWO_BUFFER_OUTPUT=args.mxfp8_output_buffers == 2)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        if args.input_mode == 'positive_mxfp8':
            # Check every output in row chunks without materializing a full MxK FP32 A.
            b_f32 = b_d.float() * torch.exp2(b_scale_obj.data.float() - 127).repeat_interleave(32, dim=1)
            for begin in range(0, args.M, 256):
                a_f32 = a_d[begin:begin + 256].float() * torch.exp2(a_scale_obj.data[begin:begin + 256].float() -
                                                                    127).repeat_interleave(32, dim=1)
                expected = a_f32 @ b_f32.T
                torch.testing.assert_close(c_d[begin:begin + 256].float(), expected, rtol=0.008, atol=1e-5)
        else:
            torch.testing.assert_close(c_d.cpu(), reference, rtol=0.01, atol=0.5)
        print('result verified', flush=True)

    return (launch, check, (a_d, b_d, c_d, as_d, bs_d))


def make_mxfp4_case(args):
    validate_gemm_shape('mxfp4', args.M, args.N, args.K)
    torch.manual_seed(args.seed)
    if args.input_mode == 'trig':
        a_bits, as_bits = _make_mxfp4_trig_matrix(args.M, args.K)
        b_bits, bs_bits = _make_mxfp4_trig_matrix(args.N, args.K)
        a = MXFP4Tensor(size=(args.M, args.K))
        b = MXFP4Tensor(size=(args.K, args.N))
        a.data = a_bits
        b.data = b_bits.T
        a_scale_obj = MXScaleTensor(size=as_bits.shape)
        b_scale_obj = MXScaleTensor(size=bs_bits.shape)
        a_scale_obj.data = as_bits
        b_scale_obj.data = bs_bits
    else:
        a = init_data('float4', args.M, args.K)
        b = init_data('float4', args.K, args.N)
        scale_k = args.K // 32
        a_scale_obj = MXScaleTensor(size=(args.M, scale_k)).random(low=1.0, high=32.0)
        b_scale_obj = MXScaleTensor(size=(args.N, scale_k)).random(low=1.0, high=32.0)
    reference = None
    if args.check:
        reference = torch_gemm_mxfp(a, b, a_scale_obj, b_scale_obj, 32, args.M, args.N, args.K).to(torch.bfloat16)
    as_d = pack_scale(a_scale_obj.data).cuda()
    bs_d = pack_scale(b_scale_obj.data).cuda()
    a_d = a.to_packed_tensor(dim=1).data.contiguous().cuda()
    b_d = b.to_packed_tensor(dim=0).data.T.contiguous().cuda()
    c_d = torch.empty((args.M, args.N), dtype=torch.bfloat16, device='cuda')
    cga = make_cga_layout([4, 4], [4, 4], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [1024, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [1024, 256], [1, 0])
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(1024, 1024, local_a, local_b, 8, [32, 16, 128],
                                                                 a_transposed=False, b_transposed=True, slice_m=256,
                                                                 slice_n=256)
    wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
                                local_wmma.instr_shape, cga)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple((tuple([basis[1], basis[0]]) for basis in dot_b.cga_layout))
    padded_a = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [1024, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [1024, 256], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(256, 256, padded_a, padded_b, 8, [32, 16, 128],
                                                                      a_transposed=False, b_transposed=True,
                                                                      slice_m=256, slice_n=256)
    shared_as = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [8, 2048], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [8, 2048], [1, 0], cga_b)
    grid = (triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch(a_d=a_d, b_d=b_d, c_d=c_d, as_d=as_d, bs_d=bs_d):
        common_kwargs = dict(GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b,
                             SHARED_SCALE_A=shared_as, SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma, num_warps=8,
                             waves_per_eu=2, num_ctas=16)
        return mxfp4_gemm_warp_pipeline[grid](a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K, a_d.stride(0),
                                              a_d.stride(1), b_d.stride(1), b_d.stride(0), c_d.stride(0), c_d.stride(1),
                                              bs_d.stride(0), **common_kwargs)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(c_d.cpu(), reference, rtol=0.01, atol=0.5)
        print('result verified', flush=True)

    return (launch, check, (a_d, b_d, c_d, as_d, bs_d))


CASE_BUILDERS = {
    'bf16': make_bf16_case,
    'mxfp8': make_mxfp8_case,
    'mxfp4': make_mxfp4_case,
    'fp8_mxfp4': make_fp8_mxfp4_case,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dtype', choices=['all', *CASE_BUILDERS], default='all')
    parser.add_argument('-M', type=int, default=4096)
    parser.add_argument('-N', type=int, default=4096)
    parser.add_argument('-K', type=int, default=65536)
    parser.add_argument(
        '--input', '--input-mode', dest='input_mode', choices=['auto', 'random', 'trig',
                                                               'positive_mxfp8'], default='auto',
        help='auto uses trig except for FP8 x MXFP4; positive_mxfp8 is MXFP8-only positive FP8 data and scales')
    parser.add_argument('--mxfp8-tile', choices=list(MXFP8_TILES),
                        help='MXFP8 CTA MxNxK tile; required when dtype is mxfp8 or all')
    parser.add_argument('--mxfp8-cluster', choices=['2x2', '4x4'],
                        help='MXFP8 CGA cluster; required when dtype is mxfp8 or all')
    parser.add_argument('--mxfp8-output-buffers', type=int, choices=[1, 2], default=1,
                        help='MXFP8 output LDS buffers: default 1; 2 overlaps BK256 N-half stores')
    parser.add_argument('--flat-pipeline', action='store_true',
                        help='fully expand the MXFP8 K traversal, including tails; specializes the kernel for K')
    parser.add_argument('--seed', type=int, default=42,
                        help='random input seed (default 42); ignored by BF16/MXFP8/MXFP4 trig inputs')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--rotate-inputs', action='store_true',
                        help='benchmark distinct copies of A, B, scales and output (disabled by default)')
    parser.add_argument('--repeats', type=int,
                        help='total timed launches with --rotate-inputs: default 100, range 1..200; excludes warmup')
    parser.add_argument('--samples', type=int, help='timing samples: default 4, or 1 with --rotate-inputs')
    parser.add_argument('--replays', type=int, help='graph replays per sample: default 20, or 1 with --rotate-inputs')
    parser.add_argument('--iters-per-graph', type=int,
                        help='launches per graph: default 50; with --rotate-inputs must equal --repeats')
    args = parser.parse_args()
    if args.input_mode == 'positive_mxfp8' and args.dtype != 'mxfp8':
        parser.error('--input positive_mxfp8 requires --dtype mxfp8')
    if args.mxfp8_output_buffers == 2 and (args.dtype not in ('mxfp8', 'all') or args.mxfp8_tile != '256x256x256'):
        parser.error('--mxfp8-output-buffers 2 requires MXFP8 with --mxfp8-tile 256x256x256')
    if args.flat_pipeline and args.dtype != 'mxfp8':
        parser.error('--flat-pipeline is unsupported yet for non-MXFP8 GEMMs; use --dtype mxfp8')
    if args.rotate_inputs:
        args.benchmark = True
        args.repeats = 100 if args.repeats is None else args.repeats
        if not 1 <= args.repeats <= 200:
            parser.error('--rotate-inputs requires --repeats between 1 and 200')
        if args.samples not in (None, 1) or args.replays not in (None, 1):
            parser.error('--rotate-inputs requires --samples 1 and --replays 1 to bound total timed launches')
        if args.iters_per_graph not in (None, args.repeats):
            parser.error('--rotate-inputs requires --iters-per-graph to equal --repeats')
        args.samples, args.replays, args.iters_per_graph = (1, 1, args.repeats)
    else:
        if args.repeats is not None:
            parser.error('--repeats requires --rotate-inputs')
        args.samples = 4 if args.samples is None else args.samples
        args.replays = 20 if args.replays is None else args.replays
        args.iters_per_graph = 50 if args.iters_per_graph is None else args.iters_per_graph
    if not args.check and (not args.benchmark):
        args.check = True
    if args.warmup < 0 or min(args.samples, args.replays, args.iters_per_graph) < 1:
        parser.error('warmup must be nonnegative and timing counts must be positive')
    dtypes = list(CASE_BUILDERS) if args.dtype == 'all' else [args.dtype]
    try:
        for dtype in dtypes:
            validate_gemm_shape(dtype, args.M, args.N, args.K, args.mxfp8_tile, args.mxfp8_cluster)
    except ValueError as error:
        parser.error(str(error))
    if not torch.cuda.is_available():
        parser.error('this example requires an AMD CDNA5 GPU')
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'hip' or target.arch != 'gfx1250':
        parser.error(f'this example requires CDNA5 (gfx1250), got {target}')
    mode = args.input_mode
    for dtype in dtypes:
        args.input_mode = ('random' if dtype == 'fp8_mxfp4' else 'trig') if mode == 'auto' else mode
        if args.input_mode == 'trig' and dtype in ('mxfp8', 'mxfp4'):
            seed_info = f'seed={TRIG_SEED} (fixed trig seed)'
        elif args.input_mode == 'trig' and dtype == 'bf16':
            seed_info = 'seed=none (deterministic trig)'
        elif args.input_mode == 'trig':
            seed_info = f'scale seed={args.seed} (deterministic trig values)'
        else:
            seed_info = f'seed={args.seed}'
        print(f'\n{dtype}: M={args.M} N={args.N} K={args.K}, {args.input_mode}, {seed_info}', flush=True)
        launch, check, inputs = CASE_BUILDERS[dtype](args)
        if args.check:
            check()
        kernel = launch()
        print(f'VGPR={kernel.n_regs}, LDS={kernel.metadata.shared}, scratch={kernel.metadata.global_scratch_size}')
        if args.benchmark:
            for _ in range(args.warmup):
                launch()
            torch.cuda.synchronize()
            pool = [inputs]
            if args.rotate_inputs:
                bytes_per_set = sum((t.numel() * t.element_size() for t in inputs))
                print(f'rotating copies: {args.repeats}, bytes per copy: {bytes_per_set}', flush=True)
                pool.extend((tuple((t.clone() for t in inputs)) for _ in range(args.repeats - 1)))
                torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            with torch.cuda.graph(graph, stream=stream):
                for i in range(args.iters_per_graph):
                    launch(*(pool[i] if args.rotate_inputs else inputs))
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.replays):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / (args.replays * args.iters_per_graph))
            us = statistics.median(samples)
            tflops = 2 * args.M * args.N * args.K / (us * 1000000.0)
            print('samples (us): ' + ', '.join((f'{v:.3f}' for v in samples)))
            print(f'median: {us:.3f} us, {tflops:.3f} TFLOPS')
            del graph, stream, pool
        del launch, check, inputs, kernel
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
