from typing import Tuple, List, Dict
import re
import torch
import pytest
import numpy as np
import hashlib

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy
from triton.experimental.gluon.language.extra import libdevice

from aiter.test_common import perftest

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
print(f"THREADS_PER_WARP={THREADS_PER_WARP}")

import os
os.environ["TRITON_CACHE_DIR"] = "/home/sijieli2/gluon_cache"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(123)


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, 
                   k: int = 5, 
                   thresholds: List[float] = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]) -> Dict:
    """
    Compare two numpy arrays and compute various difference metrics.
    
    Args:
        arr1: First input array (float32)
        arr2: Second input array (float32)
        k: Number of top differences to return
        thresholds: List of thresholds for difference magnitude analysis
        
    Returns:
        Dictionary containing:
        - top_k_diff: Top k absolute differences with their positions
        - threshold_stats: Count and percentage of differences above each threshold
        - nan_info: Information about NaN values in input arrays
    """
    # Check input shapes
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")
    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)

    result = {
        'top_k_diff': [],
        'threshold_stats': [],
        'nan_info': {}
    }

    # Check for NaN values
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    
    if np.any(nan_mask1):
        result['nan_info']['arr1_nan_count'] = np.sum(nan_mask1)
        result['nan_info']['arr1_nan_positions'] = np.argwhere(nan_mask1)
        print(f"Warning: arr1 contains {result['nan_info']['arr1_nan_count']} NaN values")
    
    if np.any(nan_mask2):
        result['nan_info']['arr2_nan_count'] = np.sum(nan_mask2)
        result['nan_info']['arr2_nan_positions'] = np.argwhere(nan_mask2)
        print(f"Warning: arr2 contains {result['nan_info']['arr2_nan_count']} NaN values")
    
    # Compute absolute differences
    diff = np.abs(arr1 - arr2)
    total_elements = arr1.size

    max_diff_thr = diff / (1.0 + np.abs(arr2))
    max_diff_thr = max_diff_thr.max()
    print(f"diff.abs.max={diff.max()}")
    print(f"max_diff_thr={max_diff_thr}")

    # Find top k differences
    flat_diff = diff.flatten()
    top_k_indices = np.argpartition(flat_diff, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(-flat_diff[top_k_indices])]

    # Convert flat indices to multi-dimensional indices
    orig_indices = np.unravel_index(top_k_indices, diff.shape)
    for i in range(k):
        idx = tuple(dim[i] for dim in orig_indices)
        result['top_k_diff'].append({
            'value': diff[idx],
            'position': idx,
            'arr1_value': arr1[idx],
            'arr2_value': arr2[idx]
        })

    # Compute threshold statistics
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]
        mask = (diff >= lower) & (diff < upper)
        count = np.sum(mask)
        result['threshold_stats'].append({
            'range': f"[{lower:.1e}, {upper:.1e})",
            'count': count,
            'percentage': 100 * count / total_elements
        })
    
    # Handle values above the largest threshold
    mask = diff >= thresholds[-1]
    count = np.sum(mask)
    result['threshold_stats'].append({
        'range': f">={thresholds[-1]:.1e}",
        'count': count,
        'percentage': 100 * count / total_elements
    })

    print("\nTop differences:")
    for item in result['top_k_diff']:
        print(f"Position {item['position']}: arr1 = {arr1[item['position']]:.6f}, arr2 = {arr2[item['position']]:.6f}, Diff = {item['value']:.6f}")

    print("\nThreshold statistics:")
    for stat in result['threshold_stats']:
        print(f"{stat['range']}: {stat['count']} ({stat['percentage']:.2f}%)")

    print("\nNaN info:")
    print(result['nan_info'])

    return result


def get_autotune_config():
    sizes = [
        # {'QUERY_GRP_SZ': 16, 'SEQ_PARTITION_SZ': 256, 'PARTITION_KV_BLK_NUM': 16, 'K_HD_SPLIT_NUM': 16, 'K_SPLIT_HEAD_SZ': 8, 'KV_BLK_SZ': 16, 'HEAD_SZ': 128},
        {'QUERY_GRP_SZ': 8, 'SEQ_PARTITION_SZ': 256, 'PARTITION_KV_BLK_NUM': 16, 'K_HD_SPLIT_NUM': 16, 'K_SPLIT_HEAD_SZ': 8, 'KV_BLK_SZ': 16, 'HEAD_SZ': 128},
    ]
    return [triton.Config(s) for s in sizes]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@gluon.jit
def gemm_qk_v2(
    q_ptr,
    k_ptr,
    qk_ptr,
    kv_len,
    q_stride0,
    q_stride1,
    q_stride2,
    q_stride3,
    k_stride0,
    k_stride1,
    k_stride2,
    k_stride3,
    k_stride4,
    k_stride5,
    k_stride6,
    qk_stride0,
    qk_stride1,
    qk_stride2,
    qk_stride3,
    qk_stride4,
    qk_stride5,
    QUERY_GRP_SZ: gl.constexpr,
    SEQ_PARTITION_SZ: gl.constexpr,
    PARTITION_KV_BLK_NUM: gl.constexpr,
    K_HD_SPLIT_NUM: gl.constexpr,
    K_SPLIT_HEAD_SZ: gl.constexpr,
    KV_BLK_SZ: gl.constexpr,
    HEAD_SZ: gl.constexpr,
    ):
    # - Q: Matrix Q with shape (batch_size, num_kv_heads * QUERY_GRP_SZ, HEAD_SZ).
    """
    Key parameters:
    - Q: Matrix Q with shape (batch_size, num_kv_heads, QUERY_GRP_SZ, HEAD_SZ).
    - K: Matrix K with shape (batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ).
    - QK: Matrix QK with shape (batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ).
    - kv_len = seq_partition_kv_num * PARTITION_KV_BLK_NUM * KV_BLK_SZ
    - K_SPLIT_HEAD_SZ = 8
    """

    seq_len = kv_len
    batch_id = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    seq_part_idx = gl.program_id(2)
    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    # if seq_start_idx >= seq_len:
    #     return

    q_base_offset = batch_id * q_stride0 + kv_head_idx * q_stride1
    k_base_offset = batch_id * k_stride0 + seq_part_idx * k_stride1 + kv_head_idx * k_stride3
    qk_base_offset = batch_id * qk_stride0 + seq_part_idx * qk_stride1 + kv_head_idx * qk_stride3
    # q_ptr = q_ptr + q_base_offset
    # k_ptr = k_ptr + k_base_offset
    # qk_ptr = qk_ptr + qk_base_offset

    # 1 x QUERY_GRP_SZ x HEAD_SZ
    # 1 x 8(mdim) x 128(kdim)
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[1, 4],
        threads_per_warp=[2, 32],
        warps_per_cta   =[4, 1],
        order           =[1, 0],
    )
    # PARTITION_KV_BLK_NUM x K_HD_SPLIT_NUM x KV_BLK_SZ x K_SPLIT_HEAD_SZ
    # 16 x 16 x 16 x 8
    blocked_k: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[1, 1, 1, 1],
        threads_per_warp=[1, 1, 8, 8],
        warps_per_cta   =[1, 2, 2, 1],
        order           =[3, 2, 1, 0],
    )
    blocked_k0: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0,1,0,0), (0,2,0,0), (0,4,0,0), (0,8,0,0), (2,0,0,0), (4,0,0,0), (8,0,0,0)),
        lane_bases=((0,0,0,1), (0,0,0,2), (0,0,0,4), (0,0,1,0), (0,0,2,0), (0,0,4,0)),
        warp_bases=((0,0,8,0), (1,0,0,0)),
        block_bases=(),
        shape=(16, 16, 16, 8)
    )
    blocked_k1: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0,0,0,1), (0,0,0,2), (0,0,0,4), (0,1,0,0), (0,2,0,0), (0,4,0,0), (0,8,0,0)),
        lane_bases=((0,0,1,0), (0,0,2,0), (0,0,4,0), (0,0,8,0), (1,0,0,0), (2,0,0,0)),
        warp_bases=((4,0,0,0), (8,0,0,0)),
        block_bases=(),
        shape=(16, 16, 16, 8) # n0, k0, n1, k1
    )
    blocked_kt0: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0,0,1,0), (0,0,2,0), (0,0,4,0), (0,1,0,0), (0,2,0,0), (0,4,0,0), (0,8,0,0)),
        lane_bases=((0,0,0,1), (0,0,0,2), (0,0,0,4), (0,0,0,8), (1,0,0,0), (2,0,0,0)),
        warp_bases=((4,0,0,0), (8,0,0,0)),
        block_bases=(),
        shape=(16, 16, 8, 16) # [0, 1, 3, 2]
    )
    blocked_kt1: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0,1,0,0), (0,2,0,0), (0,4,0,0), (1,0,0,0), (2,0,0,0), (4,0,0,0), (8,0,0,0)),
        lane_bases=((0,0,0,1), (0,0,0,2), (0,0,0,4), (0,0,0,8), (0,0,1,0), (0,0,2,0)),
        warp_bases=((0,0,4,0), (0,0,8,0)),
        block_bases=(),
        shape=(16, 8, 16, 16) # [1, 3, 0, 2]
    )
    # PARTITION_KV_BLK_NUM x HEAD_SZ x KV_BLK_SZ
    # 16 x 128(kdim) x 16(ndim)
    blocked_kt: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[4, 4, 8],
        threads_per_warp=[1, 32, 2],
        warps_per_cta   =[4, 1, 1],
        order           =[2, 1, 0],
    )

    # transposed: indicates the result tensor is transposed so that each thread holds consecutive elements
    # in the same row instead of column, which is good for chained dot and global write.
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=False, warps_per_cta=[4, 1]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=4
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=4
    )

    shared_q_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 4, (1,0))

    # q_dim_0_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, blocked_q))
    q_dim_1_layout: gl.constexpr = gl.SliceLayout(1, blocked_q)
    q_dim_2_layout: gl.constexpr = gl.SliceLayout(0, blocked_q)

    k_dim_0_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_k)))
    k_dim_1_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_k)))
    k_dim_2_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_k)))
    k_dim_3_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_k)))

    kt_dim_0_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, blocked_kt))
    kt_dim_1_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, blocked_kt))
    kt_dim_2_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, blocked_kt))

    # offs_q_dim0 = gl.arange(0, 1, layout=q_dim_0_layout)
    offs_q_dim1 = gl.arange(0, QUERY_GRP_SZ, layout=q_dim_1_layout)
    offs_q_dim2 = gl.arange(0, HEAD_SZ, layout=q_dim_2_layout)

    offs_k_dim0 = gl.arange(0, PARTITION_KV_BLK_NUM, layout=k_dim_0_layout)
    offs_k_dim1 = gl.arange(0, K_HD_SPLIT_NUM, layout=k_dim_1_layout)
    offs_k_dim2 = gl.arange(0, KV_BLK_SZ, layout=k_dim_2_layout)
    offs_k_dim3 = gl.arange(0, K_SPLIT_HEAD_SZ, layout=k_dim_3_layout)

    offs_kt_dim0 = gl.arange(0, PARTITION_KV_BLK_NUM, layout=kt_dim_0_layout)
    offs_kt_dim1 = gl.arange(0, HEAD_SZ, layout=kt_dim_1_layout)
    offs_kt_dim2 = gl.arange(0, KV_BLK_SZ, layout=kt_dim_2_layout)

    # offs_q = offs_q_dim0[:, None, None] * 0 + offs_q_dim1[None, :, None] * q_stride2 + offs_q_dim2[None, None, :] * q_stride3
    offs_q = q_base_offset + offs_q_dim1[:, None] * q_stride2 + offs_q_dim2[None, :] * q_stride3
    offs_k = k_base_offset + offs_k_dim0[:, None, None, None] * k_stride2 + offs_k_dim1[None, :, None, None] * k_stride4 + offs_k_dim2[None, None, :, None] * k_stride5 + offs_k_dim3[None, None, None, :] * k_stride6
    kt_stride0 = HEAD_SZ * KV_BLK_SZ
    kt_stride1 = KV_BLK_SZ
    kt_stride2 = 1
    offs_kt = offs_kt_dim0[:, None, None] * kt_stride0 + offs_kt_dim1[None, :, None] * kt_stride1 + offs_kt_dim2[None, None, :] * kt_stride2

    q = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=offs_q)
    k = gl.amd.cdna3.buffer_load(ptr=k_ptr, offsets=offs_k)
    # q_broadcasted = tl.broadcast_to(q, PARTITION_KV_BLK_NUM, QUERY_GRP_SZ, HEAD_SZ)
    # (PARTITION_KV_BLK_NUM, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ) --> (PARTITION_KV_BLK_NUM, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ, KV_BLK_SZ)
    k = gl.convert_layout(k, blocked_k1)
    kt_temp = tl.permute(k, [1, 3, 0, 2])
    # kt_temp = gl.permute(k, [1, 3, 0, 2], layout=blocked_kt1)
    kt = gl.reshape(kt_temp, [HEAD_SZ, PARTITION_KV_BLK_NUM * KV_BLK_SZ])
    # kt = gl.amd.cdna3.buffer_load(ptr=kt_temp, offsets=offs_kt)

    accumulator = gl.zeros((QUERY_GRP_SZ, PARTITION_KV_BLK_NUM*KV_BLK_SZ), dtype=gl.float32, layout=mfma_layout)
    shared_q = gl.allocate_shared_memory(q.dtype, q.shape, shared_q_layout, q)
    q1 = shared_q.load(dot_a_layout)
    # q1 = gl.convert_layout(q, layout=dot_a_layout)
    k1 = gl.convert_layout(kt, layout=dot_b_layout)
    accumulator = gl.amd.cdna3.mfma(q1, k1, accumulator)

    qk = accumulator.to(q_ptr.dtype.element_ty)
    offs_qk_dim1 = gl.arange(0, QUERY_GRP_SZ, layout=gl.SliceLayout(1, mfma_layout))
    offs_qk_dim2 = gl.arange(0, PARTITION_KV_BLK_NUM * KV_BLK_SZ, layout=gl.SliceLayout(0, mfma_layout))
    offs_qk = qk_base_offset + offs_qk_dim1[:, None] * qk_stride3 + offs_qk_dim2[None, :] * qk_stride5
    # qk_mask = seq_start_idx + offs_qk_dim0[:, None, None] * KV_BLK_SZ < seq_len
    # gl.amd.cdna3.buffer_store(stored_value=qk, ptr=qk_ptr, offsets=offs_qk, mask=qk_mask)
    gl.amd.cdna3.buffer_store(stored_value=qk, ptr=qk_ptr, offsets=offs_qk)


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_qk_triton(
    q_ptr,
    k_ptr,
    qk_ptr,
    kv_len,
    q_stride0,
    q_stride1,
    q_stride2,
    q_stride3,
    k_stride0,
    k_stride1,
    k_stride2,
    k_stride3,
    k_stride4,
    k_stride5,
    k_stride6,
    qk_stride0,
    qk_stride1,
    qk_stride2,
    qk_stride3,
    qk_stride4,
    compute_type: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    PARTITION_KV_BLK_NUM: tl.constexpr,
    K_HD_SPLIT_NUM: tl.constexpr,
    K_SPLIT_HEAD_SZ: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    ):
    # - Q: Matrix Q with shape (batch_size, num_kv_heads * QUERY_GRP_SZ, HEAD_SZ).
    """
    Key parameters:
    - Q: Matrix Q with shape (batch_size, num_kv_heads, QUERY_GRP_SZ, HEAD_SZ).
    - K: Matrix K with shape (batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ).
    - QK: Matrix QK with shape (batch_size, seq_partition_kv_num, num_kv_heads, QUERY_GRP_SZ, PARTITION_KV_BLK_NUM * KV_BLK_SZ).
    - kv_len = seq_partition_kv_num * PARTITION_KV_BLK_NUM * KV_BLK_SZ
    - K_SPLIT_HEAD_SZ = 8
    """

    seq_len = kv_len
    batch_id = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_part_idx = tl.program_id(2)
    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    # if seq_start_idx >= seq_len:
    #     return

    q_base_offset = batch_id * q_stride0 + kv_head_idx * q_stride1
    k_base_offset = batch_id * k_stride0 + seq_part_idx * k_stride1 + kv_head_idx * k_stride3
    qk_base_offset = batch_id * qk_stride0 + seq_part_idx * qk_stride1 + kv_head_idx * qk_stride2
    q_ptr = q_ptr + q_base_offset
    k_ptr = k_ptr + k_base_offset
    qk_ptr = qk_ptr + qk_base_offset
    # seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    # max_num_kv_blks: tl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    # num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    offs_q_dim0 = tl.arange(0, QUERY_GRP_SZ)
    offs_q_dim1 = tl.arange(0, HEAD_SZ)
    offs_q = offs_q_dim0[:, None] * q_stride2 + offs_q_dim1[None, :] * q_stride3

    offs_k_dim0 = tl.arange(0, PARTITION_KV_BLK_NUM)
    offs_k_dim1 = tl.arange(0, K_HD_SPLIT_NUM)
    offs_k_dim2 = tl.arange(0, KV_BLK_SZ)
    offs_k_dim3 = tl.arange(0, K_SPLIT_HEAD_SZ)
    offs_k = offs_k_dim0[:, None, None, None] * k_stride2 + offs_k_dim1[None, :, None, None] * k_stride4 + offs_k_dim2[None, None, :, None] * k_stride5 + offs_k_dim3[None, None, None, :] * k_stride6

    # QUERY_GRP_SZ x HEAD_SZ
    # 8(mdim) x 128(kdim)
    q = tl.load(q_ptr + offs_q)

    # PARTITION_KV_BLK_NUM x K_HD_SPLIT_NUM x KV_BLK_SZ x K_SPLIT_HEAD_SZ
    # 16 x 16 x 16 x 8
    k = tl.load(k_ptr + offs_k)
    # (PARTITION_KV_BLK_NUM, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ, KV_BLK_SZ) --> (K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ, PARTITION_KV_BLK_NUM, KV_BLK_SZ)
    kt_temp = tl.permute(k, [1, 3, 0, 2])
    kt = tl.reshape(kt_temp, [K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ, PARTITION_KV_BLK_NUM * KV_BLK_SZ])

    qk = tl.dot(q, kt, out_dtype=tl.float32)
    # qk = tl.dot(q, kt)
    qk = qk.to(compute_type)
    # qk = qk.to(tl.bfloat16)

    offs_qk_dim0 = tl.arange(0, QUERY_GRP_SZ)
    offs_qk_dim1 = tl.arange(0, PARTITION_KV_BLK_NUM * KV_BLK_SZ)
    offs_qk = offs_qk_dim0[:, None] * qk_stride3 + offs_qk_dim1[None, :] * qk_stride4
    tl.store(qk_ptr + offs_qk, qk)


@perftest()
def run_gemm_qk(q, k, dtype):
    batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ = q.shape
    _, seq_partition_kv_num, PARTITION_KV_BLK_NUM, _, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ = k.shape
    kv_len = seq_partition_kv_num * PARTITION_KV_BLK_NUM * KV_BLK_SZ
    HEAD_SZ = K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ
    # qk = torch.randn((batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ), device="cuda", dtype=dtype)
    qk = torch.empty((batch_size, seq_partition_kv_num, num_kv_heads, QUERY_GRP_SZ, PARTITION_KV_BLK_NUM, KV_BLK_SZ), device="cuda", dtype=dtype)
    q = q.reshape(batch_size, num_kv_heads, QUERY_GRP_SZ, HEAD_SZ)
    # print(f"q.stride()={q.stride()}")
    # print(f"k.stride()={k.stride()}")

    grid = (batch_size, num_kv_heads, seq_partition_kv_num)
    gemm_qk_v2[grid](
        q,
        k,
        qk,
        kv_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k.stride(4),
        k.stride(5),
        k.stride(6),
        qk.stride(0),
        qk.stride(1),
        qk.stride(2),
        qk.stride(3),
        qk.stride(4),
        qk.stride(5),
    )
    return qk.permute(0, 1, 4, 2, 3, 5)

@perftest()
def run_gemm_qk_triton(q, k, dtype):
    batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ = q.shape
    _, seq_partition_kv_num, PARTITION_KV_BLK_NUM, _, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ = k.shape
    kv_len = seq_partition_kv_num * PARTITION_KV_BLK_NUM * KV_BLK_SZ
    HEAD_SZ = K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ
    torch_to_tl_dtype = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}

    # qk = torch.randn((batch_size, seq_partition_kv_num, num_kv_heads, QUERY_GRP_SZ, PARTITION_KV_BLK_NUM * KV_BLK_SZ), device="cuda", dtype=dtype)
    qk = torch.empty((batch_size, seq_partition_kv_num, num_kv_heads, QUERY_GRP_SZ, PARTITION_KV_BLK_NUM * KV_BLK_SZ), device="cuda", dtype=dtype)
    q = q.reshape(batch_size, num_kv_heads, QUERY_GRP_SZ, HEAD_SZ)
    # print(f"q.stride()={q.stride()}")
    # print(f"k.stride()={k.stride()}")
    # print(torch_to_tl_dtype[dtype])

    grid = (batch_size, num_kv_heads, seq_partition_kv_num)
    gemm_qk_triton[grid](
        q,
        k,
        qk,
        kv_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k.stride(4),
        k.stride(5),
        k.stride(6),
        qk.stride(0),
        qk.stride(1),
        qk.stride(2),
        qk.stride(3),
        qk.stride(4),
        torch_to_tl_dtype[dtype],
    )
    qk = qk.reshape(batch_size, seq_partition_kv_num, num_kv_heads, QUERY_GRP_SZ, PARTITION_KV_BLK_NUM, KV_BLK_SZ)
    # (batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ)
    qk = qk.transpose(-3, -2).transpose(2, 3)
    return qk


def qk_gemm_ref(q, k):
    batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ = q.shape
    _, seq_partition_kv_num, PARTITION_KV_BLK_NUM, _, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ = k.shape

    # Transpose and reshape and K to align with Q's dimensions for matrix multiplication
    k_transposed = k.transpose(-2, -1)  # Swap last two dimensions
    # k_transposed = k_transposed.contiguous()
    k_reshaped = k_transposed.reshape(batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ, KV_BLK_SZ)
    # Reshape Q for matrix multiplication
    q_reshaped = q.reshape(batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ)
    q_reshaped = q_reshaped.contiguous()
    # Perform matrix multiplication using einsum
    # Reduction happens over the last dimension of Q and first of k_transposed (K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ)
    qk = torch.einsum('bhgd,bsnhdt->bsnhgt', q_reshaped, k_reshaped)
    # qk = qk.contiguous()
    # # Reshape to final output shape
    # qk = qk.reshape(batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ)
    return qk


def test_gemm_qk():
    # - Q: Matrix Q with shape (batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ).
    # - K: Matrix K with shape (batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ).
    # - QK: Matrix QK with shape (batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ).
    # shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
    # tmp_output = torch.empty(
    #     *shape_info, head_sz, dtype=output.dtype, device=output.device
    # )
    # - kv_len = seq_partition_kv_num * PARTITION_KV_BLK_NUM * KV_BLK_SZ
    # - K_SPLIT_HEAD_SZ = 8
    q_shape_list = [
        [32, 1, 8, 16, 8],
        # [80, 1, 16, 16, 8],
        # [80, 3, 16, 16, 8],
    ]
    k_shape_list = [
        [32, 16, 16, 1, 16, 16, 8],
        # [80, 16, 16, 1, 16, 16, 8],
        # [80, 16, 16, 3, 16, 16, 8],
    ]
    # q_shape_list = [
    #     [1, 1, 16, 16, 8],
    # ]
    # k_shape_list = [
    #     [1, 1, 16, 1, 16, 16, 8],
    # ]
    batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ = q_shape_list[0]
    batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ = k_shape_list[0]
    q_len = 1
    kv_len = seq_partition_kv_num * PARTITION_KV_BLK_NUM * KV_BLK_SZ
    head_dim_qk = K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ
    # dtype = torch.float16
    dtype = torch.bfloat16

    q = torch.randn((batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ), device="cuda", dtype=dtype)

    qk_gn, avg_t_us_gn = run_gemm_qk(q, k, dtype)
    qk_tn, avg_t_us_tn = run_gemm_qk_triton(q, k, dtype)

    q = q.to(torch.float32)
    k = k.to(torch.float32)
    qk_ref = qk_gemm_ref(q, k)
    # q = q.reshape(1, QUERY_GRP_SZ, head_dim_qk).contiguous()
    # q_broadcasted = q.repeat(PARTITION_KV_BLK_NUM, 1, 1)
    # k_transposed = k.transpose(-2, -1).contiguous()
    # k_transposed = k_transposed.reshape(PARTITION_KV_BLK_NUM, head_dim_qk, KV_BLK_SZ)
    # k_transposed = k_transposed.contiguous()
    # k_transposed = k_transposed.reshape(PARTITION_KV_BLK_NUM, head_dim_qk, KV_BLK_SZ)
    # qk_ref = torch.bmm(q_broadcasted, k_transposed)
    # qk_ref = qk_ref.reshape(batch_size, seq_partition_kv_num, PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ)

    print(f"kv_len={kv_len}")
    print(f"head_dim_qk={head_dim_qk}")
    print(f"q.shape={q.shape}, q.dtype={q.dtype}")
    print(f"k.shape={k.shape}, k.dtype={k.dtype}")
    print(f"qk_gn.shape={qk_gn.shape}, qk_gn.dtype={qk_gn.dtype}")
    print(f"qk_tn.shape={qk_gn.shape}, qk_tn.dtype={qk_gn.dtype}")
    print(f"qk_ref.shape={qk_gn.shape}, qk_ref.dtype={qk_gn.dtype}")
    print(f"q.stride()={q.stride()}")
    print(f"k.stride()={k.stride()}")
    print(f"qk_gn.stride()={qk_gn.stride()}")
    print(f"qk_tn.stride()={qk_tn.stride()}")
    print(f"qk_ref.stride()={qk_ref.stride()}")

    FLOPS = 2 * batch_size * num_kv_heads * QUERY_GRP_SZ * q_len * kv_len * head_dim_qk / (1e6 * avg_t_us_gn)
    rw_bytes = batch_size * num_kv_heads * (QUERY_GRP_SZ * q_len * head_dim_qk + kv_len * head_dim_qk) * (torch.finfo(dtype).bits // 8)
    TFLOPS_gn = FLOPS / (1e6 * avg_t_us_gn)
    TFLOPS_tn = FLOPS / (1e6 * avg_t_us_tn)
    band_width_gn = rw_bytes / (1.024 ** 4 * 1e3 * avg_t_us_gn)
    band_width_tn = rw_bytes / (1.024 ** 4 * 1e3 * avg_t_us_tn)

    qk_gn_md5 = hashlib.md5(qk_gn.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    qk_tn_md5 = hashlib.md5(qk_tn.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    qk_ref_md5 = hashlib.md5(qk_ref.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()

    compare_arrays(qk_gn.to(torch.float32).detach().cpu().numpy(), qk_ref.to(torch.float32).detach().cpu().numpy())
    compare_arrays(qk_tn.to(torch.float32).detach().cpu().numpy(), qk_ref.to(torch.float32).detach().cpu().numpy())

    print(f"qk_gn_md5={qk_gn_md5}")
    print(f"qk_tn_md5={qk_tn_md5}")
    print(f"qk_ref_md5={qk_ref_md5}")
    print(f"avg_t_us_gn={avg_t_us_gn:.3f} us, TFLOPS_gn={TFLOPS_gn:.1f} TFLOPS, band_width_gn={band_width_gn:.1f} GB/s")
    print(f"avg_t_us_tn={avg_t_us_tn:.3f} us, TFLOPS_tn={TFLOPS_tn:.1f} TFLOPS, band_width_tn={band_width_tn:.1f} GB/s")

    qk_ref = qk_ref.to(dtype)
    # torch.testing.assert_close(qk_gn, qk_ref, rtol=1e-3, atol=1e-3)
    # torch.testing.assert_close(qk_gn, qk_ref, rtol=1e-2, atol=1e-3)
    # torch.testing.assert_close(qk_gn, qk_ref, rtol=1.6e-2, atol=1e-5)
    # torch.testing.assert_close(qk_tn, qk_ref, rtol=1.6e-2, atol=1e-5)
    torch.testing.assert_close(qk_gn, qk_ref)
    torch.testing.assert_close(qk_tn, qk_ref)

    print("\033[92mPASSED\033[0m")


test_gemm_qk()