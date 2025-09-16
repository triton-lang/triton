import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.jit
def _paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk(
    exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,       # [num_seqs]
    scale,
    k_scale,
    v_scale,
    alibi_slopes,
    stride_max_logits_s,
    stride_max_logits_nh,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_nh,
    stride_logits_p,
    stride_logits_g,
    stride_q_s,
    stride_q_nh,
    stride_k_b,
    stride_k_nh,
    stride_k_hz,
    stride_k_bz,
    stride_v_b,
    stride_v_nh,
    stride_v_hz,
    stride_bt_s,
    compute_type: ttgl.constexpr,
    HEAD_SZ: ttgl.constexpr,
    HEAD_SZ_POW2: ttgl.constexpr, # 128
    QUERY_GRP_SZ: ttgl.constexpr,
    QUERY_GRP_SZ_POW2: ttgl.constexpr, # 16
    KV_BLK_SZ: ttgl.constexpr,
    KV_BLK_SZ_POW2: ttgl.constexpr,
    SEQ_PARTITION_SZ: ttgl.constexpr,
):
    blocked_layout0: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=(1, 1, 1, 8),
        threads_per_warp=(1, 4, 16, 1),
        warps_per_cta=(4, 1, 1, 1),
        order=(3, 2, 1, 0)
    )
    blocked_layout1: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=(1, 8),
        threads_per_warp=(4, 16),
        warps_per_cta=(4, 1),
        order=(1, 0)
    )
    blocked_layout2: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=(1,),
        threads_per_warp=(64,),
        warps_per_cta=(4,),
        order=(0,)
    )
    blocked_layout4: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=(1, 8, 1, 1),
        threads_per_warp=(4, 1, 1, 16),
        warps_per_cta=(1, 1, 1, 4),
        order=(1, 3, 0, 2)
    )
    blocked_layout5: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=(1, 8, 1),
        threads_per_warp=(1, 2, 32),
        warps_per_cta=(1, 1, 4),
        order=(1, 2, 0)
    )
    linear_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=((0,1), (0,2), (4,0), (8,0)), # 16
        lane_bases=((0,0), (0,0), (0,0), (0,0), (0,4), (0,8)), # 64
        warp_bases=((1,0), (2,0)), # 4
        block_bases=[],
        shape=[BLOCK_SIZE_K, BLOCK_SIZE_N], # todo
    )
    linear_layout1: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=((1,0), (2,0), (4,0), (32,0), (64,0), (0,64), (0,128)), # 16
        lane_bases=((0,1), (0,2), (0,4), (0,8), (8,0), (16,0)), # 64
        warp_bases=((0,16), (0,32)), # 4
        block_bases=[],
        shape=[BLOCK_SIZE_K, BLOCK_SIZE_N], # todo
    )
    linear_layout2: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=((1,0), (2,0), (4,0), (16,0), (32,0), (64,0), (128,0)), # 16
        lane_bases=((8,0), (0,1), (0,2), (0,4), (0,8), (0,16)), # 64
        warp_bases=((0,32), (0,64)), # 4
        block_bases=[],
        shape=[BLOCK_SIZE_K, BLOCK_SIZE_N], # todo
    )
    linear_layout4: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=((0,1), (0,2), (0,4), (0,32), (0,64)), # 16
        lane_bases=((1,0), (2,0), (4,0), (8,0), (0,8), (0,16)), # 64
        warp_bases=((0,0), (0,0)), # 4
        block_bases=[],
        shape=[BLOCK_SIZE_K, BLOCK_SIZE_N], # todo
    )
    blocked_layout3: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=((0,0,1), (0,0,2), (0,0,4), (0,0,8), (4,0,0), (8,0,0), (0,64,0)), # 16
        lane_bases=((0,1,0), (0,2,0), (0,4,0), (0,8,0), (1,0,0), (2,0,0)), # 64
        warp_bases=((0,16,0), (0,32,0)), # 4
        block_bases=[],
        shape=[BLOCK_SIZE_K, BLOCK_SIZE_N], # todo
    )
    linear_layout5: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=((0,1), (0,2), (0,4), (0,8), (0,64), (0,128)), # 64
        lane_bases=((1,0), (2,0), (4,0), (8,0), (0,8), (0,16)), # 64
        warp_bases=((0,0), (0,0)), # 4
        block_bases=[],
        shape=[BLOCK_SIZE_K, BLOCK_SIZE_N], # todo
    )
    mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=3,
        warps_per_cta=(2, 4), instr_shape=(16, 16), transposed=True)
    shared_layout0: ttgl.constexpr = ttgl.SwizzledSharedLayout(
        vec=4, per_phase=1, max_phase=16, order=(1,0))
    shared_layout1: ttgl.constexpr = ttgl.SwizzledSharedLayout(
        vec=4, per_phase=1, max_phase=16, order=(0,1))
    shared_layout2: ttgl.constexpr = ttgl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=(0,1))



    seq_idx = ttgl.program_id(0)
    kv_head_idx = ttgl.program_id(1)
    seq_part_idx = ttgl.program_id(2)

    log2e: ttgl.constexpr = 1.4426950408889634
    CONTIGUOUS_KV_ELEMS_16B_LOAD: ttgl.constexpr = 8

    seq_len = ttgl.load(seq_lens_ptr + seq_idx)
    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = ttgl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    max_num_kv_blks: ttgl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    num_kv_blks = ttgl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    blk_offs0 = ttgl.arange(0, KV_BLK_SZ_POW2, layout=ttgl.SliceLayout(1, 2, 3, blocked_layout0))
    blk_offs1 = ttgl.arange(0, KV_BLK_SZ_POW2, layout=blocked_layout2)
    head_sz_offs1 = ttgl.arange(0, HEAD_SZ_POW2, layout==ttgl.SliceLayout(0, blocked_layout1)) # 128
    head_sz_div_offs = ttgl.arange(0, HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD)
    q_grp_offs = ttgl.arange(0, QUERY_GRP_SZ_POW2) # 16
    contiguous_kv_elems_offs = ttgl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = ttgl.zeros([QUERY_GRP_SZ_POW2], dtype=ttgl.float32)
    else:
        alibi_slope = ttgl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    # load all kv blocks in one time
    blk_ids = ttgl.arange(0, max_num_kv_blks)
    masked_blk_ids = ttgl.where(blk_ids < num_kv_blks, blk_ids, 0)
    kv_blk_start = seq_part_idx * max_num_kv_blks
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    kv_blk_nums = ttgl.load(blk_tables_start_ptr + kv_blk_start + masked_blk_ids)

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = ( # 16x128
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    q = ttgl.load(q_ptr + q_offs, mask=q_mask, other=0.0) # blocked1
    q = (q * scale).to(compute_type) # -> ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>

    # k_blk_offs[max_num_kv_blks, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_blk_offs = (
        kv_blk_nums[:, None, None, None] * stride_k_b
        + kv_head_idx * stride_k_nh
        + head_sz_div_offs[None, :, None, None] * stride_k_hz
        + blk_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
        + contiguous_kv_elems_offs[None, None, None, :]
    )
    # blk_seq_offs[max_num_kv_blks, KV_BLK_SZ_POW2]
    blk_seq_offs = ((kv_blk_start + blk_ids[:, None]) * KV_BLK_SZ  # blk_ids: [max_num_kv_blks]
                    + blk_offs[None, :]) # blk_offs: [KV_BLK_SZ_POW2]

    k_mask = (
        (blk_seq_offs[:, None, :, None] < seq_len) &
        (blk_offs[None, None, :, None] < KV_BLK_SZ) &
        (head_sz_div_offs[None, :, None, None] < (HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD))
    )

    # k[max_num_kv_blks, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_0 = ttgl.load(k_cache_ptr + k_blk_offs) # blocked
    k = k_0.to(ttgl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
    k = k.to(compute_type)
    # k[HEAD_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2]
    k = ttgl.permute(k, [1, 3, 0, 2]) # [HEAD_SZ_POW2/x, x, max_num_kv_blks, KV_BLK_SZ_POW2]
    k = ttgl.reshape(k, [HEAD_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2]) # -> #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

    # qk[QUERY_GRP_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2]
    qk = ttgl.dot(q, k, out_dtype=ttgl.float32)
    blk_seq_flatten_offs = ttgl.reshape(blk_seq_offs, [max_num_kv_blks * KV_BLK_SZ_POW2])
    if alibi_slopes is not None:
        qk += (alibi_slope[:, None] * (blk_seq_flatten_offs - seq_len + 1)[None, :]).to(
            ttgl.float32
        )
    qk = ttgl.where(
        (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_flatten_offs[None, :] < seq_len),
        qk,
        float("-inf"),
    )

    max_logit_new = ttgl.max(qk, axis=1)
    # p[QUERY_GRP_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2]
    p = ttgl.math.exp2((qk - max_logit_new[:, None]) * log2e)
    p = p.to(compute_type) # -> ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
    exp_sum = ttgl.sum(p, axis=1)

    # v_blk_offs[max_num_kv_blks, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    v_blk_offs = (
        kv_blk_nums[:, None, None] * stride_v_b
        + kv_head_idx * stride_v_nh
        + head_sz_offs[None, :, None] * stride_v_hz
        + blk_offs[None, None, :]
    )
    v_mask = (
        (blk_seq_offs[:, None, :] < seq_len) &
        (blk_offs[None, None, :] < KV_BLK_SZ) &
        (head_sz_offs[None, :, None] < HEAD_SZ)
    )

    # v[max_num_kv_blks, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    v_0 = ttgl.load(v_cache_ptr + v_blk_offs)
    v = v_0.to(ttgl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
    v = v.to(compute_type)
    # v[max_num_kv_blks * KV_BLK_SZ_POW2, HEAD_SZ_POW2]
    v = ttgl.permute(v, [0, 2, 1])
    v = ttgl.reshape(v, [max_num_kv_blks * KV_BLK_SZ_POW2, HEAD_SZ_POW2])

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    ttgl.store(max_logits_ptr + max_logits_offs, max_logit_new, mask=m_grp_mask)
    ttgl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)

    # acc[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    acc = ttgl.dot(p, v, out_dtype=ttgl.float32)
    acc = acc / exp_sum[:, None]

    # end up computation
    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )
    ttgl.store(logits_ptr + logits_offs, acc, mask=q_mask)


def _paged_attn_decode_v2_w_dot_kernel_reshape_wrapper(
    grid,
    exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,       # [num_seqs]
    scale,
    k_scale,
    v_scale,
    alibi_slopes,
    stride_max_logits_s,
    stride_max_logits_nh,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_nh,
    stride_logits_p,
    stride_logits_g,
    stride_q_s,
    stride_q_nh,
    stride_k_b,
    stride_k_nh,
    stride_k_hz,
    stride_k_bz,
    stride_v_b,
    stride_v_hz,
    stride_v_bz,
    stride_bt_s,
    kv_type,
    compute_type,
    HEAD_SZ,
    HEAD_SZ_POW2,
    QUERY_GRP_SZ,
    QUERY_GRP_SZ_POW2,
    KV_BLK_SZ,
    KV_BLK_SZ_POW2,
    SEQ_PARTITION_SZ,
):
    _paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk[grid](
        exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
        max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
        logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
        q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
        k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
        v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
        blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
        seq_lens_ptr,       # [num_seqs]
        scale,
        k_scale,
        v_scale,
        alibi_slopes,
        stride_max_logits_s,
        stride_max_logits_nh,
        stride_max_logits_p,
        stride_logits_s,
        stride_logits_nh,
        stride_logits_p,
        stride_logits_g,
        stride_q_s,
        stride_q_nh,
        stride_k_b,
        stride_k_nh,
        stride_k_hz,
        stride_k_bz,
        stride_v_b,
        stride_v_hz,
        stride_v_bz,
        stride_bt_s,
        compute_type=compute_type,
        HEAD_SZ=HEAD_SZ,
        HEAD_SZ_POW2=HEAD_SZ_POW2,
        QUERY_GRP_SZ=QUERY_GRP_SZ,
        QUERY_GRP_SZ_POW2=QUERY_GRP_SZ_POW2,
        KV_BLK_SZ=KV_BLK_SZ,
        KV_BLK_SZ_POW2=KV_BLK_SZ_POW2,
        SEQ_PARTITION_SZ=SEQ_PARTITION_SZ,
    )

def paged_attn_decode_v2(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq],
    seq_lens: torch.Tensor,  # [num_seqs],
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: float,
    v_scale: float,
    max_num_partitions: int,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
        num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = value_cache.shape[3]
    head_sz = value_cache.shape[2]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)

    # Note: There is a bug in triton.next_power_of_2 function which causes it
    # to update the passed in arg, so that's why we have a workaround here
    # max_num_partitions_pow2 = triton.next_power_of_2(max_num_partitions)
    if max_num_partitions == 0:
        max_num_partitions_pow2 = 1
    else:
        max_num_partitions_pow2 = 2 ** math.ceil(math.log2(max_num_partitions))

    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)