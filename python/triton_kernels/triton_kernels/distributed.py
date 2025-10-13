# fmt: off

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
import random
from dataclasses import dataclass

@dataclass
class ExptAssignment:
    # torch.Tensor[n_expt_shard, n_expt_tot // 32]
    # (expt_bitmask[i, j//32] >> j%32) & 1 == 1 iff expert j is owned by shard i
    expt_bitmask: torch.Tensor
    # torch.Tensor[n_expt_shard, n_expt_tot]
    # expt_boolmask[i, j] == True iff expert j is owned by shard i
    expt_boolmask: torch.Tensor
    # torch.Tensor[n_expt_shard, n_expt_tot]
    # expt_map[i, j] is the local expert id of expert j in shard i,
    # or -1 if expert j is not owned by shard i
    expt_map: torch.Tensor
    # number of experts per shard
    n_expts_per_shard: list[int]


def make_expt_dict_uniform(n_expt_shard, n_expt_tot):
    """
    create expert assignment dictionary where shard i owns:
    [i*(n_expt_tot//n_expt_shard)...(i+1)*(n_expt_tot//n_expt_shard))
    """
    expt_dict = dict()
    for i in range(n_expt_shard):
        start = (n_expt_tot // n_expt_shard) * i
        end = (n_expt_tot // n_expt_shard) * (i + 1)
        expt_dict[i] = list(range(start, end))
    return expt_dict


def make_expt_dict_random(n_expt_shard, n_expt_tot):
    """
    create expert assignment dictionary where each shard owns
    a disjoint random subset of experts
    """
    expt_dict = dict()
    # random permutation of experts
    rng = random.Random(0)
    perm = list(range(n_expt_tot))
    rng.shuffle(perm)
    # random (distinct) cut points; ensures no empty shard
    cuts = [0] + sorted(rng.sample(range(1, n_expt_tot), n_expt_shard - 1)) + [n_expt_tot]
    for i in range(n_expt_shard):
        a, b = cuts[i], cuts[i + 1]
        expt_dict[i] = perm[a:b]
    return expt_dict


def make_expt_assignment(n_expt_shard, n_expt_tot, expt_dict: dict[int, list[int]], device) -> ExptAssignment:
    """
    n_expt_shard: int
    n_expt_tot: int
    expt_dict: dict[int, list[int]]
      expt_dict[i] is the list of expert ids owned by shard i
    """
    # make expt_bitmask
    words = (n_expt_tot + 31) // 32  # safe even if n_expt_tot not multiple of 32
    expt_bitmask = torch.zeros((n_expt_shard, words), dtype=torch.int32)
    expt_boolmask = torch.zeros((n_expt_shard, n_expt_tot), dtype=torch.bool)
    counts = {expt_id: 0 for expt_id in range(n_expt_tot)}
    for shard, experts in expt_dict.items():
        if len(experts) == 0:
            raise ValueError(f"shard {shard} has no experts")
        if shard < 0 or shard >= n_expt_shard:
            raise ValueError(f"shard {shard} out of range [0, {n_expt_shard})")
        if not isinstance(experts, (list, tuple)):
            raise TypeError(f"expt_dict[{shard}] must be a list/tuple of ints")
        for e in experts:
            counts[e] += 1
            if not (0 <= e < n_expt_tot):
                raise ValueError(f"expert id {e} out of range [0, {n_expt_tot})")
            word = e >> 5  # e // 32
            bit = e & 31  # e % 32
            expt_bitmask[shard, word] |= (1 << bit)
            expt_boolmask[shard, e] = True
    if not all(counts[e] == 1 for e in range(n_expt_tot)):
        raise ValueError("each expert must be owned by exactly one shard")
    expt_bitmask = expt_bitmask.to(device)
    expt_boolmask = expt_boolmask.to(device)
    # make expt_map
    expt_map = torch.full((n_expt_shard, n_expt_tot), -1, dtype=torch.int32)
    for shard, experts in expt_dict.items():
        for local_id, global_id in enumerate(sorted(experts)):
            expt_map[shard, global_id] = local_id
    expt_map = expt_map.to(device)
    # number of experts per shard
    n_expts_per_shard = [len(experts) for experts in expt_dict.values()]
    return ExptAssignment(expt_bitmask, expt_boolmask, expt_map, n_expts_per_shard)


# ------------------------------------------------------------

@triton.jit
def _convert_dp_to_ep(
    peer_dst_ptrs, dst_stride_m, # dst tensors
    src_ptr, src_stride_m, src_shape_n,  # src tensor
    expt_filter_ptr, expt_filter_stride_m, # expt map
    expt_indx_ptr, expt_indx_stride_m, # expt indx
    dst_row_indx_ptr, dst_row_indx_stride_m, # gate indx
    src_row_start,
    N_EXPT_ACT: tl.constexpr,
    N_RANKS: tl.constexpr,
    BLOCK: tl.constexpr
):
    # token offset
    pid_m = tl.program_id(0)
    off_m_global = pid_m + src_row_start
    off_m_local = pid_m
    # offset ptrs
    # load expt and dst indx
    offs_e = tl.arange(0, N_EXPT_ACT)
    expt_indx = tl.load(expt_indx_ptr + off_m_global*expt_indx_stride_m + offs_e)
    # load expt filter
    offs_r = tl.arange(0, N_RANKS)
    expt_filter_ptr_rows = expt_filter_ptr + offs_r[:, None] * expt_filter_stride_m
    expt_filter = (tl.load(expt_filter_ptr_rows + (expt_indx // 32)[None, :]) >> (expt_indx % 32)) & 1
    expt_rank = tl.sum(offs_r[:, None] * expt_filter, axis=0)
    # load dst indxs
    dst_row_indx = tl.load(dst_row_indx_ptr + off_m_global*dst_row_indx_stride_m + offs_e)
    # set src and d
    offs_n = tl.arange(0, BLOCK)
    dst_row_ptrs = tl.load(peer_dst_ptrs + expt_rank).to(src_ptr.dtype, bitcast=True)
    dst_row_ptrs = tl.multiple_of(dst_row_ptrs, 16)
    dst_row_ptrs = dst_row_ptrs + dst_row_indx * dst_stride_m
    dst_ptrs = dst_row_ptrs[:, None] + offs_n[None, :]
    src_ptrs = src_ptr + off_m_local * src_stride_m + offs_n
    for start_n in range(0, src_shape_n, BLOCK):
        mask_n = start_n + offs_n < src_shape_n
        src = tl.load(src_ptrs, mask=mask_n, other=0.0)
        tl.store(dst_ptrs, src[None, :], mask=mask_n[None, :])
        src_ptrs += BLOCK
        dst_ptrs += BLOCK


@triton.jit
def _create_tensor_from_tuples(Dst, Srcs: tl.tuple):
    for i in tl.static_range(len(Srcs)):
        tl.store(Dst + i, Srcs[i].to(tl.int64, bitcast=True))


def create_tensor_from_tuples(dst_shape, src_tuples, dtype, device):
    dst = torch.empty(dst_shape, dtype=dtype, device=device)
    _create_tensor_from_tuples[(1, )](dst, src_tuples)
    return dst


def convert_dp_to_ep(src, expt_assignment, expt_indx, gate_indx):
    expt_bitmask = expt_assignment.expt_bitmask
    # extract problem dimensions
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    device = src.device
    n_tokens_local, d_model = src.shape
    n_tokens_global, n_expt_act = expt_indx.shape
    # validate invariants
    assert n_ranks == expt_bitmask.size(0)
    assert all(t.device == device for t in [expt_bitmask, expt_indx, gate_indx]), "all tensors must be on the same device"
    assert expt_bitmask.dtype == torch.int32, "expt_bitmask must be int32 bitmask words"
    assert expt_bitmask.stride(-1) == 1 and expt_indx.stride(-1) == 1 and gate_indx.stride(-1) == 1
    assert n_tokens_local * n_ranks <= n_tokens_global
    assert gate_indx.shape == (n_tokens_global*n_expt_act, ), f"{tuple(gate_indx.shape)} != {(n_tokens_global*n_expt_act,)}"
    # allocate symmetric memory
    dst_local = symm_mem.empty((n_tokens_global*n_expt_act, d_model), dtype=src.dtype, device=device)
    # create tensor of peer pointers
    hdl = symm_mem.rendezvous(dst_local, dist.group.WORLD)
    peer_bufs = [hdl.get_buffer(r, dst_local.shape, dst_local.dtype) for r in range(n_ranks)]
    peer_dst_ptrs = create_tensor_from_tuples((n_ranks, ), tuple([int(buf.data_ptr()) for buf in peer_bufs]), dtype=torch.int64, device=device)
    # launch kernel
    BLOCK = 512
    grid = (n_tokens_local,)
    _convert_dp_to_ep[grid](
        peer_dst_ptrs, dst_local.stride(0),
        src, src.stride(0), src.shape[1],
        expt_bitmask, expt_bitmask.stride(0),
        expt_indx, expt_indx.stride(0),
        gate_indx, n_expt_act,
        rank * n_tokens_local,
        N_EXPT_ACT=n_expt_act,
        N_RANKS=n_ranks,
        BLOCK=BLOCK,
    )
    hdl.barrier(channel=0)
    return dst_local


# ------------------------------------------------------------

@triton.jit
def _convert_ep_to_dp(
    peer_dst_ptrs, dst_stride_m, # dst tensors
    src_ptr, src_stride_m, src_shape_n, # src tensor
    expt_filter_ptr, expt_filter_stride_m, # expt map
    expt_indx_ptr, expt_indx_stride_m, # expt indx
    dst_row_indx_ptr, # topk indx
    n_tokens_local,
    BLOCK: tl.constexpr
):
    # token offset
    pid_m = tl.program_id(0)
    # destination base pointer
    dst_indx_global = tl.load(dst_row_indx_ptr + pid_m)
    dst_rank = dst_indx_global // n_tokens_local
    dst_ptr = tl.load(peer_dst_ptrs + dst_rank)
    dst_ptr = dst_ptr.to(src_ptr.dtype, bitcast=True)
    dst_ptr = tl.multiple_of(dst_ptr, 16)
    # input / output pointers
    dst_expt_indx = tl.load(expt_indx_ptr + dst_indx_global)
    has_dst_expt = (tl.load(expt_filter_ptr + dst_expt_indx // 32) >> (dst_expt_indx % 32)) & 1
    if not has_dst_expt.to(tl.int1):
        return
    dst_indx_local = dst_indx_global - dst_rank * n_tokens_local
    offs_n = tl.arange(0, BLOCK)
    dst_ptrs = dst_ptr + dst_indx_local * dst_stride_m + offs_n
    src_ptrs = src_ptr + pid_m * src_stride_m + offs_n
    for start_n in range(0, src_shape_n, BLOCK):
        mask_n = start_n + offs_n < src_shape_n
        src = tl.load(src_ptrs, mask=mask_n, other=0.0)
        tl.store(dst_ptrs, src, mask=mask_n)
        src_ptrs += BLOCK
        dst_ptrs += BLOCK


def convert_ep_to_dp(src, expt_assignment, expt_indx, topk_indx):
    expt_bitmask = expt_assignment.expt_bitmask
    # extract problem dimensions
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    device = src.device
    n_tokens_global, d_model = src.shape
    n_tokens_local = n_tokens_global // n_ranks
    # allocate symmetric memory
    dst_local = symm_mem.empty((n_tokens_local, d_model), dtype=src.dtype, device=device)
    hdl = symm_mem.rendezvous(dst_local, dist.group.WORLD)
    peer_bufs = [hdl.get_buffer(r, dst_local.shape, dst_local.dtype) for r in range(n_ranks)]
    peer_dst_ptrs = create_tensor_from_tuples((n_ranks, ), tuple([int(buf.data_ptr()) for buf in peer_bufs]), dtype=torch.int64, device=device)
    # launch kernel
    BLOCK = 512
    grid = (n_tokens_global,)
    hdl.barrier(channel=0)
    _convert_ep_to_dp[grid](
        peer_dst_ptrs, dst_local.stride(0),
        src, src.stride(0), src.shape[1],
        expt_bitmask[rank, :], expt_bitmask.stride(0),
        expt_indx, expt_indx.stride(0),
        topk_indx,
        n_tokens_local,
        BLOCK=BLOCK,
    )
    hdl.barrier(channel=0)
    return dst_local
