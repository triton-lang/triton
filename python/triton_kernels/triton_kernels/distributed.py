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


def _convert_launch_metadata(grid, kernel, args):
    src = args["src_ptr"]
    src_rank = args["SRC_RANK"]
    n_tokens_local = args["n_tokens_local"]
    src_row_start = n_tokens_local * src_rank
    expt_filter = args["expt_filter_ptr"]
    expt_indx = args["expt_indx_ptr"].int()
    d_model = src.shape[1]
    elem_bytes = src.element_size()
    src_bytes = src.numel() * elem_bytes
    # Find out number of tokens being dispatched out from this GPU
    local_expt_indx = expt_indx[src_row_start:src_row_start + n_tokens_local]
    src_rank_filter = expt_filter[src_rank]
    local_filter = ((src_rank_filter[local_expt_indx // 32] >> (local_expt_indx % 32)) & 1).to(torch.int32)
    dst_local_tokens = torch.sum(local_filter).item()
    dst_output_tokens = local_filter.numel() - dst_local_tokens
    global_filter = ((src_rank_filter[expt_indx // 32] >> (expt_indx % 32)) & 1).to(torch.int32)
    dst_input_tokens = torch.sum(global_filter).item() - dst_local_tokens
    # Calculate the number of bytes transferred out from this GPU
    dram_bytes = src_bytes + dst_local_tokens * d_model * elem_bytes
    if "dp_to_ep" in kernel.name:
        dram_bytes += dst_input_tokens * d_model * elem_bytes
    elif "ep_to_dp" in kernel.name:
        dram_bytes += dst_output_tokens * d_model * elem_bytes
    else:
        raise ValueError(f"unknown kernel name {kernel.name}")
    nvlink_bytes = (dst_output_tokens + dst_input_tokens) * d_model * elem_bytes
    return {
        "name": f"{kernel.name} [tokens={n_tokens_local}, d_model={d_model}]",
        "bytes": dram_bytes,
        "nvlink_bytes": nvlink_bytes,
    }


@triton.jit(launch_metadata=_convert_launch_metadata)
def _convert_dp_to_ep(
    peer_dst_ptrs, dst_stride_m, # dst tensors
    src_ptr, src_stride_m, src_shape_n,  # src tensor
    expt_filter_ptr, expt_filter_stride_m, # expt map
    expt_indx_ptr, expt_indx_stride_m, # expt indx
    dst_row_indx_ptr, dst_row_indx_stride_m, # gate indx
    n_tokens_local,
    SRC_RANK: tl.constexpr,
    N_EXPT_ACT: tl.constexpr,
    N_RANKS: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid_m = tl.program_id(0)
    off_m_global = pid_m + n_tokens_local * SRC_RANK
    off_m_local = pid_m
    offs_r = tl.arange(0, N_RANKS)
    offs_e = tl.arange(0, N_EXPT_ACT)
    offs_n = tl.arange(0, BLOCK)
    dst_row_indx = tl.load(dst_row_indx_ptr + off_m_global * dst_row_indx_stride_m + offs_e)
    expt_indx = tl.load(expt_indx_ptr + off_m_global * expt_indx_stride_m + offs_e)
    expt_filter_ptr_rows = expt_filter_ptr + offs_r[:, None] * expt_filter_stride_m
    expt_filter = (tl.load(expt_filter_ptr_rows + (expt_indx // 32)[None, :]) >> (expt_indx % 32)) & 1
    expt_ranks = tl.sum(offs_r[:, None] * expt_filter, axis=0)
    dst_row_ptrs = tl.zeros((N_EXPT_ACT,), dtype=tl.int64)
    for dst_rank in tl.static_range(N_RANKS):
        peer_dst_ptr = peer_dst_ptrs[dst_rank].to(tl.int64, bitcast=True)
        dst_row_ptrs = tl.where(dst_rank == expt_ranks, peer_dst_ptr, dst_row_ptrs)
    dst_row_ptrs = dst_row_ptrs.to(src_ptr.dtype, bitcast=True)
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
    # launch kernel
    BLOCK = 512
    grid = (n_tokens_local,)
    _convert_dp_to_ep[grid](
        tuple(peer_bufs), dst_local.stride(0),
        src, src.stride(0), src.shape[1],
        expt_bitmask, expt_bitmask.stride(0),
        expt_indx, expt_indx.stride(0),
        gate_indx, n_expt_act,
        n_tokens_local,
        SRC_RANK=rank,
        N_EXPT_ACT=n_expt_act,
        N_RANKS=n_ranks,
        BLOCK=BLOCK,
    )
    hdl.barrier(channel=0)
    return dst_local


# ------------------------------------------------------------

@triton.jit(launch_metadata=_convert_launch_metadata)
def _convert_ep_to_dp(
    peer_dst_ptrs, dst_stride_m, # dst tensors
    src_ptr, src_stride_m, src_shape_n, # src tensor
    expt_filter_ptr, expt_filter_stride_m, # expt map
    expt_indx_ptr,  # expt indx
    dst_row_indx_ptr, # topk indx
    n_tokens_local,
    BLOCK: tl.constexpr,
    SRC_RANK: tl.constexpr,
    N_RANKS: tl.constexpr
):
    # token offset
    pid_m = tl.program_id(0)
    # destination base pointer
    dst_indx_global = tl.load(dst_row_indx_ptr + pid_m)
    dst_rank = dst_indx_global // n_tokens_local
    dst_ptr = tl.zeros((1,), dtype=tl.int64).item()
    for i in tl.static_range(N_RANKS):
        if dst_rank == i:
            dst_ptr = peer_dst_ptrs[i].to(tl.int64, bitcast=True)
    dst_ptr = tl.multiple_of(dst_ptr.to(src_ptr.dtype), 16)
    # input / output pointers
    dst_expt_indx = tl.load(expt_indx_ptr + dst_indx_global)
    expt_filter_ptr = expt_filter_ptr + SRC_RANK * expt_filter_stride_m
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
    # launch kernel
    BLOCK = 512
    grid = (n_tokens_global,)
    _convert_ep_to_dp[grid](
        tuple(peer_bufs), dst_local.stride(0),
        src, src.stride(0), src.shape[1],
        expt_bitmask, expt_bitmask.stride(0),
        expt_indx,
        topk_indx,
        n_tokens_local,
        BLOCK=BLOCK,
        SRC_RANK=rank,
        N_RANKS=n_ranks,
    )
    hdl.barrier(channel=0)
    return dst_local
