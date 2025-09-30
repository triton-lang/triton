# fmt: off

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from triton_kernels.routing import RoutingData, filter_expt_data


def convert_rdata_to_ep(routing_data, expt_assignment):
    rank = dist.get_rank()
    expt_data = filter_expt_data(routing_data.expt_data, expt_assignment, rank)
    routing_data = RoutingData(
        gate_scal=routing_data.gate_scal,
        expt_hist=expt_data.hist,
        n_expts_tot=expt_assignment.n_expts_per_shard[rank],
        n_expts_act=routing_data.n_expts_act,
        expt_data=expt_data,
    )
    return routing_data



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
    expt_rank = tl.sum(offs_r[:, None] * expt_filter, axis=0) > 0
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
    peer_dst_ptrs = torch.tensor([int(buf.data_ptr()) for buf in peer_bufs], device=device)
    # launch kernel
    BLOCK = 512
    grid = (n_tokens_local,)
    hdl.barrier(channel=0)
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
    torch.cuda.synchronize()
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
    peer_dst_ptrs = torch.tensor([int(buf.data_ptr()) for buf in peer_bufs], device=device)
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
