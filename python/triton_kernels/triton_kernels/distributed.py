import os
import torch
import torch.distributed as dist
import triton_bench.routing
from triton_bench.routing import RoutingData, GatherIndx, ScatterIndx
from triton_bench.topk import topk
from typing import Tuple


def _is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup() -> Tuple[int, int]:
    if _is_distributed_launch():
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        world_size = 1
        local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, world_size


def cleanup():
    if _is_distributed_launch():
        dist.destroy_process_group()
    else:
        pass


def all_gather(x: torch.Tensor, dim=0):
    if _is_distributed_launch():
        world_size = dist.get_world_size()
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_list, x)
        return torch.cat(x_list, dim=dim)
    else:
        return x


def reduce_scatter(x: torch.Tensor, token_mask: torch.Tensor = None, dim=0):
    if _is_distributed_launch():
        world_size = dist.get_world_size()
        if token_mask is not None:
            # build the padded shape
            shape = list(x.shape)
            shape[dim] = token_mask.shape[0]
            # create a zero tensor, scatter x into it where mask is True, then split
            x_new = x.new_zeros(shape)
            # Expand token_mask to match x's shape for assignment
            index = [slice(None)] * x.dim()
            index[dim] = token_mask
            x_new[index] = x
            x_list = list(x_new.chunk(world_size, dim=dim))
        else:
            x_list = list(x.chunk(world_size, dim=dim))
        # build output shape
        shape = x_list[0].shape
        # reduce scatter into the single tensor
        # check if dtype is fp8, then convert it to float16 before reducing
        if x.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            x_list = [x.to(torch.float16) for x in x_list]
            out = x.new_empty(shape, dtype=torch.float16)
        else:
            out = x.new_empty(shape, dtype=x.dtype)
        dist.reduce_scatter(out, x_list)
        return out
    else:
        return x


def routing(logits, n_expts_act, expt_indx=None, EP=1, TP=1):
    if _is_distributed_launch():
        assert expt_indx is None
        _, n_expts_tot = logits.shape
        # We need to use the same topk as triton_bench because torch's topk
        # does not have the same tie-breaking behavior as triton_bench.
        expt_scal, expt_indx, _ = topk(logits, n_expts_act)
        expt_indx = expt_indx.int()
        expt_scal = torch.softmax(expt_scal, dim=-1)
        # Sort each token's selections by expert
        expt_indx, sort_indices = torch.sort(expt_indx, dim=1, stable=True)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)
        # Distributed-DP
        expt_scal = all_gather(expt_scal, dim=0)
        expt_indx = all_gather(expt_indx, dim=0)
        chunk_size = n_expts_tot // EP
        ep_indx = dist.get_rank() // TP
        # Distributed-EP
        if EP > 1:
            # keep only the experts assigned to this rank
            local_expt_mask = (expt_indx // chunk_size) == ep_indx
            # mask out rows with all false, meaning that this token is not assigned to any expert belonging to this rank
            token_mask = torch.any(local_expt_mask, dim=1)
            expt_scal, expt_indx, local_expt_mask = [t[token_mask] for t in (expt_scal, expt_indx, local_expt_mask)]
            # normalize the expert ids
            expt_indx -= ep_indx * chunk_size
            # mask values associated with invalid expert ids
            expt_scal = expt_scal.masked_fill(~local_expt_mask, 0)
            expt_indx = expt_indx.masked_fill(~local_expt_mask, n_expts_tot)
        else:
            token_mask = None
        # flatten topk data
        expt_scal = expt_scal.reshape(-1)
        expt_indx = expt_indx.reshape(-1).to(torch.int32)
        # sort by expert_id so experts are contiguous for the matmul
        # For example:
        # expt_indx: [expt0 => row4, row1, row0, ..., expt1 => row2, row3, ..., ...]
        # topk_indx: [2 (row0), 1 (row1), 3 (row2), 4 (row3), 5 (row4), ...]
        expt_indx, topk_indx = torch.sort(expt_indx, stable=True)
        gate_indx = torch.argsort(topk_indx, stable=True)
        topk_indx[expt_indx == n_expts_tot] = -1
        gate_indx[gate_indx >= (expt_indx != n_expts_tot).sum()] = -1
        gate_scal = expt_scal[topk_indx]
        # histogram of tokens over experts
        hist = torch.histc(expt_indx[expt_indx != n_expts_tot], bins=chunk_size, min=0, max=chunk_size)
        # pack the matmul data structure
        gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
        scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
        return RoutingData(gate_scal, hist, n_expts_tot // EP, n_expts_act), gather_indx, scatter_indx, token_mask
    else:
        return *triton_bench.routing.routing(logits, n_expts_act, expt_indx, EP), None
