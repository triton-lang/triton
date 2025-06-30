import os
import torch
import torch.distributed as dist
import triton_kernels.routing
from dataclasses import dataclass
from triton_kernels.routing import RoutingData, GatherIndx, ScatterIndx, compute_expt_data
from triton_kernels.topk import topk
from typing import Tuple


@dataclass
class ReduceScatterMetadata:
    input_split_sizes: list[int]
    ep_indx: torch.Tensor
    EP: int = 1
    TP: int = 1


def _is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup() -> Tuple[int, int]:
    if _is_distributed_launch():
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", world_size=world_size, device_id=torch.device(local_rank))
    else:
        world_size = 1
        local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, world_size


def cleanup():
    if _is_distributed_launch():
        dist.barrier()
        dist.destroy_process_group()
    else:
        pass


def broadcast(x: torch.Tensor, src: int = 0, groups: list = None, group_idx: int = None) -> torch.Tensor:
    if _is_distributed_launch():
        if x.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            x = x.to(torch.float16)
        group = None
        if groups:
            groups = [dist.new_group(group) for group in groups]
            dist.barrier()
            group = groups[group_idx]
        dist.broadcast(x, src=src, group=group)
        return x
    else:
        return x


def all_gather(x: torch.Tensor, dim=0) -> torch.Tensor:
    if _is_distributed_launch():
        world_size = dist.get_world_size()
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_list, x)
        return torch.cat(x_list, dim=dim)
    else:
        return x


def reduce_scatter(input_tensor: torch.Tensor, metadata: ReduceScatterMetadata = None, dim: int = 0,
                   op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM) -> torch.Tensor:
    if _is_distributed_launch():

        def dtype_cast(dtype: torch.dtype) -> torch.dtype:
            # check if dtype is fp8, then convert it to float16 before reducing
            if dtype in [torch.float16, torch.bfloat16, torch.float32]:
                return dtype
            else:
                return torch.float16

        world_size = dist.get_world_size()
        if metadata and metadata.input_split_sizes:
            assert dim == 0, "metadata only works with dim=0"
            input_list = list(input_tensor.split(metadata.input_split_sizes, dim=0))
            output_list = all_to_all(input_list, dim=0)
            n_tokens = metadata.ep_indx.size(dim)
            other_dims = input_tensor.shape[1:]
            output_tensor = input_tensor.new_zeros((n_tokens, ) + other_dims, dtype=dtype_cast(input_tensor.dtype))
            for i in range(world_size):
                ep_rank = i // metadata.TP
                mask = torch.any(metadata.ep_indx == ep_rank, dim=1)
                if op == dist.ReduceOp.SUM:
                    output_tensor[mask] += output_list[i].to(dtype_cast(output_list[i].dtype))
                else:
                    raise NotImplementedError(f"Reduce operation {op} is not implemented.")
            return output_tensor
        else:
            input_list = list(input_tensor.chunk(world_size, dim=dim))
            shape = input_list[0].shape
            dtype = dtype_cast(input_tensor.dtype)
            input_list = [x.to(dtype) for x in input_list]
            output_tensor = input_tensor.new_empty(shape, dtype=dtype)
            dist.reduce_scatter(output_tensor, input_list, op=op)
            return output_tensor
    else:
        return input_tensor


def all_to_all(input_list: list[torch.Tensor], dim: int = 0) -> list[torch.Tensor]:
    if _is_distributed_launch():
        # Check if all tensors have only one dimension with different sizes
        for t in input_list:
            for d in range(t.dim()):
                if d != dim and t.size(d) != input_list[0].size(d):
                    raise ValueError("All tensors must have the same size in all dimensions except the specified one.")
        input_sizes = [t.size(dim) for t in input_list]
        input_sizes = torch.tensor(input_sizes, device=input_list[0].device).unsqueeze(0)
        input_sizes = all_gather(input_sizes, dim=0)
        output_split_sizes = input_sizes[:, dist.get_rank()].tolist()
        other_dims = list(input_list[0].shape[:dim] + input_list[0].shape[dim + 1:])
        output_list = [
            torch.empty([size] + other_dims, dtype=input_list[0].dtype, device=input_list[0].device)
            for size in output_split_sizes
        ]
        dist.all_to_all(output_list, input_list)
        return output_list
    else:
        return input_list


def routing(x, logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None, EP=1,
            TP=1) -> Tuple[RoutingData, GatherIndx, ScatterIndx, ReduceScatterMetadata]:
    if _is_distributed_launch():
        assert expt_indx is None, "expt_indx should be None for distributed routing"
        _, n_expts_tot = logits.shape

        # Use the same topk as triton_kernels for consistent tie-breaking behavior
        if sm_first:
            logits = torch.softmax(logits, dim=-1)
        expt_scal, expt_indx, _ = topk(logits, n_expts_act, apply_softmax=sm_first, n_rows=n_rows)
        expt_indx = expt_indx.int()

        # Sort each token's selections by expert
        expt_indx, sort_indices = torch.sort(expt_indx, dim=1, stable=True)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)

        chunk_size = n_expts_tot // EP
        output_split_sizes = None
        ep_indx = None

        if EP > 1:
            # Distributed Expert Parallelism
            ep_rank = dist.get_rank() // TP
            ep_indx = expt_indx // chunk_size

            # Partition tokens by expert parallelism rank
            expt_scal_list = []
            expt_indx_list = []
            x_list = []

            for i in range(EP):
                mask = torch.any(ep_indx == i, dim=1)
                expt_scal_masked = expt_scal[mask]
                expt_indx_masked = expt_indx[mask]
                x_masked = x[mask]

                for _ in range(TP):
                    expt_scal_list.append(expt_scal_masked)
                    expt_indx_list.append(expt_indx_masked)
                    x_list.append(x_masked)

            # Exchange data across processes
            expt_scal_list = all_to_all(expt_scal_list, dim=0)
            expt_indx_list = all_to_all(expt_indx_list, dim=0)
            x_list = all_to_all(x_list, dim=0)

            output_split_sizes = [x.size(0) for x in expt_scal_list]
            expt_scal = torch.cat(expt_scal_list, dim=0)
            expt_indx = torch.cat(expt_indx_list, dim=0)
            x = torch.cat(x_list, dim=0)

            # Filter for local experts only
            mask = (expt_indx // chunk_size) == ep_rank
            expt_indx -= ep_rank * chunk_size
            expt_scal = expt_scal.masked_fill(~mask, 0)
            expt_indx = expt_indx.masked_fill(~mask, n_expts_tot)
        else:
            # Distributed Data Parallelism
            x = all_gather(x, dim=0)
            expt_scal = all_gather(expt_scal, dim=0)
            expt_indx = all_gather(expt_indx, dim=0)

        # Flatten topk data
        expt_scal = expt_scal.reshape(-1)
        expt_indx = expt_indx.reshape(-1).to(torch.int32)

        # Sort by expert_id for contiguous experts in matmul
        expt_indx, topk_indx = torch.sort(expt_indx, stable=True)
        gate_indx = torch.argsort(topk_indx, stable=True)

        mask = expt_indx != n_expts_tot
        topk_indx[~mask] = -1
        gate_indx[gate_indx >= mask.sum()] = -1
        gate_scal = expt_scal[topk_indx]
        hist = torch.histc(expt_indx[mask], bins=chunk_size, min=0, max=chunk_size - 1)

        # Pack the matmul data structures
        gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
        scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
        n_gates = mask.sum().item()
        expt_data = compute_expt_data(hist, chunk_size, n_gates)

        return (
            x,
            RoutingData(gate_scal, hist, chunk_size, n_expts_act, expt_data=expt_data),
            gather_indx,
            scatter_indx,
            ReduceScatterMetadata(input_split_sizes=output_split_sizes, ep_indx=ep_indx, EP=EP, TP=TP),
        )
    else:
        return x, *triton_kernels.routing.routing(logits, n_expts_act, sm_first, expt_indx, EP, n_rows), None
