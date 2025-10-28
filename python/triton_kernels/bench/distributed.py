import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Optional

import triton_kernels
import triton_kernels.swiglu
from triton_kernels.reduce import reduce
from triton_kernels.matmul_ogs import RoutingData, GatherIndx, ScatterIndx
from triton_kernels.topk import topk
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.target_info import get_cdna_version, is_hip, is_cuda, cuda_capability_geq
from triton_kernels.tensor_details import layout
from triton_kernels.tensor import make_ragged_tensor_metadata, remap_ragged_tensor_metadata
from triton_kernels.distributed import make_expt_dict_uniform, make_expt_assignment, convert_dp_to_ep, convert_ep_to_dp, ExptAssignment, symm_mem_pool

from bench_utils import quantize_weight


@dataclass
class ReduceScatterMetadata:
    mode: str
    active_indx: Optional[torch.Tensor] = None
    dispatch_indx: Optional[torch.Tensor] = None
    combine_indx: Optional[torch.Tensor] = None


def _is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def create_expt_assignment(EP: int, n_expts_tot: int, device: torch.device) -> Optional[ExptAssignment]:
    if not _is_distributed_launch():
        return None
    expt_dict = make_expt_dict_uniform(EP, n_expts_tot)
    return make_expt_assignment(EP, n_expts_tot, expt_dict, device)


def initialize_matmul_ogs(
    batch: int,
    dim1: int,
    dim2: int,
    n_expts_act: int,
    n_expts_tot: int,
    dtype: torch.dtype,
) -> None:
    if not _is_distributed_launch():
        return
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    symm_mem_pool.initialize_matmul_ogs(
        n_tokens_global=batch,
        d_input=dim1,
        d_model=dim2,
        n_expts_act=n_expts_act,
        n_expts_tot=n_expts_tot,
        n_ranks=world_size,
        dtype=dtype,
        group=dist.group.WORLD,
        device=device,
    )


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


def reduce_scatter(
    input_tensor: torch.Tensor,
    n_expts_act: int,
    metadata: ReduceScatterMetadata,
    expt_assignment: Optional[ExptAssignment] = None,
    dim: int = 0,
    op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
) -> torch.Tensor:
    if _is_distributed_launch():
        if metadata.mode and metadata.mode == "ep_sharding":
            if dim != 0 or op != dist.ReduceOp.SUM:
                raise NotImplementedError("Only dim=0 and op=SUM are supported for MoE reduce_scatter.")
            output = convert_ep_to_dp(input_tensor, expt_assignment, metadata.active_indx, metadata.combine_indx)
            # weighted average of the output token from experts
            output = output.view(-1, n_expts_act, output.shape[-1])
            output, _ = reduce(output, dim=1)
            return output
        else:
            raise NotImplementedError(f"Distributed reduce_scatter mode {metadata.mode} is not implemented yet.")
    else:
        return input_tensor


# TODO: support TP > 1
# TODO: clean up duplicate code with triton_kernels.test_distributed.py
# TODO: Support nonuniform expert assignment
def routing(
    x,
    logits,
    n_expts_act,
    sm_first: bool = False,
    y_indx: Optional[torch.Tensor] = None,
    EP: int = 1,
    TP: int = 1,
    expt_assignment: Optional[ExptAssignment] = None,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, RoutingData, GatherIndx, ScatterIndx, Optional[ReduceScatterMetadata]]:
    n_expts_tot = logits.shape[-1]
    if _is_distributed_launch() and mode:
        if mode == "ep_sharding":
            if not expt_assignment:
                raise ValueError("expt_assignment must be provided for distributed routing.")
            if TP > 1:
                raise NotImplementedError("TP > 1 is not supported in distributed MoE benchmark yet.")
            rank = dist.get_rank()
            expt_map = expt_assignment.expt_map[rank, :]
            logits_global = topk(
                logits,
                n_expts_act,
                apply_softmax=not sm_first,
                y_indx=y_indx,
                all_gather=True,
            )
            active_indx = logits_global.indx
            expt_sizes = logits_global.mask_metadata.col_sum
            dispatch_indx = logits_global.mask_metadata.col_sorted_indx
            combine_indx = logits_global.mask_metadata.row_sorted_indx
            logits_global_metadata = make_ragged_tensor_metadata(expt_sizes, dispatch_indx.shape[0])
            x = convert_dp_to_ep(x, expt_assignment, active_indx, dispatch_indx)
            logits_local_metadata = remap_ragged_tensor_metadata(logits_global_metadata, expt_map)
            gate_scal = logits_global.vals.flatten()[combine_indx]
            rdata = RoutingData(gate_scal, expt_sizes, n_expts_tot // EP, n_expts_act, logits_local_metadata)
            reduce_scatter_metadata = ReduceScatterMetadata(
                mode=mode,
                active_indx=active_indx,
                dispatch_indx=dispatch_indx,
                combine_indx=combine_indx,
            )
            return x, rdata, None, None, reduce_scatter_metadata
        else:
            raise NotImplementedError(f"Distributed routing mode {mode} is not implemented yet.")
    else:
        # If mode is not specified or we have a single process, we do single-GPU routing.
        logits = topk(logits, n_expts_act, y_indx=y_indx, apply_softmax=not sm_first)
        dispatch_indx = logits.mask_metadata.col_sorted_indx
        combine_indx = logits.mask_metadata.row_sorted_indx
        ragged_batch_metadata = make_ragged_tensor_metadata(logits.mask_metadata.col_sum, dispatch_indx.shape[0])
        gate_scal = logits.vals.flatten()[combine_indx]
        routing_data = RoutingData(gate_scal, ragged_batch_metadata.slice_sizes, n_expts_tot, n_expts_act,
                                   ragged_batch_metadata)
        gather_indx = GatherIndx(combine_indx, dispatch_indx)
        scatter_indx = ScatterIndx(dispatch_indx, combine_indx)
        return x, routing_data, gather_indx, scatter_indx, None


def gather_ep(rank, world_size, param, TP, EP):
    gathered = None
    group = dist.new_group(list(range(0, world_size, TP)))
    dist.barrier()
    if rank % TP == 0:
        gathered = []
        if rank == 0:
            gathered = [torch.zeros_like(param) for _ in range(EP)]
        dist.gather(param, gathered, dst=0, group=group)
        if rank == 0:
            gathered = torch.cat(gathered, dim=0)
    return gathered


def gather_full(rank, world_size, param, TP, EP, concat_dim_inside, concat_dim_outside):
    gathered = []
    if rank == 0:
        gathered = [torch.zeros_like(param) for _ in range(world_size)]
    dist.gather(param, gathered, dst=0)
    if rank == 0:
        rows = [torch.cat(gathered[e * TP:(e + 1) * TP], dim=concat_dim_inside) for e in range(EP)]
        return torch.cat(rows, dim=concat_dim_outside)
    return None


# We compare the distributed and single-GPU versions of the model to verify correctness.
def distributed_run(rank, world_size, batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP):
    # init
    dev = f"cuda:{rank}"
    # Though we specify backend="nccl", it will be mapped to "rccl" for HIP.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(dev))
    torch.cuda.set_device(rank)

    # weights & biases
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    dist.broadcast(wg, src=0)

    bg = torch.randn((n_expts_tot, ), device=dev)
    dist.broadcast(bg, src=0)

    b2 = torch.randn((n_expts_tot // EP, dim1), device=dev)
    ep_rank = rank // TP
    groups = [dist.new_group(list(range(ep * TP, (ep + 1) * TP))) for ep in range(EP)]
    dist.barrier()
    group = groups[ep_rank]
    dist.broadcast(b2, src=ep_rank * TP, group=group)

    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    b1 = torch.randn((n_expts_tot // EP, dim2 // TP), device=dev)

    w1_full = gather_full(rank, world_size, w1, TP, EP, concat_dim_inside=2, concat_dim_outside=0)
    w2_full = gather_full(rank, world_size, w2, TP, EP, concat_dim_inside=1, concat_dim_outside=0)
    b1_full = gather_full(rank, world_size, b1, TP, EP, concat_dim_inside=1, concat_dim_outside=0)
    b2_full = gather_ep(rank, world_size, b2, TP, EP)

    # quantization
    opt1 = dict()
    opt2 = dict()
    if w_dtype == "mx4" and not is_hip():
        num_warps = 4 if batch <= 512 else 8
        value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
        scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
            mx_axis=1, num_warps=num_warps)
        opt1 = {
            "value_layout": value_layout,
            "value_layout_opts": value_layout_opts,
            "scale_layout": scale_layout,
            "scale_layout_opts": scale_layout_opts,
        }
        opt2 = deepcopy(opt1)
    wg, wg_flex, wg_scale = quantize_weight(wg, "bf16")
    w1, w1_flex, w1_scale = quantize_weight(w1, w_dtype, **opt1)
    w2, w2_flex, w2_scale = quantize_weight(w2, w_dtype, **opt2)
    if rank == 0:
        w1_full, w1_flex_full, w1_scale_full = quantize_weight(w1_full, w_dtype, **opt1)
        w2_full, w2_flex_full, w2_scale_full = quantize_weight(w2_full, w_dtype, **opt2)
    else:
        w1_full = w2_full = w1_flex_full = w2_flex_full = w1_scale_full = w2_scale_full = None

    # precision configs
    pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=wg_flex), weight_scale=wg_scale)
    act = FusedActivation(FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")), (1.0, 1.0), 2)
    pc1 = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w1_flex), weight_scale=w1_scale)
    pc2 = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex), weight_scale=w2_scale)
    if rank == 0:
        pc1_full = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w1_flex_full), weight_scale=w1_scale_full)
        pc2_full = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex_full), weight_scale=w2_scale_full)
    else:
        pc1_full = pc2_full = None

    # inputs
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fnuz if get_cdna_version() == 3 else torch.float8_e4m3fn,
    }
    xd = torch.randn((batch // world_size, dim1), device=dev).to(dtype_map[x_dtype])
    x0 = all_gather(xd, dim=0)
    expt_assignment = create_expt_assignment(EP, n_expts_tot, torch.device(dev))
    symm_mem_pool.initialize_matmul_ogs(
        n_tokens_global=batch,
        d_input=dim1,
        d_model=dim2,
        n_expts_act=n_expts_act,
        n_expts_tot=n_expts_tot,
        n_ranks=world_size,
        dtype=x0.dtype,
        group=dist.group.WORLD,
        device=torch.cuda.current_device(),
    )

    # single-GPU pass
    def single(x):
        xg = x.to(wg.dtype if n_expts_tot > 1 else x.dtype)
        if n_expts_tot > 1:
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            x, rdata, gi, si, _ = routing(x, logits, n_expts_act)
        else:
            rdata = gi = si = None
        x = matmul_ogs(x, w1_full, b1_full, rdata, gather_indx=gi, precision_config=pc1_full, fused_activation=act)
        return matmul_ogs(x, w2_full, b2_full, rdata, scatter_indx=si, precision_config=pc2_full)

    # distributed pass
    def distributed(x):
        xg = x.to(wg.dtype if n_expts_tot > 1 else x.dtype)
        if n_expts_tot > 1:  # sparse
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            x, rdata, gi, si, metadata = routing(x, logits, n_expts_act, EP=EP, TP=TP, expt_assignment=expt_assignment,
                                                 mode="ep_sharding")
        else:  # dense
            x = all_gather(x, dim=0)
            rdata = gi = si = metadata = None
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gi, precision_config=pc1, fused_activation=act)
        x = matmul_ogs(x, w2, b2 if rank % TP == 0 else None, rdata, scatter_indx=si, precision_config=pc2)
        x = reduce_scatter(x, n_expts_act, metadata=metadata, expt_assignment=expt_assignment)
        # gather the result from all GPUs, just for verification
        return all_gather(x, dim=0)

    distributed_result = distributed(xd)
    if rank == 0:
        single_result = single(x0)
        torch.testing.assert_close(distributed_result.to(torch.float16), single_result.to(torch.float16), rtol=1e-2,
                                   atol=1.0, equal_nan=True)

    dist.barrier()
    dist.destroy_process_group()


has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4


@pytest.mark.parametrize(
    "batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP",
    # dense cases
    [(1024, 1024, 1024, 1, 1, "bf16", "bf16", 1, 1), (1024, 1024, 1024, 1, 1, "fp8", "fp8", 1, 1)]
    # moe cases - test parallelism
    + [
        (1024, 1024, 1024, 128, 2, "bf16", "bf16", 1, 1),
        (1024, 1024, 1024, 128, 2, "bf16", "bf16", 1, 4),
    ] +
    # moe cases - test precision
    ([
        (1024, 1024, 1024, 128, 2, "fp8", "mx4", 1, 1),
        (1024, 1024, 1024, 128, 2, "fp8", "mx4", 1, 4),
    ] if has_native_mx4 else [
        (1024, 1024, 1024, 128, 2, "bf16", "mx4", 1, 1),
        (1024, 1024, 1024, 128, 2, "bf16", "mx4", 1, 4),
    ]),
)
def test_mlp_mp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, monkeypatch):
    parallelism = TP * EP
    if is_hip():
        pytest.skip("[TODO] HIP support for distributed MoE.")
    if torch.cuda.device_count() < parallelism:
        pytest.skip(f"Test requires at least {parallelism} GPUs.")
    if is_cuda() and not cuda_capability_geq(9, 0):
        pytest.skip("Test requires CUDA compute capability >= 9.0.")
    if TP > 1:
        pytest.skip("[TODO] TP > 1 is not supported yet in distributed mode.")

    monkeypatch.setenv("WORLD_SIZE", f"{parallelism}")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "12355")
    mp.spawn(
        distributed_run,
        args=(parallelism, batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP),
        nprocs=parallelism,
        join=True,
    )
