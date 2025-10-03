import contextlib
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
from triton_kernels.distributed import convert_dp_to_ep, convert_ep_to_dp
from triton_kernels.matmul_ogs import matmul_ogs, reduce_grouped
from triton_kernels.routing import routing, RoutingData
from .test_routing import make_expt_dict_random, make_expt_assignment, filter_expt_data
import pytest

# ------------------------------------------------------------
# fixture
# ------------------------------------------------------------

def _get_free_tcp_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

def _distributed_worker(rank, fn, master_port, world_size, args, kwargs):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ["PYTEST_DISTRIBUTED_WORKER"] = "1"

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    try:
        fn(*args, **kwargs)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def distributed_test(n_gpus):
    n_gpus_list = n_gpus if isinstance(n_gpus, (list, tuple)) else [n_gpus]

    def decorator(fn):
        def wrapped(*args, **kwargs):
            for n_gpus in n_gpus_list:
                if not torch.cuda.is_available():
                    pytest.skip("CUDA required for distributed GPU test")
                if torch.cuda.device_count() < n_gpus:
                    pytest.skip(f"requires {n_gpus} CUDA devices, found {torch.cuda.device_count()}")

                master_port = _get_free_tcp_port()
                mp.spawn(
                    _distributed_worker,
                    args=(fn, master_port, n_gpus, args, kwargs),
                    nprocs=n_gpus,
                    join=True,
                )

        return wrapped

    return decorator


# ------------------------------------------------------------


def mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, n_expts_act):
    rdata, combine_indx, dispatch_indx, _ = routing(l_global, n_expts_act)
    y_global = matmul_ogs(x_global, w_global, b_global, rdata, gather_indx=combine_indx, scatter_indx=dispatch_indx)
    return y_global


def mixture_of_expt_epsharded(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act):
    rank = dist.get_rank()
    rdata_global, combine_indx, dispatch_indx, expt_indx = routing(l_dp_local, n_expts_act, all_gather=True)
    y_ep_local = convert_dp_to_ep(x_dp_local, expt_assignment, expt_indx, dispatch_indx.src_indx)
    expt_data_local = filter_expt_data(rdata_global.expt_data, expt_assignment, rank)
    rdata_ep_local = RoutingData(
        gate_scal=rdata_global.gate_scal,
        expt_hist=expt_data_local.hist,
        n_expts_tot=expt_assignment.n_expts_per_shard[rank],
        n_expts_act=rdata_global.n_expts_act,
        expt_data=expt_data_local,
    )
    y_ep_local = matmul_ogs(y_ep_local, w_ep_local, b_ep_local, rdata_ep_local)
    y_dp_local = convert_ep_to_dp(y_ep_local, expt_assignment, expt_indx, combine_indx.src_indx)
    z_dp_local = reduce_grouped(y_dp_local, contig_group_size=n_expts_act)[0]
    return z_dp_local


@distributed_test(n_gpus=[1, 2])
@pytest.mark.parametrize("n_tokens", [16])
@pytest.mark.parametrize("d_model, n_expts_tot, n_expts_act", [(16, 4, 4)])
def test_expert_sharding(n_tokens, d_model, n_expts_tot, n_expts_act):
    torch.manual_seed(0)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dev = torch.cuda.current_device()
    n_shards = world_size

    expt_dict = make_expt_dict_random(n_shards, n_expts_tot)
    expt_assignment = make_expt_assignment(n_shards, n_expts_tot, expt_dict, device=dev)

    n_tokens_global = n_tokens
    x_global = torch.randn(n_tokens_global, d_model, device=dev, dtype=torch.bfloat16)
    l_global = torch.rand(n_tokens_global, n_expts_tot, device=dev, dtype=torch.float32)
    w_global = torch.randn((n_expts_tot, d_model, d_model), device=dev, dtype=torch.bfloat16)
    b_global = torch.randn((n_expts_tot, d_model), device=dev, dtype=torch.float32)

    n_tokens_local = n_tokens_global // n_shards
    first_token_indx, last_token_indx = rank * n_tokens_local, (rank + 1) * n_tokens_local
    w_ep_local = w_global[expt_assignment.expt_boolmask[rank, :], :, :]
    b_ep_local = b_global[expt_assignment.expt_boolmask[rank, :], :]
    x_dp_local = x_global[first_token_indx:last_token_indx, :]
    l_dp_local = l_global[first_token_indx:last_token_indx, :]

    y_global_ref = mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, n_expts_act)
    y_dp_local_tri = mixture_of_expt_epsharded(
        x_dp_local,
        l_dp_local,
        w_ep_local,
        b_ep_local,
        expt_assignment,
        n_expts_act,
    )
    y_global_tri = torch.empty_like(y_global_ref)
    dist.all_gather_into_tensor(y_global_tri, y_dp_local_tri)
    triton.testing.assert_close(y_global_ref, y_global_tri)
