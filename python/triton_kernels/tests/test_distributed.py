import contextlib
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
from triton_kernels.distributed import convert_dp_to_ep, convert_ep_to_dp, make_expt_dict_uniform, make_expt_dict_random, make_expt_assignment, SymmetricMemoryPool
from triton_kernels.distributed_details.mesh import Mesh
from triton_kernels.reduce import reduce
from triton_kernels.topk import topk
from triton_kernels.matmul import matmul
from triton_kernels.tensor import make_ragged_tensor_metadata, remap_ragged_tensor_metadata
import pytest


def _make_expt_dict_for_mode(n_shards, n_expts_tot, affinity_mode):
    factories = {
        "uniform": make_expt_dict_uniform,
        "random": make_expt_dict_random,
    }
    try:
        return factories[affinity_mode](n_shards, n_expts_tot)
    except KeyError as exc:
        raise ValueError(f"Unknown affinity mode: {affinity_mode}") from exc


def _make_y_indx_for_mode(n_tokens_global, n_expts_tot, n_expts_act, n_shards, affinity_mode, dev):
    y_indx_global = None
    if affinity_mode == "uniform":
        if n_expts_tot % n_shards != 0:
            raise ValueError("uniform affinity requires experts evenly divisible by shards")
        expts_per_rank = n_expts_tot // n_shards
        rounds = (n_expts_act + n_shards - 1) // n_shards
        if rounds > expts_per_rank:
            raise ValueError("round-robin selection exceeds experts available per shard")
        order = torch.arange(n_expts_act, device=dev, dtype=torch.int32)
        shard_order = order % n_shards
        intra_shard = order // n_shards
        round_robin_indx = (shard_order * expts_per_rank + intra_shard).to(torch.int16)
        y_indx_global = round_robin_indx.unsqueeze(0).expand(n_tokens_global, -1).contiguous()
    return y_indx_global


# ------------------------------------------------------------
# fixture
# ------------------------------------------------------------


def _get_free_tcp_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _distributed_worker(rank, fn, world_size, kwargs):
    dev = f"cuda:{rank}"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(dev))
    torch.cuda.set_device(dev)
    try:
        fn(rank=rank, world_size=world_size, **kwargs)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.fixture
def distributed_launcher(request):
    n_gpus = getattr(request, "param", None)
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for distributed GPU test")
    if torch.cuda.device_count() < n_gpus:
        pytest.skip(f"requires up to {n_gpus} CUDA devices, found {torch.cuda.device_count()}")

    master_port = _get_free_tcp_port()

    os.environ["WORLD_SIZE"] = str(n_gpus)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))

    def launch(fn, **kwargs):
        mp.spawn(
            _distributed_worker,
            args=(fn, n_gpus, kwargs),
            nprocs=n_gpus,
            join=True,
        )

    launch.world_size = n_gpus
    return launch


# ------------------------------------------------------------
# expt assignment
# ------------------------------------------------------------


@pytest.mark.parametrize("n_expts_shard, n_expts_tot", [(8, 512), (16, 64)])
@pytest.mark.parametrize("affinity_mode", ["uniform", "random"])
def test_make_expt_assignment(n_expts_shard, n_expts_tot, affinity_mode):
    device = "cuda"
    expt_dict = _make_expt_dict_for_mode(n_expts_shard, n_expts_tot, affinity_mode)
    expt_assignment = make_expt_assignment(n_expts_shard, n_expts_tot, expt_dict, device)
    # mask correctness & uniqueness: each expert set exactly once, and on the right shard
    for shard in range(n_expts_shard):
        bitmask = expt_assignment.expt_bitmask[shard, :]
        bitmask = (bitmask >> torch.arange(32, device=bitmask.device)[:, None]) & 1
        experts = bitmask.T.flatten().nonzero()[:, 0].tolist()
        assert sorted(expt_dict[shard]) == experts
        expt_map = torch.full((n_expts_tot, ), -1, device=device)
        expt_map[experts] = torch.arange(len(experts), device=expt_map.device)
        assert torch.all(expt_map == expt_assignment.expt_map[shard, :])


# ------------------------------------------------------------
# expert sharding
# ------------------------------------------------------------


def routing(logits, n_expts_act, all_gather=False, y_indx=None):
    sparse_logits = topk(logits, n_expts_act, all_gather=all_gather, y_indx=y_indx)
    dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx
    combine_indx = sparse_logits.mask_metadata.col_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(sparse_logits.mask_metadata.col_sum, dispatch_indx.shape[0])
    gather_idx = torch.div(combine_indx, n_expts_act, rounding_mode="trunc")
    scatter_idx = combine_indx
    return ragged_batch_metadata, gather_idx, scatter_idx, sparse_logits.indx


def mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, n_expts_act, y_indx=None):
    rdata, combine_indx, dispatch_indx, _ = routing(l_global, n_expts_act, y_indx=y_indx)
    y_global = matmul(x_global, w_global, b_global, rdata, gather_indx=combine_indx, scatter_indx=dispatch_indx)
    y_mask = (dispatch_indx != -1).view(y_global.shape[-2] // n_expts_act, n_expts_act, 1)
    y_global = y_global.view(y_global.shape[-2] // n_expts_act, n_expts_act, -1)
    y_mask = y_mask.expand_as(y_global)
    y_global, _ = reduce(y_global, dim=1, mask=y_mask)
    return y_global


def mixture_of_expt_epsharded(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act,
                              symm_mem_pool, y_indx=None):
    rank = dist.get_rank()
    expt_map = expt_assignment.expt_map[rank, :]
    # active global logits (sparse)
    l_global_active = topk(l_dp_local, n_expts_act, apply_softmax=True, all_gather=True, y_indx=y_indx,
                           symm_mem_pool=symm_mem_pool)
    # expert histogram, dispatch/combine indx
    active_indx = l_global_active.indx
    expt_sizes = l_global_active.mask_metadata.col_sum
    dispatch_indx = l_global_active.mask_metadata.row_sorted_indx
    combine_indx = l_global_active.mask_metadata.col_sorted_indx
    # ragged tensor metadata
    x_global_metadata = make_ragged_tensor_metadata(expt_sizes, dispatch_indx.shape[0])
    # convert x from dp-local to expert-sorted, ep-local
    y_ep_local = convert_dp_to_ep(x_dp_local, expt_assignment, active_indx, dispatch_indx, symm_mem_pool)
    y_ep_local_metadata = remap_ragged_tensor_metadata(x_global_metadata, expt_map)
    # matrix multiply
    y_ep_local = matmul(y_ep_local, w_ep_local, b_ep_local, a_ragged_metadata=y_ep_local_metadata)
    # convert x from expert-sorted, ep-local to token-sorted, dp-local
    y_dp_local = convert_ep_to_dp(y_ep_local, expt_assignment, active_indx, combine_indx, symm_mem_pool)
    # weighted average of the output token from experts
    y_dp_local = y_dp_local.view(-1, n_expts_act, y_dp_local.shape[-1])
    z_dp_local, _ = reduce(y_dp_local, dim=1)
    return z_dp_local


def _run_expert_sharding(rank, world_size, *, n_tokens, d_model, n_expts_tot, n_expts_act, affinity_mode):
    torch.manual_seed(0)

    dev = torch.cuda.current_device()
    n_shards = world_size

    expt_dict = _make_expt_dict_for_mode(n_shards, n_expts_tot, affinity_mode)
    expt_assignment = make_expt_assignment(n_shards, n_expts_tot, expt_dict, device=dev)
    # reference data
    n_tokens_global = n_tokens
    x_global = torch.randn(n_tokens_global, d_model, device=dev, dtype=torch.bfloat16)
    l_global = torch.rand(n_tokens_global, n_expts_tot, device=dev, dtype=torch.float32)
    w_global = torch.randn((n_expts_tot, d_model, d_model), device=dev, dtype=torch.bfloat16)
    b_global = torch.randn((n_expts_tot, d_model), device=dev, dtype=torch.float32)
    # initialize data shard
    n_tokens_local = n_tokens_global // n_shards
    first_token_indx, last_token_indx = rank * n_tokens_local, (rank + 1) * n_tokens_local
    w_ep_local = w_global[expt_assignment.expt_boolmask[rank, :], :, :]
    b_ep_local = b_global[expt_assignment.expt_boolmask[rank, :], :]
    x_dp_local = x_global[first_token_indx:last_token_indx, :]
    l_dp_local = l_global[first_token_indx:last_token_indx, :]
    # routing
    # test correctness
    y_indx_global = _make_y_indx_for_mode(n_tokens_global, n_expts_tot, n_expts_act, n_shards, affinity_mode, dev)
    y_global_ref = mixture_of_expt_nosharded(
        x_global,
        l_global,
        w_global,
        b_global,
        n_expts_act,
        y_indx=y_indx_global,
    )

    symm_mem_pool = SymmetricMemoryPool(Mesh(dist.group.WORLD))
    symm_mem_pool.initialize_matmul(
        n_tokens_global=n_tokens_global,
        d_input=d_model,
        d_model=d_model,
        n_expts_act=n_expts_act,
        n_expts_tot=n_expts_tot,
        dtype=torch.bfloat16,
        device=dev,
    )

    def run_moe():
        return mixture_of_expt_epsharded(
            x_dp_local,
            l_dp_local,
            w_ep_local,
            b_ep_local,
            expt_assignment,
            n_expts_act,
            y_indx=y_indx_global,
            symm_mem_pool=symm_mem_pool,
        )

    y_dp_local_tri = run_moe()
    y_global_tri = torch.empty_like(y_global_ref)

    # Validate warmup run.
    dist.all_gather_into_tensor(y_global_tri, y_dp_local_tri)
    triton.testing.assert_close(y_global_ref, y_global_tri)

    # Validate cuda graph capture + replay.
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g):
            y_dp_local_tri_graph = run_moe()

    g.replay()
    dist.all_gather_into_tensor(y_global_tri, y_dp_local_tri_graph)
    triton.testing.assert_close(y_global_ref, y_global_tri)


@pytest.mark.parametrize("distributed_launcher", [2, 4], indirect=True)
@pytest.mark.parametrize("n_tokens", [16, 128, 4096])
@pytest.mark.parametrize("d_model, n_expts_tot, n_expts_act", [(16, 4, 4), (5760, 128, 4)])
@pytest.mark.parametrize("affinity_mode", ["uniform", "random"])
def test_expert_sharding(distributed_launcher, n_tokens, d_model, n_expts_tot, n_expts_act, affinity_mode):
    if n_tokens < distributed_launcher.world_size:
        raise ValueError("n_tokens must be >= number of gpus")
    if n_tokens % distributed_launcher.world_size != 0:
        raise ValueError("n_tokens must be divisible by number of gpus")

    distributed_launcher(
        _run_expert_sharding,
        n_tokens=n_tokens,
        d_model=d_model,
        n_expts_tot=n_expts_tot,
        n_expts_act=n_expts_act,
        affinity_mode=affinity_mode,
    )
