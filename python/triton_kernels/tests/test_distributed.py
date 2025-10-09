import contextlib
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
from triton_kernels.distributed import convert_dp_to_ep, convert_ep_to_dp, make_expt_dict_random, make_expt_assignment, filter_expt_data, filter_expt_data_torch
from triton_kernels.topk import topk
from triton_kernels.matmul_ogs import matmul_ogs, reduce_grouped, RoutingData, GatherIndx, ScatterIndx
from triton_kernels.target_info import is_hip
from triton_kernels.tensor import make_ragged_tensor_metadata
import pytest

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
# filtering
# ------------------------------------------------------------

@pytest.mark.parametrize("n_expt_shard, n_expt_tot", [(8, 512), (16, 64)])
@pytest.mark.parametrize("affinity_mode", ["uniform", "random"])
def test_make_expt_assignment(n_expt_shard, n_expt_tot, affinity_mode):
    device = "cuda"
    expt_dict = {
        "uniform": make_expt_dict_uniform,
        "random": make_expt_dict_random,
    }[affinity_mode](n_expt_shard, n_expt_tot)
    expt_assignment = make_expt_assignment(n_expt_shard, n_expt_tot, expt_dict, device)
    # mask correctness & uniqueness: each expert set exactly once, and on the right shard
    for shard in range(n_expt_shard):
        bitmask = expt_assignment.expt_bitmask[shard, :]
        bitmask = (bitmask >> torch.arange(32, device=bitmask.device)[:, None]) & 1
        experts = bitmask.T.flatten().nonzero()[:, 0].tolist()
        assert sorted(expt_dict[shard]) == experts
        expt_map = torch.full((n_expt_tot, ), -1, device=device)
        expt_map[experts] = torch.arange(len(experts), device=expt_map.device)
        assert torch.all(expt_map == expt_assignment.expt_map[shard, :])


def test_filter_expt_data():
    device = "cuda"
    dtype = torch.float32
    n_expts_tot = 128
    n_expts_act = 4
    n_tokens = 1024
    n_shards = 4
    logits = torch.randn((n_tokens, n_expts_tot), dtype=dtype, device=device, requires_grad=True)
    routing_global, _, _, _ = routing(logits, n_expts_act)
    expt_data = routing_global.expt_data
    expt_dict = make_expt_dict_uniform(n_shards, n_expts_tot)
    expt_assignment = make_expt_assignment(n_shards, n_expts_tot, expt_dict, device)
    routing_local_ref = filter_expt_data_torch(expt_data, expt_assignment, 1)
    routing_local_tri = filter_expt_data(expt_data, expt_assignment, 1)
    assert torch.all(routing_local_ref.hist == routing_local_tri.hist)
    assert torch.all(routing_local_ref.token_offs_raw == routing_local_tri.token_offs_raw)
    assert torch.all(routing_local_ref.token_offs_pad_data == routing_local_tri.token_offs_pad_data)
    assert torch.all(routing_local_ref.block_pid_map_data == routing_local_tri.block_pid_map_data)


# ------------------------------------------------------------


def routing(logits, n_expts_act, all_gather=False):
    sparse_logits = topk(logits, n_expts_act, all_gather=all_gather)
    dispatch_indx = sparse_logits.mask_metadata.col_sorted_indx
    combine_indx = sparse_logits.mask_metadata.row_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(sparse_logits.mask_metadata.col_sum, dispatch_indx.shape[0])
    gate_scal = sparse_logits.vals.flatten()[combine_indx]
    routing_data = RoutingData(gate_scal, ragged_batch_metadata.batch_sizes, logits.shape[-1], n_expts_act,
                               ragged_batch_metadata)
    gather_idx = GatherIndx(combine_indx, dispatch_indx)
    scatter_idx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_idx, scatter_idx

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


@pytest.mark.parametrize("n_expt_shard, n_expt_tot", [(8, 512), (16, 64)])
@pytest.mark.parametrize("affinity_mode", ["uniform", "random"])
def test_make_expt_assignment(n_expt_shard, n_expt_tot, affinity_mode):
    device = "cuda"
    expt_dict = {
        "uniform": make_expt_dict_uniform,
        "random": make_expt_dict_random,
    }[affinity_mode](n_expt_shard, n_expt_tot)
    expt_assignment = make_expt_assignment(n_expt_shard, n_expt_tot, expt_dict, device)
    # mask correctness & uniqueness: each expert set exactly once, and on the right shard
    for shard in range(n_expt_shard):
        bitmask = expt_assignment.expt_bitmask[shard, :]
        bitmask = (bitmask >> torch.arange(32, device=bitmask.device)[:, None]) & 1
        experts = bitmask.T.flatten().nonzero()[:, 0].tolist()
        assert sorted(expt_dict[shard]) == experts
        expt_map = torch.full((n_expt_tot, ), -1, device=device)
        expt_map[experts] = torch.arange(len(experts), device=expt_map.device)
        assert torch.all(expt_map == expt_assignment.expt_map[shard, :])


def test_filter_expt_data():
    device = "cuda"
    dtype = torch.float32
    n_expts_tot = 128
    n_expts_act = 4
    n_tokens = 1024
    n_shards = 4
    logits = torch.randn((n_tokens, n_expts_tot), dtype=dtype, device=device, requires_grad=True)
    routing_global, _, _, _ = routing(logits, n_expts_act)
    expt_data = routing_global.expt_data
    expt_dict = make_expt_dict_uniform(n_shards, n_expts_tot)
    expt_assignment = make_expt_assignment(n_shards, n_expts_tot, expt_dict, device)
    routing_local_ref = filter_expt_data_torch(expt_data, expt_assignment, 1)
    routing_local_tri = filter_expt_data(expt_data, expt_assignment, 1)
    assert torch.all(routing_local_ref.hist == routing_local_tri.hist)
    assert torch.all(routing_local_ref.token_offs_raw == routing_local_tri.token_offs_raw)
    assert torch.all(routing_local_ref.token_offs_pad_data == routing_local_tri.token_offs_pad_data)
    assert torch.all(routing_local_ref.block_pid_map_data == routing_local_tri.block_pid_map_data)


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


def _run_expert_sharding(rank, world_size, *, n_tokens, d_model, n_expts_tot, n_expts_act):
    torch.manual_seed(0)

    dev = torch.cuda.current_device()
    n_shards = world_size

    expt_dict = make_expt_dict_random(n_shards, n_expts_tot)
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


@pytest.mark.parametrize("distributed_launcher", [2, 4], indirect=True)
@pytest.mark.parametrize("n_tokens", [16, 128, 4096])
@pytest.mark.parametrize("d_model, n_expts_tot, n_expts_act", [(16, 4, 4), (5760, 128, 4)])
def test_expert_sharding(distributed_launcher, n_tokens, d_model, n_expts_tot, n_expts_act):
    if is_hip():
        pytest.skip("Distributed test is not supported on AMD GPU")
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
    )
