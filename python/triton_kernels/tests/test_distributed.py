import contextlib
import os
import socket

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp
import triton
from triton_kernels.distributed import convert_dp_to_ep, convert_ep_to_dp, make_expt_dict_uniform, make_expt_dict_random, make_expt_assignment
from triton_kernels.topk import topk
from triton_kernels.matmul_ogs import matmul_ogs, reduce_grouped, RoutingData, GatherIndx, ScatterIndx
from triton_kernels.target_info import is_hip
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


def routing(logits, n_expts_act, all_gather=False):
    sparse_logits = topk(logits, n_expts_act, all_gather=all_gather)
    dispatch_indx = sparse_logits.mask_metadata.col_sorted_indx
    combine_indx = sparse_logits.mask_metadata.row_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(sparse_logits.mask_metadata.col_sum, dispatch_indx.shape[0])
    gate_scal = sparse_logits.vals.flatten()[combine_indx]
    routing_data = RoutingData(gate_scal, ragged_batch_metadata.slice_sizes, logits.shape[-1], n_expts_act,
                               ragged_batch_metadata)
    gather_idx = GatherIndx(combine_indx, dispatch_indx)
    scatter_idx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_idx, scatter_idx, sparse_logits.indx


def mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, n_expts_act):
    rdata, combine_indx, dispatch_indx, _ = routing(l_global, n_expts_act)
    y_global = matmul_ogs(x_global, w_global, b_global, rdata, gather_indx=combine_indx, scatter_indx=dispatch_indx)
    return y_global


def mixture_of_expt_epsharded(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act):
    rank = dist.get_rank()
    rdata_global, combine_indx, dispatch_indx, expt_indx = routing(l_dp_local, n_expts_act, all_gather=True)
    y_ep_local = convert_dp_to_ep(x_dp_local, expt_assignment, expt_indx, dispatch_indx.src_indx)
    expt_data_local = remap_ragged_tensor_metadata(rdata_global.expt_data, expt_assignment.expt_map[rank, :])
    rdata_ep_local = RoutingData(
        gate_scal=rdata_global.gate_scal,
        expt_hist=expt_data_local.slice_sizes,
        n_expts_tot=expt_assignment.n_expts_per_shard[rank],
        n_expts_act=rdata_global.n_expts_act,
        expt_data=expt_data_local,
    )
    y_ep_local = matmul_ogs(y_ep_local, w_ep_local, b_ep_local, rdata_ep_local)
    y_dp_local = convert_ep_to_dp(y_ep_local, expt_assignment, expt_indx, combine_indx.src_indx)
    z_dp_local = reduce_grouped(y_dp_local, contig_group_size=n_expts_act)[0]
    return z_dp_local


def _capture_with_prepared_symm_mem(fn):
    """
    Run `fn` once to record symmetric-memory allocations, preallocate them outside the CUDA graph,
    and capture a CUDA graph that reuses the recorded buffers.
    """
    orig_symm_empty = symm_mem.empty
    orig_symm_rendezvous = symm_mem.rendezvous
    recorded_empty_calls = []
    recorded_rendezvous_calls = []
    buffer_id_to_index = {}

    def recording_empty(*args, **kwargs):
        buf = orig_symm_empty(*args, **kwargs)
        idx = len(recorded_empty_calls)
        buffer_id_to_index[id(buf)] = idx
        recorded_empty_calls.append((args, dict(kwargs)))
        return buf

    def recording_rendezvous(buf, *args, **kwargs):
        buf_id = id(buf)
        if buf_id not in buffer_id_to_index:
            raise RuntimeError("symm_mem.rendezvous called on unknown buffer")
        hdl = orig_symm_rendezvous(buf, *args, **kwargs)
        recorded_rendezvous_calls.append((buffer_id_to_index[buf_id], args, dict(kwargs)))
        return hdl

    symm_mem.empty = recording_empty
    symm_mem.rendezvous = recording_rendezvous
    try:
        warmup_result = fn()
    finally:
        symm_mem.empty = orig_symm_empty
        symm_mem.rendezvous = orig_symm_rendezvous

    prepared_empty_buffers = [orig_symm_empty(*args, **kwargs) for args, kwargs in recorded_empty_calls]
    prepared_handles = [
        orig_symm_rendezvous(prepared_empty_buffers[idx], *args, **kwargs)
        for idx, args, kwargs in recorded_rendezvous_calls
    ]

    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    if recorded_empty_calls:
        empty_idx = 0
        rendezvous_idx = 0

        def reuse_empty(*args, **kwargs):
            nonlocal empty_idx
            if empty_idx >= len(prepared_empty_buffers):
                raise RuntimeError("symm_mem.empty called more times than recorded")
            expected_args, expected_kwargs = recorded_empty_calls[empty_idx]
            if expected_args != args or expected_kwargs != kwargs:
                raise RuntimeError("symm_mem.empty called with unexpected arguments")
            buf = prepared_empty_buffers[empty_idx]
            empty_idx += 1
            return buf

        def reuse_rendezvous(buf, *args, **kwargs):
            nonlocal rendezvous_idx
            if rendezvous_idx >= len(prepared_handles):
                raise RuntimeError("symm_mem.rendezvous called more times than recorded")
            expected_empty_idx, expected_args, expected_kwargs = recorded_rendezvous_calls[rendezvous_idx]
            expected_buf = prepared_empty_buffers[expected_empty_idx]
            if buf is not expected_buf:
                raise RuntimeError("symm_mem.rendezvous received unexpected buffer")
            if expected_args != args or expected_kwargs != kwargs:
                raise RuntimeError("symm_mem.rendezvous called with unexpected arguments")
            handle = prepared_handles[rendezvous_idx]
            rendezvous_idx += 1
            return handle

        symm_mem.empty = reuse_empty
        symm_mem.rendezvous = reuse_rendezvous
        try:
            with torch.cuda.stream(capture_stream):
                with torch.cuda.graph(graph):
                    fn()
        finally:
            symm_mem.empty = orig_symm_empty
            symm_mem.rendezvous = orig_symm_rendezvous
    else:
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(graph):
                fn()

    # Keep references alive for as long as the graph exists.
    graph._symm_mem_buffers = prepared_empty_buffers
    graph._symm_mem_handles = prepared_handles
    graph._capture_stream = capture_stream
    return warmup_result, graph


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
    y_global_ref = mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, n_expts_act)

    def run_mixture():
        return mixture_of_expt_epsharded(
            x_dp_local,
            l_dp_local,
            w_ep_local,
            b_ep_local,
            expt_assignment,
            n_expts_act,
        )

    # test cuda graph capture + replay with symmetric memory
    y_dp_local_tri, graph = _capture_with_prepared_symm_mem(run_mixture)
    y_global_tri = torch.empty_like(y_global_ref)

    # Validate warmup run.
    dist.all_gather_into_tensor(y_global_tri, y_dp_local_tri)
    triton.testing.assert_close(y_global_ref, y_global_tri)

    # Validate first replay with unchanged inputs.
    graph.replay()
    dist.all_gather_into_tensor(y_global_tri, y_dp_local_tri)
    triton.testing.assert_close(y_global_ref, y_global_tri)


@pytest.mark.parametrize("distributed_launcher", [2, 4], indirect=True)
@pytest.mark.parametrize("n_tokens", [16, 128, 4096])
@pytest.mark.parametrize("d_model, n_expts_tot, n_expts_act", [(16, 4, 4), (5760, 128, 4)])
@pytest.mark.parametrize("affinity_mode", ["uniform", "random"])
def test_expert_sharding(distributed_launcher, n_tokens, d_model, n_expts_tot, n_expts_act, affinity_mode):
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
        affinity_mode=affinity_mode,
    )
