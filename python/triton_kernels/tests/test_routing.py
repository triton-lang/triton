import pytest
import torch
from triton_kernels.routing import routing, routing_torch, make_expt_assignment, filter_expt_data_torch, filter_expt_data
from triton_kernels.testing import assert_close
from triton_kernels.testing import assert_equal
import random


def init_data(n_tokens, n_expts_tot, device, dtype=torch.float16):
    logits = torch.randn((n_tokens, n_expts_tot), dtype=dtype, device=device, requires_grad=True)
    return logits


n_tokens = [(x, None) for x in [371, 255, 256, 4096, 1023, 1024]]
n_tokens += [(1152, 911)]


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


@pytest.mark.parametrize("n_tokens_pad, n_tokens_raw", n_tokens)
@pytest.mark.parametrize("n_expts_tot, n_expts_act", [(128, 32), (1500, 8)])
@pytest.mark.parametrize("use_expt_indx", [False, True])
@pytest.mark.parametrize("sm_first", [True, False])
def test_op(n_tokens_pad, n_tokens_raw, n_expts_tot, n_expts_act, sm_first, use_expt_indx, device):
    torch.manual_seed(2)
    if n_tokens_raw is None:
        n_tokens_raw = n_tokens_pad
        n_routing_rows = None
    else:
        n_routing_rows = torch.tensor([n_tokens_raw], dtype=torch.int32, device=device)
    n_gates_raw = n_tokens_raw * n_expts_act
    tri_logits = init_data(n_tokens_pad, n_expts_tot, device=device, dtype=torch.float32).detach()
    tri_logits[n_tokens_raw:, :] = float("inf")  # should not be used
    tri_logits = tri_logits.requires_grad_(True)
    ref_logits = tri_logits.clone().detach().requires_grad_(True)

    if use_expt_indx:
        rand_idx = lambda: torch.randperm(n_expts_tot, device=device, dtype=torch.int64)
        tri_expt_indx = torch.stack([rand_idx()[:n_expts_act] for _ in range(n_tokens_pad)])
        tri_expt_indx, _ = torch.sort(tri_expt_indx, dim=1)
        tri_expt_indx[n_tokens_raw:] = -99999  # should not be used
        ref_expt_indx = tri_expt_indx[:n_tokens_raw]
    else:
        tri_expt_indx = ref_expt_indx = None
    ref_routing_data, ref_gather, ref_scatter, _ = routing_torch(ref_logits, n_expts_act, sm_first, ref_expt_indx,
                                                                 n_rows=n_routing_rows)
    tri_routing_data, tri_gather, tri_scatter, _ = routing(tri_logits, n_expts_act, sm_first, tri_expt_indx,
                                                           n_rows=n_routing_rows)

    def _assert_indx_equal(ref, tri):
        assert_equal(ref, tri[:len(ref)])
        assert torch.all(tri[len(ref):] == -1)

    assert_close(ref_routing_data.gate_scal, tri_routing_data.gate_scal[:n_gates_raw], 2e-2, 4e-3)
    assert_equal(ref_routing_data.expt_hist, tri_routing_data.expt_hist)

    ref_expt_data = ref_routing_data.expt_data
    tri_expt_data = tri_routing_data.expt_data
    assert_equal(ref_expt_data.hist, tri_expt_data.hist)
    assert_equal(ref_expt_data.token_offs_raw, tri_expt_data.token_offs_raw)
    assert_equal(ref_expt_data.token_offs_pad_data, tri_expt_data.token_offs_pad_data)
    assert_equal(ref_expt_data.block_pid_map_data, tri_expt_data.block_pid_map_data)

    assert ref_routing_data.n_expts_tot == ref_routing_data.n_expts_tot
    assert ref_routing_data.n_expts_act == ref_routing_data.n_expts_act

    _assert_indx_equal(ref_gather.src_indx, tri_gather.src_indx)
    _assert_indx_equal(ref_gather.dst_indx, tri_gather.dst_indx)
    _assert_indx_equal(ref_scatter.src_indx, tri_scatter.src_indx)
    _assert_indx_equal(ref_scatter.dst_indx, tri_scatter.dst_indx)

    scales_grad = torch.randn_like(tri_routing_data.gate_scal)
    ref_routing_data.gate_scal.backward(scales_grad[:n_gates_raw])
    tri_routing_data.gate_scal.backward(scales_grad)

    assert_close(ref_logits.grad[:n_tokens_raw], tri_logits.grad[:n_tokens_raw])


def bench_routing(device):
    import triton.profiler as proton
    n_tokens = 8192
    n_expts_tot, n_expts_act = 128, 4
    tri_logits = init_data(n_tokens, n_expts_tot, device)
    proton.start("routing")
    proton.activate()
    for i in range(100):
        tri_routing_data, tri_gather, tri_scatter = routing(tri_logits, n_expts_act)
    proton.finalize()
    try:
        import os
        os.system("proton-viewer -m time/ms routing.hatchet")
    except Exception:
        pass


if __name__ == "__main__":
    bench_routing()
