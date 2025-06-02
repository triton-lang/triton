import pytest
import torch
from triton_kernels.routing import routing, routing_torch
from triton_kernels.testing import assert_close
from triton_kernels.matmul_ogs_details.metadata import ExptData, compute_metadata
from triton_kernels.testing import assert_equal


def init_data(n_tokens, n_expts_tot, dtype=torch.float16, device="cuda"):
    logits = torch.randn((n_tokens, n_expts_tot), dtype=dtype, device=device, requires_grad=True)
    return logits


def ref_expt_data(routing_data, n_gates):
    n_expts_tot = routing_data.n_expts_tot
    # histogram
    hist = routing_data.expt_hist
    # offset for each experts
    device = hist.device
    token_offs_raw = torch.cumsum(hist, dim=0)
    token_offs_raw = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), token_offs_raw))
    # maximum number of tiles for all values of `block_m` considered
    block_ms = [16, 32, 64, 128]
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        # ceil_div(n_gates - n_experts + 1, d_tile) + n_experts - 1
        # ceil_div(x, y): -(-x // y)
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // min(block_ms))
    # fill up tile offset/infos for each block
    token_offs_pad = dict()
    block_id_map = dict()
    for block_m in [16, 32, 64, 128]:
        n_tiles = (hist + block_m - 1) // block_m  # matmul blocks needed
        token_offs_pad[block_m] = torch.cumsum(n_tiles, dim=0)
        token_offs_pad[block_m] = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), token_offs_pad[block_m]))
        # compute data required to drive ragged batch matmul
        block_id_map[block_m] = -torch.ones(max_n_tiles, dtype=torch.int32, device=device)
        for e in range(n_expts_tot):
            offset = token_offs_pad[block_m][e]
            for b in range(n_tiles[e]):
                block_id_map[block_m][offset + b] = (b << 16) + e
    return ExptData(hist, token_offs_raw, token_offs_pad, block_id_map)


n_tokens = [(x, None) for x in [371, 255, 256, 4096, 1023, 1024]]
n_tokens += [(1152, 911)]


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
    tri_logits = init_data(n_tokens_pad, n_expts_tot, device=device).detach()
    tri_logits[n_tokens_raw:, :] = float("inf")  # should not be used
    tri_logits = tri_logits.requires_grad_(True)
    ref_logits = tri_logits.detach()[:n_tokens_raw, :].requires_grad_(True)

    if use_expt_indx:
        rand_idx = lambda: torch.randperm(n_expts_tot, device="cuda", dtype=torch.int64)
        tri_expt_indx = torch.stack([rand_idx()[:n_expts_act] for _ in range(n_tokens_pad)])
        tri_expt_indx, _ = torch.sort(tri_expt_indx, dim=1)
        tri_expt_indx[n_tokens_raw:] = -99999  # should not be used
        ref_expt_indx = tri_expt_indx[:n_tokens_raw]
    else:
        tri_expt_indx = ref_expt_indx = None
    ref_routing_data, ref_gather, ref_scatter = routing_torch(ref_logits, n_expts_act, sm_first, ref_expt_indx,
                                                              n_rows=n_routing_rows)
    tri_routing_data, tri_gather, tri_scatter = routing(tri_logits, n_expts_act, sm_first, tri_expt_indx,
                                                        n_rows=n_routing_rows)
    ref_metadata = ref_expt_data(ref_routing_data, n_gates_raw)
    tri_metadata = compute_metadata(tri_routing_data, n_gates_raw)

    def _assert_indx_equal(ref, tri):
        assert_equal(ref, tri[:len(ref)])
        assert torch.all(tri[len(ref):] == -1)

    assert_close(ref_routing_data.gate_scal, tri_routing_data.gate_scal[:n_gates_raw], 2e-2, 4e-3)
    assert_equal(ref_routing_data.expt_hist, tri_routing_data.expt_hist)

    assert_equal(ref_metadata.hist, tri_metadata.hist)
    assert_equal(ref_metadata.token_offs_raw, tri_metadata.token_offs_raw)
    assert len(ref_metadata.token_offs_pad) == len(tri_metadata.token_offs_pad)
    assert len(ref_metadata.block_id_map) == len(tri_metadata.block_id_map)
    for block_m in ref_metadata.token_offs_pad.keys():
        assert_equal(ref_metadata.token_offs_pad[block_m], tri_metadata.token_offs_pad[block_m])
        assert_equal(ref_metadata.block_id_map[block_m], tri_metadata.block_id_map[block_m])

    assert ref_routing_data.n_expts_tot == ref_routing_data.n_expts_tot
    assert ref_routing_data.n_expts_act == ref_routing_data.n_expts_act

    _assert_indx_equal(ref_gather.src_indx, tri_gather.src_indx)
    _assert_indx_equal(ref_gather.dst_indx, tri_gather.dst_indx)
    _assert_indx_equal(ref_scatter.src_indx, tri_scatter.src_indx)
    _assert_indx_equal(ref_scatter.dst_indx, tri_scatter.dst_indx)

    scales_grad = torch.randn_like(tri_routing_data.gate_scal)
    ref_routing_data.gate_scal.backward(scales_grad[:n_gates_raw])
    tri_routing_data.gate_scal.backward(scales_grad)

    assert_close(ref_logits.grad, tri_logits.grad[:n_tokens_raw])


def bench_routing():
    import triton.profiler as proton
    n_tokens = 8192
    block_m = 128
    n_expts_tot, n_expts_act = 128, 4
    tri_logits = init_data(n_tokens, n_expts_tot)
    proton.start("routing")
    proton.activate()
    for i in range(100):
        tri_routing_data, tri_gather, tri_scatter = routing(tri_logits, n_expts_act)
        tri_metadata = compute_metadata(tri_routing_data, n_tokens * n_expts_act, block_m)  # noqa: F841
    proton.finalize()
    try:
        import os
        os.system("proton-viewer -m time/ms routing.hatchet")
    except Exception:
        pass


if __name__ == "__main__":
    bench_routing()
