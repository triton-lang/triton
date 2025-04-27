import pytest
import torch
from triton_kernels.routing import routing, routing_torch
from triton_kernels.testing import assert_close
from triton_kernels.matmul_ogs_details.metadata import compute_metadata
from triton_kernels.testing import assert_equal


def init_data(n_tokens, n_expts_tot, dtype=torch.float16, device="cuda"):
    # the reference implementation and the triton implementation do not tie-break experts the same way
    randbits = [torch.randperm(n_expts_tot) for _ in range(n_tokens)]
    x = [(-1)**i * ((16384 + ((i * 512) % 4096) + bits).to(torch.int16).view(dtype)) for i, bits in enumerate(randbits)]
    return torch.stack(x).to(device=device)


def ref_expt_data(routing_data, n_gates, block_m):
    hist = routing_data.expt_hist
    n_expts_tot = routing_data.n_expts_tot
    blks = (hist + block_m - 1) // block_m  # matmul blocks needed
    tsum = torch.cumsum(hist, dim=0)  # prefix sum of tokens
    bsum = torch.cumsum(blks, dim=0)  # prefix sum of blocks
    # Get the max number of matmul blocks of size d_tile needed (and is launched with).
    # This assumes the worst distribution of all experts with one token except for one that has the rest.
    if n_gates <= n_expts_tot:
        grid_m = n_gates
    else:
        # ceil_div(n_gates - n_experts + 1, d_tile) + n_experts - 1
        # ceil_div(x, y): -(-x // y)
        grid_m = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // block_m)
    bloc_data = -torch.ones(grid_m, dtype=torch.int32)
    # compute data required to drive ragged batch matmul
    for e in range(n_expts_tot):
        offset = bsum[e - 1] if e else 0
        for b in range(blks[e]):
            bloc_data[offset + b] = (b << 16) + e

    expt_data = torch.zeros(n_expts_tot * 3 + 2 + grid_m, dtype=torch.int32, device=hist.device)
    expt_data[:n_expts_tot] = routing_data.expt_hist
    expt_data[n_expts_tot + 1:n_expts_tot * 2 + 1] = tsum
    expt_data[n_expts_tot * 2 + 2:n_expts_tot * 3 + 2] = bsum
    expt_data[n_expts_tot * 3 + 2:] = bloc_data
    return expt_data


@pytest.mark.parametrize("n_tokens", [371, 255, 256, 8192, 1023, 1024])
@pytest.mark.parametrize("n_expts_tot, n_expts_act", [(128, 4), (1500, 8)])
@pytest.mark.parametrize("block_m", [64, 128])
def test_op(n_tokens, n_expts_tot, n_expts_act, block_m, device):
    torch.manual_seed(2)
    tri_logits = init_data(n_tokens, n_expts_tot, device=device).detach()
    ref_logits = tri_logits.clone()
    ref_routing_data, ref_gather, ref_scatter = routing_torch(ref_logits, n_expts_act)
    tri_routing_data, tri_gather, tri_scatter = routing(tri_logits, n_expts_act)
    ref_metadata = ref_expt_data(ref_routing_data, n_tokens * n_expts_act, block_m)
    tri_metadata = compute_metadata(tri_routing_data, n_tokens * n_expts_act, block_m).buffer

    assert_close(ref_routing_data.gate_scal, tri_routing_data.gate_scal, 2e-2, 4e-3)
    assert_equal(ref_routing_data.expt_hist, tri_routing_data.expt_hist)
    assert_equal(ref_metadata, tri_metadata)
    assert ref_routing_data.n_expts_tot == ref_routing_data.n_expts_tot
    assert ref_routing_data.n_expts_act == ref_routing_data.n_expts_act

    def _assert_indx_equal(ref, tri):
        assert_equal(ref, tri[:len(ref)])
        assert torch.all(tri[len(ref):] == -1)

    _assert_indx_equal(ref_gather.src_indx, tri_gather.src_indx)
    _assert_indx_equal(ref_gather.dst_indx, tri_gather.dst_indx)
    _assert_indx_equal(ref_scatter.src_indx, tri_scatter.src_indx)
    _assert_indx_equal(ref_scatter.dst_indx, tri_scatter.dst_indx)


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


@pytest.mark.parametrize("n_tokens", [371, 256, 8192])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("n_expts_tot", [128, 256])
def test_topk(n_tokens, k, n_expts_tot, device):
    x = init_data(n_tokens, n_expts_tot, device=device)
    y_vals, y_indx, _ = topk(x, k)
    y_vals_ref, y_indx_ref = torch.topk(x, k, dim=1)
    assert_equal(y_indx, y_indx_ref.sort(dim=1)[0])


if __name__ == "__main__":
    bench_routing()
