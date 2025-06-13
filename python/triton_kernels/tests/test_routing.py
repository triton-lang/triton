import pytest
import torch
from triton_kernels.routing import routing, routing_torch
from triton_kernels.testing import assert_close
from triton_kernels.testing import assert_equal
from triton_kernels.target_info import is_hip


def init_data(n_tokens, n_expts_tot, dtype=torch.float32, device="cuda"):
    logits = torch.randn((n_tokens, n_expts_tot), dtype=dtype, device=device, requires_grad=True)
    return logits


n_tokens = [(x, None) for x in [371, 255, 256, 4096, 1023, 1024]]
n_tokens += [(1152, 911)]


@pytest.mark.parametrize("n_tokens_pad, n_tokens_raw", n_tokens)
@pytest.mark.parametrize("n_expts_tot, n_expts_act", [(128, 32), (1500, 8)])
@pytest.mark.parametrize("use_expt_indx", [False, True])
@pytest.mark.parametrize("sm_first", [True, False])
@pytest.mark.skipif(is_hip(), reason="Tests are currently broken on AMD")
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
    ref_logits = tri_logits.clone().detach().requires_grad_(True)

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

    def _assert_indx_equal(ref, tri):
        assert_equal(ref, tri[:len(ref)])
        assert torch.all(tri[len(ref):] == -1)

    assert_close(ref_routing_data.gate_scal, tri_routing_data.gate_scal[:n_gates_raw], 2e-2, 4e-3)
    assert_equal(ref_routing_data.expt_hist, tri_routing_data.expt_hist)

    ref_expt_data = ref_routing_data.expt_data
    tri_expt_data = tri_routing_data.expt_data
    assert_equal(ref_expt_data.hist, tri_expt_data.hist)
    assert_equal(ref_expt_data.token_offs_raw, tri_expt_data.token_offs_raw)
    assert len(ref_expt_data.token_offs_pad) == len(tri_expt_data.token_offs_pad)
    assert len(ref_expt_data.block_pid_map) == len(tri_expt_data.block_pid_map)
    for block_m in ref_expt_data.token_offs_pad.keys():
        assert_equal(ref_expt_data.token_offs_pad[block_m], tri_expt_data.token_offs_pad[block_m])
        assert_equal(ref_expt_data.block_pid_map[block_m], tri_expt_data.block_pid_map[block_m])

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


def bench_routing():
    import triton.profiler as proton
    n_tokens = 8192
    n_expts_tot, n_expts_act = 128, 4
    tri_logits = init_data(n_tokens, n_expts_tot)
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
