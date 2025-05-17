import torch
import triton
from dataclasses import dataclass, field
from .routing_details._routing_compute import _routing_memset_indx
from .routing_details._routing_compute import _routing_compute_indx_offs
from .routing_details._routing_compute import _routing_compute_indx
from .routing_details._routing_compute import _routing_clear_bitmatrix


@dataclass
class GatherIndx:
    """
    Indices for an operation that performs:
    Y = X[src_idx, :]
    """
    # array such that `dst_idx[src_idx] = arange(0, N)`
    src_indx: torch.Tensor
    dst_indx: torch.Tensor


@dataclass
class ScatterIndx:
    """
    Indices for an operation that performs:
    Y[dst_idx, :] = X
    """
    # array such that `dst_idx[src_idx] = arange(0, N)`
    src_indx: torch.Tensor
    dst_indx: torch.Tensor


@dataclass
class RoutingData:
    gate_scal: torch.Tensor = field()
    expt_hist: torch.Tensor = field()
    n_expts_tot: int = field()
    n_expts_act: int = field()

    # Used to make perf annotation cleaner: when we use expert sharding, we can
    # use this to tell the "expected" number of local tokens per expert, because
    # the actual number can vary per each input.
    expected_tokens_per_expt: int = field(default=None)

    def n_blocks(self, n_rows, block_m):
        if n_rows <= self.n_expts_tot:
            return n_rows
        else:
            return triton.cdiv(max(n_rows - self.n_expts_tot + 1, 0), block_m) + self.n_expts_tot - 1


# --------------------------
# Triton routing
# --------------------------


def routing(logits, n_expts_act, expt_indx=None, simulated_ep=1):
    from .topk import topk
    from .reduction import sum
    from .compaction import compaction
    assert expt_indx is None
    cdiv = triton.cdiv
    HIST_BLOCK_M = 64
    INDX_OFFS_BLOCK_M = 512
    MEMSET_BLOCK = 1024
    assert logits.dtype.itemsize == 2
    n_tokens, n_expts_tot = logits.shape
    n_gates = n_tokens * n_expts_act
    device = logits.device
    expt_scal, expt_indx, bitmatrix = topk(logits, n_expts_act)
    # mutate bitmatrix
    if simulated_ep > 1:
        assert n_expts_tot % simulated_ep == 0
        _routing_clear_bitmatrix[(n_tokens, )](
            bitmatrix.data,
            bitmatrix.data.stride(0),
            bitmatrix.data.shape[1],
            n_expts_tot // simulated_ep,
            BLOCK_N=512,
        )
        expt_scal, expt_indx = compaction(expt_scal, expt_indx, bitmatrix)
        n_expts_tot = n_expts_tot // simulated_ep
        bitmatrix.shape[-1] = n_expts_tot
    # perform compaction to update expt_scal / expt_indx
    hist, partial_hist = sum(bitmatrix, partials_block_size=HIST_BLOCK_M, dim=0)
    # scratchpad
    expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
    indx_offs = torch.empty((cdiv(n_tokens, HIST_BLOCK_M), n_expts_tot), dtype=torch.int32, device=device)
    combined_indx = torch.empty(n_gates * 2, dtype=torch.int32, device=device)
    # output
    topk_indx = combined_indx[:n_gates]
    gate_indx = combined_indx[n_gates:]
    gate_scal = torch.empty(n_gates, dtype=logits.dtype, device=device)
    _routing_memset_indx[(cdiv(n_gates * 2, MEMSET_BLOCK) + 1, )](combined_indx, n_gates * 2, -1, MEMSET_BLOCK, hist,
                                                                  expt_offs, hist.shape[0], BLOCK_N=512)
    _routing_compute_indx_offs[(n_expts_tot, )](
        expt_offs, partial_hist,  # inputs
        indx_offs, partial_hist.shape[0], partial_hist.stride(0),  # outputs
        BLOCK_M=INDX_OFFS_BLOCK_M,  # tunable parameters
    )
    _routing_compute_indx[(cdiv(n_tokens, HIST_BLOCK_M), )](
        topk_indx, gate_indx, gate_scal,  # outputs
        expt_scal, expt_indx, indx_offs, indx_offs.stride(0), n_gates,  # input
        BLOCK_M=HIST_BLOCK_M,  # tunable parameters
        N_EXPTS_ACT=n_expts_act,  # constants
        num_warps=1 if HIST_BLOCK_M * n_expts_act // 32 < 4 else 4)
    # pack the matmul data structure
    gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act), gather_indx, scatter_indx


def routing_torch(logits, n_expts_act, expt_indx=None, simulated_ep=1):
    assert expt_indx is None
    assert simulated_ep == 1

    def topk(vals, k, expt_indx):
        # topk of experts
        if expt_indx is None:
            tk_idx = torch.argsort(-vals, dim=1, stable=True)[:, :k]
        else:
            tk_idx = expt_indx
        tk_val = torch.take_along_dim(vals, tk_idx, dim=1)
        return tk_val, tk_idx

    _, n_expts_tot = logits.shape
    expt_scal, expt_indx = topk(logits, n_expts_act, expt_indx)
    expt_scal = torch.softmax(expt_scal, dim=-1)
    # Sort each token's selections by expert
    expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
    expt_scal = torch.gather(expt_scal, 1, sort_indices)
    # flatten topk data
    expt_scal = expt_scal.reshape(-1)
    expt_indx = expt_indx.reshape(-1).to(torch.int32)
    # sort by expert_id so experts are contiguous for the matmul
    topk_indx = torch.argsort(expt_indx, stable=True)
    gate_indx = torch.argsort(topk_indx)
    gate_scal = expt_scal[topk_indx]
    hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1)  # histogram of tokens over experts
    # pack the matmul data structure
    gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
    scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act), gather_indx, scatter_indx
