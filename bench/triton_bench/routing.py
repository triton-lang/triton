import torch
import triton
from dataclasses import dataclass, field
from . import routing_details


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


def routing(logits, n_expts_act, expt_indx=None):
    assert expt_indx is None
    cdiv = triton.cdiv
    ROUTING_BLOCK_M = 8
    ROUTING_BLOCK_N = 128
    HIST1_BLOCK_M = 64
    HIST1_BLOCK_N = 32
    HIST1_BLOCK_N2 = 512
    HIST2_BLOCK_M = 512
    MEMSET_BLOCK = 512
    assert logits.dtype.itemsize == 2
    n_tokens, n_expts_tot = logits.shape
    n_gates = n_tokens * n_expts_act
    dev = logits.device
    n_expts_pad = cdiv(n_expts_tot, ROUTING_BLOCK_N) * ROUTING_BLOCK_N
    n_expts_words = n_expts_pad // 32
    # scratchpad tensors
    # NOTE: these are not returned
    expt_scal = torch.empty((n_tokens, n_expts_act), dtype=logits.dtype, device=dev)
    expt_indx = torch.empty((n_tokens, n_expts_act), dtype=torch.int16, device=dev)
    bitmatrix = torch.empty((n_tokens, n_expts_words), dtype=torch.uint32, device=dev)
    tok_starts = torch.empty(n_expts_tot + 1, dtype=torch.int32, device=dev)
    partial_hist = torch.empty((cdiv(n_tokens, HIST1_BLOCK_M), n_expts_tot), dtype=torch.int32, device=dev)
    partial_offs = torch.empty((cdiv(n_tokens, HIST1_BLOCK_M), n_expts_tot), dtype=torch.int32, device=dev)
    # output tensors
    hist = torch.empty(n_expts_tot, dtype=torch.int32, device=dev)
    topk_indx = torch.empty(n_gates, dtype=torch.int32, device=dev)
    gate_indx = torch.empty(n_gates, dtype=torch.int32, device=dev)
    gate_scal = torch.empty(n_gates, dtype=logits.dtype, device=dev)
    routing_details.bitmatrix._compute_bitmatrix[(cdiv(n_tokens, ROUTING_BLOCK_M), )](
        logits, logits.stride(0),  # inputs
        expt_scal, expt_indx, expt_scal.stride(0),  # output [topk]
        bitmatrix, bitmatrix.stride(0),  # output [bitmatrix]
        n_tokens, n_expts_tot,  # shapes
        BLOCK_M=ROUTING_BLOCK_M,  # tunable parameter
        N_EXPTS_PAD=n_expts_pad, N_EXPTS_ACT=n_expts_act,  # constants
        BLOCK_N=ROUTING_BLOCK_N)
    routing_details.histogram._memset_hist[(cdiv(hist.shape[0], MEMSET_BLOCK), )](
        hist, hist.shape[0], tok_starts,  # outputs
        BLOCK=MEMSET_BLOCK  # tunable parameter
    )
    routing_details.histogram._compute_hist[(cdiv(n_tokens, HIST1_BLOCK_M), cdiv(n_expts_tot, HIST1_BLOCK_N))](
        bitmatrix, bitmatrix.shape[0], bitmatrix.stride(0),  # input
        hist,  # output [histogram]
        tok_starts, partial_hist, partial_hist.stride(0), partial_hist.shape[1],  # output [cumsums]
        BLOCK_M=HIST1_BLOCK_M, BLOCK_N=HIST1_BLOCK_N,  # tunable parameters
        BLOCK_N2=HIST1_BLOCK_N2,  # constants
    )
    routing_details.histogram._finalize_hist[(n_expts_tot, )](
        tok_starts, partial_hist,  # inputs
        partial_offs, partial_hist.shape[0], partial_hist.stride(0),  # outputs
        BLOCK_M=HIST2_BLOCK_M,  # tunable parameters
    )
    routing_details.indexing._compute_indx[(cdiv(n_tokens, HIST1_BLOCK_M), )](
        topk_indx, gate_indx, gate_scal,  # outputs
        expt_scal, expt_indx, partial_offs, partial_offs.stride(0), n_gates,  # input
        BLOCK_M=HIST1_BLOCK_M,  # tunable parameters
        N_EXPTS_ACT=n_expts_act,  # constants
        num_warps=1 if HIST1_BLOCK_M * n_expts_act // 32 < 4 else 4)
    # pack the matmul data structure
    gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act), gather_indx, scatter_indx


def routing_torch(logits, n_expts_act, expt_indx=None):

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


def simulate_expert_sharded_routing(n_global_rows, routing_data, n_expt_shards, row_align=1, device="cuda", cache=None):
    n_expts_local = routing_data.n_expts_tot // n_expt_shards
    # Ignore gate projection, and create a random routing that simulates
    # routing data gathered from all expert shards. The resulting data will
    # be bogus; it's only intended for perf measurement.
    if cache is None or n_global_rows not in cache:
        # Choose n_expts_act experts for each row, with even distribution.
        weights = torch.ones(n_global_rows, routing_data.n_expts_tot, device=device)
        expt_indx = torch.multinomial(weights, num_samples=routing_data.n_expts_act, replacement=False)

        # Sort each token's selections by expert.
        expt_indx, _ = expt_indx.sort(dim=1)

        hist = torch.histc(expt_indx, bins=routing_data.n_expts_tot, max=routing_data.n_expts_tot - 1)[:n_expts_local]

        # for each row, count how many of its experts are local
        num_local_expts = (expt_indx < n_expts_local).sum(dim=1)
        # Count the number of rows that are for local experts, padded to an alignment.
        n_local_rows = (num_local_expts != 0).sum()
        n_local_rows = ((n_local_rows + row_align - 1) // row_align) * row_align

        is_active = torch.argsort((num_local_expts == 0).to(torch.int8), stable=True)
        expt_indx = expt_indx[is_active].view(-1)
        # Note: Because the number of rows routed to each expert is only known at runtime,
        # we do not drop tokens that are not routed to the local expert. This ensures that
        # the tensor shapes are fixed.
        # Create topk_indx/gate_indx.
        topk_indx = torch.argsort(expt_indx.view(-1), stable=True).to(torch.int32)
        gate_indx = torch.argsort(topk_indx).to(torch.int32)
        # Now filter out all experts >= n_expts_local
        expt_indx = torch.where(expt_indx < n_expts_local, expt_indx, -1)
        gate_indx = torch.where(expt_indx == -1, -1, gate_indx)
        topk_indx = torch.where(gate_indx[topk_indx] == -1, -1, topk_indx)

        if cache is not None:
            cache[n_global_rows] = hist, gate_indx, topk_indx, n_local_rows
    else:
        hist, gate_indx, topk_indx, n_local_rows = cache[n_global_rows]

    tokens_per_expt = ((n_global_rows // n_expt_shards) * routing_data.n_expts_act) // n_expts_local

    # Expand gate_scal to the global number of rows.
    # TODO: This currently adds a bogus "elementwise" kernel to the profile.
    gate_scal = routing_data.gate_scal.repeat_interleave(n_expt_shards, dim=0)

    return (
        n_local_rows,
        RoutingData(
            gate_scal,
            hist,
            n_expts_local,
            routing_data.n_expts_act,
            expected_tokens_per_expt=tokens_per_expt,
        ),
        GatherIndx(src_indx=topk_indx, dst_indx=gate_indx),
        ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx),
    )
