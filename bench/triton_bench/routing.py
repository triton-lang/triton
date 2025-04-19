import torch
import triton
import triton.language as tl
from dataclasses import dataclass, field


@triton.jit
def _routing_compute_expt_offs(ExpertHist, FinalExpertOffs, hist_size,  # histogram
                               BLOCK_N: tl.constexpr):
    loop_iterations = (hist_size + BLOCK_N - 1) // BLOCK_N
    x = tl.zeros([BLOCK_N], ExpertHist.dtype.element_ty)
    offs_n = tl.arange(0, BLOCK_N)
    for i in range(loop_iterations):
        hist2 = tl.load(ExpertHist + offs_n)
        tok_starts = tl.cumsum(hist2, 0) + x
        x += tl.sum(hist2, 0)
        tl.store(FinalExpertOffs, 0)
        tl.store(FinalExpertOffs + 1 + offs_n, tok_starts, mask=offs_n < hist_size)
        offs_n += BLOCK_N


@triton.jit
def _routing_compute_indx_offs(TokensStart, PartialHist, PartialOffs, shape_pm, stride_pm, BLOCK_M: tl.constexpr):
    expt_id = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    # initialize first row of the output
    start = tl.load(TokensStart + expt_id)
    tl.store(PartialOffs + expt_id, start)
    # iterate over input data
    curr_sum = start
    for _ in range(0, shape_pm, BLOCK_M):
        offs = offs_m * stride_pm + expt_id
        curr = tl.load(PartialHist + offs, mask=offs_m < shape_pm)
        out = tl.cumsum(curr, 0) + curr_sum
        curr_sum += tl.sum(curr, 0)
        offs = (1 + offs_m) * stride_pm + expt_id
        tl.store(PartialOffs + offs, out, mask=offs_m < shape_pm - 1)
        offs_m += BLOCK_M


@triton.jit
def _keyed_add(x, y):

    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xffff0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _count_previous(x):
    """
    Input  x : uint16[..., N]
    Output y : uint32[..., N]
    semantics : y[..., i] = sum_j((x[..., j] == x[..., i]) & (j < i))
    credits: @apgoucher
    """

    BLOCK_N: tl.constexpr = x.shape[-1]  # summation axis
    BATCHES: tl.constexpr = x.numel // BLOCK_N  # number of batches

    # reduce to two-dimensional case:
    y = tl.reshape(x, [BATCHES, BLOCK_N]).to(tl.uint32)

    tl.static_assert(BLOCK_N <= 32768, "compute_run_lengths requires axis to have length <= 32768")

    # sort (expert, position) ordered pairs to perform an argsort:
    kv_pairs = ((y << 16) | tl.arange(0, BLOCK_N)[None, :]).to(tl.uint32)
    sorted_kv_pairs = tl.sort(kv_pairs, 1)

    # compute run lengths in expert-sorted order:
    x = (sorted_kv_pairs & 0xffff0000 | 0x00000001)
    expts_and_inclusive_run_lengths = tl.associative_scan(x, 1, _keyed_add)
    exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xffff

    # undo permutation by doing another sort
    # TODO rewrite this when tl.scatter becomes available
    kv_pairs = ((sorted_kv_pairs << 16) | exclusive_run_lengths).to(tl.uint32)
    unsorted_run_lengths = tl.sort(kv_pairs) & 0xffff

    res = tl.reshape(unsorted_run_lengths, x.shape)
    return res


@triton.jit
def _routing_compute_indx(GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx, PartialOffs, stride_pm, n_gates,
                          BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr):
    pid_m = tl.program_id(0)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + tl.arange(0, N_EXPTS_ACT * BLOCK_M)
    mask = offs < n_gates
    indx = tl.load(ExptIndx + offs, mask=mask)
    gates = tl.load(PartialOffs + pid_m * stride_pm + indx, mask=mask)
    gates += tl.reshape(_count_previous(indx), [BLOCK_M * N_EXPTS_ACT])
    gate_scal = tl.load(ExptScal + offs, mask=mask)
    tl.store(ScatterIndx + offs, gates, mask=mask)
    tl.store(GatherIndx + gates, offs, mask=mask)
    tl.store(GateScal + gates, gate_scal, mask=mask)


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
    from .topk import topk
    from .reduce import sum
    assert expt_indx is None
    cdiv = triton.cdiv
    HIST_BLOCK_M = 64
    HIST2_BLOCK_M = 512
    assert logits.dtype.itemsize == 2
    n_tokens, n_expts_tot = logits.shape
    n_gates = n_tokens * n_expts_act
    device = logits.device
    expt_scal, expt_indx, bitmatrix = topk(logits, n_expts_act)
    hist, partial_hist = sum(bitmatrix, partials_block_size=HIST_BLOCK_M)
    # scratchpad
    expt_offs = torch.empty(n_expts_tot + 1, dtype=torch.int32, device=device)
    indx_offs = torch.empty((cdiv(n_tokens, HIST_BLOCK_M), n_expts_tot), dtype=torch.int32, device=device)
    # output
    topk_indx = torch.empty(n_gates, dtype=torch.int32, device=device)
    gate_indx = torch.empty(n_gates, dtype=torch.int32, device=device)
    gate_scal = torch.empty(n_gates, dtype=logits.dtype, device=device)
    _routing_compute_expt_offs[(1, )](
        hist, expt_offs, hist.shape[0], BLOCK_N=512  # tunable parameters
    )
    _routing_compute_indx_offs[(n_expts_tot, )](
        expt_offs, partial_hist,  # inputs
        indx_offs, partial_hist.shape[0], partial_hist.stride(0),  # outputs
        BLOCK_M=HIST2_BLOCK_M,  # tunable parameters
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
