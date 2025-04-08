import torch
import triton
from dataclasses import dataclass, field
import triton.language as tl


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
class ExptData:
    hist: torch.Tensor
    offs: torch.Tensor
    offs_sum: torch.Tensor
    blocks: torch.Tensor
    buffer: torch.Tensor


# Expert data kernel
@triton.jit
def _fill_expt_data(
    ExpertData,
    n_blocks,
    n_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    expert_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    # Load tokens per expert, compute blocks per expert, then cumsum of blocks per expert needed
    # for expert's offset into ExpertData block_map field
    ranges_offs = tl.where(offs > 0, offs - 1, 0)
    expert_mask = offs > 0 and ranges_offs < n_experts
    tokens_per_expert = tl.load(ExpertData + ranges_offs, mask=expert_mask, other=0)
    blocks_per_expert = tl.cdiv(tokens_per_expert, BLOCK_M)
    block_starts = tl.cumsum(blocks_per_expert, axis=0)

    # All program IDs compute block ranges but only first program ID stores block ranges and token
    # ranges to the expt_data buffer
    if expert_id == 0:
        # Compute token ranges per expert
        token_starts = tl.cumsum(tokens_per_expert, axis=0)

        store_mask = offs < (n_experts + 1)
        tl.store(ExpertData + n_experts + offs, token_starts, mask=store_mask)
        tl.store(ExpertData + 2 * n_experts + 1 + offs, block_starts, mask=store_mask)

    # Get our expert's first block idx
    block_starts = tl.where(offs == expert_id, block_starts, 0)
    first_block = tl.sum(block_starts, axis=0)

    # Compute per-block expert id and expert-local block idx
    n_tokens = tl.load(ExpertData + expert_id)
    n_blocks = tl.cdiv(n_tokens, BLOCK_M)
    ExpertData += 3 * n_experts + 2 + first_block
    for block_off in range(0, n_blocks, BLOCK_N):
        block_offs = block_off + tl.arange(0, BLOCK_N)
        data = (block_offs << 16) + expert_id
        tl.store(ExpertData + block_offs, data, mask=block_offs < n_blocks)


@dataclass
class RoutingData:
    gate_scal: torch.Tensor = field()
    expt_hist: torch.Tensor = field()
    n_expts_tot: int = field()
    n_expts_act: int = field()
    expt_data_map: dict[int, torch.Tensor] = field(default_factory=dict, init=False)

    # Used to make perf annotation cleaner: when we use expert sharding, we can
    # use this to tell the "expected" number of local tokens per expert, because
    # the actual number can vary per each input.
    expected_tokens_per_expt: int = field(default=None)

    def n_blocks(self, n_rows, block_m):
        if n_rows <= self.n_expts_tot:
            return n_rows
        else:
            return triton.cdiv(max(n_rows - self.n_expts_tot + 1, 0), block_m) + self.n_expts_tot - 1

    def _compute_expt_data(self, n_rows, block_m):
        routing_matrix = None
        expt_histogram = self.expt_hist
        assert routing_matrix is not None or expt_histogram is not None, ("Must pass routing_matrix or expt_histogram")
        n_experts = routing_matrix.shape[1] if routing_matrix is not None else expt_histogram.numel()
        device = routing_matrix.device if routing_matrix is not None else expt_histogram.device
        if n_rows < n_experts:
            n_blocks = n_rows
        else:
            n_blocks = triton.cdiv(n_rows - n_experts + 1, block_m) + n_experts - 1

        shape = n_experts * 3 + 2 + n_blocks
        expt_data = torch.full((shape, ), -1, dtype=torch.int32, device=device)
        if expt_histogram is not None:
            expt_data[:n_experts] = expt_histogram
        else:
            torch.sum(routing_matrix, dim=0, out=expt_data[:n_experts])

        BLOCK_N = triton.next_power_of_2(n_experts + 1)
        grid = (n_experts, )
        _fill_expt_data[grid](
            expt_data,
            n_blocks,
            n_experts,
            block_m,
            BLOCK_N,
        )
        n_expts_tot = self.n_expts_tot
        hist = expt_data[:n_expts_tot]
        offs = expt_data[n_expts_tot:2 * n_expts_tot + 1]
        offs_sum = expt_data[3 * n_expts_tot + 2 - 1]
        blocks = expt_data[n_expts_tot + 2 * (n_expts_tot + 1):]
        return ExptData(hist, offs, offs_sum, blocks, expt_data)

    def expt_data(self, n_rows, block_m):
        if self.expt_hist is None:
            return ExptData(None, None, None, None, None)
        key = (n_rows, block_m)
        if key not in self.expt_data_map:
            self.expt_data_map[key] = self._compute_expt_data(*key)
        return self.expt_data_map[key]


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
    expt_scal, expt_indx = topk(torch.softmax(logits, dim=-1), n_expts_act, expt_indx)
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
