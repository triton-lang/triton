import torch
import triton
from dataclasses import dataclass, field
from .topk import topk, topk_torch
from .tensor import RaggedTensorMetadata, make_ragged_tensor_metadata
from .tensor import make_bitmatrix_metadata


@dataclass
class RoutingData:
    gate_scal: torch.Tensor = field()
    expt_hist: torch.Tensor = field()
    n_expts_tot: int = field()
    n_expts_act: int = field()
    expt_data: RaggedTensorMetadata = None

    # Used to make perf annotation cleaner: when we use expert sharding, we can
    # use this to tell the "expected" number of local tokens per expert, because
    # the actual number can vary per each input.
    expected_tokens_per_expt: int = field(default=None)

    def n_blocks(self, n_rows, block_size):
        if n_rows <= self.n_expts_tot:
            return n_rows
        else:
            return triton.cdiv(max(n_rows - self.n_expts_tot + 1, 0), block_size) + self.n_expts_tot - 1


# --------------------------
# routing
# --------------------------


def routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act):
    bitmatrix_metadata = make_bitmatrix_metadata(expt_indx, bitmatrix)
    expt_hist = bitmatrix_metadata.col_sum
    combine_indx = bitmatrix_metadata.row_sorted_indx
    dispatch_indx = bitmatrix_metadata.col_sorted_indx
    ragged_tensor_metadata = make_ragged_tensor_metadata(expt_hist, n_expts_tot, expt_scal.numel())
    gate_scal = expt_scal.flatten()[combine_indx]
    return RoutingData(gate_scal, expt_hist, n_expts_tot, n_expts_act,
                       ragged_tensor_metadata), combine_indx, dispatch_indx


def routing(logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None):
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    expt_scal, expt_indx, bitmatrix = topk(logits, n_expts_act, apply_softmax=not sm_first, y_indx=expt_indx,
                                           n_rows=n_rows)
    return routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, logits.shape[-1], n_expts_act)


# --------------------------
# torch reference
# --------------------------


def compute_expt_data_torch(hist, n_expts_tot, n_gates):
    # offset for each experts
    device = hist.device
    batch_offs = torch.cumsum(hist, dim=0)
    batch_offs = torch.cat((torch.zeros(1, device=device), batch_offs))
    batch_offs = batch_offs.int()
    # maximum number of tiles for all values of `block_size` considered
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        # ceil_div(n_gates - n_experts + 1, d_tile) + n_experts - 1
        # ceil_div(x, y): -(-x // y)
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // min(RaggedTensorMetadata.block_sizes()))
    # fill up tile offset/infos for each block
    block_offs = dict()
    block_pid_map = dict()
    for block_size in RaggedTensorMetadata.block_sizes():
        n_tiles = (hist + block_size - 1) // block_size  # matmul blocks needed
        block_offs[block_size] = torch.cumsum(n_tiles, dim=0)
        block_offs[block_size] = torch.cat((torch.zeros(1, device=device), block_offs[block_size]))
        block_offs[block_size] = block_offs[block_size].int()
        # compute data required to drive ragged batch matmul
        block_pid_map[block_size] = -torch.ones(max_n_tiles, dtype=torch.int32, device=device)
        col = torch.arange(max_n_tiles, device=device)
        map_vals = torch.arange(n_expts_tot, device=device)[:, None] + (col << 16)[None, :]
        map_idxs = block_offs[block_size][:-1, None] + col[None, :]
        mask = col[None, :] < n_tiles[:, None]
        block_pid_map[block_size].index_put_((map_idxs[mask], ), map_vals.int()[mask])
    block_offs = torch.stack(list(block_offs.values()))
    block_pid_map = torch.stack(list(block_pid_map.values()))
    return RaggedTensorMetadata(hist, batch_offs, block_offs, block_pid_map)


def routing_torch(logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None):
    has_user_provided_indx = expt_indx is not None
    n_gates_pad = logits.shape[0] * n_expts_act

    if n_rows is not None:
        logits = logits[:n_rows, :]
    _, n_expts_tot = logits.shape
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    expt_scal, expt_indx = topk_torch(logits, n_expts_act, expt_indx)
    if not sm_first:
        expt_scal = torch.softmax(expt_scal, dim=-1)
    # sort each token's selections by expert
    if not has_user_provided_indx:
        expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)
    # flatten topk data
    expt_scal = expt_scal.reshape(-1)
    expt_indx = expt_indx.reshape(-1).to(torch.int32)
    # sort by expert_id so experts are contiguous for the matmul
    combine_indx = torch.argsort(expt_indx, stable=True).int()
    dispatch_indx = torch.argsort(combine_indx, stable=True).int()
    gate_scal = expt_scal[combine_indx]
    hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1).int()  # histogram of tokens over experts
    # pack the matmul data structure
    # compute expt_data
    expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates_pad)
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), combine_indx, dispatch_indx
