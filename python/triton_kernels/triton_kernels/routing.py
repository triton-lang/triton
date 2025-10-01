import torch
import triton
from dataclasses import dataclass, field
from .routing_details._routing_compute import _combined_routing_compute
from .routing_details._routing_compute import _combined_routing_memset
from .routing_details._expt_data import _expt_data_memset
from .routing_details._expt_data import _expt_data_compute
from .target_info import is_hip
from .topk import topk, topk_torch


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
    # hist[i] is the number of tokens routed to expert i
    hist: torch.Tensor
    # token_offs_raw[i] is the offset of the first token routed
    # to expert i in an expert-sorted array
    token_offs_raw: torch.Tensor
    # token_offs_pad_data[i, :] is the offset of the first token routed
    # to expert i in an expert-sorted array, assuming histogram
    # rounded to the next multiple of `block = 16 * i`
    token_offs_pad_data: torch.Tensor
    # block_id_map_data[i] contain one value for each `pid`` launched by
    # the matrix multiplication kernel launched with BLOCK_M=i*16:
    # - the value is -1 if the `pid` has no work to do
    # - otherwise, the value is two int16 (packed as an int32) that
    #   correspond respectively to (1) the expert assigned to
    #   the tokens processed by this pid; (2) the block assigned to the
    #   tokens processed by this pid (think `pid_m` in a regular matmul)
    # see `test_routing.py` for a reference implementation and more details
    block_pid_map_data: torch.Tensor

    def __post_init__(self):
        assert self.token_offs_pad_data.shape[0] == len(ExptData.block_ms())
        assert self.block_pid_map_data.shape[0] == len(ExptData.block_ms())
        assert self.token_offs_pad_data.dtype == torch.int32
        assert self.block_pid_map_data.dtype == torch.int32
        if self.hist is not None:
            assert self.hist.dtype == torch.int32
        if self.token_offs_raw is not None:
            assert self.token_offs_raw.dtype == torch.int32

    def token_offs_pad(self, block_m):
        return self.token_offs_pad_data[ExptData.block_ms().index(block_m)]

    def block_pid_map(self, block_m):
        return self.block_pid_map_data[ExptData.block_ms().index(block_m)]

    @staticmethod
    def block_ms_log2():
        return range(4, 9) if is_hip() else range(4, 8)

    @staticmethod
    def block_ms():
        return [2**x for x in ExptData.block_ms_log2()]


@dataclass
class RoutingData:
    gate_scal: torch.Tensor = field()
    expt_hist: torch.Tensor = field()
    n_expts_tot: int = field()
    n_expts_act: int = field()
    expt_data: ExptData = None

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
# sort tokens by expert
# --------------------------


class SortTokens(torch.autograd.Function):

    @staticmethod
    def forward(ctx, expt_scal, expt_indx, n_expts_tot, bitmatrix):
        HIST_BLOCK_M = 32
        INDX_OFFS_BLOCK_M = 512
        MEMSET_BLOCK = 1024
        MEMSET_BLOCK_A = 512
        cdiv = triton.cdiv

        device = expt_scal.device
        dtype = expt_scal.dtype
        n_tokens_raw, _ = bitmatrix.shape
        n_tokens_pad, n_expts_act = expt_scal.shape
        n_gates_pad = n_tokens_pad * n_expts_act
        block_ms_log2 = ExptData.block_ms_log2()
        block_m_num = len(block_ms_log2)

        hist, partial_hist = bitmatrix.sum(partials_block_size=HIST_BLOCK_M)
        hist = hist[:n_expts_tot]
        assert hist.dtype == torch.int32

        # allocate memory
        expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
        combined_indx = torch.empty(n_gates_pad * 2, dtype=torch.int32, device=device)
        gate_scal = torch.empty(n_gates_pad, dtype=dtype, device=device)
        token_offs_combined = empty_aligned((block_m_num + 1, n_expts_tot + 1), torch.int32, device, MEMSET_BLOCK_A)
        block_pid_map = empty_aligned((block_m_num, max_n_tiles(n_expts_tot, n_gates_pad)), torch.int32, device,
                                      MEMSET_BLOCK_A)
        # slice padded allocations
        combine_indx = combined_indx[:n_gates_pad]
        dispatch_indx = combined_indx[n_gates_pad:]
        token_offs_raw, token_offs_pad = token_offs_combined[0], token_offs_combined[1:]

        # grid sizes
        block_pid_map_n_elts = block_pid_map.untyped_storage().size() // block_pid_map.dtype.itemsize
        blocks1a = exact_div(block_pid_map_n_elts, MEMSET_BLOCK_A) + token_offs_combined.shape[0]
        blocks1b = cdiv(n_gates_pad * 2, MEMSET_BLOCK) + n_expts_tot + 1
        blocks2a = n_expts_tot * token_offs_pad.shape[0]
        blocks2b = cdiv(n_tokens_pad, HIST_BLOCK_M)

        _combined_routing_memset[(blocks1a + blocks1b, )](
            combined_indx, n_gates_pad * 2, -1, MEMSET_BLOCK, hist,  #
            expt_offs, hist.shape[0], n_expts_tot, partial_hist,  # inputs
            partial_hist.shape[0], partial_hist.stride(0), partial_hist.stride(1),  # outputs
            token_offs_combined, token_offs_combined.stride(0),  #
            blocks1a, block_pid_map,  #
            block_ms_log2[0], SIZES=len(block_ms_log2), BLOCK_A=MEMSET_BLOCK_A,  # optimization parameters
            BLOCK_N=512, BLOCK_M=INDX_OFFS_BLOCK_M,  # tunable parameters
        )

        indx_offs = partial_hist

        _combined_routing_compute[(blocks2a + blocks2b, )](
            combine_indx, dispatch_indx, gate_scal,  # outputs
            expt_scal, expt_indx, indx_offs, indx_offs.stride(0), indx_offs.stride(1),  # inputs
            expt_offs, n_tokens_raw,  # input shape
            HIST_BLOCK_M, n_expts_act,  # constants
            hist, token_offs_pad, token_offs_pad.stride(0), block_pid_map, block_pid_map.stride(0),  # outputs
            block_ms_log2[0], len(block_ms_log2), 512, blocks2a,  # etc.
        )

        ctx.n_tokens_raw = n_tokens_raw
        ctx.n_tokens_pad = n_tokens_pad
        ctx.n_expts_act = n_expts_act
        ctx.save_for_backward(dispatch_indx)

        return hist, combine_indx, dispatch_indx, gate_scal, token_offs_raw, token_offs_pad, block_pid_map

    @staticmethod
    def backward(ctx, _0, _1, _2, dgate_scal, _3, _4, _5):
        (dispatch_indx, ) = ctx.saved_tensors
        dgate_scal = dgate_scal[dispatch_indx]
        dgate_scal = dgate_scal.reshape(ctx.n_tokens_pad, ctx.n_expts_act)
        return dgate_scal, None, None, None


def sort_tokens(expt_scal, expt_indx, n_expts_tot, bitmatrix):
    return SortTokens.apply(expt_scal, expt_indx, n_expts_tot, bitmatrix)


# --------------------------
# expt_data
# --------------------------


def exact_div(x, y):
    assert x % y == 0
    return x // y


def empty_aligned(shape, dtype, device, pad_size):
    cdiv = lambda x, y: (x + y - 1) // y
    pad = lambda x: cdiv(x, pad_size) * pad_size
    ret = torch.empty((*shape[:-1], pad(shape[-1])), dtype=dtype, device=device)
    ret_slices = (*[slice(None)] * (len(shape) - 1), slice(0, shape[-1]))
    return ret[ret_slices]


def max_n_tiles(n_expts_tot, n_gates):
    if n_gates <= n_expts_tot:
        return n_gates
    return n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // ExptData.block_ms()[0])


def compute_expt_data(expt_hist, n_expts_tot, n_gates):

    if expt_hist is None:
        return ExptData(None, None, None, None)

    block_ms_log2 = ExptData.block_ms_log2()
    block_m_num = len(block_ms_log2)
    MEMSET_BLOCK = 512
    dtype = torch.int32
    device = expt_hist.device
    token_offs_combined = empty_aligned((block_m_num + 1, n_expts_tot + 1), dtype, device, MEMSET_BLOCK)
    block_pid_map = empty_aligned((block_m_num, max_n_tiles(n_expts_tot, n_gates)), dtype, device, MEMSET_BLOCK)
    token_offs_raw, token_offs_pad = token_offs_combined[0], token_offs_combined[1:]
    n_memset_blocks = exact_div(block_pid_map.storage().size(), MEMSET_BLOCK)

    _expt_data_memset[(token_offs_combined.shape[0] + n_memset_blocks, )](
        expt_hist, n_expts_tot,  #
        token_offs_combined, token_offs_combined.stride(0),  #
        block_pid_map,  #
        block_ms_log2[0], SIZES=len(block_ms_log2), BLOCK=MEMSET_BLOCK,  # optimization parameters
        num_warps=4)

    _expt_data_compute[(block_m_num * n_expts_tot, )](
        expt_hist, token_offs_pad, token_offs_pad.stride(0), block_pid_map, block_pid_map.stride(0),  # outputs
        block_ms_log2[0], SIZES=len(block_ms_log2), BLOCK=512,  # optimization parameters
        num_warps=4)

    return ExptData(expt_hist, token_offs_raw, token_offs_pad, block_pid_map)


# --------------------------
# routing
# --------------------------


def routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act):
    hist, combine_indx, dispatch_indx, gate_scal, token_offs_raw, token_offs_pad, block_pid_map = sort_tokens(
        expt_scal, expt_indx, n_expts_tot, bitmatrix)
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)
    gather_indx = GatherIndx(src_indx=combine_indx, dst_indx=dispatch_indx)
    scatter_indx = ScatterIndx(src_indx=dispatch_indx, dst_indx=combine_indx)
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), gather_indx, scatter_indx


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
    token_offs_raw = torch.cumsum(hist, dim=0)
    token_offs_raw = torch.cat((torch.zeros(1, device=device), token_offs_raw))
    token_offs_raw = token_offs_raw.int()
    # maximum number of tiles for all values of `block_m` considered
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        # ceil_div(n_gates - n_experts + 1, d_tile) + n_experts - 1
        # ceil_div(x, y): -(-x // y)
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // min(ExptData.block_ms()))
    # fill up tile offset/infos for each block
    token_offs_pad = dict()
    block_pid_map = dict()
    for block_m in ExptData.block_ms():
        n_tiles = (hist + block_m - 1) // block_m  # matmul blocks needed
        token_offs_pad[block_m] = torch.cumsum(n_tiles, dim=0)
        token_offs_pad[block_m] = torch.cat((torch.zeros(1, device=device), token_offs_pad[block_m]))
        token_offs_pad[block_m] = token_offs_pad[block_m].int()
        # compute data required to drive ragged batch matmul
        block_pid_map[block_m] = -torch.ones(max_n_tiles, dtype=torch.int32, device=device)
        col = torch.arange(max_n_tiles, device=device)
        map_vals = torch.arange(n_expts_tot, device=device)[:, None] + (col << 16)[None, :]
        map_idxs = token_offs_pad[block_m][:-1, None] + col[None, :]
        mask = col[None, :] < n_tiles[:, None]
        block_pid_map[block_m].index_put_((map_idxs[mask], ), map_vals.int()[mask])
    token_offs_pad = torch.stack(list(token_offs_pad.values()))
    block_pid_map = torch.stack(list(block_pid_map.values()))
    return ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)


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
    combine_indx = torch.argsort(expt_indx, stable=True)
    dispatch_indx = torch.argsort(combine_indx, stable=True)
    gate_scal = expt_scal[combine_indx]
    hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1).int()  # histogram of tokens over experts
    # pack the matmul data structure
    gather_indx = GatherIndx(src_indx=combine_indx.int(), dst_indx=dispatch_indx.int())
    scatter_indx = ScatterIndx(src_indx=dispatch_indx.int(), dst_indx=combine_indx.int())
    # compute expt_data
    expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates_pad)
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), gather_indx, scatter_indx
