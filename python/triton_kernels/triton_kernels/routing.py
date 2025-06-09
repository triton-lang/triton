import torch
import triton
from dataclasses import dataclass, field
from .routing_details._routing_compute import _routing_memset_indx
from .routing_details._routing_compute import _routing_compute_indx_offs
from .routing_details._routing_compute import _routing_compute_indx
from .routing_details._routing_compute import _routing_clear_bitmatrix
from .routing_details._expt_data import _expt_data_memset
from .routing_details._expt_data import _expt_data_compute


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
    # token_offs_pad[block][i] is the offset of the first token routed
    # to expert i in an expert-sorted array, assuming histogram
    # rounded to the next multiple of `block`
    token_offs_pad: dict[int, torch.Tensor]
    # block_id_map[block] contain one value for each `pid`` launched by
    # the matrix multiplication kernel launched with BLOCK_M=block:
    # - the value is -1 if the `pid` has no work to do
    # - otherwise, the value is two int16 (packed as an int32) that
    #   correspond respectively to (1) the expert assigned to
    #   the tokens processed by this pid; (2) the block assigned to the
    #   tokens processed by this pid (think `pid_m` in a regular matmul)
    # see `test_routing.py` for a reference implementation and more details
    block_pid_map: dict[int, torch.Tensor]

    def __post_init__(self):
        if self.hist is not None:
            assert self.hist.dtype == torch.int32
        if self.token_offs_raw is not None:
            assert self.token_offs_raw.dtype == torch.int32
        if self.token_offs_pad is not None:
            for v in self.token_offs_pad.values():
                assert v.dtype == torch.int32
        if self.block_pid_map is not None:
            for v in self.block_pid_map.values():
                assert v.dtype == torch.int32


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
    def forward(ctx, expt_scal, expt_indx, bitmatrix):
        HIST_BLOCK_M = 64
        INDX_OFFS_BLOCK_M = 512
        MEMSET_BLOCK = 1024
        cdiv = triton.cdiv

        device = expt_scal.device
        dtype = expt_scal.dtype
        n_tokens_raw, n_expts_tot = bitmatrix.shape_raw
        n_tokens_pad, n_expts_act = expt_scal.shape
        n_gates_pad = n_tokens_pad * n_expts_act

        hist, partial_hist = bitmatrix.sum(partials_block_size=HIST_BLOCK_M)
        assert hist.dtype == torch.int32
        # scratchpad
        expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
        combined_indx = torch.empty(n_gates_pad * 2, dtype=torch.int32, device=device)
        # output
        topk_indx = combined_indx[:n_gates_pad]
        gate_indx = combined_indx[n_gates_pad:]
        gate_scal = torch.empty(n_gates_pad, dtype=dtype, device=device)
        _routing_memset_indx[(cdiv(n_gates_pad * 2, MEMSET_BLOCK) + 1, )](
            combined_indx, n_gates_pad * 2, -1, MEMSET_BLOCK, hist,  #
            expt_offs, hist.shape[0], BLOCK_N=512  #
        )
        _routing_compute_indx_offs[(n_expts_tot, )](
            expt_offs, partial_hist,  # inputs
            partial_hist.shape[0], partial_hist.stride(0), partial_hist.stride(1),  # outputs
            BLOCK_M=INDX_OFFS_BLOCK_M,  # tunable parameters
        )
        indx_offs = partial_hist
        _routing_compute_indx[(cdiv(n_tokens_pad, HIST_BLOCK_M), )](
            topk_indx, gate_indx, gate_scal,  # outputs
            expt_scal, expt_indx, indx_offs, indx_offs.stride(0), indx_offs.stride(1),  # inputs
            n_tokens_pad, n_tokens_raw,  # input shape
            BLOCK_M=HIST_BLOCK_M,  # tunable parameters
            N_EXPTS_ACT=n_expts_act,  # constants
            num_warps=1 if HIST_BLOCK_M * n_expts_act // 32 < 4 else 4  #
        )
        ctx.n_tokens_raw = n_tokens_raw
        ctx.n_tokens_pad = n_tokens_pad
        ctx.n_expts_act = n_expts_act
        ctx.save_for_backward(gate_indx)
        return hist, topk_indx, gate_indx, gate_scal

    @staticmethod
    def backward(ctx, _0, _1, _2, dgate_scal):
        (gate_indx, ) = ctx.saved_tensors
        dgate_scal = dgate_scal[gate_indx]
        dgate_scal = dgate_scal.reshape(ctx.n_tokens_pad, ctx.n_expts_act)
        return dgate_scal, None, None


def sort_tokens(expt_scal, expt_indx, bitmatrix):
    return SortTokens.apply(expt_scal, expt_indx, bitmatrix)


# --------------------------
# prune routing
# --------------------------


class PruneRouting(torch.autograd.Function):

    @staticmethod
    def forward(ctx, expt_scal, expt_indx, bitmatrix, simulated_ep):
        from .compaction import compaction
        n_tokens_pad = expt_scal.shape[0]
        n_expts_tot = bitmatrix.shape_raw[-1]
        assert n_expts_tot % simulated_ep == 0
        _routing_clear_bitmatrix[(n_tokens_pad, )](
            bitmatrix.handle,
            bitmatrix.handle.stride(0),
            bitmatrix.handle.stride(1),
            bitmatrix.handle.shape[1],
            n_expts_tot // simulated_ep,
            BLOCK_N=512,
        )
        # perform compaction to update expt_scal / expt_indx
        expt_scal, expt_indx = compaction(expt_scal, expt_indx, bitmatrix)
        n_expts_tot = n_expts_tot // simulated_ep
        bitmatrix.shape_raw[-1] = n_expts_tot
        return expt_scal, expt_indx, bitmatrix


def prune_routing(expt_scal, expt_indx, bitmatrix, simulated_ep):
    return PruneRouting.apply(expt_scal, expt_indx, bitmatrix, simulated_ep)


# --------------------------
# expt_data
# --------------------------


def log2_power_of_two(x):
    assert x > 0 and (x & (x - 1)) == 0, "x must be a power of two"
    return x.bit_length() - 1


def compute_expt_data(expt_hist, n_expts_tot, n_gates):
    if expt_hist is None:
        return ExptData(None, None, None, None)
    MEMSET_BLOCK = 128
    HIST2_BLOCK_M = 512
    device = expt_hist.device
    n_expts_tot = n_expts_tot
    cdiv = triton.cdiv
    # block_ms are all powers-of-two between 16 and 128 (inclusive)
    block_m_log2_start = 4
    block_m_log2_end = 8
    block_m_num = block_m_log2_end - block_m_log2_start
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // 2**block_m_log2_start)
    # allocate memory
    pad = lambda x: cdiv(x, MEMSET_BLOCK) * MEMSET_BLOCK
    dtype = torch.int32
    token_offs_raw = torch.empty((n_expts_tot + 1, ), dtype=dtype, device=device)
    token_offs_pad = torch.empty((block_m_num, pad(n_expts_tot + 1)), dtype=dtype, device=device)
    block_pid_map = torch.empty((block_m_num, pad(max_n_tiles)), dtype=dtype, device=device)
    # compute outputs
    token_offs_pad = token_offs_pad[:, :n_expts_tot + 1]
    block_pid_map = block_pid_map[:, :max_n_tiles]
    memset_grid = cdiv(block_pid_map.shape[1], MEMSET_BLOCK) + 1
    _expt_data_memset[(memset_grid, block_m_num)](
        expt_hist, n_expts_tot, token_offs_raw,  #
        token_offs_pad, token_offs_pad.stride(0),  #
        block_pid_map, block_pid_map.stride(0),  #
        block_m_log2_start, BLOCK=MEMSET_BLOCK,  # optimization parameters
        num_warps=1)
    _expt_data_compute[(n_expts_tot, block_m_num)](
        expt_hist, token_offs_pad, token_offs_pad.stride(0), block_pid_map, block_pid_map.stride(0),  # outputs
        block_m_log2_start, BLOCK=HIST2_BLOCK_M,  # optimization parameters
        num_warps=4)
    # unpack into datastructure
    token_offs_pad = {2**j: token_offs_pad[i, :] for i, j in enumerate(range(block_m_log2_start, block_m_log2_end))}
    block_pid_map = {2**j: block_pid_map[i, :] for i, j in enumerate(range(block_m_log2_start, block_m_log2_end))}
    return ExptData(expt_hist, token_offs_raw, token_offs_pad, block_pid_map)


# --------------------------
# routing
# --------------------------


def routing(logits, n_expts_act, sm_first=False, expt_indx=None, simulated_ep=1, n_rows=None):
    from .topk import topk
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    expt_scal, expt_indx, bitmatrix = topk(logits, n_expts_act,  #
                                           apply_softmax=not sm_first, y_indx=expt_indx, n_rows=n_rows)
    # mutate bitmatrix
    if simulated_ep > 1:
        expt_scal, expt_indx, bitmatrix = prune_routing(expt_scal, expt_indx, bitmatrix, simulated_ep)
    hist, topk_indx, gate_indx, gate_scal = sort_tokens(expt_scal, expt_indx, bitmatrix)
    # pack the matmul data structure
    n_expts_tot = logits.shape[-1] // simulated_ep
    gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
    expt_data = compute_expt_data(hist, n_expts_tot, topk_indx.numel())
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), gather_indx, scatter_indx


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
    block_ms = [16, 32, 64, 128]
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        # ceil_div(n_gates - n_experts + 1, d_tile) + n_experts - 1
        # ceil_div(x, y): -(-x // y)
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // min(block_ms))
    # fill up tile offset/infos for each block
    token_offs_pad = dict()
    block_pid_map = dict()
    for block_m in [16, 32, 64, 128]:
        n_tiles = (hist + block_m - 1) // block_m  # matmul blocks needed
        token_offs_pad[block_m] = torch.cumsum(n_tiles, dim=0)
        token_offs_pad[block_m] = torch.cat((torch.zeros(1, device=device), token_offs_pad[block_m]))
        token_offs_pad[block_m] = token_offs_pad[block_m].int()
        # compute data required to drive ragged batch matmul
        block_pid_map[block_m] = -torch.ones(max_n_tiles, device=device)
        for e in range(n_expts_tot):
            offset = token_offs_pad[block_m][e]
            for b in range(n_tiles[e]):
                block_pid_map[block_m][offset + b] = (b << 16) + e
        block_pid_map[block_m] = block_pid_map[block_m].int()
    return ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)


def routing_torch(logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None):
    has_user_provided_indx = expt_indx is not None
    n_gates_pad = logits.shape[0] * n_expts_act

    if n_rows is not None:
        logits = logits[:n_rows, :]

    def topk(vals, k, expt_indx):
        # topk of experts
        if has_user_provided_indx:
            tk_indx = expt_indx
        else:
            tk_indx = torch.argsort(-vals, dim=1, stable=True)[:, :k]
        tk_indx = tk_indx.long()
        tk_val = torch.take_along_dim(vals, tk_indx, dim=1)
        tk_indx = tk_indx.int()
        return tk_val, tk_indx

    _, n_expts_tot = logits.shape
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    expt_scal, expt_indx = topk(logits, n_expts_act, expt_indx)
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
    topk_indx = torch.argsort(expt_indx, stable=True)
    gate_indx = torch.argsort(topk_indx, stable=True)
    gate_scal = expt_scal[topk_indx]
    hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1).int()  # histogram of tokens over experts
    # pack the matmul data structure
    gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
    scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
    # compute expt_data
    expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates_pad)
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), gather_indx, scatter_indx
