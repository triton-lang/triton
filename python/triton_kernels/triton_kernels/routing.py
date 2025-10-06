import torch
import triton
from dataclasses import dataclass, field
from .routing_details._routing_compute import _combined_routing_compute
from .routing_details._routing_compute import _combined_routing_memset
from .routing_details._expt_data import _expt_data_memset
from .routing_details._expt_data import _expt_data_compute
from .target_info import is_hip
from .topk import topk, topk_torch
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
    # hist[i] is the number of tokens routed to expert i
    hist: torch.Tensor
    # token_offs_raw[i] is the offset of the first token routed
    # to expert i in an expert-sorted array
    token_offs_raw: torch.Tensor
    # token_offs_pad_data[:, i] is the offset of the first token routed
    # to expert i in an expert-sorted array, assuming histogram
    # rounded to the next multiple of `block = 16 * i`
    token_offs_pad_data: torch.Tensor
    # block_id_map_data[i] contain one value for each `pid` launched by
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
        block_pid_map = empty_aligned((block_m_num, get_max_n_tiles(n_expts_tot, n_gates_pad)), torch.int32, device,
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


def get_max_n_tiles(n_expts_tot, n_gates):
    if n_gates <= n_expts_tot:
        return n_gates
    # ceil_div(n_gates - n_experts + 1, d_tile) + n_experts - 1
    # ceil_div(x, y): -(-x // y)
    return n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // ExptData.block_ms()[0])


def compute_expt_data(expt_hist, n_expts_tot, n_gates):

    if expt_hist is None:
        return ExptData(None, None, None, None)

    block_ms_log2 = ExptData.block_ms_log2()
    block_m_num = len(block_ms_log2)
    MEMSET_BLOCK = 512
    dtype = torch.int32
    device = expt_hist.device
    token_offs_combined = empty_aligned((block_m_num + 1, n_expts_tot + 1), dtype=dtype, device=device,
                                        align=MEMSET_BLOCK)
    block_pid_map = empty_aligned((block_m_num, get_max_n_tiles(n_expts_tot, n_gates)), dtype=dtype, device=device,
                                  align=MEMSET_BLOCK)
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


# ------------------------------------------------------------


@triton.jit
def _compaction(Out, compute_vals_and_cond_fn, compute_vals_and_cond_fn_args, sentinel, N, BLOCK: tl.constexpr):
    curr_sum = 0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        vals, conds = compute_vals_and_cond_fn(*compute_vals_and_cond_fn_args, offs)
        # compute values
        exc_cumsum = curr_sum + tl.cumsum(conds, 0) - conds
        active_flags = conds.to(tl.int1)
        rev_arange = N - start - 1 - tl.arange(0, BLOCK)
        write_indx = exc_cumsum + tl.where(active_flags, 0, rev_arange)
        out = tl.where(active_flags, vals, sentinel)
        # store
        tl.store(Out + write_indx, out, mask=offs < N)
        # update running sum
        curr_sum += tl.sum(conds, 0)
    return curr_sum


@triton.jit
def _compact_expt_id(vals_ptr, expt_filter_ptr, n_expts_tot, offs):
    expt_ids = offs
    div = expt_ids // 32
    rem = expt_ids % 32
    mask = expt_ids < n_expts_tot
    conds = (tl.load(expt_filter_ptr + div, mask=mask, other=0) >> rem) & 1
    vals = tl.load(vals_ptr + offs, mask=mask)
    return vals, conds


@triton.jit
def _compact_block_id_map(block_pid_map, expt_map_ptr, expt_filter_ptr, n_blocks, offs):
    block_id = tl.load(block_pid_map + offs, mask=offs < n_blocks, other=-1)
    block_id = block_id.to(tl.uint32, bitcast=True)
    expt_id = block_id & 0x0000FFFF
    div = expt_id // 32
    rem = expt_id % 32
    mask = expt_id != 65535
    conds = (tl.load(expt_filter_ptr + div, mask=mask, other=0) >> rem) & 1
    block_id = block_id.to(tl.int32, bitcast=True)
    conds = conds.to(tl.int32, bitcast=True)
    new_expt_id = tl.load(expt_map_ptr + expt_id, mask=mask)
    pid_mask = tl.full([
        1,
    ], 0xFFFF0000, dtype=tl.uint32)
    new_block_id = ((block_id & pid_mask) | new_expt_id).to(tl.int32, bitcast=True)
    return new_block_id, conds


@triton.jit
def _filter_expt_data(expt_hist_out, expt_hist_inp, token_offs_raw_out, token_offs_raw_inp, token_offs_pad_out,
                      token_offs_pad_out_stride_m, token_offs_pad_inp, token_offs_pad_inp_stride_m, block_pid_map_out,
                      block_pid_map_out_stride_m, block_pid_map_inp, block_pid_map_inp_stride_m, expt_filter, expt_map,
                      n_expts_tot, n_blocks, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    token_offs_pad_out += pid_m * token_offs_pad_out_stride_m
    token_offs_pad_inp += pid_m * token_offs_pad_inp_stride_m
    block_pid_map_out += pid_m * block_pid_map_out_stride_m
    block_pid_map_inp += pid_m * block_pid_map_inp_stride_m
    _compaction(expt_hist_out, _compact_expt_id, (expt_hist_inp, expt_filter, n_expts_tot), -1, n_expts_tot,
                BLOCK=BLOCK)
    _compaction(token_offs_raw_out, _compact_expt_id, (token_offs_raw_inp, expt_filter, n_expts_tot), -1,
                n_expts_tot + 1, BLOCK=BLOCK)
    compacted_tile_count = _compaction(token_offs_pad_out, _compact_expt_id,
                                       (token_offs_pad_inp, expt_filter, n_expts_tot), -1, n_expts_tot + 1, BLOCK=BLOCK)
    compacted_block_count = _compaction(block_pid_map_out, _compact_block_id_map,
                                        (block_pid_map_inp, expt_map, expt_filter, n_blocks), -1, n_blocks, BLOCK=BLOCK)
    # Record the total number of tiles in the trailing slot
    tl.store(token_offs_pad_out + compacted_tile_count, compacted_block_count)


def filter_expt_data(expt_data, expt_assignment, rank):
    expt_hist = torch.empty_like(expt_data.hist)
    token_offs_raw = torch.empty_like(expt_data.token_offs_raw)
    token_offs_pad_data = torch.empty_like(expt_data.token_offs_pad_data)
    block_pid_map_data = torch.empty_like(expt_data.block_pid_map_data)

    _filter_expt_data[(token_offs_pad_data.shape[0], )](
        expt_hist,
        expt_data.hist,
        token_offs_raw,
        expt_data.token_offs_raw,
        token_offs_pad_data,
        token_offs_pad_data.stride(0),
        expt_data.token_offs_pad_data,
        expt_data.token_offs_pad_data.stride(0),
        block_pid_map_data,
        block_pid_map_data.stride(0),
        expt_data.block_pid_map_data,
        expt_data.block_pid_map_data.stride(0),
        expt_assignment.expt_bitmask[rank, :],
        expt_assignment.expt_map[rank, :],
        len(expt_hist),
        block_pid_map_data.shape[-1],
        BLOCK=128,
    )
    return ExptData(expt_hist, token_offs_raw, token_offs_pad_data, block_pid_map_data)


def filter_expt_data_torch(expt_data, expt_assignment, rank):

    expt_bitmask = expt_assignment.expt_bitmask[rank, :]
    expt_map = expt_assignment.expt_map[rank, :]

    def compact(vals, conds, sentinel):
        assert conds.shape == vals.shape
        keep = conds.nonzero().flatten()
        sentinels = torch.full(((conds == 0).sum().item(), ), sentinel, dtype=vals.dtype, device=vals.device)
        return torch.cat((vals[keep], sentinels))

    def make_mask(block_pid_map):
        expt_id = (block_pid_map & 0x0000FFFF)
        valid_id = expt_id != 65535
        valid_expt_id = expt_id[valid_id]
        mask = torch.zeros_like(expt_id)
        mask[valid_id] = (expt_bitmask[valid_expt_id // 32] >> (valid_expt_id % 32)) & 1
        return mask

    def map_expt_id(block_pid_map):
        expt_id = (block_pid_map & 0x0000FFFF)
        valid_id = expt_id != 65535
        expt_id[valid_id] = expt_map[expt_id[valid_id]]
        return (block_pid_map & 0xFFFF0000) | expt_id

    n_expts_tot = len(expt_data.hist)
    expt_global = torch.arange(n_expts_tot, device=expt_data.hist.device)
    expt_local = (expt_bitmask[expt_global // 32] >> (expt_global % 32)) & 1
    expt_mask = torch.cat((expt_local, torch.zeros((1, ), dtype=torch.bool, device=expt_local.device)))
    expt_hist = compact(expt_data.hist, expt_mask[:-1], -1)
    token_offs_raw = compact(expt_data.token_offs_raw, expt_mask, -1)

    token_offs_pad_data = torch.stack([compact(v, expt_mask, -1) for v in expt_data.token_offs_pad_data], dim=0)
    block_pid_map_data = [compact(v, make_mask(v), -1) for v in expt_data.block_pid_map_data]
    block_pid_map_data = torch.stack([map_expt_id(v) for v in block_pid_map_data], dim=0)
    return ExptData(expt_hist, token_offs_raw, token_offs_pad_data, block_pid_map_data)


# --------------------------
# expt_assignment
# --------------------------


@dataclass
class ExptAssignment:
    # torch.Tensor[n_expt_shard, n_expt_tot // 32]
    # (expt_bitmask[i, j//32] >> j%32) & 1 == 1 iff expert j is owned by shard i
    expt_bitmask: torch.Tensor
    # torch.Tensor[n_expt_shard, n_expt_tot]
    # expt_boolmask[i, j] == True iff expert j is owned by shard i
    expt_boolmask: torch.Tensor
    # torch.Tensor[n_expt_shard, n_expt_tot]
    # expt_map[i, j] is the local expert id of expert j in shard i,
    # or -1 if expert j is not owned by shard i
    expt_map: torch.Tensor
    # number of experts per shard
    n_expts_per_shard: list[int]


def make_expt_assignment(n_expt_shard, n_expt_tot, expt_dict: dict[int, list[int]], device) -> ExptAssignment:
    """
    n_expt_shard: int
    n_expt_tot: int
    expt_dict: dict[int, list[int]]
      expt_dict[i] is the list of expert ids owned by shard i
    """
    # make expt_bitmask
    words = (n_expt_tot + 31) // 32  # safe even if n_expt_tot not multiple of 32
    expt_bitmask = torch.zeros((n_expt_shard, words), dtype=torch.int32)
    expt_boolmask = torch.zeros((n_expt_shard, n_expt_tot), dtype=torch.bool)
    counts = {expt_id: 0 for expt_id in range(n_expt_tot)}
    for shard, experts in expt_dict.items():
        if len(experts) == 0:
            raise ValueError(f"shard {shard} has no experts")
        if shard < 0 or shard >= n_expt_shard:
            raise ValueError(f"shard {shard} out of range [0, {n_expt_shard})")
        if not isinstance(experts, (list, tuple)):
            raise TypeError(f"expt_dict[{shard}] must be a list/tuple of ints")
        for e in experts:
            counts[e] += 1
            if not (0 <= e < n_expt_tot):
                raise ValueError(f"expert id {e} out of range [0, {n_expt_tot})")
            word = e >> 5  # e // 32
            bit = e & 31  # e % 32
            expt_bitmask[shard, word] |= (1 << bit)
            expt_boolmask[shard, e] = True
    if not all(counts[e] == 1 for e in range(n_expt_tot)):
        raise ValueError("each expert must be owned by exactly one shard")
    expt_bitmask = expt_bitmask.to(device)
    expt_boolmask = expt_boolmask.to(device)
    # make expt_map
    expt_map = torch.full((n_expt_shard, n_expt_tot), -1, dtype=torch.int32)
    for shard, experts in expt_dict.items():
        for local_id, global_id in enumerate(sorted(experts)):
            expt_map[shard, global_id] = local_id
    expt_map = expt_map.to(device)
    # number of experts per shard
    n_expts_per_shard = [len(experts) for experts in expt_dict.values()]
    return ExptAssignment(expt_bitmask, expt_boolmask, expt_map, n_expts_per_shard)


# --------------------------
# routing
# --------------------------


def routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act):
    hist, combine_indx, dispatch_indx, gate_scal, token_offs_raw, token_offs_pad, block_pid_map = sort_tokens(
        expt_scal, expt_indx, n_expts_tot, bitmatrix)
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)
    gather_indx = GatherIndx(src_indx=combine_indx, dst_indx=dispatch_indx)
    scatter_indx = ScatterIndx(src_indx=dispatch_indx, dst_indx=combine_indx)
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), gather_indx, scatter_indx, expt_indx


def routing(logits, n_expts_act, sm_first=False, expt_indx=None, all_gather=False, n_rows=None):
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    expt_scal, expt_indx, bitmatrix = topk(logits, n_expts_act, all_gather=all_gather,  #
                                           apply_softmax=not sm_first, y_indx=expt_indx, n_rows=n_rows)
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
    max_n_tiles = get_max_n_tiles(n_expts_tot, n_gates)
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
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), gather_indx, scatter_indx, expt_indx
