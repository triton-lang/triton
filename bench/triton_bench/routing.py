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


# --------------------------
# Triton routing
# --------------------------


@triton.jit
def vertical_popcount(x):
    """
    Input  x : uint32[..., N]
    Output y : uint32[..., 32]
    semantics : y[..., i] = sum_j((x[..., j] >> i) & 1)
    credits: @apgoucher
    """

    tl.static_assert(x.dtype == tl.uint32, "x should consist of 32-bit unsigned integers")

    BLOCK_N: tl.constexpr = x.shape[-1]  # summation axis
    BATCHES: tl.constexpr = x.numel // BLOCK_N  # number of batches
    if BLOCK_N >= 8:
        sa1: tl.constexpr = 8
    else:
        sa1: tl.constexpr = BLOCK_N
    # create 8-way sums in 4-bit fields:
    y = tl.reshape(x, [BATCHES, BLOCK_N // sa1, sa1, 1])
    y = (y >> tl.arange(0, 4)[None, None, None, :]) & 0x11111111
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // sa1, 4]
    if BLOCK_N >= 128:
        sa2: tl.constexpr = 16
    else:
        sa2: tl.constexpr = BLOCK_N // sa1
    # create 128-way sums in 8-bit fields:
    y = tl.reshape(y, [BATCHES, BLOCK_N // (sa1 * sa2), sa2, 1, 4])
    y = (y >> (4 * tl.arange(0, 2))[None, None, None, :, None]) & 0x0f0f0f0f
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // (sa1 * sa2), 2, 4]
    sa3: tl.constexpr = BLOCK_N // (sa1 * sa2)
    # create N-way sums in 32-bit fields:
    y = tl.reshape(y, [BATCHES, 1, sa3, 8])
    y = (y >> (8 * tl.arange(0, 4))[None, :, None, None]) & 0x000000ff
    y = tl.sum(y, 2)  # [BATCHES, 4, 8]
    y = tl.reshape(y, x.shape[:-1] + [32])
    return y


@triton.jit
def keyed_add(x, y):

    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xffff0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def count_previous(x):
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
    expts_and_inclusive_run_lengths = tl.associative_scan(x, 1, keyed_add)
    exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xffff

    # undo permutation by doing another sort
    # TODO rewrite this when tl.scatter becomes available
    kv_pairs = ((sorted_kv_pairs << 16) | exclusive_run_lengths).to(tl.uint32)
    unsorted_run_lengths = tl.sort(kv_pairs) & 0xffff

    res = tl.reshape(unsorted_run_lengths, x.shape)
    return res


@triton.jit
def or_combine(x, y):
    return x | y


@triton.jit
def _compute_bitmatrix(X, stride_xm,  # logits
                       Yv, Yi, stride_ym,  # topk values/indices
                       R, stride_rm, n_rows,  # routing bitmatrix
                       n_expts_tot, BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr):
    tl.static_assert(N_EXPTS_PAD % 32 == 0)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_x_n = tl.arange(0, N_EXPTS_PAD)
    mask_m = offs_m[:, None] < n_rows
    mask_n = offs_x_n[None, :] < n_expts_tot
    mask = mask_m & mask_n
    # load
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(X_ptrs, mask=mask, other=float("-inf"))
    x = (x.to(tl.uint16, bitcast=True).to(tl.int32) << 16) | offs_x_n[None, :]
    # top-k experts
    y = tl.topk(x, N_EXPTS_ACT, dim=1)
    # TODO: maybe not necessary ?
    # sort result in direction of ascending expert index
    x_sgns = (y >> 16) & 0x00008000
    y = (y << 16) | ((y >> 16) ^ x_sgns)
    y = tl.sort(y, dim=1)
    y_indices = y >> 16
    y_values = ((y & 0x0000FFFF) ^ x_sgns).to(tl.uint16).to(tl.float16, bitcast=True)
    # write back
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)
    tl.store(Yi_ptrs, y_indices, mask=mask_m)
    # pack into bitmatrix
    y_div = y_indices // 32
    y_rem = y_indices % 32
    y = tl.where(y_div[:, :, None] == tl.arange(0, N_EXPTS_PAD // 32)[None, :, :], (1 << y_rem)[:, :, None], 0)
    r = tl.reduce(y, combine_fn=or_combine, axis=1)
    offs_r_n = tl.arange(0, N_EXPTS_PAD // 32)
    R_ptrs = R + offs_m[:, None] * stride_rm + offs_r_n[None, :]
    tl.store(R_ptrs, r, mask=mask_m)


@triton.jit
def _memset_hist(Hist, hist_size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(Hist + offs, 0, mask=offs < hist_size)


@triton.jit
def _compute_hist(R, shape_rm, stride_rm,  # routing bitmatrix
                  Hist, TokensStart, PartialHist, stride_pm, shape_pn,  # histogram
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, N_EXPTS_PAD: tl.constexpr):
    tl.static_assert(BLOCK_N % 32 == 0)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    BLOCK_B: tl.constexpr = BLOCK_N // 32
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = pid_n * BLOCK_B + tl.arange(0, BLOCK_B)
    r = tl.load(R + offs_m[None, :] * stride_rm + offs_b[:, None], mask=offs_m[None, :] < shape_rm)
    hist = tl.reshape(vertical_popcount(r), [BLOCK_N])
    mask = offs_n < shape_pn
    tl.atomic_add(Hist + offs_n, hist, mask=mask)
    tl.store(PartialHist + pid_m * stride_pm + offs_n, hist, mask=mask)
    # update atomic block counter (reuse tokens offset memory)
    tl.debug_barrier()
    if tl.atomic_add(TokensStart, 1) != tl.num_programs(0) * tl.num_programs(1) - 1:
        return
    tl.debug_barrier()
    # we are the only block left and all atomics are visible; compute cumsum
    offs_n = tl.arange(0, N_EXPTS_PAD)
    hist = tl.load(Hist + offs_n)
    tok_starts = tl.cumsum(hist, 0)
    tl.store(TokensStart, 0)
    tl.store(TokensStart + 1 + offs_n, tok_starts, mask=offs_n < shape_pn)


@triton.jit
def _finalize_hist(TokensStart, PartialHist, PartialOffs, shape_pm, stride_pm, BLOCK_M: tl.constexpr):
    expt_id = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    # initialize first row of the output
    tokens_off = tl.load(TokensStart + expt_id)
    tl.store(PartialOffs + expt_id, tokens_off)
    # iterate over input data
    curr_sum = tokens_off
    for _ in range(0, shape_pm, BLOCK_M):
        offs = offs_m * stride_pm + expt_id
        curr = tl.load(PartialHist + offs, mask=offs_m < shape_pm)
        out = tl.cumsum(curr, 0) + curr_sum
        curr_sum += tl.sum(curr, 0)
        offs = (1 + offs_m) * stride_pm + expt_id
        tl.store(PartialOffs + offs, out, mask=offs_m < shape_pm - 1)
        offs_m += BLOCK_M


@triton.jit
def _compute_indx(GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx, PartialOffs, stride_pm, n_gates,
                  BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr):
    pid_m = tl.program_id(0)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + tl.arange(0, N_EXPTS_ACT * BLOCK_M)
    mask = offs < n_gates
    indx = tl.load(ExptIndx + offs, mask=mask)
    gates = tl.load(PartialOffs + pid_m * stride_pm + indx, mask=mask)
    gates += tl.reshape(count_previous(indx), [BLOCK_M * N_EXPTS_ACT])
    gate_scal = tl.load(ExptScal + offs, mask=mask)
    tl.store(ScatterIndx + offs, gates, mask=mask)
    tl.store(GatherIndx + gates, offs, mask=mask)
    tl.store(GateScal + gates, gate_scal, mask=mask)


def routing(logits, n_expts_act, expt_indx=None):
    assert expt_indx is None
    cdiv = triton.cdiv
    ROUTING_BLOCK_M = 8
    HIST1_BLOCK_M = 64
    HIST1_BLOCK_N = 32
    HIST2_BLOCK_M = 512
    MEMSET_BLOCK = 512
    assert logits.dtype.itemsize == 2
    n_tokens, n_expts_tot = logits.shape
    n_gates = n_tokens * n_expts_act
    dev = logits.device
    n_expts_pad = cdiv(n_expts_tot, 128) * 128
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
    _compute_bitmatrix[(cdiv(n_tokens, ROUTING_BLOCK_M), )](
        logits,
        logits.stride(0),
        expt_scal,
        expt_indx,
        expt_scal.stride(0),
        bitmatrix,
        bitmatrix.stride(0),
        n_tokens,
        n_expts_tot,
        BLOCK_M=ROUTING_BLOCK_M,
        N_EXPTS_PAD=n_expts_pad,
        N_EXPTS_ACT=n_expts_act,
    )
    _memset_hist[(cdiv(hist.shape[0], MEMSET_BLOCK), )](hist, hist.shape[0], BLOCK=MEMSET_BLOCK)
    _compute_hist[(cdiv(n_tokens, HIST1_BLOCK_M), cdiv(n_expts_tot, HIST1_BLOCK_N))](
        bitmatrix,
        bitmatrix.shape[0],
        bitmatrix.stride(0),
        hist,
        tok_starts,
        partial_hist,
        partial_hist.stride(0),
        partial_hist.shape[1],
        BLOCK_M=HIST1_BLOCK_M,
        BLOCK_N=HIST1_BLOCK_N,
        N_EXPTS_PAD=n_expts_pad,
    )
    _finalize_hist[(n_expts_tot, )](
        tok_starts,
        partial_hist,
        partial_offs,
        partial_hist.shape[0],
        partial_hist.stride(0),
        BLOCK_M=HIST2_BLOCK_M,
    )
    _compute_indx[(cdiv(n_tokens, HIST1_BLOCK_M), )](
        topk_indx,
        gate_indx,
        gate_scal,
        expt_scal,
        expt_indx,
        partial_offs,
        partial_offs.stride(0),
        n_gates,
        BLOCK_M=HIST1_BLOCK_M,
        N_EXPTS_ACT=n_expts_act,
    )
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
