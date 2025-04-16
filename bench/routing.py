import torch
import triton
import triton.language as tl

# fmt: off

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
                       R, stride_rm, n_rows, # routing bitmatrix
                       n_expts_tot,
                       BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr):
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
    y = (y << 16) | ((y  >> 16) ^ x_sgns)
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
def _memset_metadata(Metadata, shape, grid_m, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    vals = tl.where(offs < shape - grid_m, 0, 0xffffffff)
    mask = offs < shape
    tl.store(Metadata + offs, vals, mask=mask)


@triton.jit
def _compute_metadata(R, shape_rm, stride_rm,  # routing bitmatrix
                 Hist, TokensStart, TilesStart,
                 PartialHist, stride_pm, shape_pn,  # histogram
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 N_EXPTS_PAD: tl.constexpr,
                 TILE_DIM: tl.constexpr):
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
    if tl.atomic_add(TokensStart, 1) != tl.num_programs(0)*tl.num_programs(1) - 1:
        return
    tl.debug_barrier()
    # we are the only block left and all atomics are visible; compute cumsum
    offs_n = tl.arange(0, N_EXPTS_PAD)
    hist = tl.load(Hist + offs_n)
    tok_starts = tl.cumsum(hist, 0)
    tl.store(TokensStart, 0)
    tl.store(TokensStart + 1 + offs_n, tok_starts, mask=offs_n < shape_pn)
    tile_starts = tl.cumsum(tl.cdiv(hist, TILE_DIM), 0)
    tl.store(TilesStart, 0)
    tl.store(TilesStart + 1 + offs_n, tile_starts, mask=offs_n < shape_pn)


@triton.jit
def _finalize_metadata(TokensStart, FinalHist, PartialHist, PartialOffs, shape_pm, stride_pm,
                   TileOffs, TileMetadata, TILE_DIM: tl.constexpr,
                   BLOCK_M: tl.constexpr):
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
    # fill up metadata
    start_off = tl.load(TileOffs + expt_id)
    n_tokens = tl.load(FinalHist + expt_id)
    n_blocks = tl.cdiv(n_tokens, TILE_DIM)
    TileMetadata += start_off
    for block_off in range(0, n_blocks, BLOCK_M):
        block_offs = block_off + tl.arange(0, BLOCK_M)
        data = (block_offs << 16) + expt_id
        tl.store(TileMetadata + block_offs, data, mask=block_offs < n_blocks)

@triton.jit
def _compute_indx(GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx,
                  PartialOffs, stride_pm, n_gates,
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


def triton_routing(x, n_expts_act):
    cdiv = triton.cdiv
    ROUTING_BLOCK_M = 8
    HIST1_BLOCK_M = 64
    HIST1_BLOCK_N = 32
    HIST2_BLOCK_M = 512
    MEMSET_BLOCK = 512
    assert x.dtype.itemsize == 2
    n_tokens, n_expts_tot = x.shape
    n_gates = n_tokens * n_expts_act
    dev = x.device
    n_expts_pad = cdiv(n_expts_tot, 128) * 128
    n_expts_words = n_expts_pad // 32
    # scratchpad tensors
    # NOTE: these are not returned
    expt_scal = torch.empty((n_tokens, n_expts_act), dtype=x.dtype, device=dev)
    expt_indx = torch.empty((n_tokens, n_expts_act), dtype=torch.int16, device=dev)
    bitmatrix = torch.empty((n_tokens, n_expts_words), dtype=torch.uint32, device=dev)
    partial_hist = torch.empty((cdiv(n_tokens, HIST1_BLOCK_M), n_expts_tot), dtype=torch.int32, device=dev)
    partial_offs = torch.empty((cdiv(n_tokens, HIST1_BLOCK_M), n_expts_tot), dtype=torch.int32, device=dev)
    # output tensors
    # metadata
    tile_dim = 128
    if n_gates <= n_expts_tot:
        grid_m = n_gates
    else:
        grid_m = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // tile_dim)
    metadata = torch.empty(n_expts_tot * 3 + 2 + grid_m, dtype=torch.int32, device=x.device)
    # metadata views
    hist = metadata[:n_expts_tot]
    tokens_start = metadata[n_expts_tot: n_expts_tot * 2 + 1]
    tiles_start = metadata[n_expts_tot * 2 + 1: n_expts_tot * 3 + 2]
    blocks_info = metadata[n_expts_tot * 3 + 2:]
    # reordered indices
    topk_indx = torch.empty(n_gates, dtype=torch.int32, device=dev)
    gate_indx = torch.empty(n_gates, dtype=torch.int32, device=dev)
    gate_scal = torch.empty(n_gates, dtype=x.dtype, device=dev)
    # compute routing bitmatrix
    _compute_bitmatrix[(cdiv(n_tokens, ROUTING_BLOCK_M), )](
        x, x.stride(0),
        expt_scal, expt_indx, expt_scal.stride(0),
        bitmatrix, bitmatrix.stride(0),
        n_tokens, n_expts_tot,
        BLOCK_M=ROUTING_BLOCK_M,
        N_EXPTS_PAD=n_expts_pad, N_EXPTS_ACT=n_expts_act,
    )
    # compute metadata
    _memset_metadata[(cdiv(metadata.shape[0], MEMSET_BLOCK), )](
        metadata, metadata.shape[0], grid_m,
        BLOCK=MEMSET_BLOCK
    )
    _compute_metadata[(cdiv(n_tokens, HIST1_BLOCK_M), cdiv(n_expts_tot, HIST1_BLOCK_N))](
        bitmatrix, bitmatrix.shape[0], bitmatrix.stride(0),
        hist, tokens_start, tiles_start,
        partial_hist, partial_hist.stride(0), partial_hist.shape[1],
        BLOCK_M=HIST1_BLOCK_M, BLOCK_N=HIST1_BLOCK_N,
        N_EXPTS_PAD=n_expts_pad,
        TILE_DIM=tile_dim,
    )
    _finalize_metadata[(n_expts_tot, )](
        tokens_start, hist, partial_hist, partial_offs,
        partial_hist.shape[0], partial_hist.stride(0),
        tiles_start, blocks_info,
        BLOCK_M=HIST2_BLOCK_M,
        TILE_DIM=tile_dim,
    )
    # reorder indices
    _compute_indx[(cdiv(n_tokens, HIST1_BLOCK_M), )](
        topk_indx, gate_indx, gate_scal, expt_scal, expt_indx,
        partial_offs, partial_offs.stride(0), n_gates,
        BLOCK_M=HIST1_BLOCK_M,
        N_EXPTS_ACT=n_expts_act,
    )
    return topk_indx, gate_indx, gate_scal, metadata

def torch_routing(x, n_expts_act):
    n_tokens, n_expts_tot = x.shape
    expt_scal, expt_indx = torch.topk(x, k=n_expts_act)
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
    histogram = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1)  # histogram of tokens over experts
    tokoffs = torch.zeros_like(histogram)
    tokoffs[1:] = torch.cumsum(histogram[:-1], 0)
    # expert metadata
    # --------------
    n_gates = n_tokens * n_expts_act
    block_m = 128
    blks = (histogram + block_m - 1) // block_m  # matmul blocks needed
    tsum = torch.cumsum(histogram, dim=0)  # prefix sum of tokens
    bsum = torch.cumsum(blks, dim=0)  # prefix sum of blocks
    # Get the max number of matmul blocks of size d_tile needed (and is launched with).
    # This assumes the worst distribution of all experts with one token except for one that has the rest.
    if n_gates <= n_expts_tot:
        grid_m = n_gates
    else:
        grid_m = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // block_m)
    block_metadata = -torch.ones(grid_m, dtype=torch.int32)
    # compute data required to drive ragged batch matmul
    for e in range(n_expts_tot):
        offset = bsum[e - 1] if e else 0
        for b in range(blks[e]):
            block_metadata[offset + b] = (b << 16) + e
    metadata = torch.zeros(n_expts_tot * 3 + 2 + grid_m, dtype=torch.int32, device=x.device)
    metadata[:n_expts_tot] = histogram
    metadata[n_expts_tot + 1 : n_expts_tot * 2 + 1] = tsum
    metadata[n_expts_tot * 2 + 2 : n_expts_tot * 3 + 2] = bsum
    metadata[n_expts_tot * 3 + 2 :] = block_metadata
    return topk_indx, gate_indx, gate_scal, metadata


M = 32768
N_EXPTS_TOT, N_EXPTS_ACT = 128, 4
torch.manual_seed(0)
x = [(-1)**0 * ((16384 + ((_ * 512) % 4096) + torch.randperm(N_EXPTS_TOT)).to(torch.int16).view(torch.float16)) for _ in range(M)]
x = torch.stack(x).cuda()
# x = torch.rand((M, N_EXPTS_TOT), dtype=torch.float16, device="cuda")
import triton.profiler as proton

proton.start("routing")
tri_topk_indx, tri_gate_indx, tri_gate_scal, tri_metadata = triton_routing(x, N_EXPTS_ACT)
proton.finalize()
ref_topk_indx, ref_gate_indx, ref_gate_scal, ref_metadata = torch_routing(x, N_EXPTS_ACT)
assert torch.all(tri_gate_indx == ref_gate_indx)
assert torch.all(tri_topk_indx == ref_topk_indx)
assert torch.all(tri_gate_scal == ref_gate_scal)
assert torch.all(tri_metadata == ref_metadata)
