from dataclasses import dataclass
import triton
import triton.language as tl
import torch


@dataclass
class BitmatrixMetadata:
    """
    Example:
    `bitmatrix` = [0 0 1 0 1 1 0
                   0 1 0 0 0 1 0
                   1 1 1 0 0 0 1
                   0 0 1 0 1 0 0]
    `col_sum` = [1 2 3 0 2 2 1]
    `row_sorted_indx` = cat([3 6 8], [1 9], [0 2 4 10], [5 7])
    `col_sorted_indx` = cat([5], [3 6], [0 7], [], [9 1 10], [2 4], [8])
    """
    # the number of entries equal to 1 in each column
    col_sum: torch.Tensor
    # indices of nonzero values numbered col-major, grouped by rows, concatenated
    row_sorted_indx: torch.Tensor
    # indices of nonzero values numbered row-major, grouped by cols, concatenated
    col_sorted_indx: torch.Tensor


# `make_bitmatrix_metadata`: triton implementation
# ---------------------------------------------------------------------------- #


@triton.jit
def _bitmatrix_metadata_compute_expt_offs(ExpertHist, FinalExpertOffs, hist_size,  # histogram
                                          BLOCK_N: tl.constexpr):
    loop_iterations = (hist_size + BLOCK_N - 1) // BLOCK_N
    x = tl.zeros([BLOCK_N], ExpertHist.dtype.element_ty)
    for i in range(loop_iterations):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < hist_size
        hist2 = tl.load(ExpertHist + offs_n, mask=mask_n)
        tok_starts = tl.cumsum(hist2, 0) - hist2 + x
        x += tl.sum(hist2, 0)
        tl.store(FinalExpertOffs + offs_n, tok_starts, mask=mask_n)
        offs_n += BLOCK_N


@triton.jit
def _bitmatrix_metadata_compute_indx_offs(PartialHist, shape_pm, stride_pm, stride_pn, BLOCK_M: tl.constexpr, expt_id):
    offs_m = tl.arange(0, BLOCK_M)
    # iterate over input data
    curr_sum = 0
    for _ in range(0, shape_pm, BLOCK_M):
        offs = offs_m * stride_pm + expt_id * stride_pn
        curr = tl.load(PartialHist + offs, mask=offs_m < shape_pm)
        out = tl.cumsum(curr, 0) + curr_sum
        curr_sum += tl.sum(curr, 0)
        tl.store(PartialHist + offs, out - curr, mask=offs_m < shape_pm)
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
def _bitmatrix_metadata_compute_indx(ColSortedIndx, RowSortedIndx, NonzeroIndx, nonzero_indx_size, ColPartialSum,
                                     stride_pm, stride_pn, ColOffs, BLOCK_SIZE: tl.constexpr):
    tl.static_assert(BLOCK_SIZE <= 32768)
    pid_m = tl.program_id(0)

    offs_local = tl.arange(0, BLOCK_SIZE)
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < nonzero_indx_size
    col_indx = tl.load(NonzeroIndx + offs_global, mask=mask, other=-1).to(tl.uint32)

    # stable-sort by columns index
    kv_pairs = ((col_indx << 16) | offs_local).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    col_indx = kv_pairs >> 16
    offs_global = pid_m * BLOCK_SIZE + (kv_pairs & 0xffff)
    mask = col_indx != 0xffff

    # compute run lengths in column-sorted order:
    x = (kv_pairs & 0xffff0000 | 0x00000001)
    cols_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = (cols_and_inclusive_run_lengths - 1) & 0xffff

    row_sorted_indx = tl.load(ColPartialSum + pid_m * stride_pm + col_indx * stride_pn, mask=mask)
    row_sorted_indx += tl.load(ColOffs + col_indx, mask=mask)
    row_sorted_indx += exclusive_run_lengths

    tl.store(RowSortedIndx + offs_global, row_sorted_indx, mask=mask)
    tl.store(ColSortedIndx + row_sorted_indx, offs_global, mask=mask)


@triton.jit
def _bitmatrix_metadata_memset(Indx, size, sentinel, BLOCK: tl.constexpr, ExpertHist, FinalExpertOffs, hist_size,
                               n_expts_tot, PartialHist, shape_pm, stride_pm, stride_pn, BLOCK_N: tl.constexpr,
                               BLOCK_M: tl.constexpr):
    """
    This kernel essentially combines 6 different pieces of functionality,
    statically branching on the value of tl.program_id(0) to decide which
    codepath to take.

        pid == 0:                                  create the token cumsum
        1 <= pid <= SIZES:                         create a tile cumsum
        SIZES < pid < blocks1a:                    initialise MDTileInfo to 0xffffffff
        blocks1a <= pid < blocks1a + n_expts_tot:  compute_indx_offs
        pid == blocks1a + n_expts_tot:             compute_expt_offs
        pid > blocks1a + n_expts_tot:              initialise Indx to sentinel

    As each of these is a relatively trivial workload, launching them from
    this single trampoline is beneficial as they can execute on different
    streaming multiprocesses in parallel.
    """

    pid = tl.program_id(0)
    if pid == n_expts_tot:
        _bitmatrix_metadata_compute_expt_offs(ExpertHist, FinalExpertOffs, hist_size, BLOCK_N)
    elif pid < n_expts_tot:
        _bitmatrix_metadata_compute_indx_offs(PartialHist, shape_pm, stride_pm, stride_pn, BLOCK_M, pid)
    else:
        offs = (pid - n_expts_tot - 1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < size
        tl.store(Indx + offs, sentinel, mask=mask)


@triton.jit
def vpopc(x):
    """
    Vertical popcount
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
def _sum_bitmatrix_rows(B, shape_bm, stride_bm: tl.constexpr, stride_bn: tl.constexpr,  # input bitmatrix
                        Ret, Partials, stride_pm: tl.constexpr, stride_pn, shape_pn,  # outputs
                        BLOCK_MM: tl.constexpr, BLOCK_M: tl.constexpr):

    tl.static_assert(BLOCK_MM % BLOCK_M == 0)
    TILE_SIZE: tl.constexpr = BLOCK_MM // BLOCK_M
    if isinstance(shape_bm, tl.tensor) and shape_bm.dtype.is_ptr():
        shape_bm = tl.load(shape_bm)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_MM + tl.arange(0, BLOCK_MM)
    offs_n = pid_n * 32 + tl.arange(0, 32)
    n_rows = shape_bm
    bits = tl.load(B + pid_n * stride_bn + offs_m * stride_bm, mask=offs_m < n_rows, other=0)
    bits = tl.reshape(bits, [TILE_SIZE, BLOCK_M])
    ret = vpopc(bits)  # [TILE_SIZE, 32]

    offs_t = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)

    tl.atomic_add(Ret + offs_n, tl.sum(ret, 0), sem="relaxed")
    tl.store(Partials + offs_t[:, None] * stride_pm + offs_n[None, :] * stride_pn, ret)


def sum_bitmatrix_rows(x, out_ret, partials_block_size=None):
    assert partials_block_size is not None
    cdiv = triton.cdiv
    PARTIALS_BLOCK_M = partials_block_size
    n_rows, n_cols = x.shape
    n_rows_max = x.shape_max[0]
    assert out_ret.shape == (n_cols, )

    TILE_SIZE = max(1, 128 // PARTIALS_BLOCK_M)
    BLOCK_MM = PARTIALS_BLOCK_M * TILE_SIZE

    pids_x = cdiv(n_rows_max, BLOCK_MM)
    pids_y = cdiv(n_cols, 32)
    out_partials = torch.empty((pids_y * 32, pids_x * TILE_SIZE), device=out_ret.device, dtype=torch.int32)
    out_partials = torch.transpose(out_partials, 0, 1)

    # output tensors
    _sum_bitmatrix_rows[(pids_x, pids_y)](
        x.storage.data, n_rows, x.stride(0), x.stride(1),  # input
        out_ret,  # output [final reduction]
        out_partials, out_partials.stride(0), out_partials.stride(1),
        out_partials.shape[1],  # output [partial reductions]
        BLOCK_M=PARTIALS_BLOCK_M, BLOCK_MM=BLOCK_MM,  # constants
        num_warps=8)

    out_partials = out_partials[:cdiv(n_rows_max, PARTIALS_BLOCK_M), :]

    return out_ret, out_partials


def make_bitmatrix_metadata(nonzero_indx, bitmatrix):
    HIST_BLOCK_M = 32
    MEMSET_BLOCK = 512

    cdiv = lambda x, y: (x + y - 1) // y
    device = bitmatrix.device
    _, n_cols = bitmatrix.shape

    blocks = cdiv(n_cols, MEMSET_BLOCK)
    scratchpad = torch.zeros((blocks * MEMSET_BLOCK, ), device=device, dtype=torch.int32)[:n_cols]
    col_sum, col_partial_sum = sum_bitmatrix_rows(bitmatrix, scratchpad, HIST_BLOCK_M)

    assert col_sum.dtype == torch.int32
    nonzero_indx = nonzero_indx.to(torch.int32)
    # allocate memory
    n_indx = nonzero_indx.numel()
    n_cols = bitmatrix.shape[1]
    col_offs = torch.empty(n_cols, dtype=torch.int32, device=device)
    combined_indx = torch.empty(n_indx * 2, dtype=torch.int32, device=device)
    col_sorted_indx = combined_indx[:n_indx]
    row_sorted_indx = combined_indx[n_indx:]
    # memset the output
    MEMSET_BLOCK = 1024
    INDX_OFFS_BLOCK_M = 512
    memset_grid = (cdiv(n_indx * 2, MEMSET_BLOCK) + n_cols + 1, )
    _bitmatrix_metadata_memset[memset_grid](
        combined_indx, n_indx * 2, -1, MEMSET_BLOCK, col_sum,  #
        col_offs, col_sum.shape[0], n_cols, col_partial_sum,  # inputs
        col_partial_sum.shape[0], col_partial_sum.stride(0), col_partial_sum.stride(1),  # outputs
        BLOCK_N=512, BLOCK_M=INDX_OFFS_BLOCK_M,  # tunable parameters
    )
    # compute the output
    n_indx = nonzero_indx.numel()
    toks_per_row = nonzero_indx.shape[-1]
    compute_grid = (cdiv(bitmatrix.shape_max[0], HIST_BLOCK_M), )
    _bitmatrix_metadata_compute_indx[compute_grid](
        col_sorted_indx, row_sorted_indx,  # outputs
        nonzero_indx, nonzero_indx.numel(), col_partial_sum, col_partial_sum.stride(0),
        col_partial_sum.stride(1),  # inputs
        col_offs, BLOCK_SIZE=HIST_BLOCK_M * toks_per_row,  # constants (rows per tile * toks_per_row)
    )
    return BitmatrixMetadata(col_sum, col_sorted_indx, row_sorted_indx)


# `make_bitmatrix_metadata`: reference implementation
# ---------------------------------------------------------------------------- #


def make_bitmatrix_metadata_torch(nonzero_indx, bitmatrix):
    n_batches = bitmatrix.shape[1]
    nonzero_indx = nonzero_indx.reshape(-1).to(torch.int32)
    col_sorted_indx = torch.argsort(nonzero_indx, stable=True).int()
    row_sorted_indx = torch.argsort(col_sorted_indx, stable=True).int()
    col_sum = torch.histc(nonzero_indx, bins=n_batches, max=n_batches - 1).int()  # histogram of tokens over experts
    return BitmatrixMetadata(col_sum, col_sorted_indx, row_sorted_indx)
