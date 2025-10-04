from dataclasses import dataclass
import triton
import triton.language as tl
import torch
from ..reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows


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
def _bitmatrix_metadata_compute_indx(pid_m, GatherIndx, ScatterIndx, ExptIndx, PartialOffs, stride_pm, stride_pn,
                                     TokensStart, n_tokens, BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr):

    if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
        n_tokens = tl.load(n_tokens)
    n_gates = n_tokens * N_EXPTS_ACT

    tl.static_assert(N_EXPTS_ACT * BLOCK_M <= 32768)

    local_offs = tl.arange(0, N_EXPTS_ACT * BLOCK_M)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + local_offs
    expert = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1).to(tl.uint32)

    # stable-sort by expert ID:
    kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + (kv_pairs & 0xffff)
    mask = expert != 0xffff

    # compute run lengths in expert-sorted order:
    x = (kv_pairs & 0xffff0000 | 0x00000001)
    expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xffff

    gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn, mask=mask)
    gates += tl.load(TokensStart + expert, mask=mask)
    gates += exclusive_run_lengths

    tl.store(ScatterIndx + offs, gates, mask=mask)
    tl.store(GatherIndx + gates, offs, mask=mask)


@triton.jit
def _bitmatrix_metadata_compute(GatherIndx, ScatterIndx, ExptIndx, PartialOffs, stride_pm, stride_pn, TokensStart,
                                n_tokens, BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr):

    pid = tl.program_id(0)
    _bitmatrix_metadata_compute_indx(pid, GatherIndx, ScatterIndx, ExptIndx, PartialOffs, stride_pm, stride_pn,
                                     TokensStart, n_tokens, BLOCK_M, N_EXPTS_ACT)


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


def sum_bitmatrix(bitmatrix, partials_block_size):
    _, n_cols = bitmatrix.shape
    scratchpad = clear_sums(n_cols, bitmatrix.device)
    out_ret = scratchpad[:n_cols]
    return sum_bitmatrix_rows(bitmatrix, out_ret, partials_block_size)


def make_bitmatrix_metadata(nonzero_indx, bitmatrix):
    # TODO: `nonzero_indx` can be computed from `bitmatrix`; remove from API
    HIST_BLOCK_M = 32
    cdiv = lambda x, y: (x + y - 1) // y
    device = bitmatrix.device
    col_sum, col_partial_sum = sum_bitmatrix(bitmatrix, partials_block_size=HIST_BLOCK_M)
    assert col_sum.dtype == torch.int32
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
    compute_grid = (cdiv(bitmatrix.shape_max[0], HIST_BLOCK_M), )
    _bitmatrix_metadata_compute[compute_grid](
        col_sorted_indx, row_sorted_indx,  # outputs
        nonzero_indx, col_partial_sum, col_partial_sum.stride(0), col_partial_sum.stride(1),  # inputs
        col_offs, bitmatrix.shape[0],  # input shape
        HIST_BLOCK_M, nonzero_indx.shape[-1],  # constants
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
