import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_grouped(X, stride_xm, stride_xn,  #
                    InIndx, N, OutIndx,  #
                    K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_t = tl.program_id(0)
    # persistent along N: single program on N, iterate tiles of size BLOCK_N
    start = pid_t * K
    # load indices into a tuple
    indxs = ()
    for i in tl.static_range(0, K):
        indxs = indxs + (tl.load(InIndx + start + i), )
    # determine first valid topk row
    fi = indxs[(K - 1)]
    for i in tl.static_range(K - 2, -1, -1):
        fi = tl.where(indxs[i] != -1, indxs[i], fi)
    # record overwritten row index (may be -1 if none)
    tl.store(OutIndx + pid_t, fi)
    ColPtrs = X + tl.arange(0, BLOCK_N) * stride_xn
    for n_curr in tl.range(0, N, BLOCK_N, num_stages=4):
        n_mask = tl.arange(0, BLOCK_N) < N - n_curr
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # accumulate contributions for this tile
        for i in tl.static_range(0, K):
            is_valid = indxs[i] != -1
            row_ptr = ColPtrs + indxs[i] * stride_xm
            vals = tl.load(row_ptr, mask=n_mask & is_valid, other=0.0)
            acc += vals.to(tl.float32)
        # write-back for this tile
        out_ptr = ColPtrs + fi * stride_xm
        tl.store(out_ptr, acc, mask=n_mask & (fi != -1))
        ColPtrs += BLOCK_N * stride_xn


def reduce_grouped(x: torch.Tensor, indx: torch.Tensor, inplace: bool = True):
    """
    In-place grouped row reduction.

    - x: Tensor[AnyFloat] of shape [(num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K]
    - inplace: only True is supported; the function overwrites the first valid
      row per group with the per-group sum (fp32 accumulate).

    Returns (x, overwritten_idx):
    - x: the input tensor modified in-place
    - overwritten_idx: int32 tensor [num_groups] with the row index overwritten per group (-1 if none)
    """
    assert inplace, "only inplace=True is supported for now"
    assert x.shape[0] == indx.numel()
    num_groups = indx.shape[0]
    overwritten = torch.empty((num_groups, ), dtype=torch.int32, device=x.device)
    BLOCK_N = 512
    _reduce_grouped[(num_groups, )](
        x, x.stride(0), x.stride(1),  #
        indx, x.shape[1], overwritten,  #
        BLOCK_N=BLOCK_N, K=indx.shape[1],  #
        num_warps=1,  #
    )
    return x, overwritten


def reduce_grouped_torch(x: torch.Tensor, indx: torch.Tensor):
    """
    Torch reference for grouped reduction; same semantics as reduce_grouped.
    - x: [(num_groups*K), N]
    - indx: [num_groups, K] with -1 marking invalid entries
    Returns (x_out, overwritten_idx)
    """
    assert indx.ndim == 2
    num_groups, k = indx.shape
    src = indx.to(torch.long).reshape(-1)
    positions = torch.arange(src.numel(), device=src.device, dtype=torch.long)
    valid = src != -1
    # Sum contributions per group across valid rows (fp32)
    token_id = positions // k
    valid_pos = torch.nonzero(valid, as_tuple=False).squeeze(1)
    contrib = x.index_select(0, src.index_select(0, valid_pos)).to(torch.float32)
    sums = torch.zeros((num_groups, x.shape[-1]), dtype=torch.float32, device=x.device)
    sums.index_add_(0, token_id.index_select(0, valid_pos), contrib)
    # First valid per group (left-to-right)
    big = torch.full_like(src, src.numel(), dtype=torch.long)
    masked_pos = torch.where(valid, positions, big)
    first_pos_flat = masked_pos.view(num_groups, k).min(dim=1).values
    valid_group_mask = first_pos_flat != big[0]
    overwritten = -torch.ones((num_groups, ), dtype=torch.int32, device=x.device)
    if valid_group_mask.any():
        idx_keep = torch.nonzero(valid_group_mask, as_tuple=False).squeeze(1)
        first_rows = src.index_select(0, first_pos_flat[valid_group_mask])
        x.index_copy_(0, first_rows, sums.index_select(0, idx_keep).to(x.dtype))
        overwritten.index_copy_(0, idx_keep, first_rows.to(torch.int32))
    return x, overwritten
