import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(
    X,
    stride_xm,
    stride_xn,
    RowIndx,
    OUT,
    stride_om,
    stride_on,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    # Load selected row index for this token
    fi = tl.load(RowIndx + pid_t)
    InColPtrs = X + tl.arange(0, BLOCK_N) * stride_xn
    OutColPtrs = OUT + pid_t * stride_om + tl.arange(0, BLOCK_N) * stride_on
    for n_curr in tl.range(0, N, BLOCK_N, num_stages=3):
        n_mask = tl.arange(0, BLOCK_N) < (N - n_curr)
        eff_fi = tl.where(fi == -1, 0, fi)
        in_ptr = InColPtrs + eff_fi * stride_xm
        vals = tl.load(in_ptr, mask=n_mask & (fi != -1), other=0.0)
        tl.store(OutColPtrs, vals, mask=n_mask)
        InColPtrs += BLOCK_N * stride_xn
        OutColPtrs += BLOCK_N * stride_on


def scatter(x: torch.Tensor, indx: torch.Tensor):
    """
    Row-wise scatter (index-based row selection).

    - x: Tensor [num_rows, N]
    - indx: Tensor[int32] [M], row indices into `x` (or -1)
    Returns: out [M, N]
    """
    assert indx.ndim == 1
    num_tokens = indx.shape[0]
    n_cols = x.shape[-1]
    out = torch.empty((num_tokens, n_cols), dtype=x.dtype, device=x.device)
    BLOCK_N = 1024
    _scatter_kernel[(num_tokens, )](
        x,
        x.stride(0),
        x.stride(1),
        indx,
        out,
        out.stride(0),
        out.stride(1),
        n_cols,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


def scatter_torch(x: torch.Tensor, row_indx: torch.Tensor):
    """
    Torch reference for row-wise scatter.
    - x: [num_rows, N]
    - row_indx: [M], row indices in `x` (or -1)
    Returns: out [M, N]
    """
    num_tokens = row_indx.shape[0]
    out = torch.zeros((num_tokens, x.shape[-1]), dtype=x.dtype, device=x.device)
    valid = row_indx != -1
    if valid.any():
        rows = x.index_select(0, row_indx[valid].to(torch.long))
        out.index_copy_(0, torch.nonzero(valid, as_tuple=False).squeeze(1), rows)
    return out
