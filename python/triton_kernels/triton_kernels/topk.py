import torch
import triton
from triton_kernels.topk_details._topk_forward import _topk_forward
from triton_kernels.topk_details._topk_backward import _topk_backward
from triton_kernels.tensor import SparseMatrix, Tensor
from triton_kernels.tensor import Bitmatrix, BIT, make_bitmatrix_metadata
from typing import Optional, Union


def topk_forward(x, k, apply_softmax=True, dim=1, y_indx=None, n_rows=None):
    if not isinstance(x, Tensor):
        x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
        x_shape_max = [x.shape[0], x.shape[1]]
        x = Tensor(x, shape=x_shape, shape_max=x_shape_max)
    cdiv = lambda a, b: (a + b - 1) // b
    BLOCK_M = 32
    BLOCK_N = 32
    assert len(x.shape) == 2
    assert x.shape_max[-1] < 32768
    assert dim == 1
    n_rows, n_cols = x.shape
    n_rows_max, _ = x.shape_max
    dev = x.device
    # NOTE: these are not returned
    y_vals = torch.empty((n_rows_max, k), dtype=x.dtype, device=dev)
    if y_indx is not None:
        use_provided_indx = True
    else:
        y_indx = torch.empty((n_rows_max, k), dtype=torch.int16, device=dev)
        use_provided_indx = False
    # create bitmatrix in transposed memory layout:
    n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix_data = torch.empty((n_cols_words, cdiv(n_rows_max, 32) * 32), dtype=torch.uint32, device=dev)
    bitmatrix_data = torch.transpose(bitmatrix_data, 0, 1)[:n_rows_max]
    pids = cdiv(n_rows_max, BLOCK_M)
    _topk_forward[(pids, )](
        x, x.stride(0),  # inputs
        y_vals, y_indx, y_vals.stride(0), use_provided_indx,  # output [topk]
        bitmatrix_data, bitmatrix_data.stride(0), bitmatrix_data.stride(1),  # output [bitmatrix]
        n_rows, n_cols,  # shapes
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  # tunable parameter
        APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
    )
    bitmatrix = Bitmatrix(
        bitmatrix_data,
        dtype=BIT,
        shape=[n_rows, n_cols],
        shape_max=[n_rows_max, None],
    )
    bitmatrix.metadata = make_bitmatrix_metadata(y_indx, bitmatrix)
    return y_vals, y_indx, bitmatrix


def topk_backward(x, y_indx, dy_vals, k, n_rows, apply_softmax):
    assert dy_vals.shape[-1] == k
    n_expts_pad = triton.next_power_of_2(x.shape[-1])
    dx = torch.empty_like(x)
    _topk_backward[(dy_vals.shape[0], )](
        y_indx, y_indx.stride(0), dy_vals, dy_vals.stride(0), x, x.stride(0),  # inputs
        dx,  # outputs
        dx.stride(0), x.shape[0], n_rows, x.shape[-1], APPLY_SOFTMAX=apply_softmax, N_EXPTS_ACT=k,
        N_EXPTS_PAD=n_expts_pad)
    return dx


class TopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k, apply_softmax, dim, y_indx, n_rows):
        y_vals, y_indx, bitmatrix = topk_forward(x, k, apply_softmax, dim, y_indx, n_rows)
        ctx.save_for_backward(x, y_indx)
        ctx.apply_softmax = apply_softmax
        ctx.k = k
        ctx.n_rows = n_rows
        return y_vals, y_indx, bitmatrix

    @staticmethod
    def backward(ctx, dy_vals, _0, _1):
        x, y_indx = ctx.saved_tensors
        dx = topk_backward(x, y_indx, dy_vals, ctx.k, ctx.n_rows, ctx.apply_softmax)
        return dx, None, None, None, None, None, None


def topk(
    x: Union[Tensor, torch.Tensor],
    k: int,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: Optional[torch.Tensor] = None,
    n_rows: Optional[int] = None,
):
    """
    Computes the top-k values and indices along a specified dimension of a tensor.
    Note that the input can be either a `Tensor` or a `torch.Tensor`, but the output will always be a `torch.Tensor`.

    Parameters
    ----------
    x : Union[triton_kernels.Tensor, torch.Tensor]
        Input tensor of shape (n_tokens, n_expts).
    k : int
        Number of top elements to retrieve.
    apply_softmax : bool, default True
        Whether to apply softmax to the input tensor before computing top-k.
    dim : int, default 1
        Dimension along which to compute top-k.
    y_indx : torch.Tensor, optional
        Pre-allocated tensor for storing indices of top-k elements with shape (n_tokens, k).
        If provided, we skip the computation of top-k indices and use this tensor instead.
    n_rows : int, optional
        Number of rows to apply top-k on. If None, we consider all rows in `x`.

    Returns
    -------
    SparseMatrix: sparse matrix equal to `x` with non-selected entries set to 0
    """
    y_vals, y_indx, bitmatrix = TopK.apply(x, k, apply_softmax, dim, y_indx, n_rows)
    return SparseMatrix(vals=y_vals, indx=y_indx, mask=bitmatrix)


def topk_torch(
    x,
    k,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: Optional[torch.Tensor] = None,
    n_rows: Optional[int] = None,
) -> SparseMatrix:
    if n_rows is None:
        n_rows = x.shape[0]
    has_user_provided_indx = y_indx is not None
    cdiv = lambda a, b: (a + b - 1) // b
    device = x.device
    assert dim == 1
    assert not isinstance(x, Tensor)
    if not has_user_provided_indx:
        y_indx = torch.argsort(-x, dim=1, stable=True)[:, :k]
    y_indx = y_indx.long()
    y_vals = torch.take_along_dim(x[:n_rows, :], y_indx[:n_rows, :], dim=1)
    y_vals = torch.cat([y_vals, x[n_rows:, :k]], dim=0)
    y_indx = y_indx.int()
    # compute bitmatrix
    _, n_cols = x.shape
    bitmatrix_data = torch.zeros((cdiv(n_cols, 32), cdiv(x.shape[0], 32) * 32), dtype=torch.int32, device=device)
    bitmatrix_data = torch.transpose(bitmatrix_data, 0, 1)[:x.shape[0]]
    # fill bitmatrix
    if apply_softmax:
        y_vals = torch.softmax(y_vals.float(), dim=-1).to(x.dtype)
    if not has_user_provided_indx:
        y_indx, sort_indices = torch.sort(y_indx, dim=1)
        y_vals = torch.gather(y_vals, 1, sort_indices)
    y_indx[n_rows:, :] = -1
    rows = torch.arange(x.shape[0], device=device).unsqueeze(1).expand(-1, y_indx.shape[1]).reshape(-1)
    cols = y_indx.reshape(-1)  # 64-bit safe for div/mod
    word_idx = torch.div(cols, 32, rounding_mode='floor')
    bit_idx = cols % 32
    masks = torch.ones_like(bit_idx) << bit_idx
    bitmatrix_data.index_put_((rows, word_idx), masks, accumulate=True)
    bitmatrix_data = bitmatrix_data.view(torch.uint32)
    bitmatrix = Bitmatrix(bitmatrix_data, shape=x.shape, dtype=BIT)
    return SparseMatrix(vals=y_vals, indx=y_indx, mask=bitmatrix)
