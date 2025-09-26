import torch
import triton
from triton_kernels.topk_details._topk_forward import _topk_forward
from triton_kernels.topk_details._topk_backward import _topk_backward
from triton_kernels.tensor import Tensor, Bitmatrix
from typing import Optional, Union
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed as dist


def make_empty(shape, dtype, device, all_gather):
    if all_gather:
        n_ranks = dist.get_world_size()
        ret = symm_mem.empty(shape, dtype=dtype, device=device)

        ret_hdl = symm_mem.rendezvous(ret, dist.group.WORLD)
        ret_bufs = tuple([ret_hdl.get_buffer(r, ret.shape, ret.dtype) for r in range(n_ranks)])
        return ret_bufs, ret, ret_hdl
    ret = torch.empty(shape, dtype=dtype, device=device)
    return (ret, ), ret, None


def topk_forward(x, k, apply_softmax=True, dim=1, return_bitmatrix=True, y_indx=None, n_rows=None, all_gather=False):
    if not isinstance(x, Tensor):
        x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
        x_shape_max = [x.shape[0], x.shape[1]]
        x = Tensor(x, shape=x_shape, shape_max=x_shape_max)
    cdiv = lambda a, b: (a + b - 1) // b
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_S = 128
    use_provided_indx = y_indx is not None
    assert len(x.shape) == 2
    assert x.shape_max[-1] < 32768
    assert dim == 1
    assert return_bitmatrix
    assert not all_gather or not use_provided_indx
    n_rows, n_cols = x.shape
    n_rows_max, _ = x.shape_max
    dev = x.device
    n_rows_out_max = n_rows_max * dist.get_world_size() if all_gather else n_rows_max
    # scratchpad tensors
    # NOTE: these are not returned
    y_vals_bufs, y_vals, y_vals_hdl = make_empty((n_rows_out_max, k), x.dtype, dev, all_gather=all_gather)
    if y_indx is None:
        y_indx_bufs, y_indx, y_indx_hdl = make_empty((n_rows_out_max, k), torch.int16, dev, all_gather=all_gather)
    # create bitmatrix in transposed memory layout:
    n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix_bufs, bitmatrix, bitmatrix_hdl = make_empty((n_cols_words, cdiv(n_rows_out_max, 32) * 32), torch.uint32,
                                                          dev, all_gather=all_gather)
    bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows_out_max]
    s_blocks = cdiv(n_cols, BLOCK_S)
    s_cols = s_blocks * BLOCK_S
    scratchpad = torch.empty((s_cols, ), dtype=torch.int32, device=dev)
    pids = max(cdiv(n_rows_max, BLOCK_M), s_blocks)
    _topk_forward[(pids, )](
        x, x.stride(0),  # inputs
        y_vals_bufs, y_indx_bufs, y_vals.stride(0), use_provided_indx,  # output [topk]
        bitmatrix_bufs, bitmatrix.stride(0), bitmatrix.stride(1),  # output [bitmatrix]
        n_rows, n_cols,  # shapes
        scratchpad, BLOCK_S, s_blocks,  # thing to memset to zero
        dist.get_rank() * n_rows_max if all_gather else 0, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  # tunable parameter
        APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
    )
    if all_gather:
        y_vals_hdl.barrier(channel=0)
        y_indx_hdl.barrier(channel=0)
        bitmatrix_hdl.barrier(channel=0)
    bitmatrix_shape = [n_rows_out_max, n_cols_words * 32]
    bitmatrix_shape_max = [n_rows_out_max, None]
    bitmatrix = Bitmatrix(bitmatrix, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max, scratchpad=scratchpad)
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
    def forward(ctx, x, k, apply_softmax, dim, return_bitmatrix, y_indx, n_rows, all_gather):
        y_vals, y_indx, bitmatrix = topk_forward(x, k, apply_softmax, dim, return_bitmatrix, y_indx, n_rows, all_gather)
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
    return_bitmatrix: bool = True,
    y_indx: Optional[torch.Tensor] = None,
    n_rows: Optional[int] = None,
    all_gather: bool = False,
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
    return_bitmatrix : bool, default True
        A bitmatrix of shape (n_tokens, cdiv(n_expts, 32)).
        Each bit on [t, b] indicates whether the b-th expert was selected for the t-th token.
    y_indx : torch.Tensor, optional
        Pre-allocated tensor for storing indices of top-k elements with shape (n_tokens, k).
        If provided, we skip the computation of top-k indices and use this tensor instead.
    n_rows : int, optional
        Number of rows to apply top-k on. If None, we consider all rows in `x`.

    Returns
    -------
    (expt_scal, expt_indx, bitmatrix) : Tuple[torch.Tensor, torch.Tensor, Bitmatrix]
    """
    ret = TopK.apply(x, k, apply_softmax, dim, return_bitmatrix, y_indx, n_rows, all_gather)
    return ret
