import torch
import triton
from triton_kernels.topk_details._topk_forward import _topk_forward
from triton_kernels.topk_details._topk_backward import _topk_backward
from triton_kernels.datastruct import Bitmatrix
from triton_kernels.datastruct import Tensor


def topk_forward(x, k, apply_softmax=True, dim=1, return_bitmatrix=True, y_indx=None, n_rows=None):
    if not isinstance(x, Tensor):
        x = Tensor(x, [n_rows, None])
    cdiv = lambda a, b: (a + b - 1) // b
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_S = 128
    assert x.ndim == 2
    assert x.shape_pad[-1] < 32768
    assert dim == 1
    assert return_bitmatrix
    n_rows_pad, n_cols = x.shape_pad
    n_rows_raw = x.shape_raw[0]
    dev = x.device
    n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    # scratchpad tensors
    # NOTE: these are not returned
    y_vals = torch.empty((n_rows_pad, k), dtype=x.dtype, device=dev)
    if y_indx is not None:
        use_provided_indx = True
    else:
        y_indx = torch.empty((n_rows_pad, k), dtype=torch.int16, device=dev)
        use_provided_indx = False
    # create bitmatrix in transposed memory layout:
    bitmatrix = torch.empty((n_cols_words, cdiv(n_rows_pad, 32) * 32), dtype=torch.uint32, device=dev)
    bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows_pad]
    s_blocks = cdiv(n_cols, BLOCK_S)
    s_cols = s_blocks * BLOCK_S
    scratchpad = torch.empty((s_cols, ), dtype=torch.int32, device=dev)
    pids = max(cdiv(n_rows_pad, BLOCK_M), s_blocks)
    _topk_forward[(pids, )](
        x, x.stride(0),  # inputs
        y_vals, y_indx, y_vals.stride(0), use_provided_indx,  # output [topk]
        bitmatrix, bitmatrix.stride(0), bitmatrix.stride(1),  # output [bitmatrix]
        n_rows_pad, n_rows_raw, n_cols,  # shapes
        scratchpad, BLOCK_S, s_blocks,  # thing to memset to zero
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  # tunable parameter
        APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
    )
    return y_vals, y_indx, Bitmatrix(bitmatrix, [n_rows_raw, n_cols], scratchpad)


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
    def forward(ctx, x, k, apply_softmax, dim, return_bitmatrix, y_indx, n_rows):
        y_vals, y_indx, bitmatrix = topk_forward(x, k, apply_softmax, dim, return_bitmatrix, y_indx, n_rows)
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


def topk(x, k, apply_softmax=True, dim=1, return_bitmatrix=True, y_indx=None, n_rows=None):
    ret = TopK.apply(x, k, apply_softmax, dim, return_bitmatrix, y_indx, n_rows)
    return ret


# x = torch.randn((32, 32), dtype=torch.float16, device="cuda")
# print(topk(x, 4))
