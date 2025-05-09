import torch
from .topk_details._topk import _topk
from triton_kernels import Bitmatrix


def topk(x, k, dim=1, return_bitmatrix=True):
    cdiv = lambda a, b: (a + b - 1) // b
    BLOCK_M = 8
    BLOCK_N = 128
    assert x.dtype.itemsize == 2
    assert x.ndim == 2
    assert x.shape[-1] < 32768
    assert dim == 1
    assert return_bitmatrix
    n_rows, n_cols = x.shape
    dev = x.device
    n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    # scratchpad tensors
    # NOTE: these are not returned
    y_vals = torch.empty((n_rows, k), dtype=x.dtype, device=dev)
    y_indx = torch.empty((n_rows, k), dtype=torch.int16, device=dev)
    bitmatrix = torch.empty((n_rows, n_cols_words), dtype=torch.uint32, device=dev)
    _topk[(cdiv(n_rows, BLOCK_M), )](
        x, x.stride(0),  # inputs
        y_vals, y_indx, y_vals.stride(0),  # output [topk]
        bitmatrix, bitmatrix.stride(0),  # output [bitmatrix]
        n_rows, n_cols,  # shapes
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  # tunable parameter
        N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
    )
    return y_vals, y_indx, Bitmatrix(bitmatrix, [n_rows, n_cols])
