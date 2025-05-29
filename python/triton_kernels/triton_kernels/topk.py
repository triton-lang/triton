import torch
from .topk_details._topk import _topk
from .datastruct import Bitmatrix


def topk(x, k, dim=1, return_bitmatrix=True, y_indx=None, n_rows_raw=None):
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
    _topk[(pids, )](
        x, x.stride(0),  # inputs
        y_vals, y_indx, y_vals.stride(0), use_provided_indx,  # output [topk]
        bitmatrix, bitmatrix.stride(0), bitmatrix.stride(1),  # output [bitmatrix]
        n_rows_pad, n_rows_raw, n_cols,  # shapes
        scratchpad, BLOCK_S, s_blocks,  # thing to memset to zero
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  # tunable parameter
        N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
    )
    return y_vals, y_indx, Bitmatrix(bitmatrix, [n_rows_raw, n_cols], scratchpad)
