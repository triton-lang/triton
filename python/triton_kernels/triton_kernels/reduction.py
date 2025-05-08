import torch
import triton
from .reduction_details.reduce_bitmatrix import sum_bitmatrix_rows
from . import Bitmatrix


def sum(x, partials_block_size=None, dim=0):
    cdiv = triton.cdiv
    assert isinstance(x, Bitmatrix)
    assert dim == 0
    assert partials_block_size is not None
    n_rows, n_cols = x.shape
    dev = x.data.device
    out_ret = torch.empty(n_cols, dtype=torch.int32, device=dev)
    out_partials = torch.empty((cdiv(n_rows, partials_block_size), n_cols), dtype=torch.int32, device=dev)
    return sum_bitmatrix_rows(x, out_ret, out_partials, partials_block_size)
