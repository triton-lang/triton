from dataclasses import dataclass

import torch
import triton

from .reduction_details.reduce_bitmatrix import sum_bitmatrix_rows


@dataclass
class Bitmatrix:
    data: torch.Tensor
    shape: tuple[int]

    def sum(self, partials_block_size):
        cdiv = triton.cdiv
        n_rows, n_cols = self.shape
        dev = self.data.device
        out_ret = torch.empty(n_cols, dtype=torch.int32, device=dev)
        out_partials = torch.empty((cdiv(n_rows, partials_block_size), n_cols), dtype=torch.int32, device=dev)
        return sum_bitmatrix_rows(self, out_ret, out_partials, partials_block_size)
