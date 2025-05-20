from dataclasses import dataclass

import torch
import triton

from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows


@dataclass
class Bitmatrix:
    data: torch.Tensor
    shape: tuple[int]
    S: torch.tensor

    def sum(self, partials_block_size):
        cdiv = triton.cdiv
        n_rows, n_cols = self.shape
        dev = self.data.device
        if self.S is None:
            self.S = clear_sums(n_cols, dev)
        out_ret = self.S[:n_cols]
        self.S = None  # throw error if we try to sum again
        out_partials = torch.empty((cdiv(n_rows, partials_block_size), n_cols), dtype=torch.int32, device=dev)
        return sum_bitmatrix_rows(self, out_ret, out_partials, partials_block_size)
