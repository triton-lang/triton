from dataclasses import dataclass

import torch

from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows


@dataclass
class Bitmatrix:
    """
    Represents a boolean matrix in a packed format where each element occupies
    a single bit of memory.

    We use a Bitmatrix to represent the routing information, where each row
    corresponds to a token and each column corresponds to an expert.

    S is either None or an all-zero array of size >= n_cols; we pass it along
    with the actual bitmatrix to avoid having to launch a separate memset
    kernel when we call Bitmatrix::sum().
    """

    data: torch.Tensor
    shape: tuple[int]
    S: torch.tensor

    def sum(self, partials_block_size):
        n_rows, n_cols = self.shape
        dev = self.data.device
        if self.S is None:
            self.S = clear_sums(n_cols, dev)
        out_ret = self.S[:n_cols]
        self.S = None  # throw error if we try to sum again
        return sum_bitmatrix_rows(self, out_ret, partials_block_size)
