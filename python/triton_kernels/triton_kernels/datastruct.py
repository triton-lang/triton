import torch
from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows


class Tensor:

    def __init__(self, handle, shape_raw, shape_pad=None):
        self.handle = handle
        self.ndim = handle.ndim
        self.dtype = handle.dtype
        self.device = handle.device
        self.shape_pad = handle.shape if shape_pad is None else shape_pad
        self.shape_raw = shape_raw

    def stride(self, *args):
        return self.handle.stride(*args)

    def data_ptr(self):
        return self.handle.data_ptr()


class Bitmatrix(Tensor):
    """
    Represents a boolean matrix in a packed format where each element occupies
    a single bit of memory.

    _scratchpad is either None or an all-zero array of size >= shape[-1]; we pass it along
    with the actual bitmatrix to avoid having to launch a separate memset
    kernel when we call Bitmatrix::sum().
    """

    _scratchpad: torch.Tensor

    def __init__(self, handle, shape_raw, scratchpad=None):
        assert handle.ndim == 2
        shape_pad = [handle.shape[0], handle.shape[1] * 32]
        super().__init__(handle, shape_raw, shape_pad)
        assert self.dtype == torch.uint32
        self._scratchpad = scratchpad

    def sum(self, partials_block_size):
        _, n_cols = self.shape_raw
        dev = self.device
        if self._scratchpad is None:
            self._scratchpad = clear_sums(n_cols, dev)
        out_ret = self._scratchpad[:n_cols]
        self._scratchpad = None  # throw error if we try to sum again
        return sum_bitmatrix_rows(self, out_ret, partials_block_size, self.shape_raw[0])
