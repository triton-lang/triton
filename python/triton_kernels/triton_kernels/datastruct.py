import torch
from dataclasses import dataclass


@dataclass
class PaddedTensor:
    data: torch.Tensor
    shape_pad: tuple[int]
    shape_raw: tuple[int | torch.Tensor | None]

    @property
    def device(self):
        return self.data.device

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def stride(self, *args):
        return self.data.stride(*args)

    def data_ptr(self):
        return self.data.data_ptr()

    @property
    def shape(self):
        return self.shape_pad


@dataclass
class Bitmatrix(PaddedTensor):
    _scratchpad: torch.Tensor

    def __post_init__(self):
        assert self.data.dtype == torch.int32
        assert self.shape_pad[-1] % 32 == 0
