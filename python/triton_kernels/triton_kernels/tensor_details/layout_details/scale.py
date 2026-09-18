from abc import abstractmethod

import torch
from torch._subclasses.fake_tensor import is_fake
from triton._compile_warmup_state import is_compile_warmup

from .base import LayoutTransformation
from .strided import StridedLayoutTransformation


class ScaleLayoutTransformation(LayoutTransformation):
    """Byte-scale layouts with direct CUDA conversions to and from strided storage."""

    def _can_convert(self, data):
        # Compile warmup intercepts launches; ordinary FakeTensors must use Torch.
        return (not self.is_fp4 and data.device.type == "cuda" and data.dtype.itemsize == 1
                and (not is_fake(data) or is_compile_warmup()))

    def convert_data(self, data, destination: LayoutTransformation, *, out=None):
        if isinstance(destination, StridedLayoutTransformation) and self._can_convert(data):
            if out is None:
                out = torch.empty_strided(destination.storage_shape, destination.storage_strides, dtype=data.dtype,
                                          device=data.device)
            return self._convert(data, out, inverse=True)
        return super().convert_data(data, destination, out=out)

    def _convert_data_from(self, data, source: LayoutTransformation, *, out):
        if isinstance(source, StridedLayoutTransformation) and self._can_convert(data):
            return self._convert(data, out, inverse=False)
        return super()._convert_data_from(data, source, out=out)

    @abstractmethod
    def _convert(self, data, out, inverse):
        pass
