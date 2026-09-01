from dataclasses import dataclass
import torch
from triton_kernels.tensor_details.layout_details import strided
from .base import Layout, LayoutTransformation
from .torch_utils import repack


# ------------------- Blackwell MX Value Layout -------------------
@dataclass(frozen=True)
class BlackwellMXValueLayout(Layout):

    @property
    def name(self):
        return "BLACKWELL_MX_VALUE"

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return BlackwellMXValueLayoutTransformation(shape, is_fp4)

    def swizzle_block_shape(self, block_shape):
        return block_shape


def strides_major_dim_m2(shape):
    n = len(shape)
    if n <= 1:
        return [1] * n
    order = [n - 2, n - 1] + list(range(n - 3, -1, -1))  # fastest -> slowest
    st = [0] * n
    st[order[0]] = 1
    for prev, d in zip(order, order[1:]):
        st[d] = st[prev] * shape[prev]
    return st


# ------------------- Blackwell MX Value Layout Transformation -------------------
@dataclass(frozen=True)
class BlackwellMXValueLayoutTransformation(LayoutTransformation):

    @property
    def storage_shape(self) -> list[int]:
        *leading_shape, M, K = self.shape
        if self.is_fp4:
            K //= 2
        K *= 2
        M //= 2
        M += -M % 128
        return [*leading_shape, M, K]

    def convert_data(self, data, destination: LayoutTransformation, *, out=None):
        if (not self.is_fp4 or data.device.type != "cuda" or data.dtype != torch.uint8 or self.shape[-2] % 2
                or self.shape[-1] % 2 or not isinstance(destination, strided.StridedLayoutTransformation)
                or destination.order[0] < len(self.shape) - 2):
            return super().convert_data(data, destination, out=out)

        data = self._unpad_data(data)
        if out is None:
            out = torch.empty_strided(destination.storage_shape, destination.storage_strides, device=data.device,
                                      dtype=data.dtype)
        repack(data, -2, destination.order[0], True, out=out)
        return out

    def _convert_data_from(self, data, source: LayoutTransformation, *, out):
        if (not isinstance(source, strided.StridedLayoutTransformation) or not self.is_fp4 or self.shape[-1] % 2
                or self.shape[-2] % 2 or not source._can_convert_fp4(data)):
            return super()._convert_data_from(data, source, out=out)
        if out is None:
            out = torch.empty_strided(self.storage_shape, strides_major_dim_m2(self.storage_shape), device=data.device,
                                      dtype=data.dtype)
        repack(data, source.order[0], -2, True, out=out[..., :self.shape[-2] // 2, :])
        return out

    def swizzle_data(self, data):
        # re-pack as column-major
        ret = torch.empty_strided(self.storage_shape, strides_major_dim_m2(self.storage_shape), device=data.device,
                                  dtype=data.dtype)
        repacked_shape = list(data.shape)
        repacked_shape[-1] *= 2
        repacked_shape[-2] //= 2
        repack(data, -1, -2, self.is_fp4, out=ret[..., :repacked_shape[-2], :])
        return self._validate_storage_shape(ret)

    def _unpad_data(self, data: torch.Tensor):
        sizes = [self.shape[i] for i in range(data.ndim)]
        sizes[-2] //= 2
        return data[tuple(slice(0, s) for s in sizes)]

    def unswizzle_data(self, data: torch.Tensor):
        data = self._unpad_data(data)
        out_shape = list(self.shape)
        out_shape[-1] //= 2
        out = torch.empty(out_shape, device=data.device, dtype=data.dtype)
        repack(data, -2, -1, self.is_fp4, out=out)
        return out
