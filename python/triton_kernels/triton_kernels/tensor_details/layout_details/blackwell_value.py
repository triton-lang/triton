from dataclasses import dataclass
import torch
from .base import Layout, LayoutTransformation


# ------------------- Blackwell MX Value Layout -------------------
@dataclass(frozen=True)
class BlackwellMXValueLayout(Layout):

    @property
    def name(self):
        return "BLACKWELL_MX_VALUE"

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return BlackwellMXValueLayoutTransformation(shape, is_fp4)


def unpack(data: torch.Tensor, dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    data_lo = (data >> 0) & 0x0F
    data_hi = (data >> 4) & 0x0F
    return torch.cat([data_lo, data_hi], dim=dim)


def pack(data: torch.Tensor, dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    data_lo, data_hi = torch.chunk(data, 2, dim=dim)
    data = (data_hi << 4) | data_lo
    return data


# ------------------- Blackwell MX Value Layout Transformation -------------------
@dataclass(frozen=True)
class BlackwellMXValueLayoutTransformation(LayoutTransformation):

    def swizzle_data(self, data):
        assert data.stride(-1) == 1
        # re-pack as column-major
        data = unpack(data, -1, self.is_fp4)
        data = pack(data, -2, self.is_fp4)
        # leading dimension must be padded to be aligned to 128
        pad_last = (-data.size(-2)) % 128
        ret = torch.nn.functional.pad(data.mT, (0, pad_last)).contiguous().mT
        return ret

    def unswizzle_data(self, data: torch.Tensor):
        assert data.stride(-2) == 1
        sizes = [self.shape[i] for i in range(data.ndim)]
        sizes[-2] //= 2
        data = data[tuple(slice(0, s) for s in sizes)]
        data = unpack(data, -2, self.is_fp4)
        assert list(data.shape) == list(self.shape)
        data = pack(data, -1, self.is_fp4)
        return data

    def swizzle_block_shape(self, block_shape):
        return block_shape
