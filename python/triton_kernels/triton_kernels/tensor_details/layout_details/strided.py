import torch
from dataclasses import dataclass
from .base import Layout, LayoutTransformation

# ------------------- Strided Layout -------------------


@dataclass(frozen=True)
class StridedLayout(Layout):

    order: list[int]

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return StridedLayoutTransformation(shape, is_fp4, [x for x in self.order])

    @property
    def name(self):
        return None

    def swizzle_block_shape(self, block_shape):
        return block_shape


# ------------------- Strided Layout Transformation -------------------


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


def re_layout(x: torch.Tensor, fastest_to_slowest: list[int]) -> torch.Tensor:
    slowest_to_fastest = list(reversed(fastest_to_slowest))
    tmp = x.permute(*slowest_to_fastest).contiguous()
    inv = [0] * x.ndim
    for i, d in enumerate(slowest_to_fastest):
        inv[d] = i
    return tmp.permute(*inv)


@dataclass(frozen=True)
class StridedLayoutTransformation(LayoutTransformation):

    order: list[int]

    def swizzle_data(self, data):
        assert data.stride(-1) == 1
        data = unpack(data, -1, self.is_fp4)
        assert list(data.shape) == list(self.shape), f"{data.shape} != {self.shape}"
        ret = pack(data, self.order[0], self.is_fp4)
        ret = re_layout(ret, self.order)
        assert ret.stride(self.order[0]) == 1
        return ret

    def unswizzle_data(self, data):
        assert data.stride(self.order[0]) == 1
        data = unpack(data, self.order[0], self.is_fp4)
        assert list(data.shape) == list(self.shape), f"{data.shape} != {self.shape}"
        data = pack(data, -1, self.is_fp4)
        ret = data.contiguous()
        assert ret.stride(-1) == 1
        return ret
