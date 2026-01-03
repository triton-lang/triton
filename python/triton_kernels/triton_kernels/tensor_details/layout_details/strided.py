from dataclasses import dataclass
from .base import Layout, LayoutTransformation
from .torch_utils import unpack, pack


# ------------------- Layout Definition -------------------
@dataclass(frozen=True)
class StridedLayout(Layout):

    order: list[int]

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return StridedLayoutTransformation(shape, is_fp4, [x for x in self.order])

    @property
    def name(self):
        return "STRIDED"

    def swizzle_block_shape(self, block_shape):
        return block_shape


@dataclass(frozen=True)
class StridedLayoutTransformation(LayoutTransformation):

    order: list[int]

    def swizzle_data(self, data):
        assert data.stride(-1) == 1
        data = unpack(data, -1, self.is_fp4)
        assert list(data.shape) == list(self.shape), f"{data.shape} != {self.shape}"
        ret = pack(data, self.order[0], self.is_fp4)
        inv = [0] * len(self.order)
        for i, d in enumerate(reversed(self.order)):
            inv[d] = i
        ret = ret.permute(*reversed(self.order)).contiguous().permute(*inv)
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
