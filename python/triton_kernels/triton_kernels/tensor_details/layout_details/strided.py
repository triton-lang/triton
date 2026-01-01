from dataclasses import dataclass
from .base import Layout, LayoutTransformation

# ------------------- Strided Layout -------------------


@dataclass(frozen=True)
class StridedLayout(Layout):

    order: list[int]

    def make_transformation(self, shape: list[int]) -> LayoutTransformation:
        return StridedLayoutTransformation(shape, [x for x in self.order])

    @property
    def name(self):
        return None


# ------------------- Strided Layout Transformation -------------------


@dataclass(frozen=True)
class StridedLayoutTransformation(LayoutTransformation):

    order: list[int]

    def swizzle_data(self, data):
        return data.permute(self.order)

    def unswizzle_data(self, data):
        # permutation needed to make `data` row major
        to_row_major = sorted(range(data.ndim), key=lambda d: (data.stride(d), d))[::-1]
        # permutation  needed to retrieve original order
        inv = [0] * data.ndim
        for i, d in enumerate(to_row_major):
            inv[d] = i
        return data.permute(inv)

    def swizzle_block_shape(self, block_shape):
        return block_shape
