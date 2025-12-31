from dataclasses import dataclass
from .base import Layout, LayoutTransformation

# ------------------- Strided Layout -------------------


@dataclass(frozen=True)
class StridedLayout(Layout):

    def make_transformation(self, shape: list[int]) -> LayoutTransformation:
        return StridedLayoutTransformation(shape)

    @property
    def name(self):
        return "STRIDED"


# ------------------- Strided Layout Transformation -------------------


@dataclass(frozen=True)
class StridedLayoutTransformation(LayoutTransformation):

    def swizzle_data(self, data):
        return data

    def unswizzle_data(self, data):
        return data

    def swizzle_block_shape(self, block_shape):
        return block_shape
