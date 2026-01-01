from dataclasses import dataclass
from .base import Layout, LayoutTransformation

# ------------------- Strided Layout -------------------


@dataclass(frozen=True)
class StridedLayout(Layout):

    def make_transformation(self, shape: list[int]) -> LayoutTransformation:
        return StridedLayoutTransformation(shape)

    @property
    def name(self):
        return None


# ------------------- Strided Layout Transformation -------------------


@dataclass(frozen=True)
class StridedLayoutTransformation(LayoutTransformation):

    def swizzle_data(self, data):
        return data

    def unswizzle_data(self, data):
        return data
