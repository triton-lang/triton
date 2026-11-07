from dataclasses import dataclass

import torch
from triton import cdiv

from .base import Layout, LayoutTransformation


@dataclass(frozen=True)
class SparseMetadataLayout(Layout):
    """128-row tiles of two selector words per row for FP8 and paired NVFP4."""

    name = "SPARSE_METADATA"

    def swizzle_block_shape(self, block_shape):
        *batch, rows, words = block_shape
        return [*batch, cdiv(rows, 128), cdiv(words, 2), 128, 2]

    def make_transformation(self, shape, is_fp4):
        if is_fp4 or len(shape) not in (2, 3):
            raise ValueError("expected a matrix or batch of selector words")
        return SparseMetadataTransformation(shape, is_fp4, self)


@dataclass(frozen=True)
class SparseMetadataTransformation(LayoutTransformation):
    layout: SparseMetadataLayout

    @property
    def storage_shape(self):
        return self.layout.swizzle_block_shape(self.shape)

    def swizzle_data(self, data):
        if data.dtype != torch.int32:
            raise ValueError("sparse metadata requires int32 selector words")
        *batch, rows, words = self.shape
        data = torch.nn.functional.pad(data, (0, -words % 2, 0, -rows % 128), value=0x44444444)
        tiles = data.reshape(*batch, cdiv(rows, 128), 128, cdiv(words, 2), 2).transpose(-3, -2)
        return tiles.contiguous()

    def unswizzle_data(self, data):
        *batch, rows, words = self.shape
        tiles = data.transpose(-3, -2)
        return tiles.reshape(*batch, cdiv(rows, 128) * 128, cdiv(words, 2) * 2)[..., :rows, :words]
