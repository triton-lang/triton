from dataclasses import dataclass

import torch
from triton import cdiv

from .base import Layout, LayoutTransformation


@dataclass(frozen=True)
class SparseMetadataLayout(Layout):
    """Selector tiles for sparse matmul. FP16 uses one word per row, FP8 and paired NVFP4 use two."""
    words_per_tile: int

    def __post_init__(self):
        if self.words_per_tile not in (1, 2):
            raise ValueError("expected one or two selector words per tile row")

    name = "SPARSE_METADATA"

    def swizzle_block_shape(self, block_shape):
        *batch, rows, words = block_shape
        return [*batch, cdiv(rows, 128), cdiv(words, self.words_per_tile), 128, self.words_per_tile]

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

    def _interleave_fp16(self, tiles):
        # Transpose the two half-words of row pairs eight apart. This is its own inverse.
        if self.layout.words_per_tile == 2:
            return tiles
        rows = torch.arange(128, device=tiles.device)
        shift = ((rows & 8) * 2).int()[:, None]
        lo = tiles[..., rows & ~8, :] >> shift
        hi = tiles[..., rows | 8, :] >> shift
        return (lo & 65535) | (hi << 16)

    def swizzle_data(self, data):
        if data.dtype != torch.int32:
            raise ValueError("sparse metadata requires int32 selector words")
        *batch, rows, words = self.shape
        w = self.layout.words_per_tile
        data = torch.nn.functional.pad(data, (0, -words % w, 0, -rows % 128), value=0x44444444)
        tiles = data.reshape(*batch, cdiv(rows, 128), 128, cdiv(words, w), w).transpose(-3, -2)
        return self._interleave_fp16(tiles).contiguous()

    def unswizzle_data(self, data):
        *batch, rows, words = self.shape
        w = self.layout.words_per_tile
        tiles = self._interleave_fp16(data).transpose(-3, -2)
        return tiles.reshape(*batch, cdiv(rows, 128) * 128, cdiv(words, w) * w)[..., :rows, :words]
