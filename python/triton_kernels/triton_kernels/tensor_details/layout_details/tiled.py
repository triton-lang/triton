from dataclasses import dataclass

from .base import Layout, LayoutTransformation


@dataclass(frozen=True)
class TiledLayout(Layout):
    major_dim: int = -1
    tile_size: int = 128

    @property
    def name(self):
        return "TILED"

    def swizzle_block_shape(self, block_shape):
        return block_shape

    def make_transformation(self, shape, is_fp4):
        return TiledLayoutTransformation(shape, is_fp4, self.major_dim, self.tile_size)


@dataclass(frozen=True)
class TiledLayoutTransformation(LayoutTransformation):
    major_dim: int
    tile_size: int

    def __post_init__(self):
        if self.is_fp4 or len(self.shape) not in (2, 3):
            raise ValueError("Tiled layout requires a matrix of byte-sized or larger elements")
        if self.major_dim % len(self.shape) not in (len(self.shape) - 1, len(self.shape) - 2):
            raise ValueError("Only the last two dimensions can be tiled")
        if self.tile_size <= 0 or any(size % self.tile_size for size in self.shape[-2:]):
            raise ValueError("Matrix dimensions must be divisible by the tile size")

    @property
    def storage_shape(self):
        return list(self.shape)

    def swizzle_data(self, data):
        transpose = self.major_dim % data.ndim == data.ndim - 2
        matrix = data.mT if transpose else data
        *batch, rows, columns = matrix.shape
        tile = self.tile_size
        result = matrix.reshape(*batch, rows // tile, tile, columns // tile, tile)
        result = result.transpose(-3, -2).contiguous().reshape(matrix.shape)
        return result.mT if transpose else result

    def unswizzle_data(self, data):
        transpose = self.major_dim % data.ndim == data.ndim - 2
        matrix = data.mT if transpose else data
        *batch, rows, columns = matrix.shape
        tile = self.tile_size
        result = matrix.reshape(*batch, rows // tile, columns // tile, tile, tile)
        result = result.transpose(-3, -2).contiguous().reshape(matrix.shape)
        return result.mT if transpose else result
