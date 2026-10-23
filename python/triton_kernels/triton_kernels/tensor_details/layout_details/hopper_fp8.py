"""MXFP4 storage layouts for Hopper FP8 matmul, with reference conversions."""
from dataclasses import dataclass
import torch
import triton
from .base import Layout, LayoutTransformation
from .strided import StridedLayoutTransformation
from .torch_utils import repack


def _resolve_tile_shape(mx_axis, tile_shape):
    assert mx_axis in (-1, -2)
    if tile_shape is None:
        tile_shape = (16, 64) if mx_axis == -1 else (16, 128)
    rows, k = tile_shape
    assert rows in (8, 16, 32, 64) and k in (32, 64, 128, 256)
    return rows, k


@dataclass(frozen=True)
class HopperFP8MXValueLayout(Layout):
    """Word-transposed FP4 storage with tiles specified as (outer rows, K).

    Each word packs eight consecutive K values; rows are contiguous within
    each word plane. Tiles are stored in outer-row, then K order. Defaults
    are (16, 64) for mx_axis=-1 and (16, 128) for mx_axis=-2.
    """
    mx_axis: int
    tile_shape: tuple[int, int] | None = None

    def __post_init__(self):
        object.__setattr__(self, "tile_shape", _resolve_tile_shape(self.mx_axis, self.tile_shape))

    @property
    def name(self):
        return "HOPPER_FP8_MX_VALUE"

    def make_transformation(self, shape, is_fp4):
        assert is_fp4
        return _HopperFP8MXTransformation(shape, is_fp4, self.mx_axis, False, self.tile_shape)

    def swizzle_block_shape(self, block_shape):
        raise NotImplementedError("Use layout conversion before the reference matmul")


@dataclass(frozen=True)
class HopperFP8MXScaleLayout(Layout):
    """E8M0 bytes matching HopperFP8MXValueLayout, one scale per 32 K values.

    tile_shape uses logical value dimensions, with the same axis-dependent
    defaults as the value layout. Each tile stores scales in [K / 32, rows] order.
    """
    mx_axis: int
    tile_shape: tuple[int, int] | None = None

    def __post_init__(self):
        object.__setattr__(self, "tile_shape", _resolve_tile_shape(self.mx_axis, self.tile_shape))

    @property
    def name(self):
        return "HOPPER_FP8_MX_SCALE"

    def make_transformation(self, shape, is_fp4):
        assert not is_fp4
        return _HopperFP8MXTransformation(shape, is_fp4, self.mx_axis, True, self.tile_shape)

    def swizzle_block_shape(self, block_shape):
        raise NotImplementedError("Use layout conversion before the reference matmul")


@dataclass(frozen=True)
class _HopperFP8MXTransformation(LayoutTransformation):
    mx_axis: int
    is_scale: bool
    tile_shape: tuple[int, int]

    @property
    def packed_tile_shape(self):
        rows, k = self.tile_shape
        return rows, k // (32 if self.is_scale else 2)

    @property
    def matrix_shape(self):
        rows, k = self.shape[-2:]
        if self.mx_axis == -2:
            rows, k = k, rows
        if self.is_fp4:
            assert k % 32 == 0
            k //= 2
        return rows, k

    @property
    def storage_shape(self):
        rows, k = self.matrix_shape
        tm, tk = self.packed_tile_shape
        return [*self.shape[:-2], triton.cdiv(rows, tm) * triton.cdiv(k, tk), tm * tk]

    def _encode(self, data, major_dim):
        assert data.dtype == torch.uint8
        data = repack(data, major_dim, self.mx_axis, self.is_fp4)
        if self.mx_axis == -2:
            data = data.mT
        rows, k = self.matrix_shape
        tm, tk = self.packed_tile_shape
        mt, kt = triton.cdiv(rows, tm), triton.cdiv(k, tk)
        data = torch.nn.functional.pad(data, (0, kt * tk - k, 0, mt * tm - rows)).contiguous()
        if self.is_scale:
            data = data.reshape(-1, mt, tm, kt, tk).permute(0, 1, 3, 4, 2)
        else:
            data = data.view(torch.int32).reshape(-1, mt, tm, kt, tk // 4).permute(0, 1, 3, 4, 2)
        return data.contiguous().view(torch.uint8).reshape(self.storage_shape)

    def _decode(self, data, major_dim):
        assert data.dtype == torch.uint8
        rows, k = self.matrix_shape
        tm, tk = self.packed_tile_shape
        mt, kt = triton.cdiv(rows, tm), triton.cdiv(k, tk)
        data = data.contiguous()
        if self.is_scale:
            data = data.reshape(-1, mt, kt, tk, tm).permute(0, 1, 4, 2, 3)
        else:
            data = data.view(torch.int32).reshape(-1, mt, kt, tk // 4, tm).permute(0, 1, 4, 2, 3)
        data = data.contiguous().view(torch.uint8).reshape(*self.shape[:-2], mt * tm, kt * tk)
        data = data[..., :rows, :k]
        if self.mx_axis == -2:
            data = data.mT
        return repack(data, self.mx_axis, major_dim, self.is_fp4)

    def swizzle_data(self, data):
        return self._encode(data, -1)

    def unswizzle_data(self, data):
        return self._decode(data, -1)

    def _convert_data_from(self, data, source, *, out):
        if not isinstance(source, StridedLayoutTransformation):
            return super()._convert_data_from(data, source, out=out)
        result = self._encode(data, source.order[0])
        return result if out is None else out.copy_(result)

    def convert_data(self, data, destination, *, out=None):
        if not isinstance(destination, StridedLayoutTransformation):
            return super().convert_data(data, destination, out=out)
        result = self._decode(data, destination.order[0])
        if out is None:
            out = torch.empty_strided(destination.storage_shape, destination.storage_strides, dtype=data.dtype,
                                      device=data.device)
        return out.copy_(result)
