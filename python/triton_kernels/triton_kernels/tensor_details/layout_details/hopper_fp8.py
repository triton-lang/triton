"""MXFP4 storage layouts for Hopper FP8 matmul, with reference conversions."""
from dataclasses import dataclass
import torch
import triton
from .base import Layout, LayoutTransformation
from .strided import StridedLayoutTransformation
from .torch_utils import repack


@dataclass(frozen=True)
class HopperFP8MXValueLayout(Layout):
    """FP4 byte storage: 16x64 word-transposed LHS or 64x128 swizzled RHS.

    mx_axis=-1 selects LHS storage: eight consecutive K values per 32-bit
    word, with consecutive rows in each word plane. mx_axis=-2 selects
    RHS storage. Tiles are stored in outer-row, then K order.
    """
    mx_axis: int

    def __post_init__(self):
        assert self.mx_axis in (-1, -2)

    @property
    def name(self):
        return "HOPPER_FP8_MX_VALUE"

    def make_transformation(self, shape, is_fp4):
        assert is_fp4
        return _HopperFP8MXTransformation(shape, is_fp4, self.mx_axis, False)

    def swizzle_block_shape(self, block_shape):
        raise NotImplementedError("Use layout conversion before the reference matmul")


@dataclass(frozen=True)
class HopperFP8MXScaleLayout(Layout):
    """E8M0 bytes matching HopperFP8MXValueLayout, one scale per 32 K values.

    LHS tiles store [K / 32, rows]; RHS tiles store [rows, K / 32].
    """
    mx_axis: int

    def __post_init__(self):
        assert self.mx_axis in (-1, -2)

    @property
    def name(self):
        return "HOPPER_FP8_MX_SCALE"

    def make_transformation(self, shape, is_fp4):
        assert not is_fp4
        return _HopperFP8MXTransformation(shape, is_fp4, self.mx_axis, True)

    def swizzle_block_shape(self, block_shape):
        raise NotImplementedError("Use layout conversion before the reference matmul")


@dataclass(frozen=True)
class _HopperFP8MXTransformation(LayoutTransformation):
    mx_axis: int
    is_scale: bool

    @property
    def tile_shape(self):
        rows, k = (16, 64) if self.mx_axis == -1 else (64, 128)
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
        tm, tk = self.tile_shape
        return [*self.shape[:-2], triton.cdiv(rows, tm) * triton.cdiv(k, tk), tm * tk]

    def _encode(self, data, major_dim):
        assert data.dtype == torch.uint8
        data = repack(data, major_dim, self.mx_axis, self.is_fp4)
        if self.mx_axis == -2:
            data = data.mT
        rows, k = self.matrix_shape
        tm, tk = self.tile_shape
        mt, kt = triton.cdiv(rows, tm), triton.cdiv(k, tk)
        data = torch.nn.functional.pad(data, (0, kt * tk - k, 0, mt * tm - rows)).contiguous()
        if self.is_scale:
            data = data.reshape(-1, mt, tm, kt, tk).permute(0, 1, 3, 2, 4)
            if self.mx_axis == -1:
                data = data.transpose(-1, -2)
        elif self.mx_axis == -1:
            data = data.view(torch.int32).reshape(-1, mt, tm, kt, tk // 4).permute(0, 1, 3, 4, 2)
        else:
            data = data.view(torch.int16).reshape(-1, mt, 4, 2, 8, kt, 4, 2, 4)
            data = data.permute(0, 1, 5, 2, 4, 8, 6, 7, 3).contiguous()
            data = data.reshape(-1, mt, kt, 128, 2, 8).permute(0, 1, 2, 4, 3, 5)
        return data.contiguous().view(torch.uint8).reshape(self.storage_shape)

    def _decode(self, data, major_dim):
        assert data.dtype == torch.uint8
        rows, k = self.matrix_shape
        tm, tk = self.tile_shape
        mt, kt = triton.cdiv(rows, tm), triton.cdiv(k, tk)
        data = data.contiguous()
        if self.is_scale:
            if self.mx_axis == -1:
                data = data.reshape(-1, mt, kt, tk, tm).transpose(-1, -2)
            else:
                data = data.reshape(-1, mt, kt, tm, tk)
            data = data.permute(0, 1, 3, 2, 4)
        elif self.mx_axis == -1:
            data = data.view(torch.int32).reshape(-1, mt, kt, tk // 4, tm).permute(0, 1, 4, 2, 3)
        else:
            data = data.view(torch.int16).reshape(-1, mt, kt, 2, 128, 8).permute(0, 1, 2, 4, 3, 5)
            data = data.reshape(-1, mt, kt, 4, 8, 4, 4, 2, 2).permute(0, 1, 3, 8, 4, 2, 6, 7, 5)
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
