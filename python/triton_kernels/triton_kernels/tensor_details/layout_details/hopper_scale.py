import math
from dataclasses import dataclass
import torch
from torch._subclasses.fake_tensor import is_fake
import triton
import triton.language as tl
from .base import Layout, LayoutTransformation
from .strided import StridedLayoutTransformation

# ------------------- Hopper MX Scale Layout -------------------


@dataclass(frozen=True)
class HopperMXScaleLayout(Layout):
    mx_axis: int
    num_warps: int

    def __post_init__(self):
        assert self.num_warps & (self.num_warps - 1) == 0, "warps_n must be a power of 2"

    @property
    def name(self):
        return "HOPPER_SCALE"

    def make_transformation(self, shape: list[int], is_fp4) -> LayoutTransformation:
        return HopperMXScaleLayoutTransformation(shape, is_fp4, self.mx_axis, self.num_warps)

    def swizzle_block_shape(self, block_shape):
        # wrong ? this seems like a transposition
        if self.mx_axis == -1:
            *head, N, K = block_shape
            assert N % 32 == 0, N
            return [*head, N // 32, K * 32]
        else:
            assert self.mx_axis == -2
            *head, K, N = block_shape
            assert N % 32 == 0, N
            return [*head, K * 32, N // 32]


# ------------------- Hopper MX Scale Layout Transformation -------------------


@dataclass(frozen=True)
class HopperMXScaleLayoutTransformation(LayoutTransformation):
    mx_axis: int
    num_warps: int

    def __post_init__(self):
        *leading_shape, M, K = self.shape
        if self.mx_axis < 0:
            object.__setattr__(self, "mx_axis", self.mx_axis + len(self.shape))
        object.__setattr__(self, "leading_shape", leading_shape)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "K", K)

    @property
    def _padded_shape(self) -> list[int]:
        *leading_shape, M, K = self.shape
        if self.mx_axis == len(leading_shape):
            M, K = K, M
        align_m = 32 * self.num_warps
        M = (M + align_m - 1) // align_m * align_m
        K = (K + 1) // 2 * 2
        return [*leading_shape, M, K]

    @property
    def storage_shape(self) -> list[int]:
        *leading_shape, M, K = self._padded_shape
        if self.mx_axis == len(leading_shape):
            return [*leading_shape, K * 32, M // 32]
        return [*leading_shape, M // 32, K * 32]

    def _maybe_mT(self, data):
        if self.mx_axis == len(self.leading_shape):
            return data.contiguous().mT
        return data

    def _can_convert(self, data):
        return not self.is_fp4 and data.device.type == "cuda" and data.dtype.itemsize == 1 and not is_fake(data)

    def convert_data(self, data, destination: LayoutTransformation, *, out=None):
        if isinstance(destination, StridedLayoutTransformation) and self._can_convert(data):
            if out is None:
                out = torch.empty_strided(destination.storage_shape, destination.storage_strides, dtype=data.dtype,
                                          device=data.device)
            return self._convert(data, out, inverse=True)
        return super().convert_data(data, destination, out=out)

    def _convert_data_from(self, data, source: LayoutTransformation, *, out):
        if isinstance(source, StridedLayoutTransformation) and self._can_convert(data):
            return self._convert(data, out, inverse=False)
        return super()._convert_data_from(data, source, out=out)

    def _convert(self, data, out, inverse):
        transpose = self.mx_axis == len(self.leading_shape)
        if out is None:
            if inverse:
                out = torch.empty(self.shape, dtype=data.dtype, device=data.device)
            else:
                shape = self.storage_shape
                if transpose:
                    shape[-2:] = reversed(shape[-2:])
                out = torch.empty(shape, dtype=data.dtype, device=data.device)
                if transpose:
                    out = out.mT
        matrix, encoded = (out, data) if inverse else (data, out)
        if transpose:
            matrix, encoded = matrix.mT, encoded.mT
        *_, m_pad, k_pad = self._padded_shape
        large = math.prod(self.leading_shape) * m_pad * k_pad >= 1024 * 1024
        block_m = 32 * self.num_warps
        block_k = min(64 if large else 32, max(2, triton.next_power_of_2(k_pad)))
        num_warps = 8 if large else 4
        grid = (math.prod(self.leading_shape) * triton.cdiv(m_pad, block_m) * triton.cdiv(k_pad, block_k), )
        if grid[0]:
            with torch.cuda.device(data.device):
                _convert_scale_kernel[grid](matrix, encoded, tuple(self.leading_shape), matrix.stride(),
                                            encoded.stride(), matrix.shape[-2], matrix.shape[-1], m_pad, k_pad,
                                            self.num_warps, inverse, block_m, block_k, num_warps=num_warps)
        return out

    def swizzle_data(self, data):
        if self._can_convert(data):
            return self._convert(data, None, inverse=False)
        assert data.shape == (*self.leading_shape, self.M, self.K)
        data = self._maybe_mT(data).contiguous()
        *batch, M_in, K_in = data.shape
        *_, M, K = self._padded_shape
        pad_m = M - M_in
        pad_k = K - K_in
        data = torch.nn.functional.pad(data, (0, pad_k, 0, pad_m))
        assert data.is_contiguous()
        assert M % (
            2 * self.num_warps * 2 *
            8) == 0 and K % 2 == 0, f"Input tensor must have a subtile of shape (..., {2 * self.num_warps * 2 * 8}, 2)"
        b = len(batch)
        data = data.reshape(*batch, M // (2 * self.num_warps * 2 * 8), 2, self.num_warps, 2, 8, K // 2, 2)
        perm = [0, 2, 5, 1, 4, 6, 3]
        perm = list(range(b)) + [b + p for p in perm]
        data = data.permute(*perm)
        data = data.flatten(-5, -1)
        data = data.flatten(-3, -2)
        assert data.shape[-2] == M // 32
        assert data.shape[-1] == K * 32
        data = self._maybe_mT(data)
        return self._validate_storage_shape(data)

    def unswizzle_data(self, data):
        if self._can_convert(data):
            return self._convert(data, None, inverse=True)
        data = self._maybe_mT(data)
        *batch, M, K = data.shape
        b = len(batch)
        data = data.reshape(*batch, M // self.num_warps, self.num_warps, K // 64, 2, 8, 2, 2)
        perm = [0, 3, 1, 6, 4, 2, 5]
        perm = list(range(b)) + [b + p for p in perm]
        data = data.permute(*perm)
        data = data.reshape(*batch, M * 32, K // 32)
        data = self._maybe_mT(data)
        data = data[..., :self.M, :self.K]
        data = data.contiguous()
        return data


@triton.jit
def unswizzle_mxfp4_scale_hopper(x, mx_axis: tl.constexpr, num_warps: tl.constexpr):
    """
    Triton inverse of swizzle_mxfp4_scale_hopper
    """
    if mx_axis is not None and mx_axis < 0:
        mx_axis += len(x.shape)
    tl.static_assert(len(x.shape) == 2, "NYI")
    # implementation assumes mxfp data is packed along the last dimension
    x = x.trans() if mx_axis == 0 else x
    M: tl.constexpr = x.shape[0]
    K: tl.constexpr = x.shape[1]
    tl.static_assert(M % num_warps == 0, f"M must be divisible by {num_warps}. Got {M}")
    tl.static_assert(K % 64 == 0, f"K must be divisible by 64. Got {K}")
    x = x.reshape(M // num_warps, num_warps, K // 64, 2, 8, 2, 2)
    x = x.trans(0, 3, 1, 6, 4, 2, 5)
    x = x.reshape(M * 32, K // 32)
    # implementation assumed mxfp data is packed along the last dimension
    x = x.trans() if mx_axis == 0 else x
    return x


@triton.jit
def _convert_scale_kernel(Matrix, Encoded, LEADING_SHAPE: tl.constexpr, MATRIX_STRIDES: tl.constexpr,
                          ENCODED_STRIDES: tl.constexpr, M: tl.constexpr, K: tl.constexpr, M_PAD: tl.constexpr,
                          K_PAD: tl.constexpr, LAYOUT_WARPS: tl.constexpr, INVERSE: tl.constexpr, BLOCK_M: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    Matrix = Matrix.to(tl.pointer_type(tl.uint8))
    Encoded = Encoded.to(tl.pointer_type(tl.uint8))
    tiles_m: tl.constexpr = triton.cdiv(M_PAD, BLOCK_M)
    tiles_k: tl.constexpr = triton.cdiv(K_PAD, BLOCK_K)
    pid = tl.program_id(0).to(tl.int64)
    batch = pid // (tiles_m * tiles_k)
    tile_m, tile_k = pid // tiles_k % tiles_m, pid % tiles_k
    for dim in tl.static_range(len(LEADING_SHAPE) - 1, -1, -1):
        index = batch % LEADING_SHAPE[dim]
        batch //= LEADING_SHAPE[dim]
        Matrix += index * MATRIX_STRIDES[dim]
        Encoded += index * ENCODED_STRIDES[dim]
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    matrix_offsets = rows[:, None] * MATRIX_STRIDES[-2] + cols[None, :] * MATRIX_STRIDES[-1]
    matrix_mask = (rows[:, None] < M) & (cols[None, :] < K)
    enc_rows = tile_m * (BLOCK_M // 32) + tl.arange(0, BLOCK_M // 32)
    enc_cols = tile_k * (BLOCK_K * 32) + tl.arange(0, BLOCK_K * 32)
    enc_offsets = enc_rows[:, None] * ENCODED_STRIDES[-2] + enc_cols[None, :] * ENCODED_STRIDES[-1]
    enc_mask = (enc_rows[:, None] < M_PAD // 32) & (enc_cols[None, :] < K_PAD * 32)
    if INVERSE:
        x = tl.load(Encoded + enc_offsets, enc_mask, other=0)
        x = x.reshape(BLOCK_M // (32 * LAYOUT_WARPS), LAYOUT_WARPS, BLOCK_K // 2, 2, 8, 2, 2)
        x = x.trans(0, 3, 1, 6, 4, 2, 5).reshape(BLOCK_M, BLOCK_K)
        tl.store(Matrix + matrix_offsets, x, matrix_mask)
    else:
        x = tl.load(Matrix + matrix_offsets, matrix_mask, other=0)
        x = x.reshape(BLOCK_M // (32 * LAYOUT_WARPS), 2, LAYOUT_WARPS, 2, 8, BLOCK_K // 2, 2)
        x = x.trans(0, 2, 5, 1, 4, 6, 3).reshape(BLOCK_M // 32, BLOCK_K * 32)
        tl.store(Encoded + enc_offsets, x, enc_mask)
