from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from .base import Layout, LayoutTransformation

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

    def _maybe_mT(self, data):
        if self.mx_axis == len(self.leading_shape):
            return data.contiguous().mT
        return data

    def swizzle_data(self, data):
        assert data.shape == (*self.leading_shape, self.M, self.K)
        data = self._maybe_mT(data).contiguous()
        *batch, M, K = data.shape
        SWIZZLE_ALIGN_M = 2 * self.num_warps * 2 * 8
        SWIZZLE_ALIGN_K = 2
        pad_m = (SWIZZLE_ALIGN_M - (M % SWIZZLE_ALIGN_M)) % SWIZZLE_ALIGN_M
        pad_k = (SWIZZLE_ALIGN_K - (K % SWIZZLE_ALIGN_K)) % SWIZZLE_ALIGN_K
        data = torch.nn.functional.pad(data, (0, pad_k, 0, pad_m))
        *batch, M, K = data.shape
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
        return data

    def unswizzle_data(self, data):
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
