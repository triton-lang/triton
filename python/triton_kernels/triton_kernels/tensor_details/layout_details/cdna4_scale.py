import math
import torch
from dataclasses import dataclass
import triton
import triton.language as tl
from .base import Layout, LayoutTransformation
from .torch_utils import repack


# ------------------- CDNA4 MX Scale Layout -------------------
@dataclass(frozen=True)
class CDNA4MXScaleLayout(Layout):

    @property
    def name(self):
        return "CDNA4_SCALE"

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return CDNA4MXScaleLayoutTransformation(shape, is_fp4)

    def swizzle_block_shape(self, block_shape):
        SCALE_K = block_shape[-2]
        N = block_shape[-1]
        return block_shape[:-2] + [N // 32, SCALE_K * 32]


# ------------------- CDNA4 MX Scale Layout Transformation -------------------

NON_K_PRESHUFFLE_BLOCK_SIZE = 32


@dataclass(frozen=True)
class CDNA4MXScaleLayoutTransformation(LayoutTransformation):

    def __post_init__(self) -> None:
        *leading_shape, K_SCALE, N = self.shape
        B = math.prod(leading_shape)
        ALIGN_K_SCALE = 8
        ALIGN_N = 32
        K_SCALE_pad = math.ceil(K_SCALE / ALIGN_K_SCALE) * ALIGN_K_SCALE
        N_pad = math.ceil(N / ALIGN_N) * ALIGN_N
        object.__setattr__(self, "leading_shape", leading_shape)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "ALIGN_K_SCALE", ALIGN_K_SCALE)
        object.__setattr__(self, "ALIGN_N", ALIGN_N)
        object.__setattr__(self, "K_SCALE_pad", K_SCALE_pad)
        object.__setattr__(self, "N_pad", N_pad)
        object.__setattr__(self, "K_SCALE", K_SCALE)
        object.__setattr__(self, "N", N)

    def swizzle_data(self, data):
        assert data.stride(-1) == 1
        # re-pack as column-major
        data = repack(data, -1, -2, self.is_fp4)
        data = data.mT.contiguous().mT
        data = torch.nn.functional.pad(data, (0, self.N_pad - self.N, 0, self.K_SCALE_pad - self.K_SCALE))
        data = data.transpose(-1, -2)
        data = data.view(-1, self.N_pad // NON_K_PRESHUFFLE_BLOCK_SIZE, 2, 16, self.K_SCALE_pad // 8, 2, 4, 1)
        data = data.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
        data = data.reshape(self.B, self.N_pad // 32, self.K_SCALE_pad * 32)
        data = data.transpose(-1, -2)
        assert data.stride(-2) == 1
        return data

    def unswizzle_data(self, data):
        data = data.transpose(-1, -2)
        data = data.view(-1, self.N_pad // NON_K_PRESHUFFLE_BLOCK_SIZE, self.K_SCALE_pad // 8, 4, 16, 2, 2, 1)
        data = data.permute(0, 1, 6, 4, 2, 5, 3, 7)
        data = data.reshape(*self.leading_shape, self.N_pad, self.K_SCALE_pad)
        data = data.transpose(-1, -2)[..., :self.K_SCALE, :self.N]
        data = repack(data, -2, -1, self.is_fp4)
        data = data.contiguous()
        assert data.stride(-1) == 1
        return data


@triton.jit
def unswizzle_mx_scale_cdna4(x, BLOCK_N: tl.constexpr, MX_SCALE_BLOCK_K: tl.constexpr,
                             N_PRESHUFFLE_FACTOR: tl.constexpr = NON_K_PRESHUFFLE_BLOCK_SIZE):
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x
