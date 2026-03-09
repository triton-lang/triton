import math
import torch
from dataclasses import dataclass
import triton
import triton.language as tl
from .base import Layout, LayoutTransformation

NON_K_PRESHUFFLE_BLOCK_SIZE = 128


@dataclass(frozen=True)
class GFX1250MXScaleLayout(Layout):

    @property
    def name(self):
        return "GFX1250_SCALE"

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return GFX1250MXScaleLayoutTransformation(shape, is_fp4)

    def swizzle_block_shape(self, block_shape):
        SCALE_K = block_shape[-2]
        N = block_shape[-1]
        return block_shape[:-2] + [N // NON_K_PRESHUFFLE_BLOCK_SIZE, SCALE_K * NON_K_PRESHUFFLE_BLOCK_SIZE]


@dataclass(frozen=True)
class GFX1250MXScaleLayoutTransformation(LayoutTransformation):

    def __post_init__(self) -> None:
        *leading_shape, K_SCALE, N = self.shape
        B = math.prod(leading_shape)
        ALIGN_K_SCALE = 4 if K_SCALE > 4 else K_SCALE
        ALIGN_N = NON_K_PRESHUFFLE_BLOCK_SIZE
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
        data = torch.nn.functional.pad(data, (0, self.N_pad - self.N, 0, self.K_SCALE_pad - self.K_SCALE))
        data = data.transpose(-1, -2)
        data = data.view(-1, self.N_pad // self.ALIGN_N, 4, self.ALIGN_N // 4, self.K_SCALE_pad // self.ALIGN_K_SCALE,
                         self.ALIGN_K_SCALE)
        data = data.permute(0, 1, 4, 3, 2, 5).contiguous()
        data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.K_SCALE_pad * self.ALIGN_N)
        return data.transpose(-1, -2)

    def unswizzle_data(self, data):
        data = data.transpose(-1, -2)
        data = data.view(-1, self.N_pad // self.ALIGN_N, self.K_SCALE_pad // self.ALIGN_K_SCALE, self.ALIGN_N // 4, 4,
                         self.ALIGN_K_SCALE)
        data = data.permute(0, 1, 4, 3, 2, 5)
        data = data.reshape(*self.leading_shape, self.N_pad, self.K_SCALE_pad)
        return data.transpose(-1, -2)[..., :self.K_SCALE, :self.N].contiguous()

    def swizzle_block_shape(self, block_shape):
        SCALE_K = block_shape[-2]
        N = block_shape[-1]
        return block_shape[:-2] + [N // self.ALIGN_N, SCALE_K * self.ALIGN_N]


@triton.jit
def unswizzle_mx_scale_gfx1250(x, BLOCK_N: tl.constexpr, MX_SCALE_BLOCK_K: tl.constexpr,
                               N_PRESHUFFLE_FACTOR: tl.constexpr = NON_K_PRESHUFFLE_BLOCK_SIZE):
    SCALE_KWIDTH: tl.constexpr = 4 if MX_SCALE_BLOCK_K >= 4 else MX_SCALE_BLOCK_K
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // SCALE_KWIDTH, N_PRESHUFFLE_FACTOR // 4, 4,
                  SCALE_KWIDTH)
    x = x.permute(0, 1, 4, 3, 2, 5)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x
