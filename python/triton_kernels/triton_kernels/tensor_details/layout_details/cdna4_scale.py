import math
import torch
from dataclasses import dataclass
import triton
import triton.language as tl
from .base import Layout

NON_K_PRESHUFFLE_BLOCK_SIZE = 32


@dataclass
class CDNA4MXScaleLayout(Layout):
    name: str = "CDNA4_SCALE"

    def __init__(self, shape) -> None:
        super().__init__(shape)
        (
            *self.leading_shape,
            self.K_SCALE,
            self.N,
        ) = shape
        self.B = math.prod(self.leading_shape)
        self.ALIGN_K_SCALE = 8
        self.ALIGN_N = 32
        self.K_SCALE_pad = math.ceil(self.K_SCALE / self.ALIGN_K_SCALE) * self.ALIGN_K_SCALE
        self.N_pad = math.ceil(self.N / self.ALIGN_N) * self.ALIGN_N

    def swizzle_data(self, data):
        data = torch.nn.functional.pad(data, (0, self.N_pad - self.N, 0, self.K_SCALE_pad - self.K_SCALE))
        data = data.transpose(-1, -2)
        data = data.view(-1, self.N_pad // NON_K_PRESHUFFLE_BLOCK_SIZE, 2, 16, self.K_SCALE_pad // 8, 2, 4, 1)
        data = data.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
        data = data.reshape(self.B, self.N_pad // 32, self.K_SCALE_pad * 32)
        return data.transpose(-1, -2)

    def unswizzle_data(self, data):
        data = data.transpose(-1, -2)
        data = data.view(-1, self.N_pad // NON_K_PRESHUFFLE_BLOCK_SIZE, self.K_SCALE_pad // 8, 4, 16, 2, 2, 1)
        data = data.permute(0, 1, 6, 4, 2, 5, 3, 7)
        data = data.reshape(*self.leading_shape, self.N_pad, self.K_SCALE_pad)
        return data.transpose(-1, -2)[..., :self.K_SCALE, :self.N]

    def swizzle_block_shape(self, block_shape):
        SCALE_K = block_shape[-2]
        N = block_shape[-1]
        return block_shape[:-2] + [N // 32, SCALE_K * 32]


@triton.jit
def unswizzle_mx_scale_cdna4(x, BLOCK_N: tl.constexpr, MX_SCALE_BLOCK_K: tl.constexpr,
                             N_PRESHUFFLE_FACTOR: tl.constexpr = NON_K_PRESHUFFLE_BLOCK_SIZE):
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x
