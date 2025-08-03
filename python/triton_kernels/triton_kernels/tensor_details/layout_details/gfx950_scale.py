import triton
import triton.language as tl
from .base import Layout


class GFX950MXScaleLayout(Layout):
    name: str = "GFX950_SCALE"

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def swizzle_data(self, data):
        data = data.transpose(-1, -2).contiguous()
        E, M, SCALE_K = data.shape
        data = data.view(E, M // 32, 2, 16, SCALE_K // 8, 2, 4, 1)
        data = data.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
        data = data.reshape(E, M // 32, SCALE_K * 32)
        return data.transpose(-1, -2)

    def unswizzle_data(self, data):
        raise NotImplementedError()

    def swizzle_block_shape(self, block_shape):
        E, SCALE_K, M = block_shape
        return [E, M // 32, SCALE_K * 32]


@triton.jit
def unswizzle_mx_scale_gfx950(x, BLOCK_N: tl.constexpr, MX_SCALE_BLOCK_K: tl.constexpr):
    NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32

    x = x.reshape(BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x
