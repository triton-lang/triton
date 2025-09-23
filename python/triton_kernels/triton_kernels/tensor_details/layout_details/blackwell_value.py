import triton
import triton.language as tl
import torch
from .base import Layout


class BlackwellMXValueLayout(Layout):
    name: str = "BLACKWELL_VALUE"

    def __init__(self, shape) -> None:
        super().__init__(shape)
        self.shape = shape

    def swizzle_data(self, data):
        assert data.shape == self.shape, "Mismatch between data and recorded shape"

        major_dim = data.stride().index(1)
        minor_dim = major_dim - 1 if major_dim == data.ndim - 1 else major_dim + 1

        col_major = (major_dim, minor_dim) == (data.ndim - 2, data.ndim - 1)
        row_major = (minor_dim, major_dim) == (data.ndim - 2, data.ndim - 1)
        assert col_major or row_major

        align_to = lambda x, alignment: (x + alignment - 1) // alignment * alignment

        pad_major = align_to(data.shape[major_dim], 64) - data.shape[major_dim]
        pad_minor = align_to(data.shape[minor_dim], 2) - data.shape[minor_dim]

        padding = []
        for dim in reversed(range(min(major_dim, minor_dim), data.ndim)):
            if dim == major_dim:
                padding.extend((0, pad_major))
            elif dim == minor_dim:
                padding.extend((0, pad_minor))
            else:
                padding.extend((0, 0))
        data = torch.nn.functional.pad(data, tuple(padding))

        *leading_shape, R, C = data.shape
        leading_dims = range(data.ndim - 2)

        if col_major:
            data = data.reshape(*leading_shape, R // 64, 64, C // 2, 2)
            data = data.permute(*leading_dims, -2, -4, -1, -3)
            data = data.flatten(-2, -1)
            data = data.reshape(*leading_shape, C // 2, R // 64, 2, 64)
            data = data.transpose(-1, -2)
        else:
            data = data.reshape(*leading_shape, R // 2, 2, C // 64, 64)
            data = data.transpose(-2, -3).flatten(-2, -1).reshape(*leading_shape, R // 2, C // 64, 2, 64)

        return data

    def unswizzle_data(self, data: torch.Tensor):
        assert data.ndim == len(self.shape) + 2, "Rank mismatch between data and recorded shape"
        transpose = data.stride(-1) != 1

        if transpose:
            *leading_shape, C2, R64, a, b = data.shape
            assert (a, b) == (64, 2)
            data = data.transpose(-3, -4)
        else:
            *leading_shape, R2, C64, a, b = data.shape
            assert (a, b) == (2, 64)
        data = data.transpose(-2, -3)
        data = data.flatten(-4, -3).flatten(-2, -1)

        return data

    def swizzle_block_shape(self, block_shape):
        *leading_shape, BLOCK_N, BLOCK_K = block_shape
        return (*leading_shape, BLOCK_N // 2, BLOCK_K // 64, 2, 64)

@triton.jit
def unswizzle_mx_value_bw(x):
    shape_0: tl.constexpr = x.shape[0]
    shape_1: tl.constexpr = x.shape[1]
    tl.static_assert(x.shape[1] == 1, "unswizzle_mx_value_bw requires shape[1] == 1")
    x = x.reshape(shape_0 * 2, shape_1 * 64)
    return x
