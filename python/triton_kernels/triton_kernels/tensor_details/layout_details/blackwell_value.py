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
        print("orig data", data.shape, data.stride())

        major_dim = data.stride().index(1)
        minor_dim = major_dim - 1 if major_dim == data.ndim - 1 else major_dim + 1

        col_major = (major_dim, minor_dim) == (data.ndim - 2, data.ndim - 1)
        row_major = (minor_dim, major_dim) == (data.ndim - 2, data.ndim - 1)
        assert col_major or row_major

        align_to = lambda x, alignment: (x + alignment - 1) // alignment * alignment

        pad_major = align_to(data.shape[major_dim], 64) - data.shape[major_dim]
        pad_minor = align_to(data.shape[minor_dim], 2) - data.shape[minor_dim]

        padding = ()
        for i in range(data.ndim):
            if i == major_dim:
                padding += (0, pad_major)
            elif i == minor_dim:
                padding += (0, pad_minor)
            else:
                padding += (0, 0)
        data = torch.nn.functional.pad(data, padding)
        print("padded data", data.shape, data.stride())


        *leading_shape, R, C = data.shape

        if col_major:
            data = data.reshape(*leading_shape, R // 64, 64, C // 2, 2)
            data = data.transpose(-2, -3).transpose(-1, -2).transpose(-3, -4).flatten(-2, -1).reshape(*leading_shape, C // 2, R // 64, 2, 64).transpose(-1, -2)
        else:
            data = data.reshape(*leading_shape, R // 2, 2, C // 64, 64)
            data = data.transpose(-2, -3).flatten(-2, -1).reshape(*leading_shape, R // 2, C // 64, 2, 64)

        print("swiz data", data.shape, data.stride())
        return data

    def swizzle_data_old(self, data):
        # permutation needed to make `data` row major
        to_row_major = sorted(range(data.ndim), key=lambda d: (data.stride(d), d))[::-1]
        # permutation  needed to retrieve original order
        inv = [0] * data.ndim
        for i, d in enumerate(to_row_major):
            inv[d] = i
        # leading dimension must be padded to be aligned to 128
        align_dim = lambda x: (x + 128 - 1) // 128 * 128
        major_dim = data.stride().index(1)
        pad = align_dim(data.shape[major_dim]) - data.shape[major_dim]
        data = torch.nn.functional.pad(data.permute(to_row_major), (0, pad)).permute(inv)
        return data

    def unswizzle_data(self, data: torch.Tensor):
        assert data.ndim == len(self.shape) + 2, "Rank mismatch between data and recorded shape"
        # assert data.shape[-2:] == (2, 64) or data.shape[-2:] == (64, 2), "Expected dim -2 and -1 to be 2 and 64"

        # col_major = (data.shape[-2], data.shape[-2]) == (2, 64)
        # row_major = (data.shape[-2], data.shape[-2]) == (64, 2)
        # assert col_major or row_major
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

        # if col_major:
        #     *leading_shape, N2, K64, _, _ = data.shape
        #     data = data.transpose(-2, -3)
        # else:
        #     *leading_shape, N64, K2, _, _ = data.shape
        #     data = data.transpose(-2, -3)
        # data = data.flatten(-2, -1).flatten(-3, -1)

        # assert data.shape[-1] == 128, "Expected dim -1 to be 128"
        # assert data.shape[-2] == 2, "Expected dim -2 to be 128"
        # # *leading_shape, N2, K64, _, _ = data.shape
        # # data = data.reshape(*leading_shape, N2, K64, 2, 64)  # (*, N/2, K/64, 2, 64)
        # data = data.transpose(-2, -3)  # (*, N/2, 2, K/64, 64)
        # data = data.flatten(-2, -1).flatten(-3, -1)  # (*, N, K)
        # assert data.shape == self.shape,  "Mismatch between data and recorded shape"
        return data

    def unswizzle_data_old(self, data: torch.Tensor):
        # Trim padding along all dims back to the original shape recorded at init.
        assert data.ndim == len(self.shape), "Rank mismatch between data and recorded shape"
        sizes = [min(data.size(i), self.shape[i]) for i in range(data.ndim)]
        return data[tuple(slice(0, s) for s in sizes)]

    def swizzle_block_shape(self, block_shape):
        # print("block_shape", block_shape)
        *leading_shape, BLOCK_N, BLOCK_K = block_shape
        # *leading_shape, BLOCK_K, BLOCK_N = block_shape
        # print("swize block_shape", (*leading_shape, BLOCK_N // 2, BLOCK_K // 64, 2, 64))
        return (*leading_shape, BLOCK_N // 2, BLOCK_K // 64, 2, 64)

@triton.jit
def unswizzle_mx_value_bw(x):
    shape_0: tl.constexpr = x.shape[0]
    shape_1: tl.constexpr = x.shape[1]
    # tl.static_print("pre", x.shape)
    x = x.trans(0, 2, 1, 3).reshape(shape_0 * 2, shape_1 * 64)
    # tl.static_print("pos", x.shape)
    # tl.static_assert(x.shape[-1])
    # x = x.reshape()
    # tl.static_assert(shape_1 % SIZE_OUTER == 0)
    # tl.static_assert(shape_1 // SIZE_OUTER <= ALIGN_INNER)
    # x = x.reshape(shape_0, (shape_1 // SIZE_OUTER) // SIZE_INNER, 32, SIZE_OUTER // 32, SIZE_INNER)
    # x = x.trans(0, 3, 2, 1, 4).reshape(shape_0 * SIZE_OUTER, shape_1 // SIZE_OUTER)
    return x
