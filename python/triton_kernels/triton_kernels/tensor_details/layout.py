import math
import torch
from abc import ABC, abstractmethod


class Layout(ABC):

    def __init__(self, shape) -> None:
        self.initial_shape = shape

    @abstractmethod
    def swizzle_data(self, data):
        pass

    @abstractmethod
    def unswizzle_data(self, data):
        pass

    @abstractmethod
    def swizzle_block_shape(self, block_shape):
        pass


class DefaultLayout(Layout):
    name: str = None

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def swizzle_data(self, data):
        return data

    def unswizzle_data(self, data):
        return data

    def swizzle_block_shape(self, block_shape):
        return block_shape


class BlackwellMXScaleLayout(Layout):
    name: str = "BLACKWELL_SCALE"

    def __init__(self, shape) -> None:
        super().__init__(shape)
        *self.leading_shape, self.K, self.N, = shape
        self.B = math.prod(self.leading_shape)
        self.ALIGN_K = 8
        self.ALIGN_N = 128
        self.SWIZZLE_K = 4
        self.K_pad = (self.K + self.ALIGN_K - 1) // self.ALIGN_K * self.ALIGN_K
        self.N_pad = (self.N + self.ALIGN_N - 1) // self.ALIGN_N * self.ALIGN_N

    def swizzle_data(self, data):
        data = torch.nn.functional.pad(data, (0, self.N_pad - self.N, 0, self.K_pad - self.K))
        data = data.transpose(-1, -2).contiguous()
        data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.ALIGN_N // 32, 32, self.K_pad // self.SWIZZLE_K,
                            self.SWIZZLE_K)
        data = data.transpose(2, 4).contiguous()
        data = data.view(1, self.B * self.N_pad // 128, self.K_pad // 4, 2, 256)
        return data

    def unswizzle_data(self, data):
        data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.K_pad // self.SWIZZLE_K, 32, self.ALIGN_N // 32,
                            self.SWIZZLE_K)
        data = data.transpose(2, 4)
        data = data.reshape(*self.leading_shape, self.N_pad, self.K_pad)
        data = data.transpose(-1, -2)
        return data[..., :self.K, :self.N]

    def swizzle_block_shape(self, block_shape):
        MX_PACK_DIVISOR = 32
        MX_SCALE_BLOCK_K = block_shape[1] // MX_PACK_DIVISOR
        return [1, block_shape[0] // 128, MX_SCALE_BLOCK_K // 4, 2, 256]


class HopperMXScaleLayout(Layout):
    name: str = "HOPPER_SCALE"

    def __init__(self, shape, num_warps=8) -> None:
        assert num_warps & (num_warps - 1) == 0, "warps_n must be a power of 2"
        super().__init__(shape)
        self.num_warps = num_warps

    def swizzle_data(self, data):
        data = data.transpose(-1, -2).contiguous()
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
        data = data.transpose(-1, -2).contiguous()
        return data

    def unswizzle_data(self, data):
        data = data.transpose(-1, -2).contiguous()
        *batch, M, K = data.shape
        b = len(batch)
        data = data.reshape(*batch, M // self.num_warps, self.num_warps, K // 64, 2, 8, 2, 2)
        perm = [0, 3, 1, 6, 4, 2, 5]
        perm = list(range(b)) + [b + p for p in perm]
        data = data.permute(*perm)
        data = data.reshape(*batch, M * 32, K // 32)
        data = data.transpose(-1, -2).contiguous()
        return data

    def swizzle_block_shape(self, block_shape):
        return block_shape


class HopperMXValueLayout(Layout):
    name: str = "HOPPER_VALUE"

    def __init__(self, shape, op_idx=0, mma_version=3):
        super().__init__(shape)
        self.op_idx = op_idx
        self.mma_version = mma_version
        *self.leading_shape, self.K, self.N, = shape

    def swizzle_data(self, data):
        """
        Given a uint8 tensor of shape (*, M, K), returns a tensor of shape
        (*, M // 4, K * 4) such that:

        1) Groups contiguously all the elements owned by the same thread of 4
        mma tiles along the K axis. The following animation shows a similar
        grouping for 2 tiles along M and 2 tiles along K rather than 4 along K
        as done here:
        https://neuralmagic.com/wp-content/uploads/2024/10/animation_4.gif

        2) Moves the elements belonging to thread 4-7 to be contiguous with those
        from thread 0-3. This is done to get a full cache line when loading them
        from HBM.

        op_idx selects the lhs or rhs of the matmul.

        WARNING: Assumes that the matmul will be done in bf16 or fp16!
        Implementing it for fp8 is as easy as making the tile size (8, 8)
        """

        assert self.op_idx in (0, 1)
        batch = data.ndim - 2
        assert batch >= 0
        assert self.mma_version in (2, 3)
        # canonicalize dimensions
        if self.op_idx == 0:
            data = data.mT
        init_shape = data.shape

        # We are loading 8 bf16 elements per thread to use ld.global.v4
        # Every u8 represents 2 mxfp4 elements
        u8_kwidth = 8 // 2 if self.mma_version == 2 else 1

        # Pack the 4 // u8_kwidth subtiles of an mma into a u4x8
        contig = (1, u8_kwidth)
        scott_trick = (2, 1)
        threads = (4, 4)
        warp_tile = (2, 2)
        k_tile = (1, 4 // u8_kwidth)

        sizes = list(data.shape[:-2])
        pads = []
        # [rest, K, tile, threads] per dimension
        for i, (a, b, c, s, d) in enumerate(zip(k_tile, warp_tile, threads, scott_trick, contig)):
            pack = a * b * c * s * d
            size = data.shape[batch + i]
            pad = (pack - size % pack) % pack
            pads += [(0, pad)]
            sizes.append((size + pad) // pack)
            sizes += [a, b, c, s, d]

        pads = tuple(x for t in pads[::-1] for x in t)
        data = torch.nn.functional.pad(data, pads)
        init_shape = data.shape
        # 0: rest[0]
        # 1: k_tile[0]
        # 2: warp_tile[0]
        # 3: threads[0]
        # 4: scott_trick[0]
        # 5: contig[0]
        # 6: rest[1]
        # 7: k_tile[1]
        # 8: warp_tile[1]
        # 9: threads[1]
        # 10: scott_trick[1]
        # 11: contig[1]
        data = data.view(*sizes)
        # Want [rest[0], threads[0], rest[1], scott_trick[0], scott_trick[0], threads[1], contig[1], contig[0], k_tile[1], k_tile[0], warp_tile[1], warp_tile[0]]
        perm = [0, 3, 6, 10, 4, 9, 7, 1, 8, 2, 5, 11]
        perm = list(range(batch)) + [batch + p for p in perm]
        data = data.permute(*perm).contiguous()
        # These are views
        data = data.flatten(-10, -1)
        data = data.flatten(-3, -2)
        assert data.is_contiguous()
        assert data.shape[-2] == init_shape[-2] // 4
        assert data.shape[-1] == init_shape[-1] * 4
        # de-canonicalize
        if self.op_idx == 0:
            data = data.mT
        return data

    def unswizzle_data(self, data):
        if self.op_idx == 0:
            data = data.mT
        *batch, M, K = data.shape
        # We have two times the elements if we already upcasted to bfloat16
        mult = 2 if data.dtype == torch.bfloat16 else 1
        assert M % 4 == 0, "M must be divisible by 4"
        assert K % (4 * 8 * 2 * 2 * mult) == 0, f"K must be divisible by {4 * 8 * 2 * 2 * mult}"
        # We are loading 8 bf16 elements per thread to use ld.global.v4
        # Every u8 represents 2 mxfp4 elements
        u8_kwidth = 8 // 2 if self.mma_version == 2 else 1
        data = data.reshape(*batch, M // 4, 4, K // (4 * 8 * 2 * 2 * mult), 2, 4, 8 // u8_kwidth, 2, u8_kwidth * mult)
        b = len(batch)
        perm = [0, 6, 1, 3, 2, 5, 4, 7]
        perm = list(range(b)) + [b + p for p in perm]
        data = data.permute(*perm)
        data = data.reshape(*batch, M * 4, K // 4)
        if self.op_idx == 0:
            data = data.mT
        return data[..., :self.K, :self.N]

    def swizzle_block_shape(self, block_shape):
        return block_shape
