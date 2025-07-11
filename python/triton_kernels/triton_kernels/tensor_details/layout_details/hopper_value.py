import torch
import triton
import triton.language as tl
from .base import Layout


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


@triton.jit
def unswizzle_mxfp4_value_hopper(x, op_idx: tl.constexpr, mma_version: tl.constexpr):
    """
    Triton inverse of swizzle_mxfp4_value_hopper
    """
    tl.static_assert(op_idx == 0 or op_idx == 1, "op_idx must be 0 or 1")
    tl.static_assert(len(x.shape) == 2, "NYI")
    tl.static_assert(mma_version == 2 or mma_version == 3, "mma_version must be 2 or 3")
    if op_idx == 1:
        x = x.trans()

    # We have two times the elements if we already upcasted to bfloat16
    mult: tl.constexpr = 2 if x.dtype == tl.bfloat16 else 1
    M: tl.constexpr = x.shape[0]
    K: tl.constexpr = x.shape[1]
    tl.static_assert(M % 4 == 0, "M must be divisible by 4")
    tl.static_assert(K % (4 * 8 * 2 * 2 * mult) == 0, f"K must be divisible by {4 * 8 * 2 * 2 * mult}")

    # We are loading 8 bf16 elements per thread to use ld.global.v4
    # Every u8 represents 2 mxfp4 elements
    u8_kwidth: tl.constexpr = 8 // 2 if mma_version == 2 else 1
    x = x.reshape(M // 4, 4, K // (4 * 8 * 2 * 2 * mult), 2, 4, 8 // u8_kwidth, 2, u8_kwidth * mult)
    x = x.trans(0, 6, 1, 3, 2, 5, 4, 7)
    x = x.reshape(M * 4, K // 4)
    if op_idx == 1:
        x = x.trans()
    return x
