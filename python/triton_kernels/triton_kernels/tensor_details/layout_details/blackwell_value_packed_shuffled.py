import math
from dataclasses import dataclass

import torch

from .base import Layout, LayoutTransformation
from .torch_utils import repack


@dataclass(frozen=True)
class BlackwellMX4ValuePackedShuffledLayout(Layout):
    """
    Shuffled pair-packed Blackwell MX4 weight layout.

    The canonical FP4 value tensor is stored as [..., K, N/2]. This layout
    first repacks it to physical MX4 storage [E, K/2, N], then groups two
    consecutive K tiles into one p64 byte footprint per N row. For
    the default block_k=128 and block_n=512 the storage shape is:

        [E, ceil((K/2) / 128), ceil(N / 512), 512, 128]

    Each 16-byte group in the final dimension contains bytes [0, 8) for the
    first K tile and bytes [8, 16) for the second K tile. Columns are
    pre-xored with the 128B NVMMA shared-memory row phase so a dense TMA load
    into NVMMASharedLayout(swizzle_byte_width=128, fp4_padded=False) lands
    the bytes in physical p64 order in shared memory.
    """

    block_k: int = 128
    block_n: int = 512
    k_tiles_per_pair: int = 2
    swizzle_byte_width: int = 128

    def __post_init__(self):
        if self.k_tiles_per_pair != 2:
            raise ValueError("BlackwellMX4ValuePackedShuffledLayout currently supports exactly two K tiles per pair")
        if self.block_k <= 0 or self.block_k % 16 != 0:
            raise ValueError("block_k must be a positive multiple of 16")
        if self.block_k % 2 != 0:
            raise ValueError("block_k must be even for FP4 packing")
        if self.block_n <= 0 or self.block_n % 8 != 0:
            raise ValueError("block_n must be a positive multiple of 8 for 128B NVMMA row phases")
        if self.swizzle_byte_width != 128:
            raise ValueError("packed shuffled MX4 layout is defined for 128B NVMMA shared swizzle")

    @property
    def name(self):
        return "BLACKWELL_MX4_VALUE_PACKED_SHUFFLED"

    @property
    def packed_block_k(self):
        return self.block_k // 2

    @property
    def pair_k_packed(self):
        return self.k_tiles_per_pair * self.packed_block_k

    @property
    def p64_k_bytes(self):
        return self.pair_k_packed

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return BlackwellMX4ValuePackedShuffledTransformation(
            shape,
            is_fp4,
            block_k=self.block_k,
            block_n=self.block_n,
            k_tiles_per_pair=self.k_tiles_per_pair,
            swizzle_byte_width=self.swizzle_byte_width,
        )

    def swizzle_block_shape(self, block_shape):
        """Return the unhalved dense TMA block shape for one packed K pair.

        block_shape is expressed in logical FP4 elements as
        [1, 2 * block_k, block_n]. Existing FP4 tensor-descriptor builders
        halve the contiguous byte dimension after calling this method, producing
        the actual dense byte footprint [1, 1, 1, block_n, block_k].
        """
        if len(block_shape) != 3:
            raise ValueError(f"Expected 3D block_shape, got {len(block_shape)}D: {block_shape}")
        _, block_k, block_n = block_shape
        expected_k = self.k_tiles_per_pair * self.block_k
        if block_k != expected_k:
            raise ValueError(f"block_k={block_k} must describe one packed pair ({expected_k})")
        if block_n != self.block_n:
            raise ValueError(f"block_n={block_n} does not match layout block_n={self.block_n}")
        return [1, 1, 1, self.block_n, self.p64_k_bytes * 2]


@dataclass(frozen=True)
class BlackwellMX4ValuePackedShuffledTransformation(LayoutTransformation):
    block_k: int = 128
    block_n: int = 512
    k_tiles_per_pair: int = 2
    swizzle_byte_width: int = 128

    @property
    def packed_block_k(self):
        return self.block_k // 2

    @property
    def pair_k_packed(self):
        return self.k_tiles_per_pair * self.packed_block_k

    @property
    def p64_k_bytes(self):
        return self.pair_k_packed

    def _compute_params(self, e, k_packed, n):
        padded_k_packed = ((k_packed + self.pair_k_packed - 1) // self.pair_k_packed) * self.pair_k_packed
        num_k_pairs = padded_k_packed // self.pair_k_packed
        num_n_tiles = (n + self.block_n - 1) // self.block_n
        padded_n = num_n_tiles * self.block_n
        return padded_k_packed, padded_n, num_k_pairs, num_n_tiles

    def _canonical_to_physical(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fp4:
            raise ValueError("BlackwellMX4ValuePackedShuffledLayout only supports fp4 values")
        assert data.stride(-1) == 1
        out_shape = list(data.shape)
        out_shape[-1] *= 2
        out_shape[-2] //= 2
        out = torch.empty(out_shape, dtype=data.dtype, device=data.device)
        return repack(data, -1, -2, self.is_fp4, out=out)

    def _physical_to_canonical(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fp4:
            raise ValueError("BlackwellMX4ValuePackedShuffledLayout only supports fp4 values")
        out_shape = list(data.shape)
        out_shape[-2] *= 2
        out_shape[-1] //= 2
        out = torch.empty(out_shape, dtype=data.dtype, device=data.device)
        return repack(data, -2, -1, self.is_fp4, out=out)

    def _tma_column_indices(self, device):
        n_inner = torch.arange(self.block_n, device=device, dtype=torch.long)[:, None, None]
        tile = torch.arange(self.k_tiles_per_pair, device=device, dtype=torch.long)[None, :, None]
        k_byte = torch.arange(self.packed_block_k, device=device, dtype=torch.long)[None, None, :]

        group = k_byte // 8
        byte_in_group = k_byte % 8
        physical_col = 16 * group + 8 * tile + byte_in_group
        phase = (n_inner % 8) * 16
        return (physical_col ^ phase).reshape(self.block_n, self.p64_k_bytes)

    def swizzle_data(self, data: torch.Tensor) -> torch.Tensor:
        data = self._canonical_to_physical(data)
        leading_shape = data.shape[:-2]
        e = math.prod(leading_shape)
        k_packed, n = data.shape[-2:]
        data = data.reshape(e, k_packed, n)
        padded_k_packed, padded_n, num_k_pairs, num_n_tiles = self._compute_params(e, k_packed, n)

        if k_packed != padded_k_packed or n != padded_n:
            padded = torch.zeros((e, padded_k_packed, padded_n), dtype=data.dtype, device=data.device)
            padded[:, :k_packed, :n] = data
            data = padded

        tiles = data.reshape(
            e,
            num_k_pairs,
            self.k_tiles_per_pair,
            self.packed_block_k,
            num_n_tiles,
            self.block_n,
        )
        tiles = tiles.permute(0, 1, 4, 5, 2, 3).contiguous()
        tiles = tiles.reshape(e, num_k_pairs, num_n_tiles, self.block_n, self.p64_k_bytes)

        out = torch.zeros(
            (e, num_k_pairs, num_n_tiles, self.block_n, self.p64_k_bytes),
            dtype=data.dtype,
            device=data.device,
        )
        columns = self._tma_column_indices(data.device)
        index = columns[None, None, None, :, :].expand_as(tiles)
        out.scatter_(4, index, tiles)
        return out

    def unswizzle_data(self, data: torch.Tensor) -> torch.Tensor:
        leading_shape = self.shape[:-2]
        e = math.prod(leading_shape)
        orig_k_packed = self.shape[-2] // 2
        orig_n = self.shape[-1]
        padded_k_packed, padded_n, num_k_pairs, num_n_tiles = self._compute_params(e, orig_k_packed, orig_n)

        data = data.reshape(e, num_k_pairs, num_n_tiles, self.block_n, self.p64_k_bytes)
        columns = self._tma_column_indices(data.device)
        index = columns[None, None, None, :, :].expand_as(data)
        tiles = data.gather(4, index)
        tiles = tiles.reshape(
            e,
            num_k_pairs,
            num_n_tiles,
            self.block_n,
            self.k_tiles_per_pair,
            self.packed_block_k,
        )
        physical = tiles.permute(0, 1, 4, 5, 2, 3).contiguous()
        physical = physical.reshape(e, padded_k_packed, padded_n)
        physical = physical[:, :orig_k_packed, :orig_n].contiguous()
        canonical = self._physical_to_canonical(physical)
        if not leading_shape:
            return canonical.squeeze(0)
        return canonical.reshape(*leading_shape, canonical.shape[-2], canonical.shape[-1])
