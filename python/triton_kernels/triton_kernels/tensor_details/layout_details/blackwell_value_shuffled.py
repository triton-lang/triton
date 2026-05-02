import math
from dataclasses import dataclass
import torch
from .base import Layout, LayoutTransformation
from .torch_utils import repack


# ------------------- Blackwell MX4 Value Shuffled Layout -------------------
@dataclass(frozen=True)
class BlackwellMX4ValueShuffledLayout(Layout):
    """
    Shuffled weight layout for MX4 matmul on Blackwell GPUs.

    Physical packed storage for mxfp4 is column-major with shape [E, K_packed, N],
    where K_packed = K // 2 (two FP4 values per byte).

    Baseline TMA loads operate on the swapped view [E, N, K_packed] with block
    shape [block_n, packed_block_k], then the kernel transposes to
    [packed_block_k, block_n].

    This layout pre-arranges those tiles so each tile is contiguous in memory,
    matching the baseline's post-transpose format [block_n, packed_block_k].
    We use a 5D layout:
    [E, num_tiles_k, num_tiles_n, tile_n, tile_k_packed]

    The inner dimensions [tile_n, tile_k_packed] match the baseline TMA block
    shape after swapping, so no transpose is needed after TMA load.
    """
    block_k: int = 128
    block_n: int = 256

    @property
    def name(self):
        return "BLACKWELL_MX4_VALUE_SHUFFLED"

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return BlackwellMX4ValueShuffledTransformation(shape, is_fp4, block_k=self.block_k, block_n=self.block_n)

    def swizzle_block_shape(self, block_shape):
        """
        Convert block shape for TMA descriptor.

        Logical block shape is [1, block_k, block_n]. For this layout we want
        TMA to load [1, 1, 1, tile_n, packed_block_k] from the shuffled buffer.
        This matches the inner dimensions of our 5D layout.
        """
        if len(block_shape) != 3:
            raise ValueError(f"Expected 3D block_shape, got {len(block_shape)}D: {block_shape}")
        _, block_k, block_n = block_shape
        if block_k != self.block_k:
            raise ValueError(f"block_k={block_k} does not match layout block_k={self.block_k}")
        # Return block_k un-halved; make_dense_tma will halve it for FP4 packing
        return [1, 1, 1, block_n, block_k]


# ------------------- Blackwell MX4 Value Shuffled Transformation -------------------
@dataclass(frozen=True)
class BlackwellMX4ValueShuffledTransformation(LayoutTransformation):
    """Transformation for the shuffled MX4 weight layout."""

    block_k: int = 128
    block_n: int = 256

    def _compute_params(self, E, K_packed, N):
        """Compute tiling parameters from the physical shape."""
        packed_block_k = self.block_k // 2
        tile_k_packed = packed_block_k
        tile_n = self.block_n

        # lcm(128, tile_k_packed): 128 is the TMA alignment requirement in bytes
        align_k = (128 * tile_k_packed) // math.gcd(128, tile_k_packed)
        padded_K_packed = ((K_packed + align_k - 1) // align_k) * align_k
        num_tiles_k = (padded_K_packed + tile_k_packed - 1) // tile_k_packed
        num_tiles_n = (N + tile_n - 1) // tile_n
        padded_N = num_tiles_n * tile_n

        return tile_k_packed, tile_n, padded_K_packed, padded_N, num_tiles_k, num_tiles_n

    def _canonical_to_physical(self, data: torch.Tensor) -> torch.Tensor:
        """Repack canonical [..., K, N_packed] storage to physical [E, K_packed, N]."""
        if not self.is_fp4:
            raise ValueError("BlackwellMX4ValueShuffledLayout only supports fp4 values")
        assert data.stride(-1) == 1
        out_shape = list(data.shape)
        out_shape[-1] *= 2
        out_shape[-2] //= 2
        out = torch.empty(out_shape, dtype=data.dtype, device=data.device)
        return repack(data, -1, -2, self.is_fp4, out=out)

    def _physical_to_canonical(self, data: torch.Tensor) -> torch.Tensor:
        """Repack physical [E, K_packed, N] storage to canonical [..., K, N_packed]."""
        if not self.is_fp4:
            raise ValueError("BlackwellMX4ValueShuffledLayout only supports fp4 values")
        out_shape = list(data.shape)
        out_shape[-2] *= 2
        out_shape[-1] //= 2
        out = torch.empty(out_shape, dtype=data.dtype, device=data.device)
        return repack(data, -2, -1, self.is_fp4, out=out)

    def swizzle_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Convert data from canonical [..., K, N_packed] to 5D shuffled layout.

        Target layout: [E, num_tiles_k, num_tiles_n, tile_n, tile_k_packed]
        This matches the baseline TMA block shape [block_n, packed_block_k] after swapping.
        """
        data = self._canonical_to_physical(data)
        leading_shape = data.shape[:-2]
        E = math.prod(leading_shape)
        K_packed, N = data.shape[-2:]
        data = data.reshape(E, K_packed, N)
        tile_k_packed, tile_n, padded_K_packed, padded_N, num_tiles_k, num_tiles_n = \
            self._compute_params(E, K_packed, N)

        # Pad to tile boundaries if needed (in original [E, K_packed, N] space)
        if K_packed != padded_K_packed or N != padded_N:
            padded = torch.zeros((E, padded_K_packed, padded_N), dtype=data.dtype, device=data.device)
            padded[:, :K_packed, :N] = data
            data = padded

        # Transpose to swapped view: [E, K_packed, N] -> [E, N, K_packed]
        data = data.transpose(1, 2).contiguous()

        # [E, N, K_packed] -> [E, num_tiles_n, tile_n, num_tiles_k, tile_k_packed]
        data = data.view(E, num_tiles_n, tile_n, num_tiles_k, tile_k_packed)

        # Permute to [E, num_tiles_k, num_tiles_n, tile_n, tile_k_packed]
        # This puts K tiles first (for inner loop locality) and arranges
        # inner dims as [tile_n, tile_k_packed] to match baseline TMA block.
        data = data.permute(0, 3, 1, 2, 4).contiguous()
        return data

    def unswizzle_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Convert data from shuffled back to canonical [..., K, N_packed].

        Input layout: [E, num_tiles_k, num_tiles_n, tile_n, tile_k_packed]
        """
        E = data.shape[0]
        leading_shape = self.shape[:-2]
        # Recover original shape from self.shape (the logical shape passed to convert_layout)
        orig_K_packed = self.shape[-2] // 2 if self.is_fp4 else self.shape[-2]
        orig_N = self.shape[-1]
        tile_k_packed, tile_n, padded_K_packed, padded_N, num_tiles_k, num_tiles_n = \
            self._compute_params(E, orig_K_packed, orig_N)

        # Inverse of permute(0, 3, 1, 2, 4) is permute(0, 2, 3, 1, 4)
        # [E, num_tiles_k, num_tiles_n, tile_n, tile_k_packed] ->
        # [E, num_tiles_n, tile_n, num_tiles_k, tile_k_packed]
        data = data.permute(0, 2, 3, 1, 4).contiguous()

        # Back to swapped view [E, padded_N, padded_K_packed]
        data = data.view(E, padded_N, padded_K_packed)

        # Transpose back to physical [E, padded_K_packed, padded_N]
        data = data.transpose(1, 2).contiguous()

        # Trim padding back to original shape
        data = data[:, :orig_K_packed, :orig_N].contiguous()
        data = self._physical_to_canonical(data)
        if not leading_shape:
            return data.squeeze(0)
        return data.reshape(*leading_shape, data.shape[-2], data.shape[-1])
