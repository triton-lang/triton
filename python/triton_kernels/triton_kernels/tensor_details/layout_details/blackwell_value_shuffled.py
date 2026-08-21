import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

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

    @property
    def storage_shape(self) -> list[int]:
        if not self.is_fp4:
            raise ValueError("BlackwellMX4ValueShuffledLayout only supports fp4 values")
        if self.shape[-2] % 2:
            raise ValueError(f"FP4 packing dimension -2 must have an even size, got {self.shape[-2]}")
        E = math.prod(self.shape[:-2])
        K_packed = self.shape[-2] // 2
        N = self.shape[-1]
        tile_k_packed, tile_n, _, _, num_tiles_k, num_tiles_n = self._compute_params(K_packed, N)
        return [E, num_tiles_k, num_tiles_n, tile_n, tile_k_packed]

    def _compute_params(self, K_packed, N):
        """Compute tiling parameters from the physical shape."""
        tile_k_packed = self.block_k // 2
        tile_n = self.block_n

        # 128 is the TMA alignment requirement in bytes.
        align_k = math.lcm(128, tile_k_packed)
        padded_K_packed = ((K_packed + align_k - 1) // align_k) * align_k
        num_tiles_k = padded_K_packed // tile_k_packed
        num_tiles_n = (N + tile_n - 1) // tile_n
        padded_N = num_tiles_n * tile_n

        return tile_k_packed, tile_n, padded_K_packed, padded_N, num_tiles_k, num_tiles_n

    def swizzle_data(self, data: torch.Tensor) -> torch.Tensor:
        """Convert canonical [..., K, N_packed] bytes to shuffled 5D storage."""
        assert data.stride(-1) == 1
        return self._convert_data(data, inverse=False)

    def unswizzle_data(self, data: torch.Tensor) -> torch.Tensor:
        """Convert shuffled 5D storage back to canonical packed bytes."""
        return self._convert_data(data, inverse=True)

    def _convert_data(self, data: torch.Tensor, inverse: bool) -> torch.Tensor:
        storage_shape = self.storage_shape
        # The canonical intermediate packs N, while shuffled storage packs K.
        if self.shape[-1] % 2:
            raise ValueError(f"FP4 packing dimension -1 must have an even size, got {self.shape[-1]}")
        if data.device.type != "cuda" or data.dtype != torch.uint8:
            return self._unswizzle_data_torch(data) if inverse else self._swizzle_data_torch(data)

        canonical_shape = [*self.shape[:-1], self.shape[-1] // 2]
        out = torch.empty(canonical_shape if inverse else storage_shape, dtype=data.dtype, device=data.device)
        canonical, shuffled = (out, data) if inverse else (data, out)
        E, num_tiles_k, num_tiles_n, tile_n, tile_k = storage_shape
        K_packed, N_packed = self.shape[-2] // 2, self.shape[-1] // 2
        K_pad, N_pad = num_tiles_k * tile_k, num_tiles_n * tile_n
        block_k, block_n = 64, 128
        grid_k = triton.cdiv(K_packed if inverse else K_pad, block_k)
        grid_n = triton.cdiv(2 * N_packed if inverse else N_pad, 2 * block_n)
        # Keep indexing divisors valid even when the launch grid is empty.
        with torch.cuda.device(data.device):
            _convert_shuffled_mxfp4[(E * grid_k * grid_n, )](
                canonical,
                shuffled,
                tuple(canonical_shape),
                tuple(storage_shape),
                canonical.stride(),
                shuffled.stride(),
                GRID_K=max(grid_k, 1),
                GRID_N=max(grid_n, 1),
                INVERSE=inverse,
                BLOCK_K=block_k,
                BLOCK_N=block_n,
                num_warps=4,
            )
        return out

    def _swizzle_data_torch(self, data: torch.Tensor) -> torch.Tensor:
        """
        Convert data from canonical [..., K, N_packed] to 5D shuffled layout.

        Target layout: [E, num_tiles_k, num_tiles_n, tile_n, tile_k_packed]
        This matches the baseline TMA block shape [block_n, packed_block_k] after swapping.
        """
        data = repack(data, -1, -2, True)
        E, num_tiles_k, num_tiles_n, tile_n, tile_k_packed = self.storage_shape
        K_packed, N = data.shape[-2:]
        data = data.reshape(E, K_packed, N)
        padded_K_packed = num_tiles_k * tile_k_packed
        padded_N = num_tiles_n * tile_n

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
        return self._validate_storage_shape(data)

    def _unswizzle_data_torch(self, data: torch.Tensor) -> torch.Tensor:
        """
        Convert data from shuffled back to canonical [..., K, N_packed].

        Input layout: [E, num_tiles_k, num_tiles_n, tile_n, tile_k_packed]
        """
        E = data.shape[0]
        leading_shape = self.shape[:-2]
        # Recover original shape from self.shape (the logical shape passed to convert_layout)
        orig_K_packed = self.shape[-2] // 2
        orig_N = self.shape[-1]
        tile_k_packed, tile_n, padded_K_packed, padded_N, num_tiles_k, num_tiles_n = \
            self._compute_params(orig_K_packed, orig_N)

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
        data = repack(data, -2, -1, True)
        if not leading_shape:
            return data.squeeze(0)
        return data.reshape(*leading_shape, data.shape[-2], data.shape[-1])


@triton.jit
def _convert_shuffled_mxfp4(
    Canonical,
    Shuffled,
    CANONICAL_SHAPE: tl.constexpr,
    SHUFFLED_SHAPE: tl.constexpr,
    CANONICAL_STRIDES: tl.constexpr,
    SHUFFLED_STRIDES: tl.constexpr,
    GRID_K: tl.constexpr,
    GRID_N: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    K_PACKED: tl.constexpr = CANONICAL_SHAPE[-2] // 2
    N_PACKED: tl.constexpr = CANONICAL_SHAPE[-1]
    TILE_K: tl.constexpr = SHUFFLED_SHAPE[-1]
    TILE_N: tl.constexpr = SHUFFLED_SHAPE[-2]
    K_PAD: tl.constexpr = SHUFFLED_SHAPE[1] * TILE_K
    N_PAD: tl.constexpr = SHUFFLED_SHAPE[2] * TILE_N
    pid = tl.program_id(0).to(tl.int64)
    batch = pid // (GRID_K * GRID_N)
    k = (pid // GRID_N % GRID_K) * BLOCK_K + tl.arange(0, BLOCK_K)
    n = (pid % GRID_N) * BLOCK_N + tl.arange(0, BLOCK_N)

    # Keep leading strides instead of flattening a possibly noncontiguous input.
    batch_offset = tl.full((), 0, tl.int64)
    batch_index = batch
    for axis in tl.static_range(len(CANONICAL_SHAPE) - 3, -1, -1):
        batch_offset += (batch_index % max(CANONICAL_SHAPE[axis], 1)) * CANONICAL_STRIDES[axis]
        batch_index //= max(CANONICAL_SHAPE[axis], 1)
    canonical_k = 2 * k * CANONICAL_STRIDES[-2]
    canonical_even = Canonical + batch_offset + canonical_k[:, None] + n[None, :] * CANONICAL_STRIDES[-1]
    canonical_odd = canonical_even + CANONICAL_STRIDES[-2]

    shuffled_k = k // TILE_K * SHUFFLED_STRIDES[1] + k % TILE_K * SHUFFLED_STRIDES[4]
    even_n, odd_n = 2 * n, 2 * n + 1
    shuffled_even = Shuffled + batch * SHUFFLED_STRIDES[0] + shuffled_k[:, None] + \
        (even_n // TILE_N * SHUFFLED_STRIDES[2] + even_n % TILE_N * SHUFFLED_STRIDES[3])[None, :]
    shuffled_odd = Shuffled + batch * SHUFFLED_STRIDES[0] + shuffled_k[:, None] + \
        (odd_n // TILE_N * SHUFFLED_STRIDES[2] + odd_n % TILE_N * SHUFFLED_STRIDES[3])[None, :]
    mask = (k[:, None] < K_PACKED) & (n[None, :] < N_PACKED)

    if INVERSE:
        a = tl.load(shuffled_even, mask, other=0).to(tl.uint32)
        b = tl.load(shuffled_odd, mask, other=0).to(tl.uint32)
    else:
        a = tl.load(canonical_even, mask, other=0).to(tl.uint32)
        b = tl.load(canonical_odd, mask, other=0).to(tl.uint32)
    # Transposing a 2x2 group of FP4 nibbles is its own inverse.
    lo = (a & 0x0F) | ((b & 0x0F) << 4)
    hi = (a >> 4) | (b & 0xF0)

    if INVERSE:
        tl.store(canonical_even, lo, mask)
        tl.store(canonical_odd, hi, mask)
    else:
        tl.store(shuffled_even, lo, (k[:, None] < K_PAD) & (even_n[None, :] < N_PAD))
        tl.store(shuffled_odd, hi, (k[:, None] < K_PAD) & (odd_n[None, :] < N_PAD))
