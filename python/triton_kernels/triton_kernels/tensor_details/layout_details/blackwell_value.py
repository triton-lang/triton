import math
from dataclasses import dataclass
import torch
from torch._subclasses.fake_tensor import is_fake
import triton
import triton.language as tl
from triton._compile_warmup_state import is_compile_warmup
from triton_kernels.tensor_details.layout_details import strided
from .base import Layout, LayoutTransformation
from .torch_utils import repack


# ------------------- Blackwell MX Value Layout -------------------
@dataclass(frozen=True)
class BlackwellMXValueLayout(Layout):

    @property
    def name(self):
        return "BLACKWELL_MX_VALUE"

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return BlackwellMXValueLayoutTransformation(shape, is_fp4)

    def swizzle_block_shape(self, block_shape):
        return block_shape


def strides_major_dim_m2(shape):
    n = len(shape)
    if n <= 1:
        return [1] * n
    order = [n - 2, n - 1] + list(range(n - 3, -1, -1))  # fastest -> slowest
    st = [0] * n
    st[order[0]] = 1
    for prev, d in zip(order, order[1:]):
        st[d] = st[prev] * shape[prev]
    return st


# ------------------- Blackwell MX Value Layout Transformation -------------------
@dataclass(frozen=True)
class BlackwellMXValueLayoutTransformation(LayoutTransformation):

    def _can_convert(self, data):
        return (self.is_fp4 and data.device.type == "cuda" and data.dtype == torch.uint8 and self.shape[-2] % 2 == 0
                and self.shape[-1] % 2 == 0 and (not is_fake(data) or is_compile_warmup()))

    @property
    def storage_shape(self) -> list[int]:
        *leading_shape, M, K = self.shape
        if self.is_fp4:
            K //= 2
        K *= 2
        M //= 2
        M += -M % 128
        return [*leading_shape, M, K]

    def convert_data(self, data, destination: LayoutTransformation, *, out=None):
        if (not self._can_convert(data) or not isinstance(destination, strided.StridedLayoutTransformation)
                or destination.order[0] < len(self.shape) - 2):
            return super().convert_data(data, destination, out=out)

        return unswizzle_mxfp4(data, self.shape, destination.order[0], out=out)

    def _convert_data_from(self, data, source: LayoutTransformation, *, out):
        if (not self._can_convert(data) or not isinstance(source, strided.StridedLayoutTransformation)
                or not source._can_convert_fp4(data)):
            return super()._convert_data_from(data, source, out=out)
        return self._swizzle_mxfp4(data, source.order[0], out=out)

    def _swizzle_mxfp4(self, data, major_dim, out=None):
        if out is None:
            out = torch.empty_strided(self.storage_shape, strides_major_dim_m2(self.storage_shape), device=data.device,
                                      dtype=data.dtype)
        if major_dim % len(self.shape) == len(self.shape) - 1:
            # N-to-K repacking is K-to-N repacking with the matrix axes exchanged.
            shape = [*self.shape[:-2], self.shape[-1], self.shape[-2]]
            unswizzle_mxfp4(data.mT, shape, -1, out=out.mT)
        else:
            unswizzle_mxfp4(data, self.shape, -2, out=out)
        return out

    def swizzle_data(self, data):
        if self._can_convert(data):
            return self._swizzle_mxfp4(data, -1)
        # re-pack as column-major
        ret = torch.empty_strided(self.storage_shape, strides_major_dim_m2(self.storage_shape), device=data.device,
                                  dtype=data.dtype)
        repacked_shape = list(data.shape)
        repacked_shape[-1] *= 2
        repacked_shape[-2] //= 2
        repack(data, -1, -2, self.is_fp4, out=ret[..., :repacked_shape[-2], :])
        return self._validate_storage_shape(ret)

    def _unpad_data(self, data: torch.Tensor):
        sizes = [self.shape[i] for i in range(data.ndim)]
        sizes[-2] //= 2
        return data[tuple(slice(0, s) for s in sizes)]

    def unswizzle_data(self, data: torch.Tensor):
        if self._can_convert(data):
            return unswizzle_mxfp4(data, self.shape, -1)
        data = self._unpad_data(data)
        out_shape = list(self.shape)
        out_shape[-1] //= 2
        out = torch.empty(out_shape, device=data.device, dtype=data.dtype)
        repack(data, -2, -1, self.is_fp4, out=out)
        return out


def unswizzle_mxfp4(data, shape, major_dim, out=None, *, block_shape=None):
    """Convert plain or tile-shuffled K-packed bytes to a strided FP4 tensor."""
    if out is None:
        destination = strided.StridedLayout(major_dim).make_transformation(shape, True)
        out = torch.empty_strided(destination.storage_shape, destination.storage_strides, device=data.device,
                                  dtype=data.dtype)
    if out.numel() == 0:
        return out
    pack_k = major_dim % len(shape) == len(shape) - 2
    block_k = min(64, triton.next_power_of_2(shape[-2] // 2))
    block_n = min(128 if pack_k else 256, triton.next_power_of_2(shape[-1] // 2))
    grid_k = triton.cdiv(shape[-2] // 2, block_k)
    grid_n = triton.cdiv(shape[-1] // 2, block_n)
    grid = (math.prod(shape[:-2]) * grid_k * grid_n, )
    with torch.cuda.device(data.device):
        _unswizzle_mxfp4[grid](data, out, tuple(shape), data.stride(), out.stride(), block_shape, grid_k, grid_n,
                               pack_k, block_k, block_n, num_warps=4)
    return out


@triton.jit
def _unswizzle_mxfp4(
    Source,
    Out,
    SHAPE: tl.constexpr,
    SOURCE_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    BLOCK_SHAPE: tl.constexpr,
    GRID_K: tl.constexpr,
    GRID_N: tl.constexpr,
    PACK_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    batch = pid // (GRID_K * GRID_N)
    k = pid // GRID_N % GRID_K * BLOCK_K + tl.arange(0, BLOCK_K)
    n = pid % GRID_N * BLOCK_N + tl.arange(0, BLOCK_N)
    source_batch = tl.full((), 0, tl.int64)
    out_batch = tl.full((), 0, tl.int64)
    # Leading dimensions may be sliced or permuted independently.
    batch_index = batch
    for axis in tl.static_range(len(SHAPE) - 3, -1, -1):
        index = batch_index % SHAPE[axis]
        out_batch += index * OUT_STRIDES[axis]
        if BLOCK_SHAPE is None:
            source_batch += index * SOURCE_STRIDES[axis]
        batch_index //= SHAPE[axis]
    if BLOCK_SHAPE is None:
        offsets = source_batch + k[:, None] * SOURCE_STRIDES[-2] + 2 * n[None, :] * SOURCE_STRIDES[-1]
        odd_offsets = offsets + SOURCE_STRIDES[-1]
    else:
        TILE_K: tl.constexpr = BLOCK_SHAPE[0]
        TILE_N: tl.constexpr = BLOCK_SHAPE[1]
        source_k = k // TILE_K * SOURCE_STRIDES[1] + k % TILE_K * SOURCE_STRIDES[4]
        even_n, odd_n = 2 * n, 2 * n + 1
        offsets = batch * SOURCE_STRIDES[0] + source_k[:, None] + (even_n // TILE_N * SOURCE_STRIDES[2] +
                                                                   even_n % TILE_N * SOURCE_STRIDES[3])[None, :]
        odd_offsets = batch * SOURCE_STRIDES[0] + source_k[:, None] + (odd_n // TILE_N * SOURCE_STRIDES[2] +
                                                                       odd_n % TILE_N * SOURCE_STRIDES[3])[None, :]
    mask = (k[:, None] < SHAPE[-2] // 2) & (n[None, :] < SHAPE[-1] // 2)
    a = tl.load(Source + offsets, mask, other=0)
    b = tl.load(Source + odd_offsets, mask, other=0)
    if PACK_K:
        offsets = out_batch + k[:, None] * OUT_STRIDES[-2] + 2 * n[None, :] * OUT_STRIDES[-1]
        tl.store(Out + offsets, a, mask)
        tl.store(Out + offsets + OUT_STRIDES[-1], b, mask)
    else:
        # Transpose each 2x2 group of FP4 nibbles, then store adjacent rows together.
        lo = (a & 0x0F) | ((b & 0x0F) << 4)
        hi = (a >> 4) | (b & 0xF0)
        values = tl.join(lo, hi).permute((0, 2, 1)).reshape((2 * BLOCK_K, BLOCK_N))
        rows = pid // GRID_N % GRID_K * (2 * BLOCK_K) + tl.arange(0, 2 * BLOCK_K)
        offsets = out_batch + rows[:, None] * OUT_STRIDES[-2] + n[None, :] * OUT_STRIDES[-1]
        tl.store(Out + offsets, values, (rows[:, None] < SHAPE[-2]) & (n[None, :] < SHAPE[-1] // 2))
