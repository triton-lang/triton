import math
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import is_fake
import triton
import triton.language as tl

from triton_kernels.tensor_details.ragged_tensor import RaggedTensorMetadata
from .base import Layout, LayoutTransformation
from .strided import StridedLayoutTransformation
from triton_kernels import target_info

# ------------------- Blackwell MX Scale Layout -------------------


@dataclass(frozen=True)
class BlackwellMXScaleLayout(Layout):

    @property
    def name(self):
        return "BLACKWELL_SCALE"

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return BlackwellMXScaleLayoutTransformation(shape, is_fp4)

    def swizzle_block_shape(self, block_shape):
        K, N = block_shape
        assert K % 4 == 0, f"{block_shape[0]=} must be divisible by 4"
        assert N % 128 == 0, f"{block_shape[1]=} must be divisible by 128"
        return [1, N // 128, K // 4, 2, 256]


@dataclass(frozen=True)
class BlackwellActMXScaleLayout(Layout):

    ragged_metadata: RaggedTensorMetadata | None

    def can_preserve_storage_as(self, other: Layout, rank: int) -> bool:
        return isinstance(other, BlackwellActMXScaleLayout) and self.ragged_metadata is other.ragged_metadata

    @property
    def name(self):
        return "BLACKWELL_ACT_SCALE"

    def swizzle_block_shape(self, block_shape):
        N, K = block_shape
        assert K % 4 == 0, f"{block_shape[1]=} must be divisible by 4"
        assert N % 128 == 0, f"{block_shape[0]=} must be divisible by 128"
        return [1, block_shape[0] // 128, block_shape[1] // 4, 2, 256]

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return BlackwellActMXScaleLayoutTransformation(shape, is_fp4, self.ragged_metadata)


# ------------------- Blackwell MX Scale Layout Transformation -------------------


@dataclass(frozen=True)
class BlackwellActMXScaleLayoutTransformation(LayoutTransformation):

    ragged_metadata: RaggedTensorMetadata | None
    added_leading_batch_dim: bool = False
    ALIGN_K: int = 8
    ALIGN_M: int = 128
    SWIZZLE_K: int = 4

    def __post_init__(self):
        added_leading_batch_dim = False
        if len(self.shape) == 2:
            B, M, K = 1, *self.shape
            added_leading_batch_dim = True
            if self.ragged_metadata is None:
                M_pad = (M + self.ALIGN_M - 1) // self.ALIGN_M * self.ALIGN_M
                mode = "batched"
            else:
                # In ragged mode, input often include padded tokens
                # Out of M rows, the number of valid rows is the sum of ragged_metadata.slice_sizes
                # And the rest of rows are padded tokens
                n_slices = self.ragged_metadata.slice_sizes.shape[0]
                # this estimates the number of blocks (each block has ALIGN_M rows) we need if we have all M valid tokens
                max_n_blocks = self.ragged_metadata.n_blocks(n_slices, M, self.ALIGN_M)
                # create a static size scratchpad for output
                M_pad = self.ALIGN_M * max_n_blocks
                mode = "ragged"
        else:
            B, M, K = self.shape
            M_pad = (M + self.ALIGN_M - 1) // self.ALIGN_M * self.ALIGN_M
            mode = "batched"
        K_pad = (K + self.ALIGN_K - 1) // self.ALIGN_K * self.ALIGN_K  # min multiple of ALIGN_K
        # initialize attributes
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "M_pad", M_pad)
        object.__setattr__(self, "K_pad", K_pad)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "added_leading_batch_dim", added_leading_batch_dim)

    @property
    def storage_shape(self) -> list[int]:
        return [1, self.B * self.M_pad // 128, self.K_pad // 4, 2, 256]

    def convert_data(self, data, destination: LayoutTransformation, *, out=None):
        if isinstance(destination, StridedLayoutTransformation) and not self.is_fp4 and _can_convert_scale(data):
            if out is None:
                out = torch.empty_strided(destination.storage_shape, destination.storage_strides, dtype=data.dtype,
                                          device=data.device)
            return self._unswizzle_data(data, out)
        return super().convert_data(data, destination, out=out)

    def _convert_data_from(self, data, source: LayoutTransformation, *, out):
        if isinstance(source, StridedLayoutTransformation) and not self.is_fp4 and _can_convert_scale(data):
            return self._swizzle_data(data, out=out)
        return super()._convert_data_from(data, source, out=out)

    def _swizzle_data(self, data, out=None):
        # Activation scales use the same encoding with the matrix axes exchanged.
        shape = [*self.shape[:-2], self.shape[-1], self.shape[-2]]
        metadata = self.ragged_metadata if self.mode == "ragged" else None
        return _convert_mx_scale(data.mT, shape, self.storage_shape, out=out, ragged_metadata=metadata)

    def swizzle_data(self, data):
        if not self.is_fp4 and _can_convert_scale(data):
            return self._swizzle_data(data)
        if self.mode == "batched":
            # value of padding on left, right, top, bottom
            data = torch.nn.functional.pad(data, (0, self.K_pad - self.K, 0, self.M_pad - self.M))
        else:
            # Objective is to pad the number of rows in each slice to be multiple of ALIGN_M
            data = pad_segments_triton(
                data,
                self.ragged_metadata,
                self.ALIGN_M,
                self.M_pad,
                self.K,
                self.K_pad,
            )

        data = data.reshape(self.B, self.M_pad // 128, 4, 32, self.K_pad // 4, 4)
        data = data.transpose(2, 4).contiguous()  # [1, M//128, K//4, 32, 4, 4]
        data = data.view(*self.storage_shape)
        return self._validate_storage_shape(data)

    def _unswizzle_data(self, data, out=None):
        if out is None:
            out = torch.empty(self.shape, dtype=data.dtype, device=data.device)
        shape = [*self.shape[:-2], self.shape[-1], self.shape[-2]]
        metadata = self.ragged_metadata if self.mode == "ragged" else None
        _convert_mx_scale(data, shape, self.storage_shape, out=out.mT, ragged_metadata=metadata, inverse=True)
        return out

    def unswizzle_data(self, data):
        if not self.is_fp4 and _can_convert_scale(data):
            return self._unswizzle_data(data)
        data = data.reshape(self.B, self.M_pad // 128, self.K_pad // 4, 32, 4, 4)
        data = data.transpose(2, 4)  # [B, M//128, 4, 32, K//4, 4]
        data = data.reshape(self.B, self.M_pad, self.K_pad)

        if self.mode == "batched":
            data = data[..., :self.M, :self.K]
            if self.added_leading_batch_dim:
                return data.squeeze(0)
            return data

        # ragged path: map padded blocks back into the original ragged rows
        data = unpad_segments_triton(
            data.squeeze(0),
            self.ragged_metadata,
            self.ALIGN_M,
            self.M,
            self.K,
            self.K_pad,
        )
        return data.contiguous()


@dataclass(frozen=True)
class BlackwellMXScaleLayoutTransformation(LayoutTransformation):

    def __post_init__(self) -> None:
        *leading_shape, K, N = self.shape
        object.__setattr__(self, "leading_shape", leading_shape)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "N", N)
        object.__setattr__(self, "B", math.prod(leading_shape))
        object.__setattr__(self, "ALIGN_K", 8)
        object.__setattr__(self, "ALIGN_N", 128)
        object.__setattr__(self, "SWIZZLE_K", 4)
        object.__setattr__(self, "K_pad", (K + self.ALIGN_K - 1) // self.ALIGN_K * self.ALIGN_K)
        object.__setattr__(self, "N_pad", (N + self.ALIGN_N - 1) // self.ALIGN_N * self.ALIGN_N)

    @property
    def storage_shape(self) -> list[int]:
        return [1, self.B * self.N_pad // 128, self.K_pad // self.SWIZZLE_K, 2, 256]

    def convert_data(self, data, destination: LayoutTransformation, *, out=None):
        if isinstance(destination, StridedLayoutTransformation) and not self.is_fp4 and _can_convert_scale(data):
            if out is None:
                out = torch.empty_strided(destination.storage_shape, destination.storage_strides, dtype=data.dtype,
                                          device=data.device)
            return self._unswizzle_data(data, out)
        return super().convert_data(data, destination, out=out)

    def _convert_data_from(self, data, source: LayoutTransformation, *, out):
        if isinstance(source, StridedLayoutTransformation) and not self.is_fp4 and _can_convert_scale(data):
            return _convert_mx_scale(data, self.shape, self.storage_shape, out=out)
        return super()._convert_data_from(data, source, out=out)

    def swizzle_data(self, data):
        if not _can_convert_scale(data):
            data = torch.nn.functional.pad(data, (0, self.N_pad - self.N, 0, self.K_pad - self.K))
            data = data.transpose(-1, -2).contiguous()
            data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.ALIGN_N // 32, 32,
                                self.K_pad // self.SWIZZLE_K, self.SWIZZLE_K)
            data = data.transpose(2, 4).contiguous()
            data = data.view(*self.storage_shape)
            return self._validate_storage_shape(data)

        return _convert_mx_scale(data, self.shape, self.storage_shape)

    def _unswizzle_data(self, data, out=None):
        if out is None:
            out = torch.empty(self.shape, dtype=data.dtype, device=data.device)
        return _convert_mx_scale(data, self.shape, self.storage_shape, out=out, inverse=True)

    def unswizzle_data(self, data):
        if not self.is_fp4 and _can_convert_scale(data):
            return self._unswizzle_data(data)
        data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.K_pad // self.SWIZZLE_K, 32, self.ALIGN_N // 32,
                            self.SWIZZLE_K)
        data = data.transpose(2, 4)
        data = data.reshape(*self.leading_shape, self.N_pad, self.K_pad)
        data = data.transpose(-1, -2).contiguous()
        data = data[..., :self.K, :self.N]
        return data


def _can_convert_scale(data):
    return data.device.type == "cuda" and data.dtype.itemsize == 1 and not is_fake(data)


def _convert_mx_scale(data, shape, storage_shape, out=None, *, ragged_metadata=None, inverse=False):
    if out is None:
        out = torch.empty(storage_shape, dtype=data.dtype, device=data.device)
    if not out.numel():
        return out
    matrix, encoded = (out, data) if inverse else (data, out)
    assert tuple(matrix.shape) == tuple(shape)
    batch = math.prod(shape[:-2])
    k_pad, n_pad = storage_shape[2] * 4, storage_shape[1] // batch * 128
    metadata = (None, None, None, None)
    n_slices = 0
    if ragged_metadata is not None:
        metadata = (ragged_metadata.slice_sizes, ragged_metadata.slice_offs, ragged_metadata.block_offs(128),
                    ragged_metadata.block_schedule(128))
        n_slices = ragged_metadata.n_slices
    # Keep enough blocks for small tensors; larger tiles amortize scheduling work.
    size = batch * k_pad * n_pad
    contiguous_k = matrix.stride(-2) == 1
    block_k = min(128, triton.next_power_of_2(k_pad)) if contiguous_k else 64
    block_k = min(block_k, max(8, triton.next_power_of_2(size) // (128 * 128)))
    if contiguous_k and k_pad % 32 == 0:
        block_k = math.gcd(block_k, k_pad)
    block_n, num_warps = 128, 4
    if k_pad == 8:
        block_k = 8
        if ragged_metadata is None and size >= 2**22:
            block_n = 512 if inverse or size < 2**25 else 1024
            if inverse:
                num_warps = 8
    elif matrix.stride(-1) == 1 and k_pad <= 32 and size >= 2**22:
        block_k, num_warps = 32, 2
    grid = (batch * triton.cdiv(n_pad, block_n) * triton.cdiv(k_pad, block_k), )
    with torch.cuda.device(data.device):
        _convert_blackwell_mx_scale[grid](
            matrix.view(torch.uint8),
            encoded.view(torch.uint8),
            tuple(shape),
            matrix.stride(),
            encoded.stride(),
            k_pad,
            n_pad,
            *metadata,
            n_slices,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            INVERSE=inverse,
            num_warps=num_warps,
        )
    return out


SWIZZLE_ALIGN_INNER = tl.constexpr(8)
SWIZZLE_SIZE_INNER = tl.constexpr(4)
SWIZZLE_SIZE_OUTER = tl.constexpr(128)
SWIZZLE_SIZE_LANE = tl.constexpr(32)
SWIZZLE_SIZE_HALF = tl.constexpr(256)


@triton.jit
def pad_segments_kernel(
    data_ptr,
    out_ptr,
    slice_sizes_ptr,
    slice_offs_ptr,
    block_offs_ptr,
    block_schedule_ptr,
    K: tl.constexpr,
    K_pad: tl.constexpr,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_BLOCKS_PER_COL: tl.constexpr,
    N_SLICES: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    useful_grid_m = tl.load(block_offs_ptr + N_SLICES)  # number of valid blks we care about in the output
    num_blocks = useful_grid_m * N_BLOCKS_PER_COL

    for block_id in tl.range(tl.program_id(0), num_blocks, NUM_SMS):
        blk_m_idx = block_id // N_BLOCKS_PER_COL
        blk_n_idx = block_id % N_BLOCKS_PER_COL

        # get expert index and block index within the expert
        block_schedule = tl.load(block_schedule_ptr + blk_m_idx)  # always should get a valid block
        slice_idx = block_schedule & 0x0000FFFF
        blk_m_idx_in_slice = block_schedule >> 16

        # for the current output block, get the masked input block
        slice_size = tl.load(slice_sizes_ptr + slice_idx)  # actual rows
        input_slice_base = tl.load(slice_offs_ptr + slice_idx)  # row offset in `data`
        in_ptrs = data_ptr + input_slice_base * stride_in_m  # move in_ptrs to the start of the input slice

        in_rows = blk_m_idx_in_slice * BLOCK_M + tl.arange(0, BLOCK_M)
        in_cols = blk_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        row_in_range_in = in_rows < slice_size
        col_in_range_in = in_cols < K
        in_mask = row_in_range_in[:, None] & col_in_range_in[None, :]

        out_rows = blk_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        out_cols = blk_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        col_in_range_out = out_cols < K_pad
        out_mask = col_in_range_out[None, :]

        # default pad value = 0
        vals = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # compute linear ptrs with strides
        in_ptrs = in_ptrs + in_rows[:, None] * stride_in_m + in_cols[None, :] * stride_in_n
        vals = tl.load(in_ptrs, mask=in_mask & out_mask, other=0.0)

        # store into output
        out_ptrs = out_ptr + out_rows[:, None] * stride_out_m + out_cols[None, :] * stride_out_n
        tl.store(out_ptrs, vals, mask=out_mask)


def pad_segments_triton(data, ragged_metadata, block_size_to_align, M_pad, K, K_pad):
    """
    Pads the number of rows in each slice to be multiple of block_size_to_align
    and the number of columns to be multiple of BLOCK_N

    Input data has static shape [M, K] which include valid rows and padded rows.
    The number of valid rows equals to the sum of ragged_metadata.slice_sizes and varies across batches.
    Here we allocate enough static size for padded output but only overwrite the rows that correspond to a padded version of each expert.

    Example:
    input data: [10, 10] with 6 valid rows and 4 padded rows
    ragged_metadata.slice_sizes: [2, 1, 3] means 3 experts with 2, 1, 3 valid rows respectively
    block_size_to_align: 4 means we want to pad the number of rows in each slice to be multiple of 4

    We allocate a output with shape [16, 10] which is the maximum number of rows we need even if all 10 rows are valid;
    Each expert is padded to 4 rows;
    The output will have rows: [x, x, 0, 0, x, 0, 0, 0, x, x, x, 0, 0, 0, 0, 0] (x means valid row, 0 means padded row)

    Args:
        data: input data
        ragged_metadata: ragged metadata
        block_size_to_align: block size to align
        M_pad: padded number of rows
        K: input width
        K_pad: padded number of columns
    """
    padded_data = torch.empty(M_pad, K_pad, device=data.device, dtype=data.dtype)
    padded_data.fill_(0.0)
    if not padded_data.numel():
        return padded_data

    slice_sizes = ragged_metadata.slice_sizes
    slice_offs = ragged_metadata.slice_offs
    block_offs = ragged_metadata.block_offs(block_size_to_align)
    block_schedule = ragged_metadata.block_schedule(block_size_to_align)

    # strides (in elements, not bytes)
    stride_in_m, stride_in_n = data.stride()
    stride_out_m, stride_out_n = padded_data.stride()

    BLOCK_M = block_size_to_align
    BLOCK_N = 64

    max_grid = triton.cdiv(M_pad, BLOCK_M) * triton.cdiv(K_pad, BLOCK_N)
    num_sms = target_info.num_sms()
    grid = min(num_sms, max_grid)
    pad_segments_kernel[(grid, )](
        data,
        padded_data,
        slice_sizes,
        slice_offs,
        block_offs,
        block_schedule,
        K,
        K_pad,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        N_BLOCKS_PER_COL=triton.cdiv(K_pad, BLOCK_N),
        N_SLICES=slice_sizes.shape[0],
        NUM_SMS=num_sms,
    )

    return padded_data


@triton.jit
def unpad_segments_kernel(
    padded_ptr,
    out_ptr,
    slice_sizes_ptr,
    slice_offs_ptr,
    block_offs_ptr,
    block_schedule_ptr,
    K: tl.constexpr,
    K_pad: tl.constexpr,
    stride_pad_m,
    stride_pad_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_BLOCKS_PER_COL: tl.constexpr,
    N_SLICES: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    useful_grid_m = tl.load(block_offs_ptr + N_SLICES)
    num_blocks = useful_grid_m * N_BLOCKS_PER_COL

    for block_id in tl.range(tl.program_id(0), num_blocks, NUM_SMS):
        blk_m_idx = block_id // N_BLOCKS_PER_COL
        blk_n_idx = block_id % N_BLOCKS_PER_COL

        block_schedule = tl.load(block_schedule_ptr + blk_m_idx)
        slice_idx = block_schedule & 0x0000FFFF
        blk_m_idx_out_slice = block_schedule >> 16

        slice_size = tl.load(slice_sizes_ptr + slice_idx)
        out_slice_base = tl.load(slice_offs_ptr + slice_idx)  # output is unpadded format
        out_ptrs_base = out_ptr + out_slice_base * stride_out_m

        out_rows = blk_m_idx_out_slice * BLOCK_M + tl.arange(0, BLOCK_M)
        out_cols = blk_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        row_out_range = out_rows < slice_size
        col_out_range = out_cols < K
        mask = row_out_range[:, None] & col_out_range[None, :]

        pad_rows = blk_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        pad_cols = blk_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        pad_mask = pad_cols < K_pad

        padded_ptrs = padded_ptr + pad_rows[:, None] * stride_pad_m + pad_cols[None, :] * stride_pad_n
        vals = tl.load(padded_ptrs, mask=pad_mask[None, :], other=0.0)

        out_ptrs = out_ptrs_base + out_rows[:, None] * stride_out_m + out_cols[None, :] * stride_out_n
        tl.store(out_ptrs, vals, mask=mask)


def unpad_segments_triton(padded_data, ragged_metadata, block_size_to_align, M, K, K_pad):
    # output tensor with exact ragged rows/cols
    data = torch.empty(M, K, device=padded_data.device, dtype=padded_data.dtype)
    data.fill_(0.0)
    if not data.numel():
        return data

    slice_sizes = ragged_metadata.slice_sizes
    slice_offs = ragged_metadata.slice_offs
    block_offs = ragged_metadata.block_offs(block_size_to_align)
    block_schedule = ragged_metadata.block_schedule(block_size_to_align)

    stride_pad_m, stride_pad_n = padded_data.stride()
    stride_out_m, stride_out_n = data.stride()

    BLOCK_M = block_size_to_align
    BLOCK_N = 64

    max_grid = triton.cdiv(padded_data.shape[0], BLOCK_M) * triton.cdiv(K_pad, BLOCK_N)
    num_sms = target_info.num_sms()
    grid = min(num_sms, max_grid)

    unpad_segments_kernel[(grid, )](
        padded_data,
        data,
        slice_sizes,
        slice_offs,
        block_offs,
        block_schedule,
        K,
        K_pad,
        stride_pad_m,
        stride_pad_n,
        stride_out_m,
        stride_out_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        N_BLOCKS_PER_COL=triton.cdiv(K_pad, BLOCK_N),
        N_SLICES=slice_sizes.shape[0],
        NUM_SMS=num_sms,
    )

    return data


# ---


@triton.jit
def unswizzle_mx_scale_bw(
    x,
    SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,
    SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,
    SIZE_LANE: tl.constexpr = SWIZZLE_SIZE_LANE,
    ALIGN_INNER: tl.constexpr = SWIZZLE_ALIGN_INNER,
):
    shape_0: tl.constexpr = x.shape[0]
    shape_1: tl.constexpr = x.shape[1]
    tl.static_assert(shape_1 % SIZE_OUTER == 0)
    tl.static_assert((shape_1 // SIZE_OUTER) % SIZE_INNER == 0)
    x = x.reshape(shape_0, (shape_1 // SIZE_OUTER) // SIZE_INNER, SIZE_LANE, SIZE_OUTER // SIZE_LANE, SIZE_INNER)
    x = x.trans(0, 3, 2, 1, 4).reshape(shape_0 * SIZE_OUTER, shape_1 // SIZE_OUTER)
    return x


@triton.jit
def unswizzle_act_mx_scale_bw(
    x,
    SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,  # 128
    SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,  # 4
    SIZE_LANE: tl.constexpr = SWIZZLE_SIZE_LANE,
):
    # input block shape is [1, BLOCK_M//128, BLOCK_K//32//4, 2, 256] and we want to unswizzle it to [BLOCK_M, BLOCK_K//32]
    shape_1: tl.constexpr = x.shape[1]
    shape_2: tl.constexpr = x.shape[2]
    unswizzled_block_m: tl.constexpr = shape_1 * SIZE_OUTER  # BLOCK_M
    unswizzled_block_k: tl.constexpr = shape_2 * SIZE_INNER  # BLOCK_K // 32

    x = x.reshape(shape_1, shape_2, SIZE_LANE, SIZE_OUTER // SIZE_LANE, SIZE_INNER)
    x = x.trans(0, 3, 2, 1, 4).reshape(unswizzled_block_m, unswizzled_block_k)
    return x


@triton.jit
def swizzle_mx_scale_bw_store_ptr(base, rows, cols, leading_idx, n_cols, stride_outer, stride_inner, stride_half,
                                  stride_lane, SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,
                                  SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,
                                  SIZE_LANE: tl.constexpr = SWIZZLE_SIZE_LANE,
                                  SIZE_HALF: tl.constexpr = SWIZZLE_SIZE_HALF, INDEX_TYPE: tl.constexpr = tl.int64):
    col_block = cols // SIZE_OUTER
    outer = leading_idx * tl.cdiv(n_cols, SIZE_OUTER) + col_block
    inner_group = rows // SIZE_INNER
    col_lane = cols - col_block * SIZE_OUTER
    col_quad = col_lane // SIZE_LANE
    col_lane_inner = col_lane - col_quad * SIZE_LANE
    row_lane = rows - inner_group * SIZE_INNER
    inner_linear = ((col_lane_inner[None, :] * (SIZE_OUTER // SIZE_LANE) + col_quad[None, :]) * SIZE_INNER +
                    row_lane[:, None])
    half = inner_linear // SIZE_HALF
    inner = inner_linear - half * SIZE_HALF
    return (base + outer.to(INDEX_TYPE)[None, :] * stride_outer + inner_group.to(INDEX_TYPE)[:, None] * stride_inner +
            half.to(INDEX_TYPE) * stride_half + inner.to(INDEX_TYPE) * stride_lane)


@triton.jit
def _convert_blackwell_mx_scale(
    Matrix,
    Encoded,
    SHAPE: tl.constexpr,
    MATRIX_STRIDES: tl.constexpr,
    ENCODED_STRIDES: tl.constexpr,
    K_PAD: tl.constexpr,
    N_PAD: tl.constexpr,
    SliceSizes,
    SliceOffs,
    BlockOffs,
    BlockSchedule,
    N_SLICES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INVERSE: tl.constexpr = False,
):
    tl.static_assert(BLOCK_K % 4 == 0 and BLOCK_N % 128 == 0)
    K: tl.constexpr = SHAPE[-2]
    N: tl.constexpr = SHAPE[-1]
    k_blocks: tl.constexpr = tl.cdiv(K_PAD, BLOCK_K)
    n_blocks: tl.constexpr = N_PAD // 128
    n_tiles: tl.constexpr = tl.cdiv(N_PAD, BLOCK_N)
    pid = tl.program_id(0).to(tl.int64)
    k_block = pid % k_blocks
    n_tile = pid // k_blocks % n_tiles
    batch = pid // (k_blocks * n_tiles)
    batch_index = batch
    batch_offset = tl.full((), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 3, -1, -1):
        batch_offset += (batch_index % SHAPE[axis]) * MATRIX_STRIDES[axis]
        batch_index //= SHAPE[axis]

    rows = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    cols = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    if BlockSchedule is not None:
        tl.static_assert(BLOCK_N == 128)
        n_block = n_tile
        active = n_block < tl.load(BlockOffs + N_SLICES)
        schedule = tl.load(BlockSchedule + n_block, active, other=0)
        slice_index = schedule & 0xFFFF
        block_in_slice = schedule >> 16
        slice_size = tl.load(SliceSizes + slice_index, active, other=0)
        slice_start = tl.load(SliceOffs + slice_index, active, other=0).to(tl.int64)
        slice_rows = block_in_slice * 128 + tl.arange(0, 128)
        cols = slice_start + slice_rows
        col_mask = active & (slice_rows < slice_size)
    else:
        col_mask = cols < N
    offsets = tl.arange(0, BLOCK_K * BLOCK_N)
    n_block = n_tile * (BLOCK_N // 128) + offsets // (BLOCK_K * 128)
    group = k_block * (BLOCK_K // 4) + offsets % (BLOCK_K * 128) // 512
    lane = offsets % 512
    if ENCODED_STRIDES[2:] == (512, 256, 1):
        # A flat inner offset lets the compiler vectorize contiguous loads and stores.
        encoded_offsets = (batch * n_blocks + n_block) * ENCODED_STRIDES[1] + group * 512 + lane
    else:
        encoded_offsets = ((batch * n_blocks + n_block) * ENCODED_STRIDES[1] + group * ENCODED_STRIDES[2] +
                           lane // 256 * ENCODED_STRIDES[3] + lane % 256 * ENCODED_STRIDES[4])
    matrix_offsets = batch_offset + rows[:, None] * MATRIX_STRIDES[-2] + cols[None, :] * MATRIX_STRIDES[-1]
    matrix_mask = (rows[:, None] < K) & col_mask[None, :]
    encoded_mask = (group < K_PAD // 4) & (n_block < n_blocks)
    if INVERSE:
        vals = tl.load(Encoded + encoded_offsets, encoded_mask, other=0)
        vals = vals.reshape(BLOCK_N // 128, BLOCK_K // 4, 32, 4, 4).trans(1, 4, 0, 3, 2).reshape(BLOCK_K, BLOCK_N)
        tl.store(Matrix + matrix_offsets, vals, matrix_mask)
        if BlockSchedule is not None:
            # Valid segments occupy a prefix; the unused logical tail must be zero.
            valid_rows = tl.load(SliceOffs + N_SLICES).to(tl.int64)
            tail_cols = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
            tail_offsets = batch_offset + rows[:, None] * MATRIX_STRIDES[-2] + tail_cols[None, :] * MATRIX_STRIDES[-1]
            tail_mask = (rows[:, None] < K) & (tail_cols[None, :] >= valid_rows) & (tail_cols[None, :] < N)
            tl.store(Matrix + tail_offsets, 0, tail_mask)
    else:
        vals = tl.load(Matrix + matrix_offsets, matrix_mask, other=0)
        # A swizzle block is 4x128; reorder its 512 encoded elements.
        vals = vals.reshape(BLOCK_K // 4, 4, BLOCK_N // 128, 4, 32).trans(2, 0, 4, 3, 1).ravel()
        tl.store(Encoded + encoded_offsets, vals, encoded_mask)


@triton.jit
def swizzle_act_mx_scale_bw_store_ptr(base, rows, cols, leading_m_block, stride_outer, stride_inner, stride_half,
                                      stride_lane, SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,
                                      SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,
                                      SIZE_LANE: tl.constexpr = SWIZZLE_SIZE_LANE,
                                      SIZE_HALF: tl.constexpr = SWIZZLE_SIZE_HALF, INDEX_TYPE: tl.constexpr = tl.int64):
    row_block = rows // SIZE_OUTER
    outer = leading_m_block + row_block
    inner_group = cols // SIZE_INNER
    col_lane = cols - inner_group * SIZE_INNER
    row_lane = rows - row_block * SIZE_OUTER
    row_quad = row_lane // SIZE_LANE
    row_lane_inner = row_lane - row_quad * SIZE_LANE
    inner_linear = ((row_lane_inner[:, None] * (SIZE_OUTER // SIZE_LANE) + row_quad[:, None]) * SIZE_INNER +
                    col_lane[None, :])
    half = inner_linear // SIZE_HALF
    inner = inner_linear - half * SIZE_HALF
    return (base + outer.to(INDEX_TYPE)[:, None] * stride_outer + inner_group.to(INDEX_TYPE)[None, :] * stride_inner +
            half.to(INDEX_TYPE) * stride_half + inner.to(INDEX_TYPE) * stride_lane)
