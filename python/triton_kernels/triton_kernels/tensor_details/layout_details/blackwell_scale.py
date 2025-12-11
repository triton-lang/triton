import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from triton_kernels.tensor_details.ragged_tensor import RaggedTensorMetadata
from .base import Layout
from triton_kernels import target_info

SWIZZLE_ALIGN_INNER = tl.constexpr(8)
SWIZZLE_SIZE_INNER = tl.constexpr(4)
SWIZZLE_SIZE_OUTER = tl.constexpr(128)


@dataclass
class BlackwellMXScaleLayout(Layout):
    B: int
    ALIGN_K: int
    ALIGN_N: int
    SWIZZLE_K: int
    K_pad: int
    N_pad: int
    name: str = "BLACKWELL_SCALE"

    def __init__(self, shape) -> None:
        super().__init__(shape)
        (
            *self.leading_shape,
            self.K,
            self.N,
        ) = shape
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
        data = data.view(1, self.B * self.N_pad // 128, self.K_pad // self.SWIZZLE_K, 2, 256)
        return data

    def unswizzle_data(self, data):
        data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.K_pad // self.SWIZZLE_K, 32, self.ALIGN_N // 32,
                            self.SWIZZLE_K)
        data = data.transpose(2, 4)
        data = data.reshape(*self.leading_shape, self.N_pad, self.K_pad)
        data = data.transpose(-1, -2)
        return data[..., :self.K, :self.N]

    def swizzle_block_shape(self, block_shape):
        K, N = block_shape
        assert N >= 128, f"{block_shape[1]=} must be >= 128"
        return [1, N // 128, K // 4, 2, 256]


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
    slice_sizes = ragged_metadata.slice_sizes
    slice_offs = ragged_metadata.slice_offs
    block_offs = ragged_metadata.block_offs(block_size_to_align)
    block_schedule = ragged_metadata.block_schedule(block_size_to_align)

    padded_data = torch.empty(M_pad, K_pad, device=data.device, dtype=data.dtype)
    padded_data.fill_(0.0)

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
    slice_sizes = ragged_metadata.slice_sizes
    slice_offs = ragged_metadata.slice_offs
    block_offs = ragged_metadata.block_offs(block_size_to_align)
    block_schedule = ragged_metadata.block_schedule(block_size_to_align)

    # output tensor with exact ragged rows/cols
    data = torch.empty(M, K, device=padded_data.device, dtype=padded_data.dtype)
    data.fill_(0.0)

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


class BlackwellActMXScaleLayout(Layout):
    # Swizzling for activation tensor [M, K], M can be ragged dimension and equals to sum of expert bs
    name: str = "BLACKWELL_SCALE"

    def __init__(self, shape, ragged_metadata: RaggedTensorMetadata | None = None) -> None:
        super().__init__(shape)
        if len(shape) == 2:
            (
                self.M,
                self.K,
            ) = shape
            self.B = 1
            self.mode = "ragged"
        else:
            assert len(shape) == 3, f"Only support 3D shape for BlackwellActMXScaleLayout, got {shape}"
            (
                self.B,
                self.M,
                self.K,
            ) = shape
            self.mode = "batched"
        self.ALIGN_K = 8
        self.ALIGN_M = 128
        self.SWIZZLE_K = 4
        self.K_pad = (self.K + self.ALIGN_K - 1) // self.ALIGN_K * self.ALIGN_K  # min multiple of ALIGN_K

        if self.mode == "batched":
            self.M_pad = (self.M + self.ALIGN_M - 1) // self.ALIGN_M * self.ALIGN_M
        else:
            # In ragged mode, input often include padded tokens
            # Out of M rows, the number of valid rows is the sum of ragged_metadata.slice_sizes
            # And the rest of rows are padded tokens
            self.ragged_metadata = ragged_metadata

            n_slices = ragged_metadata.slice_sizes.shape[0]
            max_n_blocks = ragged_metadata.n_blocks(
                n_slices, self.M, self.ALIGN_M
            )  # this estimates the number of blocks (each block has ALIGN_M rows) we need if we have all M valid tokens

            # create a static size scratchpad for output
            self.M_pad = self.ALIGN_M * max_n_blocks

    def swizzle_data(self, data):
        if self.mode == "batched":
            padded_data = torch.nn.functional.pad(
                data, (0, self.K_pad - self.K, 0, self.M_pad - self.M))  # value of padding on left, right, top, bottom
            padded_data = padded_data.reshape(self.B, self.M_pad // 128, 4, 32, self.K_pad // 4, 4)
            padded_data = padded_data.transpose(2, 4).contiguous()  # [1, M//128, K//4, 32, 4, 4]
            padded_data = padded_data.view(1, self.B * self.M_pad // 128, self.K_pad // 4, 2, 256)
        else:
            # Objective is to pad the number of rows in each slice to be multiple of ALIGN_M
            padded_data = pad_segments_triton(
                data,
                self.ragged_metadata,
                self.ALIGN_M,
                self.M_pad,
                self.K,
                self.K_pad,
            )

            padded_data = padded_data.reshape(self.B, self.M_pad // 128, 4, 32, self.K_pad // 4, 4)
            padded_data = padded_data.transpose(2, 4).contiguous()  # [1, M//128, K//4, 32, 4, 4]
            padded_data = padded_data.view(1, self.B * self.M_pad // 128, self.K_pad // 4, 2, 256)

        return padded_data

    def unswizzle_data(self, data):
        data = data.reshape(self.B, self.M_pad // 128, self.K_pad // 4, 32, 4, 4)
        data = data.transpose(2, 4)  # [B, M//128, 4, 32, K//4, 4]
        data = data.reshape(self.B, self.M_pad, self.K_pad)

        if self.mode == "batched":
            return data[..., :self.M, :self.K]

        # ragged path: map padded blocks back into the original ragged rows
        assert self.B == 1, "ragged scale layout only supports 2D input"
        data = unpad_segments_triton(
            data.squeeze(0),
            self.ragged_metadata,
            self.ALIGN_M,
            self.M,
            self.K,
            self.K_pad,
        )
        return data

    def swizzle_block_shape(self, block_shape):
        assert block_shape[0] >= 128, f"{block_shape[0]=} must be >= 128"
        return [1, block_shape[0] // 128, block_shape[1] // 4, 2, 256]


@triton.jit
def unswizzle_mx_scale_bw(
    x,
    SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,
    SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,
    ALIGN_INNER: tl.constexpr = SWIZZLE_ALIGN_INNER,
):
    shape_0: tl.constexpr = x.shape[0]
    shape_1: tl.constexpr = x.shape[1]
    tl.static_assert(shape_1 % SIZE_OUTER == 0)
    tl.static_assert(shape_1 // SIZE_OUTER <= ALIGN_INNER)
    x = x.reshape(shape_0, (shape_1 // SIZE_OUTER) // SIZE_INNER, 32, SIZE_OUTER // 32, SIZE_INNER)
    x = x.trans(0, 3, 2, 1, 4).reshape(shape_0 * SIZE_OUTER, shape_1 // SIZE_OUTER)
    return x


@triton.jit
def unswizzle_act_mx_scale_bw(x, SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,  # 128
                              SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,  # 4
                              ):
    # input block shape is [1, BLOCK_M//128, BLOCK_K//32//4, 2, 256] and we want to unswizzle it to [BLOCK_M, BLOCK_K//32]
    shape_1: tl.constexpr = x.shape[1]
    shape_2: tl.constexpr = x.shape[2]
    unswizzled_block_m: tl.constexpr = shape_1 * SIZE_OUTER  # BLOCK_M
    unswizzled_block_k: tl.constexpr = shape_2 * SIZE_INNER  # BLOCK_K // 32

    x = x.reshape(shape_1, shape_2, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(unswizzled_block_m, unswizzled_block_k)
    return x
