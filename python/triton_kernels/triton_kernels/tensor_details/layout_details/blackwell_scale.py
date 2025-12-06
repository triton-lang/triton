import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .base import Layout

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
        data = data.reshape(
            self.B, self.N_pad // self.ALIGN_N, self.ALIGN_N // 32, 32, self.K_pad // self.SWIZZLE_K, self.SWIZZLE_K
        )
        data = data.transpose(2, 4).contiguous()
        data = data.view(1, self.B * self.N_pad // 128, self.K_pad // self.SWIZZLE_K, 2, 256)
        return data

    def unswizzle_data(self, data):
        data = data.reshape(
            self.B, self.N_pad // self.ALIGN_N, self.K_pad // self.SWIZZLE_K, 32, self.ALIGN_N // 32, self.SWIZZLE_K
        )
        data = data.transpose(2, 4)
        data = data.reshape(*self.leading_shape, self.N_pad, self.K_pad)
        data = data.transpose(-1, -2)
        return data[..., : self.K, : self.N]

    def swizzle_block_shape(self, block_shape):
        assert block_shape[0] >= 128, f"{block_shape[0]=} must be >= 128"
        return [1, block_shape[0] // 128, block_shape[1] // 4, 2, 256]


@triton.jit
def pad_segments_kernel(
    data_ptr,                     # *input* base pointer
    ex_hist_ptr,                  # int32[H]
    ex_hist_padded_ptr,           # int32[H]
    in_offsets_ptr,               # int32[H] row offsets in `data`
    out_offsets_ptr,              # int32[H] row offsets in `padded_data`
    K: tl.constexpr,              # input width
    K_pad: tl.constexpr,          # output width
    stride_in_m, stride_in_n,     # data.stride(0), data.stride(1)
    stride_out_m, stride_out_n,   # padded_data.stride(0), stride(1)
    out_ptr,                      # *output* base pointer
    BLOCK_M: tl.constexpr,        # block of rows
    BLOCK_N: tl.constexpr,        # block of cols
):
    # program ids
    pid_seg = tl.program_id(0)    # which segment i
    pid_row_blk = tl.program_id(1) # which row-block within padded segment

    # segment-specific metadata
    m      = tl.load(ex_hist_ptr + pid_seg)            # actual rows
    m_pad  = tl.load(ex_hist_padded_ptr + pid_seg)     # padded rows
    in_base_rows  = tl.load(in_offsets_ptr  + pid_seg) # row offset in `data`
    out_base_rows = tl.load(out_offsets_ptr + pid_seg) # row offset in `out`

    # row indices (within this segment)
    row_offsets = pid_row_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)

    # global row indices in input/output
    out_rows = out_base_rows + row_offsets[:, None]
    out_cols = col_offsets[None, :]

    in_rows = in_base_rows + row_offsets[:, None]
    in_cols = col_offsets[None, :]

    # masks
    row_in_range_out = row_offsets < m_pad
    col_in_range_out = col_offsets < K_pad
    out_mask = row_in_range_out[:, None] & col_in_range_out[None, :]

    row_in_range_in = row_offsets < m
    col_in_range_in = col_offsets < K
    in_mask = row_in_range_in[:, None] & col_in_range_in[None, :]

    # default pad value = 0
    # load from input where valid, else 0
    vals = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # compute linear ptrs with strides
    in_ptrs = data_ptr + in_rows * stride_in_m + in_cols * stride_in_n
    vals = tl.load(in_ptrs, mask=in_mask & out_mask, other=0.0)

    # store into output
    out_ptrs = out_ptr + out_rows * stride_out_m + out_cols * stride_out_n
    tl.store(out_ptrs, vals, mask=out_mask)


def pad_segments_triton(data, ex_hist, ex_hist_padded, K, K_pad, total_padded_rows):
    """
    data:           [total_M, K]       (contiguous or with known strides)
    ex_hist:        [H] int32/long     (segment lengths)
    ex_hist_padded: [H] int32/long     (padded lengths)
    returns padded_data: [sum(ex_hist_padded), K_pad]
    """
    assert data.is_cuda
    assert ex_hist.is_cuda and ex_hist_padded.is_cuda
    assert ex_hist.dim() == 1 and ex_hist_padded.dim() == 1
    assert ex_hist.shape == ex_hist_padded.shape

    device = data.device
    H = ex_hist.shape[0]

    ex_hist_int = ex_hist.to(torch.int32)
    ex_hist_padded_int = ex_hist_padded.to(torch.int32)

    # prefix sums in *rows*
    # in_offsets[i] = sum_{j < i} ex_hist[j]
    # out_offsets[i] = sum_{j < i} ex_hist_padded[j]

    # from exp_hist to offset format
    in_offsets = torch.empty_like(ex_hist_int)
    out_offsets = torch.empty_like(ex_hist_padded_int)

    if H > 0:
        in_offsets.zero_()
        out_offsets.zero_()
        if H > 1:
            torch.cumsum(ex_hist_int[:-1], dim=0, out=in_offsets[1:])
            torch.cumsum(ex_hist_padded_int[:-1], dim=0, out=out_offsets[1:])

    # total_padded_rows = int(ex_hist_padded_int.sum().item())
    padded_data = torch.empty(total_padded_rows, K_pad, device=data.device, dtype=data.dtype)

    # strides (in elements, not bytes)
    stride_in_m, stride_in_n = data.stride()
    stride_out_m, stride_out_n = padded_data.stride()

    BLOCK_M = 32
    BLOCK_N = 64

    grid = (
        H,
        triton.cdiv(128*2, BLOCK_M),
    )

    pad_segments_kernel[grid](
        data,
        ex_hist_int,
        ex_hist_padded_int,
        in_offsets,
        out_offsets,
        K,
        K_pad,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        padded_data,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return padded_data

class BlackwellActMXScaleLayout(Layout):
    # Swizzling for activation tensor [M, K], M can be ragged dimension and equals to sum of expert bs
    name: str = "BLACKWELL_SCALE"

    def __init__(self, shape, ex_hist: torch.Tensor | None = None) -> None:
        super().__init__(shape)
        if len(shape) == 2:
            (
                self.M,  # sum of expert bs
                self.K,
            ) = shape
            self.B = 1
            self.mode = "ragged"
        else:
            assert len(shape) == 3, f"Only support 3D shape for BlackwellActMXScaleLayout, got {shape}"
            (
                *self.leading_shape,
                self.M,  # sum of expert bs
                self.K,
            ) = shape
            self.B = math.prod(self.leading_shape)
            self.mode = "batched"
        self.ALIGN_K = 8
        self.ALIGN_M = 128
        self.SWIZZLE_K = 4
        self.K_pad = (self.K + self.ALIGN_K - 1) // self.ALIGN_K * self.ALIGN_K  # min multiple of ALIGN_K

        if self.mode == "batched":
            self.M_pad = (self.M + self.ALIGN_M - 1) // self.ALIGN_M * self.ALIGN_M
        else:

            # if ex_hist is None:
            #     ex_hist = [self.M]
            self.ex_hist = ex_hist

            # pad each ex_hist to be the min multiple of ALIGN_M
            # self.ex_hist_padded = [
            #     (ex_hist[i] + self.ALIGN_M - 1) // self.ALIGN_M * self.ALIGN_M for i in range(len(ex_hist))
            # ]
            self.ex_hist_padded = ((ex_hist + self.ALIGN_M - 1) // self.ALIGN_M) * self.ALIGN_M
            # self.M_pad = sum(self.ex_hist_padded)
            # print(f"ex_hist_padded {self.ex_hist_padded}")

            # upper bound static padded rows: need to be static not related to ex_hist
            n_experts = ex_hist.shape[0]
            self.M_pad = 128 * 8 * n_experts # each expert allow 8 128 blocks

    def swizzle_data(self, data):
        if self.mode == "batched":
            padded_data = torch.nn.functional.pad(data, (0, self.K_pad - self.K, 0, self.M_pad - self.M))  # value of padding on left, right, top, bottom
            padded_data = padded_data.reshape(self.B, self.M_pad // 128, 4, 32, self.K_pad // 4, 4)
            padded_data = padded_data.transpose(2, 4).contiguous()  # [1, M//128, K//4, 32, 4, 4]
            padded_data = padded_data.view(1, self.B * self.M_pad // 128, self.K_pad // 4, 2, 256)
        else:
            padded_data = pad_segments_triton(
                data,
                ex_hist=self.ex_hist,
                ex_hist_padded=self.ex_hist_padded,
                K=self.K,
                K_pad=self.K_pad,
                total_padded_rows=self.M_pad,
            )
            # print(f"padded_data after swizzling {padded_data.shape}")
            padded_data = padded_data.reshape(self.B, self.M_pad // 128, 4, 32, self.K_pad // 4, 4)
            padded_data = padded_data.transpose(2, 4).contiguous()  # [1, M//128, K//4, 32, 4, 4]
            padded_data = padded_data.view(1, self.B * self.M_pad // 128, self.K_pad // 4, 2, 256)

        return padded_data

    def unswizzle_data(self, data):
        # Kernel to reserve unswizzled data
        data = data.reshape(self.B, self.M_pad // 128, self.K_pad // 4, 32, 4, 4)
        data = data.transpose(2, 4)  # [B, M//128, 4, 32, K//4, 4]
        data = data.reshape(*self.leading_shape, self.M_pad, self.K_pad)
        return data[..., :self.M, :self.K]

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
def unswizzle_act_mx_scale_bw(
    x,
    SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,  # 128
    SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,  # 4
):
    # input block shape is [1, BLOCK_M//128, BLOCK_K//32//4, 2, 256] and we want to unswizzle it to [BLOCK_M, BLOCK_K//32]
    shape_1: tl.constexpr = x.shape[1]
    shape_2: tl.constexpr = x.shape[2]
    unswizzled_block_m: tl.constexpr = shape_1 * SIZE_OUTER  # BLOCK_M
    unswizzled_block_k: tl.constexpr = shape_2 * SIZE_INNER  # BLOCK_K // 32

    x = x.reshape(shape_1, shape_2, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(unswizzled_block_m, unswizzled_block_k)
    return x
