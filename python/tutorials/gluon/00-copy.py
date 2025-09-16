import pytest
import torch
import triton
from triton.experimental.gluon import jit
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    constexpr, program_id, BlockedLayout, SwizzledSharedLayout,
    allocate_shared_memory, SliceLayout, DistributedLinearLayout, DotOperandLayout,
    AutoLayout
)
from triton.experimental.gluon.language.amd import (
    AMDMFMALayout, cdna4, cdna3, in_thread_transpose,
    AMDRotatingSharedLayout
)

import os
os.environ["TRITON_CACHE_DIR"] = "/home/sijieli2/gluon_cache"

@jit
def grid(
        row_step, col_step, layout,
        row_end: constexpr, col_end: constexpr,
        row_start: constexpr = 0, col_start: constexpr = 0
    ):
    off_row = gl.arange(row_start, row_end, layout=SliceLayout(1, layout)) * row_step
    off_col = gl.arange(col_start, col_end, layout=SliceLayout(0, layout)) * col_step
    return off_row[:, None] + off_col[None, :]


@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=1, num_warps=4)
    ],
    key=['M', 'N', 'K'],
)
@jit
def copy_kernel0(
        y_ptr, x_ptr,
        stride_ym, stride_yn,
        stride_xm, stride_xn,
        BLOCK_SIZE_M: constexpr,
        BLOCK_SIZE_N: constexpr,
    ):

    blocked_layout1: constexpr = BlockedLayout(
        size_per_thread=(2, 8),
        threads_per_warp=(4, 16),
        warps_per_cta=(4, 1),
        order=(0,1)
    )

    blocked_layout: constexpr = DistributedLinearLayout(
        reg_bases=((0,1), (0,2), (0,4)), # 16
        lane_bases=((0,8), (0,16), (0,32), (0,64), (1,0), (2,0)), # 64
        warp_bases=((4,0), (8,0)), # 4
        block_bases=[], # 8
        shape=[16, BLOCK_SIZE_N],
    )

    blocked_layout0: constexpr = DistributedLinearLayout(
        reg_bases=((0,1), (0,2), (0,4), (1,0)), # 16
        lane_bases=((2,0), (4,0), (0,8), (0,16), (0,32), (0,64)), # 64
        warp_bases=((8,0), (16,0)), # 4
        block_bases=(), # 8
        shape=(32, BLOCK_SIZE_N)
    )

    pid_m = program_id(0) * BLOCK_SIZE_M
    pid_n = program_id(1) * BLOCK_SIZE_N

    x_ptr += pid_m * stride_xm + pid_n * stride_xn
    y_ptr += pid_m * stride_ym + pid_n * stride_yn

    offs_x = grid(stride_xm, stride_xn, blocked_layout0, BLOCK_SIZE_M, BLOCK_SIZE_N)
    offs_y = grid(stride_ym, stride_yn, blocked_layout1, BLOCK_SIZE_M, BLOCK_SIZE_N)

    x = cdna3.buffer_load(x_ptr, offs_x)
    y = gl.convert_layout(x, blocked_layout1)

    cdna3.buffer_store(y, y_ptr, offs_y)


def copy(x):
    M, N = x.shape

    y = torch.empty_like(x)

    grid0 = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    copy_kernel0[grid0](
        y, x,
        1, y.shape[0],
        # y.stride(0), y.stride(1),
        x.stride(0), x.stride(1),
    )
    return y

x = torch.randn((512, 512), device="cuda")
y = copy(x)

print(f"{x=}\n{y=}")
torch.testing.assert_close(y, x, rtol=3e-5, atol=3e-5)