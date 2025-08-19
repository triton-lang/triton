import torch
import pytest

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size


@pytest.mark.parametrize("M, N", [(32, 16), (32, 32), (32, 64), (64, 32)])
@pytest.mark.parametrize("src_layout", [
    ttgl.BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [0, 1]),
    ttgl.BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [0, 1]),
    ttgl.BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [0, 1]),
    ttgl.BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [0, 1]),
    ttgl.BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [0, 1]),
    ttgl.BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0]),
    ttgl.BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [1, 0]),
    ttgl.BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [1, 0]),
    ttgl.BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [1, 0]),
    ttgl.BlockedLayout([1, 2], [1, THREADS_PER_WARP], [1, 4], [1, 0]),
])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sanitize_overflow", [False, True])
def test_scan_layouts(M, N, src_layout, axis, sanitize_overflow, device):

    @gluon.jit
    def _combine(a, b):
        return a + b

    @gluon.jit
    def kernel(x_ptr, z_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr, axis: ttgl.constexpr):
        x_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
        x_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
        x = ttgl.load(x_ptr + x_offs_m * N + x_offs_n)
        y = ttgl.associative_scan(x, axis=axis, combine_fn=_combine)
        ttgl.store(z_ptr + x_offs_m * N + x_offs_n, y)

    torch.manual_seed(0)

    x = torch.randint(-100, 100, (M, N), dtype=torch.int32, device=device)
    z = torch.zeros((M, N), dtype=torch.int32, device=device)
    z_tri = torch.empty_like(z)

    kernel[(1, 1, 1)](x, z_tri, M, N, src_layout, axis, num_warps=4, sanitize_overflow=sanitize_overflow,
                      debug=sanitize_overflow)

    z_ref = torch.cumsum(x, dim=axis, dtype=torch.int32)
    torch.testing.assert_close(z_tri, z_ref)
