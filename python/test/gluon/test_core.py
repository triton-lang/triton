import torch
import pytest

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


@gluon.jit
def copy_kernel(Out, In, numel, XBLOCK: ttgl.constexpr, layout: ttgl.constexpr):
    xbase = ttgl.program_id(0) * XBLOCK
    xoffset = xbase + ttgl.arange(0, XBLOCK, layout=layout)
    xmask = xoffset < numel
    data = ttgl.load(In + xoffset, xmask)
    ttgl.store(Out + xoffset, data, xmask)


@pytest.mark.parametrize("layout", [
    ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[2], threads_per_warp=[32], warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[4], threads_per_warp=[32], warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[8], threads_per_warp=[32], warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[8], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[2], threads_per_warp=[32], warps_per_cta=[8], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[4], threads_per_warp=[32], warps_per_cta=[8], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[8], threads_per_warp=[32], warps_per_cta=[8], order=[0]),
])
@pytest.mark.parametrize("XBLOCK", [128, 256, 512, 1024, 2048])
def test_copy_kernel(layout, XBLOCK):
    inp = torch.randn(XBLOCK * 4 - 7, device="cuda")
    out = torch.empty_like(inp)

    copy_kernel[(4, )](out, inp, inp.numel(), XBLOCK, layout, num_warps=layout.warps_per_cta[0])
    torch.testing.assert_close(out, inp)
