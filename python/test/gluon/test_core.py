import torch
import pytest

from triton._internal_testing import is_cuda
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.hopper import tma


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


@gluon.jit
def tma_kernel(desc):
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
    value = ttgl.full(desc.block_shape, 0, desc.dtype, layout)
    alloc = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout, value)
    tma.async_copy_shared_to_global(desc, [0, 0], alloc)
    tma.store_wait(0)
    alloc._keep_alive()


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires Hopper")
def test_tma():
    out = torch.ones((16, 16), dtype=torch.float16, device="cuda")
    layout = ttgl.NVMMASharedLayout(
        swizzle_byte_width=32,
        element_bitwidth=16,
        rank=2,
        transposed=False,
        fp4_padded=False,
    )

    desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(out, [16, 16], layout)
    tma_kernel[(1, )](desc)
    torch.testing.assert_close(out, torch.zeros_like(out))
