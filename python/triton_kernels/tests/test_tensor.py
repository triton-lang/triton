import torch
import pytest
import math
from triton_kernels.testing import assert_equal
from triton_kernels.tensor_details.layout import BlackwellMXScaleLayout, HopperMXScaleLayout, HopperMXValueLayout


@pytest.mark.parametrize(
    "shape",
    [
        (3, 4096, 1024),
        (10, 254, 60),
        (1, 320, 160),
        (2, 16, 512),
        (3, 2, 36),
    ],
)
def test_mxfp_swizzle(shape: tuple[int, ...]):
    """
    Test that unswizzle is the inverse of swizzle, after removing padding.
    """
    x = torch.randn(shape, device="cuda")
    layout = BlackwellMXScaleLayout(shape)
    assert_equal(x, layout.unswizzle_data(layout.swizzle_data(x)))


@pytest.mark.parametrize("shape", [(16, 32), (16, 64), (32, 32), (32, 64), (64, 128), (128, 128)])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("op_idx", [0, 1])
@pytest.mark.parametrize("mma_version", [2, 3])
def test_swizzle_mxfp4_value(shape, trans, op_idx, mma_version):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    if trans:
        x = x.mT
    k_dim = 1 - op_idx
    if x.shape[k_dim] < 32:
        pytest.skip("Not enough elements along K")
    layout = HopperMXValueLayout(x.shape, op_idx, mma_version)
    res = layout.unswizzle_data(layout.swizzle_data(x))
    assert (res == x).all()


@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("shape", [(256, 64), (256, 128), (256, 256)])
def test_swizzle_mxfp4_scale(shape, num_warps):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    layout = HopperMXScaleLayout(x.shape, num_warps=num_warps)
    res = layout.unswizzle_data(layout.swizzle_data(x))
    assert (res[:shape[0], :shape[1]] == x).all()


def test_unswizzle_mxfp4_value_golden_value():
    shape = (16, 32)
    x = torch.arange(math.prod(shape)).view(shape).to(torch.uint8)
    layout = HopperMXValueLayout(x.shape, op_idx=1, mma_version=3)
    res = layout.swizzle_data(x)
    # Thread 0
    assert res[0, 0:16].tolist() == [0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28]
    # Thread 1
    assert res[0, 16:32].tolist() == [1, 1, 5, 5, 9, 9, 13, 13, 17, 17, 21, 21, 25, 25, 29, 29]
