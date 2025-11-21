import pytest
import torch
from triton_kernels.tensor_details.layout import CDNA4MXScaleLayout

# ------------------------------------------------------------
# Torch tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (3, 4096, 1024),
        (10, 56, 256),
        (1, 320, 160),
        (2, 8, 32),
        (3, 24, 576),
    ],
)
def test_mxfp4_scale_roundtrip(shape):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    layout = CDNA4MXScaleLayout(x.shape)
    res = layout.unswizzle_data(layout.swizzle_data(x))
    assert (res == x).all()
