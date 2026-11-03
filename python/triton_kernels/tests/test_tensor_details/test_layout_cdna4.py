import pytest
import torch
from triton_kernels.tensor_details.layout import CDNA4MXScaleLayout
from triton_kernels.tensor import convert_layout, wrap_torch_tensor

# ------------------------------------------------------------
# Torch tests
# ------------------------------------------------------------


@pytest.mark.parametrize("shape", [(0, 64), (64, 0), (2, 0), (0, 2), (0, 64, 64)])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_mxfp4_scale_zero_sized_roundtrip(shape, device):
    x = torch.empty(shape, dtype=torch.uint8, device=device)
    src = wrap_torch_tensor(x)

    swizzled = convert_layout(src, CDNA4MXScaleLayout())
    roundtrip = convert_layout(swizzled, src.storage.layout)

    assert roundtrip.storage.data.shape == x.shape


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
def test_mxfp4_scale_roundtrip(shape):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    layout = CDNA4MXScaleLayout()
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    assert (res == x).all()
