import pytest
import torch
from triton_kernels.tensor_details.layout import (
    BlackwellActMXScaleLayout,
    BlackwellMX4ValueShuffledLayout,
    BlackwellMXScaleLayout,
    BlackwellMXValueLayout,
    StridedLayout,
)
from triton_kernels.tensor_details.dtype import FP4
from triton_kernels.tensor import make_ragged_tensor_metadata, make_ragged_tensor_metadata_torch, wrap_torch_tensor, convert_layout

# ------------------------------------------------------------
# Torch tests
# ------------------------------------------------------------

ZERO_SIZED_SHAPES = [(0, 64), (64, 0), (2, 0), (0, 2), (0, 64, 64)]


def test_act_scale_storage_preservation():
    slice_sizes = torch.tensor([2, 3], dtype=torch.int32)
    metadata = make_ragged_tensor_metadata_torch(slice_sizes, 5)
    equivalent = BlackwellActMXScaleLayout(metadata)
    reconstructed = BlackwellActMXScaleLayout(make_ragged_tensor_metadata_torch(slice_sizes, 5))

    assert equivalent.can_preserve_storage_as(BlackwellActMXScaleLayout(metadata), 2)
    assert not equivalent.can_preserve_storage_as(reconstructed, 2)


@pytest.mark.parametrize("shape", ZERO_SIZED_SHAPES)
@pytest.mark.parametrize("layout", [BlackwellMXScaleLayout(), BlackwellActMXScaleLayout(None)])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_scale_zero_sized_roundtrip(shape, layout, device):
    x = torch.empty(shape, dtype=torch.uint8, device=device)
    src = wrap_torch_tensor(x)

    swizzled = convert_layout(src, layout)
    roundtrip = convert_layout(swizzled, src.storage.layout)

    assert roundtrip.storage.data.shape == x.shape


@pytest.mark.parametrize("shape", ZERO_SIZED_SHAPES)
@pytest.mark.parametrize("layout", [BlackwellMXValueLayout(), BlackwellMX4ValueShuffledLayout()])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_value_zero_sized_roundtrip(shape, layout, device):
    x = torch.empty(shape, dtype=torch.uint8, device=device)
    src = wrap_torch_tensor(x, dtype=FP4)

    swizzled = convert_layout(src, layout)
    roundtrip = convert_layout(swizzled, src.storage.layout)

    assert roundtrip.storage.data.shape == x.shape


@pytest.mark.parametrize(("slice_sizes", "shape"), [([0], (0, 64)), ([2, 0], (2, 0))])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_act_scale_zero_sized_ragged_roundtrip(slice_sizes, shape, device):
    metadata = make_ragged_tensor_metadata_torch(torch.tensor(slice_sizes, dtype=torch.int32), shape[0])
    x = torch.empty(shape, device=device)
    src = wrap_torch_tensor(x)

    swizzled = convert_layout(src, BlackwellActMXScaleLayout(metadata))
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
    layout = BlackwellMXScaleLayout()
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    assert (res == x).all()


def test_mxfp4_scale_roundtrip_noncontiguous():
    x = torch.randint(0, 256, (2, 16, 1024), dtype=torch.uint8, device="cuda")[..., ::2]
    assert not x.is_contiguous()
    layout = BlackwellMXScaleLayout()
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    assert torch.equal(res, x)


def test_mxfp4_scale_swizzle_meta():
    x = torch.empty((2, 16, 32), dtype=torch.uint8, device="meta")
    layout = BlackwellMXScaleLayout()
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    swizzled = transformation.swizzle_data(x)
    assert swizzled.device.type == "meta"
    assert swizzled.shape == (1, 2, 4, 2, 256)


@pytest.mark.parametrize("shape", [(2, 256, 192), (1, 128, 64)])
def test_act_scale_roundtrip_batched(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    layout = BlackwellActMXScaleLayout(ragged_metadata=None)
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    torch.testing.assert_close(res, x)


@pytest.mark.parametrize("shape", [(256, 192), (128, 64), (130, 65)])
def test_act_scale_roundtrip_2d_without_ragged_metadata(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    layout = BlackwellActMXScaleLayout(ragged_metadata=None)
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    assert transformation.mode == "batched"
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    assert res.shape == shape
    assert torch.equal(res, x)


@pytest.mark.parametrize("shape", [(256, 192), (128, 64), (130, 65)])
def test_act_scale_convert_layout_roundtrip_2d_without_ragged_metadata(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    x_tri = wrap_torch_tensor(x)
    scale_layout = BlackwellActMXScaleLayout(ragged_metadata=None)
    x_tri_scale = convert_layout(x_tri, scale_layout)
    x_tri_roundtrip = convert_layout(x_tri_scale, StridedLayout(-1))
    assert torch.equal(x_tri_roundtrip.data, x)


@pytest.mark.parametrize(
    "slice_sizes, m, k, align_m",
    [
        ([17, 0, 33, 5], 100, 94, 8),
        ([1, 2, 3, 4, 5], 50, 15, 16),
    ],
)
def test_act_scale_roundtrip_ragged(slice_sizes, m, k, align_m):
    slice_sizes = torch.tensor(slice_sizes, device="cuda", dtype=torch.int32)
    m = max(m, slice_sizes.sum().item())  # there can be padded tokens in the input
    ragged_metadata = make_ragged_tensor_metadata(slice_sizes, m)
    x = torch.randn((m, k), device="cuda", dtype=torch.float32)
    layout = BlackwellActMXScaleLayout(ragged_metadata=ragged_metadata)
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))

    x_useful_rows = x[ragged_metadata.slice_offs[:-1], :]
    res_useful_rows = res[ragged_metadata.slice_offs[:-1], :]
    torch.testing.assert_close(res_useful_rows, x_useful_rows)
