import pytest
import torch
from triton_kernels.tensor_details.layout import (
    BlackwellActMXScaleLayout,
    BlackwellMX4ValuePackedShuffledLayout,
    BlackwellMXScaleLayout,
    StridedLayout,
)
from triton_kernels.tensor import make_ragged_tensor_metadata, wrap_torch_tensor, convert_layout

# ------------------------------------------------------------
# Torch tests
# ------------------------------------------------------------


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(256, 512), (384, 768), (2, 384, 768)])
def test_mxfp4_value_packed_shuffled_roundtrip(shape):
    storage_shape = (*shape[:-1], shape[-1] // 2)
    x = torch.randint(0, 256, storage_shape, dtype=torch.uint8, device="cuda")
    layout = BlackwellMX4ValuePackedShuffledLayout()
    transformation = layout.make_transformation(list(shape), is_fp4=True)

    swizzled = transformation.swizzle_data(x)
    result = transformation.unswizzle_data(swizzled)

    assert torch.equal(result, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mxfp4_value_packed_shuffled_column_mapping():
    shape = (256, 512)
    x = torch.arange(shape[0] * (shape[1] // 2), dtype=torch.int32, device="cuda").to(torch.uint8)
    x = x.reshape(shape[0], shape[1] // 2)
    layout = BlackwellMX4ValuePackedShuffledLayout()
    transformation = layout.make_transformation(list(shape), is_fp4=True)

    swizzled = transformation.swizzle_data(x)
    physical = transformation._canonical_to_physical(x).reshape(1, shape[0] // 2, shape[1])

    assert tuple(swizzled.shape) == (1, 1, 1, 512, 128)
    for n_inner in [0, 1, 7, 8, 255, 511]:
        phase = (n_inner % 8) * 16
        for k_tile in [0, 1]:
            for k_byte in [0, 1, 7, 8, 31, 63]:
                physical_col = 16 * (k_byte // 8) + 8 * k_tile + (k_byte % 8)
                tma_col = physical_col ^ phase
                expected = physical[0, k_tile * 64 + k_byte, n_inner]
                assert swizzled[0, 0, 0, n_inner, tma_col] == expected


def test_mxfp4_value_packed_shuffled_block_shape():
    layout = BlackwellMX4ValuePackedShuffledLayout()
    assert layout.swizzle_block_shape([1, 256, 512]) == [1, 1, 1, 512, 256]
    with pytest.raises(ValueError, match="one packed pair"):
        layout.swizzle_block_shape([1, 128, 512])
