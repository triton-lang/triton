import pytest
import torch
from triton_kernels.tensor_details.dtype import BIT, FP4, UINT8
from triton_kernels.tensor import (
    convert_layout,
    make_ragged_tensor_metadata,
    make_ragged_tensor_metadata_torch,
    remap_ragged_tensor_metadata,
    remap_ragged_tensor_metadata_torch,
    make_bitmatrix_metadata,
    make_bitmatrix_metadata_torch,
    wrap_torch_tensor,
)
from triton_kernels.testing import assert_equal
from triton_kernels.tensor_details.layout import (
    BlackwellActMXScaleLayout,
    BlackwellMX4ValueShuffledLayout,
    BlackwellMXScaleLayout,
    BlackwellMXValueLayout,
    CDNA4MXScaleLayout,
    GFX1250MXScaleLayout,
    HopperMXScaleLayout,
    HopperMXValueLayout,
    StridedLayout,
)


@pytest.mark.parametrize(
    ("transpose", "layout"),
    [
        (False, StridedLayout(-1)),
        (False, StridedLayout(1)),
        (True, StridedLayout(-2)),
        (True, StridedLayout(0)),
    ],
)
def test_convert_layout_noop(transpose, layout):
    data = torch.randn((7, 11))
    if transpose:
        data = data.T
    tensor = wrap_torch_tensor(data)

    assert convert_layout(tensor, layout) is tensor


def test_convert_layout_noop_preserves_strided_view():
    tensor = wrap_torch_tensor(torch.randn((14, 11))[::2])

    assert convert_layout(tensor, StridedLayout(-1)) is tensor
    assert tensor.storage.data.stride() == (22, 1)


def test_convert_layout_rejects_strided_view_without_contiguous_dimension():
    tensor = wrap_torch_tensor(torch.randn((14, 22))[::2, ::2])

    with pytest.raises(ValueError):
        convert_layout(tensor, tensor.storage.layout)


def test_convert_layout_noop_does_not_ignore_transformation_kwargs():
    tensor = wrap_torch_tensor(torch.randn((7, 11)))

    with pytest.raises(TypeError):
        convert_layout(tensor, tensor.storage.layout, unsupported=True)


@pytest.mark.parametrize(
    ("storage_shape", "logical_shape", "dtype", "layout", "equivalent_layout"),
    [
        ((10, 254, 60), None, UINT8, BlackwellMXScaleLayout(), BlackwellMXScaleLayout()),
        ((130, 65), None, UINT8, BlackwellActMXScaleLayout(None), BlackwellActMXScaleLayout(None)),
        ((256, 64), (256, 128), FP4, BlackwellMXValueLayout(), BlackwellMXValueLayout()),
        ((128, 256), (128, 512), FP4, BlackwellMX4ValueShuffledLayout(), BlackwellMX4ValueShuffledLayout()),
        ((70, 65), None, UINT8, HopperMXScaleLayout(-2, 4), HopperMXScaleLayout(-2, 4)),
        ((64, 64), (64, 128), FP4, HopperMXValueLayout(-2, 3), HopperMXValueLayout(-2, 3)),
        ((10, 254, 60), None, UINT8, CDNA4MXScaleLayout(), CDNA4MXScaleLayout()),
        ((10, 254, 60), None, UINT8, GFX1250MXScaleLayout(), GFX1250MXScaleLayout()),
    ],
)
def test_convert_layout_noop_for_equivalent_layout(storage_shape, logical_shape, dtype, layout, equivalent_layout):
    tensor = wrap_torch_tensor(torch.randint(0, 256, storage_shape, dtype=torch.uint8), dtype=dtype,
                               shape=logical_shape)
    converted = convert_layout(tensor, layout)

    assert converted is not tensor
    assert convert_layout(converted, equivalent_layout) is converted


@pytest.mark.parametrize(
    ("storage_shape", "logical_shape", "dtype", "layout", "different_layout"),
    [
        ((70, 65), None, UINT8, HopperMXScaleLayout(-2, 4), HopperMXScaleLayout(-2, 8)),
        ((64, 64), (64, 128), FP4, HopperMXValueLayout(-2, 3), HopperMXValueLayout(-2, 2)),
        ((128, 256), (128, 512), FP4, BlackwellMX4ValueShuffledLayout(), BlackwellMX4ValueShuffledLayout(block_n=128)),
    ],
)
def test_convert_layout_converts_different_parameterized_layout(storage_shape, logical_shape, dtype, layout,
                                                                different_layout):
    tensor = wrap_torch_tensor(torch.randint(0, 256, storage_shape, dtype=torch.uint8), dtype=dtype,
                               shape=logical_shape)
    converted = convert_layout(tensor, layout)

    assert convert_layout(converted, different_layout) is not converted


@pytest.mark.parametrize("n_slices", [1, 7, 33, 911, 1025])
def test_make_ragged_tensor_metadata(n_slices):
    torch.manual_seed(0)
    device = "cuda"
    max_slice_size = 200
    n_total_rows = max_slice_size * n_slices
    slice_sizes = torch.randint(0, max_slice_size, (n_slices, ), dtype=torch.int32, device=device)
    slice_sizes[torch.randint(0, n_slices, (1, ))] = 0
    meta = make_ragged_tensor_metadata(slice_sizes, n_total_rows)
    ref = make_ragged_tensor_metadata_torch(slice_sizes, n_total_rows)
    assert_equal(meta.slice_sizes, ref.slice_sizes)
    assert_equal(meta.slice_offs, ref.slice_offs)
    assert_equal(meta.block_offs_data, ref.block_offs_data)
    assert_equal(meta.block_schedule_data, ref.block_schedule_data)


@pytest.mark.parametrize("n_slices", [9, 32, 911, 1025])
def test_remap_ragged_tensor_metadata(n_slices):
    device = "cuda"
    max_slice_size = 200
    n_total_rows = max_slice_size * n_slices
    slice_sizes = torch.randint(0, max_slice_size, (n_slices, ), dtype=torch.int32, device=device)
    slice_sizes[torch.randint(0, n_slices, (1, ))] = 0
    # randomly permute slices
    slice_map = torch.randperm(n_slices, device=device, dtype=torch.int32)
    # discard random slices
    slice_map[torch.randint(0, len(slice_map), (5, ))] = -1
    tri_metadata = make_ragged_tensor_metadata(slice_sizes, n_total_rows)
    ref_metadata = make_ragged_tensor_metadata_torch(slice_sizes, n_total_rows)
    tri_metadata = remap_ragged_tensor_metadata(tri_metadata, slice_map)
    ref_metadata = remap_ragged_tensor_metadata_torch(ref_metadata, slice_map)
    assert_equal(tri_metadata.slice_sizes, ref_metadata.slice_sizes)
    assert_equal(tri_metadata.slice_offs, ref_metadata.slice_offs)
    assert_equal(tri_metadata.block_offs_data, ref_metadata.block_offs_data)
    assert_equal(tri_metadata.block_schedule_data, ref_metadata.block_schedule_data)


@pytest.mark.parametrize("n_rows", [7, 256, 17111])
@pytest.mark.parametrize("n_cols", [13, 32, 128, 811])
@pytest.mark.parametrize("k", [1, 4, 8])
def test_make_bitmatrix_metadata(n_rows, n_cols, k):
    if k > n_cols:
        pytest.skip("k must be <= n_cols")
    device = "cuda"
    torch.manual_seed(0)
    # random permutation of column indices
    # NOTE: `indx` *must* be sorted
    indx = torch.rand(n_rows, n_cols, device=device).argsort(dim=1).int()[:, :k]
    indx = torch.sort(indx, dim=1)[0]
    # create bitmask
    rows = torch.arange(n_rows, device=device).unsqueeze(1).expand_as(indx)
    bitmask_data = torch.zeros((n_rows, (n_cols + 31) // 32), dtype=torch.int32, device=device)
    bitmask_data.index_put_((rows, indx // 32), 1 << (indx % 32), accumulate=True)
    bitmask = wrap_torch_tensor(bitmask_data.view(torch.uint32), dtype=BIT, shape=(n_rows, n_cols))
    # make metadata and compare
    metadata_tri = make_bitmatrix_metadata(indx, bitmask)
    metadata_ref = make_bitmatrix_metadata_torch(indx, bitmask)
    assert_equal(metadata_tri.col_sum, metadata_ref.col_sum)
    assert_equal(metadata_tri.row_sorted_indx, metadata_ref.row_sorted_indx)
    assert_equal(metadata_tri.col_sorted_indx, metadata_ref.col_sorted_indx)
