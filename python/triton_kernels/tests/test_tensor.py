import pytest
import torch
from triton_kernels.tensor import (
    make_ragged_tensor_metadata,
    make_ragged_tensor_metadata_torch,
    make_bitmatrix_metadata,
    make_bitmatrix_metadata_torch,
)
from triton_kernels.tensor_details.ragged_tensor import filter_ragged_tensor_metadata, filter_ragged_tensor_metadata_torch
from triton_kernels.topk import topk
from triton_kernels.testing import assert_equal
from triton_kernels.distributed import make_expt_assignment
from .test_distributed import _make_expt_dict_for_mode


@pytest.mark.parametrize("n_slices", [1, 7, 33, 911])
def test_make_ragged_tensor_metadata(n_slices):
    torch.manual_seed(0)
    device = "cuda"
    max_batch_size = 200
    max_n_blocks = (1 + (max_batch_size // 16)) * n_slices
    batch_sizes = torch.randint(0, max_batch_size, (n_slices, ), dtype=torch.int32, device=device)
    batch_sizes[torch.randint(0, n_slices, (1, ))] = 0
    meta = make_ragged_tensor_metadata(batch_sizes, max_n_blocks)
    ref = make_ragged_tensor_metadata_torch(batch_sizes, max_n_blocks)
    assert_equal(meta.slice_sizes, ref.slice_sizes)
    assert_equal(meta.slice_offs, ref.slice_offs)
    assert_equal(meta.block_offs_data, ref.block_offs_data)
    assert_equal(meta.block_schedule_data, ref.block_schedule_data)


@pytest.mark.parametrize("n_slices", [9, 32, 911, 1024])
@pytest.mark.parametrize("n_shards", [1, 2, 4, 8])
@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("affinity_mode", ["uniform", "random"])
def test_filter_ragged_tensor_metadata(n_slices, n_shards, rank, affinity_mode):
    if n_slices % n_shards != 0 and affinity_mode == "uniform":
        pytest.skip("n_slices must be divisible by n_shards for uniform affinity mode")
    if rank >= n_shards:
        pytest.skip("rank must be less than n_shards")
    device = "cuda"
    max_batch_size = 200
    max_n_blocks = (1 + (max_batch_size // 16)) * n_slices
    batch_sizes = torch.randint(n_shards, max_batch_size, (n_slices, ), dtype=torch.int32, device=device)
    batch_sizes[torch.randint(0, n_slices, (1, ))] = 0
    expt_dict = _make_expt_dict_for_mode(n_shards, n_slices, affinity_mode)
    expt_assignment = make_expt_assignment(n_shards, n_slices, expt_dict, device)
    expt_bitmask, expt_map = expt_assignment.expt_bitmask, expt_assignment.expt_map
    tri_metadata = make_ragged_tensor_metadata(batch_sizes, max_n_blocks)
    ref_metadata = make_ragged_tensor_metadata_torch(batch_sizes, max_n_blocks)
    tri_metadata = filter_ragged_tensor_metadata(tri_metadata, expt_bitmask[rank, :], expt_map[rank, :])
    ref_metadata = filter_ragged_tensor_metadata_torch(ref_metadata, expt_bitmask[rank, :], expt_map[rank, :])
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
    x = torch.randn((n_rows, n_cols), dtype=torch.float32, device=device)
    sparse_x = topk(x, k)
    metadata_tri = make_bitmatrix_metadata(sparse_x.indx, sparse_x.mask)
    metadata_ref = make_bitmatrix_metadata_torch(sparse_x.indx, sparse_x.mask)
    assert_equal(metadata_tri.col_sum, metadata_ref.col_sum)
    assert_equal(metadata_tri.row_sorted_indx, metadata_ref.row_sorted_indx)
    assert_equal(metadata_tri.col_sorted_indx, metadata_ref.col_sorted_indx)
