import pytest
import torch
from triton_kernels.tensor import (
    make_ragged_tensor_metadata,
    make_ragged_tensor_metadata_torch,
    make_bitmatrix_metadata,
    make_bitmatrix_metadata_torch,
)
from triton_kernels.topk import topk
from triton_kernels.testing import assert_equal


@pytest.mark.parametrize("n_gates", [4, 16, 128, 1990])
@pytest.mark.parametrize("n_batches", [1, 7, 33])
def test_make_ragged_tensor_metadata(n_gates, n_batches):
    torch.manual_seed(0)
    device = "cuda"
    batch_sizes = torch.randint(0, 200, (n_batches, ), dtype=torch.int32, device=device)
    batch_sizes[torch.randint(0, n_batches, (1, ))] = 0
    meta = make_ragged_tensor_metadata(batch_sizes, n_gates)
    ref = make_ragged_tensor_metadata_torch(batch_sizes, n_gates)
    assert_equal(meta.batch_offs, ref.batch_offs)
    assert_equal(meta.block_offs_data, ref.block_offs_data)
    assert_equal(meta.block_schedule_data, ref.block_schedule_data)


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
