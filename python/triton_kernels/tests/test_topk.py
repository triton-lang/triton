import pytest
import torch
from triton_kernels.topk import topk, topk_torch
from triton_kernels.testing import assert_equal, assert_close


def canonicalize(vals, idx):
    order = torch.argsort(vals, dim=1, descending=True, stable=True)
    vals_sorted = torch.take_along_dim(vals, order, dim=1)
    idx_sorted = torch.take_along_dim(idx, order, dim=1)
    return vals_sorted, idx_sorted.int()


@pytest.mark.parametrize("n_rows", [1, 7, 256, 300])
@pytest.mark.parametrize("n_cols", [13, 32, 128, 200])
@pytest.mark.parametrize("k", [8])
@pytest.mark.parametrize("apply_softmax", [True, False])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_topk(n_rows, n_cols, k, apply_softmax, dtype):
    device = "cuda"
    torch.manual_seed(0)
    dtype = getattr(torch, dtype)
    x = torch.randn((n_rows, n_cols), dtype=torch.float32, device=device)
    y_vals_tri, y_indx_tri, bitmatrix_tri = topk(x, k, apply_softmax=apply_softmax)
    y_vals_ref, y_indx_ref, bitmatrix_ref = topk_torch(x, k, apply_softmax=apply_softmax)
    y_vals_tri_sorted, y_indx_tri_sorted = canonicalize(y_vals_tri, y_indx_tri)
    y_vals_ref_sorted, y_indx_ref_sorted = canonicalize(y_vals_ref, y_indx_ref)
    assert_close(y_vals_tri_sorted, y_vals_ref_sorted)
    assert_equal(y_indx_tri_sorted, y_indx_ref_sorted)
    assert_equal(bitmatrix_tri.storage.data, bitmatrix_ref.storage.data)
    assert bitmatrix_tri.storage.data.stride() == bitmatrix_ref.storage.data.stride()
    assert bitmatrix_tri.storage.data.shape == bitmatrix_ref.storage.data.shape
