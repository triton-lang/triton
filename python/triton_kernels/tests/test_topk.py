import pytest
import torch
from triton_kernels.topk import topk, topk_torch
from triton_kernels.testing import assert_equal, assert_close


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
    sparse_x_tri = topk(x, k, apply_softmax=apply_softmax)
    sparse_x_ref = topk_torch(x, k, apply_softmax=apply_softmax)
    assert_close(sparse_x_tri.vals, sparse_x_ref.vals)
    assert_equal(sparse_x_tri.indx, sparse_x_ref.indx)
    assert_equal(sparse_x_tri.mask.storage.data, sparse_x_ref.mask.storage.data)
    assert sparse_x_tri.mask.storage.data.stride() == sparse_x_ref.mask.storage.data.stride()
    assert sparse_x_tri.mask.storage.data.shape == sparse_x_ref.mask.storage.data.shape
