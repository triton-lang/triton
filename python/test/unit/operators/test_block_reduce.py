import torch
import triton
import pytest

@pytest.mark.parametrize("SPLIT_K, M, N, dtype",
  [
    (SPLIT_K, M, N, dtype) for SPLIT_K in [1, 2, 4, 8, 16]
                           for M in [512, 1871, 8192]
                           for N in [512, 1871, 8192]
                           for dtype in ['float16', 'float32']
  ]
)
def test_op(SPLIT_K, M, N, dtype):
  dtype = {'float16': torch.float16, 'float32': torch.float32}[dtype]
  # create inputs
  x = torch.randn((SPLIT_K, M, N), device='cuda', dtype=dtype)
  th_y = torch.einsum('kmn->mn', x)
  tt_y = triton.ops.block_reduce(x)
  triton.testing.assert_almost_equal(th_y, tt_y)