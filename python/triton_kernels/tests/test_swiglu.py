from triton_kernels.swiglu import swiglu, swiglu_torch, PrecisionConfig
from triton_kernels.testing import assert_close
import torch
import pytest

# ---------------
# initialize data
# ---------------


def alloc_rand(shape, device, dtype, requires_grad=True):
    if dtype.itemsize == 1:
        tmp = 2**-(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype).requires_grad_(requires_grad)
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


# ---------------
# unit tests
# ---------------


@pytest.mark.parametrize("M, N", [(1311, 4352)])
@pytest.mark.parametrize("limit", [1e-2, 10])
def test_op(M, N, limit, device, alpha=0.5):
    torch.manual_seed(2)
    # initialize data
    x = alloc_rand([M, N], device=device, dtype=torch.bfloat16)
    precision_config = PrecisionConfig(limit=limit)
    tri_y = swiglu(x, alpha, precision_config)
    ref_y = swiglu_torch(x, alpha, precision_config)
    assert_close(tri_y, ref_y)
