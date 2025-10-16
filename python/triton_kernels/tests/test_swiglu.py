from triton_kernels.swiglu import swiglu, swiglu_torch, standard_swiglu, standard_swiglu_torch, PrecisionConfig
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
@pytest.mark.parametrize("add_bias", [True, False])
def test_op(M, N, limit, add_bias, device, alpha=0.5):
    torch.manual_seed(2)
    # initialize data
    x = alloc_rand([M, N], device=device, dtype=torch.bfloat16)
    precision_config = PrecisionConfig(limit=limit)
    tri_y = swiglu(x, alpha, precision_config, add_bias=add_bias)
    ref_y = swiglu_torch(x, alpha, precision_config, add_bias=add_bias)
    assert_close(tri_y, ref_y)
    
    
@pytest.mark.parametrize("M, N", [(1311, 4352)])
def test_op_standard_swiglu(M, N, device):
    torch.manual_seed(2)
    # initialize data
    x = alloc_rand([M, N], device=device, dtype=torch.bfloat16)
    precision_config = PrecisionConfig(limit=None)
    tri_y = standard_swiglu(x, precision_config)
    ref_y = standard_swiglu_torch(x)
    assert_close(tri_y, ref_y)
