from triton_bench.routing import routing_torch
from triton_bench.swiglu import swiglu, swiglu_torch, PrecisionConfig
from triton_bench.testing import assert_close
import torch
import pytest

from .test_routing import init_data as init_routing_data
from .test_routing import ref_expt_data

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
def test_op(M, N, limit, alpha=0.5):
    torch.manual_seed(2)
    dev = "cuda"
    dtype = torch.bfloat16
    # initialize expert data
    n_expts_tot = 6
    n_expts_act = 2
    logits = init_routing_data(M, n_expts_tot).detach()
    routing_data, _, _ = routing_torch(logits, n_expts_act)
    expt_data = ref_expt_data(routing_data, M * n_expts_act, block_m=128)
    n_tokens = expt_data[2 * n_expts_tot].sum()

    # initialize data
    x = alloc_rand([n_tokens, N], device=dev, dtype=dtype)
    precision_config = PrecisionConfig(limit=limit)
    tri_y = swiglu(x, alpha, precision_config, expt_data, n_expts_tot)
    ref_y = swiglu_torch(x, alpha, precision_config)
    assert_close(tri_y, ref_y)
