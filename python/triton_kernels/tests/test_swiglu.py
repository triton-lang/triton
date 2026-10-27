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


@pytest.mark.parametrize(
    "layout",
    ["sliced_rows", "padded_rows", "broadcast_rows", "batched_sliced_rows", "transposed_batch", "vector", "batched"])
@pytest.mark.parametrize("N", [256, 258])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("limit", [None, 1e-2, 10])
def test_swiglu_input_layout(layout, N, dtype, limit, device):
    torch.manual_seed(0)
    M = 37
    if layout == "sliced_rows":
        x = alloc_rand((2 * M, N), device, dtype, requires_grad=False)[1::2]
    elif layout == "padded_rows":
        x = alloc_rand((M, N + 8), device, dtype, requires_grad=False)[:, 4:-4]
    elif layout == "broadcast_rows":
        x = alloc_rand((1, N), device, dtype, requires_grad=False).expand(M, N)
    elif layout == "batched_sliced_rows":
        x = alloc_rand((3, 2 * M, N), device, dtype, requires_grad=False)[:, 1::2]
    elif layout == "transposed_batch":
        x = alloc_rand((3, M, N), device, dtype, requires_grad=False).transpose(0, 1)
    elif layout == "vector":
        x = alloc_rand((N, ), device, dtype, requires_grad=False)
    else:
        x = alloc_rand((3, M, N), device, dtype, requires_grad=False)

    alpha = 0.5
    precision_config = PrecisionConfig(limit=limit)
    actual = swiglu(x, alpha, precision_config)
    expected = swiglu_torch(x, alpha, precision_config)

    assert actual.shape == expected.shape
    assert_close(expected, actual)
