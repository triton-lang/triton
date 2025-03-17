import pytest
import torch
from triton._internal_testing import is_cuda


@pytest.mark.parametrize("m, n, k", [(16, 16, 16), (32, 16, 16), (16, 32, 16), (16, 16, 32)])
@pytest.mark.parametrize("dtype_str", ["float8_e4m3fn", "float16"])
def test_cublas(m, n, k, dtype_str, device):
    dtype = getattr(torch, dtype_str)
    if not is_cuda():
        pytest.skip("test_cublas is only supported on CUDA")
    if dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fp8 is only supported on CUDA with cc >= 90")

    from triton._C.libtriton import nvidia

    torch.manual_seed(123)
    workspace_size = 32 * 1024 * 1024

    def limited_rand(elements, shape):
        total_elems = torch.prod(torch.tensor(shape)).item()
        indices = torch.randint(0, len(elements), (total_elems, ), device=device)
        return elements[indices].view(shape)

    elements = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device=device)
    a = limited_rand(elements, (m, k)).to(dtype)
    b = limited_rand(elements, (k, n)).to(dtype)
    c = torch.zeros((m, n), dtype=dtype, device=device)

    b = b.T.contiguous()

    workspace = torch.empty(workspace_size, dtype=torch.int8, device=device)

    cublas = nvidia.cublas.CublasLt(workspace)
    cublas.matmul(a, b, c)

    ref = torch.matmul(a.to(torch.float16), b.to(torch.float16).T)

    assert torch.allclose(c.to(torch.float16), ref, atol=2.0)
