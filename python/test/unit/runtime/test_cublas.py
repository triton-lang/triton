import pytest
import torch
import triton
import os


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cuda():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cuda"


@pytest.mark.parametrize("m, n, k", [(16, 16, 16), (32, 16, 16), (16, 32, 16), (16, 16, 32)])
def test_cublas_fp8(m, n, k, device):
    if is_cuda():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9 and capability[1] < 9:
            pytest.skip("test_cublas_fp8 is only supported on CUDA with cc >= 89")

    from triton._C.libtriton import nvidia

    torch.manual_seed(123)
    workspace_size = 32 * 1024 * 1024

    def limited_rand(elements, shape):
        total_elems = torch.prod(torch.tensor(shape)).item()
        indices = torch.randint(0, len(elements), (total_elems, ), device=device)
        return elements[indices].view(shape)

    elements = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device=device)
    a = limited_rand(elements, (m, k)).to(torch.float8_e4m3fn)
    b = limited_rand(elements, (k, n)).to(torch.float8_e4m3fn)
    c = torch.zeros((m, n), dtype=torch.float8_e4m3fn, device=device)

    b = b.T.contiguous()

    workspace = torch.empty(workspace_size, dtype=torch.int8, device=device)

    cublas = nvidia.cublas.CublasLt(workspace)
    cublas.fp8_matmul(a, b, c)

    ref = torch.matmul(a.to(torch.float16), b.to(torch.float16).T)

    assert torch.allclose(c.to(torch.float16), ref, atol=2.0)
