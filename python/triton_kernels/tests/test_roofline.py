import pytest
from triton_kernels.roofline import get_memset_tbps, get_blas_tflops


def test_get_memset_tbps():
    tbps = get_memset_tbps()
    assert tbps > 0


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp8"])
def test_get_blas_tflops(dtype):
    tflops = get_blas_tflops(dtype)
    assert tflops > 0
