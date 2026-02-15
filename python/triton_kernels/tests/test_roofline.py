import pytest
from triton_kernels.roofline import get_memset_tbps, get_blas_tflops
from triton_kernels.target_info import cuda_capability_geq, is_cuda


def test_get_memset_tbps():
    tbps = get_memset_tbps()
    assert tbps > 0


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp8"])
def test_get_blas_tflops(dtype):
    if dtype in ["fp8"] and is_cuda() and not cuda_capability_geq(9, 0):
        pytest.skip("FP8 not supported on this GPU")
    tflops = get_blas_tflops(dtype)
    assert tflops > 0
