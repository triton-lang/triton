from contextlib import nullcontext
import pytest
import torch
from triton.tools.tensor_descriptor import TensorDescriptor


@pytest.mark.parametrize("M, BLOCK_M, expect_error", [(128, 32, False), (127, 32, False), (128, 31, True)])
def test_1d_tma_descriptor_exception(M, BLOCK_M, expect_error):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("Test requires Hopper or Blackwell target.")
        return

    device = "cuda"
    x = torch.randn(M, dtype=torch.float32, device=device)
    # globalAddress in the tma descriptor must be aligned to 16 bytes for CU_TENSOR_MAP_INTERLEAVE_NONE.
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY
    assert x.data_ptr() % 16 == 0

    ctx = pytest.raises(ValueError, match="Shape element 0 must be a power of 2") if expect_error else nullcontext()
    with ctx:
        _ = TensorDescriptor.from_tensor(x, [BLOCK_M])


@pytest.mark.parametrize("M, BLOCK_M, expect_error_m", [(128, 32, False), (125, 33, True)])
@pytest.mark.parametrize("N, BLOCK_N, expect_error_n", [(128, 32, False), (128, 30, True), (127, 32, False)])
def test_2d_tma_descriptor_exception(M, N, BLOCK_M, BLOCK_N, expect_error_n, expect_error_m):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("Test requires Hopper or Blackwell target.")
        return

    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, N), dtype=torch.float16, device=device)
    # globalAddress in the tma descriptor must be aligned to 16 bytes for CU_TENSOR_MAP_INTERLEAVE_NONE.
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY
    assert A.data_ptr() % 16 == 0

    shape_error = expect_error_n or expect_error_m
    error_alignment = (N % 16) != 0
    expect_error = shape_error or error_alignment

    exc_type = ValueError if shape_error else AssertionError
    match = "Shape element . must be a power of 2" if shape_error else "strides must be 16-byte aligned"
    ctx = pytest.raises(exc_type, match=match) if expect_error else nullcontext()
    with ctx:
        _ = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_N])
