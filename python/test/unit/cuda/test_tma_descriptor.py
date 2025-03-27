import pytest
import torch
from triton.tools.experimental_descriptor import create_1d_tma_descriptor, create_2d_tma_descriptor


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
    is_error = False

    try:
        create_1d_tma_descriptor(x.data_ptr(), M, BLOCK_M, x.element_size())
    except RuntimeError as e:
        is_error = True
        assert e.args[0] == "Triton Error [CUDA]: invalid argument"

    assert is_error == expect_error


@pytest.mark.parametrize("M, BLOCK_M", [(128, 32), (125, 33)])
@pytest.mark.parametrize("N, BLOCK_N, expect_error", [(128, 32, False), (128, 30, True), (127, 32, True)])
def test_2d_tma_descriptor_exception(M, N, BLOCK_M, BLOCK_N, expect_error):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("Test requires Hopper or Blackwell target.")
        return

    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, N), dtype=torch.float16, device=device)
    # globalAddress in the tma descriptor must be aligned to 16 bytes for CU_TENSOR_MAP_INTERLEAVE_NONE.
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY
    assert A.data_ptr() % 16 == 0
    is_error = False

    try:
        create_2d_tma_descriptor(A.data_ptr(), M, N, BLOCK_M, BLOCK_N, A.element_size())
    except RuntimeError as e:
        is_error = True
        assert e.args[0] == "Triton Error [CUDA]: invalid argument"

    assert is_error == expect_error
