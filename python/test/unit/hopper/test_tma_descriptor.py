import pytest
import torch
from triton.tools.experimental_descriptor import create_1d_tma_descriptor, create_2d_tma_descriptor


@pytest.mark.parametrize("M, BLOCK_M, expect_error", [(128, 32, False), (127, 32, False), (128, 31, True)])
def test_1d_tma_descriptor_exception(M, BLOCK_M, expect_error):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Test requires Hopper target.")
        return

    device = "cuda"
    x = torch.randn(M, dtype=torch.float32, device=device)
    is_error = False

    try:
        create_1d_tma_descriptor(x.data_ptr(), M, BLOCK_M, x.element_size())
    except SystemError as e:
        is_error = True
        assert e.args[0] == "<built-in function fill_1d_tma_descriptor> returned a result with an exception set"

    assert is_error == expect_error


@pytest.mark.parametrize("M, BLOCK_M", [(128, 32), (125, 33)])
@pytest.mark.parametrize("N, BLOCK_N, expect_error", [(128, 32, False), (128, 30, True), (127, 32, True)])
def test_2d_tma_descriptor_exception(M, N, BLOCK_M, BLOCK_N, expect_error):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Test requires Hopper target.")
        return

    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, N), dtype=torch.float16, device=device)
    is_error = False

    try:
        create_2d_tma_descriptor(A.data_ptr(), M, N, BLOCK_M, BLOCK_N, A.element_size())
    except SystemError as e:
        is_error = True
        assert e.args[0] == "<built-in function fill_2d_tma_descriptor> returned a result with an exception set"

    assert is_error == expect_error
