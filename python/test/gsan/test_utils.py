import pytest
import torch

from triton._internal_testing import is_cuda
from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_uint8_cuda_tensor_from_ptr_delete_tensor():
    if torch.cuda.device_count() < 1:
        pytest.skip("requires at least 1 CUDA device")

    torch.cuda.set_device(0)
    view = uint8_cuda_tensor_from_ptr(12345, 10, 1)
    assert view.data_ptr() == 12345
    assert view.shape == (10, )
    assert view.dtype == torch.uint8
    del view
