import pytest
import torch

from triton._internal_testing import is_cuda
from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_uint8_cuda_tensor_from_ptr_delete_tensor():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    backing = torch.arange(10, dtype=torch.uint8, device=device)
    view = uint8_cuda_tensor_from_ptr(backing.data_ptr(), backing.numel(), device.index)
    assert view.data_ptr() == backing.data_ptr()
    assert view.shape == (10, )
    assert view.dtype == torch.uint8
    assert view.device == device
    assert torch.equal(view, backing)
