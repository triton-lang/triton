import pytest
import torch

devices = ['cpu']

if torch.cuda.is_available():
    devices += ['cuda']


@pytest.fixture(params=devices, scope="session")
def device(request):
    """
    Fixture for CPU/GPU device in pytorch
    """
    return request.param
