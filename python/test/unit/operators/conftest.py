# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default='cuda')


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
