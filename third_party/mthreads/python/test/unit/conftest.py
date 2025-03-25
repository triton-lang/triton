# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default='musa')


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
