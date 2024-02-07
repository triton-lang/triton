# content of conftest.py

import os
import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default='cuda')


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture(scope="function", params=[False, True])
def add_interpreter_test(request):
    mode = request.param
    os.environ['TRITON_INTERPRET'] = '1' if mode else '0'
    yield
    os.environ['TRITON_INTERPRET'] = '0'
