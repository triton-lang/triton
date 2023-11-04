# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default="", help="Codegen backend")


@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--backend")
