import os
import pytest
import tempfile


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.environ["TRITON_CACHE_DIR"] = tmpdir
            yield tmpdir
        finally:
            os.environ.pop("TRITON_CACHE_DIR", None)
