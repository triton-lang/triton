import os
import tempfile
from contextlib import contextmanager

import pytest


@contextmanager
def enable_remark_context():
    try:
        os.environ["MLIR_ENABLE_REMARK"] = "1"
        yield
    finally:
        os.environ["MLIR_ENABLE_REMARK"] = "0"


@contextmanager
def enable_dump_context(pass_name="1"):
    try:
        os.environ["MLIR_ENABLE_REMARK"] = pass_name
        yield
    finally:
        os.environ["MLIR_ENABLE_REMARK"] = "0"


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
