import os
import importlib

import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture(autouse=True)
def proton_test_backend(monkeypatch):
    backend = os.environ.get("PROTON_TEST_BACKEND")
    if backend:
        profile = importlib.import_module("triton.profiler.profile")

        monkeypatch.setattr(profile, "_select_backend", lambda: backend)


@pytest.fixture
def fresh_knobs():
    from triton._internal_testing import _fresh_knobs_impl

    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()
