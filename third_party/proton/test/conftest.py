import os

import pytest

_RUNTIME_SELECTION_ENV = (
    "TRITON_LIBHIP_PATH",
    "TRITON_HSA_RUNTIME_PATH",
    "TRITON_HSA_RUNTIME_LIBRARY",
    "TRITON_ROCPROFILER_SDK_INCLUDE_PATH",
    "TRITON_ROCPROFILER_SDK_LIB_PATH",
    "TRITON_ROCPROFILER_SDK_LIBRARY",
    "TRITON_ROCTRACER_LIB_PATH",
    "TRITON_ROCTRACER_LIBRARY",
    "TRITON_ROCTX_LIB_PATH",
    "TRITON_ROCTX_LIBRARY",
)


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_knobs():
    from triton._internal_testing import _fresh_knobs_impl

    # TheRock installs ROCm libraries outside the system loader paths. Proton
    # selects their absolute paths at import time, so keep those selections
    # while resetting mutable test knobs such as TRITON_PROTON_DISABLE.
    runtime_selection_env = {key: os.environ[key] for key in _RUNTIME_SELECTION_ENV if key in os.environ}
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        fresh = fresh_function()
        os.environ.update(runtime_selection_env)
        yield fresh
    finally:
        reset_function()
