import os
import tempfile

import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


def pytest_configure(config):
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    requested = os.environ.get("TRITON_TEST_NUM_GPUS")
    if not requested or not worker.startswith("gw"):
        return

    visible = os.environ.get("TRITON_TEST_VISIBLE_GPUS")
    if visible is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
    else:
        devices = [str(index) for index in range(int(requested))]
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[int(worker[2:]) % int(requested)]


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as directory:
        from triton import knobs

        with knobs.cache.scope(), knobs.runtime.scope():
            knobs.cache.dir = directory
            yield directory


@pytest.fixture
def fresh_knobs():
    from triton import knobs
    from triton._internal_testing import _fresh_knobs_impl

    fresh, reset = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
    with knobs.amd.scope():
        try:
            yield fresh()
        finally:
            reset()


@pytest.fixture
def fresh_compilation_knobs():
    from triton import knobs

    with knobs.compilation.scope():
        yield knobs


@pytest.fixture
def fresh_knobs_including_libraries():
    from triton._internal_testing import _fresh_knobs_impl

    fresh, reset = _fresh_knobs_impl()
    try:
        yield fresh()
    finally:
        reset()
