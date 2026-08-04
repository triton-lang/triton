import tempfile

import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


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
        yield fresh()
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
    yield fresh()
    reset()
