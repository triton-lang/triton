import os
import pytest
import tempfile
from typing import Optional, Set


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs
        with knobs.cache.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir


def _fresh_knobs_impl(monkeypatch, skipped_attr: Optional[Set[str]] = None):
    from triton import knobs

    if skipped_attr is None:
        skipped_attr = set()

    knobs_map = {
        name: knobset
        for name, knobset in knobs.__dict__.items()
        if isinstance(knobset, knobs.base_knobs) and knobset != knobs.base_knobs and name not in skipped_attr
    }

    # We store which variables we need to unset below in finally because
    # monkeypatch doesn't appear to reset variables that were never set
    # before the monkeypatch.delenv call below.
    env_to_unset = []
    prev_propagate_env = knobs.propagate_env

    def fresh_function():
        nonlocal env_to_unset
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset.copy().reset())
            for knob in knobset.knob_descriptors.values():
                if knob.key in os.environ:
                    monkeypatch.delenv(knob.key, raising=False)
                else:
                    env_to_unset.append(knob.key)
        knobs.propagate_env = True
        return knobs

    def reset_function():
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset)
        for k in env_to_unset:
            if k in os.environ:
                del os.environ[k]
        knobs.propagate_env = prev_propagate_env

    return fresh_function, reset_function


@pytest.fixture
def fresh_knobs(monkeypatch):
    fresh_function, reset_function = _fresh_knobs_impl(monkeypatch)
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_except_libraries(monkeypatch):
    """
    A variant of `fresh_knobs` that keeps library path
    information from the environment as these may be
    needed to successfully compile kernels.
    """
    fresh_function, reset_function = _fresh_knobs_impl(monkeypatch, skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()
