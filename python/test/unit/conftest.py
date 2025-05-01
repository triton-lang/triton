import os
import pytest
import tempfile


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
        try:
            os.environ["TRITON_CACHE_DIR"] = tmpdir
            yield tmpdir
        finally:
            os.environ.pop("TRITON_CACHE_DIR", None)


@pytest.fixture
def fresh_knobs(request, monkeypatch):
    from triton import knobs
    knobs_map = {
        name: knobset
        for name, knobset in knobs.__dict__.items()
        if isinstance(knobset, knobs.base_knobs) and knobset != knobs.base_knobs
    }
    try:
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset.copy().reset())
            for knob in knobset.knob_descriptors.values():
                monkeypatch.delenv(knob.key, raising=False)
        prev_propagate_env = knobs.propagate_env
        knobs.propagate_env = True
        yield knobs
        knobs.propagate_env = prev_propagate_env
    finally:
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset)
