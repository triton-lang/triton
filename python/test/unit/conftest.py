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
def fresh_config(request):
    from triton import config
    test_name = request.node.name
    config_map = {
        name: cls
        for name, cls in config.__dict__.items()
        if isinstance(cls, config._base) and cls != config._base
    }
    try:
        for name, cls in config_map.items():
            new_cls = type(f"{name}_{test_name}", cls.__bases__, dict(cls.__dict__))
            new_cls.reset()
            setattr(config, name, new_cls)
        yield
    finally:
        for name, cls in config_map.items():
            setattr(config, name, cls)
