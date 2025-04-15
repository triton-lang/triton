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
    config_map = {
        name: conf
        for name, conf in config.__dict__.items()
        if isinstance(conf, config.base_config) and conf != config.base_config
    }
    try:
        for name, conf in config_map.items():
            setattr(config, name, conf.copy().reset())
        yield config
    finally:
        for name, conf in config_map.items():
            setattr(config, name, conf)
