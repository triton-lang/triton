import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gsan_fine_granularity(reason): requires sub-16-byte GSan tracking")
    # If pytest-sugar is not active, enable instafail
    if not config.pluginmanager.hasplugin("sugar"):
        config.option.instafail = True


# Exercise synchronization/ordering at the default granularity. Tests whose
# behavior depends on cell size explicitly parametrize this fixture indirectly.
@pytest.fixture(params=[4], ids=lambda granularity: f"granularity-{granularity}")
def shadow_granularity(request):
    marker = request.node.get_closest_marker("gsan_fine_granularity")
    if request.param == 16 and marker is not None:
        pytest.skip(marker.args[0])
    return request.param


@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())
