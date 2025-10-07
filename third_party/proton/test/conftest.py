import pytest


@pytest.fixture
def fresh_knobs():
    from triton._internal_testing import _fresh_knobs_impl

    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()
