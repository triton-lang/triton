import pytest


@pytest.fixture
def fresh_knobs():
    """
    Resets all knobs except ``build``, ``nvidia``, and ``amd`` (preserves
    library paths needed to compile kernels).
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_including_libraries():
    """
    Resets ALL knobs including ``build``, ``nvidia``, and ``amd``.
    Use for tests that verify initial values of these knobs.
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()
