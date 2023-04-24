import sys

import triton


def test_is_lazy():
    mod = sys.modules[triton.runtime.driver.__module__]
    assert isinstance(triton.runtime.driver, getattr(mod, "LazyProxy"))
    assert triton.runtime.driver._obj is None
    utils = triton.runtime.driver.utils  # noqa: F841
    assert issubclass(triton.runtime.driver._obj.__class__, getattr(mod, "DriverBase"))
