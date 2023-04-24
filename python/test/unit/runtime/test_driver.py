import sys

import triton


def test_is_lazy():
    mod = sys.modules[triton.runtime.driver.__module__]
    assert isinstance(triton.runtime.driver, getattr(mod, "LazyProxy"))
