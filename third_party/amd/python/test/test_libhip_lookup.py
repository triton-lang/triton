import os
import platform

import pytest

import triton.backends.amd.driver as amd_driver

pytestmark = pytest.mark.skipif(platform.system() != "Linux", reason="dl_iterate_phdr is Linux-only")


def test_find_loaded_library():
    path = amd_driver._find_already_mmapped_dylib_on_linux("libc.so")
    assert path is not None
    assert "libc.so" in os.path.basename(path)
    assert os.path.exists(path)


def test_find_loaded_library_returns_none_for_no_match():
    assert amd_driver._find_already_mmapped_dylib_on_linux("lib-not-loaded-by-triton-test.so") is None
