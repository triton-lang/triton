import concurrent.futures
import os
import platform

import pytest

import triton.backends.amd.driver as amd_driver


pytestmark = pytest.mark.skipif(platform.system() != "Linux", reason="dl_iterate_phdr is Linux-only")


@pytest.fixture(autouse=True)
def clear_lookup_caches():
    amd_driver._load_dl_helper_module.cache_clear()
    amd_driver._get_path_to_hip_runtime_dylib.cache_clear()
    yield
    amd_driver._load_dl_helper_module.cache_clear()
    amd_driver._get_path_to_hip_runtime_dylib.cache_clear()


def test_find_loaded_library():
    path = amd_driver._find_already_mmapped_dylib_on_linux("libc.so")
    assert path is not None
    assert "libc.so" in os.path.basename(path)
    assert os.path.exists(path)


def test_find_loaded_library_returns_none_for_no_match():
    assert amd_driver._find_already_mmapped_dylib_on_linux("lib-not-loaded-by-triton-test.so") is None


def test_concurrent_find_loaded_library():
    module = amd_driver._load_dl_helper_module()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(module.find_loaded_library, ["libc.so"] * 64))

    assert all(path is not None for path in paths)
    assert all("libc.so" in os.path.basename(path) for path in paths)


def test_dl_helper_failure_returns_none(monkeypatch):
    def fail_to_load():
        raise RuntimeError("failed to build native helper")

    monkeypatch.setattr(amd_driver, "_load_dl_helper_module", fail_to_load)

    assert amd_driver._find_already_mmapped_dylib_on_linux("libamdhip64.so") is None
