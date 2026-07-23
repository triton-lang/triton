import importlib.util
import sys
import types
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[3] / "third_party" / "proton" / "proton" / "_libpath.py"
)


def _load_libpath_module():
    spec = importlib.util.spec_from_file_location("proton_libpath_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_rocprofiler_sdk_library(monkeypatch, tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    library_names = {
        "amdhip64": "libamdhip64.so.7",
        "rocprofiler-sdk": "librocprofiler-sdk.so.1",
        "rocprofiler-sdk-roctx": "librocprofiler-sdk-roctx.so.1",
        "roctracer64": "libroctracer64.so.4",
        "roctx64": "libroctx64.so.4",
    }
    libraries = {}
    for shortname, filename in library_names.items():
        path = lib_dir / filename
        path.touch()
        libraries[shortname] = path
    (lib_dir / "libhsa-runtime64.so.1").touch()
    rocm_sdk = types.ModuleType("rocm_sdk")
    calls = []

    def find_libraries(*shortnames):
        calls.append(shortnames)
        return [libraries[shortnames[0]]]

    rocm_sdk.find_libraries = find_libraries
    monkeypatch.setitem(sys.modules, "rocm_sdk", rocm_sdk)

    module = _load_libpath_module()
    found = module.find_therock_rocm_libraries()
    assert found == {
        **{name: str(path) for name, path in libraries.items()},
        "hsa-runtime64": str(lib_dir / "libhsa-runtime64.so.1"),
    }
    assert calls == [(name,) for name in library_names]
    calls.clear()
    assert module.find_rocprofiler_sdk_library() == str(libraries["rocprofiler-sdk"])


def test_find_rocprofiler_sdk_library_legacy_fallback(monkeypatch, tmp_path):
    rocm_sdk = types.ModuleType("rocm_sdk")

    def find_libraries(*shortnames):
        raise ModuleNotFoundError

    rocm_sdk.find_libraries = find_libraries
    monkeypatch.setitem(sys.modules, "rocm_sdk", rocm_sdk)

    core_dir = tmp_path / "_rocm_sdk_core"
    lib_dir = core_dir / "lib"
    lib_dir.mkdir(parents=True)
    core = types.ModuleType("_rocm_sdk_core")
    core.__file__ = str(core_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "_rocm_sdk_core", core)

    assert _load_libpath_module().find_rocprofiler_sdk_library() == str(lib_dir)


def test_find_rocprofiler_sdk_library_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "rocm_sdk", None)
    monkeypatch.setitem(sys.modules, "_rocm_sdk_core", None)

    assert _load_libpath_module().find_rocprofiler_sdk_library() is None
