from __future__ import annotations

import pytest
import tempfile

from pathlib import Path

import triton

from triton.runtime.build import compile_module_from_src


@pytest.fixture
def amd_codegen_build_setup(tmp_path, monkeypatch):
    import argparse
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    monkeypatch.setattr(build_helpers, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(build_helpers, "get_llvm_package_info",
                        lambda *args: pytest.fail("backend consulted core LLVM"))
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake/llvm-info.json").write_text("core LLVM metadata must not be read")
    info = {
        "llvm_hash": "a" * 40,
        "build_number": 2,
        "sha256sum": {},
        "bootstrap_llvm": {
            "build_number": 1,
            "sha256sum": {"macos-arm64": "backend-sdk-checksum"},
        },
    }
    manifest = tmp_path / "cmake/amd-llvm-info.json"
    manifest.write_text(json.dumps(info))
    parser = argparse.ArgumentParser()
    build_helpers.add_common_args(parser)
    args = build_helpers.normalize_parsed_args(
        parser.parse_args([
            "--triton-cache-path",
            str(tmp_path / "cache"),
            "--triton-llvm-system-suffix",
            "macos-arm64",
            "--llvm-syspath",
            str(tmp_path / "unusable-core-llvm"),
        ]))
    return build_helpers, args, info, manifest


@pytest.mark.parametrize("offline", [False, True])
def test_amd_codegen_bootstrap_ignores_core_llvm(amd_codegen_build_setup, monkeypatch, offline):
    from dataclasses import replace

    helpers, args, info, _ = amd_codegen_build_setup
    args = replace(args, offline_build=offline)
    library = helpers.amd_codegen_library_name()
    sdk = Path(args.cache_path) / "amd/llvm/llvm-aaaaaaaa-macos-arm64-1"
    config = sdk / "lib/cmake/llvm/LLVMConfig.cmake"
    downloads = []

    def download(url, package_root, name, archives_path, checksum):
        assert not offline
        assert name == "llvm-aaaaaaaa-macos-arm64-1"
        assert checksum == "backend-sdk-checksum"
        assert Path(package_root) == sdk.parent
        downloads.append(url)
        config.parent.mkdir(parents=True)
        config.touch()

    if offline:
        config.parent.mkdir(parents=True)
        config.touch()
    monkeypatch.setattr(helpers, "_download_and_extract", download)
    commands = []

    def cmake(command):
        commands.append(command)
        if "--install" in command:
            prefix = next(flag.split("=", 1)[1] for flag in commands[0] if flag.startswith("-DCMAKE_INSTALL_PREFIX="))
            output = Path(prefix) / "lib" / library
            output.parent.mkdir(parents=True)
            output.write_bytes(b"packaging fixture")

    monkeypatch.setattr(helpers.subprocess, "check_call", cmake)
    helpers.download_and_copy_amd_codegen(args)
    assert len(downloads) == (0 if offline else 1)
    assert f"-DLLVM_DIR={sdk / 'lib/cmake/llvm'}" in commands[0]
    assert f"-DTRITON_AMD_LLVM_REVISION={info['llvm_hash']}-2" in commands[0]
    assert all(args.llvm_syspath not in flag for command in commands for flag in command)


def test_amd_codegen_download_ignores_core_llvm(amd_codegen_build_setup, monkeypatch):
    import json

    helpers, args, info, manifest = amd_codegen_build_setup
    info["sha256sum"]["macos-arm64"] = "backend-library-checksum"
    manifest.write_text(json.dumps(info))
    monkeypatch.setattr(helpers, "build_amd_codegen", lambda *args: pytest.fail("published backend was rebuilt"))
    downloads = []

    def download(url, package_root, name, archives_path, checksum):
        assert name == "amd-codegen-aaaaaaaa-macos-arm64-2"
        assert checksum == "backend-library-checksum"
        downloads.append(url)
        library = Path(package_root) / name / "lib" / helpers.amd_codegen_library_name()
        library.parent.mkdir(parents=True)
        library.write_bytes(b"published packaging fixture")

    monkeypatch.setattr(helpers, "_download_and_extract", download)
    helpers.download_and_copy_amd_codegen(args)
    assert len(downloads) == 1


def test_amd_codegen_bootstrap_requires_cached_llvm_offline(amd_codegen_build_setup, monkeypatch):
    from dataclasses import replace

    helpers, args, info, _ = amd_codegen_build_setup
    monkeypatch.setattr(helpers, "_download_and_extract", lambda *args: pytest.fail("offline download attempted"))
    with pytest.raises(RuntimeError, match="amd bootstrap LLVM is missing"):
        helpers.download_codegen_llvm("amd", info, replace(args, offline_build=True))


TEST_MODULE_C = """
#include <Python.h>
#include <string.h>

static PyObject* go(PyObject* self, PyObject* args) {
    const char *command;
    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    const char* res;
    if (strcmp(command, "hello") == 0) {
        res = "hiya";
    } else {
        res = "huh";
    }
    return PyUnicode_FromString(res);
}

static PyMethodDef ModuleMethods[] = {
  {"go", go, METH_VARARGS, "test_module.go for testing"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "test_module",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit_test_module(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""


def test_compile_module(fresh_triton_cache):
    mod = compile_module_from_src(TEST_MODULE_C, "test_module")

    with pytest.raises(Exception):
        mod.go()

    assert mod.go("huh") == "huh"
    assert mod.go("hello") == "hiya"

    # Make sure the module is cached
    mod2 = compile_module_from_src(TEST_MODULE_C, "test_module")
    assert mod2.__file__ == mod.__file__


def test_compile_module_bad_cache(fresh_knobs):
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        called_get_file = False

        class InvalidFileCacheManager(triton.runtime.cache.FileCacheManager):

            def get_file(self, filename: str) -> str | None:
                nonlocal called_get_file
                called_get_file = True
                (tmp / filename).write_text("not an so")
                return str(tmp / filename)

        # First corrupt the cache
        fresh_knobs.cache.manager_class = InvalidFileCacheManager

        mod = compile_module_from_src(TEST_MODULE_C, "test_module")
        assert called_get_file

        with pytest.raises(Exception):
            mod.go()

        assert mod.go("huh") == "huh"
        assert mod.go("hello") == "hiya"
