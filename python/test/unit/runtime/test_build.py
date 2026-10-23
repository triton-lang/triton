from __future__ import annotations

import argparse
import hashlib
import io
import json
import pytest
import tarfile
import tempfile

from pathlib import Path

import triton

from triton.runtime.build import compile_module_from_src

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


@pytest.mark.parametrize("artifact", ["bootstrap", "codegen"])
@pytest.mark.parametrize(
    "mirror", ["", "https://mirror.example/generic/oaiartifacts", "https://mirror.example/generic/oaiartifacts/"])
def test_amd_codegen_download_mirror(artifact, mirror, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    monkeypatch.setenv("OAIARTIFACTS_BASE_URL", mirror)
    monkeypatch.delenv("TRITON_CACHE_DEPENDENCY_DOWNLOADS", raising=False)
    monkeypatch.setattr(build_helpers, "get_base_dir", lambda: str(tmp_path))
    parser = argparse.ArgumentParser()
    build_helpers.add_common_args(parser)
    args = build_helpers.normalize_parsed_args(
        parser.parse_args([
            "--triton-cache-path",
            str(tmp_path / "cache"),
            "--triton-llvm-system-suffix",
            "ubuntu-arm64",
            "--llvm-syspath",
            str(tmp_path / "unrelated-core-llvm"),
        ]))

    revision = "123456789abcdef"
    name = f"{'llvm' if artifact == 'bootstrap' else 'amd-codegen'}-{revision[:8]}-ubuntu-arm64-1"
    relative_path = "lib/cmake/llvm/LLVMConfig.cmake" if artifact == "bootstrap" else (
        f"lib/{build_helpers.amd_codegen_library_name()}")
    contents = b"test artifact"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(f"{name}/{relative_path}")
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))
    archive_bytes = buffer.getvalue()
    checksum = hashlib.sha256(archive_bytes).hexdigest()
    llvm_info = {"llvm_hash": revision, "build_number": 1, "sha256sum": {"ubuntu-arm64": checksum}}
    if artifact == "bootstrap":
        llvm_info["bootstrap_llvm"] = {"build_number": 1, "sha256sum": llvm_info.pop("sha256sum")}

    downloads = []

    def download(url, path, label):
        downloads.append(url)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(archive_bytes)

    monkeypatch.setattr(build_helpers, "_download_file", download)
    if artifact == "bootstrap":
        result = Path(build_helpers.download_codegen_llvm("amd", llvm_info, args)) / relative_path
    else:
        (tmp_path / "cmake").mkdir()
        (tmp_path / "cmake" / "amd-llvm-info.json").write_text(json.dumps(llvm_info))
        build_helpers.download_and_copy_amd_codegen(args)
        result = tmp_path / "third_party" / "amd" / "backend" / relative_path

    base = "https://mirror.example/generic/triton-llvm" if mirror else (
        "https://oaitriton.blob.core.windows.net/public/llvm-builds")
    assert downloads == [f"{base}/{name}.tar.gz"]
    assert result.read_bytes() == contents


@pytest.mark.parametrize("alias_type", ["symlink", "directory", "indirect_symlink"])
def test_llvm_syspath_preserves_cached_alias(alias_type, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    cache_path = tmp_path / "cache"
    alias = cache_path / "llvm" / "llvm-ubuntu-x64"
    configured_path = tmp_path / "custom-llvm" if alias_type == "indirect_symlink" else alias
    parser = argparse.ArgumentParser()
    build_helpers.add_common_args(parser)
    args = build_helpers.normalize_parsed_args(
        parser.parse_args([
            "--triton-cache-path",
            str(cache_path), "--triton-offline-build", "--llvm-syspath",
            str(configured_path), "--triton-llvm-system-suffix", "ubuntu-x64"
        ]))
    package = build_helpers.get_llvm_package_info(args)
    source = alias if alias_type == "directory" else cache_path / "llvm" / package.name
    marker = Path("include/llvm/marker.h")
    (source / marker).parent.mkdir(parents=True)
    (source / marker).write_text("LLVM headers")
    if alias_type != "directory":
        alias.symlink_to(source, target_is_directory=True)
    if alias_type == "indirect_symlink":
        configured_path.symlink_to(alias, target_is_directory=True)

    variables = build_helpers._get_thirdparty_package_cmake_vars(package, args)

    assert (alias / marker).read_text() == "LLVM headers"
    assert (source / marker).read_text() == "LLVM headers"
    assert Path(variables["LLVM_INCLUDE_DIRS"]) / "llvm/marker.h" == configured_path / marker
