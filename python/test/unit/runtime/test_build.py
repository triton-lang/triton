from __future__ import annotations

import argparse
import hashlib
import io
import json
import pytest
import runpy
import sys
import tarfile
import tempfile

from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("backends", [None, (), ("nvidia", ), ("amd", ), ("nvidia", "amd")])
@pytest.mark.parametrize("build_proton", [False, True])
def test_backend_dependency_downloads(backends, build_proton, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    downloads = []
    monkeypatch.setattr(build_helpers, "download_and_copy_amd_codegen", lambda args: downloads.append("amd"))
    monkeypatch.setattr(build_helpers, "download_and_copy", lambda **kwargs: downloads.append(kwargs["name"]))
    args = ["download_and_copy_dependencies", "--triton-cache-path", str(tmp_path / "cache")]
    if backends is not None:
        args.extend(["--backends", *backends])
    if build_proton:
        args.append("--build-proton")

    build_helpers.main(args)

    selected = ("nvidia", "amd") if backends is None else backends
    expected = ["amd"] if "amd" in selected else []
    if "nvidia" in selected:
        expected.extend(["nvcc", "cuobjdump", "nvdisasm", "nvcc-crt", "cudart", "cupti", "cupti", "cupti-blackwell"])
    elif build_proton:
        expected.extend(["nvcc-crt", "cudart", "cupti", "cupti", "cupti-blackwell"])
    assert downloads == expected


@pytest.fixture
def build_setup(monkeypatch):
    import setuptools

    # Load the packaging functions without installing packages or preparing backends.
    monkeypatch.setenv("TRITON_CODEGEN_BACKENDS", "")
    monkeypatch.delenv("TRITON_PLUGIN_DIRS", raising=False)
    monkeypatch.setenv("TRITON_BUILD_PROTON", "OFF")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    root = Path(__file__).resolve().parents[4]
    setup = runpy.run_path(str(root / "setup.py"))
    return setup["get_packages"].__globals__


@pytest.mark.parametrize("value,expected", [
    (None, ["nvidia", "amd"]),
    ("", []),
    ("nvidia", ["nvidia"]),
    ("amd", ["amd"]),
    (" amd ; nvidia ; amd ", ["amd", "nvidia"]),
])
def test_codegen_backends_selection(value, expected, build_setup, monkeypatch):
    if value is None:
        monkeypatch.delenv("TRITON_CODEGEN_BACKENDS")
    else:
        monkeypatch.setenv("TRITON_CODEGEN_BACKENDS", value)
    assert build_setup["get_codegen_backends"]() == expected


def test_codegen_backends_invalid(build_setup, monkeypatch):
    monkeypatch.setenv("TRITON_CODEGEN_BACKENDS", "example")
    with pytest.raises(ValueError, match="Use TRITON_PLUGIN_DIRS for external backends"):
        build_setup["get_codegen_backends"]()


@pytest.mark.parametrize("selected", [(), ("nvidia", ), ("example", )])
def test_backend_packages_selection(selected, build_setup, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(build_setup, "__file__", str(tmp_path / "setup.py"))
    for package in ("triton", "triton/backends"):
        directory = tmp_path / "python" / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").touch()

    available = {}
    for name in ("nvidia", "amd", "example"):
        source = tmp_path / "third_party" / name / "backend"
        (source / "nested").mkdir(parents=True)
        (source / "__init__.py").touch()
        (source / "nested/__init__.py").touch()
        link = tmp_path / "python/triton/backends" / name
        link.symlink_to(source, target_is_directory=True)
        available[name] = SimpleNamespace(name=name, backend_dir=str(source), install_dir=str(link), language_dir=None,
                                          tools_dir=None, is_external=name == "example")
    # GPU links represent a prior editable install; retain the external plugin.
    monkeypatch.setitem(build_setup, "backends", [available[name] for name in selected])
    if "example" not in selected:
        Path(available["example"].install_dir).unlink()
    packages = set(build_setup["get_packages"]())
    assert build_setup["get_entry_points"]()["triton.backends"] == [
        f"{name} = triton.backends.{name}" for name in selected
    ]
    build_setup["add_link_to_backends"](external_only=True)
    for name in available:
        assert (f"triton.backends.{name}" in packages) == (name in selected)
        assert (f"triton.backends.{name}.nested" in packages) == (name in selected)
        assert Path(available[name].install_dir).is_symlink() == (name in selected)
