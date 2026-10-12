from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sysconfig
import tarfile
import tempfile

import pytest

from pathlib import Path

import triton

from triton.runtime import build
from triton.runtime.build import _get_cache_manager, compile_module_from_src

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

TEST_MODULE_COMPILER = TEST_MODULE_C.replace(
    "#include <Python.h>", """#include <Python.h>
#if defined(__clang__)
#define COMPILER_NAME "clang"
#elif defined(__GNUC__)
#define COMPILER_NAME "gcc"
#else
#define COMPILER_NAME "unknown"
#endif
""").replace('res = "hiya";', "res = COMPILER_NAME;")


def test_compile_module(fresh_triton_cache):
    mod = compile_module_from_src(TEST_MODULE_C, "test_module")

    with pytest.raises(Exception):
        mod.go()

    assert mod.go("huh") == "huh"
    assert mod.go("hello") == "hiya"

    # Make sure the module is cached
    mod2 = compile_module_from_src(TEST_MODULE_C, "test_module")
    assert mod2.__file__ == mod.__file__


@pytest.mark.parametrize("option", ["include_dirs", "library_dirs", "libraries", "ccflags"])
def test_build_cache_distinguishes_colons_in_options(fresh_triton_cache, option):
    first = _get_cache_manager(b"source", {option: ["a:b", "c"]})
    second = _get_cache_manager(b"source", {option: ["a", "b:c"]})
    assert first.key != second.key


def test_compile_module_distinguishes_include_directories(fresh_triton_cache, tmp_path, monkeypatch):
    first_dirs = [tmp_path / "a:b", tmp_path / "c"]
    second_dirs = [tmp_path / "a", tmp_path / "b:c"]
    for directory in first_dirs + second_dirs:
        directory.mkdir()
    (first_dirs[0] / "greeting.h").write_text('#define GREETING "first"\n')
    (second_dirs[0] / "greeting.h").write_text('#define GREETING "second"\n')
    source = TEST_MODULE_C.replace("#include <Python.h>", '#include <Python.h>\n#include "greeting.h"')
    source = source.replace('"hiya"', "GREETING")
    monkeypatch.chdir(tmp_path)

    first = compile_module_from_src(source, "test_module", include_dirs=[directory.name for directory in first_dirs])
    second = compile_module_from_src(source, "test_module", include_dirs=[directory.name for directory in second_dirs])
    assert first.go("hello") == "first"
    assert second.go("hello") == "second"
    assert first.__file__ != second.__file__


@pytest.mark.parametrize(
    ("language", "env_var", "compilers"),
    [("c", "CC", ("gcc", "clang")), ("c++", "CXX", ("g++", "clang++"))],
)
def test_compile_module_cache_keys_compiler(language, env_var, compilers, fresh_triton_cache, monkeypatch):
    compiler_paths = [shutil.which(compiler) for compiler in compilers]
    if any(path is None for path in compiler_paths):
        pytest.skip(f"requires both {compilers[0]} and {compilers[1]}")
    compiler_paths = [path for path in compiler_paths if path is not None]
    if Path(compiler_paths[0]).resolve() == Path(compiler_paths[1]).resolve():
        pytest.skip("requires two distinct compiler executables")

    module_paths = []
    expected_compilers = ("gcc", "clang")
    try:
        for compiler, expected in zip(compilers, expected_compilers):
            monkeypatch.setenv(env_var, compiler)
            # A separate process would select the compiler from its current environment.
            build._find_compiler.cache_clear()
            mod = compile_module_from_src(TEST_MODULE_COMPILER, "test_module", language=language)
            assert mod.go("hello") == expected
            module_paths.append(mod.__file__)

            cached_mod = compile_module_from_src(TEST_MODULE_COMPILER, "test_module", language=language)
            assert cached_mod.__file__ == mod.__file__
            assert cached_mod.go("hello") == expected
    finally:
        build._find_compiler.cache_clear()

    assert module_paths[0] != module_paths[1]


def test_compile_module_cache_keys_resolved_compiler_path(fresh_triton_cache, monkeypatch, tmp_path):
    if os.name != "posix":
        pytest.skip("requires POSIX executable symlinks")

    compiler_paths = [shutil.which(compiler) for compiler in ("gcc", "clang")]
    if any(path is None for path in compiler_paths):
        pytest.skip("requires both gcc and clang")
    compiler_paths = [path for path in compiler_paths if path is not None]

    module_paths = []
    original_path = os.environ.get("PATH", "")
    try:
        for compiler, compiler_path, expected in zip(("gcc", "clang"), compiler_paths, ("gcc", "clang")):
            alias_dir = tmp_path / compiler
            alias_dir.mkdir()
            (alias_dir / "cc").symlink_to(compiler_path)
            monkeypatch.setenv("CC", "cc")
            monkeypatch.setenv("PATH", os.pathsep.join((str(alias_dir), original_path)))
            build._find_compiler.cache_clear()

            mod = compile_module_from_src(TEST_MODULE_COMPILER, "test_module")
            assert mod.go("hello") == expected
            module_paths.append(mod.__file__)
    finally:
        build._find_compiler.cache_clear()

    assert module_paths[0] != module_paths[1]


def test_custom_build_impl_does_not_require_compiler(fresh_triton_cache, monkeypatch):
    mod = compile_module_from_src(TEST_MODULE_C, "test_module")

    def custom_impl(name, src, srcdir, library_dirs, include_dirs, libraries):
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        output = Path(srcdir) / f"{name}{suffix}"
        shutil.copyfile(mod.__file__, output)
        return str(output)

    monkeypatch.setattr(triton.knobs.build, "impl", custom_impl)
    monkeypatch.setattr(build, "_find_compiler", lambda language: pytest.fail("custom builder selected a compiler"))

    custom_mod = compile_module_from_src(f"/* custom builder */\n{TEST_MODULE_C}", "test_module")
    assert custom_mod.go("hello") == "hiya"


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
