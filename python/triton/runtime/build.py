from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sysconfig
import tempfile
import logging
from pathlib import Path

from types import ModuleType

from .cache import get_cache_manager
from .. import knobs


@functools.lru_cache()
def _find_compiler(language: str) -> str:
    if language == "c":
        cc = os.environ.get("CC")
        if cc is not None:
            return cc
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is not None:
            return cc
        raise RuntimeError(
            "Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.")

    assert language == "c++"
    cxx = os.environ.get("CXX")
    if cxx is not None:
        return cxx

    clangxx = shutil.which("clang++")
    gxx = shutil.which("g++")
    cxx = gxx if gxx is not None else clangxx
    if cxx is not None:
        return cxx

    raise RuntimeError(
        "Failed to find C++ compiler. Please specify via CXX environment variable or set triton.knobs.build.impl.")


def _language_from_filename(source_name: str) -> str:
    ext = Path(source_name).suffix
    if ext == ".c":
        return "c"
    if ext in {".cc", ".cpp", ".cxx"}:
        return "c++"
    print(source_name)
    raise ValueError(f"Unrecognized file extension: {source_name}")


def _build(name: str, src: str, srcdir: str, library_dirs: list[str], include_dirs: list[str], libraries: list[str],
           ccflags: list[str], language: str = "c") -> str:
    if impl := knobs.build.impl:
        return impl(name, src, srcdir, library_dirs, include_dirs, libraries)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, f'{name}{suffix}')
    cc = _find_compiler(language)
    scheme = sysconfig.get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = knobs.build.backend_dirs
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
    # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
    cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so]
    if language == "c++":
        cc_cmd.insert(3, "-std=c++17")
    cc_cmd += [_library_flag(lib) for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    cc_cmd.extend(ccflags)
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    return so


def _library_flag(lib: str) -> str:
    # Match .so files with optional version numbers (e.g., .so, .so.1, .so.513.50.1)
    if re.search(r'\.so(\.\d+)*$', lib) or lib.endswith(".a"):
        return f"-l:{lib}"
    return f"-l{lib}"


@functools.lru_cache
def platform_key() -> str:
    from platform import machine, system, architecture
    return ",".join([machine(), system(), *architecture()])


def _get_file_extension(language):
    if language == "c":
        return ".c"
    if language == "c++":
        return ".cpp"
    raise ValueError(f"Unexpected languange: {language}")


def _load_module_from_path(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load newly compiled {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_cache_manager(src: bytes, language: str):
    digest = hashlib.sha256()
    digest.update(src)
    digest.update(platform_key().encode("utf-8"))
    digest.update(language.encode("utf-8"))
    key = digest.hexdigest()
    return get_cache_manager(key)


def _compile_so(src: bytes, src_path: str, name: str, library_dirs: list[str] | None, include_dirs: list[str] | None,
                libraries: list[str] | None, ccflags: list[str] | None, load_module: bool, language: str):
    cache = _get_cache_manager(src, language)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    cache_path = cache.get_file(f"{name}{suffix}")
    if cache_path is not None:
        if not load_module:
            return cache_path
        try:
            return _load_module_from_path(name, cache_path)
        except (RuntimeError, ImportError):
            log = logging.getLogger(__name__)
            log.warning(f"Triton cache error: compiled module {name}.so could not be loaded")

    with tempfile.TemporaryDirectory() as tmpdir:
        so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [],
                    language=language)
        with open(so, "rb") as f:
            cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)

    return _load_module_from_path(name, cache_path) if load_module else cache_path


def _compile_so_from_file(src_path: str, name: str, library_dirs: list[str] | None, include_dirs: list[str] | None,
                          libraries: list[str] | None, ccflags: list[str] | None, load_module: bool):
    src_path = os.path.abspath(src_path)
    src_name = os.path.basename(src_path)
    with open(src_path, "rb") as f:
        src = f.read()

    language = _language_from_filename(src_name)
    return _compile_so(src=src, src_path=src_path, name=name, library_dirs=library_dirs, include_dirs=include_dirs,
                       libraries=libraries, ccflags=ccflags, language=language, load_module=load_module)


def _compile_so_from_src(src: str, name: str, library_dirs: list[str] | None, include_dirs: list[str] | None,
                         libraries: list[str] | None, ccflags: list[str] | None, language, load_module: bool):
    src_bytes = src.encode("utf-8")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"{name}{_get_file_extension(language)}")
        with open(src_path, "wb") as f:
            f.write(src_bytes)
        return _compile_so(src=src_bytes, src_path=src_path, name=name, library_dirs=library_dirs,
                           include_dirs=include_dirs, libraries=libraries, ccflags=ccflags, language=language,
                           load_module=load_module)


def compile_so_from_file(src_path: str, name: str, library_dirs: list[str] | None = None,
                         include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                         ccflags: list[str] | None = None) -> str:
    return _compile_so_from_file(src_path, name, library_dirs, include_dirs, libraries, ccflags, load_module=False)


def compile_so_from_src(src: str, name: str, library_dirs: list[str] | None = None,
                        include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                        ccflags: list[str] | None = None, language="c") -> str:
    return _compile_so_from_src(src, name, library_dirs, include_dirs, libraries, ccflags, language, load_module=False)


def compile_module_from_file(src_path: str, name: str, library_dirs: list[str] | None = None,
                             include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                             ccflags: list[str] | None = None) -> ModuleType:
    return _compile_so_from_file(src_path, name, library_dirs, include_dirs, libraries, ccflags, load_module=True)


def compile_module_from_src(src: str, name: str, library_dirs: list[str] | None = None,
                            include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                            ccflags: list[str] | None = None, language="c") -> ModuleType:
    return _compile_so_from_src(src, name, library_dirs, include_dirs, libraries, ccflags, language, load_module=True)
