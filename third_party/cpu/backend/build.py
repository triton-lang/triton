import hashlib
import importlib.resources
import importlib.util
import os
import platform
import subprocess
import tempfile

import triton
from triton import knobs
from triton.runtime.build import _build as _upstream_build
from triton.runtime.build import _find_compiler
from triton.runtime.cache import get_cache_manager

_system_root = os.getenv("TRITON_SYS_PATH", default="/usr/local")

# Locate the packaged TritonCPU runtime libraries.
try:
    _triton_c_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9.
    _triton_c_dir = importlib.resources.path(triton, "_C").__enter__()

_include_dirs = []
_library_dirs = [_triton_c_dir]

_system_include_dir = os.path.join(_system_root, "include")
if os.path.exists(_system_include_dir):
    _include_dirs.append(_system_include_dir)

_system_library_dir = os.path.join(_system_root, "lib")
if os.path.exists(_system_library_dir):
    _library_dirs.append(_system_library_dir)


def _is_apple_clang(compiler):
    if platform.system() != "Darwin":
        return False
    result = subprocess.run([compiler, "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return "Apple clang" in result.stdout


def _build_cpu_shared_object(name, src, srcdir, libraries, ccflags, source_kind):
    library_dirs = list(_library_dirs)
    include_dirs = list(_include_dirs)
    libraries = list(libraries)
    ccflags = list(ccflags)

    # Preserve upstream's build override behavior: custom implementations run
    # before TritonCPU adds any compiler or linker policy.
    if knobs.build.impl:
        return _upstream_build(name, src, srcdir, library_dirs, include_dirs, libraries, ccflags)

    compiler = _find_compiler("c")
    system = platform.system()
    machine = platform.machine()
    apple_clang = _is_apple_clang(compiler)

    libraries.append("gcc")
    cpu_flags = []

    if system == "Darwin":
        # Python extension modules resolve Python symbols from the interpreter.
        cpu_flags += ["-undefined", "dynamic_lookup"]
        if apple_clang:
            libraries.pop()

    for library_dir in library_dirs:
        cpu_flags.extend(["-Wl,-rpath", library_dir])

    if source_kind == "launcher":
        # Keep using CC plus explicit libstdc++, matching the existing CPU
        # launcher build rather than switching compiler selection to CXX.
        cpu_flags.append("-std=c++17")
        if not os.environ.get("TRITON_DISABLE_OPENMP"):
            libomp_path = os.environ.get("TRITON_LOCAL_LIBOMP_PATH")
            if apple_clang:
                if libomp_path:
                    cpu_flags += [
                        "-Xclang",
                        "-fopenmp",
                        f"-I{libomp_path}/include",
                        f"-L{libomp_path}/lib",
                        "-lomp",
                    ]
                else:
                    print("Warning: TRITON_LOCAL_LIBOMP_PATH is not set for Apple clang. OpenMP is disabled.")
            else:
                cpu_flags.append("-fopenmp")
                if libomp_path:
                    print("Info: Ignoring TRITON_LOCAL_LIBOMP_PATH for non-Apple clang compiler")
    elif source_kind == "assembly":
        # Preserve .file directives in generated host assembly.
        cpu_flags.append("-g")
        if system == "Darwin" and machine in ("aarch64", "arm64"):
            # LLVM can emit CSSC instructions on Apple Silicon, but Apple clang
            # does not enable that extension for standalone assembly by default.
            cpu_flags.append("-mcpu=native+cssc")
        if system == "Linux" and machine in ("aarch64", "arm64"):
            # Arm assembly may need an explicit target to enable host features.
            cpu_flags.append("-mcpu=native")
    else:
        raise ValueError(f"Unexpected CPU source kind: {source_kind}")

    return _upstream_build(name, src, srcdir, library_dirs, include_dirs, libraries, cpu_flags + ccflags)


def compile_launcher_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as src_file:
                src_file.write(src)
            so = _build_cpu_shared_object(
                name,
                src_path,
                tmpdir,
                libraries=["stdc++"],
                ccflags=[],
                source_kind="launcher",
            )
            with open(so, "rb") as so_file:
                cache_path = cache.put(so_file.read(), f"{name}.so", binary=True)

    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_kernel_from_asm(src_path, srcdir):
    return _build_cpu_shared_object(
        "kernel",
        src_path,
        srcdir,
        libraries=["m", "TritonCPURuntime", "sleef"],
        ccflags=[],
        source_kind="assembly",
    )
