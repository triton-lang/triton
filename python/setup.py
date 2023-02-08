import distutils
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from distutils.version import LooseVersion
from typing import NamedTuple

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_build_type():
    if check_env_flag("DEBUG"):
        return "Debug"
    elif check_env_flag("REL_WITH_DEB_INFO"):
        return "RelWithDebInfo"
    elif check_env_flag("TRITON_REL_BUILD_WITH_ASSERTS"):
        return "TritonRelBuildWithAsserts"
    else:
        # TODO: change to release when stable enough
        return "TritonRelBuildWithAsserts"

# --- third party packages -----


class Package(NamedTuple):
    package: str
    name: str
    url: str
    test_file: str
    include_flag: str
    lib_flag: str
    syspath_var_name: str

# pybind11


def get_pybind11_package_info():
    name = "pybind11-2.10.0"
    url = "https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.tar.gz"
    return Package("pybind11", name, url, "include/pybind11/pybind11.h", "PYBIND11_INCLUDE_DIR", "", "PYBIND11_SYSPATH")

# llvm


def get_llvm_package_info():
    # download if nothing is installed
    system = platform.system()
    if system == "Darwin":
        system_suffix = "apple-darwin"
    elif system == "Linux":
        vglibc = tuple(map(int, platform.libc_ver()[1].split('.')))
        vglibc = vglibc[0] * 100 + vglibc[1]
        linux_suffix = 'ubuntu-18.04' if vglibc > 217 else 'centos-7'
        system_suffix = f"linux-gnu-{linux_suffix}"
    else:
        raise RuntimeError(f"unsupported system: {system}")
    use_assert_enabled_llvm = check_env_flag("TRITON_USE_ASSERT_ENABLED_LLVM", "False")
    release_suffix = "assert" if use_assert_enabled_llvm else "release"
    name = f'llvm+mlir-14.0.6-x86_64-{system_suffix}-{release_suffix}'
    url = f"https://github.com/ptillet/triton-llvm-releases/releases/download/llvm-14.0.6-f28c006a5895/{name}.tar.xz"
    return Package("llvm", name, url, "lib", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")


def get_thirdparty_packages(triton_cache_path):
    packages = [get_pybind11_package_info(), get_llvm_package_info()]
    thirdparty_cmake_args = []
    for p in packages:
        package_root_dir = os.path.join(triton_cache_path, p.package)
        package_dir = os.path.join(package_root_dir, p.name)
        if p.syspath_var_name in os.environ:
            package_dir = os.environ[p.syspath_var_name]
        test_file_path = os.path.join(package_dir, p.test_file)
        if not os.path.exists(test_file_path):
            try:
                shutil.rmtree(package_root_dir)
            except Exception:
                pass
            os.makedirs(package_root_dir, exist_ok=True)
            print(f'downloading and extracting {p.url} ...')
            ftpstream = urllib.request.urlopen(p.url)
            file = tarfile.open(fileobj=ftpstream, mode="r|*")
            file.extractall(path=package_root_dir)
        if p.include_flag:
            thirdparty_cmake_args.append(f"-D{p.include_flag}={package_dir}/include")
        if p.lib_flag:
            thirdparty_cmake_args.append(f"-D{p.lib_flag}={package_dir}/lib")
    return thirdparty_cmake_args

# ---- package data ---


def download_and_copy_ptxas():
    base_dir = os.path.dirname(__file__)
    src_path = "bin/ptxas"
    url = "https://conda.anaconda.org/nvidia/label/cuda-12.0.0/linux-64/cuda-nvcc-12.0.76-0.tar.bz2"
    dst_prefix = os.path.join(base_dir, "triton")
    dst_suffix = os.path.join("third_party", "cuda", src_path)
    dst_path = os.path.join(dst_prefix, dst_suffix)
    if not os.path.exists(dst_path):
        print(f'downloading and extracting {url} ...')
        ftpstream = urllib.request.urlopen(url)
        file = tarfile.open(fileobj=ftpstream, mode="r|*")
        with tempfile.TemporaryDirectory() as temp_dir:
            file.extractall(path=temp_dir)
            src_path = os.path.join(temp_dir, src_path)
            os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
            shutil.copy(src_path, dst_path)
    return dst_suffix


# ---- cmake extension ----


class CMakeExtension(Extension):
    def __init__(self, name, path, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path


class CMakeBuild(build_ext):

    user_options = build_ext.user_options + [('base-dir=', None, 'base directory of Triton')]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        lit_dir = shutil.which('lit')
        triton_cache_path = os.path.join(os.environ["HOME"], ".triton")
        # lit is used by the test suite
        thirdparty_cmake_args = get_thirdparty_packages(triton_cache_path)
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # python directories
        python_include_dir = distutils.sysconfig.get_python_inc()
        cmake_args = [
            "-DLLVM_ENABLE_WERROR=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DTRITON_BUILD_TUTORIALS=OFF",
            "-DTRITON_BUILD_PYTHON_MODULE=ON",
            "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
            "-DPYTHON_INCLUDE_DIRS=" + python_include_dir,
        ]
        if lit_dir is not None:
            cmake_args.append("-DLLVM_EXTERNAL_LIT=" + lit_dir)
        cmake_args.extend(thirdparty_cmake_args)

        # configuration
        cfg = get_build_type()
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            import multiprocessing
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ['-j' + str(2 * multiprocessing.cpu_count())]

        env = os.environ.copy()
        subprocess.check_call(["cmake", self.base_dir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


download_and_copy_ptxas()

setup(
    name="triton",
    version="2.0.0",
    author="Philippe Tillet",
    author_email="phil@openai.com",
    description="A language and compiler for custom Deep Learning operations",
    long_description="",
    packages=["triton", "triton/_C", "triton/language", "triton/tools", "triton/impl", "triton/ops", "triton/runtime", "triton/ops/blocksparse"],
    install_requires=[
        "cmake",
        "filelock",
        "torch",
        "lit",
    ],
    package_data={"triton": ["third_party/**/*"]},
    include_package_data=True,
    ext_modules=[CMakeExtension("triton", "triton/_C/")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    # for PyPI
    keywords=["Compiler", "Deep Learning"],
    url="https://github.com/openai/triton/",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    test_suite="tests",
    extras_require={
        "tests": [
            "autopep8",
            "flake8",
            "isort",
            "numpy",
            "pytest",
            "scipy>=1.7.1",
        ],
        "tutorials": [
            "matplotlib",
            "pandas",
            "tabulate",
        ],
    },
)
