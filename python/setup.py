import distutils
import distutils.spawn
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
    else:
        return "Release"


# --- third party packages -----

class Package(NamedTuple):
    package: str
    name: str
    url: str
    test_file: str
    include_flag: str
    lib_flag: str


def get_pybind11_package_info():
    name = "pybind11-2.10.0"
    url = "https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.tar.gz"
    return Package("pybind11", name, url, "include/pybind11/pybind11.h", "PYBIND11_INCLUDE_DIR", "")


def get_llvm_package_info():
    # download if nothing is installed
    system = platform.system()
    system_suffix = {"Linux": "linux-gnu-ubuntu-18.04", "Darwin": "apple-darwin"}[system]
    use_assert_enabled_llvm = check_env_flag("TRITON_USE_ASSERT_ENABLED_LLVM", "False")
    if use_assert_enabled_llvm:
        name = 'llvm+mlir-14.0.0-x86_64-{}-assert'.format(system_suffix)
        url = "https://github.com/shintaro-iwasaki/llvm-releases/releases/download/llvm-14.0.0-329fda39c507/{}.tar.xz".format(name)
    else:
        name = 'clang+llvm-14.0.0-x86_64-{}'.format(system_suffix)
        url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/{}.tar.xz".format(name)
    return Package("llvm", name, url, "lib", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR")


def get_thirdparty_packages(triton_cache_path):
    packages = [get_pybind11_package_info(), get_llvm_package_info()]
    thirdparty_cmake_args = []
    for p in packages:
        package_root_dir = os.path.join(triton_cache_path, p.package)
        package_dir = os.path.join(package_root_dir, p.name)
        test_file_path = os.path.join(package_dir, p.test_file)
        if not os.path.exists(test_file_path):
            try:
                shutil.rmtree(package_root_dir)
            except Exception:
                pass
            os.makedirs(package_root_dir, exist_ok=True)
            print('downloading and extracting {} ...'.format(p.url))
            ftpstream = urllib.request.urlopen(p.url)
            file = tarfile.open(fileobj=ftpstream, mode="r|*")
            file.extractall(path=package_root_dir)
        if p.include_flag:
            thirdparty_cmake_args.append("-D{}={}/include".format(p.include_flag, package_dir))
        if p.lib_flag:
            thirdparty_cmake_args.append("-D{}={}/lib".format(p.lib_flag, package_dir))
    return thirdparty_cmake_args

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
        self.debug = True
        lit_dir = shutil.which('lit')
        triton_cache_path = os.path.join(os.environ["HOME"], ".triton")
        # lit is used by the test suite
        thirdparty_cmake_args = get_thirdparty_packages(triton_cache_path)
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # create build directories
        build_suffix = 'debug' if self.debug else 'release'
        llvm_build_dir = os.path.join(tempfile.gettempdir(), "llvm-" + build_suffix)
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if not os.path.exists(llvm_build_dir):
            os.makedirs(llvm_build_dir)
        # python directories
        python_include_dirs = [distutils.sysconfig.get_python_inc()] + ['/usr/local/cuda/include']
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DTRITON_BUILD_TUTORIALS=OFF",
            "-DTRITON_BUILD_PYTHON_MODULE=ON",
            # '-DPYTHON_EXECUTABLE=' + sys.executable,
            # '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON',
            "-DPYTHON_INCLUDE_DIRS=" + ";".join(python_include_dirs),
            "-DLLVM_EXTERNAL_LIT=" + lit_dir
        ] + thirdparty_cmake_args

        # configuration
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            import multiprocessing
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", '-j' + str(2 * multiprocessing.cpu_count())]

        env = os.environ.copy()
        subprocess.check_call(["cmake", self.base_dir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


setup(
    name="triton",
    version="2.0.0",
    author="Philippe Tillet",
    author_email="phil@openai.com",
    description="A language and compiler for custom Deep Learning operations",
    long_description="",
    packages=["triton", "triton/_C", "triton/language", "triton/tools", "triton/ops", "triton/runtime", "triton/ops/blocksparse"],
    install_requires=[
        "cmake",
        "filelock",
        "torch",
        "lit",
    ],
    package_data={"triton/ops": ["*.c"], "triton/ops/blocksparse": ["*.c"]},
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
