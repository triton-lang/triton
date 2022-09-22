import distutils
import distutils.spawn
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from distutils.version import LooseVersion

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


def check_submodule():
    submodule_paths = ["third-party/pybind11/include/pybind11"]
    if not all([os.path.exists(p) for p in submodule_paths]):
        print("initializing submodules ...")
        try:
            cwd = os.path.abspath(os.path.dirname(__file__))
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=cwd)
            print("submodule initialization succeeded")
        except Exception:
            print("submodule initialization failed")
            print(" Please run:\n\tgit submodule update --init --recursive")
            exit(-1)


def get_llvm():
    # tries to find system LLVM
    versions = ['-11.0', '-11', '-11-64']
    supported = ['llvm-config{v}'.format(v=v) for v in versions]
    paths = [distutils.spawn.find_executable(cfg) for cfg in supported]
    paths = [p for p in paths if p is not None]
    if paths:
        return '', ''
    if platform.system() == "Windows":
        return '', ''
    # download if nothing is installed
    name = 'clang+llvm-11.0.1-x86_64-linux-gnu-ubuntu-16.04'
    dir = os.path.join(os.environ["HOME"], ".triton", "llvm")
    llvm_include_dir = '{dir}/{name}/include'.format(dir=dir, name=name)
    llvm_library_dir = '{dir}/{name}/lib'.format(dir=dir, name=name)
    if not os.path.exists(llvm_library_dir):
        os.makedirs(dir, exist_ok=True)
        try:
            shutil.rmtree(os.path.join(dir, name))
        except Exception:
            pass
        url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/{name}.tar.xz".format(name=name)
        print('downloading and extracting ' + url + '...')
        ftpstream = urllib.request.urlopen(url)
        file = tarfile.open(fileobj=ftpstream, mode="r|xz")
        file.extractall(path=dir)
    return llvm_include_dir, llvm_library_dir


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
        check_submodule()
        llvm_include_dir, llvm_library_dir = get_llvm()
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # python directories
        python_include_dirs = [distutils.sysconfig.get_python_inc()]
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DBUILD_TUTORIALS=OFF",
            "-DBUILD_PYTHON_MODULE=ON",
            "-DLLVM_INCLUDE_DIRS=" + llvm_include_dir,
            "-DLLVM_LIBRARY_DIR=" + llvm_library_dir,
            # '-DPYTHON_EXECUTABLE=' + sys.executable,
            # '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON',
            "-DPYTHON_INCLUDE_DIRS=" + ";".join(python_include_dirs)
        ]
        # configuration
        cfg = get_build_type()
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
    packages=["triton", "triton/_C", "triton/language", "triton/runtime", "triton/tools", "triton/ops", "triton/ops/blocksparse"],
    install_requires=[
        "cmake",
        "filelock",
        "torch",
    ],
    package_data={
        "triton/ops": ["*.c"],
        "triton/ops/blocksparse": ["*.c"],
        "triton/language": ["*.bc"],
    },
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
