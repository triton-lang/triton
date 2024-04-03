import multiprocessing
import os
import subprocess

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


def get_cuda_root():
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]
    elif "CUDA_ROOT" in os.environ:
        return os.environ["CUDA_ROOT"]
    else:
        return ""


def get_triton_cache_path():
    user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".triton")


class CMakeExtension(Extension):

    def __init__(self, name, path, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path


class CMakeBuild(build_ext):

    def initialize_options(self):
        build_ext.initialize_options(self)

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def get_cmake_args(self, extdir):
        # XXX: Check if pybind11 is installed
        pybind11_path = get_triton_cache_path() + "/pybind11/pybind11-2.11.1/include"
        if not os.path.exists(pybind11_path):
            raise RuntimeError(f"pybind11 is not found under {pybind11_path}. Please install triton first.")
        cmake_args = [
            "-DPYBIND11_INCLUDE_DIR=" + pybind11_path,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DCMAKE_BUILD_TYPE=" + get_build_type(),
        ]
        if get_build_type() != "Release":
            cmake_args += ["-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
        if cuda_root := get_cuda_root():
            cmake_args += ["-DCUDAToolkit_ROOT=" + cuda_root]
        return cmake_args

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # Create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmake_args = self.get_cmake_args(extdir)
        build_args = ["--", "-j" + str(multiprocessing.cpu_count())]

        base_dir = os.path.abspath(os.path.dirname(__file__))
        env = os.environ.copy()
        subprocess.check_call(["cmake", base_dir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


setup(
    name="proton",
    version="0.0.1",
    author="Keren Zhou",
    author_email="kerenzhou@outlook.com",
    description="A profiler for Triton",
    long_description="",
    packages=["proton", "proton/_C"],
    install_requires=[
        "setuptools",
        "cmake",
        "triton",
        "llnl-hatchet",
    ],
    py_modules=["proton"],
    entry_points={"console_scripts": ["proton-viewer = proton.viewer:main"]},
    package_data={"python/_C": ["*.so", "*.pyi"]},
    include_package_data=True,
    ext_modules=[CMakeExtension("proton", "proton/_C/")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    keywords=["Profiler", "Deep Learning"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require={"dev": ["pytest", "pre-commit"]},
)
