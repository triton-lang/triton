import os
import sysconfig
import sys
from pathlib import Path


def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    return Path(get_base_dir()) / "build" / dir_name


def get_cmake_dir():
    cmake_dir = os.getenv("TRITON_BUILD_DIR", default=_get_cmake_dir())
    cmake_dir = Path(cmake_dir)
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir
