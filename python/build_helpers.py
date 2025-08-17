import os
import sysconfig
import sys
from pathlib import Path


def get_base_dir():
    return Path(__file__).parent.parent


def _get_build_base():
    build_base = os.getenv("TRITON_BUILD_DIR", default=(get_base_dir() / "build"))
    return Path(build_base)


def _get_dir_common(prefix):
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"{prefix}.{plat_name}-{sys.implementation.name}-{python_version}"
    path = _get_build_base() / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cmake_dir():
    return _get_dir_common("cmake")


def get_build_lib():
    return _get_dir_common("lib")


def get_build_temp():
    return _get_dir_common("temp")


def get_bdist_dir():
    return _get_dir_common("bdist")
