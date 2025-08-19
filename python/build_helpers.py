import os
from pathlib import Path

# base_dir: Root of source code


def _get_base_dir() -> Path:
    return Path(__file__).parent.parent


def get_base_dir() -> str:
    return _get_base_dir().as_posix()


def get_build_base() -> str:
    build_base = os.getenv("TRITON_BUILD_DIR", default=(_get_base_dir() / "build"))
    build_base = Path(build_base)
    build_base.mkdir(parents=True, exist_ok=True)
    return build_base.as_posix()


# cmd: setuptools.Command
# ext: setuptools.Extension
# Cannot use type hints due to ruff F821
def get_cmake_dir(cmd, ext):
    cmake_dir = Path(cmd.build_temp) / ext.name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir
