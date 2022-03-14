"""isort:skip_file"""
# flake8: noqa: F401
__version__ = '2.0.0'

# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch
# submodules
from .code_gen import cdiv, next_power_of_2, jit, autotune, heuristics, \
    JITFunction, Config, Autotuner, reinterpret
from . import language
from . import code_gen
from . import testing
from . import ops
