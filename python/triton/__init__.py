"""isort:skip_file"""
# flake8: noqa: F401
__version__ = '2.0.0'

# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch
# submodules
from .utils import *
from .runtime import Config, autotune, heuristics, JITFunction, KernelInterface
from .runtime.jit import jit
from .compiler import compile, CompilationError
from . import language
from . import testing
from . import ops
