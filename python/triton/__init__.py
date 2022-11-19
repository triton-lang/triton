"""isort:skip_file"""
# flake8: noqa: F401
__version__ = "2.0.0"

# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch

# submodules

from triton._C.libtriton.triton import ir

from . import utils
from .jitlib import (
    KernelInterface,
    JITFunction,
    jit,
    ExternalFunction,
    extern,
)
from .compiler import compile, CompiledKernel, CompilationError, OutOfResources

# No @tr.jit() interface can be called until after .compiler is loaded;
# and .compiler depends upon the core stack.

from .tuning import (
    autotune,
    Config,
    heuristics,
    Heuristics,
)

from . import language
from . import testing
from . import ops

# unconstrained
