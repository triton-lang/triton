"""isort:skip_file"""
# flake8: noqa: F401
__version__ = "2.0.0"

# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch

# submodules

from triton._C.libtriton.triton import ir

from . import utils

# No @tr.jit() interface can be called until after .compiler is loaded;
# and .compiler depends upon the core stack.
from .jitlib import (
    extern,
    ExternalFunction,
    jit,
    JITFunction,
    KernelInterface,
)

# .compiler depends upon core.minimum and core.where
from .compiler import (
    CompilationError,
    compile,
    CompiledKernel,
    OutOfResources,
)

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

__all__ = [
    "autotune",
    "CompilationError",
    "compile",
    "CompiledKernel",
    "Config",
    "extern",
    "ExternalFunction",
    "heuristics",
    "Heuristics",
    "ir",
    "jit",
    "JITFunction",
    "KernelInterface",
    "language",
    "ops",
    "OutOfResources",
    "testing",
    "utils",
]
