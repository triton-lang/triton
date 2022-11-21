"""isort:skip_file"""
# flake8: noqa: F401
__version__ = "2.0.0"

# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch

# Import order is significant here.

from .utils import cdiv, next_power_of_2

from triton._C.libtriton.triton import ir

from .impl import (
    autotune,
    CompilationError,
    compile,
    CompiledKernel,
    Config,
    extern,
    heuristics,
    Heuristics,
    jit,
    JITFunction,
    KernelInterface,
    MockTensor,
    OutOfResources,
    reinterpret,
    TensorWrapper,
)

from . import language

from . import runtime
from . import testing
from . import ops

# unconstrained

__all__ = [
    "autotune",
    "cdiv",
    "CompilationError",
    "compile",
    "CompiledKernel",
    "Config",
    "extern",
    "heuristics",
    "Heuristics",
    "ir",
    "jit",
    "JITFunction",
    "KernelInterface",
    "language",
    "MockTensor",
    "next_power_of_2",
    "ops",
    "OutOfResources",
    "reinterpret",
    "runtime",
    "TensorWrapper",
    "testing",
    "utils",
]
