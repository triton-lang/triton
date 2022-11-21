"""isort:skip_file"""
# flake8: noqa: F401
__version__ = "2.0.0"

# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch

# submodules

from .impl import (
    autotune,
    cdiv,
    CompilationError,
    compile,
    CompiledKernel,
    Config,
    extern,
    ExternalFunction,
    heuristics,
    Heuristics,
    ir,
    jit,
    JITFunction,
    KernelInterface,
    MockTensor,
    next_power_of_2,
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
    "ExternalFunction",
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
