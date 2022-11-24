"""isort:skip_file"""
__version__ = '2.0.0'

# ---------------------------------------
# Note: import order is significant here.

# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch  # noqa: F401

# submodules
from . import impl
from .utils import (
    cdiv,
    MockTensor,
    next_power_of_2,
    reinterpret,
    TensorWrapper,
)
from .runtime import (
    autotune,
    Config,
    heuristics,
    JITFunction,
    KernelInterface,
)
from .runtime.jit import jit
from .compiler import compile, CompilationError
from . import language
from . import testing
from . import ops

__all__ = [
    "autotune",
    "cdiv",
    "CompilationError",
    "compile",
    "Config",
    "heuristics",
    "impl",
    "jit",
    "JITFunction",
    "KernelInterface",
    "language",
    "MockTensor",
    "next_power_of_2",
    "ops",
    "reinterpret",
    "runtime",
    "TensorWrapper",
    "testing",
]
