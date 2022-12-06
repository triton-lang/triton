"""isort:skip_file"""
__version__ = '2.0.0'

# ---------------------------------------
# Note: import order is significant here.

# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch  # noqa: F401

# submodules
from .utils import (
    cdiv,
    MockTensor,
    next_power_of_2,
)
from . import impl
from .impl import (
    reinterpret,
    TensorWrapper,
    jit,
    builtin,
    extern,
    compile,
    CompilationError,
)
from .runtime import (
    autotune,
    Config,
    heuristics,
    JITFunction,
    KernelInterface,
)
from .compiler import compile, CompilationError
from . import language
from . import testing
from . import ops

__all__ = [
    "autotune",
    "builtin",
    "cdiv",
    "CompilationError",
    "compile",
    "Config",
    "extern",
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
