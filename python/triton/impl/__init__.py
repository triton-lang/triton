"""isort:skip_file"""
# Import order is significant here.

from triton._C.libtriton.triton import ir

from ..utils import (
    cdiv,
    next_power_of_2,
    MockTensor,
)
from .base import TensorWrapper, reinterpret

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

from .autotuner import (
    autotune,
    Config,
    heuristics,
    Heuristics,
)

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
    "MockTensor",
    "next_power_of_2",
    "OutOfResources",
    "reinterpret",
    "TensorWrapper",
]
