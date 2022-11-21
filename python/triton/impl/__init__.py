"""isort:skip_file"""
# Import order is significant here.

from .. import ir

from ..utils import MockTensor

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
    "OutOfResources",
    "reinterpret",
    "TensorWrapper",
]
