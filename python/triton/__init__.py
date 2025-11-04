"""isort:skip_file"""
__version__ = '3.5.1'

# ---------------------------------------
# Note: import order is significant here.

# submodules
from .runtime import (
    autotune,
    Config,
    heuristics,
    JITFunction,
    KernelInterface,
    reinterpret,
    TensorWrapper,
    OutOfResources,
    InterpreterError,
    MockTensor,
)
from .runtime.jit import constexpr_function, jit
from .runtime._async_compile import AsyncCompileMode, FutureKernel
from .compiler import compile, CompilationError
from .errors import TritonError
from .runtime._allocation import set_allocator

from . import language
from . import testing
from . import tools

must_use_result = language.core.must_use_result

__all__ = [
    "AsyncCompileMode",
    "autotune",
    "cdiv",
    "CompilationError",
    "compile",
    "Config",
    "constexpr_function",
    "FutureKernel",
    "heuristics",
    "InterpreterError",
    "jit",
    "JITFunction",
    "KernelInterface",
    "language",
    "MockTensor",
    "must_use_result",
    "next_power_of_2",
    "OutOfResources",
    "reinterpret",
    "runtime",
    "set_allocator",
    "TensorWrapper",
    "TritonError",
    "testing",
    "tools",
]

# -------------------------------------
# misc. utilities that  don't fit well
# into any specific module
# -------------------------------------


@constexpr_function
def cdiv(x: int, y: int):
    return (x + y - 1) // y


@constexpr_function
def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n
