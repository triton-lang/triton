"""isort:skip_file"""
__version__ = '3.2.0'

# ---------------------------------------
# Note: import order is significant here.

# submodules
from . import language, testing, tools
from .compiler import CompilationError, compile
from .errors import TritonError
from .runtime import (
    Config,
    InterpreterError,
    JITFunction,
    KernelInterface,
    MockTensor,
    OutOfResources,
    TensorWrapper,
    autotune,
    heuristics,
    reinterpret,
)
from .runtime._allocation import set_allocator
from .runtime.jit import jit

__all__ = [
    "autotune",
    "cdiv",
    "CompilationError",
    "compile",
    "Config",
    "heuristics",
    "impl",
    "InterpreterError",
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


def cdiv(x: int, y: int):
    return (x + y - 1) // y


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
