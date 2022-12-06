"""Triton internal implementation details.

Client libraries should not import interfaces from the `triton.impl` module;
as the details are subject to change.

APIs defined in the `triton.impl` module which are public will be re-exported
in other relevant `triton` module namespaces.
"""

from triton._C.libtriton.triton import ir
from . import base
from .base import (
    builtin,
    extern,
    is_builtin,
    reinterpret,
    TensorWrapper,
)
from .jitlib import (
    jit,
    JITFunction,
    KernelInterface,
)
from .compiler import (
    CompilationError,
    compile,
    CompiledKernel,
    OutOfResources,
)
from . import core

__all__ = [
    "base",
    "builtin",
    "CompilationError",
    "compile",
    "CompiledKernel",
    "core",
    "extern",
    "ir",
    "is_builtin",
    "jit",
    "OutOfResources",
]
