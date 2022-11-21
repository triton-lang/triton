"""
Compat module for legacy code.
"""

from ..utils import version_key
from ..impl import jit, JITFunction, KernelInterface, TensorWrapper
from ..impl.base import reinterpret

__all__ = [
    "jit",
    "JITFunction",
    "KernelInterface",
    "reinterpret",
    "TensorWrapper",
    "version_key",
]
