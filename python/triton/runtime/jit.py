"""
Compat module for legacy code.
"""

from ..utils import version_key
from ..jitlib import jit, JITFunction, KernelInterface
from ..base import TensorWrapper, reinterpret

__all__ = [
    "jit",
    "JITFunction",
    "KernelInterface",
    "reinterpret",
    "TensorWrapper",
    "version_key",
]
