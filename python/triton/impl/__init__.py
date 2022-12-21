"""Triton internal implementation details.

Client libraries should not import interfaces from the `triton.impl` module;
as the details are subject to change.

APIs defined in the `triton.impl` module which are public will be re-exported
in other relevant `triton` module namespaces.
"""

from .base import builtin, extern, is_builtin
from triton._C.libtriton.triton import ir

__all__ = [
    "builtin",
    "extern",
    "ir",
    "is_builtin",
]
