from ..._core import builtin
from .._ops import _wmma
from ..cdna5._cache_policy import CachePolicy
from . import buffer

__all__ = ["buffer", "CachePolicy", "wmma"]


@builtin
def wmma(a, b, acc, _semantic=None):
    """
    Computes matrix multiplication ``a * b + acc`` using an AMD WMMA instruction.

    Args:
        a (tensor): The operand a to be multiplied.
        b (tensor): The operand b to be multiplied.
        acc (tensor): The accumulator tensor.
    """
    return _wmma(2, a, b, acc, _semantic)
