from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check

from .._layouts import AMDWMMALayout
from ..._core import builtin

__all__ = ["wmma"]


@builtin
def wmma(a, b, acc, _semantic=None):
    """
    Computes matrix-multiplication of a * b + acc using AMD WMMA instruction.

    Args:
        a (tensor): The operand a to be multiplied.
        b (tensor): The operand b to be multiplied.
        acc (tensor): The accumulator tensor.
    """
    _check(acc is not None, lambda: "acc is required")
    layout = acc.type.layout
    _check(
        isinstance(layout, AMDWMMALayout) and layout.version == 1,
        lambda: "Expected layout to be an instance of AMDWMMALayout with version 1")

    handle = _semantic.dot(a, b, acc, input_precision=knobs.language.fp32_default, max_num_imprecise_acc=None,
                           out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)
