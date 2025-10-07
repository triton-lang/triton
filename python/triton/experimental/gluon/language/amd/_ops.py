from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check

from .._layouts import DotOperandLayout
from ._layouts import AMDWMMALayout


def _verify_wmma(version, a, b, acc):
    _check(acc is not None, lambda: "acc is required")

    layout = acc.type.layout
    _check(
        isinstance(layout, AMDWMMALayout) and layout.version == version,
        lambda: f"Expected layout to be an instance of AMDWMMALayout with version {version}")

    a_layout = a.type.layout
    _check(
        isinstance(a_layout, DotOperandLayout) and isinstance(a_layout.parent, AMDWMMALayout)
        and a_layout.parent.version == version,
        lambda: "Expected a's layout to be a DotOperandLayout with parent matching AMDWMMALayout")

    b_layout = b.type.layout
    _check(
        isinstance(b_layout, DotOperandLayout) and isinstance(b_layout.parent, AMDWMMALayout)
        and b_layout.parent.version == version,
        lambda: "Expected b's layout to be a DotOperandLayout with parent matching AMDWMMALayout")


def _wmma(version, a, b, acc, semantic):
    """ Shared implementation for AMD WMMA operations for Gluon builtins """
    _verify_wmma(version, a, b, acc)

    handle = semantic.dot(a, b, acc, input_precision=knobs.language.fp32_default, max_num_imprecise_acc=None,
                          out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)
