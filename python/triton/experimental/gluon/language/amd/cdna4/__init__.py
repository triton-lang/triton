from triton.experimental.gluon.language import _core as ttgl
from ..._core import builtin, float32
from ..._layouts import DotOperandLayout
from .._layouts import AMDMFMALayout
from ..cdna3 import *  # NOQA: F403
from ..cdna3 import __all__ as __cdna3_all
from . import async_copy

__all__ = [*__cdna3_all, "async_copy", "mfma_scaled"]


@builtin
def mfma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None):
    """
    AMD Scaled MFMA operation.

    ```
    c = a * a_scale @ b * b_scale + acc
    ```

    `a` and `b` use microscaling formats described in
    "OCP Microscaling Formats (MX) Specification":
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.
    Currently supported only on CDNA4 hardware.

    Args:
        a (tensor): The operand A to be multiplied.
        a_scale (tensor): Scale factor for operand A.
        a_format (str): Format of the operand A. Available formats: `e2m1`, `e4m3`, `e5m2`.
        b (tensor): The operand B to be multiplied.
        b_scale (tensor): Scale factor for operand B. Available formats: `e2m1`, `e4m3`, `e5m2`.
        b_format (str): Format of the operand B.
        acc (tensor): Accumulator tensor.
    """
    layout = acc.type.layout
    assert isinstance(layout, AMDMFMALayout), "Expected layout to be an instance of AMDMFMALayout"
    assert (isinstance(a.type.layout, DotOperandLayout) and a.type.layout.parent== layout), \
            "Expected lhs layout to be a DotOperandLayout with parent matching MFMA layout"
    assert (isinstance(b.type.layout, DotOperandLayout) and b.type.layout.parent == layout), \
            "Expected rhs layout to be a DotOperandLayout with parent matching MFMA layout"

    assert a_format.value in {"e2m1", "e4m3", "e5m2"}, f"Unsupported lhs_format: {a_format.value}"
    assert b_format.value in {"e2m1", "e4m3", "e5m2"}, f"Unsupported rhs_format: {b_format.value}"

    tensor = _semantic.dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, False, True, True, float32)

    ret_ty = ttgl.distributed_type(tensor.dtype, tensor.shape, layout)
    return ttgl.tensor(tensor.handle, ret_ty)
