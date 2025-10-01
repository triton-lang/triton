from ..._core import builtin
from .._ops import _wmma, _verify_wmma
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check
from ..._layouts import DotOperandLayout
from .._layouts import AMDWMMALayout
from . import tdm

__all__ = ["tdm", "wmma", "wmma_scaled"]


@builtin
def wmma(a, b, acc, _semantic=None):
    """
    Computes matrix-multiplication of a * b + acc using AMD WMMA instruction.

    Args:
        a (tensor): The operand a to be multiplied.
        b (tensor): The operand b to be multiplied.
        acc (tensor): The accumulator tensor.
    """
    return _wmma(3, a, b, acc, _semantic)


@builtin
def wmma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None):
    """
    AMD Scaled WMMA operation.

    ```
    c = a * a_scale @ b * b_scale + acc
    ```

    `a` and `b` use microscaling formats described in
    "OCP Microscaling Formats (MX) Specification":
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.

    Args:
        a (tensor): The operand A to be multiplied.
        a_scale (tensor): Scale factor for operand A.
        a_format (str): Format of the operand A. Available formats: `e2m1'.
        b (tensor): The operand B to be multiplied.
        b_scale (tensor): Scale factor for operand B.
        b_format (str): Format of the operand B. Available formats: `e2m1'.
        acc (tensor): Accumulator tensor.
    """
    _verify_wmma(3, a, b, acc)
    if a_format.value == "e2m1":
        wmma_layout = a.type.layout.parent
        assert isinstance(wmma_layout, AMDWMMALayout) and wmma_layout.instr_shape == (16, 16, 64), \
            "e2m1 format expects instr_shape to be (16, 16, 64)"
    if b_format.value == "e2m1":
        wmma_layout = b.type.layout.parent
        assert isinstance(wmma_layout, AMDWMMALayout) and wmma_layout.instr_shape == (16, 16, 64), \
            "e2m1 format expects instr_shape to be (16, 16, 64)"

    acc_layout = acc.type.layout
    assert isinstance(acc_layout, AMDWMMALayout) and acc_layout.instr_shape == (16, 16, 128), \
    "accumulator tensor's layout must be (16, 16, 128)"

    # TODO: Add more formats
    assert a_format.value in {"e2m1"}, f"Unsupported lhs_format: {a_format.value}"
    assert b_format.value in {"e2m1"}, f"Unsupported rhs_format: {b_format.value}"

    assert a_scale is not None and b_scale is not None, "Scales must not be None"

    handle = _semantic.dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, fast_math=False, lhs_k_pack=True,
                                  rhs_k_pack=True, out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)
