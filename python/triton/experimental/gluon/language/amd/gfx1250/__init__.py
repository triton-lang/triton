from ..._core import builtin, _unwrap_if_constexpr
from .._ops import _wmma, _verify_wmma, _mma_scaled
from .._layouts import AMDWMMALayout
from ..cdna3 import buffer_load, buffer_store
from . import tdm

__all__ = ["tdm", "wmma", "wmma_scaled", "buffer_load", "buffer_store", "get_wmma_scale_layout"]


def _get_wmma_scale_layout(dot_operand_layout, shape, semantic):
    dot_operand_layout = _unwrap_if_constexpr(dot_operand_layout)
    shape = _unwrap_if_constexpr(shape)

    op_idx = dot_operand_layout.operand_index
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDWMMALayout), "Expected parent to be an instance of AMDMFMALayout"
    warps_per_cta = parent.warps_per_cta
    return semantic.builder.get_amd_wmma_scale_layout(op_idx, shape, warps_per_cta)


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
        a_scale (Optional[tensor]): Scale factor for operand A.
        a_format (str): Format of the operand A. Available formats: `e2m1`, `e4m3`, `e5m2`.
        b (tensor): The operand B to be multiplied.
        b_scale (Optional[tensor]): Scale factor for operand B.
        b_format (str): Format of the operand B. Available formats: `e2m1`, `e4m3`, `e5m2`.
        acc (tensor): Accumulator tensor.
    """
    _verify_wmma(3, a, b, acc)
    if a_format.value == "e2m1":
        wmma_layout = a.type.layout.parent
        assert isinstance(wmma_layout, AMDWMMALayout) and wmma_layout.instr_shape == [16, 16, 64], \
            "e2m1 format expects instr_shape to be [16, 16, 64]"
    if b_format.value == "e2m1":
        wmma_layout = b.type.layout.parent
        assert isinstance(wmma_layout, AMDWMMALayout) and wmma_layout.instr_shape == [16, 16, 64], \
            "e2m1 format expects instr_shape to be [16, 16, 64]"

    acc_layout = acc.type.layout
    assert isinstance(acc_layout, AMDWMMALayout) and acc_layout.instr_shape == [16, 16, 128], \
    "accumulator tensor's layout must be [16, 16, 128]"

    assert a_format.value in {"e2m1", "e4m3", "e5m2"}, f"Unsupported lhs_format: {a_format.value}"
    assert b_format.value in {"e2m1", "e4m3", "e5m2"}, f"Unsupported rhs_format: {b_format.value}"

    return _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _get_wmma_scale_layout, _semantic)


@builtin
def get_wmma_scale_layout(dot_operand_layout, shape, _semantic=None):
    """ Get the scale layout for WMMA scaled operands.

    Args:
        dot_operand_layout (DotOperandLayout): The dot operand layout.
        shape (List[int]): The shape of the scale tensor.

    Return:
        layout (DistributedLinearLayout): The scale layout.
    """
    return _get_wmma_scale_layout(dot_operand_layout, shape, _semantic)
