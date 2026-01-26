from triton.runtime.jit import constexpr_function
from triton._C.libtriton.gluon_ir import get_amd_wmma_scale_layout as _get_wmma_scale_layout

from ..._core import builtin
from .._ops import _wmma, _verify_wmma, _mma_scaled
from .._layouts import AMDWMMALayout
from ..cdna3 import buffer_load, buffer_store
from . import tdm
from . import async_copy
from . import mbarrier
from . import cluster

__all__ = [
    "async_copy", "tdm", "mbarrier", "cluster", "wmma", "wmma_scaled", "buffer_load", "buffer_store",
    "get_wmma_scale_layout"
]


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

    return _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, get_wmma_scale_layout, _semantic)


def _get_wmma_scale_layout_impl(*args, **kwargs):
    return _get_wmma_scale_layout(*args, **kwargs)


_get_wmma_scale_layout_impl.__triton_builtin__ = True


@constexpr_function
def get_wmma_scale_layout(dot_operand_layout, shape):
    """ Get the scale layout for WMMA scaled operands.

    Args:
        dot_operand_layout (DotOperandLayout): The dot operand layout.
        shape (List[int]): The shape of the scale tensor.

    Return:
        layout (DistributedLinearLayout): The scale layout.
    """
    op_idx = dot_operand_layout.operand_index
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDWMMALayout), "Expected parent to be an instance of AMDMFMALayout"
    mdim = parent.instr_shape[0]
    reg_bases = parent.reg_bases
    warp_bases = parent.warp_bases
    return _get_wmma_scale_layout_impl(op_idx, shape, mdim, reg_bases, warp_bases)
