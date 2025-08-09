from triton.experimental.gluon.language import _core as ttgl
from ..._core import builtin, float32
from ..._layouts import DotOperandLayout
from .._layouts import AMDMFMALayout
from ..cdna3 import buffer_load_to_shared, buffer_load, buffer_store

__all__ = ["buffer_load_to_shared", "buffer_load", "buffer_store", "mfma_scaled"]


@builtin
def mfma_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, _semantic=None):
    """
    AMD MFMA scaled operation, supported only on CDNA4 hardware. This is thin
    wrapper around the `tl.dot_scaled` operation, to propagate MFMA layout
    from the accumulator tensor to the result.

    Args:
        lhs (tensor): The operand A to be multiplied.
        lhs_scale (tensor): Scale factor for operand A.
        lhs_format (str): Format of the operand A. Available formats: `e2m1`, `e4m3`, `e5m2`.
        rhs (tensor): The operand B to be multiplied.
        rhs_scale (tensor): Scale factor for operand B. Available formats: `e2m1`, `e4m3`, `e5m2`.
        rhs_format (str): Format of the operand B.
        acc (tensor): Accumulator tensor.
    """
    layout = acc.type.layout
    assert isinstance(layout, AMDMFMALayout), "Expected layout to be an instance of AMDMFMALayout"
    assert (isinstance(lhs.type.layout, DotOperandLayout) and lhs.type.layout.parent== layout), \
            "Expected lhs layout to be a DotOperandLayout with parent matching MFMA layout"
    assert (isinstance(rhs.type.layout, DotOperandLayout) and rhs.type.layout.parent == layout), \
            "Expected rhs layout to be a DotOperandLayout with parent matching MFMA layout"
    assert lhs_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported lhs_format: {lhs_format}"
    assert rhs_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported rhs_format: {rhs_format}"

    tensor = _semantic.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, False, True, True,
                                  float32)

    ret_ty = ttgl.distributed_type(tensor.dtype, tensor.shape, layout)
    return ttgl.tensor(tensor.handle, ret_ty)
