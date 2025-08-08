from triton.experimental.gluon.language import _core as ttgl
from ..._core import builtin, float32, _unwrap_if_constexpr
from .._layouts import AMDMFMALayout
from ..cdna3 import buffer_load_to_shared, buffer_load, buffer_store

__all__ = ["buffer_load_to_shared", "buffer_load", "buffer_store", "mfma_scaled"]


@builtin
def mfma_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math=False, layout=None,
                _semantic=None):
    """
    AMD MFMA scaled operation, supported only on CDNA4 hardware. This is thin
    wrapper around the `tl.dot_scaled` operation, with enforced mfma layout.

    Args:
        lhs (tensor): Left-hand side tensor.
        lhs_scale (tensor): Scale tensor for the left-hand side.
        lhs_format (str): Format of the left-hand side tensor.
        rhs (tensor): Right-hand side tensor.
        rhs_scale (tensor): Scale tensor for the right-hand side.
        rhs_format (str): Format of the right-hand side tensor.
        acc (tensor): Accumulator tensor.
        fast_math (bool, optional): Enable fast math. Defaults to False.
        layout (ttgl.amd.AMDMFMALayout): Layout for the operation.
    """
    layout = _unwrap_if_constexpr(layout)
    assert isinstance(layout, AMDMFMALayout), "Expected layout to be an instance of AMDMFMALayout"

    tensor = _semantic.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math, True, True,
                                  float32)

    ret_ty = ttgl.distributed_type(tensor.dtype, tensor.shape, layout)
    return ttgl.tensor(tensor.handle, ret_ty)
