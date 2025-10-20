from triton.experimental.gluon.language import _core as ttgl
from ..._core import builtin, float32, _unwrap_if_constexpr
from ..._layouts import DotOperandLayout
from .._layouts import AMDMFMALayout
from ..cdna3 import _buffer_atomic_rmw_impl
from ..cdna3 import *  # NOQA: F403
from ..cdna3 import __all__ as __cdna3_all
from . import async_copy

__all__ = [*__cdna3_all, "async_copy", "mfma_scaled", "get_mfma_scale_layout"]


def _get_mfma_scale_layout(dot_operand_layout, shape, semantic):
    dot_operand_layout = _unwrap_if_constexpr(dot_operand_layout)
    shape = _unwrap_if_constexpr(shape)

    op_idx = dot_operand_layout.operand_index
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDMFMALayout), "Expected parent to be an instance of AMDMFMALayout"
    mdim = parent.instr_shape[0]
    tiles_per_warp = parent.tiles_per_warp
    warps_per_cta = parent.warps_per_cta
    return semantic.builder.get_amd_mfma_scale_layout(op_idx, shape, mdim, tiles_per_warp, warps_per_cta)


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
        a_scale (Optional[tensor]): Scale factor for operand A.
        a_format (str): Format of the operand A. Available formats: `e2m1`, `e4m3`, `e5m2`.
        b (tensor): The operand B to be multiplied.
        b_scale (Optional[tensor]): Scale factor for operand B. Available formats: `e2m1`, `e4m3`, `e5m2`.
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

    def _process_scale(op_idx, scale, format):
        operand = a if op_idx == 0 else b

        operand_shape = [s for s in operand.type.shape]
        scale_shape = operand_shape
        unpack_factor = 2 if format.value == "e2m1" else 1
        if op_idx == 0:
            k = scale_shape[-1] * unpack_factor
            scale_shape[-1] = k // 32
        else:
            k = scale_shape[-2] * unpack_factor
            scale_shape[-2] = k // 32
            scale_shape[-2], scale_shape[-1] = scale_shape[-1], scale_shape[-2]
        scale_layout = _get_mfma_scale_layout(operand.type.layout, scale_shape, _semantic)

        if isinstance(scale, ttgl.tensor) and scale.numel.value != 1:
            assert scale.type.shape == scale_shape, \
                f"Expect scale tensor to have shape {scale_shape}, but got {scale.type.shape}"
            return scale

        scale_value = _unwrap_if_constexpr(scale)
        scale_value = 0x7F if scale_value is None else scale_value
        return _semantic.full(scale_shape, scale_value, ttgl.uint8, scale_layout)

    a_scale = _process_scale(0, a_scale, a_format)
    b_scale = _process_scale(1, b_scale, b_format)
    output = _semantic.dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, False, True, True, float32)
    return ttgl.tensor(output.handle, acc.type)


@builtin
def get_mfma_scale_layout(dot_operand_layout, shape, _semantic=None):
    """ Get the scale layout for MFMA scaled operands.

    Args:
        dot_operand_layout (DotOperandLayout): The dot operand layout.
        shape (List[int]): The shape of the scale tensor.

    Returns:
        layout (DistributedLinearLayout): The scale layout.
    """
    return _get_mfma_scale_layout(dot_operand_layout, shape, _semantic)


"""
buffer_atomic_rmw of cnda4 shares the same signature and functionalities as cdna3.buffer_atomic_rmw.
The cdna4 version additionally supports `fadd` with `bf16`.
"""


@builtin
def buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):
    return _buffer_atomic_rmw_impl('max', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('min', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('add', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('and', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('or', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xor', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xchg(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xchg', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)
