from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check

from .._core import _unwrap_if_constexpr
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


def _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, scale_fn, semantic):
    """ Shared implementation for AMD WMMA scaled and MFMA scaled operation. """

    def _get_scale_shape(op_idx, operand, format):
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
        return scale_shape

    def _create_and_broadcast_default_scale(op_idx, scale, format):
        operand = a if op_idx == 0 else b

        scale_shape = _get_scale_shape(op_idx, operand, format)
        scale_layout = scale_fn(operand.type.layout, scale_shape, semantic)

        if isinstance(scale, ttgl.tensor) and scale.numel.value != 1:
            assert scale.type.shape == scale_shape, \
                f"Expect scale tensor to have shape {scale_shape}, but got {scale.type.shape}"
            return scale

        scale_value = _unwrap_if_constexpr(scale)
        scale_value = 0x7F if scale_value is None else scale_value
        return semantic.full(scale_shape, scale_value, ttgl.uint8, scale_layout)

    a_scale = _create_and_broadcast_default_scale(0, a_scale, a_format)
    b_scale = _create_and_broadcast_default_scale(1, b_scale, b_format)
    output = semantic.dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, fast_math=False, lhs_k_pack=True,
                                 rhs_k_pack=True, out_dtype=ttgl.float32)
    return ttgl.tensor(output.handle, acc.type)
