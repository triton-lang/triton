import math

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check

from .._core import _unwrap_if_constexpr
from .._layouts import DotOperandLayout
from ._layouts import AMDWMMALayout


def _wrap_scaled_upcast_result(handle, elem_type, semantic):
    shape = semantic.builder.get_shape_from_tensor(handle)
    layout = semantic.builder.get_gluon_layout_from_tensor(handle)
    ret_ty = ttgl.distributed_type(elem_type, shape, layout)
    return ttgl.tensor(handle, ret_ty)


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

    def _get_scale_shape(op_idx, operand, format, scale_factor):
        operand_shape = [s for s in operand.type.shape]
        scale_shape = operand_shape
        unpack_factor = 2 if format == "e2m1" else 1
        if op_idx == 0:
            k = scale_shape[-1] * unpack_factor
            scale_shape[-1] = k // scale_factor
        else:
            k = scale_shape[-2] * unpack_factor
            scale_shape[-2] = k // scale_factor
            scale_shape[-2], scale_shape[-1] = scale_shape[-1], scale_shape[-2]
        return scale_shape

    def _get_default_scale_dtype_and_unit_value(op_idx):
        default_value_by_dtype = {ttgl.uint8: 0x7F, ttgl.float8e4nv: 1.0}

        if a_scale is None and b_scale is None:
            return ttgl.uint8, 0x7F

        if a_format == b_format == "e2m1":
            # Fp4 x Fp4 requries to use the same scale dtype for both operands.
            other_scale = b_scale if op_idx == 0 else a_scale
            return other_scale.dtype, default_value_by_dtype[other_scale.dtype]

        return ttgl.uint8, 0x7F

    def _create_and_broadcast_default_scale(op_idx, scale, format, scale_factor):
        operand = a if op_idx == 0 else b

        scale_shape = _get_scale_shape(op_idx, operand, format, scale_factor)
        if isinstance(scale, ttgl.tensor) and scale.numel.value != 1:
            # In the case of scale pre-shuffling, the input shape is different from the default shape. We only check
            # the number of elements here.
            assert math.prod(scale_shape) == scale.numel.value, "Incompatible scale shape"
            return scale

        scale_layout = scale_fn(operand.type.layout, scale_shape, scale_factor)
        scale_value = _unwrap_if_constexpr(scale)
        if scale_value is None:
            scale_dtype, scale_value = _get_default_scale_dtype_and_unit_value(op_idx)
        elif isinstance(scale_value, int):
            scale_dtype = ttgl.uint8
        elif isinstance(scale_value, float):
            scale_dtype = ttgl.float8e4nv
        else:
            scale_dtype = scale.dtype

        return semantic.full(scale_shape, scale_value, scale_dtype, scale_layout)

    scale_factor = semantic.deduce_scale_factor(a, a_scale, a_format, True, b, b_scale, b_format, True)

    a_scale = _create_and_broadcast_default_scale(0, a_scale, a_format, scale_factor)
    b_scale = _create_and_broadcast_default_scale(1, b_scale, b_format, scale_factor)
    output = semantic.dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, fast_math=False, lhs_k_pack=True,
                                 rhs_k_pack=True, out_dtype=ttgl.float32)
    return ttgl.tensor(output.handle, acc.type)


def _scaled_upcast(src, scale, elem_type, axis, semantic):
    _check(isinstance(src.type, ttgl.distributed_type),
           lambda: f"Expected src to have a distributed_type but got {src.type}")
    _check(isinstance(scale.type, ttgl.distributed_type),
           lambda: f"Expected scale to have a distributed_type but got {scale.type}")
    _check(elem_type in {ttgl.float16, ttgl.bfloat16},
           lambda: f"Expected elem_type to be fp16 or bf16 but got {elem_type}")

    if src.dtype in {ttgl.float8e4nv, ttgl.float8e5}:
        _check(axis is None, lambda: "axis must be None for fp8 scaled_upcast")
        _check(scale.type.shape == src.type.shape,
               lambda: f"Expected scale shape for fp8 scaled_upcast to be {src.type.shape} but got {scale.type.shape}")
        _check(
            scale.type.layout == src.type.layout,
            lambda: f"Expected scale layout for fp8 scaled_upcast to be {src.type.layout} but got {scale.type.layout}")
        # Note: bf16 is allowed due to CDNA3/CDNA4 conversion before passing to scaled_upcast
        _check(scale.dtype in {ttgl.int8, ttgl.uint8, ttgl.bfloat16},
               lambda: f"Unsupported scale dtype for fp8 scaled_upcast: {scale.dtype}")
        ret_ty = scale.type.with_element_ty(elem_type)
        handle = semantic.builder.create_scaled_upcast_fp8(ret_ty.to_ir(semantic.builder), src.handle, scale.handle)
        return _wrap_scaled_upcast_result(handle, elem_type, semantic)

    _check(src.dtype in {ttgl.int8, ttgl.uint8},
           lambda: f"Expected packed fp4 input in int8/uint8 or fp8 input, but got {src.dtype}")
    _check(axis is not None, lambda: "axis is required for packed fp4 scaled_upcast")

    rank = len(src.type.shape)
    _check(-rank <= axis < rank, lambda: f"axis {axis} out of range for rank {rank}")
    if axis < 0:
        axis += rank

    expected_shape = list(src.type.shape)
    expected_shape[axis] *= 2
    _check(scale.type.shape == expected_shape,
           lambda: f"Expected scale shape for fp4 scaled_upcast to be {expected_shape} but got {scale.type.shape}")
    _check(scale.dtype in {ttgl.int8, ttgl.uint8, ttgl.bfloat16},
           lambda: f"Unsupported scale dtype for fp4 scaled_upcast: {scale.dtype}")

    handle = semantic.builder.create_scaled_upcast_fp4(src.handle, scale.handle, elem_type.to_ir(semantic.builder),
                                                       axis)
    return _wrap_scaled_upcast_result(handle, elem_type, semantic)
