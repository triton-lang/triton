import math

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check

from .._core import builtin, _unwrap_if_constexpr
from .._layouts import DotOperandLayout
from ._layouts import AMDWMMALayout

_DOWNCAST_FORMAT_TO_ELEM_TYPE = {
    "e4m3": ttgl.float8e4nv,
    "e5m2": ttgl.float8e5,
    "e2m1": ttgl.uint8,
}


def _downcast_format_to_elem_type(format):
    format = _unwrap_if_constexpr(format)
    _check(
        format in _DOWNCAST_FORMAT_TO_ELEM_TYPE, lambda:
        f"Unsupported scaled_downcast format {format!r}; expected one of {_DOWNCAST_FORMAT_TO_ELEM_TYPE.keys()}")
    return _DOWNCAST_FORMAT_TO_ELEM_TYPE[format]


def _wrap_scaled_upcast_result(handle, elem_type, semantic):
    shape = semantic.builder.get_shape_from_tensor(handle)
    layout = semantic.builder.get_gluon_layout_from_tensor(handle)
    ret_ty = ttgl.distributed_type(elem_type, shape, layout)
    return ttgl.tensor(handle, ret_ty)


def _validate_scaled_upcast_fp4_args(src, scale, axis):
    _check(src.dtype in {ttgl.int8, ttgl.uint8},
           lambda: f"Expected packed fp4 input in int8/uint8, but got {src.dtype}")
    _check(axis is not None, lambda: "axis is required for packed fp4 scaled_upcast")

    rank = len(src.type.shape)
    _check(-rank <= axis < rank, lambda: f"axis {axis} out of range for rank {rank}")
    if axis < 0:
        axis += rank

    expected_shape = list(src.type.shape)
    expected_shape[axis] *= 2
    _check(
        scale.type.shape[:axis] + scale.type.shape[axis + 1:] == expected_shape[:axis] + expected_shape[axis + 1:],
        lambda: f"Expected scale shape for scaled_upcast to match output shape on non-axis dims: "
        f"{expected_shape}, but got {scale.type.shape}")
    _check(
        scale.type.shape[axis] > 0 and expected_shape[axis] % scale.type.shape[axis] == 0,
        lambda: f"Expected output axis extent {expected_shape[axis]} to be divisible by scale axis extent "
        f"{scale.type.shape[axis]}")
    _check(scale.dtype in {ttgl.int8, ttgl.uint8, ttgl.bfloat16},
           lambda: f"Unsupported scale dtype for fp4 scaled_upcast: {scale.dtype}")
    return axis


@ttgl.builtin
def get_scaled_upcast_fp4_scale_layout(src, scale, elem_type, axis, _semantic=None):
    """Return the scale layout required by an FP4 scaled_upcast.

    The caller must explicitly convert the scale tensor to the returned layout.
    ``scale`` is only inspected for its shape and layout; if it already has a
    compatible layout its own layout is returned so the convert is a no-op.
    Raises if the layout of ``src`` admits no valid scale layout for the
    requested ``scale`` shape.
    """
    _check(isinstance(src.type, ttgl.distributed_type),
           lambda: f"Expected src to have a distributed_type but got {src.type}")
    _check(isinstance(scale.type, ttgl.distributed_type),
           lambda: f"Expected scale to have a distributed_type but got {scale.type}")
    _check(elem_type in {ttgl.float16, ttgl.bfloat16},
           lambda: f"Expected elem_type to be fp16 or bf16 but got {elem_type}")
    axis = _validate_scaled_upcast_fp4_args(src, scale, axis)
    return _semantic.builder.get_scaled_upcast_fp4_scale_layout(
        src.handle,
        scale.handle,
        elem_type.to_ir(_semantic.builder),
        axis,
    )


def _infer_fp4_repacked_shape(src_shape, op_idx):
    result_shape = list(src_shape)
    k_dim = -1 if op_idx == 0 else -2
    packed_dim = -2 if op_idx == 0 else -1
    _check(src_shape[k_dim] % 2 == 0, lambda: f"Expected K dimension {src_shape[k_dim]} to be even")
    result_shape[k_dim] //= 2
    result_shape[packed_dim] *= 2
    return result_shape


def _load_shared_fp4_repacked(mem_desc, layout, semantic, parent_type):
    _check(isinstance(mem_desc, ttgl.shared_memory_descriptor),
           lambda: f"Expected mem_desc to be a shared_memory_descriptor but got {type(mem_desc)}")
    _check(isinstance(layout, DotOperandLayout), lambda: f"Expected layout to be a DotOperandLayout but got {layout}")
    _check(isinstance(layout.parent, parent_type),
           lambda: f"Expected layout parent to be an instance of {parent_type} but got {layout.parent}")
    _check(mem_desc.dtype in {ttgl.int8, ttgl.uint8},
           lambda: f"Expected packed fp4 input in int8/uint8 but got {mem_desc.dtype}")

    src_shape = list(mem_desc.shape)
    rank = len(src_shape)
    _check(rank in {2, 3}, lambda: f"Expected mem_desc rank to be 2 or 3 but got {rank}")
    _check(layout.operand_index in {0, 1},
           lambda: f"Expected operand_index to be 0 or 1 but got {layout.operand_index}")

    result_shape = _infer_fp4_repacked_shape(src_shape, layout.operand_index)
    ret_ty = ttgl.distributed_type(mem_desc.dtype, result_shape, layout)
    handle = semantic.builder.create_local_load_packed_transposed(ret_ty.to_ir(semantic.builder), mem_desc.handle)
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

    axis = _validate_scaled_upcast_fp4_args(src, scale, axis)

    handle = semantic.builder.create_scaled_upcast_fp4(src.handle, scale.handle, elem_type.to_ir(semantic.builder),
                                                       axis)
    return _wrap_scaled_upcast_result(handle, elem_type, semantic)


def _normalize_axis(axis, rank, axis_required_message):
    _check(axis is not None, lambda: axis_required_message)
    _check(-rank <= axis < rank, lambda: f"axis {axis} out of range for rank {rank}")
    if axis < 0:
        axis += rank
    return axis


def _validate_scaled_downcast_common(input, scale):
    _check(isinstance(input.type, ttgl.distributed_type),
           lambda: f"Expected input to have a distributed_type but got {input.type}")
    _check(isinstance(scale.type, ttgl.distributed_type),
           lambda: f"Expected scale to have a distributed_type but got {scale.type}")
    supported_dtypes = {ttgl.float16, ttgl.bfloat16, ttgl.float32}
    _check(input.dtype in supported_dtypes, lambda: f"Expected fp16, bf16, or fp32 input but got {input.dtype}")
    _check(scale.dtype in {ttgl.int8, ttgl.uint8},
           lambda: f"Expected raw E8M0 scale in int8/uint8 but got {scale.dtype}")
    return len(input.type.shape)


def _check_scaled_downcast_scale_shape(input_shape, scale_shape, axis, axis_extent, non_axis_error_message):
    _check(scale_shape[:axis] + scale_shape[axis + 1:] == input_shape[:axis] + input_shape[axis + 1:],
           lambda: f"{non_axis_error_message}, but got scale {scale_shape} and input {input_shape}")
    _check(scale_shape[axis] > 0 and axis_extent % scale_shape[axis] == 0,
           lambda: f"Expected axis extent {axis_extent} to be divisible by scale axis extent "
           f"{scale_shape[axis]}")
    _check(
        scale_shape[axis] < axis_extent, lambda: "scaled_downcast requires compact scales along the scaled axis; "
        "expanded scales (one scale per output element) are not supported")
    block_size = input_shape[axis] // scale_shape[axis]
    _check(
        block_size % 8 == 0, lambda: f"Expected each scale block to span a multiple of 8 consecutive input elements "
        f"along axis {axis}, but got {block_size}")


def _scaled_downcast(input, scale, elem_type, axis, semantic):
    rank = _validate_scaled_downcast_common(input, scale)
    input_shape = input.type.shape
    scale_shape = scale.type.shape

    # fp8: elementwise downcast; scale axis is based on input shape.
    if elem_type in {ttgl.float8e4nv, ttgl.float8e5}:
        axis = _normalize_axis(axis, rank, "axis is required for fp8 scaled_downcast")
        _check_scaled_downcast_scale_shape(
            input_shape, scale_shape, axis, input_shape[axis],
            "Expected scale shape for fp8 scaled_downcast to match the input shape on non-axis dimensions")

        handle = semantic.builder.create_scaled_downcast_fp8(input.handle, scale.handle,
                                                             elem_type.to_ir(semantic.builder), axis)
        ret_ty = input.type.with_element_ty(elem_type)
        return ttgl.tensor(handle, ret_ty)

    _check(elem_type in {ttgl.int8, ttgl.uint8},
           lambda: f"Expected elem_type to be fp8 (e4m3/e5m2) or packed fp4 (int8/uint8) but got {elem_type}")
    axis = _normalize_axis(axis, rank, "axis is required for packed fp4 scaled_downcast")
    _check(input_shape[axis] % 2 == 0, lambda: f"Expected even input axis extent but got {input_shape[axis]}")

    # fp4: packed output halves axis extent; scale axis uses packed output size.
    packed_output_axis_extent = input_shape[axis] // 2
    _check_scaled_downcast_scale_shape(
        input_shape, scale_shape, axis, packed_output_axis_extent,
        "Expected scale shape for packed fp4 scaled_downcast to match the packed output shape on non-axis dimensions")

    handle = semantic.builder.create_scaled_downcast_fp4(input.handle, scale.handle, axis)
    shape = semantic.builder.get_shape_from_tensor(handle)
    layout = semantic.builder.get_gluon_layout_from_tensor(handle)
    ret_ty = ttgl.distributed_type(elem_type, shape, layout)
    return ttgl.tensor(handle, ret_ty)


@builtin
def scaled_downcast(input, scale, format, axis=-1, _semantic=None):
    """
    Scale and convert FP16, BF16, or FP32 values to a low-precision MX format,
    dividing by the raw E8M0 ``scale`` payload (``int8`` or ``uint8``).

    ``format`` selects the target type and packing behavior:

    * ``"e4m3"`` / ``"e5m2"`` (fp8): elementwise, so the output keeps the
      shape and layout of ``input``.
    * ``"e2m1"`` (packed fp4): consecutive pairs of values along ``axis`` are
      packed into one output byte (even element -> low nibble, next element ->
      high nibble), so the result extent along ``axis`` is halved (the inverse
      of ``scaled_upcast``).

    ``axis`` (default: last dim) selects the dimension along which scales are
    shared. ``scale`` must be compact along ``axis`` with one E8M0 byte per
    block of consecutive input elements along ``axis``, and each block must
    span a multiple of 8 consecutive input elements.
    """
    axis = _unwrap_if_constexpr(axis)
    elem_type = _downcast_format_to_elem_type(format)
    return _scaled_downcast(input, scale, elem_type, axis, _semantic)
