from enum import Enum
import triton
import triton.language as tl
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#                      Dequantization / Quantization Utilities
# -----------------------------------------------------------------------------


def get_max_quant_val(dtype: torch.dtype):
    d = {torch.uint8: 6.0, torch.float8_e5m2: 57344.0, torch.float8_e4m3fn: 448.0}
    assert dtype in d
    return d[dtype]


@tl.constexpr_function
def get_scaled_dot_format_string(dtype: tl.dtype):
    mapping = {
        tl.float16: "fp16",
        tl.bfloat16: "bf16",
        tl.uint8: "e2m1",
        tl.float8e4nv: "e4m3",
        tl.float8e5: "e5m2",
    }
    return mapping[dtype]


@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


# fmt: off
@triton.jit
def _compute_quant_and_scale(src_tensor, valid_src_mask, mx_tensor_dtype: tl.constexpr,
                             DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0):
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    # Explicit cast to fp32 since most ops are not supported on bfloat16. We avoid needless conversions to and from bf16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # DequantScaleRoundingMode.ROUND_UP
        # compute 2 ** ceil(log2(dequant_scale))
        # Adding 0x007FFFFF adds exponent by 1 unless mantissa is all zeros
        # A corner case: exponent is 0xFF that will overflow but that's already
        # NaN so assume we don't care.
        dequant_scale_exponent = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    else:
        # DequantScaleRoundingMode.ROUND_DOWN
        # compute 2 ** floor(log2(dequant_scale))
        assert DEQUANT_SCALE_ROUNDING_MODE == 1
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0. This will ensure that any padding tensors are 0 in the mx format.
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    # First, we simply extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    # Now we must convert the tensors to the mx format.
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas = (quant_tensor & 0x7FFFFF)

        # 0.25 <= x < 0.75 maps to 0.5, a denormal number
        E8_BIAS = 127
        E2_BIAS = 1
        # Move implicit bit 1 at the beginning to mantissa for denormals
        adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
        mantissas = tl.where(exponents < E8_BIAS, (0x400000 | (mantissas >> 1)) >> adjusted_exponents, mantissas)

        # For normal numbers, we change the bias from 127 to 1, and for subnormals, we keep exponent as 0.
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((exponents << 2) | (mantissas >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent

@triton.jit
def _downcast_to_mxfp(mx_tensor_ptr, stride_mxt_outer, stride_mxt_quant: tl.constexpr,
                      mx_scale_ptr, stride_mx_scale_outer, stride_mx_scale_quant,
                      src_ptr, stride_src_outer, stride_src_quant,
                      outer_dim, quant_dim,
                      BLOCK_SIZE_OUT_DIM: tl.constexpr, BLOCK_SIZE_QUANT_DIM: tl.constexpr,
                      DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr):

    tl.static_assert(stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1.")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % 32 == 0, f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32")

    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(mx_tensor_dtype == tl.uint8 or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
                     f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.")

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, f"{mx_scale_ptr.dtype.element_ty=} must be uint8")
    tl.static_assert((src_dtype == tl.bfloat16) or (src_dtype == tl.float16), f"{src_dtype=} must be bfloat16 or float16")
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_mxt = mask_mxt_quant & mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, 32)
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    mx_scale_offsets = offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_quant_and_scale(src_tensor, full_mask_src, mx_tensor_dtype,
                                                        DEQUANT_SCALE_ROUNDING_MODE)

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


@triton.jit
def _upcast_from_mxfp(out_ptr, stride_o_outer, stride_o_quant: tl.constexpr,
                      mx_scale_ptr, stride_scale_outer, stride_scale_quant,
                      mx_tensor_ptr, stride_tensor_outer, stride_tensor_quant: tl.constexpr,
                      outer_dim, quant_dim,
                      BLOCK_SIZE_OUT_DIM: tl.constexpr, BLOCK_SIZE_QUANT_DIM: tl.constexpr):

    tl.static_assert(stride_o_quant == 1, "the weight must be contiguous in the k dimension for mx")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % 32 == 0, "BLOCK_SIZE_K must be a multiple of 32")
    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype: tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16)
    tl.static_assert(mx_tensor_dtype == tl.uint8 or
                     ((mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5) or
                       mx_tensor_dtype == dst_dtype),
                     "mx_tensor_ptr must be uint8 or float8 or dst_dtype")
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")

    # Determine if we are dealing with fp8 types.
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    # Compute starting indices for the quantized (packed) dimension and the outer dimension.
    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    mx_tensor_ptr += start_mxt_quant * stride_tensor_quant + start_out * stride_tensor_outer
    mx_scale_ptr += start_mx_scale_quant * stride_scale_quant + start_out * stride_scale_outer
    out_ptr += start_out * stride_o_outer + start_out_quant * stride_o_quant

    # Compute offsets and masks.
    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_out_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)

    mask_outer = start_out + offs_outer < outer_dim
    mask_out_quant = start_out_quant + offs_out_quant < quant_dim
    full_mask_out = mask_out_quant & mask_outer

    mask_src_quant = start_mxt_quant + offs_src_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_src = mask_src_quant & mask_outer

    mask_scale = start_mx_scale_quant + offs_scale < tl.cdiv(quant_dim, 32)
    full_scale_mask = mask_scale & mask_outer

    tensor_offsets = offs_src_quant * stride_tensor_quant + offs_outer * stride_tensor_outer
    scale_offsets = offs_scale * stride_scale_quant + offs_outer * stride_scale_outer
    out_offsets = offs_out_quant * stride_o_quant + offs_outer * stride_o_outer

    # Load the packed tensor and scale.
    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask_src)
    scale = tl.load(mx_scale_ptr + scale_offsets, mask=full_scale_mask)

    # Upcast the scale to the destination type.
    if dst_dtype == tl.bfloat16:
        dst_scale = (scale.to(tl.uint16) << 7).to(dst_dtype, bitcast=True)
    else:
        tl.static_assert(dst_dtype == tl.float16)
        dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        dst_scale = dst_scale.to(tl.float16)

    # Now upcast the tensor.
    if is_fp8:
        dst_tensor = tensor.to(dst_dtype)
        if tensor.dtype == tl.float8e5:
            from_e_bits: tl.constexpr = 5
            from_m_bits: tl.constexpr = 2
            to_e_bits: tl.constexpr = 8 if dst_dtype == tl.bfloat16 else 5
            to_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10

            # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
            non_finite_mask_src: tl.constexpr = ((1 << from_e_bits) - 1) << from_m_bits
            non_finite_mask_dst: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
            dst_tensor = tl.where(
                (tensor.to(tl.uint8, bitcast=True) & non_finite_mask_src) == non_finite_mask_src,
                (dst_tensor.to(tl.uint16, bitcast=True) | non_finite_mask_dst).to(dst_dtype, bitcast=True),
                dst_tensor,
            )
    else:
        assert is_fp4
        dst_bias: tl.constexpr = 127 if dst_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if dst_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10
        # e2m1
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((tensor & 0x08).to(tl.uint16) << 12)
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((tensor & 0x80).to(tl.uint16) << 8)
        # Three cases:
        # 1) x is normal and non-zero: Correct bias
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in the dst type
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        # 3) x is zero, do nothing
        dst_tensor = tl.interleave(x0, x1).to(dst_dtype, bitcast=True)

    # Reshape for proper broadcasting: the scale was stored with a 32‐sized “inner” grouping.
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    # Correct any NaNs encoded via the scale.
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)


class SwizzlingType(Enum):
    HOPPER = 0
    BLACKWELL = 1


class DequantScaleRoundingMode(Enum):
    ROUND_UP = 0
    ROUND_DOWN = 1


SWIZZLE_ALIGN_INNER = 8
SWIZZLE_SIZE_INNER = 4
SWIZZLE_SIZE_OUTER = 128

@triton.jit
def unswizzle_mx_scale_bw(x,
                          SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,
                          SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,
                          ALIGN_INNER: tl.constexpr = SWIZZLE_ALIGN_INNER):
    shape_0: tl.constexpr = x.shape[0]
    shape_1: tl.constexpr = x.shape[1]
    tl.static_assert(shape_1 % SIZE_OUTER == 0)
    tl.static_assert(shape_1 // SIZE_OUTER <= ALIGN_INNER)
    x = x.reshape(shape_0, (shape_1 // SIZE_OUTER) // SIZE_INNER, 32, SIZE_OUTER // 32, SIZE_INNER)
    x = x.trans(0, 3, 2, 1, 4).reshape(shape_0 * SIZE_OUTER, shape_1 // SIZE_OUTER)
    return x

def perm_to_contig(ndim: int, axis: int, swizzle_axis: int | None = None) -> tuple[int, ...]:
    """
    Permute the shape so that axis is the last dimension and swizzle_axis is the second to last dimension.
    """
    # FIXME(Lezcano): This API is not very good as it's too generic.
    # Chances are we just care about the cases
    # - axis=-2 and swizzle_axis=-1
    # - axis=-1 and swizzle_axis=-2
    # - axis=anything and swizzle_axis=None
    # We could probably just implement
    # perm_to_contig(ndim, transpose: bool)
    # where we transpose the last two dimensions if transpose is True and otherwise we leave them as is.
    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim

    assert axis != swizzle_axis
    shape = list(range(ndim))
    shape[axis], shape[-1] = shape[-1], shape[axis]
    if swizzle_axis is not None:
        if swizzle_axis == len(shape) - 1:
            swizzle_axis = axis
        shape[swizzle_axis], shape[-2] = shape[-2], shape[swizzle_axis]
    return tuple(shape)

def perm_from_contig(ndim: int, axis: int, swizzle_axis: int | None = None) -> tuple[int, ...]:
    # Invert the permutation via argsort
    perm = perm_to_contig(ndim, axis, swizzle_axis)
    inv = [0] * ndim
    for i, v in enumerate(perm):
        inv[v] = i
    return tuple(inv)

def perm_tuple_to_contig(shape: tuple[int, ...], axis: int, swizzle_axis: int | None = None) -> tuple[int, ...]:
    """
    Permute the shape so that axis is the last dimension and swizzle_axis is the second to last dimension.
    """
    perm = perm_to_contig(len(shape), axis, swizzle_axis)
    return tuple(shape[i] for i in perm)

def perm_tuple_from_contig(shape: tuple[int, ...], axis: int, swizzle_axis: int | None = None) -> tuple[int, ...]:
    """
    Permute the shape moving the last dimension to axis and the second to last dimension to swizzle_axis.
    """
    perm = perm_from_contig(len(shape), axis, swizzle_axis)
    return tuple(shape[i] for i in perm)

def perm_tensor_to_contig(x: torch.Tensor, axis: int, swizzle_axis: int | None = None) -> torch.Tensor:
    """
    Permute the tensor x moving axis to the last dimension and swizzle_axis to the second to last dimension.
    """
    return x.permute(perm_to_contig(x.ndim, axis, swizzle_axis))


def perm_tensor_from_contig(x: torch.Tensor, axis: int, swizzle_axis: int | None = None) -> torch.Tensor:
    """
    Permute the tensor x moving the last dimension to axis and the second to last dimension to swizzle_axis.
    """
    return x.permute(perm_from_contig(x.ndim, axis, swizzle_axis))

def downcast_to_mxfp_impl(src_tensor: torch.Tensor, out_quant_type: torch.dtype,
                          DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode,
                          BLOCK_OUT_DIM: int, BLOCK_QUANT_DIM: int,
                          *, out_quant_tensor: torch.Tensor | None = None, out_scale: torch.Tensor | None = None):
    """
    Downcast a contiguous tensor to mxfp along the last dimension.
    """
    is_fp4 = out_quant_type == torch.uint8
    is_fp8 = out_quant_type in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert is_fp4 or is_fp8
    divisor = 2 if is_fp4 else 1
    L = src_tensor.shape[-1]
    if is_fp4:
        assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"
    out_shape = src_tensor.shape[:-1] + (L // divisor,)
    out_scale_shape = src_tensor.shape[:-1] + (triton.cdiv(L, 32),)

    if out_quant_tensor is None:
        out_quant_tensor = src_tensor.new_empty(out_shape, dtype=out_quant_type)
    else:
        assert out_quant_tensor.shape == out_shape
    if out_scale is None:
        out_scale = src_tensor.new_empty(out_scale_shape, dtype=torch.uint8)
    else:
        assert out_scale.shape == out_scale_shape

    kernel_src_tensor = src_tensor.reshape(-1, src_tensor.shape[-1])
    kernel_quant_tensor = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
    kernel_scale = out_scale.view(-1, out_scale.shape[-1])

    blocks_out_dim = triton.cdiv(kernel_src_tensor.shape[0], BLOCK_OUT_DIM)
    blocks_quant_dim = triton.cdiv(kernel_src_tensor.shape[1], BLOCK_QUANT_DIM)

    _downcast_to_mxfp[(blocks_out_dim, blocks_quant_dim)](
        kernel_quant_tensor, *kernel_quant_tensor.stride(),
        kernel_scale, *kernel_scale.stride(),
        kernel_src_tensor, *kernel_src_tensor.stride(),
        *kernel_src_tensor.shape,
        BLOCK_OUT_DIM, BLOCK_QUANT_DIM, DEQUANT_SCALE_ROUNDING_MODE.value,
        num_warps=8
    )

    return out_quant_tensor, out_scale

def downcast_to_mxfp(src_tensor: torch.Tensor, out_quant_type: torch.dtype, axis: int, swizzle_axis: int | None = None,
                     swizzle_value: SwizzlingType | None = None, swizzle_scale: SwizzlingType | None = None,
                     out_quant_tensor: torch.Tensor | None = None, out_scale: torch.Tensor | None = None,
                     DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode=DequantScaleRoundingMode.ROUND_UP,
                     BLOCK_OUT_DIM: int = 128, BLOCK_QUANT_DIM: int = 32):
    """
         Convert the src weights to mx format. The src weight is quantized along the axis dimension.

         If weight_quant_type is torch.uint8, we output mxfp4 where two e2m1 values are packed into a single byte.
         Note that this means the k_dim of the tensor will be half of the logical k_dim.

         If weight_quant_type is torch.float8_e4m3fn or torch.float8_e5m2, we output mxfp8 with the float8s are stored
         in their respective formats.

         When swizzle_axis is provided, the downcast will quantize along the quantization axis and swizzle the scales
         with the swizzle_axis. See the relevant swizzle_* functions for more details.
    """
    # This should probably be packed into its own tiny class
    if swizzle_axis is None:
        assert swizzle_scale is None, "Swizzle scale must be None if swizzle axis is None"
        assert swizzle_value is None, "Swizzle value must be None if swizzle axis is None"
    else:
        assert swizzle_scale is not None or swizzle_value is not None, "At least one of swizzle_scale or swizzle_value must be provided"
        assert swizzle_value is None or swizzle_value == SwizzlingType.HOPPER, "Just implemented Hopper swizzle for now"

    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        assert -ndim <= swizzle_axis < ndim, f"Invalid swizzle axis {swizzle_axis=}"
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim
        assert axis != swizzle_axis, f"Axis and swizzle axis cannot be the same {axis=} {swizzle_axis=}"

    if out_quant_tensor is not None or out_scale is not None:
        assert swizzle_axis is None, "out= not supported together with swizzling"

    # Permute the tensor so that axis is the last dimension and swizzle_axis is the second to last dimension.
    src_tensor = perm_tensor_to_contig(src_tensor, axis, swizzle_axis)
    if out_quant_tensor is not None:
        out_quant_tensor = perm_tensor_to_contig(out_quant_tensor, axis, swizzle_axis)
    if out_scale is not None:
        out_scale = perm_tensor_to_contig(out_scale, axis, swizzle_axis)

    quant_tensor, scale = downcast_to_mxfp_impl(src_tensor, out_quant_type, DEQUANT_SCALE_ROUNDING_MODE, BLOCK_OUT_DIM, BLOCK_QUANT_DIM,
                                                out_quant_tensor=out_quant_tensor, out_scale=out_scale)

    # Swizzling
    if swizzle_value == SwizzlingType.HOPPER:
        quant_tensor = swizzle_mxfp4_value_hopper(quant_tensor, op_idx=0, mma_version=3)
    assert quant_tensor.is_contiguous()
    quant_tensor = perm_tensor_from_contig(quant_tensor, axis, swizzle_axis)

    orig_scale_shape = scale.shape
    if swizzle_scale == SwizzlingType.BLACKWELL:
        scale = swizzle_mx_scale_bw(scale, allow_pad=True)
    elif swizzle_scale == SwizzlingType.HOPPER:
        scale = swizzle_mxfp4_scale_hopper(scale, num_warps=8)
    assert scale.is_contiguous()
    scale = perm_tensor_from_contig(scale, axis, swizzle_axis)

    if out_scale is not None:
        out_scale = perm_tensor_from_contig(out_scale, axis, swizzle_axis)
        assert scale.data_ptr() == out_scale.data_ptr()
    if out_quant_tensor is not None:
        out_quant_tensor = perm_tensor_from_contig(out_quant_tensor, axis, swizzle_axis)
        assert quant_tensor.data_ptr() == out_quant_tensor.data_ptr()
    return quant_tensor, scale, perm_tuple_from_contig(orig_scale_shape, axis, swizzle_axis)


def upcast_from_mxfp(tensor: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, axis: int, swizzle_axis: int | None = None,
                     swizzle_value: SwizzlingType | None = None, swizzle_scale: SwizzlingType | None = None,
                     BLOCK_OUT_DIM: int = 128, BLOCK_QUANT_DIM: int = 32):
    """
    Upcasts an mxfp (packed) weight tensor back to float16 or bfloat16.

    The function assumes that the tensors were quantized along the given axis.
    It permutes the tensor so that the quantized axis is last, reshapes to 2D,
    launches the Triton upcast kernel, and then unpermutes back to the original order.
    """
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        assert -ndim <= swizzle_axis < ndim, f"Invalid swizzle axis {swizzle_axis=}"
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim
        assert axis != swizzle_axis, f"Axis and swizzle axis cannot be the same {axis=} {swizzle_axis=}"
        assert swizzle_scale is not None or swizzle_value is not None, "At least one of swizzle_scale or swizzle_value must be provided"
    else:
        assert swizzle_scale is None, "Swizzle scale must be None if swizzle axis is None"
        assert swizzle_value is None, "Swizzle value must be None if swizzle axis is None"

    assert tensor.ndim == scale.ndim, (f"Weight and scale must have the same number of dimensions. "
                                       f"Got {tensor.ndim=} and {scale.ndim=}")
    # dtype checks
    assert tensor.dtype in {torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn}, \
        f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == torch.uint8, f"Invalid scale dtype {scale.dtype=}"
    assert dtype in (torch.float16, torch.bfloat16), f"Invalid output dtype {dtype=}"

    # Bring the quantized axis to the end.
    # For the scales, bring the swizzle axis second to last.
    tensor = perm_tensor_to_contig(tensor, axis, swizzle_axis)
    assert tensor.is_contiguous()
    scale = perm_tensor_to_contig(scale, axis, swizzle_axis)
    assert scale.is_contiguous()

    # Unswizzle the value tensor
    if swizzle_value == SwizzlingType.HOPPER:
        tensor = unswizzle_mxfp4_value_hopper_torch(tensor, op_idx=0, mma_version=3)

    logical_quant_dim = tensor.shape[-1] * (2 if tensor.dtype == torch.uint8 else 1)

    # Unswizzle the scale tensor
    if swizzle_scale == SwizzlingType.BLACKWELL:
        scale = unswizzle_mx_scale_bw_torch(scale)

        # Peel off padding
        unpadded_scale_shape = (*tensor.shape[:-1], triton.cdiv(logical_quant_dim, 32))
        assert triton.cdiv(unpadded_scale_shape[-1], SWIZZLE_ALIGN_INNER) * SWIZZLE_ALIGN_INNER == scale.shape[-1], \
            f"Scale shape mismatch. Got {scale.shape[axis]=} and {triton.cdiv(unpadded_scale_shape[-1], SWIZZLE_ALIGN_INNER) * SWIZZLE_ALIGN_INNER=}"

        slices = tuple(slice(0, size) for size in unpadded_scale_shape)
        scale = scale[slices].contiguous()
    elif swizzle_scale == SwizzlingType.HOPPER:
        scale = unswizzle_mxfp4_scale_hopper_torch(scale, num_warps=8)

    assert scale.is_contiguous()

    assert tensor.is_contiguous()

    out = torch.empty((*tensor.shape[:-1], logical_quant_dim), dtype=dtype, device=tensor.device)

    reshaped_out = out.view(-1, out.shape[-1])
    reshaped_tensor = tensor.view(-1, tensor.shape[-1])
    reshaped_scale = scale.view(-1, scale.shape[-1])

    blocks_out_dim = triton.cdiv(reshaped_out.shape[0], BLOCK_OUT_DIM)
    blocks_quant_dim = triton.cdiv(reshaped_out.shape[1], BLOCK_QUANT_DIM)

    _upcast_from_mxfp[(blocks_out_dim, blocks_quant_dim)](
        reshaped_out, *reshaped_out.stride(),
        reshaped_scale, *reshaped_scale.stride(),
        reshaped_tensor, *reshaped_tensor.stride(),
        *reshaped_out.shape, BLOCK_OUT_DIM, BLOCK_QUANT_DIM, num_warps=8
    )

    return perm_tensor_from_contig(out, axis, swizzle_axis)


def right_shift_unsigned(x, shift):
    # CUDA torch does not support bit ops on uint32, so we need to mask to get unsigned right shift
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def downcast_to_mxfp_torch(src_tensor: torch.Tensor, out_quant_type: torch.dtype, axis: int, swizzle_axis: int | None = None,
                           swizzle_value: SwizzlingType | None = None, swizzle_scale: SwizzlingType | None = None,
                           out_quant_tensor: torch.Tensor | None = None, out_scale: torch.Tensor | None = None,
                           DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP):
    """
    Converts the src tensor to the output format specified by out_quant_type.
      axis: The axis along which the tensors are contiguous and quantization is applied.
      DEQUANT_SCALE_ROUNDING_MODE: 0 for ROUND_UP, 1 for ROUND_DOWN.

    Returns:
      out_quant_tensor: Quantized tensor in mx format.
         • For mxfp8, the output has the same shape as src_tensor.
         • For mxfp4, the size along the axis is halved, and the tensor is returned as a torch.uint8.
      scale: Scale tensor (stored as uint8) computed per group of 32 elements along the axis.
             Its shape is the same as src_tensor except that the axis is replaced by ceil(L/32),
             where L is the original length along that axis.
    """
    # This should probably be packed into its own tiny class
    if swizzle_axis is None:
        assert swizzle_scale is None, "Swizzle scale must be None if swizzle axis is None"
        assert swizzle_value is None, "Swizzle value must be None if swizzle axis is None"
    else:
        assert swizzle_scale is not None or swizzle_value is not None, "At least one of swizzle_scale or swizzle_value must be provided"
        assert swizzle_value is None or swizzle_value == SwizzlingType.HOPPER, "Just implemented Hopper swizzle for now"


    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    assert src_tensor.dtype in {torch.float32, torch.bfloat16, torch.float16}, f"Invalid input tensor dtype {src_tensor.dtype}"

    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        assert -ndim <= swizzle_axis < ndim, f"Invalid swizzle axis {swizzle_axis=}"
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim
        assert axis != swizzle_axis, f"Axis and swizzle axis cannot be the same {axis=} {swizzle_axis=}"
    is_fp4 = out_quant_type == torch.uint8
    is_fp8 = "float8" in str(out_quant_type)
    assert is_fp4 or is_fp8, f"Invalid input tensor dtype {out_quant_type}"

    device = src_tensor.device

    # For mxfp4 conversion, we assume the contiguous axis length is even.
    if is_fp4:
        axis_shape = src_tensor.size(axis)
        assert axis_shape % 2 == 0, "For mxfp4 conversion the contiguous axis length must be even."

    # Permute the tensor so that the contiguous axis becomes the last dimension.
    # For the scales, make the swizzle axis is second to last.
    src = perm_tensor_to_contig(src_tensor, axis, swizzle_axis).to(torch.float32)  # now shape: (..., axis_shape)
    axis_shape = src.shape[-1]

    # Pad the axis to be divisible by 32, in case it is not.
    next_multiple = (axis_shape + 31) // 32 * 32
    pad_amount = next_multiple - axis_shape
    padded_src = F.pad(src, (0, pad_amount))
    valid_mask = F.pad(torch.ones_like(src, dtype=torch.bool), (0, pad_amount))
    padded_axis_shape = padded_src.size(-1)  # now divisible by 32

    # --- Compute per-group maximums for scale ---
    # Set padded entries to -1 so they don’t affect the max.
    abs_f = torch.abs(padded_src)
    abs_f = torch.where(valid_mask, abs_f, torch.tensor(-1.0, device=device, dtype=padded_src.dtype))
    # Reshape the last dimension into groups of 32.
    new_shape = padded_src.shape[:-1] + (padded_axis_shape // 32, 32)
    abs_groups = abs_f.view(*new_shape)
    # Compute maximum along the group dimension (of size 32).
    max_val, _ = abs_groups.max(dim=-1, keepdim=True)

    # Choose a max quantization value depending on type.
    max_quant_val = get_max_quant_val(out_quant_type)
    dequant_scale = max_val / max_quant_val  # shape: (..., padded_axis_shape//32, 1)

    # Convert to int to round the FP32 scale, prior to quantization!
    ds_int = dequant_scale.view(torch.int32)
    if DEQUANT_SCALE_ROUNDING_MODE == DequantScaleRoundingMode.ROUND_UP:
        ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
    else:
        ds_int_rounded = ds_int & 0x7F800000
    # Reinterpret back as float32.
    dequant_scale_rounded = ds_int_rounded.view(torch.float32)

    # Compute the quantization scale.
    quant_scale = torch.where(dequant_scale_rounded == 0,
                              torch.tensor(0.0, device=device),
                              1.0 / dequant_scale_rounded)

    # Quantize the tensor
    orig_padded_shape = padded_src.shape
    padded_src_groups = padded_src.view(*new_shape)
    quant_tensor = padded_src_groups * quant_scale
    # Reshape back to the original shape and trim padding
    quant_tensor = quant_tensor.view(orig_padded_shape)
    quant_tensor = quant_tensor[..., :axis_shape]

    # Finally, convert the quantized tensor to the target format
    if is_fp8:
        # Conversion must use satfinite PTX, so clamp before the conversion in torch to emulate this behavior
        quant_tensor = torch.clamp(quant_tensor, -max_quant_val, max_quant_val)
        out_weight = quant_tensor.to(out_quant_type)
    else:
        assert is_fp4, f"Invalid output quantization type {out_quant_type}"
        # For mxfp4, perform bit-level manipulation and pack two 4-bit values per uint8.
        # First, reinterpret the quantized tensor bits.
        q_int = quant_tensor.contiguous().view(torch.int32)
        # Extract sign, exponent, and mantissa.
        signs = q_int & 0x80000000
        exponents = right_shift_unsigned(q_int, 23) & 0xFF
        mantissas = q_int & 0x7FFFFF

        E8_BIAS = 127
        E2_BIAS = 1
        # Adjust mantissas for subnormals.
        mantissas = torch.where(exponents < E8_BIAS,
                                (0x400000 | right_shift_unsigned(mantissas, 1)) >> (E8_BIAS - exponents - 1),
                                mantissas)
        exponents = torch.maximum(exponents,
                                  torch.tensor(E8_BIAS - E2_BIAS, device=device)) - (E8_BIAS - E2_BIAS)
        e2m1_tmp = right_shift_unsigned(((exponents << 2) | right_shift_unsigned(mantissas, 21)) + 1, 1)
        e2m1_tmp = torch.minimum(e2m1_tmp, torch.tensor(0x7, device=device))
        e2m1_value = (right_shift_unsigned(signs, 28) | e2m1_tmp).to(torch.uint8)  # shape: (..., even_axis_shape)

        # Pack pairs of 4-bit values along the last dimension.
        e2m1_value = e2m1_value.view(*e2m1_value.shape[:-1], axis_shape // 2, 2)
        evens = e2m1_value[..., 0]
        odds = e2m1_value[..., 1]
        out_weight = evens | (odds << 4)  # shape: (..., axis_shape//2)

    # --- Process and output the scale ---
    dq_scale = (ds_int_rounded.view(*dequant_scale.shape) >> 23).to(torch.uint8)  # shape: (..., axis_shape//32, 1)
    dq_scale = dq_scale.squeeze(-1)

    if swizzle_scale == SwizzlingType.BLACKWELL:
        dq_scale = swizzle_mx_scale_bw(dq_scale)
    elif swizzle_scale == SwizzlingType.HOPPER:
        dq_scale = swizzle_mxfp4_scale_hopper(dq_scale, num_warps=8)
    if swizzle_value == SwizzlingType.HOPPER:
        out_weight = swizzle_mxfp4_value_hopper(out_weight, op_idx=0, mma_version=3)

    # Now, invert the permutation so that the contiguous axis returns to its original position.
    out_weight = perm_tensor_from_contig(out_weight, axis, swizzle_axis)
    dq_scale = perm_tensor_from_contig(dq_scale, axis, swizzle_axis)

    if out_quant_tensor is not None:
        assert out_quant_tensor.shape == out_weight.shape, f"Invalid shape {out_quant_tensor.shape} != {out_weight.shape}"
        assert out_quant_tensor.dtype == out_weight.dtype, f"Invalid dtype {out_quant_tensor.dtype} != {out_weight.dtype}"
        out_quant_tensor.copy_(out_weight)
    else:
        out_quant_tensor = out_weight

    if out_scale is not None:
        assert out_scale.shape == dq_scale.shape, f"Invalid shape {out_scale.shape} != {dq_scale.shape}"
        assert out_scale.dtype == dq_scale.dtype, f"Invalid dtype {out_scale.dtype} != {dq_scale.dtype}"
        out_scale.copy_(dq_scale)
    else:
        out_scale = dq_scale

    return out_quant_tensor, out_scale


def cvt_e2m1_to_fp32(input_tensor):
    assert input_tensor.dtype == torch.uint8

    input_tensor = input_tensor.to(torch.int32)
    evens = input_tensor & 0xF
    odds = (input_tensor >> 4) & 0xF

    vals = [0.0, 0.5, 1, 1.5, 2, 3, 4, 6]
    outputs = torch.tensor(vals, dtype=torch.float32, device=input_tensor.device)
    outputs = torch.cat([outputs, -outputs])

    even_floats = outputs[evens]
    odd_floats = outputs[odds]
    output_tensor = torch.stack([even_floats, odd_floats], dim=-1)
    output_tensor = output_tensor.view(*input_tensor.shape[:-1], -1)
    return output_tensor


def upcast_from_mxfp_torch(tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype, axis: int, swizzle_axis: int | None = None,
                           swizzle_value: SwizzlingType | None = None, swizzle_scale: SwizzlingType | None = None):
    """
    Converts the mxfp4/mxfp8 tensor to the target format specified by target_dtype.
      axis: The axis along which dequantization is applied.

    Returns:
      out_weight: Tensor in the target format.
    """

    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    is_fp8 = tensor.dtype == torch.float8_e4m3fn or tensor.dtype == torch.float8_e5m2
    assert is_fp8 or tensor.dtype == torch.uint8, f"Invalid input quantization type {tensor.dtype}"

    # Permute the tensor and scale so that the quantization axis becomes the last dimension
    # For the scales, also permute so the swizzle axis is second to last.
    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        assert -ndim <= swizzle_axis < ndim, f"Invalid swizzle axis {swizzle_axis=}"
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim
        assert axis != swizzle_axis, f"Axis and swizzle axis cannot be the same {axis=} {swizzle_axis=}"
        assert swizzle_scale is not None or swizzle_value is not None, "At least one of swizzle_scale or swizzle_value must be provided"
        assert swizzle_value is None or swizzle_value == SwizzlingType.HOPPER, "Just implemented Hopper swizzle for now"
    else:
        assert swizzle_scale is None, "Swizzle scale must be None if swizzle axis is None"
        assert swizzle_value is None, "Swizzle value must be None if swizzle axis is None"

    tensor = perm_tensor_to_contig(tensor, axis, swizzle_axis)
    scale = perm_tensor_to_contig(scale, axis, swizzle_axis)

    if swizzle_value == SwizzlingType.HOPPER:
        tensor = unswizzle_mxfp4_value_hopper_torch(tensor, op_idx=0, mma_version=3)

    if swizzle_scale == SwizzlingType.BLACKWELL:
        scale = unswizzle_mx_scale_bw_torch(scale)
    elif swizzle_scale == SwizzlingType.HOPPER:
        scale = unswizzle_mxfp4_scale_hopper_torch(scale, num_warps=8)

    dq_scale = (scale.to(torch.int32) << 23).view(torch.float32)  # Shift to the exponent and bitcast to fp32

    if tensor.dtype == torch.uint8:
        fp32_tensor = cvt_e2m1_to_fp32(tensor)
    else:
        fp32_tensor = tensor.to(torch.float32)

    # Trim padding
    dq_scale = dq_scale[..., :fp32_tensor.shape[-2], :(fp32_tensor.shape[-1] + 31) // 32]

    axis_shape = fp32_tensor.size(-1)
    padded_axis_shape = dq_scale.size(-1) * 32
    pad_size = padded_axis_shape - axis_shape
    padded_tensor = F.pad(fp32_tensor, (0, pad_size))

    new_axis_shape = padded_tensor.shape[-1]
    new_shape = padded_tensor.shape[:-1] + (new_axis_shape // 32, 32)
    padded_tensor = padded_tensor.view(*new_shape)
    dq_scale_padded = dq_scale.unsqueeze(-1)  # shape: [..., ceil(axis_shape/32), 1]
    out_padded = padded_tensor * dq_scale_padded

    # Flatten back and remove the padded tail
    out_padded = out_padded.view(*fp32_tensor.shape[:-1], new_axis_shape)
    out_tensor = out_padded[..., :axis_shape]

    out_tensor = out_tensor.to(target_dtype).contiguous()
    out_tensor = perm_tensor_from_contig(out_tensor, axis, swizzle_axis)

    return out_tensor


def swizzle_mx_scale_bw(tensor: torch.Tensor, allow_pad=True):
    """
    Swizzle the input tensor of shape (A, B, ... N, K) to (A, B, ... N // 128, K // 4, 32, 4, 4).
    Padding is applied if N and K are not multiples of 128 and 4 respectively.
    Returns the swizzled tensor repacked as (A, B, ... N, K), with padding.
    """
    *leading_shape, N, K, = tensor.shape
    pad_k = (SWIZZLE_ALIGN_INNER - (K % SWIZZLE_ALIGN_INNER)) % SWIZZLE_ALIGN_INNER
    pad_n = (SWIZZLE_SIZE_OUTER - (N % SWIZZLE_SIZE_OUTER)) % SWIZZLE_SIZE_OUTER
    if pad_k or pad_n > 0:
        assert allow_pad, "Padding is required for swizzling, but it was explicitly disabled."
        tensor = torch.nn.functional.pad(tensor, (0, pad_k, 0, pad_n))
    padded_shape = tensor.shape
    tensor = tensor.reshape(*leading_shape, padded_shape[-2] // SWIZZLE_SIZE_OUTER, SWIZZLE_SIZE_OUTER // 32, 32, padded_shape[-1] // SWIZZLE_SIZE_INNER, SWIZZLE_SIZE_INNER)
    permute_order = list(range(len(tensor.shape)))
    permute_order[-2], permute_order[-4] = permute_order[-4], permute_order[-2]
    return tensor.permute(permute_order).reshape(*padded_shape)


def unswizzle_mx_scale_bw_torch(tensor: torch.Tensor):
    """
    Unswizzle the input tensor of shape (A, B, ... N // 128, K // 4, 32, 4, 4) packed as (A, B, ... N, K). (Testing only)
    """
    assert tensor.shape[-1] % SWIZZLE_SIZE_INNER == 0, f"{tensor.shape[-1]=} must be a multiple of {SWIZZLE_SIZE_INNER}"
    assert tensor.shape[-2] % SWIZZLE_SIZE_OUTER == 0, f"{tensor.shape[-2]=} must be a multiple of {SWIZZLE_SIZE_OUTER}"
    *leading_shape, N, K, = tensor.shape
    tensor = tensor.reshape(*leading_shape, N // SWIZZLE_SIZE_OUTER, K // SWIZZLE_SIZE_INNER, 32, SWIZZLE_SIZE_OUTER // 32, SWIZZLE_SIZE_INNER)
    permute_order = list(range(len(tensor.shape)))
    permute_order[-2], permute_order[-4] = permute_order[-4], permute_order[-2]
    return tensor.permute(permute_order).reshape(*leading_shape, N, K)

def swizzle_mxfp4_value_hopper(x: torch.Tensor, op_idx: int, mma_version: int):
    """
    Given a uint8 tensor of shape (*, M, K), returns a tensor of shape
    (*, M // 4, K * 4) such that:

    1) Groups contiguously all the elements owned by the same thread of 4
    mma tiles along the K axis. The following animation shows a similar
    grouping for 2 tiles along M and 2 tiles along K rather than 4 along K
    as done here:
    https://neuralmagic.com/wp-content/uploads/2024/10/animation_4.gif

    2) Moves the elements belonging to thread 4-7 to be contiguous with those
    from thread 0-3. This is done to get a full cache line when loading them
    from HBM.

    op_idx selects the lhs or rhs of the matmul.

    WARNING: Assumes that the matmul will be done in bf16 or fp16!
    Implementing it for fp8 is as easy as making the tile size (8, 8)
    """
    assert x.dtype == torch.uint8
    assert op_idx in (0, 1)
    batch = x.ndim - 2
    assert batch >= 0
    assert mma_version in (2, 3)

    if op_idx == 1:
        x = x.mT
    init_shape = x.shape

    # We are loading 8 bf16 elements per thread to use ld.global.v4
    # Every u8 represents 2 mxfp4 elements
    u8_kwidth = 8 // 2 if mma_version == 2 else 1

    # Pack the 4 // u8_kwidth subtiles of an mma into a u4x8
    contig = (1, u8_kwidth)
    scott_trick = (2, 1)
    threads = (4, 4)
    warp_tile = (2, 2)
    k_tile = (1, 4 // u8_kwidth)

    sizes = list(x.shape[:-2])
    # [rest, K, tile, threads] per dimension
    for i, (a, b, c, s, d) in enumerate(zip(k_tile, warp_tile, threads, scott_trick, contig)):
        pack = a * b * c * s * d
        assert x.shape[batch + i] % pack == 0, (
            f"Shape should be divisible by {pack}. Got {x.shape[batch + i]}"
        )
        sizes.append(x.shape[batch + i] // pack)
        sizes += [a, b, c, s, d]

    # 0: rest[0]
    # 1: k_tile[0]
    # 2: warp_tile[0]
    # 3: threads[0]
    # 4: scott_trick[0]
    # 5: contig[0]
    # 6: rest[1]
    # 7: k_tile[1]
    # 8: warp_tile[1]
    # 9: threads[1]
    # 10: scott_trick[1]
    # 11: contig[1]

    x = x.view(*sizes)
    # Want [rest[0], threads[0], rest[1], scott_trick[0], scott_trick[0], threads[1], contig[1], contig[0], k_tile[1], k_tile[0], warp_tile[1], warp_tile[0]]
    perm = [0, 3,
            6, 10, 4, 9, 7, 1, 8, 2, 5, 11]
    perm = list(range(batch)) + [batch + p for p in perm]
    x = x.permute(*perm)
    x = x.contiguous()
    # These are views
    x = x.flatten(-10, -1)
    x = x.flatten(-3, -2)
    assert x.is_contiguous()
    assert x.shape[-2] == init_shape[-2] // 4
    assert x.shape[-1] == init_shape[-1] * 4

    if op_idx == 1:
        x = x.mT

    return x

def swizzle_mxfp4_scale_hopper(x: torch.Tensor, num_warps: int):
    """
    Make the 64x2 tile of scales of a 64x64 tile of mxfp4 values contiguous.
    """
    *batch, M, K = x.shape
    assert x.is_contiguous()
    assert num_warps & (num_warps - 1) == 0, "warps_n must be a power of 2"
    assert M % (2 * num_warps * 2 * 8) == 0 and K % 2 == 0, f"Input tensor must have a subtile of shape (..., {2 * num_warps * 2 * 8}, 2)"
    b = len(batch)
    x = x.reshape(*batch, M // (2 * num_warps * 2 * 8), 2, num_warps, 2, 8, K // 2, 2)
    perm = [0, 2, 5, 1, 4, 6, 3]
    perm = list(range(b)) + [b + p for p in perm]
    x = x.permute(*perm)
    x = x.flatten(-5, -1)
    x = x.flatten(-3, -2)
    assert x.shape[-2] == M // 32
    assert x.shape[-1] == K * 32
    return x

@triton.jit
def unswizzle_mxfp4_scale_hopper(x, num_warps: tl.constexpr):
    """
    Triton inverse of swizzle_mxfp4_scale_hopper
    """
    tl.static_assert(len(x.shape) == 2, "NYI")
    M: tl.constexpr = x.shape[0]
    K: tl.constexpr = x.shape[1]
    tl.static_assert(M % num_warps == 0, f"M must be divisible by {num_warps}. Got {M}")
    tl.static_assert(K % 64 == 0, f"K must be divisible by 64. Got {K}")
    x = x.reshape(M // num_warps, num_warps, K // 64, 2, 8, 2, 2)
    x = x.trans(0, 3, 1, 6, 4, 2, 5)
    x = x.reshape(M * 32, K // 32)
    return x

def unswizzle_mxfp4_scale_hopper_torch(x: torch.Tensor, num_warps: int):
    """
    PyTorch inverse of unswizzle_mxfp4_scale_hopper
    """
    assert num_warps & (num_warps - 1) == 0, "num_warps must be a power of 2"
    *batch, M, K = x.shape
    b = len(batch)
    x = x.reshape(*batch, M // num_warps, num_warps, K // 64, 2, 8, 2, 2)
    perm = [0, 3, 1, 6, 4, 2, 5]
    perm = list(range(b)) + [b + p for p in perm]
    x = x.permute(*perm)
    x = x.reshape(*batch, M * 32, K // 32)
    return x

@triton.jit
def unswizzle_mxfp4_value_hopper(x, op_idx: tl.constexpr, mma_version: tl.constexpr):
    """
    Triton inverse of swizzle_mxfp4_value_hopper
    """
    tl.static_assert(op_idx == 0 or op_idx == 1, "op_idx must be 0 or 1")
    tl.static_assert(len(x.shape) == 2, "NYI")
    tl.static_assert(mma_version == 2 or mma_version == 3, "mma_version must be 2 or 3")
    if op_idx == 1:
        x = x.trans()

    # We have two times the elements if we already upcasted to bfloat16
    mult: tl.constexpr = 2 if x.dtype == tl.bfloat16 else 1
    M: tl.constexpr = x.shape[0]
    K: tl.constexpr = x.shape[1]
    tl.static_assert(M % 4 == 0, "M must be divisible by 4")
    tl.static_assert(K % (4 * 8 * 2 * 2 * mult) == 0, f"K must be divisible by {4 * 8 * 2 * 2 * mult}")

    # We are loading 8 bf16 elements per thread to use ld.global.v4
    # Every u8 represents 2 mxfp4 elements
    u8_kwidth: tl.constexpr = 8 // 2 if mma_version == 2 else 1
    x = x.reshape(M // 4, 4, K // (4 * 8 * 2 * 2 * mult), 2, 4, 8 // u8_kwidth, 2, u8_kwidth * mult)
    x = x.trans(0, 6, 1, 3,
                2, 5, 4, 7)
    x = x.reshape(M * 4, K // 4)
    if op_idx == 1:
        x = x.trans()
    return x

def unswizzle_mxfp4_value_hopper_torch(x, op_idx: int, mma_version: int):
    """
    PyTorch inverse of swizzle_mxfp4_value_hopper. (Testing only)
    """
    assert op_idx in (0, 1), "op_idx must be 0 or 1"
    assert mma_version in (2, 3), "mma_version must be 2 or 3"
    if op_idx == 1:
        x = x.transpose(-2, -1)

    *batch, M, K = x.shape
    # We have two times the elements if we already upcasted to bfloat16
    mult = 2 if x.dtype == torch.bfloat16 else 1
    assert M % 4 == 0, "M must be divisible by 4"
    assert K % (4 * 8 * 2 * 2 * mult) == 0, f"K must be divisible by {4 * 8 * 2 * 2 * mult}"

    # We are loading 8 bf16 elements per thread to use ld.global.v4
    # Every u8 represents 2 mxfp4 elements
    u8_kwidth = 8 // 2 if mma_version == 2 else 1
    x = x.reshape(*batch, M // 4, 4, K // (4 * 8 * 2 * 2 * mult), 2, 4, 8 // u8_kwidth, 2, u8_kwidth * mult)
    b = len(batch)
    perm = [0, 6, 1, 3,
            2, 5, 4, 7]
    perm = list(range(b)) + [b + p for p in perm]
    x = x.permute(*perm)
    x = x.reshape(*batch, M * 4, K // 4)
    if op_idx == 1:
        x = x.transpose(-2, -1)
    return x
