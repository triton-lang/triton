import triton
import triton.language as tl

from ._downcast_to_mxfp import MXFP_BLOCK_SIZE, NVFP_BLOCK_SIZE
from triton_kernels.target_info import cuda_capability_geq


@triton.jit
def upcast_ue8m0_scale(scale, dst_dtype: tl.constexpr, handle_nan: tl.constexpr = True):
    scale = scale.to(tl.uint8)
    if dst_dtype == tl.bfloat16 and scale.numel % 2 == 0 and cuda_capability_geq(10, 0):
        return tl.inline_asm_elementwise(
            "cvt.rn.bf16x2.ue8m0x2 $0, $1;",
            constraints="=r,h",
            args=[scale],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=2,
        )
    # E8M0 byte zero is 2**-127, which is subnormal in BF16 and FP32.
    if dst_dtype == tl.bfloat16:
        bits = tl.maximum(scale.to(tl.uint16) << 7, 0x0040)
        scale = bits.to(tl.uint16).to(dst_dtype, bitcast=True)
    else:
        bits = scale.to(tl.uint32) << 23
        if dst_dtype == tl.float32:
            bits = tl.maximum(bits, 0x00400000)
        scale = bits.to(tl.float32, bitcast=True)
    # Saturating callers restore NaNs after clamping instead.
    if handle_nan:
        scale = tl.fma(scale, tl.zeros((), scale.dtype), scale)
    return scale.to(dst_dtype)


@triton.jit
def _upcast_mxfp4_values(tensor, dst_dtype: tl.constexpr):
    tl.static_assert(tensor.dtype == tl.uint8)
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16 or dst_dtype == tl.float32)
    intermediate_dtype: tl.constexpr = tl.bfloat16 if dst_dtype == tl.float32 else dst_dtype
    if cuda_capability_geq(10, 0):
        packed_u32 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 in_8;
            .reg .f16x2 out;
            cvt.u8.u32 in_8, $1;
            cvt.rn.f16x2.e2m1x2 out, in_8;
            mov.b32 $0, out;
            }
            """,
            constraints="=r,r",
            args=[tensor],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        lo_u16 = (packed_u32 & 0xFFFF).to(tl.uint16)
        hi_u16 = (packed_u32 >> 16).to(tl.uint16)
        lo_f16 = lo_u16.to(tl.float16, bitcast=True)
        hi_f16 = hi_u16.to(tl.float16, bitcast=True)
        if intermediate_dtype == tl.float16:
            x0, x1 = lo_f16, hi_f16
        else:
            x0 = lo_f16.to(intermediate_dtype)
            x1 = hi_f16.to(intermediate_dtype)
        dst_tensor = tl.interleave(x0, x1)
    else:
        dst_bias: tl.constexpr = 127 if intermediate_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if intermediate_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if intermediate_dtype == tl.bfloat16 else 10
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((tensor & 0x08).to(tl.uint16) << 12)
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((tensor & 0x80).to(tl.uint16) << 8)
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        dst_tensor = tl.interleave(x0, x1).to(intermediate_dtype, bitcast=True)
    return dst_tensor.to(dst_dtype)


@triton.jit
def upcast_mxfp4_tile(tensor, scale, dst_dtype: tl.constexpr):
    tl.static_assert(len(tensor.shape) == 2)
    tl.static_assert(len(scale.shape) == 2)
    tl.static_assert(tensor.dtype == tl.uint8)
    tl.static_assert(scale.dtype == tl.uint8)
    tl.static_assert(tensor.shape[0] == scale.shape[0])
    tl.static_assert(tensor.shape[1] * 2 == scale.shape[1] * MXFP_BLOCK_SIZE)

    dst_scale = upcast_ue8m0_scale(scale, dst_dtype, handle_nan=False)
    dst_tensor = _upcast_mxfp4_values(tensor, dst_dtype)
    dst_tensor = dst_tensor.reshape([tensor.shape[0], scale.shape[1], MXFP_BLOCK_SIZE])
    dst_scale = dst_scale.reshape([scale.shape[0], scale.shape[1], 1])
    scale = scale.reshape(dst_scale.shape)
    out_tensor = dst_tensor * dst_scale
    if dst_dtype == tl.float32:
        max_fin = 3.4028234663852886e+38
    elif dst_dtype == tl.bfloat16:
        max_fin = 3.3895313892515355e+38
    else:
        tl.static_assert(dst_dtype == tl.float16)
        max_fin = 65504
    out_tensor = tl.clamp(out_tensor, min=-max_fin, max=max_fin)
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    return out_tensor.to(dst_dtype).reshape([tensor.shape[0], tensor.shape[1] * 2])


@triton.jit
def upcast_nvfp4_tile(tensor, scale, dst_dtype: tl.constexpr):
    tl.static_assert(len(tensor.shape) == 2)
    tl.static_assert(len(scale.shape) == 2)
    tl.static_assert(tensor.dtype == tl.uint8)
    tl.static_assert(scale.dtype == tl.float8e4nv)
    tl.static_assert(tensor.shape[0] == scale.shape[0])
    tl.static_assert(tensor.shape[1] * 2 == scale.shape[1] * NVFP_BLOCK_SIZE)

    # Only apply block scales here. Row/fiber scales are applied once to the
    # accumulated matmul result, not once per K tile.
    dst_tensor = _upcast_mxfp4_values(tensor, dst_dtype)
    dst_tensor = dst_tensor.reshape([tensor.shape[0], scale.shape[1], NVFP_BLOCK_SIZE])
    dst_scale = scale.to(dst_dtype).reshape([scale.shape[0], scale.shape[1], 1])
    out_tensor = dst_tensor * dst_scale
    out_tensor = out_tensor.reshape([tensor.shape[0], tensor.shape[1] * 2])
    return out_tensor


@triton.jit
def _mxfp4_maximum_code(values):
    # Pack explicit groups before the SIMD reduction; inline-asm pack ordering
    # alone does not guarantee that elements belong to the same scale block.
    even, odd = tl.split(values.reshape(values.shape[0], values.shape[1], 2, 2, 2))
    byte0, byte2 = tl.split(even)
    byte1, byte3 = tl.split(odd)
    packed = (byte0.to(tl.uint32) | (byte1.to(tl.uint32) << 8)
              | (byte2.to(tl.uint32) << 16) | (byte3.to(tl.uint32) << 24))
    # Each halfword holds one magnitude code. Native packed maxima avoid
    # expanding all sixteen FP4 magnitudes into separate scalar reductions.
    word_maximum = tl.inline_asm_elementwise(
        """{
        .reg .b32 a, b;
        and.b32 a, $1, 0x00070007;
        shr.u32 b, $1, 4;  and.b32 b, b, 0x00070007; max.u16x2 a, a, b;
        shr.u32 b, $1, 8;  and.b32 b, b, 0x00070007; max.u16x2 a, a, b;
        shr.u32 b, $1, 12; and.b32 b, b, 0x00070007; max.u16x2 a, a, b;
        shr.u32 b, a, 16;
        and.b32 a, a, 7;
        max.u32 $0, a, b;
        }""",
        constraints="=r,r",
        args=[packed],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    # Keep the reduction visible to prevent duplicating it into both scale layouts.
    return tl.max(word_maximum, 2)


@triton.jit
def nvfp4_to_mxfp8_tile(values, scales):
    # FP4 magnitude codes are ordered. Reduce before decoding so scale selection
    # does not duplicate the full activation decode in a second register layout.
    maximum_code = _mxfp4_maximum_code(values.reshape(values.shape[0], scales.shape[1], 8))
    maximum_bits = (maximum_code << 22) + 0x3F000000
    maximum = tl.where(maximum_code < 2, maximum_code.to(tl.float32) * 0.5, maximum_bits.to(tl.float32, bitcast=True))
    maximum *= tl.abs(scales.to(tl.float32))
    maximum = tl.max(maximum.reshape(values.shape[0], scales.shape[1] // 2, 2), 2)
    dequant_scale = maximum / 448.0
    scale_bits = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    rounded_scale = scale_bits.to(tl.float32, bitcast=True)
    inverse_scale = tl.where(rounded_scale == 0, 0, 1.0 / rounded_scale)

    # The MX scale is a power of two, so folding its inverse into each NVFP4
    # block scale preserves FP32 rounding while avoiding per-value scale work.
    block_scale = scales.to(tl.float32).reshape(values.shape[0], scales.shape[1] // 2, 2)
    block_scale = (block_scale * inverse_scale[:, :, None]).reshape(values.shape[0], scales.shape[1], 1)
    # Raw FP4 values fit exactly in FP16; promote before any scale arithmetic.
    # This also lets the compiler use matrix loads for the packed input tile.
    decoded = _upcast_mxfp4_values(values, tl.float16).to(tl.float32)
    decoded = decoded.reshape(values.shape[0], scales.shape[1], 16)
    quantized = (decoded * block_scale).reshape(values.shape[0], values.shape[1] * 2).to(tl.float8e4nv)
    # Outer tensor/row scales remain in the matmul epilogue.
    return quantized, (scale_bits >> 23).to(tl.uint8)


# fmt: off
@triton.jit
def _upcast_from_mxfp(
    out_desc,
    mx_tensor_desc,
    mx_scale_ptr,
    stride_scale_outer,
    stride_scale_quant,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    MX_BLOCK_SIZE: tl.constexpr,
):

    tl.static_assert(MX_BLOCK_SIZE == MXFP_BLOCK_SIZE or MX_BLOCK_SIZE == NVFP_BLOCK_SIZE)
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MX_BLOCK_SIZE == 0, f"Block size along quantization block must be a multiple of {MX_BLOCK_SIZE=}")
    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_desc.dtype
    dst_dtype: tl.constexpr = out_desc.dtype
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16 or dst_dtype == tl.float32)
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or ((mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5) or mx_tensor_dtype == dst_dtype),
        "mx_tensor_ptr must be uint8 or float8 or dst_dtype")
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8 or mx_scale_ptr.dtype.element_ty == tl.float8e4nv,
        "mx_scale_ptr must be uint8 or float8e4nv",
    )

    # Determine if we are dealing with fp8 types.
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    scale_is_ocp: tl.constexpr = mx_scale_ptr.dtype.element_ty == tl.uint8
    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MX_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    # Compute starting indices for the quantized (packed) dimension and the outer dimension.
    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    # Load the quantized value tensor.
    tensor = mx_tensor_desc.load([start_out.to(tl.int32), start_mxt_quant.to(tl.int32)])

    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    mask_outer = start_out + offs_outer < outer_dim
    mask_scale = start_mx_scale_quant + offs_scale < tl.cdiv(quant_dim, MX_BLOCK_SIZE)
    full_scale_mask = mask_scale & mask_outer
    scale_offsets = offs_scale * stride_scale_quant + offs_outer * stride_scale_outer
    scale_ptr_base = mx_scale_ptr + start_out * stride_scale_outer + start_mx_scale_quant * stride_scale_quant
    scale = tl.load(scale_ptr_base + scale_offsets, mask=full_scale_mask)

    # Upcast the scale to the destination type.
    if scale_is_ocp:
        dst_scale = upcast_ue8m0_scale(scale, dst_dtype, handle_nan=False)
    else:
        dst_scale = scale.to(dst_dtype)

    # Now upcast the tensor.
    intermediate_dtype: tl.constexpr = tl.bfloat16 if dst_dtype == tl.float32 else dst_dtype
    if is_fp8:
        dst_tensor = tensor.to(intermediate_dtype)
        if tensor.dtype == tl.float8e5:
            from_e_bits: tl.constexpr = 5
            from_m_bits: tl.constexpr = 2
            to_e_bits: tl.constexpr = 8 if intermediate_dtype == tl.bfloat16 else 5
            to_m_bits: tl.constexpr = 7 if intermediate_dtype == tl.bfloat16 else 10

            # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
            non_finite_mask_src: tl.constexpr = ((1 << from_e_bits) - 1) << from_m_bits
            non_finite_mask_dst: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
            dst_tensor = tl.where(
                (tensor.to(tl.uint8, bitcast=True) & non_finite_mask_src) == non_finite_mask_src,
                (dst_tensor.to(tl.uint16, bitcast=True) | non_finite_mask_dst).to(intermediate_dtype, bitcast=True),
                dst_tensor,
            )

    else:
        assert is_fp4
        dst_tensor = _upcast_mxfp4_values(tensor, dst_dtype)

    # Reshape for proper broadcasting over the microscale block.
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MX_BLOCK_SIZE])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    if dst_dtype == tl.float32:
        max_fin = 3.4028234663852886e+38
    elif dst_dtype == tl.bfloat16:
        max_fin = 3.3895313892515355e+38
    else:
        tl.static_assert(dst_dtype == tl.float16)
        max_fin = 65504
    # TODO: handle infinity same as upcast_from_mxfp_torch together with the
    # above FIXME
    out_tensor = tl.clamp(out_tensor, min=-max_fin, max=max_fin)
    # Correct any NaNs encoded via OCP E8M0 scales.
    if scale_is_ocp:
        out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    out_desc.store([start_out.to(tl.int32), start_out_quant.to(tl.int32)], out_tensor)
