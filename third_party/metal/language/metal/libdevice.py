"""Metal math library (libdevice equivalent) for Triton.

Provides math intrinsics mapped to Metal Shading Language (MSL) standard library
functions. This is the Metal equivalent of CUDA's libdevice / NVIDIA's __nv_*
functions.

Metal does NOT support FP64 (double precision) on Apple Silicon GPUs, so all
double variants are omitted. Functions map to MSL standard library names or
Metal-specific intrinsics (__metal_* prefix).

MSL naming conventions used here:
  - FP32 functions use C99-style 'f' suffix (e.g., sinf, cosf, expf)
  - Metal-specific builtins use __metal_ prefix (e.g., __metal_clz_u32)
  - Fast-math variants use __metal_fast_ prefix
  - FP16 (half) functions use 'h' suffix (e.g., sinh, cosh, exph)
"""

from triton.language import core


# ---------------------------------------------------------------------------
# Bit manipulation / integer intrinsics
# ---------------------------------------------------------------------------


@core.extern
def clz(arg0, _semantic=None):
    """Count leading zeros."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__metal_clz_u32", core.dtype("int32")),
            (core.dtype("uint32"), ): ("__metal_clz_u32", core.dtype("int32")),
            (core.dtype("int64"), ): ("__metal_clz_u64", core.dtype("int32")),
            (core.dtype("uint64"), ): ("__metal_clz_u64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def popc(arg0, _semantic=None):
    """Population count (number of set bits)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__metal_popcount_u32", core.dtype("int32")),
            (core.dtype("uint32"), ): ("__metal_popcount_u32", core.dtype("int32")),
            (core.dtype("int64"), ): ("__metal_popcount_u64", core.dtype("int32")),
            (core.dtype("uint64"), ): ("__metal_popcount_u64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def brev(arg0, _semantic=None):
    """Bit reverse."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__metal_reverse_bits_u32", core.dtype("int32")),
            (core.dtype("uint32"), ): ("__metal_reverse_bits_u32", core.dtype("uint32")),
            (core.dtype("int64"), ): ("__metal_reverse_bits_u64", core.dtype("int64")),
            (core.dtype("uint64"), ): ("__metal_reverse_bits_u64", core.dtype("uint64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ctz(arg0, _semantic=None):
    """Count trailing zeros."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__metal_ctz_u32", core.dtype("int32")),
            (core.dtype("uint32"), ): ("__metal_ctz_u32", core.dtype("int32")),
            (core.dtype("int64"), ): ("__metal_ctz_u64", core.dtype("int32")),
            (core.dtype("uint64"), ): ("__metal_ctz_u64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mulhi(arg0, arg1, _semantic=None):
    """High bits of integer multiplication."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__metal_mulhi_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__metal_mulhi_u32", core.dtype("uint32")),
            (core.dtype("int64"), core.dtype("int64")): ("__metal_mulhi_i64", core.dtype("int64")),
            (core.dtype("uint64"), core.dtype("uint64")): ("__metal_mulhi_u64", core.dtype("uint64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def hadd(arg0, arg1, _semantic=None):
    """Integer half-add: (arg0 + arg1) >> 1 without overflow."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__metal_hadd_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__metal_hadd_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rhadd(arg0, arg1, _semantic=None):
    """Integer rounded half-add: (arg0 + arg1 + 1) >> 1 without overflow."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__metal_rhadd_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__metal_rhadd_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Absolute value
# ---------------------------------------------------------------------------


@core.extern
def abs(arg0, _semantic=None):
    """Absolute value (integer and floating-point)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__metal_abs_i32", core.dtype("int32")),
            (core.dtype("int64"), ): ("__metal_abs_i64", core.dtype("int64")),
            (core.dtype("fp16"), ): ("__metal_fabs_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_fabs_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Rounding functions
# ---------------------------------------------------------------------------


@core.extern
def floor(arg0, _semantic=None):
    """Round toward negative infinity."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_floor_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_floor_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ceil(arg0, _semantic=None):
    """Round toward positive infinity."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_ceil_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_ceil_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def trunc(arg0, _semantic=None):
    """Round toward zero."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_trunc_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_trunc_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def round(arg0, _semantic=None):
    """Round to nearest integer, ties away from zero."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_round_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_round_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rint(arg0, _semantic=None):
    """Round to nearest integer (may raise inexact)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_rint_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_rint_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def nearbyint(arg0, _semantic=None):
    """Round to nearest integer using current rounding mode (no inexact exception)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_nearbyint_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Trigonometric functions
# ---------------------------------------------------------------------------


@core.extern
def sin(arg0, _semantic=None):
    """Sine."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_sin_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_sin_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cos(arg0, _semantic=None):
    """Cosine."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_cos_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_cos_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tan(arg0, _semantic=None):
    """Tangent."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_tan_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_tan_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asin(arg0, _semantic=None):
    """Arc sine."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_asin_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_asin_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acos(arg0, _semantic=None):
    """Arc cosine."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_acos_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_acos_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    """Arc tangent."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_atan_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_atan_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    """Arc tangent of y/x (two-argument form)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("fp16")): ("__metal_atan2_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_atan2_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinpi(arg0, _semantic=None):
    """Sine of pi*x: sin(pi * arg0). More accurate than sin(pi*x)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_sinpi_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_sinpi_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cospi(arg0, _semantic=None):
    """Cosine of pi*x: cos(pi * arg0). More accurate than cos(pi*x)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_cospi_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_cospi_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanpi(arg0, _semantic=None):
    """Tangent of pi*x: tan(pi * arg0). More accurate than tan(pi*x)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_tanpi_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Hyperbolic functions
# ---------------------------------------------------------------------------


@core.extern
def sinh(arg0, _semantic=None):
    """Hyperbolic sine."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_sinh_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_sinh_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cosh(arg0, _semantic=None):
    """Hyperbolic cosine."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_cosh_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_cosh_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    """Hyperbolic tangent."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_tanh_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_tanh_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asinh(arg0, _semantic=None):
    """Inverse hyperbolic sine."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_asinh_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_asinh_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acosh(arg0, _semantic=None):
    """Inverse hyperbolic cosine."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_acosh_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_acosh_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atanh(arg0, _semantic=None):
    """Inverse hyperbolic tangent."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_atanh_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_atanh_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Exponential functions
# ---------------------------------------------------------------------------


@core.extern
def exp(arg0, _semantic=None):
    """Base-e exponential."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_exp_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_exp_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp2(arg0, _semantic=None):
    """Base-2 exponential."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_exp2_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_exp2_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp10(arg0, _semantic=None):
    """Base-10 exponential."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_exp10_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_exp10_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def expm1(arg0, _semantic=None):
    """exp(x) - 1, accurate for small x."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_expm1_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Logarithmic functions
# ---------------------------------------------------------------------------


@core.extern
def log(arg0, _semantic=None):
    """Natural logarithm."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_log_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_log_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log2(arg0, _semantic=None):
    """Base-2 logarithm."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_log2_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_log2_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log10(arg0, _semantic=None):
    """Base-10 logarithm."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_log10_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_log10_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log1p(arg0, _semantic=None):
    """log(1 + x), accurate for small x."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_log1p_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Power and root functions
# ---------------------------------------------------------------------------


@core.extern
def sqrt(arg0, _semantic=None):
    """Square root."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_sqrt_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_sqrt_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt(arg0, _semantic=None):
    """Inverse square root (1/sqrt(x)). Metal provides rsqrt as a native operation."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_rsqrt_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_rsqrt_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cbrt(arg0, _semantic=None):
    """Cube root."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_cbrt_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    """Power function (x raised to the power y)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("fp16")): ("__metal_pow_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("int32")): ("__metal_powi_f32", core.dtype("fp32")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_pow_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def powr(arg0, arg1, _semantic=None):
    """Power function for x >= 0 (undefined for x < 0). Can be faster than pow."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("fp16")): ("__metal_powr_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_powr_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def hypot(arg0, arg1, _semantic=None):
    """Hypotenuse: sqrt(x*x + y*y) without overflow."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_hypot_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Fused multiply-add
# ---------------------------------------------------------------------------


@core.extern
def fma(arg0, arg1, arg2, _semantic=None):
    """Fused multiply-add: (arg0 * arg1) + arg2 with a single rounding."""
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp16"), core.dtype("fp16"), core.dtype("fp16")):
                ("__metal_fma_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_fma_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Remainder and modulus
# ---------------------------------------------------------------------------


@core.extern
def fmod(arg0, arg1, _semantic=None):
    """Floating-point remainder (same sign as dividend)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("fp16")): ("__metal_fmod_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_fmod_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def remainder(arg0, arg1, _semantic=None):
    """IEEE 754 remainder (result may have different sign from dividend)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_remainder_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Min / max (floating-point)
# ---------------------------------------------------------------------------


@core.extern
def fmin(arg0, arg1, _semantic=None):
    """Floating-point minimum. If one argument is NaN, returns the other."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("fp16")): ("__metal_fmin_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_fmin_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmax(arg0, arg1, _semantic=None):
    """Floating-point maximum. If one argument is NaN, returns the other."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("fp16")): ("__metal_fmax_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_fmax_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Integer min / max
# ---------------------------------------------------------------------------


@core.extern
def min(arg0, arg1, _semantic=None):
    """Integer minimum."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__metal_min_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__metal_min_u32", core.dtype("uint32")),
            (core.dtype("int64"), core.dtype("int64")): ("__metal_min_i64", core.dtype("int64")),
            (core.dtype("uint64"), core.dtype("uint64")): ("__metal_min_u64", core.dtype("uint64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def max(arg0, arg1, _semantic=None):
    """Integer maximum."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__metal_max_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__metal_max_u32", core.dtype("uint32")),
            (core.dtype("int64"), core.dtype("int64")): ("__metal_max_i64", core.dtype("int64")),
            (core.dtype("uint64"), core.dtype("uint64")): ("__metal_max_u64", core.dtype("uint64")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Sign, classification, and comparison helpers
# ---------------------------------------------------------------------------


@core.extern
def copysign(arg0, arg1, _semantic=None):
    """Copy sign of arg1 onto magnitude of arg0."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("fp16")): ("__metal_copysign_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_copysign_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def signbit(arg0, _semantic=None):
    """Return non-zero if sign bit is set."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_signbit_f32", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    """Test for infinity."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_isinf_f16", core.dtype("int32")),
            (core.dtype("fp32"), ): ("__metal_isinf_f32", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isnan(arg0, _semantic=None):
    """Test for NaN."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_isnan_f16", core.dtype("int32")),
            (core.dtype("fp32"), ): ("__metal_isnan_f32", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isfinite(arg0, _semantic=None):
    """Test for finite value (not inf, not NaN)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_isfinite_f16", core.dtype("int32")),
            (core.dtype("fp32"), ): ("__metal_isfinite_f32", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isnormal(arg0, _semantic=None):
    """Test for normal value (not zero, subnormal, inf, or NaN)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_isnormal_f32", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isunordered(arg0, arg1, _semantic=None):
    """Test if either argument is NaN (unordered comparison)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_isunordered_f32", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Positive difference
# ---------------------------------------------------------------------------


@core.extern
def fdim(arg0, arg1, _semantic=None):
    """Positive difference: max(arg0 - arg1, 0)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_fdim_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Error functions
# ---------------------------------------------------------------------------


@core.extern
def erf(arg0, _semantic=None):
    """Error function."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_erf_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfc(arg0, _semantic=None):
    """Complementary error function: 1 - erf(x)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_erfc_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfinv(arg0, _semantic=None):
    """Inverse error function."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_erfinv_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcinv(arg0, _semantic=None):
    """Inverse complementary error function."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_erfcinv_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcx(arg0, _semantic=None):
    """Scaled complementary error function: exp(x*x) * erfc(x)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_erfcx_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Normal distribution functions
# ---------------------------------------------------------------------------


@core.extern
def normcdf(arg0, _semantic=None):
    """Standard normal cumulative distribution function."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_normcdf_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def normcdfinv(arg0, _semantic=None):
    """Inverse standard normal cumulative distribution function."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_normcdfinv_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Gamma functions
# ---------------------------------------------------------------------------


@core.extern
def lgamma(arg0, _semantic=None):
    """Log-gamma function: log(|gamma(x)|)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_lgamma_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tgamma(arg0, _semantic=None):
    """True gamma function."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_tgamma_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Floating-point manipulation
# ---------------------------------------------------------------------------


@core.extern
def nextafter(arg0, arg1, _semantic=None):
    """Next representable float after arg0 in direction of arg1."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_nextafter_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ldexp(arg0, arg1, _semantic=None):
    """Load exponent: arg0 * 2^arg1."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__metal_ldexp_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def scalbn(arg0, arg1, _semantic=None):
    """Scale by power of radix: arg0 * FLT_RADIX^arg1."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__metal_scalbn_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ilogb(arg0, _semantic=None):
    """Extract unbiased exponent as integer."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_ilogb_f32", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def logb(arg0, _semantic=None):
    """Extract exponent as floating-point value."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_logb_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal-specific: saturate, clamp, mix, step, smoothstep, fract
# ---------------------------------------------------------------------------


@core.extern
def saturate(arg0, _semantic=None):
    """Clamp value to [0.0, 1.0] range. Metal native operation."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_saturate_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_saturate_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fract(arg0, _semantic=None):
    """Return fractional part: x - floor(x). Always in [0, 1). Metal native."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_fract_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_fract_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def clamp(arg0, arg1, arg2, _semantic=None):
    """Clamp arg0 to range [arg1, arg2]. Returns min(max(arg0, arg1), arg2)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp16"), core.dtype("fp16"), core.dtype("fp16")):
                ("__metal_clamp_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_clamp_f32", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("int32"), core.dtype("int32")):
                ("__metal_clamp_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32"), core.dtype("uint32")):
                ("__metal_clamp_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mix(arg0, arg1, arg2, _semantic=None):
    """Linear interpolation: arg0 + (arg1 - arg0) * arg2. Metal native."""
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp16"), core.dtype("fp16"), core.dtype("fp16")):
                ("__metal_mix_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_mix_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def step(arg0, arg1, _semantic=None):
    """Step function: returns 0.0 if arg1 < arg0, else 1.0."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("fp16")): ("__metal_step_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_step_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def smoothstep(arg0, arg1, arg2, _semantic=None):
    """Hermite interpolation between 0 and 1 when arg0 < arg2 < arg1."""
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp16"), core.dtype("fp16"), core.dtype("fp16")):
                ("__metal_smoothstep_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_smoothstep_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sign(arg0, _semantic=None):
    """Sign function: returns -1.0, 0.0, or 1.0."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_sign_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_sign_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Reciprocal
# ---------------------------------------------------------------------------


@core.extern
def rcp(arg0, _semantic=None):
    """Reciprocal: 1.0 / x. Metal can use fast hardware reciprocal."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__metal_rcp_f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__metal_rcp_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Fast math variants (reduced precision, higher performance)
# ---------------------------------------------------------------------------


@core.extern
def fast_sinf(arg0, _semantic=None):
    """Fast sine approximation (reduced precision)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_sin_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_cosf(arg0, _semantic=None):
    """Fast cosine approximation (reduced precision)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_cos_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_tanf(arg0, _semantic=None):
    """Fast tangent approximation (reduced precision)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_tan_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_sinpif(arg0, _semantic=None):
    """Fast sin(pi*x) approximation."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_sinpi_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_cospif(arg0, _semantic=None):
    """Fast cos(pi*x) approximation."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_cospi_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_tanpif(arg0, _semantic=None):
    """Fast tan(pi*x) approximation."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_tanpi_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_expf(arg0, _semantic=None):
    """Fast exponential approximation (reduced precision)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_exp_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_exp2f(arg0, _semantic=None):
    """Fast base-2 exponential approximation."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_exp2_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_exp10f(arg0, _semantic=None):
    """Fast base-10 exponential approximation."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_exp10_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_logf(arg0, _semantic=None):
    """Fast natural log approximation (reduced precision)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_log_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log2f(arg0, _semantic=None):
    """Fast base-2 log approximation."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_log2_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log10f(arg0, _semantic=None):
    """Fast base-10 log approximation."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_log10_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_powf(arg0, arg1, _semantic=None):
    """Fast power approximation (reduced precision)."""
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__metal_fast_pow_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_dividef(arg0, arg1, _semantic=None):
    """Fast divide approximation (reduced precision)."""
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__metal_fast_divide_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_rsqrtf(arg0, _semantic=None):
    """Fast inverse square root approximation."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_rsqrt_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_rcpf(arg0, _semantic=None):
    """Fast reciprocal approximation."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_fast_rcp_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Type reinterpretation (bitcast)
# ---------------------------------------------------------------------------


@core.extern
def int_as_float(arg0, _semantic=None):
    """Reinterpret int32 bits as float32."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__metal_as_type_f32_i32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_int(arg0, _semantic=None):
    """Reinterpret float32 bits as int32."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_as_type_i32_f32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint_as_float(arg0, _semantic=None):
    """Reinterpret uint32 bits as float32."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__metal_as_type_f32_u32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_uint(arg0, _semantic=None):
    """Reinterpret float32 bits as uint32."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_as_type_u32_f32", core.dtype("uint32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def short_as_half(arg0, _semantic=None):
    """Reinterpret int16 bits as fp16 (half)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int16"), ): ("__metal_as_type_f16_i16", core.dtype("fp16")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def half_as_short(arg0, _semantic=None):
    """Reinterpret fp16 (half) bits as int16."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp16"), ): ("__metal_as_type_i16_f16", core.dtype("int16")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ushort_as_half(arg0, _semantic=None):
    """Reinterpret uint16 bits as fp16 (half)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint16"), ): ("__metal_as_type_f16_u16", core.dtype("fp16")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def half_as_ushort(arg0, _semantic=None):
    """Reinterpret fp16 (half) bits as uint16."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp16"), ): ("__metal_as_type_u16_f16", core.dtype("uint16")),
    }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Type conversion utilities
# ---------------------------------------------------------------------------


@core.extern
def float2int_rn(arg0, _semantic=None):
    """Convert float32 to int32 with round-to-nearest-even."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_float2int_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rz(arg0, _semantic=None):
    """Convert float32 to int32 with round-toward-zero (truncation)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_float2int_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rn(arg0, _semantic=None):
    """Convert float32 to uint32 with round-to-nearest-even."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_float2uint_rn", core.dtype("uint32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rz(arg0, _semantic=None):
    """Convert float32 to uint32 with round-toward-zero (truncation)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_float2uint_rz", core.dtype("uint32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rn(arg0, _semantic=None):
    """Convert int32 to float32 with round-to-nearest-even."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__metal_int2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rn(arg0, _semantic=None):
    """Convert uint32 to float32 with round-to-nearest-even."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__metal_uint2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2half_rn(arg0, _semantic=None):
    """Convert float32 to fp16 with round-to-nearest-even."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_float2half_rn", core.dtype("fp16")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def half2float(arg0, _semantic=None):
    """Convert fp16 to float32 (exact, no rounding needed)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp16"), ): ("__metal_half2float", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal SIMD / warp-level intrinsics (simdgroup operations)
# ---------------------------------------------------------------------------


@core.extern
def simd_shuffle(arg0, arg1, _semantic=None):
    """SIMD shuffle: read value from lane arg1 in the simdgroup."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("int32")): ("__metal_simd_shuffle_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("int32")): ("__metal_simd_shuffle_f32", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("int32")): ("__metal_simd_shuffle_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("int32")): ("__metal_simd_shuffle_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def simd_shuffle_xor(arg0, arg1, _semantic=None):
    """SIMD shuffle XOR: read value from lane (current_lane XOR arg1)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__metal_simd_shuffle_xor_f32", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("int32")): ("__metal_simd_shuffle_xor_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("int32")): ("__metal_simd_shuffle_xor_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def simd_shuffle_up(arg0, arg1, _semantic=None):
    """SIMD shuffle up: read value from lane (current_lane - arg1)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__metal_simd_shuffle_up_f32", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("int32")): ("__metal_simd_shuffle_up_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("int32")): ("__metal_simd_shuffle_up_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def simd_shuffle_down(arg0, arg1, _semantic=None):
    """SIMD shuffle down: read value from lane (current_lane + arg1)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__metal_simd_shuffle_down_f32", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("int32")): ("__metal_simd_shuffle_down_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("int32")): ("__metal_simd_shuffle_down_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal-specific: precise math (full precision, no fast-math)
# ---------------------------------------------------------------------------


@core.extern
def precise_sin(arg0, _semantic=None):
    """Precise sine (full IEEE 754 compliance, no fast-math optimization)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_precise_sin_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def precise_cos(arg0, _semantic=None):
    """Precise cosine (full IEEE 754 compliance, no fast-math optimization)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_precise_cos_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def precise_sqrt(arg0, _semantic=None):
    """Precise square root (correctly rounded)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_precise_sqrt_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def precise_exp(arg0, _semantic=None):
    """Precise exponential (full IEEE 754 compliance)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_precise_exp_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def precise_exp2(arg0, _semantic=None):
    """Precise base-2 exponential (full IEEE 754 compliance)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_precise_exp2_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def precise_log(arg0, _semantic=None):
    """Precise natural logarithm (full IEEE 754 compliance)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_precise_log_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def precise_log2(arg0, _semantic=None):
    """Precise base-2 logarithm (full IEEE 754 compliance)."""
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__metal_precise_log2_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def precise_pow(arg0, arg1, _semantic=None):
    """Precise power function (full IEEE 754 compliance)."""
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__metal_precise_pow_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def precise_divide(arg0, arg1, _semantic=None):
    """Precise division (correctly rounded)."""
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__metal_precise_divide_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal-specific: dot product and distance (commonly used in shaders)
# ---------------------------------------------------------------------------


@core.extern
def distance(arg0, arg1, _semantic=None):
    """Euclidean distance between two scalar values: |arg0 - arg1|."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__metal_distance_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def length(arg0, _semantic=None):
    """Length (magnitude) of a scalar value: |arg0|. Same as fabs for scalars."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_length_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def normalize(arg0, _semantic=None):
    """Normalize: arg0 / |arg0|. Returns sign for scalars."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_normalize_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal-specific: extract_bits and insert_bits
# ---------------------------------------------------------------------------


@core.extern
def extract_bits(arg0, arg1, arg2, _semantic=None):
    """Extract a bitfield from arg0 starting at bit arg1 with width arg2."""
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("int32"), core.dtype("int32"), core.dtype("int32")):
                ("__metal_extract_bits_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("int32"), core.dtype("int32")):
                ("__metal_extract_bits_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def insert_bits(arg0, arg1, arg2, arg3, _semantic=None):
    """Insert bits from arg1 into arg0 starting at bit arg2 with width arg3."""
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("int32"), core.dtype("int32"), core.dtype("int32"), core.dtype("int32")):
                ("__metal_insert_bits_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32"), core.dtype("int32"), core.dtype("int32")):
                ("__metal_insert_bits_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal-specific: rotate
# ---------------------------------------------------------------------------


@core.extern
def rotate(arg0, arg1, _semantic=None):
    """Bitwise rotate left by arg1 positions."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__metal_rotate_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__metal_rotate_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal-specific: select (ternary)
# ---------------------------------------------------------------------------


@core.extern
def select(arg0, arg1, arg2, _semantic=None):
    """Conditional select: returns arg1 if arg2 is true (MSB set), else arg0."""
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("int32")):
                ("__metal_select_f32", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("int32"), core.dtype("int32")):
                ("__metal_select_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32"), core.dtype("int32")):
                ("__metal_select_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal-specific: mad (multiply-add, potentially faster than fma on some HW)
# ---------------------------------------------------------------------------


@core.extern
def mad(arg0, arg1, arg2, _semantic=None):
    """Multiply and add: arg0 * arg1 + arg2. May not be fused (unlike fma)."""
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp16"), core.dtype("fp16"), core.dtype("fp16")):
                ("__metal_mad_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_mad_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal-specific: absdiff (absolute difference)
# ---------------------------------------------------------------------------


@core.extern
def absdiff(arg0, arg1, _semantic=None):
    """Absolute difference: |arg0 - arg1| without overflow for integers."""
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__metal_absdiff_i32", core.dtype("uint32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__metal_absdiff_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal-specific: popcount for smaller types
# ---------------------------------------------------------------------------


@core.extern
def popcount(arg0, _semantic=None):
    """Population count (alias for popc with additional type support)."""
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int16"), ): ("__metal_popcount_u16", core.dtype("int16")),
            (core.dtype("uint16"), ): ("__metal_popcount_u16", core.dtype("int16")),
            (core.dtype("int32"), ): ("__metal_popcount_u32", core.dtype("int32")),
            (core.dtype("uint32"), ): ("__metal_popcount_u32", core.dtype("int32")),
            (core.dtype("int64"), ): ("__metal_popcount_u64", core.dtype("int32")),
            (core.dtype("uint64"), ): ("__metal_popcount_u64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal threadgroup barrier (memory fence)
# ---------------------------------------------------------------------------


@core.extern
def threadgroup_barrier(_semantic=None):
    """Metal threadgroup memory barrier (equivalent to __syncthreads in CUDA)."""
    return core.extern_elementwise("", "", [], {
        (): ("__metal_threadgroup_barrier", core.dtype("int32")),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def simdgroup_barrier(_semantic=None):
    """Metal simdgroup memory barrier (equivalent to __syncwarp in CUDA)."""
    return core.extern_elementwise("", "", [], {
        (): ("__metal_simdgroup_barrier", core.dtype("int32")),
    }, is_pure=False, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Metal 4 features (macOS 26+ / MTLLanguageVersion4_0)
# ---------------------------------------------------------------------------


@core.extern
def simdgroup_matrix_multiply(a, b, _semantic=None):
    """Metal 4 simdgroup matrix multiply-accumulate (8x8 tiles).

    Uses Apple GPU's hardware matrix units for accelerated matmul.
    Equivalent to NVIDIA's wmma/mma instructions.
    """
    return core.extern_elementwise("", "", [a, b], {
        (core.dtype("fp16"), core.dtype("fp16")): ("__metal_simdgroup_matrix_multiply_f16", core.dtype("fp16")),
        (core.dtype("fp32"), core.dtype("fp32")): ("__metal_simdgroup_matrix_multiply_f32", core.dtype("fp32")),
        (core.dtype("bf16"), core.dtype("bf16")): ("__metal_simdgroup_matrix_multiply_bf16", core.dtype("bf16")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def simdgroup_load(src, _semantic=None):
    """Load data into a simdgroup matrix tile from device memory."""
    return core.extern_elementwise("", "", [src], {
        (core.dtype("fp16"), ): ("__metal_simdgroup_load_f16", core.dtype("fp16")),
        (core.dtype("fp32"), ): ("__metal_simdgroup_load_f32", core.dtype("fp32")),
        (core.dtype("bf16"), ): ("__metal_simdgroup_load_bf16", core.dtype("bf16")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def simdgroup_store(val, dst, _semantic=None):
    """Store simdgroup matrix tile data to device memory."""
    return core.extern_elementwise("", "", [val, dst], {
        (core.dtype("fp16"), core.dtype("fp16")): ("__metal_simdgroup_store_f16", core.dtype("fp16")),
        (core.dtype("fp32"), core.dtype("fp32")): ("__metal_simdgroup_store_f32", core.dtype("fp32")),
        (core.dtype("bf16"), core.dtype("bf16")): ("__metal_simdgroup_store_bf16", core.dtype("bf16")),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def atomic_load_explicit(ptr, _semantic=None):
    """Metal 4 explicit atomic load (relaxed memory order)."""
    return core.extern_elementwise("", "", [ptr], {
        (core.dtype("int32"), ): ("__metal_atomic_load_explicit_i32", core.dtype("int32")),
        (core.dtype("uint32"), ): ("__metal_atomic_load_explicit_u32", core.dtype("uint32")),
        (core.dtype("fp32"), ): ("__metal_atomic_load_explicit_f32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def atomic_store_explicit(ptr, val, _semantic=None):
    """Metal 4 explicit atomic store (relaxed memory order)."""
    return core.extern_elementwise("", "", [ptr, val], {
        (core.dtype("int32"), core.dtype("int32")): ("__metal_atomic_store_explicit_i32", core.dtype("int32")),
        (core.dtype("uint32"), core.dtype("uint32")): ("__metal_atomic_store_explicit_u32", core.dtype("uint32")),
        (core.dtype("fp32"), core.dtype("fp32")): ("__metal_atomic_store_explicit_f32", core.dtype("fp32")),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def threadgroup_async_copy(dst, src, num_elements, _semantic=None):
    """Asynchronous threadgroup memory copy (Metal 4).

    Initiates an async copy from device to threadgroup memory,
    similar to CUDA's cp.async.
    """
    return core.extern_elementwise("", "", [dst, src, num_elements], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("int32")):
            ("__metal_threadgroup_async_copy_f32", core.dtype("int32")),
        (core.dtype("fp16"), core.dtype("fp16"), core.dtype("int32")):
            ("__metal_threadgroup_async_copy_f16", core.dtype("int32")),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def threadgroup_async_copy_wait(_semantic=None):
    """Wait for all pending async threadgroup copies to complete."""
    return core.extern_elementwise("", "", [], {
        (): ("__metal_threadgroup_async_copy_wait", core.dtype("int32")),
    }, is_pure=False, _semantic=_semantic)
