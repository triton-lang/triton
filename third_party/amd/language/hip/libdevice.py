from triton.language import core


@core.extern
def clz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__ockl_clz_u32", core.dtype("int32")),
            (core.dtype("int64"), ): ("__ockl_clz_u64", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic).to(core.int32, _semantic=_semantic)


@core.extern
def popc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__ockl_popcount_u32", core.dtype("int32")),
            (core.dtype("int64"), ): ("__ockl_popcount_u64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def abs(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__triton_hip_iabs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__triton_hip_iabs", core.dtype("int64")),
            (core.dtype("fp32"), ): ("__triton_hip_fabs", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__triton_hip_fabs", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def floor(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_floor_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_floor_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_rsqrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_rsqrt_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ceil(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_ceil_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_ceil_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def trunc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_trunc_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_trunc_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_exp2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_exp2_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_exp_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_exp_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_expf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__triton_hip_fast_expf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_tanhf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__triton_hip_fast_tanhf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_dividef(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__triton_hip_fast_fdividef", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sqrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sqrt_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__triton_hip_rint", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__triton_hip_rint", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def llrint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__triton_hip_llrint", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__triton_hip_llrint", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def nearbyint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_nearbyint_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_nearbyint_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isnan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_isnan_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_isnan_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def signbit(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_signbit_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_signbit_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def copysign(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_copysign_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_copysign_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_isinf_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_isinf_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def nextafter(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_nextafter_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_nextafter_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sin_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sin_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_cos_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_cos_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_tan_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_tan_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log2_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_cosh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_cosh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sinh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sinh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_tanh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_tanh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_atan2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_atan2_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_atan_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_atan_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_asin_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_asin_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_acos_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_acos_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log10_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log10_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log1p(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log1p_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log1p_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_acosh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_acosh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_asinh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_asinh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_atanh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_atanh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def expm1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_expm1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_expm1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def hypot(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_hypot_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_hypot_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_j0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_j0_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_j1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_j1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def y0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_y0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_y0_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def y1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_y1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_y1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_i0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_i0_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_i1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_i1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erf_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erf_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfinv_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfinv_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfc_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfc_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcx(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfcx_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfcx_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def lgamma(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_lgamma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_lgamma_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ldexp(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ocml_ldexp_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ocml_ldexp_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmod(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_fmod_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_fmod_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ocml_fma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ocml_fma_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ocml_pown_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ocml_pown_f64", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_pow_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_pow_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ilogb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_ilogb_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_ilogb_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def round(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_round_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_round_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def finitef(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ocml_isfinite_f32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isfinited(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ocml_isfinite_f64", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Functions below were missing from the HIP libdevice but have direct OCML
# equivalents. Added to close the parity gap with CUDA libdevice.
# ---------------------------------------------------------------------------


@core.extern
def cbrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_cbrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_cbrt_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcbrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_rcbrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_rcbrt_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_exp10_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_exp10_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfcinv_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfcinv_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tgamma(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_tgamma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_tgamma_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def logb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_logb_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_logb_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def remainder(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_remainder_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_remainder_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rhypot(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_rhypot_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_rhypot_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fdim(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_fdim_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_fdim_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def scalbn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ocml_scalbn_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ocml_scalbn_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def llround(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_llround_f32", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__ocml_llround_f64", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinpi(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sinpi_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sinpi_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cospi(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_cospi_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_cospi_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def normcdf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_ncdf_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_ncdf_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def normcdfinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_ncdfinv_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_ncdfinv_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def jn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("fp32")): ("__ocml_jn_f32", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("fp64")): ("__ocml_jn_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def yn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("fp32")): ("__ocml_yn_f32", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("fp64")): ("__ocml_yn_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def norm3d(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ocml_len3_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ocml_len3_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rnorm3d(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ocml_rlen3_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ocml_rlen3_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def norm4d(arg0, arg1, arg2, arg3, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ocml_len4_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ocml_len4_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rnorm4d(arg0, arg1, arg2, arg3, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ocml_rlen4_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ocml_rlen4_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_sinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_native_sin_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_cosf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_native_cos_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_logf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_native_log_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log2f(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_native_log2_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log10f(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_native_log10_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_exp10f(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_native_exp10_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_tanf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_native_tan_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_powf(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_native_powr_f32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
