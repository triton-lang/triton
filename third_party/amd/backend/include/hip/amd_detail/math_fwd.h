/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#if !defined(__HIPCC_RTC__)
#include "host_defines.h"
#include "amd_hip_vector_types.h"  // For Native_vec_
#endif

#if defined(__cplusplus)
    extern "C" {
#endif

// DOT FUNCTIONS
#if defined(__clang__) && defined(__HIP__)
__device__
__attribute__((const))
int __ockl_sdot2(
    HIP_vector_base<short, 2>::Native_vec_,
    HIP_vector_base<short, 2>::Native_vec_,
    int, bool);

__device__
__attribute__((const))
unsigned int __ockl_udot2(
    HIP_vector_base<unsigned short, 2>::Native_vec_,
    HIP_vector_base<unsigned short, 2>::Native_vec_,
    unsigned int, bool);

__device__
__attribute__((const))
int __ockl_sdot4(
    HIP_vector_base<char, 4>::Native_vec_,
    HIP_vector_base<char, 4>::Native_vec_,
    int, bool);

__device__
__attribute__((const))
unsigned int __ockl_udot4(
    HIP_vector_base<unsigned char, 4>::Native_vec_,
    HIP_vector_base<unsigned char, 4>::Native_vec_,
    unsigned int, bool);

__device__
__attribute__((const))
int __ockl_sdot8(int, int, int, bool);

__device__
__attribute__((const))
unsigned int __ockl_udot8(unsigned int, unsigned int, unsigned int, bool);
#endif

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
// BEGIN FLOAT
__device__
__attribute__((const))
float __ocml_acos_f32(float);
__device__
__attribute__((pure))
float __ocml_acosh_f32(float);
__device__
__attribute__((const))
float __ocml_asin_f32(float);
__device__
__attribute__((pure))
float __ocml_asinh_f32(float);
__device__
__attribute__((const))
float __ocml_atan2_f32(float, float);
__device__
__attribute__((const))
float __ocml_atan_f32(float);
__device__
__attribute__((pure))
float __ocml_atanh_f32(float);
__device__
__attribute__((pure))
float __ocml_cbrt_f32(float);
__device__
__attribute__((const))
float __ocml_ceil_f32(float);
__device__
__attribute__((const))
__device__
float __ocml_copysign_f32(float, float);
__device__
float __ocml_cos_f32(float);
__device__
float __ocml_native_cos_f32(float);
__device__
__attribute__((pure))
__device__
float __ocml_cosh_f32(float);
__device__
float __ocml_cospi_f32(float);
__device__
float __ocml_i0_f32(float);
__device__
float __ocml_i1_f32(float);
__device__
__attribute__((pure))
float __ocml_erfc_f32(float);
__device__
__attribute__((pure))
float __ocml_erfcinv_f32(float);
__device__
__attribute__((pure))
float __ocml_erfcx_f32(float);
__device__
__attribute__((pure))
float __ocml_erf_f32(float);
__device__
__attribute__((pure))
float __ocml_erfinv_f32(float);
__device__
__attribute__((pure))
float __ocml_exp10_f32(float);
__device__
__attribute__((pure))
float __ocml_native_exp10_f32(float);
__device__
__attribute__((pure))
float __ocml_exp2_f32(float);
__device__
__attribute__((pure))
float __ocml_exp_f32(float);
__device__
__attribute__((pure))
float __ocml_native_exp_f32(float);
__device__
__attribute__((pure))
float __ocml_expm1_f32(float);
__device__
__attribute__((const))
float __ocml_fabs_f32(float);
__device__
__attribute__((const))
float __ocml_fdim_f32(float, float);
__device__
__attribute__((const))
float __ocml_floor_f32(float);
__device__
__attribute__((const))
float __ocml_fma_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fmax_f32(float, float);
__device__
__attribute__((const))
float __ocml_fmin_f32(float, float);
__device__
__attribute__((const))
__device__
float __ocml_fmod_f32(float, float);
__device__
float __ocml_frexp_f32(float, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
float __ocml_hypot_f32(float, float);
__device__
__attribute__((const))
int __ocml_ilogb_f32(float);
__device__
__attribute__((const))
int __ocml_isfinite_f32(float);
__device__
__attribute__((const))
int __ocml_isinf_f32(float);
__device__
__attribute__((const))
int __ocml_isnan_f32(float);
__device__
float __ocml_j0_f32(float);
__device__
float __ocml_j1_f32(float);
__device__
__attribute__((const))
float __ocml_ldexp_f32(float, int);
__device__
float __ocml_lgamma_f32(float);
__device__
__attribute__((pure))
float __ocml_log10_f32(float);
__device__
__attribute__((pure))
float __ocml_native_log10_f32(float);
__device__
__attribute__((pure))
float __ocml_log1p_f32(float);
__device__
__attribute__((pure))
float __ocml_log2_f32(float);
__device__
__attribute__((pure))
float __ocml_native_log2_f32(float);
__device__
__attribute__((const))
float __ocml_logb_f32(float);
__device__
__attribute__((pure))
float __ocml_log_f32(float);
__device__
__attribute__((pure))
float __ocml_native_log_f32(float);
__device__
float __ocml_modf_f32(float, __attribute__((address_space(5))) float*);
__device__
__attribute__((const))
float __ocml_nearbyint_f32(float);
__device__
__attribute__((const))
float __ocml_nextafter_f32(float, float);
__device__
__attribute__((const))
float __ocml_len3_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_len4_f32(float, float, float, float);
__device__
__attribute__((pure))
float __ocml_ncdf_f32(float);
__device__
__attribute__((pure))
float __ocml_ncdfinv_f32(float);
__device__
__attribute__((pure))
float __ocml_pow_f32(float, float);
__device__
__attribute__((pure))
float __ocml_pown_f32(float, int);
__device__
__attribute__((pure))
float __ocml_rcbrt_f32(float);
__device__
__attribute__((const))
float __ocml_remainder_f32(float, float);
__device__
float __ocml_remquo_f32(float, float, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
float __ocml_rhypot_f32(float, float);
__device__
__attribute__((const))
float __ocml_rint_f32(float);
__device__
__attribute__((const))
float __ocml_rlen3_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_rlen4_f32(float, float, float, float);
__device__
__attribute__((const))
float __ocml_round_f32(float);
__device__
__attribute__((pure))
float __ocml_rsqrt_f32(float);
__device__
__attribute__((const))
float __ocml_scalb_f32(float, float);
__device__
__attribute__((const))
float __ocml_scalbn_f32(float, int);
__device__
__attribute__((const))
int __ocml_signbit_f32(float);
__device__
float __ocml_sincos_f32(float, __attribute__((address_space(5))) float*);
__device__
float __ocml_sincospi_f32(float, __attribute__((address_space(5))) float*);
__device__
float __ocml_sin_f32(float);
__device__
float __ocml_native_sin_f32(float);
__device__
__attribute__((pure))
float __ocml_sinh_f32(float);
__device__
float __ocml_sinpi_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_f32(float);
__device__
__attribute__((const))
float __ocml_native_sqrt_f32(float);
__device__
float __ocml_tan_f32(float);
__device__
__attribute__((pure))
float __ocml_tanh_f32(float);
__device__
float __ocml_tgamma_f32(float);
__device__
__attribute__((const))
float __ocml_trunc_f32(float);
__device__
float __ocml_y0_f32(float);
__device__
float __ocml_y1_f32(float);

// BEGIN INTRINSICS
__device__
__attribute__((const))
float __ocml_add_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_add_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_add_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_add_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_sqrt_rte_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_rtn_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_rtp_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_rtz_f32(float);
__device__
__attribute__((const))
float __ocml_fma_rte_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fma_rtn_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fma_rtp_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fma_rtz_f32(float, float, float);
// END INTRINSICS
// END FLOAT

// BEGIN DOUBLE
__device__
__attribute__((const))
double __ocml_acos_f64(double);
__device__
__attribute__((pure))
double __ocml_acosh_f64(double);
__device__
__attribute__((const))
double __ocml_asin_f64(double);
__device__
__attribute__((pure))
double __ocml_asinh_f64(double);
__device__
__attribute__((const))
double __ocml_atan2_f64(double, double);
__device__
__attribute__((const))
double __ocml_atan_f64(double);
__device__
__attribute__((pure))
double __ocml_atanh_f64(double);
__device__
__attribute__((pure))
double __ocml_cbrt_f64(double);
__device__
__attribute__((const))
double __ocml_ceil_f64(double);
__device__
__attribute__((const))
double __ocml_copysign_f64(double, double);
__device__
double __ocml_cos_f64(double);
__device__
__attribute__((pure))
double __ocml_cosh_f64(double);
__device__
double __ocml_cospi_f64(double);
__device__
double __ocml_i0_f64(double);
__device__
double __ocml_i1_f64(double);
__device__
__attribute__((pure))
double __ocml_erfc_f64(double);
__device__
__attribute__((pure))
double __ocml_erfcinv_f64(double);
__device__
__attribute__((pure))
double __ocml_erfcx_f64(double);
__device__
__attribute__((pure))
double __ocml_erf_f64(double);
__device__
__attribute__((pure))
double __ocml_erfinv_f64(double);
__device__
__attribute__((pure))
double __ocml_exp10_f64(double);
__device__
__attribute__((pure))
double __ocml_exp2_f64(double);
__device__
__attribute__((pure))
double __ocml_exp_f64(double);
__device__
__attribute__((pure))
double __ocml_expm1_f64(double);
__device__
__attribute__((const))
double __ocml_fabs_f64(double);
__device__
__attribute__((const))
double __ocml_fdim_f64(double, double);
__device__
__attribute__((const))
double __ocml_floor_f64(double);
__device__
__attribute__((const))
double __ocml_fma_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fmax_f64(double, double);
__device__
__attribute__((const))
double __ocml_fmin_f64(double, double);
__device__
__attribute__((const))
double __ocml_fmod_f64(double, double);
__device__
double __ocml_frexp_f64(double, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
double __ocml_hypot_f64(double, double);
__device__
__attribute__((const))
int __ocml_ilogb_f64(double);
__device__
__attribute__((const))
int __ocml_isfinite_f64(double);
__device__
__attribute__((const))
int __ocml_isinf_f64(double);
__device__
__attribute__((const))
int __ocml_isnan_f64(double);
__device__
double __ocml_j0_f64(double);
__device__
double __ocml_j1_f64(double);
__device__
__attribute__((const))
double __ocml_ldexp_f64(double, int);
__device__
double __ocml_lgamma_f64(double);
__device__
__attribute__((pure))
double __ocml_log10_f64(double);
__device__
__attribute__((pure))
double __ocml_log1p_f64(double);
__device__
__attribute__((pure))
double __ocml_log2_f64(double);
__device__
__attribute__((const))
double __ocml_logb_f64(double);
__device__
__attribute__((pure))
double __ocml_log_f64(double);
__device__
double __ocml_modf_f64(double, __attribute__((address_space(5))) double*);
__device__
__attribute__((const))
double __ocml_nearbyint_f64(double);
__device__
__attribute__((const))
double __ocml_nextafter_f64(double, double);
__device__
__attribute__((const))
double __ocml_len3_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_len4_f64(double, double, double, double);
__device__
__attribute__((pure))
double __ocml_ncdf_f64(double);
__device__
__attribute__((pure))
double __ocml_ncdfinv_f64(double);
__device__
__attribute__((pure))
double __ocml_pow_f64(double, double);
__device__
__attribute__((pure))
double __ocml_pown_f64(double, int);
__device__
__attribute__((pure))
double __ocml_rcbrt_f64(double);
__device__
__attribute__((const))
double __ocml_remainder_f64(double, double);
__device__
double __ocml_remquo_f64(
    double, double, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
double __ocml_rhypot_f64(double, double);
__device__
__attribute__((const))
double __ocml_rint_f64(double);
__device__
__attribute__((const))
double __ocml_rlen3_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_rlen4_f64(double, double, double, double);
__device__
__attribute__((const))
double __ocml_round_f64(double);
__device__
__attribute__((pure))
double __ocml_rsqrt_f64(double);
__device__
__attribute__((const))
double __ocml_scalb_f64(double, double);
__device__
__attribute__((const))
double __ocml_scalbn_f64(double, int);
__device__
__attribute__((const))
int __ocml_signbit_f64(double);
__device__
double __ocml_sincos_f64(double, __attribute__((address_space(5))) double*);
__device__
double __ocml_sincospi_f64(double, __attribute__((address_space(5))) double*);
__device__
double __ocml_sin_f64(double);
__device__
__attribute__((pure))
double __ocml_sinh_f64(double);
__device__
double __ocml_sinpi_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_f64(double);
__device__
double __ocml_tan_f64(double);
__device__
__attribute__((pure))
double __ocml_tanh_f64(double);
__device__
double __ocml_tgamma_f64(double);
__device__
__attribute__((const))
double __ocml_trunc_f64(double);
__device__
double __ocml_y0_f64(double);
__device__
double __ocml_y1_f64(double);

// BEGIN INTRINSICS
__device__
__attribute__((const))
double __ocml_add_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_add_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_add_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_add_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_sqrt_rte_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_rtn_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_rtp_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_rtz_f64(double);
__device__
__attribute__((const))
double __ocml_fma_rte_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fma_rtn_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fma_rtp_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fma_rtz_f64(double, double, double);
// END INTRINSICS
// END DOUBLE

#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__

#if defined(__cplusplus)
    } // extern "C"
#endif
