/**
 * MIT License
 *
 * Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * \file
 * \brief amd_hip_fp8.h header, for AMD fp8 data types
 */

#ifndef _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP8_H_
#define _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP8_H_

#if (defined(__gfx942__) || defined(__gfx1200__) || defined(__gfx1201__) ||                        \
     defined(__gfx950__)) &&                                                                       \
    __HIP_DEVICE_COMPILE__
#define HIP_FP8_CVT_FAST_PATH 1
#else
#define HIP_FP8_CVT_FAST_PATH 0
#endif

#if defined(__gfx942__) && __HIP_DEVICE_COMPILE__
#define HIP_FP8_TYPE_OCP 0
#define HIP_FP8_TYPE_FNUZ 1
#elif (defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx950__)) &&                     \
    __HIP_DEVICE_COMPILE__
#define HIP_FP8_TYPE_OCP 1
#define HIP_FP8_TYPE_FNUZ 0
#else
#define HIP_FP8_TYPE_FNUZ 1
#define HIP_FP8_TYPE_OCP 1
#endif

#if defined(__HIPCC_RTC__)
#if HIP_FP8_TYPE_FNUZ
#define ENABLE_FNUZ_HIPRTC 1
#else
#define ENABLE_FNUZ_HIPRTC 0
#endif
#if HIP_FP8_TYPE_OCP
#define ENABLE_OCP_HIPRTC 1
#else
#define ENABLE_OCP_HIPRTC 0
#endif
#endif

// Include it explicitly for HIPRTC
#include "amd_hip_bf16.h"

#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_common.h>
#include <climits>

#include "host_defines.h"          // __hip_internal::
#include "amd_hip_vector_types.h"  // float2 etc
#include "amd_hip_fp16.h"          // __half_raw
#include "math_fwd.h"              // ocml device functions
#include "hip_assert.h"            // hip assertions
#define __HIP_SCHAR_MAX SCHAR_MAX
#define __HIP_SCHAR_MIN SCHAR_MIN
#define __HIP_UCHAR_MAX UCHAR_MAX
#define __HIP_SHRT_MIN SHRT_MIN
#define __HIP_SHRT_MAX SHRT_MAX
#define __HIP_CHAR_MIN CHAR_MIN
#define __HIP_CHAR_MAX CHAR_MAX
#else
// fp8 header uses all this, since we do not include standard header, we include this
#define __HIP_SCHAR_MAX __SCHAR_MAX__
#define __HIP_SCHAR_MIN (-__SCHAR_MAX__ - 1)
#define __HIP_UCHAR_MAX (__SCHAR_MAX__ * 2 + 1)
#define __HIP_SHRT_MIN (-__SHRT_MAX__ - 1)
#define __HIP_SHRT_MAX __SHRT_MAX__
#ifdef __CHAR_UNSIGNED__ /* -funsigned-char */
#define __HIP_CHAR_MIN 0
#define __HIP_CHAR_MAX __HIP_UCHAR_MAX
#else
#define __HIP_CHAR_MIN __HIP_SCHAR_MIN
#define __HIP_CHAR_MAX __SCHAR_MAX__
#endif
#endif  // !defined(__HIPCC_RTC__)

#if defined(__HIPCC_RTC__)
#define __FP8_HOST_DEVICE__ __device__
#define __FP8_HOST_DEVICE_STATIC__ __FP8_HOST_DEVICE__ static
#else
#define __FP8_HOST_DEVICE__ __host__ __device__
#define __FP8_HOST_DEVICE_STATIC__ __FP8_HOST_DEVICE__ static inline
#endif  // __HIPCC_RTC__

#define __FP8_HOST__ __host__
#define __FP8_HOST_STATIC__ __FP8_HOST__ static inline


#if !defined(__HIPCC_RTC__)
static_assert(CHAR_BIT == 8, "byte size should be of 8 bits");
#endif
static_assert(sizeof(unsigned char) == 1);
static_assert(sizeof(unsigned short int) == 2);
static_assert(sizeof(unsigned int) == 4);

/**
 * \brief Describes FP8 interpretation
 */
enum __hip_fp8_interpretation_t {
  __HIP_E4M3 = 0,      /**< OCP E4M3 */
  __HIP_E5M2 = 1,      /**< OCP E5M2 */
  __HIP_E4M3_FNUZ = 2, /**< Standard FP8*/
  __HIP_E5M2_FNUZ = 3, /**< BF8 */
};

/**
 * \brief Describes saturation behavior
 */
enum __hip_saturation_t {
  __HIP_NOSAT = 0,     /**< No saturation */
  __HIP_SATFINITE = 1, /**< Saturate to finite */
};

/** \typedef __hip_fp8_storage_t
 *
 * \brief type to store single fp8 number
 */
typedef unsigned char __hip_fp8_storage_t;


/** \typedef __hip_fp8x2_storage_t
 *
 * \brief type to store two fp8 numbers
 */
typedef unsigned short int __hip_fp8x2_storage_t;


/** \typedef __hip_fp8x4_storage_t
 *
 * \brief type to store four fp8 numbers
 */
typedef unsigned int __hip_fp8x4_storage_t;


namespace internal {

// Assertions to check for supported conversion types
#define __assert_ocp_support(interp)                                                               \
  {                                                                                                \
    if (interp != __HIP_E4M3 && interp != __HIP_E5M2) {                                            \
      __hip_assert(false && "type is unsupported by current target device");                       \
    }                                                                                              \
  }
#define __assert_fnuz_support(interp)                                                              \
  {                                                                                                \
    if (interp != __HIP_E4M3_FNUZ && interp != __HIP_E5M2_FNUZ) {                                  \
      __hip_assert(false && "type is unsupported by current target device");                       \
    }                                                                                              \
  }

__FP8_HOST_DEVICE_STATIC__ void __is_interpret_supported(__hip_fp8_interpretation_t interp) {
#if __HIP_DEVICE_COMPILE__
#if HIP_FP8_TYPE_OCP
  __assert_ocp_support(interp);
#endif
#if HIP_FP8_TYPE_FNUZ
  __assert_fnuz_support(interp);
#endif
#endif
}

// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_hip_f8_impl.h#L39
// This has been modified to add double types conversion as well
template <typename T, bool is_fnuz>
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t cast_to_f8(T _x, int wm, int we, bool clip = false,
                                                          bool stoch = false,
                                                          unsigned int rng = 0) {
#if defined(__clang__) and defined(__HIP__)
  constexpr bool is_half = __hip_internal::is_same<T, _Float16>::value;
  constexpr bool is_float = __hip_internal::is_same<T, float>::value;
  constexpr bool is_double = __hip_internal::is_same<T, double>::value;
#else   // compiling for host
  constexpr bool is_half = std::is_same<T, _Float16>::value;
  constexpr bool is_float = std::is_same<T, float>::value;
  constexpr bool is_double = std::is_same<T, double>::value;
#endif  // defined(__clang__) and defined(__HIP__)
  static_assert(is_half || is_float || is_double, "Only half, float and double can be cast to f8");

  const int mfmt = (sizeof(T) == 8) ? 52 : ((sizeof(T) == 4) ? 23 : 10);
  unsigned long long x;

  if (sizeof(T) == 8)
    x = reinterpret_cast<unsigned long long&>(_x);
  else if (sizeof(T) == 4)
    x = reinterpret_cast<unsigned int&>(_x);
  else
    x = reinterpret_cast<unsigned short int&>(_x);


  unsigned long long head, mantissa;
  int exponent, bias;
  unsigned int sign;
  unsigned long long fInf, mask;

  if (sizeof(T) == 8) {
    head = x & 0xFFF0000000000000ull;
    mantissa = x & 0xFFFFFFFFFFFFFull;
    exponent = (head >> 52) & 0x7FF;
    sign = head >> 63;
    bias = 1023;
    fInf = 0x7FF0000000000000ull;
    mask = 0x7FFFFFFFFFFFFFFFull;
  } else if (sizeof(T) == 4) {
    head = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head >> 23) & 0xFF;
    sign = head >> 31;
    bias = 127;
    fInf = 0x7F800000;
    mask = 0x7FFFFFFF;
  } else {
    head = x & 0xFC00;
    mantissa = x & 0x3FF;
    exponent = (head >> 10) & 0x1F;
    sign = head >> 15;
    bias = 15;
    fInf = 0x7C00;
    mask = 0x7FFF;
  }
  unsigned int signed_inf = 0;
  unsigned int nan = 0;
  if (is_fnuz) {
    signed_inf = clip ? ((sign << 7) + 0x7f) : 0x80;
    nan = 0x80;
  } else {
    if (we == 4) {  // e4m3
      signed_inf = (sign << 7) + (clip ? 0x7e : 0x7f);
    } else {  // e5m2
      signed_inf = (sign << 7) + (clip ? 0x7b : 0x7c);
    }
    nan = (sign << 7) + 0x7f;
  }
  // Max values
  unsigned long long ifmax = 0;
  if (sizeof(T) == 8) {
    if (we == 5) {  // 57344
      ifmax = 0x40EC000000000000ull;
    } else {
      if (is_fnuz) {  // 240
        ifmax = 0x406E000000000000ull;
      } else {  // 448
        ifmax = 0x407C000000000000ull;
      }
    }
  } else if (sizeof(T) == 4) {
    if (we == 5) {
      ifmax = 0x47600000;
    } else {
      if (is_fnuz) {
        ifmax = 0x43700000;
      } else {
        ifmax = 0x43E00000;
      }
    }
  } else {
    if (we == 5) {
      ifmax = 0x7B00;
    } else {
      if (is_fnuz) {
        ifmax = 0x5B80;
      } else {
        ifmax = 0x5F00;
      }
    }
  }
  // Deal with inf and NaNs
  if ((x & fInf) == fInf) {
    if (is_fnuz || we == 4) return nan;  // funz and OCP E4M3 has no INF
    if (mantissa != 0) return nan;       // NaN
    return sign == 0 ? 0x7C : 0xFC;      // E5M2 Inf
  }

  if ((x & mask) > ifmax) {
    return signed_inf;
  }

  if (x == 0) {
    return 0;
  }

  // First need to check if it is normal or denorm as there is a difference of implict 1
  // Then need to adjust the exponent to align with the F8 exponent, in the meanwhile, shift
  // The mantissa. Then for stochastic rounding, add rng to mantissa and truncate. And for
  // RNE, no need to add rng. Then probably need to check whether there is carry and adjust
  // exponent and mantissa again

  // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent bits
  const int f8_bias = (1 << (we - 1)) - 1 + (is_fnuz ? 1 : 0);
  const int f8_denormal_act_exponent = 1 - f8_bias;  // actual exponent of f8 denormal
  // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
  // f8_exponent is the converted f8 exponent with bias encoding
  // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
  // the difference needs to be adjusted and mantissa shifted
  int act_exponent, f8_exponent, exponent_diff;

  if (exponent == 0) {  // fp32/fp16 is in denormal.
    /* fp32 denormal is below 2^-127 so it is usually not a concern here, we mostly concern fp16
here. In this case, f8 is usually in denormal. But there could be exceptions. fp16 denormal has
exponent bias 15 while bf8 with NANOO has exponent bias 16. It means that there are some numbers in
fp16 denormal but they are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15. fp16 numbers
where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8 (NANOO) normal. In
this case, the fp16 mantissa should be shift left by 1  */
    act_exponent = exponent - bias + 1;
    exponent_diff = f8_denormal_act_exponent -
                    act_exponent;  // actual exponent is exponent-bias+1 as it is denormal
  } else {                         // fp32/fp16 is normal with implicit 1
    act_exponent = exponent - bias;
    if (act_exponent <= f8_denormal_act_exponent) {
      /* This is the case where fp32/fp16 is normal but it is in f8 denormal range.
For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
actual exponent is -7, it is actually larger due to the implict 1,
Therefore it needs to be adjust to -6 and mantissa shift right by 1.
So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
      exponent_diff = f8_denormal_act_exponent - act_exponent;
    } else {              // both fp32/fp16 and f8 are in normal range
      exponent_diff = 0;  // exponent_diff=0 does not mean there is no difference for this case,
                          // act_exponent could be larger. Just that it does not need shift mantissa
    }
    mantissa += (1ull << mfmt);  // Add the implicit 1 into mantissa
  }

  bool midpoint = (mantissa & ((1ull << (mfmt - wm + exponent_diff)) - 1)) ==
                  (1ull << (mfmt - wm + exponent_diff - 1));
  /* This part is a bit tricky. The judgment of whether it is a tie needs to be done before we shift
right as shift right could rip off some residual part and make something not midpoint look like
midpoint. For example, the fp16 number 0x1002 (0 00100 0000000010), it is larger than midpoint, but
after shift right by 4 bits, it would look like midpoint.
*/

  if (exponent_diff > 0)
    mantissa >>= exponent_diff;
  else if (exponent_diff == -1)
    mantissa <<= -exponent_diff;
  bool implicit_one = mantissa & (1ull << mfmt);
  // if there is no implict 1, it  means the f8 is denormal and need to adjust to denorm exponent
  f8_exponent =
      (act_exponent + exponent_diff) /*actual f8 exponent*/ + f8_bias - (implicit_one ? 0 : 1);

  // Now we have the exponent and mantissa adjusted
  unsigned long long drop_mask = (1ull << (mfmt - wm)) - 1;
  bool odd =
      mantissa & (1ull << (mfmt - wm));  // if the least significant bit that is not truncated is 1
  mantissa +=
      (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1ull) : mantissa)) & drop_mask;

  // Now we deal with overflow
  if (f8_exponent == 0) {
    if ((1ull << mfmt) & mantissa) {
      f8_exponent = 1;  // denormal overflow to become normal, promote exponent
    }
  } else {
    if ((1ull << (mfmt + 1)) & mantissa) {
      mantissa >>= 1;
      f8_exponent++;
    }
  }

  mantissa >>= (mfmt - wm);

  // above range: quantize to maximum possible float of the same sign
  const int max_exp = (1 << we) - 1;
  if (f8_exponent > max_exp) {
    if (clip) {
      mantissa = (1 << wm) - 1;
      f8_exponent = max_exp;
    } else {
      return signed_inf;
    }
  }

  if (f8_exponent == 0 && mantissa == 0) return is_fnuz ? 0 : (sign << 7);
  mantissa &= (1 << wm) - 1;
  return (sign << 7) | (f8_exponent << wm) | mantissa;
}
// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_hip_f8_impl.h#L220
// This has been modified to handle double types as well
template <typename T, bool is_fnuz> __FP8_HOST_DEVICE_STATIC__ T cast_from_f8(__hip_fp8_storage_t x,
                                                                              int wm, int we,
                                                                              bool clip = false) {
#if defined(__clang__) and defined(__HIP__)
  constexpr bool is_half = __hip_internal::is_same<T, _Float16>::value;
  constexpr bool is_float = __hip_internal::is_same<T, float>::value;
  constexpr bool is_double = __hip_internal::is_same<T, double>::value;
#else
  constexpr bool is_half = std::is_same<T, _Float16>::value;
  constexpr bool is_float = std::is_same<T, float>::value;
  constexpr bool is_double = std::is_same<T, double>::value;
#endif  // defined(__clang__) and defined(__HIP__)
  static_assert(is_half || is_float || is_double, "only half, float and double are supported");

  constexpr int weo = is_half ? 5 : (is_float ? 8 : 11);
  constexpr int wmo = is_half ? 10 : (is_float ? 23 : 52);

  T fInf, fNegInf, fNaN, fNeg0, fmax, fmin;
  if (is_half) {
    const unsigned short int ihInf = 0x7C00;
    const unsigned short int ihNegInf = 0xFC00;
    const unsigned short int ihNaN = 0x7C01;
    const unsigned short int ihNeg0 = 0x8000;
    /* Max number in e5m2 57344*/
    const unsigned short int ifmax = 0x7B00;
    const unsigned short int ifmin = 0xFB00;
    fInf = reinterpret_cast<const _Float16&>(ihInf);
    fNegInf = reinterpret_cast<const _Float16&>(ihNegInf);
    fNaN = reinterpret_cast<const _Float16&>(ihNaN);
    fNeg0 = reinterpret_cast<const _Float16&>(ihNeg0);
    fmax = reinterpret_cast<const _Float16&>(ifmax);
    fmin = reinterpret_cast<const _Float16&>(ifmin);
  } else if (is_float) {
    const unsigned int ifInf = 0x7F800000;
    const unsigned int ifNegInf = 0xFF800000;
    const unsigned int ifNaN = 0x7F800001;
    const unsigned int ifNeg0 = 0x80000000;
    /* Max number in e5m2 57344*/
    const unsigned int ifmax = 0x47600000;
    const unsigned int ifmin = 0xC7600000;
    fInf = reinterpret_cast<const float&>(ifInf);
    fNegInf = reinterpret_cast<const float&>(ifNegInf);
    fNaN = reinterpret_cast<const float&>(ifNaN);
    fNeg0 = reinterpret_cast<const float&>(ifNeg0);
    fmax = reinterpret_cast<const float&>(ifmax);
    fmin = reinterpret_cast<const float&>(ifmin);
  } else if (is_double) {
    const unsigned long long ifInf = 0x7FF0000000000000ull;
    const unsigned long long ifNegInf = 0xFFF0000000000000ull;
    const unsigned long long ifNaN = 0x7FF0000000000001ull;
    const unsigned long long ifNeg0 = 0x8000000000000000ull;
    /* Max number in e5m2 57344*/
    const unsigned long long ifmax = 0x40EC000000000000ull;
    const unsigned long long ifmin = 0xC0EC000000000000ull;
    fInf = reinterpret_cast<const double&>(ifInf);
    fNegInf = reinterpret_cast<const double&>(ifNegInf);
    fNaN = reinterpret_cast<const double&>(ifNaN);
    fNeg0 = reinterpret_cast<const double&>(ifNeg0);
    fmax = reinterpret_cast<const double&>(ifmax);
    fmin = reinterpret_cast<const double&>(ifmin);
  }

  if (x == 0) {
    return 0;
  }

  unsigned long long sign = x >> 7;
  unsigned long long mantissa = x & ((1 << wm) - 1);
  int exponent = (x & 0x7F) >> wm;
  if (is_fnuz) {
    if (x == 0x80) {
      return fNaN;
    }
  } else {
    if (x == 0x80) {
      return fNeg0;
    }
    if (we == 4) {  // e4m3
      if ((x & 0x7F) == 0x7F) {
        return fNaN;
      }
    } else if ((x & 0x7C) == 0x7C) {  // e5m2 NaN/Inf
      if ((x & 0x3) == 0) {           // Inf
        if (clip) {
          return sign ? fmin : fmax;
        }
        return sign ? fNegInf : fInf;
      }
      return fNaN;
    }
  }

#if defined(__clang__) and defined(__HIP__)
  typename __hip_internal::conditional<
      sizeof(T) == 2, unsigned short int,
      typename __hip_internal::conditional<sizeof(T) == 4, unsigned int,
                                           unsigned long long>::type>::type retval;
#else
  typename std::conditional<sizeof(T) == 2, unsigned short int,
                            typename std::conditional<sizeof(T) == 4, unsigned int,
                                                      unsigned long long>::type>::type retval;
#endif

  if (we == 5 && is_half && !is_fnuz) {
    retval = x << 8;
    return reinterpret_cast<const T&>(retval);
  }

  const int exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (is_fnuz ? 1 : 0);

  // subnormal input
  if (exponent == 0) {
#if __HIP_DEVICE_COMPILE__
    // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
    int sh = 1 + __clz(mantissa) - (32 - wm);
#else
    int sh = 1 + __builtin_clz(mantissa) - (32 - wm);
#endif
    mantissa <<= sh;
    exponent += 1 - sh;
    mantissa &= ((1ull << wm) - 1);
  }
  exponent += exp_low_cutoff - 1;
  mantissa <<= wmo - wm;

  // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
  if (exponent <= 0) {
    mantissa |= 1ull << wmo;
    mantissa >>= 1 - exponent;
    exponent = 0;
  }

  if (sizeof(T) == 2)
    retval = (sign << 15) | (exponent << 10) | mantissa;
  else if (sizeof(T) == 4)
    retval = (sign << 31) | (exponent << 23) | mantissa;
  else
    retval = (sign << 63) | (static_cast<unsigned long long>(exponent) << 52) | mantissa;
  return reinterpret_cast<const T&>(retval);
}

#if HIP_FP8_CVT_FAST_PATH
// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_float8.h#L79
template <bool stochastic_rounding = false>
static __device__ __hip_fp8_storage_t cast_to_f8_from_f32(float v, bool saturate,
                                                          __hip_fp8_interpretation_t interpret,
                                                          unsigned int rng = 0) {
  __hip_fp8_storage_t i8data;
  union {
    float fval;
    unsigned int i32val;
    unsigned char i8val[4];  // NOTE: not endian independent
  } val;

  unsigned int ival = 0;
  val.fval = v;

  if (saturate) {
    if (interpret == __HIP_E4M3_FNUZ) {
      if ((val.i32val & 0x7F800000) != 0x7F800000) {  /// propagate NAN/INF, no clipping
        val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
      }
    } else if (interpret == __HIP_E4M3) {             // OCP type
      if ((val.i32val & 0x7F800000) != 0x7F800000) {  /// propagate NAN/INF, no clipping
        val.fval = __builtin_amdgcn_fmed3f(val.fval, 448.0, -448.0);
      }
    } else {
      if ((val.i32val & 0x7F800000) != 0x7F800000) {  /// propagate NAN/INF, no clipping
        val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);
      }
    }
  }

  if (stochastic_rounding) {
    ival = (interpret == __HIP_E4M3_FNUZ) || (interpret == __HIP_E4M3)
               ? __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0)
               : __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0);  // 0 pos
    val.i32val = ival;
    i8data = val.i8val[0];  // little endian
  } else {                  // RNE CVT
    ival =
        (interpret == __HIP_E4M3_FNUZ) || (interpret == __HIP_E4M3)
            ? __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival, false)
            : __builtin_amdgcn_cvt_pk_bf8_f32(val.fval, val.fval, ival, false);  // false -> WORD0
    val.i32val = ival;
    i8data = val.i8val[0];
  }
  return i8data;
}

static __device__ __hip_fp8x2_storage_t
cast_to_f8x2_from_f32x2(float2 v, bool saturate, __hip_fp8_interpretation_t interpret) {
  union {
    static_assert(sizeof(float2) == sizeof(unsigned int[2]), "size mismatch");
    static_assert(sizeof(float2) == sizeof(unsigned short[4]), "size mismatch");
    float2 fval;
    unsigned int i32val[2];
    unsigned short i16val[4];
  } f2val;

  f2val.fval = v;

  if (saturate) {  /// propagate NAN/INF, no clipping
    if (interpret == __HIP_E4M3_FNUZ) {
      if ((f2val.i32val[0] & 0x7F800000) != 0x7F800000) {
        f2val.fval.x = __builtin_amdgcn_fmed3f(f2val.fval.x, 240.0, -240.0);
      }
      if ((f2val.i32val[1] & 0x7F800000) != 0x7F800000) {
        f2val.fval.y = __builtin_amdgcn_fmed3f(f2val.fval.x, 240.0, -240.0);
      }
    } else if (interpret == __HIP_E4M3) {
      if ((f2val.i32val[0] & 0x7F800000) != 0x7F800000) {
        f2val.fval.x = __builtin_amdgcn_fmed3f(f2val.fval.x, 448.0, -448.0);
      }
      if ((f2val.i32val[1] & 0x7F800000) != 0x7F800000) {
        f2val.fval.y = __builtin_amdgcn_fmed3f(f2val.fval.x, 448.0, -448.0);
      }
    } else {
      if ((f2val.i32val[0] & 0x7F800000) != 0x7F800000) {
        f2val.fval.x = __builtin_amdgcn_fmed3f(f2val.fval.x, 57344.0, -57344.0);
      }
      if ((f2val.i32val[1] & 0x7F800000) != 0x7F800000) {
        f2val.fval.y = __builtin_amdgcn_fmed3f(f2val.fval.x, 57344.0, -57344.0);
      }
    }
  }

  f2val.i32val[0] = (interpret == __HIP_E4M3_FNUZ) || (interpret == __HIP_E4M3)
                        ? __builtin_amdgcn_cvt_pk_fp8_f32(v.x, v.y, 0, false)
                        : __builtin_amdgcn_cvt_pk_bf8_f32(v.x, v.y, 0, false);

  return static_cast<__hip_fp8x2_storage_t>(f2val.i16val[0]);
}

static __device__ float cast_to_f32_from_f8(__hip_fp8_storage_t v,
                                            __hip_fp8_interpretation_t interpret) {
  union {
    unsigned int i32val;
    unsigned char i8val[4];
  } val;
  val.i8val[0] = v;

  float fval = (interpret == __HIP_E4M3_FNUZ) || (interpret == __HIP_E4M3)
                   ? __builtin_amdgcn_cvt_f32_fp8(val.i32val, 0)
                   : __builtin_amdgcn_cvt_f32_bf8(val.i32val, 0);
  return fval;
}

static __device__ float2 cast_to_f32x2_from_f8x2(__hip_fp8x2_storage_t v,
                                                 __hip_fp8_interpretation_t interpret) {
  union {
    unsigned int i32val;
    unsigned short i16val[2];
  } val;
  val.i16val[0] = v;

  auto f2 = (interpret == __HIP_E4M3_FNUZ) || (interpret == __HIP_E4M3)
                ? __builtin_amdgcn_cvt_pk_f32_fp8(val.i32val, false)
                : __builtin_amdgcn_cvt_pk_f32_bf8(val.i32val, false);
  return float2{f2[0], f2[1]};
}
#endif  // HIP_FP8_CVT_FAST_PATH

/* For fp8 fnuz types, finite and NaN values are supported. Zero is unsigned.
Inf are not supported. This gives us one additional number to represent.
NaN are represented by 1-0000-000 or 1-00000-00 */
__FP8_HOST_DEVICE_STATIC__ bool hip_fp8_fnuz_is_nan(__hip_fp8_storage_t a) {
  return static_cast<unsigned char>(a) == 0x80;
}

__FP8_HOST_DEVICE_STATIC__ bool hip_fp8_ocp_is_nan(__hip_fp8_storage_t a,
                                                   const __hip_fp8_interpretation_t type) {
  return (type == __HIP_E4M3)   ? ((a & 0x7f) == 0x7f)
         : (type == __HIP_E5M2) ? ((a & 0x7f) > 0x7c)
                                : false;
}

__FP8_HOST_DEVICE_STATIC__ bool hip_fp8_ocp_is_inf(__hip_fp8_storage_t a,
                                                   const __hip_fp8_interpretation_t type) {
  return (type == __HIP_E5M2) ? (a & 0x7f) == 0x7c : false;
}

}  // namespace internal

/**
 * \brief convert float to @p __hip_fp8_storage_t
 *
 * \param f float number
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_float_to_fp8(
    const float f, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
  return internal::cast_to_f8_from_f32<false>(f, sat == __HIP_SATFINITE, interp);
#else
#if HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_float_to_fp8(
    const float f, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8_storage_t __hip_cvt_float_to_fp8(
    const float f, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  if (interp == __HIP_E4M3_FNUZ || interp == __HIP_E5M2_FNUZ) {
    int we = interp == __HIP_E4M3_FNUZ ? 4 : 5;
    int wm = interp == __HIP_E4M3_FNUZ ? 3 : 2;
    return internal::cast_to_f8<float, true>(f, wm, we, sat == __HIP_SATFINITE);
  } else {
    int we = interp == __HIP_E4M3 ? 4 : 5;
    int wm = interp == __HIP_E4M3 ? 3 : 2;
    return internal::cast_to_f8<float, false>(f, wm, we, sat == __HIP_SATFINITE);
  }
#endif  // HIP_FP8_CVT_FAST_PATH
}


/**
 * \brief convert float2 to @p __hip_fp8x2_storage_t
 *
 * \param f2 float2 number
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8x2_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_float2_to_fp8x2(
    const float2 f2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
  return internal::cast_to_f8x2_from_f32x2(f2, sat == __HIP_SATFINITE, interp);
#else
#if HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_float2_to_fp8x2(
    const float2 f2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8x2_storage_t __hip_cvt_float2_to_fp8x2(
    const float2 f2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  return static_cast<__hip_fp8x2_storage_t>(
      static_cast<unsigned short int>(__hip_cvt_float_to_fp8(f2.y, sat, interp)) << 8 |
      static_cast<unsigned short int>(__hip_cvt_float_to_fp8(f2.x, sat, interp)));
#endif  // HIP_FP8_CVT_FAST_PATH
}

/**
 * \brief convert double to @p __hip_fp8_storage_t
 *
 * \param d double val
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_double_to_fp8(
    const double d, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_double_to_fp8(
    const double d, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8_storage_t __hip_cvt_double_to_fp8(
    const double d, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  if (interp == __HIP_E4M3_FNUZ || interp == __HIP_E5M2_FNUZ) {
    int we = interp == __HIP_E4M3_FNUZ ? 4 : 5;
    int wm = interp == __HIP_E4M3_FNUZ ? 3 : 2;
    return internal::cast_to_f8<double, true>(d, wm, we, sat == __HIP_SATFINITE);
  } else {
    int we = interp == __HIP_E4M3 ? 4 : 5;
    int wm = interp == __HIP_E4M3 ? 3 : 2;
    return internal::cast_to_f8<double, false>(d, wm, we, sat == __HIP_SATFINITE);
  }
}

/**
 * \brief convert double2 to @p __hip_fp8x2_storage_t
 *
 * \param d2 double2 val
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8x2_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_double2_to_fp8x2(
    const double2 d2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_double2_to_fp8x2(
    const double2 d2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8x2_storage_t __hip_cvt_double2_to_fp8x2(
    const double2 d2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  return static_cast<__hip_fp8x2_storage_t>(
      static_cast<unsigned short int>(__hip_cvt_double_to_fp8(d2.y, sat, interp)) << 8 |
      static_cast<unsigned short int>(__hip_cvt_double_to_fp8(d2.x, sat, interp)));
}

/**
 * \brief convert __hip_bfloat16_raw to @p __hip_fp8_storage_t
 *
 * \param hr __hip_bfloat16_raw val
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t
__hip_cvt_bfloat16raw_to_fp8(const __hip_bfloat16_raw hr, const __hip_saturation_t sat,
                             const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t
__hip_cvt_bfloat16raw_to_fp8(const __hip_bfloat16_raw hr, const __hip_saturation_t sat,
                             const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8_storage_t
__hip_cvt_bfloat16raw_to_fp8(const __hip_bfloat16_raw hr, const __hip_saturation_t sat,
                             const __hip_fp8_interpretation_t interp) {
#endif
  float fval = __hip_bfloat16(hr);
  return __hip_cvt_float_to_fp8(fval, sat, interp);
}

/**
 * \brief convert double2 to @p __hip_fp8x2_storage_t
 *
 * \param hr __hip_bfloat162_raw value
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8x2_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t
__hip_cvt_bfloat16raw2_to_fp8x2(const __hip_bfloat162_raw hr, const __hip_saturation_t sat,
                                const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t
__hip_cvt_bfloat16raw2_to_fp8x2(const __hip_bfloat162_raw hr, const __hip_saturation_t sat,
                                const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8x2_storage_t
__hip_cvt_bfloat16raw2_to_fp8x2(const __hip_bfloat162_raw hr, const __hip_saturation_t sat,
                                const __hip_fp8_interpretation_t interp) {
#endif
  float2 f2 = __hip_bfloat162(hr);
  return __hip_cvt_float2_to_fp8x2(f2, sat, interp);
}

/**
 * \brief convert @p __hip_fp8_storage_t to __half_raw
 *
 * \param x __hip_fp8_storage_t val
 * \param interp interpretation of fp8
 * \return __half_raw
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __half_raw
__hip_cvt_fp8_to_halfraw(const __hip_fp8_storage_t x, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __half_raw
__hip_cvt_fp8_to_halfraw(const __hip_fp8_storage_t x, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __half_raw __hip_cvt_fp8_to_halfraw(const __hip_fp8_storage_t x,
                                                        const __hip_fp8_interpretation_t interp) {
#endif
  if (interp == __HIP_E4M3_FNUZ || interp == __HIP_E5M2_FNUZ) {
    unsigned int we = interp == __HIP_E4M3_FNUZ ? 4 : 5;
    unsigned int wm = interp == __HIP_E4M3_FNUZ ? 3 : 2;
    return __half_raw{internal::cast_from_f8<_Float16, true>(x, wm, we)};
  } else {
    unsigned int we = interp == __HIP_E4M3 ? 4 : 5;
    unsigned int wm = interp == __HIP_E4M3 ? 3 : 2;
    return __half_raw{internal::cast_from_f8<_Float16, false>(x, wm, we)};
  }
}

/**
 * \brief convert @p __hip_fp8x2_storage_t to __half2_raw
 *
 * \param x __hip_fp8x2_storage_t val
 * \param interp interpretation of fp8
 * \return __half2_raw
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp8x2_to_halfraw2(
    const __hip_fp8x2_storage_t x, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp8x2_to_halfraw2(
    const __hip_fp8x2_storage_t x, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __half2_raw __hip_cvt_fp8x2_to_halfraw2(
    const __hip_fp8x2_storage_t x, const __hip_fp8_interpretation_t interp) {
#endif
  __half2 ret(static_cast<__half>(
                  __hip_cvt_fp8_to_halfraw(static_cast<__hip_fp8_storage_t>(x & 0xFF), interp)),
              static_cast<__half>(
                  __hip_cvt_fp8_to_halfraw(static_cast<__hip_fp8_storage_t>(x >> 8), interp)));
  return static_cast<__half2_raw>(ret);
}

/**
 * \brief convert __half_raw to @p __hip_fp8_storage_t
 *
 * \param x __half_raw value
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_halfraw_to_fp8(
    const __half_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_halfraw_to_fp8(
    const __half_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8_storage_t __hip_cvt_halfraw_to_fp8(
    const __half_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  return __hip_cvt_float_to_fp8(__half2float(__half(x)), sat, interp);
}

/**
 * \brief convert __half2_raw to @p __hip_fp8x2_storage_t
 *
 * \param x __half2_raw value
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8x2_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_halfraw2_to_fp8x2(
    const __half2_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_halfraw2_to_fp8x2(
    const __half2_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8x2_storage_t __hip_cvt_halfraw2_to_fp8x2(
    const __half2_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  return __hip_cvt_float2_to_fp8x2(__half22float2(__half2(x)), sat, interp);
}

/**
 * \brief struct representing single fp8 number with e4m3 interpretation
 *
 */

#if !defined(ENABLE_FNUZ_HIPRTC) || ENABLE_FNUZ_HIPRTC
struct __hip_fp8_e4m3_fnuz {
  __hip_fp8_storage_t __x;  //! raw storage of fp8 number
  constexpr static __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  constexpr static __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3_FNUZ;
  constexpr static unsigned int __we = 4;
  constexpr static unsigned int __wm = 3;

  // TODO: SWDEV-452411
  // Add cast from unsigned long long, long long to fp8

  /*! create fp8 e4m3 from long */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const long int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from short int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const short int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from unsigned long */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const unsigned long int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const unsigned long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from unsigned int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const unsigned int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const unsigned int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from unsigned short */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const unsigned short int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const unsigned short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from double */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const double f)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const double f)
#endif
      : __x(__hip_cvt_double_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e4m3 from float */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const float f)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const float f)
#endif
      : __x(__hip_cvt_float_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e4m3 from __hip_bfloat16 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const __hip_bfloat16 f)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const __hip_bfloat16 f)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(f), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from __half */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const __half f)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const __half f)
#endif
      : __x(__hip_cvt_halfraw_to_fp8(static_cast<__half_raw>(f), __default_saturation,
                                     __default_interpret)) {
  }

  /*! default construct fp8 e4m3 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz() = default;
#endif

  /*! convert fp8 e4m3 to __half */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __half() const {
#else
  __FP8_HOST__ operator __half() const {
#endif
    return __half(__hip_cvt_fp8_to_halfraw(__x, __default_interpret));
  }

  /*! convert fp8 e4m3 to __hip_bfloat16 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __hip_bfloat16() const {
#else
  __FP8_HOST__ operator __hip_bfloat16() const {
#endif
    float f = *this;
    return __hip_bfloat16(f);
  }

  /*! convert fp8 e4m3 to bool, return false if value is 0, true otherwise */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator bool() const {
#else
  __FP8_HOST__ operator bool() const {
#endif
    // it can be 0x00 (+0.0) since 0x80 will be nan
    return !(static_cast<unsigned short>(__x) == 0);
  }

  /*! convert fp8 e4m3 to char, clamp number to __HIP_CHAR_MIN/__HIP_CHAR_MAX if its out of range */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator char() const {
#else
  __FP8_HOST__ operator char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    auto fval = internal::cast_from_f8<float, true>(__x, __wm, __we);
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_CHAR_MIN) {
      return __HIP_CHAR_MIN;
    } else if (llval >= __HIP_CHAR_MAX) {
      return __HIP_CHAR_MAX;
    }
    return static_cast<char>(fval);
  }

  /*! convert fp8 e4m3 to double */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator double() const {
#else
  __FP8_HOST__ operator double() const {
#endif
    return internal::cast_from_f8<double, true>(__x, __wm, __we);
  }

  /*! convert fp8 e4m3 to float */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float() const {
#else
  __FP8_HOST__ operator float() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
    return internal::cast_to_f32_from_f8(__x, __default_interpret);
#else
    return internal::cast_from_f8<float, true>(__x, __wm, __we);
#endif
  }

  /*! convert fp8 e4m3 to int, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator int() const {
#else
  __FP8_HOST__ operator int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<int>(fval);
  }

  /*! convert fp8 e4m3 to long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator long int() const {
#else
  __FP8_HOST__ operator long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long>(fval);
  }

  /*! convert fp8 e4m3 to long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator long long int() const {
#else
  __FP8_HOST__ operator long long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long long>(fval);
  }

  /*! convert fp8 e4m3 to short int, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator short int() const {
#else
  __FP8_HOST__ operator short int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SHRT_MIN) {
      return __HIP_SHRT_MIN;
    } else if (llval >= __HIP_SHRT_MAX) {
      return __HIP_SHRT_MAX;
    }
    return static_cast<short>(fval);
  }

  /*! convert fp8 e4m3 to signed char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator signed char() const {
#else
  __FP8_HOST__ operator signed char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SCHAR_MIN) {
      return __HIP_SCHAR_MIN;
    } else if (llval >= __HIP_SCHAR_MAX) {
      return __HIP_SCHAR_MAX;
    }
    return static_cast<signed char>(fval);
  }

  /*! convert fp8 e4m3 to unsigned char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned char() const {
#else
  __FP8_HOST__ operator unsigned char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    } else if (llval >= __HIP_UCHAR_MAX) {
      return __HIP_UCHAR_MAX;
    }
    return static_cast<unsigned char>(fval);
  }

  /*! convert fp8 e4m3 to unsigned int, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned int() const {
#else
  __FP8_HOST__ operator unsigned int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned int>(fval);
  }

  /*! convert fp8 e4m3 to unsigned long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned long int() const {
#else
  __FP8_HOST__ operator unsigned long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long>(fval);
  }

  /*! convert fp8 e4m3 to long long int, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned long long int() const {
#else
  __FP8_HOST__ operator unsigned long long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long long>(fval);
  }

  /*! convert fp8 e4m3 to unsigned short, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned short int() const {
#else
  __FP8_HOST__ operator unsigned short int() const {
#endif
      if (internal::hip_fp8_fnuz_is_nan(__x)) {
        return 0;
}

float fval = *this;
auto llval = static_cast<long long>(fval);
if (llval <= 0) {
  return 0;
}
return static_cast<unsigned short>(fval);
}
}
;

/**
 * \brief struct representing two fp8 numbers with e4m3 interpretation
 *
 */
struct __hip_fp8x2_e4m3_fnuz {
  __hip_fp8x2_storage_t __x;  //! raw storage of two fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3_FNUZ;
  static constexpr unsigned int __we = 4;
  static constexpr unsigned int __wm = 3;

  /*! create fp8x2 e4m3 type from double2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3_fnuz(const double2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3_fnuz(const double2 val)
#endif
      : __x(__hip_cvt_double2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from float2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3_fnuz(const float2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3_fnuz(const float2 val)
#endif
      : __x(__hip_cvt_float2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from __hip_bfloat162 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3_fnuz(const __hip_bfloat162 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3_fnuz(const __hip_bfloat162 val)
#endif
      : __x(__hip_cvt_bfloat16raw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3_fnuz(const __half2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3_fnuz(const __half2 val)
#endif
      : __x(__hip_cvt_halfraw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! Default construct of fp8x2 e4m3 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8x2_e4m3_fnuz() = default;
#endif

  /*! convert fp8x2 e4m3 to __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __half2() const {
#else
  __FP8_HOST__ operator __half2() const {
#endif
    return __half2(__hip_cvt_fp8x2_to_halfraw2(__x, __default_interpret));
  }

  /*! convert fp8x2 e4m3 to float2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float2() const {
#else
  __FP8_HOST__ operator float2() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
      return internal::cast_to_f32x2_from_f8x2(__x, __default_interpret);
#else
    return float2(internal::cast_from_f8<float, true>(static_cast<__hip_fp8_storage_t>(__x & 0xFF),
                                                      __wm, __we),
                  internal::cast_from_f8<float, true>(static_cast<__hip_fp8_storage_t>(__x >> 8),
                                                      __wm, __we));
#endif
}
}
;

/**
 * \brief struct representing four fp8 numbers with e4m3 interpretation
 *
 */
struct __hip_fp8x4_e4m3_fnuz {
  __hip_fp8x4_storage_t __x;  //! raw storage of four fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3_FNUZ;
  static constexpr unsigned int __we = 4;
  static constexpr unsigned int __wm = 3;

  /*! create fp8x4 e4m3 type from double4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3_fnuz(const double4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3_fnuz(const double4 val)
#endif
      : __x{reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))} {
  }

  /*! create fp8x4 e4m3 type from float4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3_fnuz(const float4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3_fnuz(const float4 val)
#endif
      : __x{reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))} {
  }

  /*! create fp8x4 e4m3 type from two __hip_bfloat162 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3_fnuz(const __hip_bfloat162 low, const __hip_bfloat162 high)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3_fnuz(const __hip_bfloat162 low, const __hip_bfloat162 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(static_cast<unsigned int>(
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(high, __default_saturation, __default_interpret)) |
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(low, __default_saturation, __default_interpret))
                << 16))) {
  }

  /*! create fp8x4 e4m3 type from two __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3_fnuz(const __half2 low, const __half2 high)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3_fnuz(const __half2 low, const __half2 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          high, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          low, __default_saturation, __default_interpret))
                                          << 16))) {
  }

  /*! Default construct fp8x4 e4m3 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8x4_e4m3_fnuz() = default;
#endif

  /*! convert fp8x4 e4m3 to float4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float4() const {
#else
  __FP8_HOST__ operator float4() const {
#endif
      auto x = __x;                                                // bypass const
  auto fp8x2_low = *reinterpret_cast<__hip_fp8x2_storage_t*>(&x);  // Little E
  auto fp8x2_high = *(reinterpret_cast<__hip_fp8x2_storage_t*>(&x) + 1);
#if HIP_FP8_CVT_FAST_PATH
  float2 high = internal::cast_to_f32x2_from_f8x2(fp8x2_high, __default_interpret);
  float2 low = internal::cast_to_f32x2_from_f8x2(fp8x2_low, __default_interpret);
#else
    float2 high = float2(internal::cast_from_f8<float, true>(
                             static_cast<__hip_fp8_storage_t>((fp8x2_high << 8) >> 8), __wm, __we),
                         internal::cast_from_f8<float, true>(
                             static_cast<__hip_fp8_storage_t>(fp8x2_high >> 8), __wm, __we));
    float2 low = float2(internal::cast_from_f8<float, true>(
                            static_cast<__hip_fp8_storage_t>((fp8x2_low << 8) >> 8), __wm, __we),
                        internal::cast_from_f8<float, true>(
                            static_cast<__hip_fp8_storage_t>(fp8x2_low >> 8), __wm, __we));
#endif
  return float4(low.x, low.y, high.x, high.y);
}
}
;

/**
 * \brief struct representing one fp8 number with e5m2 interpretation
 *
 */
struct __hip_fp8_e5m2_fnuz {
  __hip_fp8_storage_t __x;  //! raw storage of one fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2_FNUZ;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;


  // TODO: SWDEV-452411
  // Add cast from unsigned long long, long long to fp8

  /*! create fp8 e5m2 type from long */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const long int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from short int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const short int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned long */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const unsigned long int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const unsigned long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const unsigned int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const unsigned int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned short */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const unsigned short int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const unsigned short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from double */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const double f)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const double f)
#endif
      : __x(__hip_cvt_double_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e5m2 type from float */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const float f)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const float f)
#endif
      : __x(__hip_cvt_float_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e5m2 type from __hip_bfloat16 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const __hip_bfloat16 f)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const __hip_bfloat16 f)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(f), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from __hip_bfloat16 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const __half f)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const __half f)
#endif
      : __x(__hip_cvt_halfraw_to_fp8(static_cast<__half_raw>(f), __default_saturation,
                                     __default_interpret)) {
  }

  /*! default construct fp8 e5m2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz() = default;
#endif

  /*! convert fp8 e5m2 to float */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float() const {
#else
  __FP8_HOST__ operator float() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
    return internal::cast_to_f32_from_f8(__x, __default_interpret);
#else
    return internal::cast_from_f8<float, true>(__x, __wm, __we);
#endif
  }

  /*! convert fp8 e5m2 to __half */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __half() const {
#else
  __FP8_HOST__ operator __half() const {
#endif
    return __half(__hip_cvt_fp8_to_halfraw(__x, __default_interpret));
  }

  /*! convert fp8 e5m2 to __hip_bfloat16 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __hip_bfloat16() const {
#else
  __FP8_HOST__ operator __hip_bfloat16() const {
#endif
    float f = *this;
    return __hip_bfloat16(f);
  }

  /*! convert fp8 e4m3 to bool, return false if value is 0, true otherwise */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator bool() const {
#else
  __FP8_HOST__ operator bool() const {
#endif
    // it can be 0x00 (+0.0) since 0x80 will be nan
    return !(static_cast<unsigned short>(__x) == 0);
  }

  /*! convert fp8 e5m2 to char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator char() const {
#else
  __FP8_HOST__ operator char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_CHAR_MIN) {
      return __HIP_CHAR_MIN;
    } else if (llval >= __HIP_CHAR_MAX) {
      return __HIP_CHAR_MAX;
    }
    return static_cast<char>(fval);
  }

  /*! convert fp8 e5m2 to double */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator double() const {
#else
  __FP8_HOST__ operator double() const {
#endif
    return internal::cast_from_f8<double, true>(__x, __wm, __we);
  }

  /*! convert fp8 e5m2 to int, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator int() const {
#else
  __FP8_HOST__ operator int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<int>(fval);
  }

  /*! convert fp8 e5m2 to long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator long int() const {
#else
  __FP8_HOST__ operator long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long>(fval);
  }

  /*! convert fp8 e5m2 to long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator long long int() const {
#else
  __FP8_HOST__ operator long long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long long>(fval);
  }

  /*! convert fp8 e5m2 to short, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator short int() const {
#else
  __FP8_HOST__ operator short int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SHRT_MIN) {
      return __HIP_SHRT_MIN;
    } else if (llval >= __HIP_SHRT_MAX) {
      return __HIP_SHRT_MAX;
    }
    return static_cast<short>(fval);
  }

  /*! convert fp8 e5m2 to signed char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator signed char() const {
#else
  __FP8_HOST__ operator signed char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SCHAR_MIN) {
      return __HIP_SCHAR_MIN;
    } else if (llval >= __HIP_SCHAR_MAX) {
      return __HIP_SCHAR_MAX;
    }
    return static_cast<signed char>(fval);
  }

  /*! convert fp8 e5m2 to unsigned char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned char() const {
#else
  __FP8_HOST__ operator unsigned char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    } else if (llval >= __HIP_UCHAR_MAX) {
      return __HIP_UCHAR_MAX;
    }
    return static_cast<unsigned char>(fval);
  }

  /*! convert fp8 e5m2 to unsigned int, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned int() const {
#else
  __FP8_HOST__ operator unsigned int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned int>(fval);
  }

  /*! convert fp8 e5m2 to unsigned long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned long int() const {
#else
  __FP8_HOST__ operator unsigned long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long>(fval);
  }

  /*! convert fp8 e5m2 to unsigned long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned long long int() const {
#else
  __FP8_HOST__ operator unsigned long long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long long>(fval);
  }

  /*! convert fp8 e5m2 to unsigned short, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned short int() const {
#else
  __FP8_HOST__ operator unsigned short int() const {
#endif
      if (internal::hip_fp8_fnuz_is_nan(__x)) {
        return 0;
}

float fval = *this;
auto llval = static_cast<long long>(fval);
if (llval <= 0) {
  return 0;
}
return static_cast<unsigned short>(fval);
}
}
;

/**
 * \brief struct representing two fp8 numbers with e5m2 interpretation
 *
 */
struct __hip_fp8x2_e5m2_fnuz {
  __hip_fp8x2_storage_t __x;  //! raw storage of two fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2_FNUZ;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;

  /*! create fp8x2 e5m2 type from double2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2_fnuz(const double2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2_fnuz(const double2 val)
#endif
      : __x(__hip_cvt_double2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from float2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2_fnuz(const float2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2_fnuz(const float2 val)
#endif
      : __x(__hip_cvt_float2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from __hip_bfloat162 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2_fnuz(const __hip_bfloat162 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2_fnuz(const __hip_bfloat162 val)
#endif
      : __x(__hip_cvt_bfloat16raw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2_fnuz(const __half2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2_fnuz(const __half2 val)
#endif
      : __x(__hip_cvt_halfraw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! default construct fp8x2 e5m2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8x2_e5m2_fnuz() = default;
#endif

  /*! convert fp8x2 e5m2 to __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __half2() const {
#else
  __FP8_HOST__ operator __half2() const {
#endif
    return __half2(__hip_cvt_fp8x2_to_halfraw2(__x, __default_interpret));
  }

  /*! convert fp8x2 e5m2 to float2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float2() const {
#else
  __FP8_HOST__ operator float2() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
      return internal::cast_to_f32x2_from_f8x2(__x, __default_interpret);
#else
    return float2(internal::cast_from_f8<float, true>(static_cast<__hip_fp8_storage_t>(__x & 0xFF),
                                                      __wm, __we),
                  internal::cast_from_f8<float, true>(static_cast<__hip_fp8_storage_t>(__x >> 8),
                                                      __wm, __we));
#endif
}
}
;

/**
 * \brief struct representing four fp8 numbers with e5m2 interpretation
 *
 */
struct __hip_fp8x4_e5m2_fnuz {
  __hip_fp8x4_storage_t __x;  //! raw storage of four fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2_FNUZ;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;

  /*! create fp8x4 e5m2 type from double4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2_fnuz(const double4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2_fnuz(const double4 val)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))) {
  }

  /*! create fp8x4 e5m2 type from float4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2_fnuz(const float4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2_fnuz(const float4 val)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))) {
  }

  /*! create fp8x4 e5m2 type from two __hip_bfloat162 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2_fnuz(const __hip_bfloat162 low, const __hip_bfloat162 high)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2_fnuz(const __hip_bfloat162 low, const __hip_bfloat162 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(static_cast<unsigned int>(
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(high, __default_saturation, __default_interpret)) |
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(low, __default_saturation, __default_interpret))
                << 16))) {
  }

  /*! create fp8x4 e5m2 type from two __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2_fnuz(const __half2 low, const __half2 high)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2_fnuz(const __half2 low, const __half2 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          high, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          low, __default_saturation, __default_interpret))
                                          << 16))) {
  }

  /* default construct fp8x4 e5m2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8x4_e5m2_fnuz() = default;
#endif

  /*! convert fp8x4 e5m2 to float4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float4() const {
#else
  __FP8_HOST__ operator float4() const {
#endif
      auto x = __x;                                                // bypass const
  auto fp8x2_low = *reinterpret_cast<__hip_fp8x2_storage_t*>(&x);  // Little E
  auto fp8x2_high = *(reinterpret_cast<__hip_fp8x2_storage_t*>(&x) + 1);
#if HIP_FP8_CVT_FAST_PATH
  float2 high = internal::cast_to_f32x2_from_f8x2(fp8x2_high, __default_interpret);
  float2 low = internal::cast_to_f32x2_from_f8x2(fp8x2_low, __default_interpret);
#else
    float2 high = float2(internal::cast_from_f8<float, true>(
                             static_cast<__hip_fp8_storage_t>((fp8x2_high << 8) >> 8), __wm, __we),
                         internal::cast_from_f8<float, true>(
                             static_cast<__hip_fp8_storage_t>(fp8x2_high >> 8), __wm, __we));
    float2 low = float2(internal::cast_from_f8<float, true>(
                            static_cast<__hip_fp8_storage_t>((fp8x2_low << 8) >> 8), __wm, __we),
                        internal::cast_from_f8<float, true>(
                            static_cast<__hip_fp8_storage_t>(fp8x2_low >> 8), __wm, __we));
#endif
  return float4(low.x, low.y, high.x, high.y);
}
}
;

#endif  // ENABLE_FNUZ_HIPRTC

/**
 * \brief struct representing ocp fp8 numbers with e4m3 interpretation
 *
 * */

#if !defined(ENABLE_OCP_HIPRTC) || ENABLE_OCP_HIPRTC

struct __hip_fp8_e4m3 {
  __hip_fp8_storage_t __x;  //! raw storage of fp8 number
  constexpr static __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  constexpr static __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3;
  constexpr static unsigned int __we = 4;
  constexpr static unsigned int __wm = 3;

  // TODO: SWDEV-452411
  // Add cast from unsigned long long, long long to fp8

  /*! create fp8 e4m3 from long */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const long int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from int */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from short int */
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const short int val)
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {}

  /*! create fp8 e4m3 from unsigned long */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const unsigned long int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const unsigned long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from unsigned int */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const unsigned int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const unsigned int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from unsigned short */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const unsigned short int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const unsigned short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from double */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const double f)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const double f)
#endif
      : __x(__hip_cvt_double_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e4m3 from float */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const float f)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const float f)
#endif
      : __x(__hip_cvt_float_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e4m3 from __hip_bfloat16 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const __hip_bfloat16 f)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const __hip_bfloat16 f)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(f), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from __half */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const __half f)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const __half f)
#endif
      : __x(__hip_cvt_halfraw_to_fp8(static_cast<__half_raw>(f), __default_saturation,
                                     __default_interpret)) {
  }

  /*! default construct fp8 e4m3 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3() = default;
#else
  __FP8_HOST__ __hip_fp8_e4m3() = default;
#endif

  /*! convert fp8 e4m3 to __half */

#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __half() const {
#else
  __FP8_HOST__ operator __half() const {
#endif
    return __half(__hip_cvt_fp8_to_halfraw(__x, __default_interpret));
  }

  /*! convert fp8 e4m3 to __hip_bfloat16 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __hip_bfloat16() const {
#else
  __FP8_HOST__ operator __hip_bfloat16() const {
#endif
    float f = *this;
    return __hip_bfloat16(f);
  }

  /*! convert fp8 e4m3 to bool, return false if value is 0, true otherwise */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator bool() const {
#else
  __FP8_HOST__ operator bool() const {
#endif
    // it can be 0x00 (+0.0) since 0x80 will be nan
    return !(static_cast<unsigned short>(__x) == 0 || static_cast<unsigned short>(__x) == 0x80);
  }

  /*! convert fp8 e4m3 to char, clamp number to __HIP_CHAR_MIN/__HIP_CHAR_MAX if its out of range */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator char() const {
#else
  __FP8_HOST__ operator char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    auto fval = internal::cast_from_f8<float, false>(__x, __wm, __we);
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_CHAR_MIN) {
      return __HIP_CHAR_MIN;
    } else if (llval >= __HIP_CHAR_MAX) {
      return __HIP_CHAR_MAX;
    }
    return static_cast<char>(fval);
  }

  /*! convert fp8 e4m3 to double */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator double() const {
#else
  __FP8_HOST__ operator double() const {
#endif
    return internal::cast_from_f8<double, false>(__x, __wm, __we);
  }

  /*! convert fp8 e4m3 to float */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float() const {
#else
  __FP8_HOST__ operator float() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
    return internal::cast_to_f32_from_f8(__x, __default_interpret);
#else
    return internal::cast_from_f8<float, false>(__x, __wm, __we);
#endif
  }

  /*! convert fp8 e4m3 to int, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator int() const {
#else
  __FP8_HOST__ operator int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<int>(fval);
  }

  /*! convert fp8 e4m3 to long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator long int() const {
#else
  __FP8_HOST__ operator long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long>(fval);
  }

  /*! convert fp8 e4m3 to long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator long long int() const {
#else
  __FP8_HOST__ operator long long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long long>(fval);
  }

  /*! convert fp8 e4m3 to short int, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator short int() const {
#else
  __FP8_HOST__ operator short int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SHRT_MIN) {
      return __HIP_SHRT_MIN;
    } else if (llval >= __HIP_SHRT_MAX) {
      return __HIP_SHRT_MAX;
    }
    return static_cast<short>(fval);
  }

  /*! convert fp8 e4m3 to signed char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator signed char() const {
#else
  __FP8_HOST__ operator signed char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SCHAR_MIN) {
      return __HIP_SCHAR_MIN;
    } else if (llval >= __HIP_SCHAR_MAX) {
      return __HIP_SCHAR_MAX;
    }
    return static_cast<signed char>(fval);
  }

  /*! convert fp8 e4m3 to unsigned char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned char() const {
#else
  __FP8_HOST__ operator unsigned char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    } else if (llval >= __HIP_UCHAR_MAX) {
      return __HIP_UCHAR_MAX;
    }
    return static_cast<unsigned char>(fval);
  }

  /*! convert fp8 e4m3 to unsigned int, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned int() const {
#else
  __FP8_HOST__ operator unsigned int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned int>(fval);
  }

  /*! convert fp8 e4m3 to unsigned long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned long int() const {
#else
  __FP8_HOST__ operator unsigned long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long>(fval);
  }

  /*! convert fp8 e4m3 to long long int, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned long long int() const {
#else
  __FP8_HOST__ operator unsigned long long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long long>(fval);
  }

  /*! convert fp8 e4m3 to unsigned short, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned short int() const {
#else
  __FP8_HOST__ operator unsigned short int() const {
#endif
      if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
        return 0;
}

float fval = *this;
auto llval = static_cast<long long>(fval);
if (llval <= 0) {
  return 0;
}
return static_cast<unsigned short>(fval);
}
}
;

/**
 * \brief struct representing two ocp fp8 numbers with e4m3 interpretation
 *
 * */
struct __hip_fp8x2_e4m3 {
  __hip_fp8x2_storage_t __x;  //! raw storage of two fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3;
  static constexpr unsigned int __we = 4;
  static constexpr unsigned int __wm = 3;

  /*! create fp8x2 e4m3 type from double2 */

#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3(const double2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3(const double2 val)
#endif
      : __x(__hip_cvt_double2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from float2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3(const float2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3(const float2 val)
#endif
      : __x(__hip_cvt_float2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from __hip_bfloat162 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3(const __hip_bfloat162 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3(const __hip_bfloat162 val)
#endif
      : __x(__hip_cvt_bfloat16raw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3(const __half2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3(const __half2 val)
#endif
      : __x(__hip_cvt_halfraw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! Default construct of fp8x2 e4m3 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3() = default;
#else
  __FP8_HOST__ __hip_fp8x2_e4m3() = default;
#endif

  /*! convert fp8x2 e4m3 to __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __half2() const {
#else
  __FP8_HOST__ operator __half2() const {
#endif
    return __half2(__hip_cvt_fp8x2_to_halfraw2(__x, __default_interpret));
  }

  /*! convert fp8x2 e4m3 to float2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float2() const {
#else
  __FP8_HOST__ operator float2() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
      return internal::cast_to_f32x2_from_f8x2(__x, __default_interpret);
#else
    return float2(internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(__x & 0xFF),
                                                       __wm, __we),
                  internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(__x >> 8),
                                                       __wm, __we));
#endif
}
}
;

/**
 * \brief struct representing four ocp fp8 numbers with e4m3 interpretation
 *
 * */
struct __hip_fp8x4_e4m3 {
  __hip_fp8x4_storage_t __x;  //! raw storage of four fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3;
  static constexpr unsigned int __we = 4;
  static constexpr unsigned int __wm = 3;

  /*! create fp8x4 e4m3 type from double4 */

#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3(const double4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3(const double4 val)
#endif
      : __x{reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))} {
  }

  /*! create fp8x4 e4m3 type from float4 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3(const float4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3(const float4 val)
#endif
      : __x{reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))} {
  }

  /*! create fp8x4 e4m3 type from two __hip_bfloat162 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3(const __hip_bfloat162 low, const __hip_bfloat162 high)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3(const __hip_bfloat162 low, const __hip_bfloat162 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(static_cast<unsigned int>(
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(high, __default_saturation, __default_interpret)) |
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(low, __default_saturation, __default_interpret))
                << 16))) {
  }

  /*! create fp8x4 e4m3 type from two __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3(const __half2 low, const __half2 high)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3(const __half2 low, const __half2 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          high, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          low, __default_saturation, __default_interpret))
                                          << 16))) {
  }

  /*! Default construct fp8x4 e4m3 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3() = default;
#else
  __FP8_HOST__ __hip_fp8x4_e4m3() = default;
#endif

  /*! convert fp8x4 e4m3 to float4 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float4() const {
#else
  __FP8_HOST__ operator float4() const {
#endif
      auto x = __x;                                                // bypass const
  auto fp8x2_low = *reinterpret_cast<__hip_fp8x2_storage_t*>(&x);  // Little E
  auto fp8x2_high = *(reinterpret_cast<__hip_fp8x2_storage_t*>(&x) + 1);
#if HIP_FP8_CVT_FAST_PATH
  float2 high = internal::cast_to_f32x2_from_f8x2(fp8x2_high, __default_interpret);
  float2 low = internal::cast_to_f32x2_from_f8x2(fp8x2_low, __default_interpret);
#else
    float2 high = float2(internal::cast_from_f8<float, false>(
                             static_cast<__hip_fp8_storage_t>((fp8x2_high << 8) >> 8), __wm, __we),
                         internal::cast_from_f8<float, false>(
                             static_cast<__hip_fp8_storage_t>(fp8x2_high >> 8), __wm, __we));
    float2 low = float2(internal::cast_from_f8<float, false>(
                            static_cast<__hip_fp8_storage_t>((fp8x2_low << 8) >> 8), __wm, __we),
                        internal::cast_from_f8<float, false>(
                            static_cast<__hip_fp8_storage_t>(fp8x2_low >> 8), __wm, __we));
#endif
  return float4(low.x, low.y, high.x, high.y);
}
}
;

/**
 * \brief struct representing  ocp fp8 numbers with e5m2 interpretation
 *
 * */
struct __hip_fp8_e5m2 {
  __hip_fp8_storage_t __x;  //! raw storage of one fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;


  // TODO: SWDEV-452411
  // Add cast from unsigned long long, long long to fp8

  /*! create fp8 e5m2 type from long */

#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const long int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from int */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from short int */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const short int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned long */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const unsigned long int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const unsigned long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned int */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const unsigned int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const unsigned int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned short */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const unsigned short int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const unsigned short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from double */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const double f)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const double f)
#endif
      : __x(__hip_cvt_double_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e5m2 type from float */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const float f)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const float f)
#endif
      : __x(__hip_cvt_float_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e5m2 type from __hip_bfloat16 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const __hip_bfloat16 f)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const __hip_bfloat16 f)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(f), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from __hip_bfloat16 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const __half f)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const __half f)
#endif
      : __x(__hip_cvt_halfraw_to_fp8(static_cast<__half_raw>(f), __default_saturation,
                                     __default_interpret)) {
  }

  /*! default construct fp8 e5m2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2() = default;
#else
  __FP8_HOST__ __hip_fp8_e5m2() = default;
#endif

  /*! convert fp8 e5m2 to float */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float() const {
#else
  __FP8_HOST__ operator float() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
    return internal::cast_to_f32_from_f8(__x, __default_interpret);
#else
    return internal::cast_from_f8<float, false>(__x, __wm, __we);
#endif
  }

  /*! convert fp8 e5m2 to __half */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __half() const {
#else
  __FP8_HOST__ operator __half() const {
#endif
    return __half(__hip_cvt_fp8_to_halfraw(__x, __default_interpret));
  }

  /*! convert fp8 e5m2 to __hip_bfloat16 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __hip_bfloat16() const {
#else
  __FP8_HOST__ operator __hip_bfloat16() const {
#endif
    float f = *this;
    return __hip_bfloat16(f);
  }

  /*! convert fp8 e4m3 to bool, return false if value is 0, true otherwise */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator bool() const {
#else
  __FP8_HOST__ operator bool() const {
#endif
    // it can be 0x00 (+0.0) since 0x80 will be nan
    return !(static_cast<unsigned short>(__x) == 0 || static_cast<unsigned short>(__x) == 0x80);
  }

  /*! convert fp8 e5m2 to char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator char() const {
#else
  __FP8_HOST__ operator char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_CHAR_MIN) {
      return __HIP_CHAR_MIN;
    } else if (llval >= __HIP_CHAR_MAX) {
      return __HIP_CHAR_MAX;
    }
    return static_cast<char>(fval);
  }

  /*! convert fp8 e5m2 to double */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator double() const {
#else
  __FP8_HOST__ operator double() const {
#endif
    return internal::cast_from_f8<double, false>(__x, __wm, __we,
                                                 __default_saturation == __HIP_SATFINITE);
  }

  /*! convert fp8 e5m2 to int, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator int() const {
#else
  __FP8_HOST__ operator int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<int>(fval);
  }

  /*! convert fp8 e5m2 to long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator long int() const {
#else
  __FP8_HOST__ operator long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long>(fval);
  }

  /*! convert fp8 e5m2 to long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator long long int() const {
#else
  __FP8_HOST__ operator long long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long long>(fval);
  }

  /*! convert fp8 e5m2 to short, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator short int() const {
#else
  __FP8_HOST__ operator short int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SHRT_MIN) {
      return __HIP_SHRT_MIN;
    } else if (llval >= __HIP_SHRT_MAX) {
      return __HIP_SHRT_MAX;
    }
    return static_cast<short>(fval);
  }

  /*! convert fp8 e5m2 to signed char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator signed char() const {
#else
  __FP8_HOST__ operator signed char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SCHAR_MIN) {
      return __HIP_SCHAR_MIN;
    } else if (llval >= __HIP_SCHAR_MAX) {
      return __HIP_SCHAR_MAX;
    }
    return static_cast<signed char>(fval);
  }

  /*! convert fp8 e5m2 to unsigned char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned char() const {
#else
  __FP8_HOST__ operator unsigned char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    } else if (llval >= __HIP_UCHAR_MAX) {
      return __HIP_UCHAR_MAX;
    }
    return static_cast<unsigned char>(fval);
  }

  /*! convert fp8 e5m2 to unsigned int, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned int() const {
#else
  __FP8_HOST__ operator unsigned int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned int>(fval);
  }

  /*! convert fp8 e5m2 to unsigned long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned long int() const {
#else
  __FP8_HOST__ operator unsigned long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long>(fval);
  }

  /*! convert fp8 e5m2 to unsigned long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned long long int() const {
#else
  __FP8_HOST__ operator unsigned long long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long long>(fval);
  }

  /*! convert fp8 e5m2 to unsigned short, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned short int() const {
#else
  __FP8_HOST__ operator unsigned short int() const {
#endif
      if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
        return 0;
}

float fval = *this;
auto llval = static_cast<long long>(fval);
if (llval <= 0) {
  return 0;
}
return static_cast<unsigned short>(fval);
}
}
;

/**
 * \brief struct representing two ocp fp8 numbers with e5m2 interpretation
 *
 */
struct __hip_fp8x2_e5m2 {
  __hip_fp8x2_storage_t __x;  //! raw storage of two fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;

  /*! create fp8x2 e5m2 type from double2 */

#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2(const double2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2(const double2 val)
#endif
      : __x(__hip_cvt_double2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from float2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2(const float2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2(const float2 val)
#endif
      : __x(__hip_cvt_float2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from __hip_bfloat162 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2(const __hip_bfloat162 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2(const __hip_bfloat162 val)
#endif
      : __x(__hip_cvt_bfloat16raw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2(const __half2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2(const __half2 val)
#endif
      : __x(__hip_cvt_halfraw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! default construct fp8x2 e5m2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2() = default;
#else
  __FP8_HOST__ __hip_fp8x2_e5m2() = default;
#endif

  /*! convert fp8x2 e5m2 to __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __half2() const {
#else
  __FP8_HOST__ operator __half2() const {
#endif
    return __half2(__hip_cvt_fp8x2_to_halfraw2(__x, __default_interpret));
  }

  /*! convert fp8x2 e5m2 to float2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float2() const {
#else
  __FP8_HOST__ operator float2() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
      return internal::cast_to_f32x2_from_f8x2(__x, __default_interpret);
#else
    return float2(
        internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(__x & 0xFF), __wm,
                                             __we, __default_saturation == __HIP_SATFINITE),
        internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(__x >> 8), __wm, __we,
                                             __default_saturation == __HIP_SATFINITE));
#endif
}
}
;

/**
 * \brief struct representing four ocp fp8 numbers with e5m2 interpretation
 *
 */
struct __hip_fp8x4_e5m2 {
  __hip_fp8x4_storage_t __x;  //! raw storage of four fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;

  /*! create fp8x4 e5m2 type from double4 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2(const double4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2(const double4 val)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))) {
  }

  /*! create fp8x4 e5m2 type from float4 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2(const float4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2(const float4 val)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))) {
  }

  /*! create fp8x4 e5m2 type from two __hip_bfloat162 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2(const __hip_bfloat162 low, const __hip_bfloat162 high)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2(const __hip_bfloat162 low, const __hip_bfloat162 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(static_cast<unsigned int>(
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(high, __default_saturation, __default_interpret)) |
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(low, __default_saturation, __default_interpret))
                << 16))) {
  }

  /*! create fp8x4 e5m2 type from two __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2(const __half2 low, const __half2 high)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2(const __half2 low, const __half2 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          high, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          low, __default_saturation, __default_interpret))
                                          << 16))) {
  }

  /* default construct fp8x4 e5m2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2() = default;
#else
  __FP8_HOST__ __hip_fp8x4_e5m2() = default;
#endif

  /*! convert fp8x4 e5m2 to float4 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float4() const {
#else
  __FP8_HOST__ operator float4() const {
#endif
      auto x = __x;                                                // bypass const
  auto fp8x2_low = *reinterpret_cast<__hip_fp8x2_storage_t*>(&x);  // Little E
  auto fp8x2_high = *(reinterpret_cast<__hip_fp8x2_storage_t*>(&x) + 1);
#if HIP_FP8_CVT_FAST_PATH
  float2 high = internal::cast_to_f32x2_from_f8x2(fp8x2_high, __default_interpret);
  float2 low = internal::cast_to_f32x2_from_f8x2(fp8x2_low, __default_interpret);
#else
    float2 high = float2(
        internal::cast_from_f8<float, false>(
            static_cast<__hip_fp8_storage_t>((fp8x2_high << 8) >> 8), __wm, __we,
            __default_saturation == __HIP_SATFINITE),
        internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(fp8x2_high >> 8),
                                             __wm, __we, __default_saturation == __HIP_SATFINITE));
    float2 low = float2(
        internal::cast_from_f8<float, false>(
            static_cast<__hip_fp8_storage_t>((fp8x2_low << 8) >> 8), __wm, __we,
            __default_saturation == __HIP_SATFINITE),
        internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(fp8x2_low >> 8), __wm,
                                             __we, __default_saturation == __HIP_SATFINITE));
#endif
  return float4(low.x, low.y, high.x, high.y);
}
}
;
#endif  // ENABLE_OCP_HIPRTC
#endif  // _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP8_H_
