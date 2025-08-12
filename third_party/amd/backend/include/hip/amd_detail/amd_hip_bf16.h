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
 * \brief hip_bf16.h provides struct for __hip_bfloat16 types
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16 bfloat16 Precision Intrinsics
 * This section describes hip_bfloat16 precision intrinsic functions.
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_ARITH Bfloat16 Arithmetic Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_COMP Bfloat16 Comparision Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_COMP Bfloat162 Comparision Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_ARITH Bfloat162 Arithmetic Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_CONV Bfloat16 Conversion Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_CONV Bfloat162 Conversion Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_MATH Bfloat16 Math Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_MATH Bfloat162 Math Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_RAW Bfloat16 Raw Struct
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_RAW Bfloat162 Raw Struct
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

#ifndef _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BF16_H_
#define _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BF16_H_

#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_common.h>
#endif  // !defined(__HIPCC_RTC__)

#include "amd_hip_vector_types.h"  // float2 etc
#include "device_library_decls.h"  // ocml conversion functions
#include "math_fwd.h"              // ocml device functions

#define __BF16_DEVICE__ __device__
#if defined(__HIPCC_RTC__)
#define __BF16_HOST_DEVICE__ __BF16_DEVICE__
#else
#include <algorithm>
#include <climits>
#include <cmath>
#define __BF16_HOST_DEVICE__ __host__ __BF16_DEVICE__
#endif
#define __BF16_DEVICE_STATIC__ __BF16_DEVICE__ static inline
#define __BF16_HOST_DEVICE_STATIC__ __BF16_HOST_DEVICE__ static inline

#if defined(__AVX512VL__) and defined(__AVX512BF16__) and not defined(__HIP_DEVICE_COMPILE__)
// Enable with -mavx512vl -mavx512bf16
#if defined(__MINGW64__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#define HIP_BF16_AVX512_OP 1
static_assert(sizeof(__bf16) == sizeof(unsigned short),
              "sizeof __bf16 should match sizeof unsigned short");
#else
#define HIP_BF16_AVX512_OP 0
#endif

#define HIPRT_ONE_BF16 __float2bfloat16(1.0f)
#define HIPRT_ZERO_BF16 __float2bfloat16(0.0f)
#define HIPRT_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#define HIPRT_MAX_NORMAL_BF16 __ushort_as_bfloat16((unsigned short)0x7F7FU)
#define HIPRT_MIN_DENORM_BF16 __ushort_as_bfloat16((unsigned short)0x0001U)
#define HIPRT_NAN_BF16 __ushort_as_bfloat16((unsigned short)0x7FFFU)
#define HIPRT_NEG_ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x8000U)

// Since we are using unsigned short to represent data in bfloat16, it can be of different sizes on
// different machines. These naive checks should prevent some undefined behavior on systems which
// have different sizes for basic types.
#if !defined(__HIPCC_RTC__)
static_assert(CHAR_BIT == 8, "byte size should be of 8 bits");
#endif
static_assert(sizeof(unsigned short) == 2, "size of unsigned short should be 2 bytes");

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_RAW
 * \brief represents raw bfloat16 type
 */
typedef struct __attribute__((aligned(2))) {
  unsigned short x;
} __hip_bfloat16_raw;

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_RAW
 * \brief represents raw bfloat16x2 vector type
 */
typedef struct __attribute__((aligned(4))) {
  unsigned short x;
  unsigned short y;
} __hip_bfloat162_raw;

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_STRUCT
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * \brief Struct to represent a 16 bit brain floating point number.
 * @{
 */
struct __attribute__((aligned(2))) __hip_bfloat16 {
 private:
  __BF16_HOST_DEVICE_STATIC__ float bfloatraw_2_float(unsigned short val) {
#if HIP_BF16_AVX512_OP
    union {
      unsigned short us;
      __bf16 bf16;
    } u = {val};
    return _mm_cvtsbh_ss(u.bf16);
#else
    unsigned int uval = val << 16;
    union {
      unsigned int u32;
      float fp32;
    } u = {uval};
    return u.fp32;
#endif
  }
  __BF16_HOST_DEVICE_STATIC__ unsigned short float_2_bfloatraw(float f) {
#if HIP_BF16_AVX512_OP
    union {
      __bf16 bf16;
      unsigned short us;
    } u = {_mm_cvtness_sbh(f)};
    return u.us;
#else
    union {
      float fp32;
      unsigned int u32;
    } u = {f};
    if (~u.u32 & 0x7f800000) {
      // When the exponent bits are not all 1s, then the value is zero, normal,
      // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
      // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
      // This causes the bfloat16's mantissa to be incremented by 1 if the 16
      // least significant bits of the float mantissa are greater than 0x8000,
      // or if they are equal to 0x8000 and the least significant bit of the
      // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
      // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
      // has the value 0x7f, then incrementing it causes it to become 0x00 and
      // the exponent is incremented by one, which is the next higher FP value
      // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
      // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
      // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
      // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
      // incrementing it causes it to become an exponent of 0xFF and a mantissa
      // of 0x00, which is Inf, the next higher value to the unrounded value.
      u.u32 += 0x7fff + ((u.u32 >> 16) & 1);  // Round to nearest, round to even
    } else if (u.u32 & 0xffff) {
      // When all of the exponent bits are 1, the value is Inf or NaN.
      // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
      // mantissa bit. Quiet NaN is indicated by the most significant mantissa
      // bit being 1. Signaling NaN is indicated by the most significant
      // mantissa bit being 0 but some other bit(s) being 1. If any of the
      // lower 16 bits of the mantissa are 1, we set the least significant bit
      // of the bfloat16 mantissa, in order to preserve signaling NaN in case
      // the bloat16's mantissa bits are all 0.
      u.u32 |= 0x10000;  // Preserve signaling NaN
    }
    return static_cast<unsigned short>(u.u32 >> 16);
#endif
  }

  __BF16_HOST_DEVICE_STATIC__ unsigned short double_2_bfloatraw(double d_in) {
    union {
      float fp32;
      unsigned int u32;
    } u = {static_cast<float>(d_in)};
    double d = u.fp32;

    // Round to odd
    if ((d_in > 0.0 && d > d_in) || (d_in < 0.0 && d < d_in)) {
      u.u32--;
      u.u32 |= 1;
    }

    return float_2_bfloatraw(u.fp32);
  }

 protected:
  /*! \brief raw representation of bfloat16 */
  unsigned short __x;

 public:
  // TODO: SWDEV-452411
  // Need to add constructor of __hip_bfloat16 from
  // unsigned long long
  // long long
  // long
  // unsigned long
  // Casting directly to double might lead to double rounding.

  /*! \brief create __hip_bfloat16 from an unsigned int */
  __BF16_HOST_DEVICE__ __hip_bfloat16(unsigned int val)
      : __x(double_2_bfloatraw(static_cast<double>(val))) {}

  /*! \brief create __hip_bfloat16 from a int */
  __BF16_HOST_DEVICE__ __hip_bfloat16(int val)
      : __x(double_2_bfloatraw(static_cast<double>(val))) {}

  /*! \brief create __hip_bfloat16 from an unsigned short */
  __BF16_HOST_DEVICE__ __hip_bfloat16(unsigned short val)
      : __x(float_2_bfloatraw(static_cast<float>(val))) {}

  /*! \brief create __hip_bfloat16 from a short */
  __BF16_HOST_DEVICE__ __hip_bfloat16(short val)
      : __x(float_2_bfloatraw(static_cast<float>(val))) {}

  /*! \brief create __hip_bfloat16 from a double */
  __BF16_HOST_DEVICE__ __hip_bfloat16(const double val) : __x(double_2_bfloatraw(val)) {}

  /*! \brief create __hip_bfloat16 from a float */
  __BF16_HOST_DEVICE__ __hip_bfloat16(const float val) : __x(float_2_bfloatraw(val)) {}

  /*! \brief create __hip_bfloat16 from a __hip_bfloat16_raw */
  __BF16_HOST_DEVICE__ __hip_bfloat16(const __hip_bfloat16_raw& val) : __x(val.x) {}

  /*! \brief default constructor */
  __BF16_HOST_DEVICE__ __hip_bfloat16() = default;

  /*! \brief return a __hip_bfloat16_raw */
  __BF16_HOST_DEVICE__ operator __hip_bfloat16_raw() const { return __hip_bfloat16_raw{__x}; }

  /*! \brief return a __hip_bfloat16_raw cv qualifier */
  __BF16_HOST_DEVICE__ operator __hip_bfloat16_raw() const volatile {
    return __hip_bfloat16_raw{__x};
  }

  /*! \brief return false if bfloat value is +0.0 or -0.0, returns true otherwise */
  __BF16_HOST_DEVICE__ operator bool() const {
    auto val = bfloatraw_2_float(__x);
    return val != 0.0f && val != -0.0f;
  }

  /*! \brief return a casted char from underlying float val */
  __BF16_HOST_DEVICE__ operator char() const { return static_cast<char>(bfloatraw_2_float(__x)); }

  /*! \brief return a float */
  __BF16_HOST_DEVICE__ operator float() const { return bfloatraw_2_float(__x); }

  /*! \brief return a casted int casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator int() const { return static_cast<int>(bfloatraw_2_float(__x)); }

  /*! \brief return a casted long casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator long() const { return static_cast<long>(bfloatraw_2_float(__x)); }

  /*! \brief return a casted long long casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator long long() const {
    return static_cast<long long>(bfloatraw_2_float(__x));
  }

  /*! \brief return a casted short casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator short() const { return static_cast<short>(bfloatraw_2_float(__x)); }

  /*! \brief return a casted signed char from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator signed char() const {
    return static_cast<signed char>(bfloatraw_2_float(__x));
  }

  /*! \brief return a casted unsigned char casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned char() const {
    return static_cast<unsigned char>(bfloatraw_2_float(__x));
  }

  /*! \brief return a casted unsigned int casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned int() const {
    return static_cast<unsigned int>(bfloatraw_2_float(__x));
  }

  /*! \brief return a casted unsigned from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned long() const {
    return static_cast<unsigned long>(bfloatraw_2_float(__x));
  }

  /*! \brief return a casted unsigned long long from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned long long() const {
    return static_cast<unsigned long long>(bfloatraw_2_float(__x));
  }

  /*! \brief return a casted unsigned short from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned short() const {
    return static_cast<unsigned short>(bfloatraw_2_float(__x));
  }

  // TODO: SWDEV-452411 add operator which converts unsigned long long and long long to bfloat

  /*! \brief assign value from an unsigned int */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(unsigned int val) {
    __x = float_2_bfloatraw(static_cast<float>(val));
    return *this;
  }

  /*! \brief assign value from a int */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(int val) {
    __x = float_2_bfloatraw(static_cast<float>(val));
    return *this;
  }

  /*! \brief assign value from an unsigned short */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(unsigned short val) {
    __x = float_2_bfloatraw(static_cast<float>(val));
    return *this;
  }

  /*! \brief assign value from a short int */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(short val) {
    __x = float_2_bfloatraw(static_cast<float>(val));
    return *this;
  }

  /*! \brief assign value from a double */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(const double f) {
    __x = float_2_bfloatraw(static_cast<float>(f));
    return *this;
  }

  /*! \brief assign value from a float */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(const float f) {
    __x = float_2_bfloatraw(f);
    return *this;
  }

  /*! \brief assign value from a __hip_bfloat16_raw */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(const __hip_bfloat16_raw& hr) {
    __x = hr.x;
    return *this;
  }

  /*! \brief assign value from a __hip_bfloat16_raw volatile */
  __BF16_HOST_DEVICE__ volatile __hip_bfloat16& operator=(const __hip_bfloat16_raw& hr) volatile {
    __x = hr.x;
    return *this;
  }

  /*! \brief assign value from a __hip_bfloat16_raw cv qualifier */
  __BF16_HOST_DEVICE__ volatile __hip_bfloat16& operator=(
      const volatile __hip_bfloat16_raw& hr) volatile {
    __x = hr.x;
    return *this;
  }
};
/**@}*/

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_STRUCT
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * \brief Struct to represent a two 16 bit brain floating point number.
 * @{
 */
struct __attribute__((aligned(4))) __hip_bfloat162 {
 public:
  __hip_bfloat16 x; /*! \brief raw representation of bfloat16 */
  __hip_bfloat16 y; /*! \brief raw representation of bfloat16 */


 public:
  /*! \brief create __hip_bfloat162 from __hip_bfloat162_raw */
  __BF16_HOST_DEVICE__ __hip_bfloat162(const __hip_bfloat162_raw& h2r)
      : x(__hip_bfloat16(__hip_bfloat16_raw{h2r.x})),
        y(__hip_bfloat16(__hip_bfloat16_raw{h2r.y})) {}

  /*! \brief copy constructor of __hip_bfloat162 */
  __BF16_HOST_DEVICE__ __hip_bfloat162(const __hip_bfloat162& val) {
    __hip_bfloat162_raw hr = val;
    x = __hip_bfloat16_raw{hr.x};
    y = __hip_bfloat16_raw{hr.y};
  }

  /*! \brief create __hip_bfloat162 from two __hip_bfloat16 */
  __BF16_HOST_DEVICE__ __hip_bfloat162(const __hip_bfloat16& a, const __hip_bfloat16& b)
      : x(a), y(b) {}

  /*! \brief default constructor of __hip_bfloat162 */
  __BF16_HOST_DEVICE__ __hip_bfloat162() = default;

  /*! \brief return a __hip_bfloat162_raw */
  __BF16_HOST_DEVICE__ operator __hip_bfloat162_raw() const {
    __hip_bfloat16_raw l = x;
    __hip_bfloat16_raw r = y;
    return __hip_bfloat162_raw{l.x, r.x};
  }

  /*! \brief return a float2 */
  __BF16_HOST_DEVICE__ operator float2() const {
#if HIP_BF16_AVX512_OP
    union {
      __hip_bfloat162_raw raw2;
      __bf16 bf162[2];
      static_assert(sizeof(__bf16[2]) == sizeof(__hip_bfloat162_raw));
    } u;
    u.raw2 = *this;
    __m128bh pbf16{u.bf162[0], u.bf162[1], 0, 0};
    __m128 pf32 = _mm_cvtpbh_ps(pbf16);
    float2 ret(pf32[0], pf32[1]);
#else
    float2 ret(x, y);
#endif
    return ret;
  }

  /*! \brief assign value from __hip_bfloat162_raw */
  __BF16_HOST_DEVICE__ __hip_bfloat162& operator=(const __hip_bfloat162_raw& h2r) {
    x = __hip_bfloat16(__hip_bfloat16_raw{h2r.x});
    y = __hip_bfloat16(__hip_bfloat16_raw{h2r.y});
    return *this;
  }

  /*! \brief assign value from __hip_bfloat162 */
  __BF16_HOST_DEVICE__ __hip_bfloat162& operator=(const __hip_bfloat162& src) {
    __hip_bfloat162_raw hr = src;
    x = __hip_bfloat16(__hip_bfloat16_raw{hr.x});
    y = __hip_bfloat16(__hip_bfloat16_raw{hr.y});
    return *this;
  }
};
/**@}*/

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Converts bfloat16 to float
 */
__BF16_HOST_DEVICE_STATIC__ float __bfloat162float(__hip_bfloat16 a) {
  float ret = a;
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Converts float to bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __float2bfloat16(float f) {
  __hip_bfloat16 ret{f};
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts and moves bfloat162 to float2
 */
__BF16_HOST_DEVICE_STATIC__ float2 __bfloat1622float2(const __hip_bfloat162 a) {
  float2 ret = a;
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Moves bfloat16 value to bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __bfloat162bfloat162(const __hip_bfloat16 a) {
  return __hip_bfloat162(a, a);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets bits in a __hip_bfloat16 as a signed short integer
 */
__BF16_HOST_DEVICE_STATIC__ short int __bfloat16_as_short(const __hip_bfloat16 h) {
  short ret = h;
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets bits in a __hip_bfloat16 as an unsigned signed short integer
 */
__BF16_HOST_DEVICE_STATIC__ unsigned short int __bfloat16_as_ushort(const __hip_bfloat16 h) {
  unsigned short ret = h;
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Convert double to __hip_bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __double2bfloat16(const double a) {
  __hip_bfloat16 ret{a};
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Convert float2 to __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __float22bfloat162_rn(const float2 a) {
  return __hip_bfloat162{__float2bfloat16(a.x), __float2bfloat16(a.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Combine two __hip_bfloat16 to __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __halves2bfloat162(const __hip_bfloat16 a,
                                                               const __hip_bfloat16 b) {
  return __hip_bfloat162(a, b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns high 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __high2bfloat16(const __hip_bfloat162 a) {
  __hip_bfloat162_raw hr = a;
  return __hip_bfloat16(__hip_bfloat16_raw{hr.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns high 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __high2bfloat162(const __hip_bfloat162 a) {
  __hip_bfloat162_raw hr = a;
  return __hip_bfloat162(__hip_bfloat16_raw{hr.y}, __hip_bfloat16_raw{hr.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts high 16 bits of __hip_bfloat162 to float and returns the result
 */
__BF16_HOST_DEVICE_STATIC__ float __high2float(const __hip_bfloat162 a) {
  __hip_bfloat162_raw hr = a;
  return __bfloat162float(__hip_bfloat16(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Extracts high 16 bits from each and combines them
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __highs2bfloat162(const __hip_bfloat162 a,
                                                              const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162(__hip_bfloat162_raw{hr_a.y, hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns low 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __low2bfloat16(const __hip_bfloat162 a) {
  __hip_bfloat162_raw hr = a;
  return __hip_bfloat16(hr.x);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns low 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __low2bfloat162(const __hip_bfloat162 a) {
  __hip_bfloat162_raw hr = a;
  return __hip_bfloat162(hr.x, hr.x);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts low 16 bits of __hip_bfloat162 to float and returns the result
 */
__BF16_HOST_DEVICE_STATIC__ float __low2float(const __hip_bfloat162 a) {
  __hip_bfloat162_raw hr = a;
  return __bfloat162float(__hip_bfloat16(__hip_bfloat16_raw{hr.x}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Swaps both halves
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __lowhigh2highlow(const __hip_bfloat162 a) {
  __hip_bfloat162_raw hr = a;
  return __hip_bfloat162(__hip_bfloat162_raw{hr.y, hr.x});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Extracts low 16 bits from each and combines them
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __lows2bfloat162(const __hip_bfloat162 a,
                                                             const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162(__hip_bfloat162_raw{hr_a.x, hr_b.x});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets short int into a bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __short_as_bfloat16(const short int a) {
  return __hip_bfloat16(a);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets unsigned short int into a bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __ushort_as_bfloat16(const unsigned short int a) {
  return __hip_bfloat16(a);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Adds two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hadd(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Subtracts two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hsub(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) - __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Divides two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hdiv(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) / __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Performs FMA of given bfloat16 values
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __hfma(const __hip_bfloat16 a, const __hip_bfloat16 b,
                                             const __hip_bfloat16 c) {
  return __float2bfloat16(
      __ocml_fma_f32(__bfloat162float(a), __bfloat162float(b), __bfloat162float(c)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Multiplies two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmul(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Negate a bfloat16 value
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hneg(const __hip_bfloat16 a) {
  __hip_bfloat16_raw hr = a;
  hr.x ^= 0x8000;
  return __hip_bfloat16(hr);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Returns absolute of a bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __habs(const __hip_bfloat16 a) {
  __hip_bfloat16_raw hr = a;
  hr.x &= 0x7FFF;
  return __hip_bfloat16(hr);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Divides bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __h2div(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162(__float2bfloat16(__bfloat162float(__hip_bfloat16_raw{hr_a.x}) /
                                          __bfloat162float(__hip_bfloat16_raw{hr_b.x})),
                         __float2bfloat16(__bfloat162float(__hip_bfloat16_raw{hr_a.y}) /
                                          __bfloat162float(__hip_bfloat16_raw{hr_b.y})));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Returns absolute of a bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __habs2(const __hip_bfloat162 a) {
  __hip_bfloat162_raw hr_a = a;
  return __hip_bfloat162(__habs(__hip_bfloat16_raw{hr_a.x}), __habs(__hip_bfloat16_raw{hr_a.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Adds two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hadd2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162(__hadd(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}),
                         __hadd(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Performs FMA of given bfloat162 values
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __hfma2(const __hip_bfloat162 a, const __hip_bfloat162 b,
                                               const __hip_bfloat162 c) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  __hip_bfloat162_raw hr_c = c;
  return __hip_bfloat162(
      __hfma(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}, __hip_bfloat16_raw{hr_c.x}),
      __hfma(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}, __hip_bfloat16_raw{hr_c.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Multiplies two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmul2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162(__hmul(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}),
                         __hmul(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Converts a bfloat162 into negative
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hneg2(const __hip_bfloat162 a) {
  __hip_bfloat162_raw hr_a = a;
  return __hip_bfloat162(__hneg(__hip_bfloat16_raw{hr_a.x}), __hneg(__hip_bfloat16_raw{hr_a.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Subtracts two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hsub2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162(__hsub(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}),
                         __hsub(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to multiply two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator*(const __hip_bfloat16& l,
                                                     const __hip_bfloat16& r) {
  return __hmul(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to multiply-assign two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator*=(__hip_bfloat16& l, const __hip_bfloat16& r) {
  l = __hmul(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to unary+ on a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator+(const __hip_bfloat16& l) { return l; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to add two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator+(const __hip_bfloat16& l,
                                                     const __hip_bfloat16& r) {
  return __hadd(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to negate a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator-(const __hip_bfloat16& l) { return __hneg(l); }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to subtract two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator-(const __hip_bfloat16& l,
                                                     const __hip_bfloat16& r) {
  return __hsub(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to post increment a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator++(__hip_bfloat16& l, const int) {
  auto ret = l;
  l = __hadd(l, HIPRT_ONE_BF16);
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to pre increment a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator++(__hip_bfloat16& l) {
  l = __hadd(l, HIPRT_ONE_BF16);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to post decrement a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator--(__hip_bfloat16& l, const int) {
  auto ret = l;
  l = __hsub(l, HIPRT_ONE_BF16);
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to pre decrement a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator--(__hip_bfloat16& l) {
  l = __hsub(l, HIPRT_ONE_BF16);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to add-assign two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator+=(__hip_bfloat16& l, const __hip_bfloat16& r) {
  l = __hadd(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to subtract-assign two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator-=(__hip_bfloat16& l, const __hip_bfloat16& r) {
  l = __hsub(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to divide two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator/(const __hip_bfloat16& l,
                                                     const __hip_bfloat16& r) {
  return __hdiv(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to divide-assign two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator/=(__hip_bfloat16& l, const __hip_bfloat16& r) {
  l = __hdiv(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to multiply two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator*(const __hip_bfloat162& l,
                                                      const __hip_bfloat162& r) {
  return __hmul2(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to multiply-assign two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator*=(__hip_bfloat162& l,
                                                        const __hip_bfloat162& r) {
  l = __hmul2(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to unary+ on a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator+(const __hip_bfloat162& l) { return l; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to add two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator+(const __hip_bfloat162& l,
                                                      const __hip_bfloat162& r) {
  return __hadd2(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to negate a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator-(const __hip_bfloat162& l) {
  return __hneg2(l);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to subtract two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator-(const __hip_bfloat162& l,
                                                      const __hip_bfloat162& r) {
  return __hsub2(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to post increment a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator++(__hip_bfloat162& l, const int) {
  auto ret = l;
  l = __hadd2(l, {HIPRT_ONE_BF16, HIPRT_ONE_BF16});
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to pre increment a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator++(__hip_bfloat162& l) {
  l = __hadd2(l, {HIPRT_ONE_BF16, HIPRT_ONE_BF16});
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to post decrement a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator--(__hip_bfloat162& l, const int) {
  auto ret = l;
  l = __hsub2(l, {HIPRT_ONE_BF16, HIPRT_ONE_BF16});
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to pre decrement a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator--(__hip_bfloat162& l) {
  l = __hsub2(l, {HIPRT_ONE_BF16, HIPRT_ONE_BF16});
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to add-assign two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator+=(__hip_bfloat162& l,
                                                        const __hip_bfloat162& r) {
  l = __hadd2(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to subtract-assign two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator-=(__hip_bfloat162& l,
                                                        const __hip_bfloat162& r) {
  l = __hsub2(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to divide two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator/(const __hip_bfloat162& l,
                                                      const __hip_bfloat162& r) {
  return __h2div(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to divide-assign two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator/=(__hip_bfloat162& l,
                                                        const __hip_bfloat162& r) {
  l = __h2div(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ bool __heq(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) == __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hequ(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) < __bfloat162float(b)) &&
      !(__bfloat162float(a) > __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - greater than
 */
__BF16_HOST_DEVICE_STATIC__ bool __hgt(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) > __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered greater than
 */
__BF16_HOST_DEVICE_STATIC__ bool __hgtu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) <= __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - greater than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hge(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) >= __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered greater than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hgeu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) < __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - not equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hne(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) != __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered not equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hneu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) == __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - return max
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmax(const __hip_bfloat16 a, const __hip_bfloat16 b) {
#if __HIP_DEVICE_COMPILE__
  return __float2bfloat16(__ocml_fmax_f32(__bfloat162float(a), __bfloat162float(b)));
#else
  return __float2bfloat16(std::max(__bfloat162float(a), __bfloat162float(b)));
#endif
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - return min
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmin(const __hip_bfloat16 a, const __hip_bfloat16 b) {
#if __HIP_DEVICE_COMPILE__
  return __float2bfloat16(__ocml_fmin_f32(__bfloat162float(a), __bfloat162float(b)));
#else
  return __float2bfloat16(std::min(__bfloat162float(a), __bfloat162float(b)));
#endif
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - less than operator
 */
__BF16_HOST_DEVICE_STATIC__ bool __hlt(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) < __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered less than
 */
__BF16_HOST_DEVICE_STATIC__ bool __hltu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) >= __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - less than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hle(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) <= __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered less than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hleu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) > __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Checks if number is inf
 */
__BF16_HOST_DEVICE_STATIC__ int __hisinf(const __hip_bfloat16 a) {
  __hip_bfloat16_raw hr = a;
  return !(~hr.x & 0x7f80) && !(hr.x & 0x7f);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Checks if number is nan
 */
__BF16_HOST_DEVICE_STATIC__ bool __hisnan(const __hip_bfloat16 a) {
  __hip_bfloat16_raw hr = a;
  return !(~hr.x & 0x7f80) && +(hr.x & 0x7f);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks if two numbers are equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbeq2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __heq(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) &&
      __heq(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks if two numbers are equal - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbequ2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hequ(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) &&
      __hequ(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbge2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hge(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) &&
      __hge(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbgeu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hgeu(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) &&
      __hgeu(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbgt2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hgt(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) &&
      __hgt(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbgtu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hgtu(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) &&
      __hgtu(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hble2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hle(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) &&
      __hle(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbleu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hleu(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) &&
      __hleu(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hblt2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hlt(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) &&
      __hlt(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbltu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hltu(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) &&
      __hltu(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbne2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hne(__hip_bfloat16(__hip_bfloat16_raw{hr_a.x}),
               __hip_bfloat16(__hip_bfloat16_raw{hr_b.x})) &&
      __hne(__hip_bfloat16(__hip_bfloat16_raw{hr_a.y}), __hip_bfloat16(__hip_bfloat16_raw{hr_b.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbneu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hneu(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) ||
      __hneu(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y});
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b, returns 1.0 if equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __heq2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162{
      {__heq(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16},
      {__heq(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hge2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162{
      {__hge(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16},
      {__hge(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hgt2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162{
      {__hgt(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16},
      {__hgt(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ONE_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a is NaN, returns 1.0 if NaN, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hisnan2(const __hip_bfloat162 a) {
  __hip_bfloat162_raw hr_a = a;
  return __hip_bfloat162{{__hisnan(__hip_bfloat16_raw{hr_a.x}) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hisnan(__hip_bfloat16_raw{hr_a.y}) ? HIPRT_ONE_BF16 : HIPRT_ONE_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hle2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162{
      {__hle(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16},
      {__hle(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hlt2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162{
      {__hlt(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16},
      {__hlt(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Returns max of two elements
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmax2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162(__hmax(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}),
                         __hmax(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Returns min of two elements
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmin2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162(__hmin(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}),
                         __hmin(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks for not equal to
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hne2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  __hip_bfloat162_raw hr_a = a;
  __hip_bfloat162_raw hr_b = b;
  return __hip_bfloat162{
      {__hne(__hip_bfloat16_raw{hr_a.x}, __hip_bfloat16_raw{hr_b.x}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16},
      {__hne(__hip_bfloat16_raw{hr_a.y}, __hip_bfloat16_raw{hr_b.y}) ? HIPRT_ONE_BF16
                                                                     : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform an equal compare on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator==(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __heq(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a not equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator!=(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __hne(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a less than on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator<(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __hlt(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a less than equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator<=(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __hle(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a greater than on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator>(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __hgt(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a greater than equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator>=(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __hge(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Operator to perform an equal compare on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator==(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  float2 ret = __heq2(l, r);
  return ret.x != 0.0f && ret.y != 0.0f;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Operator to perform a not equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator!=(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  return !(l == r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Operator to perform a less than on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator<(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  float2 fl = l, fr = r;
  return fl.x < fr.x && fl.x < fr.y;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Operator to perform a less than equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator<=(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  float2 fl = l, fr = r;
  return fl.x <= fr.x && fl.x <= fr.y;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Operator to perform a greater than on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator>(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  float2 fl = l, fr = r;
  return fl.x > fr.x && fl.x > fr.y;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a greater than equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator>=(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  float2 fl = l, fr = r;
  return fl.x >= fr.x && fl.x >= fr.y;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate ceil of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hceil(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_ceil_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate cosine of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hcos(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_cos_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate exponential of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hexp(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_exp_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate exponential 10 of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hexp10(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_exp10_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate exponential 2 of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hexp2(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_exp2_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate floor of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hfloor(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_floor_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate natural log of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hlog(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_log_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate log 10 of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hlog10(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_log10_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate log 2 of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hlog2(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_log2_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate reciprocal
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hrcp(const __hip_bfloat16 h) {
  return __float2bfloat16(1.0f / (__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Round to nearest int
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hrint(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_rint_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Reciprocal square root
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hrsqrt(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_rsqrt_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate sin of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hsin(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_sin_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate sqrt of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hsqrt(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_sqrt_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate truncate of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 htrunc(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_trunc_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate ceil of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2ceil(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hceil(__hip_bfloat16_raw{hr.x}), hceil(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate cosine of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2cos(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hcos(__hip_bfloat16_raw{hr.x}), hcos(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2exp(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hexp(__hip_bfloat16_raw{hr.x}), hexp(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential 10 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2exp10(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hexp10(__hip_bfloat16_raw{hr.x}), hexp10(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential 2 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2exp2(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hexp2(__hip_bfloat16_raw{hr.x}), hexp2(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate floor of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2floor(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hfloor(__hip_bfloat16_raw{hr.x}), hfloor(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate natural log of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2log(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hlog(__hip_bfloat16_raw{hr.x}), hlog(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate log 10 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2log10(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hlog10(__hip_bfloat16_raw{hr.x}), hlog10(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate log 2 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2log2(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hlog2(__hip_bfloat16_raw{hr.x}), hlog2(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector reciprocal
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2rcp(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hrcp(__hip_bfloat16_raw{hr.x}), hrcp(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector round to nearest int
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2rint(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hrint(__hip_bfloat16_raw{hr.x}), hrint(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector reciprocal square root
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2rsqrt(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hrsqrt(__hip_bfloat16_raw{hr.x}), hrsqrt(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate sin of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2sin(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hsin(__hip_bfloat16_raw{hr.x}), hsin(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate sqrt of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2sqrt(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(hsqrt(__hip_bfloat16_raw{hr.x}), hsqrt(__hip_bfloat16_raw{hr.y}));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate truncate of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2trunc(const __hip_bfloat162 h) {
  __hip_bfloat162_raw hr = h;
  return __hip_bfloat162(htrunc(__hip_bfloat16_raw{hr.x}), htrunc(__hip_bfloat16_raw{hr.y}));
}
#endif
