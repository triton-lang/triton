/**
 * MIT License
 *
 * Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * \defgroup HIP_INTRINSIC_BFLOAT16_MOVE Bfloat16 Data Movement Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_MOVE Bfloat162 Data Movement Functions
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
#include "amd_hip_vector_types.h"  // float2 etc
#include "device_library_decls.h"  // ocml conversion functions
#include "math_fwd.h"              // ocml device functions
#if defined(__clang__) and defined(__HIP__)
#include <hip/amd_detail/amd_warp_functions.h>       // define warpSize
#include <hip/amd_detail/amd_warp_sync_functions.h>  // Sync functions
#endif
#endif  // !defined(__HIPCC_RTC__)

#define __BF16_DEVICE__ __device__
#if defined(__HIPCC_RTC__)
#define __BF16_HOST_DEVICE__ __BF16_DEVICE__
#else
#include <climits>
#include <cmath>
#define __BF16_HOST_DEVICE__ __host__ __BF16_DEVICE__
#endif
#define __BF16_DEVICE_STATIC__ __BF16_DEVICE__ static inline
#define __BF16_HOST_DEVICE_STATIC__ __BF16_HOST_DEVICE__ static inline

#pragma push_macro("MAYBE_UNDEF")
#if defined(__has_attribute) && __has_attribute(maybe_undef)
#define MAYBE_UNDEF __attribute__((maybe_undef))
#else
#define MAYBE_UNDEF
#endif

#define HIPRT_ONE_BF16 __ushort_as_bfloat16((unsigned short)0x3F80U)
#define HIPRT_ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x0000U)
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
static_assert(sizeof(__bf16) == sizeof(unsigned short));

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
 protected:
  union {
    /*! \brief raw representation of bfloat16 */
    unsigned short __x;
    /*! \brief bf16 represenation */
    __bf16 __x_bf16;
  };

 public:
  // TODO: SWDEV-452411
  // Need to add constructor of __hip_bfloat16 from
  // unsigned long long
  // long long
  // long
  // unsigned long
  // Casting directly to double might lead to double rounding.

  /*! \brief create __hip_bfloat16 from an unsigned int */
  __BF16_HOST_DEVICE__ __hip_bfloat16(unsigned int val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from a int */
  __BF16_HOST_DEVICE__ __hip_bfloat16(int val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from an unsigned short */
  __BF16_HOST_DEVICE__ __hip_bfloat16(unsigned short val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from a short */
  __BF16_HOST_DEVICE__ __hip_bfloat16(short val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from a double */
  __BF16_HOST_DEVICE__ __hip_bfloat16(const double val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from a float */
  __BF16_HOST_DEVICE__ __hip_bfloat16(const float val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from a __hip_bfloat16_raw */
  __BF16_HOST_DEVICE__ constexpr __hip_bfloat16(const __hip_bfloat16_raw& val) : __x(val.x) {}

  /*! \brief create __hip_bfloat16 from __bf16 */
  __BF16_HOST_DEVICE__ __hip_bfloat16(const __bf16 val) : __x_bf16(val) {}

  /*! \brief default constructor */
  __BF16_HOST_DEVICE__ __hip_bfloat16() = default;

  /*! \brief return a __hip_bfloat16_raw */
  __BF16_HOST_DEVICE__ operator __hip_bfloat16_raw() const { return __hip_bfloat16_raw{__x}; }

  /*! \brief return a __hip_bfloat16_raw cv qualifier */
  __BF16_HOST_DEVICE__ operator __hip_bfloat16_raw() const volatile {
    return __hip_bfloat16_raw{__x};
  }

  /*! \brief return false if bfloat value is +0.0 or -0.0, returns true otherwise */
  __BF16_HOST_DEVICE__ constexpr operator bool() const { return __x_bf16 != 0.0f; }

  /*! \brief return a casted char from underlying float val */
  __BF16_HOST_DEVICE__ operator char() const { return static_cast<char>(__x_bf16); }

  /*! \brief return a float */
  __BF16_HOST_DEVICE__ operator float() const { return static_cast<float>(__x_bf16); }

  /*! \brief return a casted int casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator int() const { return static_cast<int>(__x_bf16); }

  /*! \brief return a casted long casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator long() const { return static_cast<long>(__x_bf16); }

  /*! \brief return a casted long long casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator long long() const { return static_cast<long long>(__x_bf16); }

  /*! \brief return a casted short casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator short() const { return static_cast<short>(__x_bf16); }

  /*! \brief return a casted signed char from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator signed char() const { return static_cast<signed char>(__x_bf16); }

  /*! \brief return a casted unsigned char casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned char() const {
    return static_cast<unsigned char>(__x_bf16);
  }

  /*! \brief return a casted unsigned int casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned int() const { return static_cast<unsigned int>(__x_bf16); }

  /*! \brief return a casted unsigned from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned long() const {
    return static_cast<unsigned long>(__x_bf16);
  }

  /*! \brief return a casted unsigned long long from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned long long() const {
    return static_cast<unsigned long long>(__x_bf16);
  }

  /*! \brief return a casted unsigned short from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned short() const {
    return static_cast<unsigned short>(__x_bf16);
  }

  __BF16_HOST_DEVICE__ operator __bf16() const { return __x_bf16; }

  // TODO: SWDEV-452411 add operator which converts unsigned long long and long long to bfloat

  /*! \brief assign value from an unsigned int */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(unsigned int val) {
    __x_bf16 = static_cast<__bf16>(val);
    return *this;
  }

  /*! \brief assign value from a int */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(int val) {
    __x_bf16 = static_cast<__bf16>(val);
    return *this;
  }

  /*! \brief assign value from an unsigned short */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(unsigned short val) {
    __x_bf16 = static_cast<__bf16>(val);
    return *this;
  }

  /*! \brief assign value from a short int */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(short val) {
    __x_bf16 = static_cast<__bf16>(val);
    return *this;
  }

  /*! \brief assign value from a double */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(const double f) {
    __x_bf16 = static_cast<__bf16>(f);
    return *this;
  }

  /*! \brief assign value from a float */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(const float f) {
    __x_bf16 = static_cast<__bf16>(f);
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

typedef __bf16 __bf16_2 __attribute__((ext_vector_type(2)));

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_STRUCT
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * \brief Struct to represent a two 16 bit brain floating point number.
 * @{
 */
struct __attribute__((aligned(4))) __hip_bfloat162 {
  static_assert(sizeof(__hip_bfloat16[2]) == sizeof(__bf16_2));

 public:
  union {
    struct {
      __hip_bfloat16 x; /*! \brief raw representation of bfloat16 */
      __hip_bfloat16 y; /*! \brief raw representation of bfloat16 */
    };
    __bf16_2 __xy_bf162;
  };


 public:
  /*! \brief create __hip_bfloat162 from __hip_bfloat162_raw */
  __BF16_HOST_DEVICE__ __hip_bfloat162(const __hip_bfloat162_raw& h2r)
      : x(__hip_bfloat16(__hip_bfloat16_raw{h2r.x})),
        y(__hip_bfloat16(__hip_bfloat16_raw{h2r.y})) {}

  /*! \brief copy constructor of __hip_bfloat162 */
  __BF16_HOST_DEVICE__ __hip_bfloat162(const __hip_bfloat162& val) : x(val.x), y(val.y) {}

  /*! \brief create __hip_bfloat162 from two __hip_bfloat16 */
  __BF16_HOST_DEVICE__ constexpr __hip_bfloat162(const __hip_bfloat16& a, const __hip_bfloat16& b)
      : x(a), y(b) {}

  /*! \brief create __hip_bfloat162 from vector of __bf16_2 */
  __BF16_HOST_DEVICE__ __hip_bfloat162(const __bf16_2 in) : __xy_bf162(in) {}

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
    float2 ret(x, y);
    return ret;
  }

  /*! \brief return a vector of bf16 */
  __BF16_HOST_DEVICE__ operator __bf16_2() const { return __xy_bf162; }

  /*! \brief return a vector of bf16 */
  __BF16_HOST_DEVICE__ __hip_bfloat162& operator=(const __bf16_2 in) {
    __xy_bf162 = in;
    return *this;
  }

  /*! \brief assign value from __hip_bfloat162_raw */
  __BF16_HOST_DEVICE__ __hip_bfloat162& operator=(const __hip_bfloat162_raw& h2r) {
    x = __hip_bfloat16(__hip_bfloat16_raw{h2r.x});
    y = __hip_bfloat16(__hip_bfloat16_raw{h2r.y});
    return *this;
  }

  /*! \brief assign value from __hip_bfloat162 */
  __BF16_HOST_DEVICE__ __hip_bfloat162& operator=(const __hip_bfloat162& src) {
    x = src.x;
    y = src.y;
    return *this;
  }
};
/**@}*/

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Checks if number is nan
 */
__BF16_HOST_DEVICE_STATIC__ bool __hisnan(const __hip_bfloat16 a) {
  __hip_bfloat16_raw hr = a;
  return !(~hr.x & 0x7f80) && +(hr.x & 0x7f);
}

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
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Reinterprets bits in a __hip_bfloat16 as a signed short integer
 */
__BF16_HOST_DEVICE_STATIC__ short int __bfloat16_as_short(const __hip_bfloat16 h) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(short int));
  union {
    __hip_bfloat16 bf16;
    short int si;
  } u{h};
  return u.si;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Reinterprets bits in a __hip_bfloat16 as an unsigned signed short integer
 */
__BF16_HOST_DEVICE_STATIC__ unsigned short int __bfloat16_as_ushort(const __hip_bfloat16 h) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(unsigned short int));
  union {
    __hip_bfloat16 bf16;
    unsigned short int usi;
  } u{h};
  return u.usi;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Convert double to __hip_bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __double2bfloat16(const double a) {
  __hip_bfloat16 ret{a};
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Convert int to __hip_bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __int2bfloat16_rn(const int a) {
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
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __high2bfloat16(const __hip_bfloat162 a) { return a.y; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns high 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __high2bfloat162(const __hip_bfloat162 a) {
  return __hip_bfloat162(a.y, a.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts high 16 bits of __hip_bfloat162 to float and returns the result
 */
__BF16_HOST_DEVICE_STATIC__ float __high2float(const __hip_bfloat162 a) {
  return __bfloat162float(a.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Extracts high 16 bits from each and combines them
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __highs2bfloat162(const __hip_bfloat162 a,
                                                              const __hip_bfloat162 b) {
  return __hip_bfloat162(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns low 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __low2bfloat16(const __hip_bfloat162 a) { return a.x; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns low 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __low2bfloat162(const __hip_bfloat162 a) {
  return __hip_bfloat162(a.x, a.x);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts low 16 bits of __hip_bfloat162 to float and returns the result
 */
__BF16_HOST_DEVICE_STATIC__ float __low2float(const __hip_bfloat162 a) {
  return __bfloat162float(a.x);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Swaps both halves
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __lowhigh2highlow(const __hip_bfloat162 a) {
  return __hip_bfloat162(a.y, a.x);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Extracts low 16 bits from each and combines them
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __lows2bfloat162(const __hip_bfloat162 a,
                                                             const __hip_bfloat162 b) {
  return __hip_bfloat162(a.x, b.x);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets short int into a bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __short_as_bfloat16(const short int a) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(short int));
  union {
    short int si;
    __hip_bfloat16 bf16;
  } u{a};
  return u.bf16;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets unsigned short int into a bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __ushort_as_bfloat16(const unsigned short int a) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(unsigned short int));
  union {
    unsigned short int usi;
    __hip_bfloat16 bf16;
  } u{a};
  return u.bf16;
}

#if defined(__clang__) && defined(__HIP__)
/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_SHFL
 * \brief shfl warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__
__hip_bfloat16 __shfl(MAYBE_UNDEF __hip_bfloat16 var, int src_lane, int width = warpSize) {
  union {
    int i;
    __hip_bfloat16 f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl(tmp.i, src_lane, width);
  return tmp.f;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_SHFL
 * \brief shfl up warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__
__hip_bfloat16 __shfl_up(MAYBE_UNDEF __hip_bfloat16 var, unsigned int lane_delta,
                         int width = warpSize) {
  union {
    int i;
    __hip_bfloat16 f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_up(tmp.i, lane_delta, width);
  return tmp.f;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_SHFL
 * \brief shfl down warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__
__hip_bfloat16 __shfl_down(MAYBE_UNDEF __hip_bfloat16 var, unsigned int lane_delta,
                           int width = warpSize) {
  union {
    int i;
    __hip_bfloat16 f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_down(tmp.i, lane_delta, width);
  return tmp.f;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_SHFL
 * \brief shfl xor warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__
__hip_bfloat16 __shfl_xor(MAYBE_UNDEF __hip_bfloat16 var, int lane_mask, int width = warpSize) {
  union {
    int i;
    __hip_bfloat16 f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_xor(tmp.i, lane_mask, width);
  return tmp.f;
}

#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)
/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MOVE
 * \brief shfl down warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __shfl_down_sync(const unsigned long long mask,
                                                       const __hip_bfloat16 in,
                                                       const unsigned int delta,
                                                       const int width = warpSize) {
  return __ushort_as_bfloat16(__shfl_down_sync(mask, __bfloat16_as_ushort(in), delta, width));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MOVE
 * \brief shfl down warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __shfl_down_sync(const unsigned long long mask,
                                                        const __hip_bfloat162 in,
                                                        const unsigned int delta,
                                                        const int width = warpSize) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(unsigned int));
  union {
    __hip_bfloat162 bf162;
    unsigned int ui;
  } u{in};
  u.ui = __shfl_down_sync<unsigned long long, unsigned int>(mask, u.ui, delta, width);
  return u.bf162;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MOVE
 * \brief shfl sync warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __shfl_sync(const unsigned long long mask,
                                                  const __hip_bfloat16 in, const int delta,
                                                  const int width = warpSize) {
  return __ushort_as_bfloat16(__shfl_sync(mask, __bfloat16_as_ushort(in), delta, width));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MOVE
 * \brief shfl sync warp intrinsic for bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __shfl_sync(const unsigned long long mask,
                                                   const __hip_bfloat162 in, const int delta,
                                                   const int width = warpSize) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(unsigned int));
  union {
    __hip_bfloat162 bf162;
    unsigned int ui;
  } u{in};
  u.ui = __shfl_sync<unsigned long long, unsigned int>(mask, u.ui, delta, width);
  return u.bf162;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MOVE
 * \brief shfl up sync warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __shfl_up_sync(const unsigned long long mask,
                                                     const __hip_bfloat16 in,
                                                     const unsigned int delta,
                                                     const int width = warpSize) {
  return __ushort_as_bfloat16(__shfl_up_sync(mask, __bfloat16_as_ushort(in), delta, width));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MOVE
 * \brief shfl up sync warp intrinsic for bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __shfl_up_sync(const unsigned long long mask,
                                                      const __hip_bfloat162 in,
                                                      const unsigned int delta,
                                                      const int width = warpSize) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(unsigned int));
  union {
    __hip_bfloat162 bf162;
    unsigned int ui;
  } u{in};
  u.ui = __shfl_up_sync<unsigned long long, unsigned int>(mask, u.ui, delta, width);
  return u.bf162;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MOVE
 * \brief shfl xor sync warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __shfl_xor_sync(const unsigned long long mask,
                                                      const __hip_bfloat16 in, const int delta,
                                                      const int width = warpSize) {
  return __ushort_as_bfloat16(__shfl_xor_sync(mask, __bfloat16_as_ushort(in), delta, width));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MOVE
 * \brief shfl xor sync warp intrinsic for bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __shfl_xor_sync(const unsigned long long mask,
                                                       const __hip_bfloat162 in, const int delta,
                                                       const int width = warpSize) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(unsigned int));
  union {
    __hip_bfloat162 bf162;
    unsigned int ui;
  } u{in};
  u.ui = __shfl_xor_sync<unsigned long long, unsigned int>(mask, u.ui, delta, width);
  return u.bf162;
}
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS
#endif  // defined(__clang__) && defined(__HIP__)

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Adds two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hadd(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a + (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Adds two bfloat16 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hadd_rn(const __hip_bfloat16 a,
                                                     const __hip_bfloat16 b) {
#pragma clang fp contract(off)
  return (__bf16)a + (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Subtracts two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hsub(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a - (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Subtracts two bfloat16 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hsub_rn(const __hip_bfloat16 a,
                                                     const __hip_bfloat16 b) {
#pragma clang fp contract(off)
  return (__bf16)a - (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Divides two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hdiv(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a / (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Performs FMA of given bfloat16 values
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __hfma(const __hip_bfloat16 a, const __hip_bfloat16 b,
                                             const __hip_bfloat16 c) {
  return __hip_bfloat16(__builtin_elementwise_fma(__bf16(a), __bf16(b), __bf16(c)));
  ;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Multiplies two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmul(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a * (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Multiplies two bfloat16 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmul_rn(const __hip_bfloat16 a,
                                                     const __hip_bfloat16 b) {
#pragma clang fp contract(off)
  return (__bf16)a * (__bf16)b;
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
  return __hip_bfloat162{__bf16_2(a) / __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Returns absolute of a bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __habs2(const __hip_bfloat162 a) {
  return __hip_bfloat162{__habs(a.x), __habs(a.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Adds two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hadd2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162{__bf16_2(a) + __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Adds two bfloat162 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hadd2_rn(const __hip_bfloat162 a,
                                                       const __hip_bfloat162 b) {
#pragma clang fp contract(off)
  return __hip_bfloat162{__bf16_2(a) + __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Performs FMA of given bfloat162 values
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __hfma2(const __hip_bfloat162 a, const __hip_bfloat162 b,
                                               const __hip_bfloat162 c) {
  return __hip_bfloat162{__builtin_elementwise_fma(__bf16_2(a), __bf16_2(b), __bf16_2(c))};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Multiplies two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmul2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162{__bf16_2(a) * __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Multiplies two bfloat162 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmul2_rn(const __hip_bfloat162 a,
                                                       const __hip_bfloat162 b) {
#pragma clang fp contract(off)
  return __hip_bfloat162{__bf16_2(a) * __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Converts a bfloat162 into negative
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hneg2(const __hip_bfloat162 a) {
  return __hip_bfloat162{__hneg(a.x), __hneg(a.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Subtracts two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hsub2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162{__bf16_2(a) - __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Subtracts two bfloat162 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hsub2_rn(const __hip_bfloat162 a,
                                                       const __hip_bfloat162 b) {
#pragma clang fp contract(off)
  return __hip_bfloat162{__bf16_2(a) - __bf16_2(b)};
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
  return (__bf16)a == (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hequ(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a < (__bf16)b) && !((__bf16)a > (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - greater than
 */
__BF16_HOST_DEVICE_STATIC__ bool __hgt(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a > (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered greater than
 */
__BF16_HOST_DEVICE_STATIC__ bool __hgtu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a <= (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - greater than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hge(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a >= (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered greater than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hgeu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a < (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - not equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hne(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a != (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered not equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hneu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a == (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - return max
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmax(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  auto a_nan = __hisnan(a), b_nan = __hisnan(b);
  if (a_nan || b_nan) {
    if (a_nan && b_nan) return HIPRT_NAN_BF16;  // return canonical NaN
    return a_nan ? b : a;
  }
  return (__bf16)a > (__bf16)b ? a : b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - return min
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmin(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  auto a_nan = __hisnan(a), b_nan = __hisnan(b);
  if (a_nan || b_nan) {
    if (a_nan && b_nan) return HIPRT_NAN_BF16;  // return canonical NaN
    return a_nan ? b : a;
  }
  return (__bf16)a < (__bf16)b ? a : b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - less than operator
 */
__BF16_HOST_DEVICE_STATIC__ bool __hlt(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a < (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered less than
 */
__BF16_HOST_DEVICE_STATIC__ bool __hltu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a >= (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - less than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hle(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a <= (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered less than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hleu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a > (__bf16)b);
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
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks if two numbers are equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbeq2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __heq(a.x, b.x) && __heq(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks if two numbers are equal - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbequ2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hequ(a.x, b.x) && __hequ(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbge2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hge(a.x, b.x) && __hge(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbgeu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hgeu(a.x, b.x) && __hgeu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbgt2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hgt(a.x, b.x) && __hgt(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbgtu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hgtu(a.x, b.x) && __hgtu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hble2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hle(a.x, b.x) && __hle(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbleu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hleu(a.x, b.x) && __hleu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hblt2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hlt(a.x, b.x) && __hlt(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbltu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hltu(a.x, b.x) && __hltu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbne2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hne(a.x, b.x) && __hne(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbneu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hneu(a.x, b.x) || __hneu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b, returns 1.0 if equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __heq2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__heq(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__heq(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hge2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hge(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hge(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hgt2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hgt(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hgt(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ONE_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a is NaN, returns 1.0 if NaN, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hisnan2(const __hip_bfloat162 a) {
  return __hip_bfloat162{{__hisnan(a.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hisnan(a.y) ? HIPRT_ONE_BF16 : HIPRT_ONE_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hle2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hle(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hle(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hlt2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hlt(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hlt(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Returns max of two elements
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmax2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162(__hmax(a.x, b.x), __hmax(a.y, b.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Returns min of two elements
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmin2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162(__hmin(a.x, b.x), __hmin(a.y, b.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks for not equal to
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hne2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hne(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hne(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16}};
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
  return __float2bfloat16(__builtin_elementwise_ceil(__bfloat162float(h)));
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
  return __float2bfloat16(__builtin_elementwise_floor(__bfloat162float(h)));
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
  return __float2bfloat16(__builtin_elementwise_rint(__bfloat162float(h)));
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
  return __float2bfloat16(__builtin_elementwise_trunc(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate ceil of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2ceil(const __hip_bfloat162 h) {
  return __hip_bfloat162(hceil(h.x), hceil(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate cosine of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2cos(const __hip_bfloat162 h) {
  return __hip_bfloat162(hcos(h.x), hcos(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2exp(const __hip_bfloat162 h) {
  return __hip_bfloat162(hexp(h.x), hexp(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential 10 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2exp10(const __hip_bfloat162 h) {
  return __hip_bfloat162(hexp10(h.x), hexp10(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential 2 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2exp2(const __hip_bfloat162 h) {
  return __hip_bfloat162(hexp2(h.x), hexp2(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate floor of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2floor(const __hip_bfloat162 h) {
  return __hip_bfloat162(hfloor(h.x), hfloor(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate natural log of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2log(const __hip_bfloat162 h) {
  return __hip_bfloat162(hlog(h.x), hlog(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate log 10 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2log10(const __hip_bfloat162 h) {
  return __hip_bfloat162(hlog10(h.x), hlog10(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate log 2 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2log2(const __hip_bfloat162 h) {
  return __hip_bfloat162(hlog2(h.x), hlog2(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector reciprocal
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2rcp(const __hip_bfloat162 h) {
  return __hip_bfloat162(hrcp(h.x), hrcp(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector round to nearest int
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2rint(const __hip_bfloat162 h) {
  return __hip_bfloat162(hrint(h.x), hrint(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector reciprocal square root
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2rsqrt(const __hip_bfloat162 h) {
  return __hip_bfloat162(hrsqrt(h.x), hrsqrt(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate sin of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2sin(const __hip_bfloat162 h) {
  return __hip_bfloat162(hsin(h.x), hsin(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate sqrt of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2sqrt(const __hip_bfloat162 h) {
  return __hip_bfloat162(hsqrt(h.x), hsqrt(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate truncate of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2trunc(const __hip_bfloat162 h) {
  return __hip_bfloat162(htrunc(h.x), htrunc(h.y));
}

#if defined(__clang__) && defined(__HIP__)
/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Atomic add bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 unsafeAtomicAdd(__hip_bfloat162* address,
                                                       __hip_bfloat162 value) {
#if __has_builtin(__builtin_amdgcn_flat_atomic_fadd_v2bf16)
  typedef short __attribute__((ext_vector_type(2))) vec_short2;
  static_assert(sizeof(vec_short2) == sizeof(__hip_bfloat162_raw));
  union {
    __hip_bfloat162_raw bf162_raw;
    vec_short2 vs2;
  } u{static_cast<__hip_bfloat162_raw>(value)};
  u.vs2 = __builtin_amdgcn_flat_atomic_fadd_v2bf16((vec_short2*)address, u.vs2);
  return static_cast<__hip_bfloat162>(u.bf162_raw);
#else
  static_assert(sizeof(unsigned int) == sizeof(__hip_bfloat162_raw));
  union u_hold {
    __hip_bfloat162_raw h2r;
    unsigned int u32;
  };
  u_hold old_val, new_val;
  old_val.u32 =
      __hip_atomic_load((unsigned int*)address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  do {
    new_val.h2r = __hadd2(old_val.h2r, value);
  } while (!__hip_atomic_compare_exchange_strong((unsigned int*)address, &old_val.u32, new_val.u32,
                                                 __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT));
  return old_val.h2r;
#endif
}
__BF16_DEVICE_STATIC__ __hip_bfloat16 unsafeAtomicAdd(__hip_bfloat16* address,
                                                      __hip_bfloat16 value) {
  static_assert(sizeof(unsigned short int) == sizeof(__hip_bfloat16_raw));
  unsigned short int* address_as_short = reinterpret_cast<unsigned short int*>(address);
  // Align to 4 bytes
  unsigned int* aligned_addr = __builtin_bit_cast(
      unsigned int*, __builtin_bit_cast(unsigned long long int, address_as_short) &
                         (unsigned long long int)(~0x3));

  bool is_lower = __builtin_bit_cast(unsigned long long int, aligned_addr) ==
                  __builtin_bit_cast(unsigned long long int, address);

  __hip_bfloat162 fval;
  if (is_lower)
    fval = __halves2bfloat162(value, __float2bfloat16(0.0f));
  else
    fval = __halves2bfloat162(__float2bfloat16(0.0f), value);

  __hip_bfloat162* in = (__hip_bfloat162*)(aligned_addr);
  __hip_bfloat162 out = unsafeAtomicAdd(in, fval);
  if (is_lower) return __low2bfloat16(out);
  return __high2bfloat16(out);
}
#endif  // defined(__clang__) && defined(__HIP__)
#pragma pop_macro("MAYBE_UNDEF")
#endif
