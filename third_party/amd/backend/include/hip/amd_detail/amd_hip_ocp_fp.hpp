/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT

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

#include <hip/amd_detail/amd_hip_bf16.h>

#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_common.h>
#include <hip/amd_detail/host_defines.h>
#include <hip/amd_detail/amd_hip_ocp_types.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <climits>
#include <cstdio>

static_assert(sizeof(uint8_t) * CHAR_BIT == 8);
static_assert(sizeof(uint16_t) * CHAR_BIT == 16);
static_assert(sizeof(uint32_t) * CHAR_BIT == 32);
static_assert(sizeof(uint64_t) * CHAR_BIT == 64);
#endif  // !defined(__HIPCC_RTC__)

#include <hip/amd_detail/amd_hip_ocp_host.hpp>  // Host Conversion

// HW Detection
#if defined(__gfx950__)
#define HIP_ENABLE_GFX950_OCP_BUILTINS 1
#else
#define HIP_ENABLE_GFX950_OCP_BUILTINS 0
#endif
#if !defined(__gfx950__)
#define HIP_ENABLE_HOST_OCP_CONVERSIONS 1
#else
#define HIP_ENABLE_HOST_OCP_CONVERSIONS 0
#endif

/** FP8 format
 * +---------------+--------------------+----------------------+
 * |   Attribute   |        E4M3        |         E5M2         |
 * +---------------+--------------------+----------------------+
 * | Exponent Bias | 7                  | 15                   |
 * | Inf           | N/A                | S-11111-00           |
 * | NAN           | S-1111-111         | S-11111-**           |
 * | Zero          | S-0000-000         | S-00000-00           |
 * | Max Normal    | S-1111-110: +-448  | S-11110-11: +-57,344 |
 * | Min Normal    | S-0001-000: +-2^-6 | S-00001-00: +-2^-14  |
 * | Max Subnorm   | S-0000-111         | S-00000-11           |
 * | Min Subnorm   | S-0000-001         | S-00000-01           |
 * +---------------+--------------------+----------------------+
 */
enum __amd_fp8_interpretation_t {
  __AMD_OCP_E4M3 = 0, /* FP8 */
  __AMD_OCP_E5M2 = 1, /* BF8 */
};

/** FP6 format
 * +---------------+------------------------------------+-------------------------------------+
 * |   Attribute   |               E2M3                 |                E3M2                 |
 * +---------------+------------------------------------+-------------------------------------+
 * | Exponent bias | 1                                  | 3                                   |
 * | Infinities    | N/A                                | N/A                                 |
 * | NaN           | N/A                                | N/A                                 |
 * | Zeros         | S-00-000                           | S-000-00                            |
 * | Max normal    | S-11-111 = +-2^2 × 1.875 = +-7.5   | S-111-11 = +-2^4 × 1.75 = +-28.0    |
 * | Min normal    | S-01-000 = +-2^0 × 1.0 = +-1.0     | S-001-00 = +-2^-2 × 1.0 = +-0.25    |
 * | Max subnorm   | S-00-111 = +-2^0 × 0.875 = +-0.875 | S-000-11 = +-2^-2 × 0.75 = +-0.1875 |
 * | Min subnorm   | S-00-001 = +-2^0 × 0.125 = +-0.125 | S-000-01 = +-2^-2 × 0.25 = +-0.0625 |
 * +---------------+------------------------------------+-------------------------------------+
 */
enum __amd_fp6_interpretation_t {
  __AMD_OCP_E2M3 = 0, /* FP6 */
  __AMD_OCP_E3M2 = 1, /* BF6 */
};

/** FP4 format
 * +---------------+------------------------------+
 * |   Attribute   |            E2M1              |
 * +---------------+------------------------------+
 * | Exponent bias | 1                            |
 * | Infinities    | N/A                          |
 * | NaN           | N/A                          |
 * | Zeros         | S-00-0                       |
 * | Max normal    | S-11-1 = +-2^2 × 1.5 = +-6.0 |
 * | Min normal    | S-01-0 = +-2^0 × 1.0 = +-1.0 |
 * | Max subnorm   | S-00-1 = +-2^0 × 0.5 = +-0.5 |
 * | Min subnorm   | S-00-1 = +-2^0 × 0.5 = +-0.5 |
 * +---------------+------------------------------+
 */
enum __amd_fp4_interpretation_t {
  __AMD_OCP_E2M1 = 0, /* FP4 */
};

/**
 * @brief Create fp8x2 from two fp8 numbers
 *
 * @param x
 * @param y
 * @return __amd_fp8x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x2_storage_t
__amd_create_fp8x2(const __amd_fp8_storage_t x, const __amd_fp8_storage_t y) {
  __amd_fp8x2_storage_t ret = 0;
  ret = x | (y << 8);
  return ret;
}

/**
 * @brief Create fp4x2 from two fp4 numbers
 *
 * @param x
 * @param y
 * @return __amd_fp4x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x2_storage_t __amd_create_fp4x2(const uint8_t x,
                                                                       const uint8_t y) {
  __amd_fp4x2_storage_t ret = 0;
  ret = x | (y << 4);
  return ret;
}

/**
 * @brief Create fp4x8 from fp4x2
 *
 * @param x
 * @param y
 * @param z
 * @param w
 * @return __amd_fp4x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x8_storage_t
__amd_create_fp4x8(const __amd_fp4x2_storage_t x, const __amd_fp4x2_storage_t y,
                   const __amd_fp4x2_storage_t z, const __amd_fp4x2_storage_t w) {
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(__amd_fp4x8_storage_t));
  union {
    __amd_fp4x2_storage_t fp4x2[4];
    __amd_fp4x8_storage_t fp4x8;
  } u{{x, y, z, w}};
  return u.fp4x8;
}

/**
 * @brief Create fp8x2 from two fp8x2 numbers
 *
 * @param x
 * @param y
 * @return __amd_fp8x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x8_storage_t
__amd_create_fp8x8(const __amd_fp8x2_storage_t x, const __amd_fp8x2_storage_t y,
                   const __amd_fp8x2_storage_t z, const __amd_fp8x2_storage_t w) {
  static_assert(sizeof(__amd_fp8x2_storage_t[4]) == sizeof(__amd_fp8x8_storage_t));
  union {
    __amd_fp8x2_storage_t fp8x2[4];
    __amd_fp8x8_storage_t fp8x8;
  } u{{x, y, z, w}};
  return u.fp8x8;
}

/**
 * @brief Get fp8 number from an fp8x2
 *
 * @param x
 * @param index
 * @return __amd_fp8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8_storage_t __amd_extract_fp8(const __amd_fp8x2_storage_t x,
                                                                    const size_t index) {
  static_assert(sizeof(__amd_fp8_storage_t[2]) == sizeof(__amd_fp8x2_storage_t));
  union {
    __amd_fp8x2_storage_t fp8x2;
    __amd_fp8_storage_t fp8[2];
  } u{x};
  if (index == 0) return u.fp8[0];
  return u.fp8[1];
}

/**
 * @brief Get fp8x2 from fp8x4
 *
 * @param x
 * @param index
 * @return __amd_fp8x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x2_storage_t
__amd_extract_fp8x2(const __amd_fp8x8_storage_t x, const size_t index) {
  static_assert(sizeof(__amd_fp8x8_storage_t) == sizeof(__amd_fp8x2_storage_t[4]));
  union {
    __amd_fp8x8_storage_t fp8x8;
    __amd_fp8x2_storage_t fp8x2[4];
  } u{x};
  return u.fp8x2[index];
}

/**
 * @brief Extract a fp4 from fp4x2
 *
 * @param x __amd_fp4x2_storage_t type
 * @param index 0 or 1
 * @return uint8_t populated with 4 bits of fp4 number
 */
__OCP_FP_HOST_DEVICE_STATIC__ uint8_t __amd_extract_fp4(const __amd_fp4x2_storage_t x,
                                                        const size_t index) {
  if (index == 0) return (x & 0xFu);
  return (x >> 4);
}

/**
 * @brief extract fp4x2 from fp4x4
 *
 * @param x __amd_fp4x8_storage_t type
 * @param index 0, 1, 2 or 3 index
 * @return __amd_fp4x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x2_storage_t
__amd_extract_fp4x2(const __amd_fp4x8_storage_t x, const size_t index) {
  return (x >> (index * 8 /* char bits*/)) & 0xFFu;
}

/**
 * @brief Convert scale E8M0 to float scale which is used by gfx950
 *
 * @param scale
 * @return float embed with E8M0 in its exponent
 */
__OCP_FP_HOST_DEVICE_STATIC__ float __amd_scale_to_float(const __amd_scale_t scale_exp) {
  constexpr int8_t OCP_SCALE_EXP_NAN = -128;
  const uint32_t SCALE_EXP_BIAS = 127;  // OCP MX E8M0 "scale" bias

  // On gfx950 the "scale" operand is encoded in the exponent bits of
  // an IEEE-754 float - always in the form: 1.0 * 2**scale.
  const size_t SCALE_EXP_SHIFT = 23;  // IEEE-754 "float" mantissa bits

  uint32_t s;
  if (scale_exp == OCP_SCALE_EXP_NAN)
    s = 0xff;
  else
    s = ((uint32_t)scale_exp + SCALE_EXP_BIAS) & 0xffu;

  return fcbx::F32(s << SCALE_EXP_SHIFT);
}

/**
 * @brief Convert FP8 to float
 *
 * @param val input fp8 val
 * @param interpret
 * @return float
 */
__OCP_FP_HOST_DEVICE_STATIC__ float __amd_cvt_fp8_to_float(
    const __amd_fp8_storage_t val, const __amd_fp8_interpretation_t interpret) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (interpret == __AMD_OCP_E4M3) {
    return __builtin_amdgcn_cvt_f32_fp8(val, 0);
  } else {
    return __builtin_amdgcn_cvt_f32_bf8(val, 0);
  }
#else  // Host
  using namespace fcbx;
  if (interpret == __AMD_OCP_E4M3) {
    return to_float<float, Encoding::E4M3, true>(static_cast<uint32_t>(val), 0);
  } else {
    return to_float<float, Encoding::E5M2, true>(static_cast<uint32_t>(val), 0);
  }
#endif
}

/**
 * @brief Convert float to FP8 with stochastic rounding.
 *
 * @param val input float value.
 * @param interpret
 * @param seed
 * @return __amd_fp8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8_storage_t __amd_cvt_float_to_fp8_sr(
    const float val, const __amd_fp8_interpretation_t interpret, const unsigned int seed) {
  static_assert(sizeof(float) == sizeof(__amd_fp8_storage_t[4]));
  static_assert(sizeof(unsigned int) == sizeof(uint32_t));
  union {
    unsigned int ui32;
    uint32_t ui32t;
    float f32;
    __amd_fp8_storage_t fp8[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (interpret == __AMD_OCP_E4M3) {
    u.ui32 = __builtin_amdgcn_cvt_sr_fp8_f32(val, seed, u.ui32, 0);
  } else {
    u.ui32 = __builtin_amdgcn_cvt_sr_bf8_f32(val, seed, u.ui32, 0);
  }
#else
  using namespace fcbx;
  if (interpret == __AMD_OCP_E4M3) {
    u.ui32t = from_float_sr<float, Encoding::E4M3, true>(val, seed, 0 /*scale*/);
  } else {
    u.ui32t = from_float_sr<float, Encoding::E5M2, true>(val, seed, 0 /*scale*/);
  }
#endif
  return u.fp8[0];
}


/**
 * @brief Convert fp8 to float with scale.
 *
 * @param in fp8 number
 * @param interpret
 * @param scale
 * @return float
 */
__OCP_FP_HOST_DEVICE_STATIC__ float __amd_cvt_fp8_to_float_scale(
    const __amd_fp8_storage_t val, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  return interpret == __AMD_OCP_E4M3
             ? __builtin_amdgcn_cvt_scalef32_f32_fp8(val, __amd_scale_to_float(scale), 0)
             : __builtin_amdgcn_cvt_scalef32_f32_bf8(val, __amd_scale_to_float(scale), 0);
#else
  using namespace fcbx;
  return interpret == __AMD_OCP_E4M3
             ? to_float<float, Encoding::E4M3, true>(static_cast<uint32_t>(val), scale)
             : to_float<float, Encoding::E5M2, true>(static_cast<uint32_t>(val), scale);
#endif
}

/**
 * @brief Convert float to fp8 with stochastic rounding and scale.
 *
 * @param val
 * @param interpret
 * @param seed
 * @param scale
 * @return __amd_fp8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8_storage_t
__amd_cvt_float_to_fp8_sr_scale(const float val, const __amd_fp8_interpretation_t interpret,
                                const unsigned int seed, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp8_storage_t[4]) == sizeof(unsigned int));
  union u {
    unsigned int ui32;
    __amd_fp8_storage_t fp8[4];
  } u{0};
  if (interpret == __AMD_OCP_E4M3) {
    u.ui32 =
        __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(u.ui32, val, seed, __amd_scale_to_float(scale), 0);
  } else {
    u.ui32 =
        __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(u.ui32, val, seed, __amd_scale_to_float(scale), 0);
  }
  return u.fp8[0];
#else
  static_assert(sizeof(uint32_t) == sizeof(unsigned int));
  union u {
    uint32_t ui32t;
    __amd_fp8_storage_t fp8[4];
  } u{0};
  using namespace fcbx;
  u.ui32t = interpret == __AMD_OCP_E4M3
                ? from_float_sr<float, Encoding::E4M3, true>(val, seed, scale)
                : from_float_sr<float, Encoding::E5M2, true>(val, seed, scale);
  return u.fp8[0];
#endif
}

/**
 * @brief Convert packed fp8x2 to floatx2.
 *
 * @param val input fp8x2 value
 * @param interpret
 * @return __amd_floatx2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_floatx2_storage_t __amd_cvt_fp8x2_to_floatx2(
    const __amd_fp8x2_storage_t val, const __amd_fp8_interpretation_t interpret) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (interpret == __AMD_OCP_E4M3) {
    return __builtin_amdgcn_cvt_pk_f32_fp8(val, false);
  } else {
    return __builtin_amdgcn_cvt_pk_f32_bf8(val, false);
  }
#else
  using namespace fcbx;
  __amd_floatx2_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] =
        to_float<float, Encoding::E4M3, false>(static_cast<uint32_t>(__amd_extract_fp8(val, 0)), 0);
    ret[1] =
        to_float<float, Encoding::E4M3, false>(static_cast<uint32_t>(__amd_extract_fp8(val, 1)), 0);
  } else {
    ret[0] =
        to_float<float, Encoding::E5M2, true>(static_cast<uint32_t>(__amd_extract_fp8(val, 0)), 0);
    ret[1] =
        to_float<float, Encoding::E5M2, true>(static_cast<uint32_t>(__amd_extract_fp8(val, 1)), 0);
  }
  return ret;
#endif
}

/**
 * @brief Convert floatx2 to fp8x2.
 *
 * @param val Input floatx2 value.
 * @param interpret
 * @return __amd_fp8x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x2_storage_t __amd_cvt_floatx2_to_fp8x2(
    const __amd_floatx2_storage_t val, const __amd_fp8_interpretation_t interpret) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp8x2_storage_t[2]) == sizeof(int));
  union {
    int i32;
    __amd_fp8x2_storage_t fp8x2[2];
  } u{0};
  if (interpret == __AMD_OCP_E4M3) {
    u.i32 = __builtin_amdgcn_cvt_pk_fp8_f32(val[0], val[1], u.i32, false);
  } else {
    u.i32 = __builtin_amdgcn_cvt_pk_bf8_f32(val[0], val[1], u.i32, false);
  }
  return u.fp8x2[0];
#else
  using namespace fcbx;
  __amd_fp8_storage_t l, r;
  if (interpret == __AMD_OCP_E4M3) {
    l = from_float<float, Encoding::E4M3, true>(val[0], 0 /*scale*/);
    r = from_float<float, Encoding::E4M3, true>(val[1], 0 /*scale*/);
  } else {
    l = from_float<float, Encoding::E5M2, true>(val[0], 0 /*scale*/);
    r = from_float<float, Encoding::E5M2, true>(val[1], 0 /*scale*/);
  }
  return __amd_create_fp8x2(l, r);
#endif
}

/**
 * @brief Convert packed floatx2 to fp4x2 with stochastic rounding and scale.
 *
 * @param val
 * @param seed
 * @param scale
 * @return __amd_fp4x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x2_storage_t __amd_cvt_floatx2_to_fp4x2_sr_scale(
    const __amd_floatx2_storage_t val, const __amd_fp4_interpretation_t, const unsigned int seed,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  u.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(u.ui32, val, seed,
                                                       __amd_scale_to_float(scale), 1);
  return u.fp4x2[1];
#else
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(uint32_t));
  union u {
    uint32_t ui32t;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  using namespace fcbx;
  auto l = from_float_sr<float, Encoding::E2M1, true>(val[0], seed, scale);
  auto r = from_float_sr<float, Encoding::E2M1, true>(val[1], seed, scale);
  r <<= 4;
  l |= r;
  u.ui32t = l;
  return u.fp4x2[0];
#endif
}

/**
 * @brief Convert packed fp4x2 to floatx2. This is wrapper for gfx950 builtin.
 *
 * @param in
 * @param scale
 * @return __amd_floatx2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_floatx2_storage_t __amd_cvt_fp4x2_to_floatx2_scale(
    const __amd_fp4x2_storage_t val, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(val, __amd_scale_to_float(scale), 0);
#else
  using namespace fcbx;
  __amd_floatx2_storage_t ret{to_float<float, Encoding::E2M1, true>(val & 0xFu, scale),
                              to_float<float, Encoding::E2M1, true>(val >> 4, scale)};
  return ret;
#endif
}

/**
 * @brief Convert packed floatx2 to fp4x2.
 *
 * @param in
 * @param scale
 * @return __amd_fp4x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x2_storage_t
__amd_cvt_floatx2_to_fp4x2_scale(const __amd_floatx2_storage_t val,
                                 const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(unsigned int) == sizeof(__amd_fp4x2_storage_t[4]));
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, val[0], val[1],
                                                    __amd_scale_to_float(scale), 0);
  return u.fp4x2[0];
#else
  using namespace fcbx;
  auto l = from_float<float, Encoding::E2M1, true>(val[1], scale);
  auto r = from_float<float, Encoding::E2M1, true>(val[0], scale);
  __amd_fp4x2_storage_t ret(l << 4 | r);
  return ret;
#endif
}

/**
 * @brief Convert packed fp8x2 to floatx2. This is direct mapping of gfx950.
 *
 * @param in input fp8x2
 * @param interpret
 * @param scale
 * @return __amd_floatx2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_floatx2_storage_t __amd_cvt_fp8x2_to_floatx2_scale(
    const __amd_fp8x2_storage_t val, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  return interpret == __AMD_OCP_E4M3
             ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(val, __amd_scale_to_float(scale), false)
             : __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(val, __amd_scale_to_float(scale), false);
#else
  using namespace fcbx;
  __amd_floatx2_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = to_float<float, Encoding::E4M3, true>(val >> 8, scale);
    ret[1] = to_float<float, Encoding::E4M3, true>((val << 8) >> 8, scale);
  } else {
    ret[0] = to_float<float, Encoding::E5M2, true>(val >> 8, scale);
    ret[1] = to_float<float, Encoding::E5M2, true>((val << 8) >> 8, scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert packed floatx2 to fp8x2 with scale.
 *
 * @param in
 * @param interpret
 * @param scale
 * @return __amd_fp8x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x2_storage_t __amd_cvt_floatx2_to_fp8x2_scale(
    const __amd_floatx2_storage_t val, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_shortx2_storage_t) == sizeof(__amd_fp8x2_storage_t[2]));
  union {
    __amd_shortx2_storage_t shortx2;
    __amd_fp8x2_storage_t fp8x2[2];
  } u{0};
  u.shortx2 = interpret == __AMD_OCP_E4M3
                  ? __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(u.shortx2, val[0], val[1],
                                                             __amd_scale_to_float(scale), false)
                  : __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(u.shortx2, val[0], val[1],
                                                             __amd_scale_to_float(scale), false);
  return u.fp8x2[0];
#else
  using namespace fcbx;
  uint8_t l, r;
  if (interpret == __AMD_OCP_E4M3) {
    l = from_float<float, Encoding::E4M3, true>(val[0], scale);
    r = from_float<float, Encoding::E4M3, true>(val[1], scale);
  } else {
    l = from_float<float, Encoding::E5M2, true>(val[0], scale);
    r = from_float<float, Encoding::E5M2, true>(val[1], scale);
  }
  __amd_fp8x2_storage_t ret(l << 8 | r);
  return ret;
#endif
}

/**
 * @brief Convert packed bf16x32 to fp6x32 with scale.
 *
 * @param in
 * @param interpret
 * @param scale
 * @return __amd_fp6x32_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp6x32_storage_t __amd_cvt_bf16x32_to_fp6x32_scale(
    const __amd_bf16x32_storage_t in, const __amd_fp6_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (interpret == __AMD_OCP_E2M3) {
    return __builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16(in, __amd_scale_to_float(scale));
  }
  return __builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16(in, __amd_scale_to_float(scale));
#else
  if (interpret == __AMD_OCP_E2M3) {
    return fcbx::fp6_cvt_packedx32<__amd_bf16x32_storage_t, __amd_fp6x32_storage_t,
                                   __amd_bf16_storage_t, fcbx::Encoding::E8M7,
                                   fcbx::Encoding::E2M3>(in, scale);
  } else {
    return fcbx::fp6_cvt_packedx32<__amd_bf16x32_storage_t, __amd_fp6x32_storage_t,
                                   __amd_bf16_storage_t, fcbx::Encoding::E8M7,
                                   fcbx::Encoding::E3M2>(in, scale);
  }
#endif
}

/**
 * @brief Convert packed fp16x32 to fp6x32 with scale.
 *
 * @param in
 * @param interpret
 * @param scale
 * @return __amd_fp6x32_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp6x32_storage_t __amd_cvt_fp16x32_to_fp6x32_scale(
    const __amd_fp16x32_storage_t in, const __amd_fp6_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (interpret == __AMD_OCP_E2M3) {
    return __builtin_amdgcn_cvt_scalef32_pk32_fp6_f16(in, __amd_scale_to_float(scale));
  }
  return __builtin_amdgcn_cvt_scalef32_pk32_bf6_f16(in, __amd_scale_to_float(scale));
#else
  if (interpret == __AMD_OCP_E2M3) {
    return fcbx::fp6_cvt_packedx32<__amd_fp16x32_storage_t, __amd_fp6x32_storage_t,
                                   __amd_fp16_storage_t, fcbx::Encoding::E5M10,
                                   fcbx::Encoding::E2M3>(in, scale);
  } else {
    return fcbx::fp6_cvt_packedx32<__amd_fp16x32_storage_t, __amd_fp6x32_storage_t,
                                   __amd_fp16_storage_t, fcbx::Encoding::E5M10,
                                   fcbx::Encoding::E3M2>(in, scale);
  }
#endif
}

/**
 * @brief Convert fp8x2 to fp16x2. This is direct mapping of gfx950 builtin.
 *
 * @param val
 * @param interpret
 * @param scale
 * @return __amd_fp16x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16x2_storage_t __amd_cvt_fp8x2_to_fp16x2_scale(
    const __amd_fp8x2_storage_t val, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(unsigned int) == sizeof(__amd_fp8x2_storage_t[2]));
  union {
    __amd_fp8x2_storage_t fp8x2[2];
    unsigned int ui32;
  } u;
  u.fp8x2[0] = val;
  return interpret == __AMD_OCP_E4M3
             ? __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(u.ui32, __amd_scale_to_float(scale), false)
             : __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(u.ui32, __amd_scale_to_float(scale), false);
#else
  using namespace fcbx;
  __amd_fp16x2_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val & 0xFF, scale);
    ret[1] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val >> 8, scale);
  } else {
    ret[0] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val & 0xFF, scale);
    ret[1] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val >> 8, scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert fp8 packed 8 to fp16 packed 8
 *
 * @param val fp8x8 value
 * @param interpret interpretation of fp8
 * @param scale
 * @return __amd_fp16x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16x8_storage_t __amd_cvt_fp8x8_to_fp16x8_scale(
    const __amd_fp8x8_storage_t val, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp16x8_storage_t) == sizeof(__amd_fp16x2_storage_t[4]));
  static_assert(sizeof(__amd_fp8x8_storage_t) == sizeof(__amd_fp8x2_storage_t[4]));
  union {
    __amd_fp16x8_storage_t fp16x8;
    __amd_fp16x2_storage_t fp16x2[4];
  } ret;
  union {
    __amd_fp8x8_storage_t fp8x8;
    __amd_fp8x2_storage_t fp8x2[4];
  } input{val};
  union {
    __amd_fp8x2_storage_t fp8x2[2];
    unsigned int ui32;
  } u;

  u.fp8x2[0] = input.fp8x2[0];
  auto adjusted_scale = __amd_scale_to_float(scale);
  if (interpret == __AMD_OCP_E4M3) {
    ret.fp16x2[0] = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[1];
    ret.fp16x2[1] = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[2];
    ret.fp16x2[2] = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[3];
    ret.fp16x2[3] = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(u.ui32, adjusted_scale, false);
  } else {
    ret.fp16x2[0] = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[1];
    ret.fp16x2[1] = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[2];
    ret.fp16x2[2] = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[3];
    ret.fp16x2[3] = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(u.ui32, adjusted_scale, false);
  }
  return ret.fp16x8;
#else
  using namespace fcbx;
  __amd_fp16x8_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[0], scale);
    ret[1] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[1], scale);
    ret[2] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[2], scale);
    ret[3] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[3], scale);
    ret[4] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[4], scale);
    ret[5] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[5], scale);
    ret[6] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[6], scale);
    ret[7] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[7], scale);
  } else {
    ret[0] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[0], scale);
    ret[1] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[1], scale);
    ret[2] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[2], scale);
    ret[3] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[3], scale);
    ret[4] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[4], scale);
    ret[5] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[5], scale);
    ret[6] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[6], scale);
    ret[7] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[7], scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert fp8x2 to bf16x2 with scale. This is direct mapping of gfx950 builtin.
 *
 * @param in fp8x2 input
 * @param interpret
 * @param scale
 * @return __amd_bf16x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_bf16x2_storage_t __amd_cvt_fp8x2_to_bf16x2_scale(
    const __amd_fp8x2_storage_t in, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(unsigned int) == sizeof(__amd_fp8x2_storage_t[2]));
  union {
    __amd_fp8x2_storage_t fp8x2[2];
    unsigned int ui32;
  } u;
  u.fp8x2[0] = in;
  return interpret == __AMD_OCP_E4M3
             ? __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(u.ui32, __amd_scale_to_float(scale), false)
             : __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(u.ui32, __amd_scale_to_float(scale),
                                                         false);
#else
  using namespace fcbx;
  __amd_bf16x2_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(in & 0xFF, scale);
    ret[1] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(in >> 8, scale);
  } else {
    ret[0] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(in & 0xFF, scale);
    ret[1] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(in >> 8, scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert fp8 packed 8 to bf16 packed 8.
 *
 * @param val fp8x8 value
 * @param interpret interpretation of fp8
 * @param scale
 * @return __amd_bf16x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_bf16x8_storage_t __amd_cvt_fp8x8_to_bf16x8_scale(
    const __amd_fp8x8_storage_t val, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_bf16x8_storage_t) == sizeof(__amd_bf16x2_storage_t[4]));
  static_assert(sizeof(__amd_fp8x8_storage_t) == sizeof(__amd_fp8x2_storage_t[4]));
  union {
    __amd_bf16x8_storage_t bf16x8;
    __amd_bf16x2_storage_t bf16x2[4];
  } ret;
  union {
    __amd_fp8x8_storage_t fp8x8;
    __amd_fp8x2_storage_t fp8x2[4];
  } input{val};
  union {
    __amd_fp8x2_storage_t fp8x2[2];
    unsigned int ui32;
  } u;

  auto adjusted_scale = __amd_scale_to_float(scale);
  if (interpret == __AMD_OCP_E4M3) {
    u.fp8x2[0] = input.fp8x2[0];
    ret.bf16x2[0] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[1];
    ret.bf16x2[1] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[2];
    ret.bf16x2[2] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[3];
    ret.bf16x2[3] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(u.ui32, adjusted_scale, false);
  } else {
    u.fp8x2[0] = input.fp8x2[0];
    ret.bf16x2[0] = __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[1];
    ret.bf16x2[1] = __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[2];
    ret.bf16x2[2] = __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(u.ui32, adjusted_scale, false);
    u.fp8x2[0] = input.fp8x2[3];
    ret.bf16x2[3] = __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(u.ui32, adjusted_scale, false);
  }
  return ret.bf16x8;
#else
  using namespace fcbx;
  __amd_bf16x8_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[0], scale);
    ret[1] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[1], scale);
    ret[2] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[2], scale);
    ret[3] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[3], scale);
    ret[4] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[4], scale);
    ret[5] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[5], scale);
    ret[6] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[6], scale);
    ret[7] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[7], scale);
  } else {
    ret[0] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[0], scale);
    ret[1] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[1], scale);
    ret[2] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[2], scale);
    ret[3] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[3], scale);
    ret[4] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[4], scale);
    ret[5] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[5], scale);
    ret[6] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[6], scale);
    ret[7] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[7], scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert packed fp6x32 to fp16x32 with scale.
 *
 * @param in fp6x32 value
 * @param interpret
 * @param scale
 * @return __amd_fp16x32_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16x32_storage_t __amd_cvt_fp6x32_to_fp16x32_scale(
    const __amd_fp6x32_storage_t in, const __amd_fp6_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  // gfx950 expects scale to be in float
  return interpret == __AMD_OCP_E2M3
             ? __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6(in, __amd_scale_to_float(scale))
             : __builtin_amdgcn_cvt_scalef32_pk32_f16_bf6(in, __amd_scale_to_float(scale));
#else
  using namespace fcbx;
  if (interpret == __AMD_OCP_E2M3) {
    return fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_fp16x32_storage_t, __amd_fp16_storage_t,
                             Encoding::E2M3, Encoding::E5M10>(in, scale);
  } else {
    return fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_fp16x32_storage_t, __amd_fp16_storage_t,
                             Encoding::E3M2, Encoding::E5M10>(in, scale);
  }
#endif
}

/**
 * @brief Convert packed fp6x32 to bf16x32 with scale.
 *
 * @param in fp6x32 value
 * @param interpret
 * @param scale
 * @return __amd_bf16x32_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_bf16x32_storage_t __amd_cvt_fp6x32_to_bf16x32_scale(
    const __amd_fp6x32_storage_t in, const __amd_fp6_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  return interpret == __AMD_OCP_E2M3
             ? __builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6(in, __amd_scale_to_float(scale))
             : __builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6(in, __amd_scale_to_float(scale));
#else
  using namespace fcbx;
  if (interpret == __AMD_OCP_E2M3) {
    return fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_bf16x32_storage_t, __amd_bf16_storage_t,
                             Encoding::E2M3, Encoding::E8M7>(in, scale);
  } else {
    return fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_bf16x32_storage_t, __amd_bf16_storage_t,
                             Encoding::E3M2, Encoding::E8M7>(in, scale);
  }
#endif
}

__OCP_FP_HOST_DEVICE_STATIC__ __amd_floatx32_storage_t __amd_cvt_fp6x32_to_floatx32_scale(
    const __amd_fp6x32_storage_t val, const __amd_fp6_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  return interpret == __AMD_OCP_E2M3
             ? __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(val, __amd_scale_to_float(scale))
             : __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(val, __amd_scale_to_float(scale));
#else
  using namespace fcbx;
  return interpret == __AMD_OCP_E2M3
             ? fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_floatx32_storage_t, float,
                                 Encoding::E2M3, Encoding::IEEE754>(val, scale)
             : fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_floatx32_storage_t, float,
                                 Encoding::E3M2, Encoding::IEEE754>(val, scale);
#endif
}

/**
 * @brief Convert packed 2 of fp4 to fp16.
 *
 * @param in packed fp4x2
 * @param scale
 * @return __amd_fp16x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16x2_storage_t __amd_cvt_fp4x2_to_fp16x2_scale(
    const __amd_fp4x2_storage_t in, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(in, __amd_scale_to_float(scale), 0);
#else
  using namespace fcbx;
  __amd_fp16x2_storage_t ret{to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(in & 0xFu, scale),
                             to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(in >> 4, scale)};
  return ret;
#endif
}

/**
 * @brief convert packed fp4x8 to fp16 x16.
 *
 * @param in
 * @param scale
 * @return __amd_fp16x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16x8_storage_t __amd_cvt_fp4x8_to_fp16x8_scale(
    const __amd_fp4x8_storage_t in, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  static_assert(sizeof(__amd_fp16x2_storage_t[4]) == sizeof(__amd_fp16x8_storage_t));
  union {
    __amd_fp4x8_storage_t fp4x8;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{in};
  union {
    __amd_fp16x8_storage_t fp16x8;
    __amd_fp16x2_storage_t fp16x2[4];
  } ret;
  ret.fp16x2[0] =
      __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(u.fp4x2[0], __amd_scale_to_float(scale), 0);
  ret.fp16x2[1] =
      __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(u.fp4x2[1], __amd_scale_to_float(scale), 0);
  ret.fp16x2[2] =
      __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(u.fp4x2[2], __amd_scale_to_float(scale), 0);
  ret.fp16x2[3] =
      __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(u.fp4x2[3], __amd_scale_to_float(scale), 0);
  return ret.fp16x8;
#else
  using namespace fcbx;
  __amd_fp16x8_storage_t ret;
  ret[0] = to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(in & 0xFu, scale);
  ret[1] = to_float<__amd_fp16_storage_t, Encoding::E2M1, true>((in >> 4) & 0xFu, scale);
  ret[2] = to_float<__amd_fp16_storage_t, Encoding::E2M1, true>((in >> 8) & 0xFu, scale);
  ret[3] = to_float<__amd_fp16_storage_t, Encoding::E2M1, true>((in >> 12) & 0xFu, scale);
  ret[4] = to_float<__amd_fp16_storage_t, Encoding::E2M1, true>((in >> 16) & 0xFu, scale);
  ret[5] = to_float<__amd_fp16_storage_t, Encoding::E2M1, true>((in >> 20) & 0xFu, scale);
  ret[6] = to_float<__amd_fp16_storage_t, Encoding::E2M1, true>((in >> 24) & 0xFu, scale);
  ret[7] = to_float<__amd_fp16_storage_t, Encoding::E2M1, true>((in >> 28) & 0xFu, scale);
  return ret;
#endif
}

/**
 * @brief Convert packed fp4x2 to bf16x2.
 *
 * @param in
 * @param scale
 * @return __OCP_FP_DEVICE_STATIC__
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_bf16x2_storage_t __amd_cvt_fp4x2_to_bf16x2_scale(
    const __amd_fp4x2_storage_t in, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(in, __amd_scale_to_float(scale), 0);
#else
  using namespace fcbx;
  __amd_bf16x2_storage_t ret{to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(in & 0xFu, scale),
                             to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(in >> 4, scale)};
  return ret;
#endif
}

/**
 * @brief Convert packed fp4x8 to bf16x8.
 *
 * @param in
 * @param scale
 * @return __amd_bf16x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_bf16x8_storage_t __amd_cvt_fp4x8_to_bf16x8_scale(
    const __amd_fp4x8_storage_t in, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  static_assert(sizeof(__amd_bf16x2_storage_t[4]) == sizeof(__amd_bf16x8_storage_t));
  union {
    __amd_fp4x8_storage_t fp4x8;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{in};
  union {
    __amd_bf16x8_storage_t bf16x8;
    __amd_bf16x2_storage_t bf16x2[4];
  } ret;
  // Unrolled
  ret.bf16x2[0] =
      __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(u.fp4x2[0], __amd_scale_to_float(scale), 0);
  ret.bf16x2[1] =
      __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(u.fp4x2[1], __amd_scale_to_float(scale), 0);
  ret.bf16x2[2] =
      __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(u.fp4x2[2], __amd_scale_to_float(scale), 0);
  ret.bf16x2[3] =
      __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(u.fp4x2[3], __amd_scale_to_float(scale), 0);
  return ret.bf16x8;
#else
  using namespace fcbx;
  __amd_bf16x8_storage_t ret;
  ret[0] = to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(in & 0xFu, scale);
  ret[1] = to_float<__amd_bf16_storage_t, Encoding::E2M1, true>((in >> 4) & 0xFu, scale);
  ret[2] = to_float<__amd_bf16_storage_t, Encoding::E2M1, true>((in >> 8) & 0xFu, scale);
  ret[3] = to_float<__amd_bf16_storage_t, Encoding::E2M1, true>((in >> 12) & 0xFu, scale);
  ret[4] = to_float<__amd_bf16_storage_t, Encoding::E2M1, true>((in >> 16) & 0xFu, scale);
  ret[5] = to_float<__amd_bf16_storage_t, Encoding::E2M1, true>((in >> 20) & 0xFu, scale);
  ret[6] = to_float<__amd_bf16_storage_t, Encoding::E2M1, true>((in >> 24) & 0xFu, scale);
  ret[7] = to_float<__amd_bf16_storage_t, Encoding::E2M1, true>((in >> 28) & 0xFu, scale);
  return ret;
#endif
}

/**
 * @brief Convert packed fp4x8 to floatx8 with scale.
 *
 * @param val
 * @param scale
 * @return __amd_floatx8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_floatx8_storage_t __amd_cvt_fp4x8_to_floatx8_scale(
    const __amd_fp4x8_storage_t val, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(__amd_fp4x8_storage_t));
  union {
    __amd_fp4x8_storage_t fp4x8;
    __amd_fp4x2_storage_t fp8x2[4];
  } u{val};
  __amd_floatx2_storage_t op;
  __amd_floatx8_storage_t ret;
  op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(u.fp8x2[0], __amd_scale_to_float(scale), 0);
  ret[0] = op[0];
  ret[1] = op[1];
  op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(u.fp8x2[1], __amd_scale_to_float(scale), 0);
  ret[2] = op[0];
  ret[3] = op[1];
  op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(u.fp8x2[2], __amd_scale_to_float(scale), 0);
  ret[4] = op[0];
  ret[5] = op[1];
  op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(u.fp8x2[3], __amd_scale_to_float(scale), 0);
  ret[6] = op[0];
  ret[7] = op[1];
  return ret;
#else
  using namespace fcbx;
  __amd_floatx8_storage_t ret;
  ret[0] = to_float<float, Encoding::E2M1, true>(val & 0xFu, scale);
  ret[1] = to_float<float, Encoding::E2M1, true>((val >> 4) & 0xFu, scale);
  ret[2] = to_float<float, Encoding::E2M1, true>((val >> 8) & 0xFu, scale);
  ret[3] = to_float<float, Encoding::E2M1, true>((val >> 12) & 0xFu, scale);
  ret[4] = to_float<float, Encoding::E2M1, true>((val >> 16) & 0xFu, scale);
  ret[5] = to_float<float, Encoding::E2M1, true>((val >> 20) & 0xFu, scale);
  ret[6] = to_float<float, Encoding::E2M1, true>((val >> 24) & 0xFu, scale);
  ret[7] = to_float<float, Encoding::E2M1, true>((val >> 28) & 0xFu, scale);
  return ret;
#endif
}

/**
 * @brief Convert packed floatx8 to fp4x8.
 *
 * @param in
 * @param interpret
 * @param scale
 * @return __amd_fp4x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x8_storage_t __amd_cvt_floatx8_to_fp4x8_scale(
    const __amd_floatx8_storage_t in, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(unsigned int) == sizeof(__amd_fp4x2_storage_t[4]));
  static_assert(sizeof(__amd_fp4x8_storage_t) == sizeof(__amd_fp4x2_storage_t[4]));
  union hold_u {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
    __amd_fp4x8_storage_t fp4x8;
  } ret{0}, tmp{0};
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp.ui32, in[0], in[1],
                                                      __amd_scale_to_float(scale), 0);
  ret.fp4x2[0] = tmp.fp4x2[0];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp.ui32, in[2], in[3],
                                                      __amd_scale_to_float(scale), 0);
  ret.fp4x2[1] = tmp.fp4x2[0];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp.ui32, in[4], in[5],
                                                      __amd_scale_to_float(scale), 0);
  ret.fp4x2[2] = tmp.fp4x2[0];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp.ui32, in[6], in[7],
                                                      __amd_scale_to_float(scale), 0);
  ret.fp4x2[3] = tmp.fp4x2[0];
  return ret.fp4x8;
#else
  __amd_fp4x8_storage_t ret = 0;
  using namespace fcbx;
  auto tmp = from_float<float, Encoding::E2M1, true>(in[7], scale);
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(in[6], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(in[5], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(in[4], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(in[3], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(in[2], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(in[1], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(in[0], scale);
  ret <<= 4;
  ret |= tmp;
  return ret;
#endif
}

/**
 * @brief Convert packed fp16x2 to fp8x2 with scale.
 *
 * @param in
 * @param interpret
 * @param scale
 * @return __amd_fp8x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x2_storage_t __amd_cvt_fp16x2_to_fp8x2_scale(
    const __amd_fp16x2_storage_t in, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_shortx2_storage_t) == sizeof(__amd_fp8x2_storage_t[2]));
  static_assert(sizeof(uint32_t) == sizeof(__amd_fp8x2_storage_t[2]));
  union {
    __amd_shortx2_storage_t shortx2;
    __amd_fp8x2_storage_t fp8x2[2];
  } u{0};
  u.shortx2 = interpret == __AMD_OCP_E4M3 ? __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(
                                                u.shortx2, in, __amd_scale_to_float(scale), false)
                                          : __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(
                                                u.shortx2, in, __amd_scale_to_float(scale), false);
  return u.fp8x2[0];
#else
  static_assert(sizeof(__amd_fp8x2_storage_t[2]) == sizeof(uint32_t));
  using namespace fcbx;
  union {
    uint32_t ui32;
    __amd_fp8x2_storage_t fp8x2[2];
  } u{0};
  if (interpret == __AMD_OCP_E4M3) {
    u.ui32 = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(in[1], scale);
    u.ui32 <<= 8;
    u.ui32 |= from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(in[0], scale);
  } else {
    u.ui32 = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(in[1], scale);
    u.ui32 <<= 8;
    u.ui32 |= from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(in[0], scale);
  }
  return u.fp8x2[0];
#endif
}

/**
 * @brief Convert packed bf16x2 to fp8x2 with scale.
 *
 * @param in
 * @param interpret
 * @param scale
 * @return __amd_fp8x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x2_storage_t __amd_cvt_bf16x2_to_fp8x2_scale(
    const __amd_bf16x2_storage_t in, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_shortx2_storage_t) == sizeof(__amd_fp8x2_storage_t[2]));
  union {
    __amd_shortx2_storage_t shortx2;
    __amd_fp8x2_storage_t fp8x2[2];
  } u{0};
  u.shortx2 = interpret == __AMD_OCP_E4M3 ? __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(
                                                u.shortx2, in, __amd_scale_to_float(scale), false)
                                          : __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16(
                                                u.shortx2, in, __amd_scale_to_float(scale), false);
  return u.fp8x2[0];
#else
  using namespace fcbx;
  union {
    uint32_t ui32;
    __amd_fp8x2_storage_t fp8x2[2];
  } u{0};
  if (interpret == __AMD_OCP_E4M3) {
    u.ui32 = from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(in[1], scale);
    u.ui32 <<= 8;
    u.ui32 |= from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(in[0], scale);
  } else {
    u.ui32 = from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(in[1], scale);
    u.ui32 <<= 8;
    u.ui32 |= from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(in[0], scale);
  }
  return u.fp8x2[0];
#endif
}

/**
 * @brief Convert bf16 pack 8 to fp8 packed 8
 *
 * @param val bf16x8 value
 * @param interpret interpretation of fp8
 * @param scale
 * @return __amd_fp8x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x8_storage_t __amd_cvt_bf16x8_to_fp8x8_scale(
    const __amd_bf16x8_storage_t val, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp8x2_storage_t[4]) == sizeof(__amd_fp8x8_storage_t));
  static_assert(sizeof(__amd_fp8x2_storage_t[2]) == sizeof(unsigned int));
  union {
    __amd_shortx2_storage_t shortx2;
    __amd_fp8x2_storage_t fp8x2[2];
    unsigned int ui32;
  } u{0};
  union {
    __amd_fp8x2_storage_t fp8x2[4];
    __amd_fp8x8_storage_t fp8x8;
  } result;
  __amd_shortx2_storage_t t_shortx2{0, 0};
  if (interpret == __AMD_OCP_E4M3) {
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(
        t_shortx2, __amd_bf16x2_storage_t{val[0], val[1]}, __amd_scale_to_float(scale), false);
    result.fp8x2[0] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(
        t_shortx2, __amd_bf16x2_storage_t{val[2], val[3]}, __amd_scale_to_float(scale), false);
    result.fp8x2[1] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(
        t_shortx2, __amd_bf16x2_storage_t{val[4], val[5]}, __amd_scale_to_float(scale), false);
    result.fp8x2[2] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(
        t_shortx2, __amd_bf16x2_storage_t{val[6], val[7]}, __amd_scale_to_float(scale), false);
    result.fp8x2[3] = u.fp8x2[0];
  } else {
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16(
        t_shortx2, __amd_bf16x2_storage_t{val[0], val[1]}, __amd_scale_to_float(scale), false);
    result.fp8x2[0] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16(
        t_shortx2, __amd_bf16x2_storage_t{val[2], val[3]}, __amd_scale_to_float(scale), false);
    result.fp8x2[1] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16(
        t_shortx2, __amd_bf16x2_storage_t{val[4], val[5]}, __amd_scale_to_float(scale), false);
    result.fp8x2[2] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16(
        t_shortx2, __amd_bf16x2_storage_t{val[6], val[7]}, __amd_scale_to_float(scale), false);
    result.fp8x2[3] = u.fp8x2[0];
  }
  return result.fp8x8;
#else
  using namespace fcbx;
  __amd_fp8x8_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[0], scale);
    ret[1] = from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[1], scale);
    ret[2] = from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[2], scale);
    ret[3] = from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[3], scale);
    ret[4] = from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[4], scale);
    ret[5] = from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[5], scale);
    ret[6] = from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[6], scale);
    ret[7] = from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val[7], scale);
  } else {
    ret[0] = from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[0], scale);
    ret[1] = from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[1], scale);
    ret[2] = from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[2], scale);
    ret[3] = from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[3], scale);
    ret[4] = from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[4], scale);
    ret[5] = from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[5], scale);
    ret[6] = from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[6], scale);
    ret[7] = from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val[7], scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert fp8 packed 8 to float packed 8.
 *
 * @param val fp8x8 value
 * @param interpret interpretation of fp8
 * @param scale
 * @return __amd_floatx8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_floatx8_storage_t __amd_cvt_fp8x8_to_floatx8_scale(
    const __amd_fp8x8_storage_t val, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp8x8_storage_t) == sizeof(__amd_fp8x2_storage_t[4]));
  union {
    __amd_fp8x8_storage_t fp8x8;
    __amd_fp8x2_storage_t fp8x2[4];
  } u{val};
  __amd_floatx8_storage_t ret;
  __amd_floatx2_storage_t out;
  if (interpret == __AMD_OCP_E4M3) {
    out = __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(u.fp8x2[0], __amd_scale_to_float(scale), false);
    ret[0] = out[0];
    ret[1] = out[1];
    out = __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(u.fp8x2[1], __amd_scale_to_float(scale), false);
    ret[2] = out[0];
    ret[3] = out[1];
    out = __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(u.fp8x2[2], __amd_scale_to_float(scale), false);
    ret[4] = out[0];
    ret[5] = out[1];
    out = __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(u.fp8x2[3], __amd_scale_to_float(scale), false);
    ret[6] = out[0];
    ret[7] = out[1];
  } else {
    out = __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(u.fp8x2[0], __amd_scale_to_float(scale), false);
    ret[0] = out[0];
    ret[1] = out[1];
    out = __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(u.fp8x2[1], __amd_scale_to_float(scale), false);
    ret[2] = out[0];
    ret[3] = out[1];
    out = __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(u.fp8x2[2], __amd_scale_to_float(scale), false);
    ret[4] = out[0];
    ret[5] = out[1];
    out = __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(u.fp8x2[3], __amd_scale_to_float(scale), false);
    ret[6] = out[0];
    ret[7] = out[1];
  }
  return ret;
#else
  using namespace fcbx;
  __amd_floatx8_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = to_float<float, Encoding::E4M3, true>(val[0], scale);
    ret[1] = to_float<float, Encoding::E4M3, true>(val[1], scale);
    ret[2] = to_float<float, Encoding::E4M3, true>(val[2], scale);
    ret[3] = to_float<float, Encoding::E4M3, true>(val[3], scale);
    ret[4] = to_float<float, Encoding::E4M3, true>(val[4], scale);
    ret[5] = to_float<float, Encoding::E4M3, true>(val[5], scale);
    ret[6] = to_float<float, Encoding::E4M3, true>(val[6], scale);
    ret[7] = to_float<float, Encoding::E4M3, true>(val[7], scale);
  } else {
    ret[0] = to_float<float, Encoding::E5M2, true>(val[0], scale);
    ret[1] = to_float<float, Encoding::E5M2, true>(val[1], scale);
    ret[2] = to_float<float, Encoding::E5M2, true>(val[2], scale);
    ret[3] = to_float<float, Encoding::E5M2, true>(val[3], scale);
    ret[4] = to_float<float, Encoding::E5M2, true>(val[4], scale);
    ret[5] = to_float<float, Encoding::E5M2, true>(val[5], scale);
    ret[6] = to_float<float, Encoding::E5M2, true>(val[6], scale);
    ret[7] = to_float<float, Encoding::E5M2, true>(val[7], scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert fp8 to fp16 with scale.
 *
 * @param val
 * @param interpret
 * @param scale
 * @return __amd_fp16_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16_storage_t
__amd_cvt_fp8_to_fp16_scale(const __amd_fp8_storage_t val,
                            const __amd_fp8_interpretation_t interpret, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_fp16x2_storage_t ret;
  ret =
      interpret == __AMD_OCP_E4M3
          ? __builtin_amdgcn_cvt_scalef32_f16_fp8(ret, val, __amd_scale_to_float(scale), 0, false)
          : __builtin_amdgcn_cvt_scalef32_f16_bf8(ret, val, __amd_scale_to_float(scale), 0, false);
  return ret[0];
#else
  using namespace fcbx;
  if (interpret == __AMD_OCP_E4M3) {
    return to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val, scale);
  } else {
    return to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val, scale);
  }
#endif
}

/**
 * @brief Convert fp8 to bf16 with scale.
 *
 * @param val
 * @param interpret
 * @param scale
 * @return __amd_bf16_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_bf16_storage_t
__amd_cvt_fp8_to_bf16_scale(const __amd_fp8_storage_t val,
                            const __amd_fp8_interpretation_t interpret, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(unsigned int) == sizeof(__amd_fp8x2_storage_t[2]));
  static_assert(sizeof(__amd_fp8_storage_t[4]) == sizeof(__amd_fp8x2_storage_t[2]));
  union {
    __amd_fp8_storage_t fp8[4];
    __amd_fp8x2_storage_t fp8x2[2];
    unsigned int ui32;
  } u{0};
  u.fp8[0] = val;
  auto ret =
      interpret == __AMD_OCP_E4M3
          ? __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(u.ui32, __amd_scale_to_float(scale), false)
          : __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(u.ui32, __amd_scale_to_float(scale), false);
  return ret[0];
#else
  using namespace fcbx;
  if (interpret == __AMD_OCP_E4M3) {
    return to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(val, scale);
  } else {
    return to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(val, scale);
  }
#endif
}

/**
 * @brief Convert two packed float16x16 to fp6x32.
 *
 * @param in1
 * @param in2
 * @param interpret
 * @param scale
 * @return __amd_fp6x32_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp6x32_storage_t __amd_cvt_floatx16_floatx16_to_fp6x32_scale(
    const __amd_floatx16_storage_t in1, const __amd_floatx16_storage_t in2,
    const __amd_fp6_interpretation_t interpret, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  return interpret == __AMD_OCP_E2M3
             ? __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, __amd_scale_to_float(scale))
             : __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, __amd_scale_to_float(scale));
#else
  __amd_floatx32_storage_t tmp;
  for (size_t i = 0; i < 16; i++) {
    tmp[i] = in1[i];
  }
  for (size_t i = 0; i < 16; i++) {
    tmp[i + 16] = in2[i];
  }
  using namespace fcbx;
  return interpret == __AMD_OCP_E2M3
             ? fp6_cvt_packedx32<__amd_floatx32_storage_t, __amd_fp6x32_storage_t, float,
                                 Encoding::IEEE754, Encoding::E2M3>(tmp, scale)
             : fp6_cvt_packedx32<__amd_floatx32_storage_t, __amd_fp6x32_storage_t, float,
                                 Encoding::IEEE754, Encoding::E3M2>(tmp, scale);
#endif
}

/**
 * @brief Convert packed floatx32 to fp6x32.
 *
 * @param in1
 * @param in2
 * @param interpret
 * @param scale
 * @return __amd_fp6x32_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp6x32_storage_t __amd_cvt_floatx32_to_fp6x32_scale(
    const __amd_floatx32_storage_t val, const __amd_fp6_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_floatx16_storage_t in1{val[0],  val[1],  val[2],  val[3], val[4],  val[5],
                               val[6],  val[7],  val[8],  val[9], val[10], val[11],
                               val[12], val[13], val[14], val[15]},
      in2 = {val[16], val[17], val[18], val[19], val[20], val[21], val[22], val[23],
             val[24], val[25], val[26], val[27], val[28], val[29], val[30], val[31]};
  return interpret == __AMD_OCP_E2M3
             ? __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, __amd_scale_to_float(scale))
             : __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, __amd_scale_to_float(scale));
#else
  using namespace fcbx;
  return interpret == __AMD_OCP_E2M3
             ? fp6_cvt_packedx32<__amd_floatx32_storage_t, __amd_fp6x32_storage_t, float,
                                 Encoding::IEEE754, Encoding::E2M3>(val, scale)
             : fp6_cvt_packedx32<__amd_floatx32_storage_t, __amd_fp6x32_storage_t, float,
                                 Encoding::IEEE754, Encoding::E3M2>(val, scale);
#endif
}

/**
 * @brief Convert packed floatx32 to fp6x32 with stochastic rounding and scale.
 *
 * @param val
 * @param interpret
 * @param round
 * @param scale
 * @return __amd_fp6x32_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp6x32_storage_t __amd_cvt_floatx32_to_fp6x32_sr_scale(
    const __amd_floatx32_storage_t val, const __amd_fp6_interpretation_t interpret,
    const unsigned int round, const __amd_scale_t scale) {
#if __has_builtin(__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32) and                               \
    __has_builtin(__builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f32)
  return interpret == __AMD_OCP_E2M3 ? __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32(
                                           val, round, __amd_scale_to_float(scale))
                                     : __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f32(
                                           val, round, __amd_scale_to_float(scale));
#else
  using namespace fcbx;
  return interpret == __AMD_OCP_E2M3
             ? fp6_cvt_packedx32<__amd_floatx32_storage_t, __amd_fp6x32_storage_t, float,
                                 Encoding::IEEE754, Encoding::E2M3, true>(val, scale, round)
             : fp6_cvt_packedx32<__amd_floatx32_storage_t, __amd_fp6x32_storage_t, float,
                                 Encoding::IEEE754, Encoding::E3M2, true>(val, scale, round);
#endif
}

/**
 * @brief Convert float to fp16 with stochastic rounding.
 *
 * @param in input float val
 * @param round
 * @return __amd_fp16_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16_storage_t
__amd_cvt_float_to_fp16_sr(const float in, const unsigned int round) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_fp16x2_storage_t ret;
  ret = __builtin_amdgcn_cvt_sr_f16_f32(ret, in, round, 0);
  return ret[0];
#else
  __builtin_trap();
#endif
}

/**
 * @brief Convert two float inputs to fp16x2.
 *
 * @param in1 input float val
 * @param in2 input float val
 * @param round
 * @return __amd_fp16x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16x2_storage_t
__amd_cvt_float_float_to_fp16x2_sr(const float in1, const float in2, const unsigned int round) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_fp16x2_storage_t ret1, ret2;
  ret1 = __builtin_amdgcn_cvt_sr_f16_f32(ret1, in1, round, 0);
  ret2 = __builtin_amdgcn_cvt_sr_f16_f32(ret2, in2, round, 0);
  return __amd_fp16x2_storage_t{ret1[0], ret2[0]};
#else
  __builtin_trap();
#endif
}

/**
 * @brief Convert float to bfloat16 with stochastic rounding.
 *
 * @param in
 * @param round
 * @return __amd_bf16_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_bf16_storage_t
__amd_cvt_float_to_bf16_sr(const float in, const unsigned int round) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_bf16x2_storage_t ret;
  ret = __builtin_amdgcn_cvt_sr_bf16_f32(ret, in, round, 0);
  return ret[0];
#else
  __builtin_trap();
#endif
}

/**
 * @brief Convert packed fp16x32 to fp6x32 with stochastic rounding and scale.
 *
 * @param in
 * @param interpret
 * @param round
 * @param scale
 * @return __amd_fp6x32_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp6x32_storage_t __amd_cvt_fp16x32_to_fp6x32_sr_scale(
    const __amd_fp16x32_storage_t in, const __amd_fp6_interpretation_t interpret,
    const unsigned int round, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  return interpret == __AMD_OCP_E2M3
             ? __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f16(in, round, __amd_scale_to_float(scale))
             : __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f16(in, round,
                                                             __amd_scale_to_float(scale));
#else
  return interpret == __AMD_OCP_E2M3
             ? fcbx::fp6_cvt_packedx32<__amd_fp16x32_storage_t, __amd_fp6x32_storage_t,
                                       __amd_fp16_storage_t, fcbx::Encoding::E5M10,
                                       fcbx::Encoding::E2M3, true>(in, scale, round)
             : fcbx::fp6_cvt_packedx32<__amd_fp16x32_storage_t, __amd_fp6x32_storage_t,
                                       __amd_fp16_storage_t, fcbx::Encoding::E5M10,
                                       fcbx::Encoding::E3M2, true>(in, scale, round);
#endif
}

__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp6x32_storage_t __amd_cvt_bf16x32_to_fp6x32_sr_scale(
    const __amd_bf16x32_storage_t in, const __amd_fp6_interpretation_t interpret,
    const unsigned int round, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  return interpret == __AMD_OCP_E2M3 ? __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_bf16(
                                           in, round, __amd_scale_to_float(scale))
                                     : __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_bf16(
                                           in, round, __amd_scale_to_float(scale));
#else
  return interpret == __AMD_OCP_E2M3
             ? fcbx::fp6_cvt_packedx32<__amd_bf16x32_storage_t, __amd_fp6x32_storage_t,
                                       __amd_bf16_storage_t, fcbx::Encoding::E8M7,
                                       fcbx::Encoding::E2M3, true>(in, scale, round)
             : fcbx::fp6_cvt_packedx32<__amd_bf16x32_storage_t, __amd_fp6x32_storage_t,
                                       __amd_bf16_storage_t, fcbx::Encoding::E8M7,
                                       fcbx::Encoding::E3M2, true>(in, scale, round);
#endif
}

/**
 * @brief Convert packed bf16x2 to fp4x2 with scale.
 *
 * @param val
 * @param scale
 * @return __amd_fp4x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x2_storage_t __amd_cvt_bf16x2_to_fp4x2_scale(
    const __amd_bf16x2_storage_t val, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u.ui32, val, __amd_scale_to_float(scale), 0);
  return u.fp4x2[0];
#else
  using namespace fcbx;
  __amd_fp4x2_storage_t ret;
  ret = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(val[1], scale);
  ret <<= 4;
  ret |= from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(val[0], scale);
  return ret;
#endif
}

/**
 * @brief Convert packed bf16x8 to fp4x8.
 *
 * @param val
 * @param scale
 * @return __amd_fp4x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x8_storage_t __amd_cvt_bf16x8_to_fp4x8_scale(
    const __amd_bf16x8_storage_t val, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(__amd_fp4x8_storage_t));
  union hold_u {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
    __amd_fp4x8_storage_t fp4x8;
  } ret{0}, tmp{0};
  __amd_bf16x2_storage_t tmp_in{val[0], val[1]};
  tmp.ui32 =
      __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(tmp.ui32, tmp_in, __amd_scale_to_float(scale), 0);
  ret.fp4x2[0] = tmp.fp4x2[0];
  tmp_in[0] = val[2];
  tmp_in[1] = val[3];
  tmp.ui32 =
      __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(tmp.ui32, tmp_in, __amd_scale_to_float(scale), 0);
  ret.fp4x2[1] = tmp.fp4x2[0];
  tmp_in[0] = val[4];
  tmp_in[1] = val[5];
  tmp.ui32 =
      __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(tmp.ui32, tmp_in, __amd_scale_to_float(scale), 0);
  ret.fp4x2[2] = tmp.fp4x2[0];
  tmp_in[0] = val[6];
  tmp_in[1] = val[7];
  tmp.ui32 =
      __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(tmp.ui32, tmp_in, __amd_scale_to_float(scale), 0);
  ret.fp4x2[3] = tmp.fp4x2[0];
  return ret.fp4x8;
#else
  __amd_fp4x8_storage_t ret = 0;
  using namespace fcbx;
  auto tmp = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(val[7], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(val[6], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(val[5], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(val[4], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(val[3], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(val[2], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(val[1], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(val[0], scale);
  ret <<= 4;
  ret |= tmp;
  return ret;
#endif
}

/**
 * @brief Convert packed fp16x2 to fp4x2.
 *
 * @param val
 * @param scale
 * @return __amd_fp4x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x2_storage_t __amd_cvt_fp16x2_to_fp4x2_scale(
    const __amd_fp16x2_storage_t val, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(u.ui32, val, __amd_scale_to_float(scale), 0);
  return u.fp4x2[0];
#else
  using namespace fcbx;
  __amd_fp4x2_storage_t ret;
  ret = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(val[1], scale);
  ret <<= 4;
  ret |= from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(val[0], scale);
  return ret;
#endif
}

/**
 * @brief Convert packed fp16x8 to fp4x8.
 *
 * @param val
 * @param scale
 * @return __amd_fp4x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x8_storage_t __amd_cvt_fp16x8_to_fp4x8_scale(
    const __amd_fp16x8_storage_t val, const __amd_fp4_interpretation_t, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(__amd_fp4x8_storage_t));
  union hold_u {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
    __amd_fp4x8_storage_t fp4x8;
  } ret{0}, tmp{0};
  __amd_fp16x2_storage_t tmp_in{val[0], val[1]};
  tmp.ui32 =
      __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(tmp.ui32, tmp_in, __amd_scale_to_float(scale), 0);
  ret.fp4x2[0] = tmp.fp4x2[0];
  tmp_in[0] = val[2];
  tmp_in[1] = val[3];
  tmp.ui32 =
      __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(tmp.ui32, tmp_in, __amd_scale_to_float(scale), 0);
  ret.fp4x2[1] = tmp.fp4x2[0];
  tmp_in[0] = val[4];
  tmp_in[1] = val[5];
  tmp.ui32 =
      __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(tmp.ui32, tmp_in, __amd_scale_to_float(scale), 0);
  ret.fp4x2[2] = tmp.fp4x2[0];
  tmp_in[0] = val[6];
  tmp_in[1] = val[7];
  tmp.ui32 =
      __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(tmp.ui32, tmp_in, __amd_scale_to_float(scale), 0);
  ret.fp4x2[3] = tmp.fp4x2[0];
  return ret.fp4x8;
#else
  __amd_fp4x8_storage_t ret = 0;
  using namespace fcbx;
  auto tmp = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(val[7], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(val[6], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(val[5], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(val[4], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(val[3], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(val[2], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(val[1], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(val[0], scale);
  ret <<= 4;
  ret |= tmp;
  return ret;
#endif
}

/**
 * @brief Convert floatx8 to fp4x8 with stochastic rounding and scale.
 *
 * @param val
 * @param seed
 * @param scale
 * @return __amd_fp4x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x8_storage_t __amd_cvt_floatx8_to_fp4x8_sr_scale(
    const __amd_floatx8_storage_t val, const __amd_fp4_interpretation_t, const unsigned int seed,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } tmp{0};
  union {
    __amd_fp4x2_storage_t fp4x2[4];
    __amd_fp4x8_storage_t fp4x8;
  } ret;
  __amd_floatx2_storage_t in{val[0], val[1]};
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(tmp.ui32, in, seed,
                                                         __amd_scale_to_float(scale), 1);
  ret.fp4x2[0] = tmp.fp4x2[1];
  in[0] = val[2];
  in[1] = val[3];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(tmp.ui32, in, seed,
                                                         __amd_scale_to_float(scale), 1);
  ret.fp4x2[1] = tmp.fp4x2[1];
  in[0] = val[4];
  in[1] = val[5];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(tmp.ui32, in, seed,
                                                         __amd_scale_to_float(scale), 1);
  ret.fp4x2[2] = tmp.fp4x2[1];
  in[0] = val[6];
  in[1] = val[7];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(tmp.ui32, in, seed,
                                                         __amd_scale_to_float(scale), 1);
  ret.fp4x2[3] = tmp.fp4x2[1];
  return ret.fp4x8;
#else
  __amd_fp4x8_storage_t ret = 0;
  using namespace fcbx;
  auto tmp = from_float<float, Encoding::E2M1, true>(val[7], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(val[6], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(val[5], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(val[4], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(val[3], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(val[2], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(val[1], scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float<float, Encoding::E2M1, true>(val[0], scale);
  ret <<= 4;
  ret |= tmp;
  return ret;
#endif
}

/**
 * @brief Convert packed bf16x2 to fp4x2 with stochastic rounding and scale.
 *
 * @param val
 * @param seed
 * @param scale
 * @return __amd_fp4x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x2_storage_t __amd_cvt_bf16x2_to_fp4x2_sr_scale(
    const __amd_bf16x2_storage_t val, const __amd_fp4_interpretation_t, const unsigned int seed,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  u.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16(u.ui32, val, seed,
                                                        __amd_scale_to_float(scale), 1);
  return u.fp4x2[1];
#else
  __amd_fp4x2_storage_t ret;
  using namespace fcbx;
  ret = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(val[1], seed, scale);
  ret <<= 4;
  ret |= from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(val[0], seed, scale);
  return ret;
#endif
}

/**
 * @brief Convert bf16x8 to fp4x8 with stochastic rounding and scale.
 *
 * @param val
 * @param seed
 * @param scale
 * @return __amd_fp4x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x8_storage_t __amd_cvt_bf16x8_to_fp4x8_sr_scale(
    const __amd_bf16x8_storage_t val, const __amd_fp4_interpretation_t, const unsigned int seed,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } tmp{0};
  union {
    __amd_fp4x2_storage_t fp4x2[4];
    __amd_fp4x8_storage_t fp4x8;
  } ret;
  __amd_bf16x2_storage_t in{val[0], val[1]};
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16(tmp.ui32, in, seed,
                                                          __amd_scale_to_float(scale), 1);
  ret.fp4x2[0] = tmp.fp4x2[1];
  in[0] = val[2];
  in[1] = val[3];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16(tmp.ui32, in, seed,
                                                          __amd_scale_to_float(scale), 1);
  ret.fp4x2[1] = tmp.fp4x2[1];
  in[0] = val[4];
  in[1] = val[5];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16(tmp.ui32, in, seed,
                                                          __amd_scale_to_float(scale), 1);
  ret.fp4x2[2] = tmp.fp4x2[1];
  in[0] = val[6];
  in[1] = val[7];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16(tmp.ui32, in, seed,
                                                          __amd_scale_to_float(scale), 1);
  ret.fp4x2[3] = tmp.fp4x2[1];
  return ret.fp4x8;
#else
  __amd_fp4x8_storage_t ret = 0;
  using namespace fcbx;
  auto tmp = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(val[7], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(val[6], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(val[5], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(val[4], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(val[3], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(val[2], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(val[1], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(val[0], seed, scale);
  ret <<= 4;
  ret |= tmp;
  return ret;
#endif
}

/**
 * @brief Convert packed fp16x2 to fp4x2 with stochastic rounding and scale.
 *
 * @param val
 * @param seed
 * @param scale
 * @return __amd_fp4x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x2_storage_t __amd_cvt_fp16x2_to_fp4x2_sr_scale(
    const __amd_fp16x2_storage_t val, const __amd_fp4_interpretation_t, const unsigned int seed,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  u.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16(u.ui32, val, seed,
                                                       __amd_scale_to_float(scale), 1);
  return u.fp4x2[1];
#else
  __amd_fp4x2_storage_t ret;
  using namespace fcbx;
  ret = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(val[1], seed, scale);
  ret <<= 4;
  ret |= from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(val[0], seed, scale);
  return ret;
#endif
}

/**
 * @brief Convert fp16x8 to fp4x8 with stochastic rounding and scale.
 *
 * @param val
 * @param seed
 * @param scale
 * @return __amd_fp4x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp4x8_storage_t __amd_cvt_fp16x8_to_fp4x8_sr_scale(
    const __amd_fp16x8_storage_t val, const __amd_fp4_interpretation_t, const unsigned int seed,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } tmp{0};
  union {
    __amd_fp4x2_storage_t fp4x2[4];
    __amd_fp4x8_storage_t fp4x8;
  } ret;
  __amd_fp16x2_storage_t in{val[0], val[1]};
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16(tmp.ui32, in, seed,
                                                         __amd_scale_to_float(scale), 1);
  ret.fp4x2[0] = tmp.fp4x2[1];
  in[0] = val[2];
  in[1] = val[3];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16(tmp.ui32, in, seed,
                                                         __amd_scale_to_float(scale), 1);
  ret.fp4x2[1] = tmp.fp4x2[1];
  in[0] = val[4];
  in[1] = val[5];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16(tmp.ui32, in, seed,
                                                         __amd_scale_to_float(scale), 1);
  ret.fp4x2[2] = tmp.fp4x2[1];
  in[0] = val[6];
  in[1] = val[7];
  tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16(tmp.ui32, in, seed,
                                                         __amd_scale_to_float(scale), 1);
  ret.fp4x2[3] = tmp.fp4x2[1];
  return ret.fp4x8;
#else
  __amd_fp4x8_storage_t ret = 0;
  using namespace fcbx;
  auto tmp = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(val[7], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(val[6], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(val[5], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(val[4], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(val[3], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(val[2], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(val[1], seed, scale);
  ret <<= 4;
  ret |= tmp;
  tmp = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(val[0], seed, scale);
  ret <<= 4;
  ret |= tmp;
  return ret;
#endif
}

/**
 * @brief Convert packed floatx8 to fp8x8 with stochastic rounding and scale.
 *
 * @param val
 * @param interpret
 * @param seed
 * @param scale
 * @return __amd_fp8x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x8_storage_t __amd_cvt_floatx8_to_fp8x8_sr_scale(
    const __amd_floatx8_storage_t val, const __amd_fp8_interpretation_t interpret,
    const unsigned int seed, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp8_storage_t[4]) == sizeof(unsigned int));
  __amd_fp8x8_storage_t ret;
  union hold_u {
    unsigned int ui32;
    __amd_fp8_storage_t fp8[4];
  } tmp{0};
  if (interpret == __AMD_OCP_E4M3) {
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(tmp.ui32, val[0], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[0] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(tmp.ui32, val[1], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[1] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(tmp.ui32, val[2], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[2] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(tmp.ui32, val[3], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[3] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(tmp.ui32, val[4], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[4] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(tmp.ui32, val[5], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[5] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(tmp.ui32, val[6], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[6] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(tmp.ui32, val[7], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[7] = tmp.fp8[0];
  } else {
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(tmp.ui32, val[0], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[0] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(tmp.ui32, val[1], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[1] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(tmp.ui32, val[2], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[2] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(tmp.ui32, val[3], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[3] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(tmp.ui32, val[4], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[4] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(tmp.ui32, val[5], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[5] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(tmp.ui32, val[6], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[6] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(tmp.ui32, val[7], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[7] = tmp.fp8[0];
  }
  return ret;
#else
  using namespace fcbx;
  __amd_fp8x8_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = from_float_sr<float, Encoding::E4M3, true>(val[0], seed, scale);
    ret[1] = from_float_sr<float, Encoding::E4M3, true>(val[1], seed, scale);
    ret[2] = from_float_sr<float, Encoding::E4M3, true>(val[2], seed, scale);
    ret[3] = from_float_sr<float, Encoding::E4M3, true>(val[3], seed, scale);
    ret[4] = from_float_sr<float, Encoding::E4M3, true>(val[4], seed, scale);
    ret[5] = from_float_sr<float, Encoding::E4M3, true>(val[5], seed, scale);
    ret[6] = from_float_sr<float, Encoding::E4M3, true>(val[6], seed, scale);
    ret[7] = from_float_sr<float, Encoding::E4M3, true>(val[7], seed, scale);
  } else {
    ret[0] = from_float_sr<float, Encoding::E5M2, true>(val[0], seed, scale);
    ret[1] = from_float_sr<float, Encoding::E5M2, true>(val[1], seed, scale);
    ret[2] = from_float_sr<float, Encoding::E5M2, true>(val[2], seed, scale);
    ret[3] = from_float_sr<float, Encoding::E5M2, true>(val[3], seed, scale);
    ret[4] = from_float_sr<float, Encoding::E5M2, true>(val[4], seed, scale);
    ret[5] = from_float_sr<float, Encoding::E5M2, true>(val[5], seed, scale);
    ret[6] = from_float_sr<float, Encoding::E5M2, true>(val[6], seed, scale);
    ret[7] = from_float_sr<float, Encoding::E5M2, true>(val[7], seed, scale);
  }
  return ret;
#endif
}

/**
 * @brief Convret fp16 to fp8 with stochastic rounding and scale.
 *
 * @param val
 * @param interpret
 * @param seed
 * @param scale
 * @return __amd_fp8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8_storage_t __amd_cvt_fp16_to_fp8_sr_scale(
    const __amd_fp16_storage_t val, const __amd_fp8_interpretation_t interpret,
    const unsigned int seed, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp8_storage_t[4]) == sizeof(unsigned int));
  union u {
    unsigned int ui32;
    __amd_fp8_storage_t fp8[4];
  } u{0};
  if (interpret == __AMD_OCP_E4M3) {
    u.ui32 =
        __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(u.ui32, val, seed, __amd_scale_to_float(scale), 0);
  } else {
    u.ui32 =
        __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(u.ui32, val, seed, __amd_scale_to_float(scale), 0);
  }
  return u.fp8[0];
#else
  using namespace fcbx;
  if (interpret == __AMD_OCP_E4M3) {
    return from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(val, seed, scale);
  } else {
    return from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(val, seed, scale);
  }
#endif
}

/**
 * @brief Convert packed fp16x8 to fp8x8 with stochastic rounding and scale.
 *
 * @param val
 * @param interpret
 * @param seed
 * @param scale
 * @return __amd_fp8x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x8_storage_t __amd_cvt_fp16x8_to_fp8x8_sr_scale(
    const __amd_fp16x8_storage_t val, const __amd_fp8_interpretation_t interpret,
    const unsigned int seed, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp8_storage_t[4]) == sizeof(unsigned int));
  __amd_fp8x8_storage_t ret;
  union hold_u {
    unsigned int ui32;
    __amd_fp8_storage_t fp8[4];
  } tmp{0};
  if (interpret == __AMD_OCP_E4M3) {
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(tmp.ui32, val[0], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[0] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(tmp.ui32, val[1], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[1] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(tmp.ui32, val[2], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[2] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(tmp.ui32, val[3], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[3] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(tmp.ui32, val[4], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[4] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(tmp.ui32, val[5], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[5] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(tmp.ui32, val[6], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[6] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(tmp.ui32, val[7], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[7] = tmp.fp8[0];
  } else {
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(tmp.ui32, val[0], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[0] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(tmp.ui32, val[1], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[1] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(tmp.ui32, val[2], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[2] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(tmp.ui32, val[3], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[3] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(tmp.ui32, val[4], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[4] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(tmp.ui32, val[5], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[5] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(tmp.ui32, val[6], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[6] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(tmp.ui32, val[7], seed,
                                                        __amd_scale_to_float(scale), 0);
    ret[7] = tmp.fp8[0];
  }
  return ret;
#else
  using namespace fcbx;
  __amd_fp8x8_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(val[0], seed, scale);
    ret[1] = from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(val[1], seed, scale);
    ret[2] = from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(val[2], seed, scale);
    ret[3] = from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(val[3], seed, scale);
    ret[4] = from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(val[4], seed, scale);
    ret[5] = from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(val[5], seed, scale);
    ret[6] = from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(val[6], seed, scale);
    ret[7] = from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(val[7], seed, scale);
  } else {
    ret[0] = from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(val[0], seed, scale);
    ret[1] = from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(val[1], seed, scale);
    ret[2] = from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(val[2], seed, scale);
    ret[3] = from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(val[3], seed, scale);
    ret[4] = from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(val[4], seed, scale);
    ret[5] = from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(val[5], seed, scale);
    ret[6] = from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(val[6], seed, scale);
    ret[7] = from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(val[7], seed, scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert bf16 to fp8 with stochastic rounding and scale
 *
 * @param val
 * @param interpret
 * @param seed
 * @param scale
 * @return __amd_fp8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8_storage_t __amd_cvt_bf16_to_fp8_sr_scale(
    const __amd_bf16_storage_t val, const __amd_fp8_interpretation_t interpret,
    const unsigned int seed, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp8_storage_t[4]) == sizeof(unsigned int));
  union u {
    unsigned int ui32;
    __amd_fp8_storage_t fp8[4];
  } u{0};
  if (interpret == __AMD_OCP_E4M3) {
    u.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(u.ui32, val, seed,
                                                       __amd_scale_to_float(scale), 0);
  } else {
    u.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(u.ui32, val, seed,
                                                       __amd_scale_to_float(scale), 0);
  }
  return u.fp8[0];
#else
  using namespace fcbx;
  if (interpret == __AMD_OCP_E4M3) {
    return from_float_sr<__amd_bf16_storage_t, Encoding::E4M3, true>(val, seed, scale);
  } else {
    return from_float_sr<__amd_bf16_storage_t, Encoding::E5M2, true>(val, seed, scale);
  }
#endif
}

/**
 * @brief Convert packed bf16x8 to fp8x8 with stochastic rounding and scale.
 *
 * @param val
 * @param interpret
 * @param seed
 * @param scale
 * @return __amd_fp8x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x8_storage_t __amd_cvt_bf16x8_to_fp8x8_sr_scale(
    const __amd_bf16x8_storage_t val, const __amd_fp8_interpretation_t interpret,
    const unsigned int seed, const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp8_storage_t[4]) == sizeof(unsigned int));
  __amd_fp8x8_storage_t ret;
  union hold_u {
    unsigned int ui32;
    __amd_fp8_storage_t fp8[4];
  } tmp{0};
  if (interpret == __AMD_OCP_E4M3) {
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(/*tmp.ui32*/ 0, val[0], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[0] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(/*tmp.ui32*/ 0, val[1], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[1] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(/*tmp.ui32*/ 0, val[2], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[2] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(/*tmp.ui32*/ 0, val[3], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[3] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(/*tmp.ui32*/ 0, val[4], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[4] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(/*tmp.ui32*/ 0, val[5], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[5] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(/*tmp.ui32*/ 0, val[6], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[6] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(/*tmp.ui32*/ 0, val[7], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[7] = tmp.fp8[0];
  } else {
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(/*tmp.ui32*/ 0, val[0], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[0] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(/*tmp.ui32*/ 0, val[1], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[1] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(/*tmp.ui32*/ 0, val[2], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[2] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(/*tmp.ui32*/ 0, val[3], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[3] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(/*tmp.ui32*/ 0, val[4], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[4] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(/*tmp.ui32*/ 0, val[5], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[5] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(/*tmp.ui32*/ 0, val[6], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[6] = tmp.fp8[0];
    tmp.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(/*tmp.ui32*/ 0, val[7], seed,
                                                         __amd_scale_to_float(scale), 0);
    ret[7] = tmp.fp8[0];
  }
  return ret;
#else
  using namespace fcbx;
  __amd_fp8x8_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = from_float_sr<__amd_bf16_storage_t, Encoding::E4M3, true>(val[0], seed, scale);
    ret[1] = from_float_sr<__amd_bf16_storage_t, Encoding::E4M3, true>(val[1], seed, scale);
    ret[2] = from_float_sr<__amd_bf16_storage_t, Encoding::E4M3, true>(val[2], seed, scale);
    ret[3] = from_float_sr<__amd_bf16_storage_t, Encoding::E4M3, true>(val[3], seed, scale);
    ret[4] = from_float_sr<__amd_bf16_storage_t, Encoding::E4M3, true>(val[4], seed, scale);
    ret[5] = from_float_sr<__amd_bf16_storage_t, Encoding::E4M3, true>(val[5], seed, scale);
    ret[6] = from_float_sr<__amd_bf16_storage_t, Encoding::E4M3, true>(val[6], seed, scale);
    ret[7] = from_float_sr<__amd_bf16_storage_t, Encoding::E4M3, true>(val[7], seed, scale);
  } else {
    ret[0] = from_float_sr<__amd_bf16_storage_t, Encoding::E5M2, true>(val[0], seed, scale);
    ret[1] = from_float_sr<__amd_bf16_storage_t, Encoding::E5M2, true>(val[1], seed, scale);
    ret[2] = from_float_sr<__amd_bf16_storage_t, Encoding::E5M2, true>(val[2], seed, scale);
    ret[3] = from_float_sr<__amd_bf16_storage_t, Encoding::E5M2, true>(val[3], seed, scale);
    ret[4] = from_float_sr<__amd_bf16_storage_t, Encoding::E5M2, true>(val[4], seed, scale);
    ret[5] = from_float_sr<__amd_bf16_storage_t, Encoding::E5M2, true>(val[5], seed, scale);
    ret[6] = from_float_sr<__amd_bf16_storage_t, Encoding::E5M2, true>(val[6], seed, scale);
    ret[7] = from_float_sr<__amd_bf16_storage_t, Encoding::E5M2, true>(val[7], seed, scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert fp8 to fp16.
 *
 * @param val fp8 number
 * @param interpret interpretation of fp8
 * @return __amd_fp16_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16_storage_t
__amd_cvt_fp8_to_fp16(const __amd_fp8_storage_t val, const __amd_fp8_interpretation_t interpret) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_fp16x2_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret = __builtin_amdgcn_cvt_scalef32_f16_fp8(ret, val, __amd_scale_to_float(0), 0, false);
  } else {
    ret = __builtin_amdgcn_cvt_scalef32_f16_bf8(ret, val, __amd_scale_to_float(0), 0, false);
  }
  return ret[0];
#else
  using namespace fcbx;
  if (interpret == __AMD_OCP_E4M3) {
    return to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val, 0);
  } else {
    return to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val, 0);
  }
#endif
}

/**
 * @brief Convert packed fp16 vector of size 2 to fp8 vector of size 2
 *
 * @param val fp8x2 value
 * @param interpret interpretation of fp8
 * @return __amd_fp16x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16x2_storage_t __amd_cvt_fp8x2_to_fp16x2(
    const __amd_fp8x2_storage_t val, const __amd_fp8_interpretation_t interpret) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(unsigned int) == sizeof(__amd_fp8x2_storage_t[2]));
  union {
    __amd_fp8x2_storage_t fp8x2[2];
    unsigned int ui32;
  } u;
  u.fp8x2[0] = val;
  return interpret == __AMD_OCP_E4M3
             ? __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(u.ui32, __amd_scale_to_float(0), false)
             : __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(u.ui32, __amd_scale_to_float(0), false);
#else
  using namespace fcbx;
  __amd_fp16x2_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val & 0xFF, 0);
    ret[1] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val >> 8, 0);
  } else {
    ret[0] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val & 0xFF, 0);
    ret[1] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val >> 8, 0);
  }
  return ret;
#endif
}

/**
 * @brief Convert packed fp16 of size 2 to fp8
 *
 * @param val fp8x2 value
 * @param interpret interpretation of fp8
 * @return __amd_fp8x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x2_storage_t __amd_cvt_fp16x2_to_fp8x2(
    const __amd_fp16x2_storage_t val, const __amd_fp8_interpretation_t interpret) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_shortx2_storage_t) == sizeof(__amd_fp8x2_storage_t[2]));
  union {
    __amd_shortx2_storage_t shortx2;
    __amd_fp8x2_storage_t fp8x2[2];
  } u{0};
  u.shortx2 = interpret == __AMD_OCP_E4M3 ? __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(
                                                u.shortx2, val, __amd_scale_to_float(0), false)
                                          : __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(
                                                u.shortx2, val, __amd_scale_to_float(0), false);
  return u.fp8x2[0];
#else
  using namespace fcbx;
  __amd_fp8x2_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[1], 0);
    ret <<= 8;
    ret |= from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[0], 0);
  } else {
    ret = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[1], 0);
    ret <<= 8;
    ret |= from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[0], 0);
  }
  return ret;
#endif
}

/**
 * @brief Convert fp16 pack 8 to fp8 packed 8
 *
 * @param val fp16x8 value
 * @param interpret interpretation of fp8
 * @param scale
 * @return __amd_fp8x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x8_storage_t __amd_cvt_fp16x8_to_fp8x8_scale(
    const __amd_fp16x8_storage_t val, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_shortx2_storage_t) == sizeof(__amd_fp8x2_storage_t[2]));
  static_assert(sizeof(__amd_fp8x8_storage_t) == sizeof(__amd_fp8x2_storage_t[4]));
  union {
    __amd_shortx2_storage_t shortx2;
    __amd_fp8x2_storage_t fp8x2[2];
  } tmp{0};
  union {
    __amd_fp8x2_storage_t fp8x2[4];
    __amd_fp8x8_storage_t fp8x8;
  } ret;
  __amd_fp16x2_storage_t in{val[0], val[1]};
  if (interpret == __AMD_OCP_E4M3) {
    tmp.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(tmp.shortx2, in,
                                                           __amd_scale_to_float(scale), false);
    ret.fp8x2[0] = tmp.fp8x2[0];
    in[0] = val[2];
    in[1] = val[3];
    tmp.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(tmp.shortx2, in,
                                                           __amd_scale_to_float(scale), false);
    ret.fp8x2[1] = tmp.fp8x2[0];
    in[0] = val[4];
    in[1] = val[5];
    tmp.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(tmp.shortx2, in,
                                                           __amd_scale_to_float(scale), false);
    ret.fp8x2[2] = tmp.fp8x2[0];
    in[0] = val[6];
    in[1] = val[7];
    tmp.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(tmp.shortx2, in,
                                                           __amd_scale_to_float(scale), false);
    ret.fp8x2[3] = tmp.fp8x2[0];
  } else {
    tmp.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(tmp.shortx2, in,
                                                           __amd_scale_to_float(scale), false);
    ret.fp8x2[0] = tmp.fp8x2[0];
    in[0] = val[2];
    in[1] = val[3];
    tmp.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(tmp.shortx2, in,
                                                           __amd_scale_to_float(scale), false);
    ret.fp8x2[1] = tmp.fp8x2[0];
    in[0] = val[4];
    in[1] = val[5];
    tmp.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(tmp.shortx2, in,
                                                           __amd_scale_to_float(scale), false);
    ret.fp8x2[2] = tmp.fp8x2[0];
    in[0] = val[6];
    in[1] = val[7];
    tmp.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(tmp.shortx2, in,
                                                           __amd_scale_to_float(scale), false);
    ret.fp8x2[3] = tmp.fp8x2[0];
  }
  return ret.fp8x8;
#else
  __amd_fp8x8_storage_t ret;
  using namespace fcbx;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[0], scale);
    ret[1] = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[1], scale);
    ret[2] = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[2], scale);
    ret[3] = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[3], scale);
    ret[4] = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[4], scale);
    ret[5] = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[5], scale);
    ret[6] = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[6], scale);
    ret[7] = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(val[7], scale);
  } else {
    ret[0] = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[0], scale);
    ret[1] = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[1], scale);
    ret[2] = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[2], scale);
    ret[3] = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[3], scale);
    ret[4] = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[4], scale);
    ret[5] = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[5], scale);
    ret[6] = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[6], scale);
    ret[7] = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(val[7], scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert float pack 8 to fp8 packed 8
 *
 * @param val floatx8 value
 * @param interpret interpretation of fp8
 * @param scale
 * @return __amd_fp8x8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8x8_storage_t __amd_cvt_floatx8_to_fp8x8_scale(
    const __amd_floatx8_storage_t val, const __amd_fp8_interpretation_t interpret,
    const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_shortx2_storage_t) == sizeof(__amd_fp8x2_storage_t[2]));
  static_assert(sizeof(__amd_fp8x8_storage_t) == sizeof(__amd_fp8x2_storage_t[4]));
  union {
    __amd_shortx2_storage_t shortx2;
    __amd_fp8x2_storage_t fp8x2[2];
  } u{0};
  union {
    __amd_fp8x2_storage_t fp8x2[4];
    __amd_fp8x8_storage_t fp8x8;
  } ret;
  if (interpret == __AMD_OCP_E4M3) {
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(u.shortx2, val[0], val[1],
                                                         __amd_scale_to_float(scale), false);
    ret.fp8x2[0] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(u.shortx2, val[2], val[3],
                                                         __amd_scale_to_float(scale), false);
    ret.fp8x2[1] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(u.shortx2, val[4], val[5],
                                                         __amd_scale_to_float(scale), false);
    ret.fp8x2[2] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(u.shortx2, val[6], val[7],
                                                         __amd_scale_to_float(scale), false);
    ret.fp8x2[3] = u.fp8x2[0];
  } else {
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(u.shortx2, val[0], val[1],
                                                         __amd_scale_to_float(scale), false);
    ret.fp8x2[0] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(u.shortx2, val[2], val[3],
                                                         __amd_scale_to_float(scale), false);
    ret.fp8x2[1] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(u.shortx2, val[4], val[5],
                                                         __amd_scale_to_float(scale), false);
    ret.fp8x2[2] = u.fp8x2[0];
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(u.shortx2, val[6], val[7],
                                                         __amd_scale_to_float(scale), false);
    ret.fp8x2[3] = u.fp8x2[0];
  }
  return ret.fp8x8;
#else
  using namespace fcbx;
  __amd_fp8x8_storage_t ret;
  if (interpret == __AMD_OCP_E4M3) {
    ret[0] = from_float<float, Encoding::E4M3, true>(val[0], scale);
    ret[1] = from_float<float, Encoding::E4M3, true>(val[1], scale);
    ret[2] = from_float<float, Encoding::E4M3, true>(val[2], scale);
    ret[3] = from_float<float, Encoding::E4M3, true>(val[3], scale);
    ret[4] = from_float<float, Encoding::E4M3, true>(val[4], scale);
    ret[5] = from_float<float, Encoding::E4M3, true>(val[5], scale);
    ret[6] = from_float<float, Encoding::E4M3, true>(val[6], scale);
    ret[7] = from_float<float, Encoding::E4M3, true>(val[7], scale);
  } else {
    ret[0] = from_float<float, Encoding::E5M2, true>(val[0], scale);
    ret[1] = from_float<float, Encoding::E5M2, true>(val[1], scale);
    ret[2] = from_float<float, Encoding::E5M2, true>(val[2], scale);
    ret[3] = from_float<float, Encoding::E5M2, true>(val[3], scale);
    ret[4] = from_float<float, Encoding::E5M2, true>(val[4], scale);
    ret[5] = from_float<float, Encoding::E5M2, true>(val[5], scale);
    ret[6] = from_float<float, Encoding::E5M2, true>(val[6], scale);
    ret[7] = from_float<float, Encoding::E5M2, true>(val[7], scale);
  }
  return ret;
#endif
}

/**
 * @brief Convert fp16 to fp8 with stochastic rounding.
 *
 * @param val
 * @param interpret
 * @param sr
 * @return __amd_fp8_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp8_storage_t __amd_cvt_fp16_to_fp8_sr(
    const __amd_fp16_storage_t val, const __amd_fp8_interpretation_t interpret, const short sr) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  static_assert(sizeof(__amd_fp8_storage_t[4]) == sizeof(unsigned int));
  union u {
    unsigned int ui32;
    __amd_fp8_storage_t fp8[4];
  } u{0};
  if (interpret == __AMD_OCP_E4M3) {
    u.ui32 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(u.ui32, val, sr, __amd_scale_to_float(0), 0);
  } else {
    u.ui32 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(u.ui32, val, sr, __amd_scale_to_float(0), 0);
  }
  return u.fp8[0];
#else
  using namespace fcbx;
  return interpret == __AMD_OCP_E4M3
             ? from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(val, sr, 0)
             : from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(val, sr, 0);
#endif
}

/**
 * @brief Convert __amd_floatx2_storage_t to hip float2 type
 *
 * @param val
 * @return float2
 */
__OCP_FP_HOST_DEVICE_STATIC__ float2
__amd_cvt_floatx2_to_float2(const __amd_floatx2_storage_t val) {
  return float2{val[0], val[1]};
}

/**
 * @brief Convert __amd_fp16_storage_t type to __half type
 *
 * @param val
 * @return __half
 */
__OCP_FP_HOST_DEVICE_STATIC__ __half __amd_cvt_fp16_to_half(const __amd_fp16_storage_t val) {
  __half_raw hr;
  hr.data = val;
  return hr;
}

/**
 * @brief Convert __amd_fp16x2_storage_t to __half2 type
 *
 * @param val
 * @return __half2
 */
__OCP_FP_HOST_DEVICE_STATIC__ __half2 __amd_cvt_fp16x2_to_half2(const __amd_fp16x2_storage_t val) {
  __half2_raw hr;
  hr.data = val;
  return hr;
}

/**
 * @brief convert __half to __amd_fp16_storage_t
 *
 * @param val
 * @return __amd_fp16_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16_storage_t __amd_cvt_half_to_fp16(const __half val) {
  __half_raw tmp = val;
  return tmp.data;
}

/**
 * @brief Convert __half2 to __amd_fp16x2_storage_t
 *
 * @param val
 * @return __amd_fp16x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_fp16x2_storage_t __amd_cvt_half2_to_fp16x2(const __half2 val) {
  __half2_raw tmp = val;
  return tmp.data;
}

/**
 * @brief Convert __amd_bf16_storage_t to __hip_bfloat16
 *
 * @param val
 * @return __hip_bfloat16
 */
__OCP_FP_HOST_DEVICE_STATIC__ __hip_bfloat16
__amd_cvt_bf16_to_hipbf16(const __amd_bf16_storage_t val) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(__amd_bf16_storage_t));
  union {
    __amd_bf16_storage_t bf16;
    __hip_bfloat16 hip_bf16;
  } u{val};
  return u.hip_bf16;
}

/**
 * @brief Convert __amd_bf16x2_storage_t to __hip_bfloat162
 *
 * @param val
 * @return __hip_bfloat162
 */
__OCP_FP_HOST_DEVICE_STATIC__ __hip_bfloat162
__amd_cvt_bf16x2_to_hipbf162(const __amd_bf16x2_storage_t val) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(__amd_bf16x2_storage_t));
  union {
    __amd_bf16x2_storage_t bf16;
    __hip_bfloat162 hip_bf16;
  } u{val};
  return u.hip_bf16;
}

/**
 * @brief Convert __hip_bfloat16 to __amd_bf16_storage_t
 *
 * @param val
 * @return __amd_bf16_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_bf16_storage_t
__amd_cvt_hipbf16_to_bf16(const __hip_bfloat16 val) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(__amd_bf16_storage_t));
  union {
    __hip_bfloat16 hip_bf16;
    __amd_bf16_storage_t bf16;
  } u{val};
  return u.bf16;
}

/**
 * @brief Convert __hip_bfloat162 to __amd_bf16x2_storage_t
 *
 * @param val
 * @return __amd_bf16x2_storage_t
 */
__OCP_FP_HOST_DEVICE_STATIC__ __amd_bf16x2_storage_t
__amd_cvt_hipbf162_to_bf16x2(const __hip_bfloat162 val) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(__amd_bf16x2_storage_t));
  union {
    __hip_bfloat162 hip_bf16;
    __amd_bf16x2_storage_t bf16;
  } u{val};
  return u.bf16;
}
