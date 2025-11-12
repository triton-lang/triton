/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include "amd_hip_mx_common.h"

#include "amd_hip_fp16.h"
#include "amd_hip_bf16.h"
#include "amd_hip_fp8.h"

#include "amd_hip_ocp_types.h"
#include "amd_hip_ocp_host.hpp"
#include "hip/amd_detail/amd_hip_mx_common.h"

#if defined(__HIPCC_RTC__)
#define __FP6_HOST_DEVICE__ __device__
#define __FP6_HOST_DEVICE_STATIC__ __FP6_HOST_DEVICE__ static
#else
#define __FP6_HOST_DEVICE__ __host__ __device__
#define __FP6_HOST_DEVICE_STATIC__ __FP6_HOST_DEVICE__ static inline
#endif  // __HIPCC_RTC__

typedef __hip_fp8_storage_t __hip_fp6_storage_t;
typedef __hip_fp8x2_storage_t __hip_fp6x2_storage_t;
typedef __hip_fp8x4_storage_t __hip_fp6x4_storage_t;

static_assert(sizeof(__hip_fp6_storage_t[4]) == sizeof(uint32_t));
static_assert(sizeof(__hip_fp6x2_storage_t[2]) == sizeof(uint32_t));
static_assert(sizeof(__hip_fp6x4_storage_t[2]) == sizeof(uint64_t));

enum __hip_fp6_interpretation_t {
  __HIP_E3M2 = 0, /**< FP6 E3M2 Type*/
  __HIP_E2M3 = 1, /**< FP6 E2M3 Type */
};

namespace internal {
__FP6_HOST_DEVICE_STATIC__ __amd_fp16_storage_t half_to_f16(const __half val) {
  __half_raw tmp = val;
  return tmp.data;
}
__FP6_HOST_DEVICE_STATIC__ __amd_fp16x2_storage_t half2_to_f16x2(const __half2 val) {
  __half2_raw tmp = val;
  return tmp.data;
}
__FP6_HOST_DEVICE_STATIC__ __amd_bf16_storage_t hipbf16_to_bf16(const __hip_bfloat16 val) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(__amd_bf16_storage_t));
  union {
    __hip_bfloat16 hip_bf16;
    __amd_bf16_storage_t bf16;
  } u{val};
  return u.bf16;
}
__FP6_HOST_DEVICE_STATIC__ __amd_bf16x2_storage_t hipbf162_to_bf16x2(const __hip_bfloat162 val) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(__amd_bf16x2_storage_t));
  union {
    __hip_bfloat162 hip_bf16;
    __amd_bf16x2_storage_t bf16;
  } u{val};
  return u.bf16;
}
}  // namespace internal


// Note: Ignore rounding input on AMD GPUs for now. At the moment AMD GPUs do not support rounding
// modes, all the inputs are rounded to nearest or use an input to do stochastic rounding.
// We hide the rounding variable to not trigger the unused variable compiler warning.
__FP6_HOST_DEVICE_STATIC__ __hip_fp6_storage_t __hip_cvt_bfloat16raw_to_fp6(
    const __hip_bfloat16_raw x, const __hip_fp6_interpretation_t fp6_interpretation,
    const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp6_storage_t fp6[4];
  } u{0};
#if __gfx950__
  __amd_bf16x32_storage_t in;
  __amd_fp6x32_storage_t out;
  in[0] = internal::hipbf16_to_bf16(x);
  if (fp6_interpretation == __HIP_E2M3)
    out = __builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16(in, 1.0f /* scale */);
  else if (fp6_interpretation == __HIP_E3M2)
    out = __builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16(in, 1.0f /* scale */);
  u.ui32 = out[0];
  return u.fp6[0];
#else
  if (fp6_interpretation == __HIP_E2M3)
    u.ui32 = fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M3, true>(
        internal::hipbf16_to_bf16(x), 0);
  else if (fp6_interpretation == __HIP_E3M2)
    u.ui32 = fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E3M2, true>(
        internal::hipbf16_to_bf16(x), 0);
  return u.fp6[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6x2_storage_t __hip_cvt_bfloat16raw2_to_fp6x2(
    const __hip_bfloat162_raw x, const __hip_fp6_interpretation_t fp6_interpretation,
    const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp6x2_storage_t fp6x2[2];
  } u{0};
#if __gfx950__
  __amd_bf16x32_storage_t in;
  in[0] = internal::hipbf16_to_bf16(x.x);
  in[1] = internal::hipbf16_to_bf16(x.y);
  __amd_fp6x32_storage_t out;
  if (fp6_interpretation == __HIP_E2M3)
    out = __builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16(in, 1.0f /* scale */);
  else if (fp6_interpretation == __HIP_E3M2)
    out = __builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16(in, 1.0f /* scale */);
  u.ui32 = out[0];
  return u.fp6x2[0];
#else
  if (fp6_interpretation == __HIP_E2M3) {
    auto bf16x2 = internal::hipbf162_to_bf16x2(x);
    u.ui32 |= fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M3, true>(bf16x2[1], 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M3, true>(bf16x2[0], 0);
  } else if (fp6_interpretation == __HIP_E3M2) {
    auto bf16x2 = internal::hipbf162_to_bf16x2(x);
    u.ui32 |= fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E3M2, true>(bf16x2[1], 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E3M2, true>(bf16x2[0], 0);
  }
  return u.fp6x2[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6_storage_t
__hip_cvt_double_to_fp6(const double x, const __hip_fp6_interpretation_t fp6_interpretation_t,
                        const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp6_storage_t fp6[4];
  } u{0};
#if __gfx950__
  __amd_floatx16_storage_t in1;
  __amd_floatx16_storage_t in2;
  __amd_fp6x32_storage_t out;
  in1[0] = float(x);
  in2[0] = 0.0f;
  if (fp6_interpretation_t == __HIP_E2M3)
    out = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, 1.0f /* scale */);
  else if (fp6_interpretation_t == __HIP_E3M2)
    out = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, 1.0f /* scale */);
  u.ui32 = out[0];
  return u.fp6[0];
#else
  if (fp6_interpretation_t == __HIP_E2M3) {
    u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M3, true>(float(x), 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    u.ui32 = fcbx::from_float<float, fcbx::Encoding::E3M2, true>(float(x), 0);
  }
  return u.fp6[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6x2_storage_t
__hip_cvt_double2_to_fp6x2(const double2 x, const __hip_fp6_interpretation_t fp6_interpretation_t,
                           const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp6x2_storage_t fp6x2[2];
  } u{0};
#if __gfx950__
  __amd_floatx16_storage_t in1;
  __amd_floatx16_storage_t in2;
  __amd_fp6x32_storage_t out;
  in1[0] = float(x.x);
  in2[0] = float(x.y);
  if (fp6_interpretation_t == __HIP_E2M3)
    out = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, 1.0f /* scale */);
  else if (fp6_interpretation_t == __HIP_E3M2)
    out = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, 1.0f /* scale */);
  u.ui32 = out[0] & 0x3Fu;
  u.ui32 |= ((out[0] & 0xFC0u) << 2);
  return u.fp6x2[0];
#else
  if (fp6_interpretation_t == __HIP_E2M3) {
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M3, true>(float(x.y), 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M3, true>(float(x.x), 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E3M2, true>(float(x.y), 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E3M2, true>(float(x.x), 0);
  }
  return u.fp6x2[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6_storage_t
__hip_cvt_float_to_fp6(const float x, const __hip_fp6_interpretation_t fp6_interpretation_t,
                       const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp6_storage_t fp6[4];
  } u{0};
#if __gfx950__
  __amd_floatx16_storage_t in1;
  __amd_floatx16_storage_t in2;
  __amd_fp6x32_storage_t out;
  in1[0] = x;
  in2[0] = 0.0f;
  if (fp6_interpretation_t == __HIP_E2M3)
    out = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, 1.0f /* scale */);
  else if (fp6_interpretation_t == __HIP_E3M2)
    out = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, 1.0f /* scale */);
  u.ui32 = out[0];
  return u.fp6[0];
#else
  if (fp6_interpretation_t == __HIP_E2M3)
    u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M3, true>(x, 0);
  else if (fp6_interpretation_t == __HIP_E3M2)
    u.ui32 = fcbx::from_float<float, fcbx::Encoding::E3M2, true>(x, 0);
  return u.fp6[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6x2_storage_t
__hip_cvt_float2_to_fp6x2(const float2 x, const __hip_fp6_interpretation_t fp6_interpretation_t,
                          const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp6x2_storage_t fp6x2[2];
  } u{0};
#if __gfx950__
  __amd_floatx16_storage_t in1;
  __amd_floatx16_storage_t in2;
  __amd_fp6x32_storage_t out;
  in1[0] = x.x;
  in2[0] = x.y;
  if (fp6_interpretation_t == __HIP_E2M3)
    out = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, 1.0f /* scale */);
  else if (fp6_interpretation_t == __HIP_E3M2)
    out = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, 1.0f /* scale */);
  u.ui32 = out[0] & 0x3Fu;
  u.ui32 |= ((out[0] & 0xFC0u) << 2);
  return u.fp6x2[0];
#else
  if (fp6_interpretation_t == __HIP_E2M3) {
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M3, true>(x.y, 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M3, true>(x.x, 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E3M2, true>(x.y, 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E3M2, true>(x.x, 0);
  }
  return u.fp6x2[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __half_raw __hip_cvt_fp6_to_halfraw(
    const __hip_fp6_storage_t x, const __hip_fp6_interpretation_t fp6_interpretation_t) {
  __half_raw ret;
#if __gfx950__
  __amd_fp16x32_storage_t out;
  __amd_fp6x32_storage_t in;
  in[0] = (uint32_t)x;
  if (fp6_interpretation_t == __HIP_E2M3)
    out = __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6(in, 1.0f);
  else if (fp6_interpretation_t == __HIP_E3M2)
    out = __builtin_amdgcn_cvt_scalef32_pk32_f16_bf6(in, 1.0f);
  ret.data = out[0];
#else
  using namespace fcbx;
  if (fp6_interpretation_t == __HIP_E2M3) {
    ret.data = __amd_fp16_storage_t{to_float<__amd_fp16_storage_t, Encoding::E2M3, true>(x, 0)};
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    ret.data = __amd_fp16_storage_t{to_float<__amd_fp16_storage_t, Encoding::E3M2, true>(x, 0)};
  }
#endif
  return ret;
}
__FP6_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp6x2_to_halfraw2(
    const __hip_fp6x2_storage_t x, const __hip_fp6_interpretation_t fp6_interpretation_t) {
  __half2_raw ret;
#if __gfx950__
  __amd_fp16x32_storage_t out;
  __amd_fp6x32_storage_t in;
  in[0] = x & 0x3Fu;            // first 6 bits
  in[0] |= (x & 0x3F00u) >> 2;  // next 6 bits
  if (fp6_interpretation_t == __HIP_E2M3)
    out = __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6(in, 1.0f);
  else if (fp6_interpretation_t == __HIP_E3M2)
    out = __builtin_amdgcn_cvt_scalef32_pk32_f16_bf6(in, 1.0f);
  ret.data = {out[0], out[1]};
#else
  using namespace fcbx;
  if (fp6_interpretation_t == __HIP_E2M3) {
    ret.data =
        __amd_fp16x2_storage_t{to_float<__amd_fp16_storage_t, Encoding::E2M3, true>(x & 0xFFu, 0),
                               to_float<__amd_fp16_storage_t, Encoding::E2M3, true>(x >> 8, 0)};
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    ret.data =
        __amd_fp16x2_storage_t{to_float<__amd_fp16_storage_t, Encoding::E3M2, true>(x & 0xFFu, 0),
                               to_float<__amd_fp16_storage_t, Encoding::E3M2, true>(x >> 8, 0)};
  }
#endif
  return ret;
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6_storage_t
__hip_cvt_halfraw_to_fp6(const __half_raw x, const __hip_fp6_interpretation_t fp6_interpretation_t,
                         const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp6_storage_t fp6[4];
  } u{0};
#if __gfx950__
  __amd_fp16x32_storage_t in;
  __amd_fp6x32_storage_t out;
  in[0] = x.data;
  if (fp6_interpretation_t == __HIP_E2M3)
    out = __builtin_amdgcn_cvt_scalef32_pk32_fp6_f16(in, 1.0f);
  else if (fp6_interpretation_t == __HIP_E3M2)
    out = __builtin_amdgcn_cvt_scalef32_pk32_bf6_f16(in, 1.0f);
  u.ui32 = out[0];
  return u.fp6[0];
#else
  if (fp6_interpretation_t == __HIP_E2M3) {
    u.ui32 = fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M3, true>(
        internal::half_to_f16(x), 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    u.ui32 = fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E3M2, true>(
        internal::half_to_f16(x), 0);
  }
  return u.fp6[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6x2_storage_t __hip_cvt_halfraw2_to_fp6x2(
    const __half2_raw x, const __hip_fp6_interpretation_t fp6_interpretation_t,
    const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp6x2_storage_t fp6x2[2];
  } u{0};
#if __gfx950__
  __amd_fp16x32_storage_t in;
  __amd_fp6x32_storage_t out;
  in[0] = x.data[0];
  in[1] = x.data[1];
  if (fp6_interpretation_t == __HIP_E2M3)
    out = __builtin_amdgcn_cvt_scalef32_pk32_fp6_f16(in, 1.0f);
  else if (fp6_interpretation_t == __HIP_E3M2)
    out = __builtin_amdgcn_cvt_scalef32_pk32_bf6_f16(in, 1.0f);
  u.ui32 = out[0] & 0x3Fu;
  u.ui32 |= ((out[0] & 0xFC0u) << 2);
  return u.fp6x2[0];
#else
  auto fp16x2 = internal::half2_to_f16x2(x);
  if (fp6_interpretation_t == __HIP_E2M3) {
    u.ui32 |= fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M3, true>(fp16x2[1], 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M3, true>(fp16x2[0], 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    u.ui32 |= fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E3M2, true>(fp16x2[1], 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E3M2, true>(fp16x2[0], 0);
  }
  return u.fp6x2[0];
#endif
}

//======================== structs ====================
struct __hip_fp6_e2m3 {
  __hip_fp6_storage_t __x;

  __FP6_HOST_DEVICE__ __hip_fp6_e2m3() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const __half f)
      : __x(__hip_cvt_halfraw_to_fp6(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const __hip_bfloat16 f)
      : __x(__hip_cvt_bfloat16raw_to_fp6(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const double f)
      : __x(__hip_cvt_double_to_fp6(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const float f)
      : __x(__hip_cvt_float_to_fp6(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const long long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const short int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const unsigned int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const unsigned long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const unsigned long long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const unsigned short int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
#endif  // !defined(__HIP_NO_FP6_CONVERSIONS__)

#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator __half_raw() const {
    return __hip_cvt_fp6_to_halfraw(__x, __HIP_E2M3);
  }
  __FP6_HOST_DEVICE__ operator __hip_bfloat16_raw() const {
    static_assert(sizeof(__hip_bfloat16_raw) == sizeof(__amd_bf16_storage_t));
    union {
      __hip_bfloat16_raw bf16_raw;
      __amd_bf16_storage_t bf16;
    } u;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in;
    __amd_bf16x32_storage_t out;
    in[0] = (uint32_t)__x;
    out = __builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6(in, 1.0f /* scale */);
    u.bf16 = out[0];
#else
    using namespace fcbx;
    u.bf16 = to_float<__amd_bf16_storage_t, Encoding::E2M3, true>(__x, 0);
#endif
    return u.bf16_raw;
  }
  __FP6_HOST_DEVICE__ operator float() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in;
    __amd_floatx32_storage_t out;
    in[0] = (uint32_t)__x;
    out = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(in, 1.0f /* scale */);
    auto ret = out[0];
#else
    using namespace fcbx;
    float ret{to_float<float, Encoding::E2M3, true>(__x, 0)};
#endif
    return ret;
  }
  __FP6_HOST_DEVICE__ operator double() const { return double(float(*this)); }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

struct __hip_fp6_e3m2 {
  __hip_fp6_storage_t __x;

  __FP6_HOST_DEVICE__ __hip_fp6_e3m2() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const __half f)
      : __x(__hip_cvt_halfraw_to_fp6(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const __hip_bfloat16 f)
      : __x(__hip_cvt_bfloat16raw_to_fp6(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const double f)
      : __x(__hip_cvt_double_to_fp6(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const float f)
      : __x(__hip_cvt_float_to_fp6(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const long long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const short int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const unsigned int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const unsigned long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const unsigned long long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const unsigned short int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
#endif  // !defined(__HIP_NO_FP6_CONVERSIONS__)
#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator __half_raw() const {
    return __hip_cvt_fp6_to_halfraw(__x, __HIP_E3M2);
  }
  __FP6_HOST_DEVICE__ operator __hip_bfloat16_raw() const {
    static_assert(sizeof(__hip_bfloat16_raw) == sizeof(__amd_bf16_storage_t));
    union {
      __hip_bfloat16_raw bf16_raw;
      __amd_bf16_storage_t bf16;
    } u;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in;
    __amd_bf16x32_storage_t out;
    in[0] = (uint32_t)__x;
    out = __builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6(in, 1.0f /* scale */);
    u.bf16 = out[0];
#else
    using namespace fcbx;
    u.bf16 = to_float<__amd_bf16_storage_t, Encoding::E3M2, true>(__x, 0);
#endif
    return u.bf16_raw;
  }
  __FP6_HOST_DEVICE__ operator float() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in;
    __amd_floatx32_storage_t out;
    in[0] = (uint32_t)__x;
    out = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(in, 1.0f /* scale */);
    auto ret = out[0];
#else
    using namespace fcbx;
    float ret{to_float<float, Encoding::E3M2, true>(__x, 0)};
#endif
    return ret;
  }
  __FP6_HOST_DEVICE__ operator double() const { return double(float(*this)); }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

struct __hip_fp6x2_e2m3 {
  __hip_fp6x2_storage_t __x;
  __FP6_HOST_DEVICE__ inline __hip_fp6x2_e2m3() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e2m3(const __half2 f)
      : __x(__hip_cvt_halfraw2_to_fp6x2(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e2m3(const __hip_bfloat162 f)
      : __x(__hip_cvt_bfloat16raw2_to_fp6x2(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e2m3(const double2 f)
      : __x(__hip_cvt_double2_to_fp6x2(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e2m3(const float2 f)
      : __x(__hip_cvt_float2_to_fp6x2(f, __HIP_E2M3, hipRoundNearest)) {}
#endif  // !defined(__HIP_NO_FP6_CONVERSIONS__)
#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator __half2_raw() const {
    return __hip_cvt_fp6x2_to_halfraw2(__x, __HIP_E2M3);
  }
  __FP6_HOST_DEVICE__ operator __hip_bfloat162_raw() const {
    static_assert(sizeof(__hip_bfloat162_raw) == sizeof(__amd_bf16x2_storage_t));
    union {
      __hip_bfloat162_raw bf162_raw;
      __amd_bf16x2_storage_t bf16x2;
    } u;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in;
    __amd_bf16x32_storage_t out;
    in[0] = __x & 0x3Fu;          // first 6 bits
    in[0] |= (__x & FC00u) >> 2;  // next 6 bits
    out = __builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6(in, 1.0f /* scale */);
    u.bf16x2 = {out[0], out[1]};
#else
    using namespace fcbx;
    u.bf16x2 =
        __amd_bf16x2_storage_t{to_float<__amd_bf16_storage_t, Encoding::E2M3, true>(__x & 0xFFu, 0),
                               to_float<__amd_bf16_storage_t, Encoding::E2M3, true>(__x >> 8, 0)};
#endif
    return u.bf162_raw;
  }
  __FP6_HOST_DEVICE__ operator float2() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in;
    __amd_floatx32_storage_t out;
    in[0] = __x & 0x3Fu;          // first 6 bits
    in[0] |= (__x & FC00u) >> 2;  // next 6 bits
    out = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(in, 1.0f /* scale */);
    auto fp32x2 = {out[0], out[1]};
#else
    using namespace fcbx;
    auto fp32x2 = __amd_floatx2_storage_t{to_float<float, Encoding::E2M3, true>(__x & 0xFFu, 0),
                                          to_float<float, Encoding::E2M3, true>(__x >> 8, 0)};
#endif
    return float2(fp32x2[0], fp32x2[1]);
  }
  __FP6_HOST_DEVICE__ operator double2() const {
    auto fp32 = float2(*this);
    return double2(fp32.x, fp32.y);
  }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

struct __hip_fp6x2_e3m2 {
  __hip_fp6x2_storage_t __x;
  __FP6_HOST_DEVICE__ inline __hip_fp6x2_e3m2() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e3m2(const __half2 f)
      : __x(__hip_cvt_halfraw2_to_fp6x2(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e3m2(const __hip_bfloat162 f)
      : __x(__hip_cvt_bfloat16raw2_to_fp6x2(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e3m2(const double2 f)
      : __x(__hip_cvt_double2_to_fp6x2(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e3m2(const float2 f)
      : __x(__hip_cvt_float2_to_fp6x2(f, __HIP_E3M2, hipRoundNearest)) {}
#endif  //! defined(__HIP_NO_FP6_CONVERSIONS__)
#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator __half2_raw() const {
    return __hip_cvt_fp6x2_to_halfraw2(__x, __HIP_E3M2);
  }
  __FP6_HOST_DEVICE__ operator __hip_bfloat162_raw() const {
    static_assert(sizeof(__hip_bfloat162_raw) == sizeof(__amd_bf16x2_storage_t));
    union {
      __hip_bfloat162_raw bf162_raw;
      __amd_bf16x2_storage_t bf16x2;
    } u;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in;
    __amd_bf16x32_storage_t out;
    in[0] = __x & 0x3Fu;          // first 6 bits
    in[0] |= (__x & FC00u) >> 2;  // next 6 bits
    out = __builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6(in, 1.0f /* scale */);
    u.bf16x2 = {out[0], out[1]};
#else
    using namespace fcbx;
    u.bf16x2 =
        __amd_bf16x2_storage_t{to_float<__amd_bf16_storage_t, Encoding::E3M2, true>(__x & 0xFFu, 0),
                               to_float<__amd_bf16_storage_t, Encoding::E3M2, true>(__x >> 8, 0)};
#endif
    return u.bf162_raw;
  }
  __FP6_HOST_DEVICE__ operator float2() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in;
    __amd_floatx32_storage_t out;
    in[0] = __x & 0x3Fu;          // first 6 bits
    in[0] |= (__x & FC00u) >> 2;  // next 6 bits
    out = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(in, 1.0f /* scale */);
    auto fp32x2 = {out[0], out[1]};
#else
    using namespace fcbx;
    auto fp32x2 = __amd_floatx2_storage_t{to_float<float, Encoding::E3M2, true>(__x & 0xFFu, 0),
                                          to_float<float, Encoding::E3M2, true>(__x >> 8, 0)};
#endif
    return float2(fp32x2[0], fp32x2[1]);
  }
  __FP6_HOST_DEVICE__ operator double2() const {
    auto fp32 = float2(*this);
    return double2(fp32.x, fp32.y);
  }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

struct __hip_fp6x4_e2m3 {
  __hip_fp6x4_storage_t __x;
  __FP6_HOST_DEVICE__ inline __hip_fp6x4_e2m3() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e2m3(const __half2 low, const __half2 high)
      : __x(__hip_cvt_halfraw2_to_fp6x2(high, __HIP_E2M3, hipRoundNearest) << 16 |
            __hip_cvt_halfraw2_to_fp6x2(low, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e2m3(const __hip_bfloat162 low,
                                                       const __hip_bfloat162 high)
      : __x(__hip_cvt_bfloat16raw2_to_fp6x2(high, __HIP_E2M3, hipRoundNearest) << 16 |
            __hip_cvt_bfloat16raw2_to_fp6x2(low, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e2m3(const double4 f)
      : __x(__hip_cvt_double2_to_fp6x2(double2(f.z, f.w), __HIP_E2M3, hipRoundNearest) << 16 |
            __hip_cvt_double2_to_fp6x2(double2(f.x, f.y), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e2m3(const float4 f)
      : __x(__hip_cvt_float2_to_fp6x2(float2(f.z, f.w), __HIP_E2M3, hipRoundNearest) << 16 |
            __hip_cvt_float2_to_fp6x2(float2(f.x, f.y), __HIP_E2M3, hipRoundNearest)) {}
#endif  // !defined(__HIP_NO_FP6_CONVERSIONS__)
#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator float4() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in;
    __amd_floatx32_storage_t out;
    in[0] = __x & 0x3Fu;                 // first 6 bits
    in[0] |= ((__x >> 8) & 0x3Fu) << 6;  // second 6 bits
    in[0] |= ((__x >> 16) & 0x3Fu) << 12;
    in[0] |= ((__x >> 24) & 0x3Fu) << 18;
    out = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(in, 1.0f /* scale */);
    auto fp32x2_1 = {out[0], out[1]};
    auto fp32x2_2 = {out[2], out[3]};
#else
    using namespace fcbx;
    auto fp32x2_1 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E2M3, true>(__x & 0xFFu, 0),
                                to_float<float, Encoding::E2M3, true>((__x >> 8) & 0xFFu, 0)};
    auto fp32x2_2 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E2M3, true>((__x >> 16) & 0xFFu, 0),
                                to_float<float, Encoding::E2M3, true>(__x >> 24, 0)};
#endif
    return float4{fp32x2_1[0], fp32x2_1[1], fp32x2_2[0], fp32x2_2[1]};
  }
  __FP6_HOST_DEVICE__ operator double4() const {
    auto fp32 = float4(*this);
    return double4{fp32.x, fp32.y, fp32.z, fp32.w};
  }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

struct __hip_fp6x4_e3m2 {
  __hip_fp6x4_storage_t __x;
  __FP6_HOST_DEVICE__ inline __hip_fp6x4_e3m2() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e3m2(const __half2 high, const __half2 low)
      : __x(__hip_cvt_halfraw2_to_fp6x2(high, __HIP_E3M2, hipRoundNearest) << 16 |
            __hip_cvt_halfraw2_to_fp6x2(low, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e3m2(const __hip_bfloat162 high,
                                                       const __hip_bfloat162 low)
      : __x(__hip_cvt_bfloat16raw2_to_fp6x2(high, __HIP_E3M2, hipRoundNearest) << 16 |
            __hip_cvt_bfloat16raw2_to_fp6x2(low, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e3m2(const double4 f)
      : __x(__hip_cvt_double2_to_fp6x2(double2(f.z, f.w), __HIP_E3M2, hipRoundNearest) << 16 |
            __hip_cvt_double2_to_fp6x2(double2(f.x, f.y), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e3m2(const float4 f)
      : __x(__hip_cvt_float2_to_fp6x2(float2(f.z, f.w), __HIP_E3M2, hipRoundNearest) << 16 |
            __hip_cvt_float2_to_fp6x2(float2(f.x, f.y), __HIP_E3M2, hipRoundNearest)) {}
#endif  //! defined(__HIP_NO_FP6_CONVERSIONS__)
#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator float4() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in;
    __amd_floatx32_storage_t out;
    in[0] = __x & 0x3Fu;                 // first 6 bits
    in[0] |= ((__x >> 8) & 0x3Fu) << 6;  // second 6 bits
    in[0] |= ((__x >> 16) & 0x3Fu) << 12;
    in[0] |= ((__x >> 24) & 0x3Fu) << 18;
    out = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(in, 1.0f /* scale */);
    auto fp32x2_1 = {out[0], out[1]};
    auto fp32x2_2 = {out[2], out[3]};
#else
    using namespace fcbx;
    auto fp32x2_1 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E3M2, true>(__x & 0xFFu, 0),
                                to_float<float, Encoding::E3M2, true>((__x >> 8) & 0xFFu, 0)};
    auto fp32x2_2 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E3M2, true>((__x >> 16) & 0xFFu, 0),
                                to_float<float, Encoding::E3M2, true>(__x >> 24, 0)};
#endif
    return float4{fp32x2_1[0], fp32x2_1[1], fp32x2_2[0], fp32x2_2[1]};
  }
  __FP6_HOST_DEVICE__ operator double4() const {
    auto fp32 = float4(*this);
    return double4{fp32.x, fp32.y, fp32.z, fp32.w};
  }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};
