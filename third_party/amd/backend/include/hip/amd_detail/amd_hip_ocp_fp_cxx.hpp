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

#include "amd_hip_ocp_host.hpp"

static_assert(sizeof(unsigned int) == sizeof(__amd_fp8_storage_t[4]));
static_assert(sizeof(uint32_t) == sizeof(__amd_fp8_storage_t[4]));
static_assert(sizeof(int) == sizeof(__amd_fp8x2_storage_t[2]));
static_assert(sizeof(uint32_t) == sizeof(__amd_fp8x2_storage_t[2]));
static_assert(sizeof(__amd_shortx2_storage_t) == sizeof(__amd_fp8x2_storage_t[2]));

struct __hipext_ocp_fp8_e4m3 {
  __amd_fp8_storage_t __x;
  static const __amd_fp8_interpretation_t __default_interpret = __AMD_OCP_E4M3;

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e4m3() = default;

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e4m3(const float in) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = __builtin_amdgcn_cvt_pk_fp8_f32(in, in, 0, false);
    __x = u.fp8[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = from_float<float, Encoding::E4M3, true>(in, 0 /*scale*/);
    __x = u.fp8[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e4m3(const float in, const unsigned int seed) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = __builtin_amdgcn_cvt_sr_fp8_f32(in, seed, 0, 0);
    __x = u.fp8[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = from_float_sr<float, Encoding::E4M3, true>(in, seed, 0 /*scale*/);
    __x = u.fp8[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e4m3(const float in, const unsigned int seed,
                                                        const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 =
        __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(u.ui32, in, seed, __amd_scale_to_float(scale), 0);
    __x = u.fp8[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = from_float_sr<float, Encoding::E4M3, true>(in, seed, scale);
    __x = u.fp8[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e4m3(const __amd_fp16_storage_t in,
                                                        const unsigned int seed,
                                                        const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union u {
      unsigned int ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 =
        __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(u.ui32, in, seed, __amd_scale_to_float(scale), 0);
    __x = u.fp8[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = from_float_sr<__amd_fp16_storage_t, Encoding::E4M3, true>(in, seed, scale);
    __x = u.fp8[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e4m3(const __amd_bf16_storage_t in,
                                                        const unsigned int seed,
                                                        const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    static_assert(sizeof(__amd_fp8_storage_t[4]) == sizeof(unsigned int));
    union u {
      unsigned int ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 =
        __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(u.ui32, in, seed, __amd_scale_to_float(scale), 0);
    __x = u.fp8[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = from_float_sr<__amd_bf16_storage_t, Encoding::E4M3, true>(in, seed, scale);
    __x = u.fp8[0];
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_fp16_storage_t get_scaled_fp16(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp16x2_storage_t ret;
    ret = __builtin_amdgcn_cvt_scalef32_f16_fp8(ret, __x, __amd_scale_to_float(scale), 0, false);
    return ret[0];
#else
    using namespace fcbx;
    return to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(__x, scale);
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_bf16_storage_t get_scaled_bf16(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    static_assert(sizeof(unsigned int) == sizeof(__amd_fp8_storage_t[4]));
    union {
      __amd_fp8_storage_t fp8[4];
      unsigned int ui32;
    } u;
    u.fp8[0] = __x;
    auto ret =
        __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(u.ui32, __amd_scale_to_float(scale), false);
    return ret[0];
#else
    using namespace fcbx;
    return to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(__x, scale);
#endif
  }

  __OCP_FP_HOST_DEVICE__ float get_scaled_float(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_f32_fp8(__x, __amd_scale_to_float(scale), 0);
#else
    using namespace fcbx;
    return to_float<float, Encoding::E4M3, true>(__x, scale);
#endif
  }

  __OCP_FP_HOST_DEVICE__ operator float() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_f32_fp8(__x, 0);
#else
    using namespace fcbx;
    return to_float<float, Encoding::E4M3, true>(__x, 0);
#endif
  }
};

struct __hipext_ocp_fp8_e5m2 {
  __amd_fp8_storage_t __x;
  static const __amd_fp8_interpretation_t __default_interpret = __AMD_OCP_E5M2;

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e5m2() = default;

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e5m2(const float in) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = __builtin_amdgcn_cvt_pk_bf8_f32(in, in, 0, false);
    __x = u.fp8[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = from_float<float, Encoding::E5M2, true>(in, 0 /*scale*/);
    __x = u.fp8[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e5m2(const float in, const unsigned int seed) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = __builtin_amdgcn_cvt_sr_bf8_f32(in, seed, 0, 0);
    __x = u.fp8[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = from_float_sr<float, Encoding::E5M2, true>(in, seed, 0 /*scale*/);
    __x = u.fp8[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e5m2(const float in, const unsigned int seed,
                                                        const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 =
        __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(u.ui32, in, seed, __amd_scale_to_float(scale), 0);
    __x = u.fp8[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = from_float_sr<float, Encoding::E5M2, true>(in, seed, scale);
    __x = u.fp8[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e5m2(const __amd_fp16_storage_t in,
                                                        const unsigned int seed,
                                                        const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union u {
      unsigned int ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 =
        __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(u.ui32, in, seed, __amd_scale_to_float(scale), 0);
    __x = u.fp8[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = from_float_sr<__amd_fp16_storage_t, Encoding::E5M2, true>(in, seed, scale);
    __x = u.fp8[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8_e5m2(const __amd_bf16_storage_t in,
                                                        const unsigned int seed,
                                                        const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union u {
      unsigned int ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 =
        __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(u.ui32, in, seed, __amd_scale_to_float(scale), 0);
    __x = u.fp8[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8_storage_t fp8[4];
    } u{0};
    u.ui32 = from_float_sr<__amd_bf16_storage_t, Encoding::E5M2, true>(in, seed, scale);
    __x = u.fp8[0];
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_fp16_storage_t get_scaled_fp16(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp16x2_storage_t ret;
    ret = __builtin_amdgcn_cvt_scalef32_f16_bf8(ret, __x, __amd_scale_to_float(scale), 0, false);
    return ret[0];
#else
    using namespace fcbx;
    return to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(__x, scale);
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_bf16_storage_t get_scaled_bf16(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    static_assert(sizeof(unsigned int) == sizeof(__amd_fp8_storage_t[4]));
    union {
      __amd_fp8_storage_t fp8[4];
      unsigned int ui32;
    } u;
    u.fp8[0] = __x;
    auto ret =
        __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(u.ui32, __amd_scale_to_float(scale), false);
    return ret[0];
#else
    using namespace fcbx;
    return to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(__x, scale);
#endif
  }

  __OCP_FP_HOST_DEVICE__ float get_scaled_float(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_f32_bf8(__x, __amd_scale_to_float(scale), 0);
#else
    using namespace fcbx;
    return to_float<float, Encoding::E5M2, true>(__x, scale);
#endif
  }

  __OCP_FP_HOST_DEVICE__ operator float() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_f32_bf8(__x, 0);
#else
    using namespace fcbx;
    return to_float<float, Encoding::E5M2, true>(__x, 0);
#endif
  }
};

struct __hipext_ocp_fp8x2_e4m3 {
  __amd_fp8x2_storage_t __x;
  static const __amd_fp8_interpretation_t __default_interpret = __AMD_OCP_E4M3;

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e4m3() = default;

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e4m3(const float a, const float b) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      int i32;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.i32 = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, u.i32, false);
    __x = u.fp8x2[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.ui32 = from_float<float, Encoding::E4M3, true>(b, 0 /*scale*/);
    u.ui32 <<= 8;
    u.ui32 |= from_float<float, Encoding::E4M3, true>(a, 0 /*scale*/);
    __x = u.fp8x2[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e4m3(const __amd_floatx2_storage_t in)
      : __hipext_ocp_fp8x2_e4m3(in[0], in[1]) {}

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e4m3(const float a, const float b,
                                                          __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      __amd_shortx2_storage_t shortx2;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(u.shortx2, a, b,
                                                         __amd_scale_to_float(scale), false);
    __x = u.fp8x2[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.ui32 = from_float<float, Encoding::E4M3, true>(b, scale);
    u.ui32 <<= 8;
    u.ui32 |= from_float<float, Encoding::E4M3, true>(a, scale);
    __x = u.fp8x2[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e4m3(const __amd_floatx2_storage_t in,
                                                          const __amd_scale_t scale)
      : __hipext_ocp_fp8x2_e4m3(in[0], in[1], scale) {}


  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e4m3(const __amd_fp16x2_storage_t in,
                                                 const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      __amd_shortx2_storage_t shortx2;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.shortx2 =
        __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(u.shortx2, in, __amd_scale_to_float(scale), false);
    __x = u.fp8x2[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.ui32 = from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(in[1], scale);
    u.ui32 <<= 8;
    u.ui32 |= from_float<__amd_fp16_storage_t, Encoding::E4M3, true>(in[0], scale);
    __x = u.fp8x2[0];
#endif
  }

  explicit __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e4m3(const __amd_bf16x2_storage_t in,
                                                          const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      __amd_shortx2_storage_t shortx2;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(u.shortx2, in,
                                                          __amd_scale_to_float(scale), false);
    __x = u.fp8x2[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.ui32 = from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(in[1], scale);
    u.ui32 <<= 8;
    u.ui32 |= from_float<__amd_bf16_storage_t, Encoding::E4M3, true>(in[0], scale);
    __x = u.fp8x2[0];
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_fp16x2_storage_t get_scaled_fp16x2(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      __amd_fp8x2_storage_t fp8x2[2];
      unsigned int ui32;
    } u;
    u.fp8x2[0] = __x;
    return __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(u.ui32, __amd_scale_to_float(scale), false);
#else
    using namespace fcbx;
    __amd_fp16x2_storage_t ret;
    ret[0] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(__x & 0xFF, scale);
    ret[1] = to_float<__amd_fp16_storage_t, Encoding::E4M3, true>(__x >> 8, scale);
    return ret;
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_bf16x2_storage_t get_scaled_bf16x2(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      __amd_fp8x2_storage_t fp8x2[2];
      unsigned int ui32;
    } u;
    u.fp8x2[0] = __x;
    return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(u.ui32, __amd_scale_to_float(scale), false);
#else
    using namespace fcbx;
    __amd_bf16x2_storage_t ret;
    ret[0] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(__x & 0xFF, scale);
    ret[1] = to_float<__amd_bf16_storage_t, Encoding::E4M3, true>(__x >> 8, scale);
    return ret;
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_floatx2_storage_t
  get_scaled_floatx2(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(__x, __amd_scale_to_float(scale), false);
#else
    using namespace fcbx;
    __amd_floatx2_storage_t ret;
    ret[0] = to_float<float, Encoding::E4M3, true>(__x & 0xFF, scale);
    ret[1] = to_float<float, Encoding::E4M3, true>(__x >> 8, scale);
    return ret;
#endif
  }

  __OCP_FP_HOST_DEVICE__
  operator __amd_floatx2_storage_t() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_pk_f32_fp8(__x, false);
#else
    using namespace fcbx;
    __amd_floatx2_storage_t ret;
    ret[0] = to_float<float, Encoding::E4M3, true>(__x & 0xFF, 0);
    ret[1] = to_float<float, Encoding::E4M3, true>(__x >> 8, 0);
    return ret;
#endif
  }
};

struct __hipext_ocp_fp8x2_e5m2 {
  __amd_fp8x2_storage_t __x;
  static const __amd_fp8_interpretation_t __default_interpret = __AMD_OCP_E5M2;

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e5m2() = default;

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e5m2(const float a, const float b) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      int i32;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.i32 = __builtin_amdgcn_cvt_pk_bf8_f32(a, b, u.i32, false);
    __x = u.fp8x2[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.ui32 = from_float<float, Encoding::E5M2, true>(b, 0 /*scale*/);
    u.ui32 <<= 8;
    u.ui32 |= from_float<float, Encoding::E5M2, true>(a, 0 /*scale*/);
    __x = u.fp8x2[0];
#endif
  }

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e5m2(const __amd_floatx2_storage_t in)
      : __hipext_ocp_fp8x2_e5m2(in[0], in[1]) {}

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e5m2(const float a, const float b,
                                                 const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      __amd_shortx2_storage_t shortx2;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(u.shortx2, a, b,
                                                         __amd_scale_to_float(scale), false);
    __x = u.fp8x2[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.ui32 = from_float<float, Encoding::E5M2, true>(b, scale);
    u.ui32 <<= 8;
    u.ui32 |= from_float<float, Encoding::E5M2, true>(a, scale);
    __x = u.fp8x2[0];
#endif
  }

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e5m2(const __amd_floatx2_storage_t in,
                                                 const __amd_scale_t scale)
      : __hipext_ocp_fp8x2_e5m2(in[0], in[1], scale) {}

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e5m2(const __amd_fp16x2_storage_t in,
                                                 const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      __amd_shortx2_storage_t shortx2;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.shortx2 =
        __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(u.shortx2, in, __amd_scale_to_float(scale), false);
    __x = u.fp8x2[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.ui32 = from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(in[1], scale);
    u.ui32 <<= 8;
    u.ui32 |= from_float<__amd_fp16_storage_t, Encoding::E5M2, true>(in[0], scale);
    __x = u.fp8x2[0];
#endif
  }

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp8x2_e5m2(const __amd_bf16x2_storage_t in,
                                                 const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      __amd_shortx2_storage_t shortx2;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.shortx2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16(u.shortx2, in,
                                                          __amd_scale_to_float(scale), false);
    __x = u.fp8x2[0];
#else
    using namespace fcbx;
    union {
      uint32_t ui32;
      __amd_fp8x2_storage_t fp8x2[2];
    } u{0};
    u.ui32 = from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(in[1], scale);
    u.ui32 <<= 8;
    u.ui32 |= from_float<__amd_bf16_storage_t, Encoding::E5M2, true>(in[0], scale);
    __x = u.fp8x2[0];
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_fp16x2_storage_t get_scaled_fp16x2(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      __amd_fp8x2_storage_t fp8x2[2];
      unsigned int ui32;
    } u;
    u.fp8x2[0] = __x;
    return __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(u.ui32, __amd_scale_to_float(scale), false);
#else
    using namespace fcbx;
    __amd_fp16x2_storage_t ret;
    ret[0] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(__x & 0xFF, scale);
    ret[1] = to_float<__amd_fp16_storage_t, Encoding::E5M2, true>(__x >> 8, scale);
    return ret;
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_bf16x2_storage_t get_scaled_bf16x2(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      __amd_fp8x2_storage_t fp8x2[2];
      unsigned int ui32;
    } u;
    u.fp8x2[0] = __x;
    return __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(u.ui32, __amd_scale_to_float(scale), false);
#else
    using namespace fcbx;
    __amd_bf16x2_storage_t ret;
    ret[0] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(__x & 0xFF, scale);
    ret[1] = to_float<__amd_bf16_storage_t, Encoding::E5M2, true>(__x >> 8, scale);
    return ret;
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_floatx2_storage_t
  get_scaled_floatx2(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(__x, __amd_scale_to_float(scale), false);
#else
    using namespace fcbx;
    __amd_floatx2_storage_t ret;
    ret[0] = to_float<float, Encoding::E5M2, true>(__x & 0xFF, scale);
    ret[1] = to_float<float, Encoding::E5M2, true>(__x >> 8, scale);
    return ret;
#endif
  }

  __OCP_FP_HOST_DEVICE__
  operator __amd_floatx2_storage_t() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_pk_f32_bf8(__x, false);
#else
    using namespace fcbx;
    __amd_floatx2_storage_t ret;
    ret[0] = to_float<float, Encoding::E5M2, true>(__x & 0xFF, 0);
    ret[1] = to_float<float, Encoding::E5M2, true>(__x >> 8, 0);
    return ret;
#endif
  }
};

struct __hipext_ocp_fp6x32_e2m3 {
  __amd_fp6x32_storage_t __x;
  static const __amd_fp6_interpretation_t __default_interpret = __AMD_OCP_E2M3;

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp6x32_e2m3(const __amd_floatx16_storage_t in1,
                                                  const __amd_floatx16_storage_t in2,
                                                  const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, __amd_scale_to_float(scale))) {
  }
#else
  {
    using namespace fcbx;
    __amd_floatx32_storage_t tmp_in;
    for (size_t i = 0; i < 16; i++) {
      tmp_in[i] = in1[i];
      tmp_in[16 + i] = in2[i];
    }
    __x = fp6_cvt_packedx32<__amd_floatx32_storage_t, __amd_fp6x32_storage_t, float,
                            Encoding::IEEE754, Encoding::E2M3>(tmp_in, scale);
  }
#endif

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp6x32_e2m3(const __amd_floatx32_storage_t in,
                                                  const unsigned int round,
                                                  const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32(in, round, __amd_scale_to_float(scale))){}
#else
      : __x(fcbx::fp6_cvt_packedx32<__amd_floatx32_storage_t, __amd_fp6x32_storage_t, float,
                                    fcbx::Encoding::IEEE754, fcbx::Encoding::E2M3>(in, scale)) {
  }
#endif

        __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp6x32_e2m3(
            const __amd_fp16x32_storage_t in, const unsigned int round, const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f16(in, round, __amd_scale_to_float(scale))){}
#else
      : __x(fcbx::fp6_cvt_packedx32<__amd_fp16x32_storage_t, __amd_fp6x32_storage_t,
                                    __amd_fp16_storage_t, fcbx::Encoding::E5M10,
                                    fcbx::Encoding::E2M3, true>(in, scale, round)) {
  }
#endif

        __OCP_FP_HOST_DEVICE__
        __hipext_ocp_fp6x32_e2m3(const __amd_fp16x32_storage_t in, const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_pk32_fp6_f16(in, __amd_scale_to_float(scale))){}
#else
      : __x(fcbx::fp6_cvt_packedx32<__amd_fp16x32_storage_t, __amd_fp6x32_storage_t,
                                    __amd_fp16_storage_t, fcbx::Encoding::E5M10,
                                    fcbx::Encoding::E2M3>(in, scale)) {
  }
#endif

        __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp6x32_e2m3(
            const __amd_bf16x32_storage_t in, const unsigned int round, const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_bf16(in, round,
                                                           __amd_scale_to_float(scale))){}
#else
      : __x(fcbx::fp6_cvt_packedx32<__amd_bf16x32_storage_t, __amd_fp6x32_storage_t,
                                    __amd_bf16_storage_t, fcbx::Encoding::E8M7,
                                    fcbx::Encoding::E2M3, true>(in, scale, round)) {
  }
#endif

        __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp6x32_e2m3(const __amd_bf16x32_storage_t in,
                                                        const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16(in, __amd_scale_to_float(scale))){}
#else
      : __x(fcbx::fp6_cvt_packedx32<__amd_bf16x32_storage_t, __amd_fp6x32_storage_t,
                                    __amd_bf16_storage_t, fcbx::Encoding::E8M7,
                                    fcbx::Encoding::E2M3>(in, scale)) {
  }
#endif

        __OCP_FP_HOST_DEVICE__ __amd_floatx32_storage_t
        get_scaled_floatx32(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(__x, __amd_scale_to_float(scale));
#else
    using namespace fcbx;
    return fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_floatx32_storage_t, float,
                             Encoding::E2M3, Encoding::IEEE754>(__x, scale);
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_fp16x32_storage_t
  get_scaled_fp16x32(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6(__x, __amd_scale_to_float(scale));
#else
    using namespace fcbx;
    return fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_fp16x32_storage_t, __amd_fp16_storage_t,
                             Encoding::E2M3, Encoding::E5M10>(__x, scale);
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_bf16x32_storage_t
  get_scaled_bf16x32(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6(__x, __amd_scale_to_float(scale));
#else
    using namespace fcbx;
    return fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_bf16x32_storage_t, __amd_bf16_storage_t,
                             Encoding::E2M3, Encoding::E8M7>(__x, scale);
#endif
  }
};

struct __hipext_ocp_fp6x32_e3m2 {
  __amd_fp6x32_storage_t __x;
  static const __amd_fp6_interpretation_t __default_interpret = __AMD_OCP_E3M2;

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp6x32_e3m2(const __amd_floatx16_storage_t in1,
                                                  const __amd_floatx16_storage_t in2,
                                                  const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, __amd_scale_to_float(scale))) {
  }
#else
  {
    using namespace fcbx;
    __amd_floatx32_storage_t tmp_in;
    for (size_t i = 0; i < 16; i++) {
      tmp_in[i] = in1[i];
      tmp_in[16 + i] = in2[i];
    }
    __x = fp6_cvt_packedx32<__amd_floatx32_storage_t, __amd_fp6x32_storage_t, float,
                            Encoding::IEEE754, Encoding::E3M2>(tmp_in, scale);
  }
#endif

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp6x32_e3m2(const __amd_floatx32_storage_t in,
                                                  const unsigned int round,
                                                  const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f32(in, round, __amd_scale_to_float(scale))){}
#else
      : __x(fcbx::fp6_cvt_packedx32<__amd_floatx32_storage_t, __amd_fp6x32_storage_t, float,
                                    fcbx::Encoding::IEEE754, fcbx::Encoding::E3M2>(in, scale)) {
  }
#endif

        __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp6x32_e3m2(
            const __amd_fp16x32_storage_t in, const unsigned int round, const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f16(in, round, __amd_scale_to_float(scale))){}
#else
      : __x(fcbx::fp6_cvt_packedx32<__amd_fp16x32_storage_t, __amd_fp6x32_storage_t,
                                    __amd_fp16_storage_t, fcbx::Encoding::E5M10,
                                    fcbx::Encoding::E3M2, true>(in, scale, round)) {
  }
#endif

        __OCP_FP_HOST_DEVICE__
        __hipext_ocp_fp6x32_e3m2(const __amd_fp16x32_storage_t in, const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_pk32_bf6_f16(in, __amd_scale_to_float(scale))){}
#else
      : __x(fcbx::fp6_cvt_packedx32<__amd_fp16x32_storage_t, __amd_fp6x32_storage_t,
                                    __amd_fp16_storage_t, fcbx::Encoding::E5M10,
                                    fcbx::Encoding::E3M2>(in, scale)) {
  }
#endif

        __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp6x32_e3m2(
            const __amd_bf16x32_storage_t in, const unsigned int round, const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_bf16(in, round,
                                                           __amd_scale_to_float(scale))){}
#else
      : __x(fcbx::fp6_cvt_packedx32<__amd_bf16x32_storage_t, __amd_fp6x32_storage_t,
                                    __amd_bf16_storage_t, fcbx::Encoding::E8M7,
                                    fcbx::Encoding::E3M2, true>(in, scale, round)) {
  }
#endif

        __OCP_FP_HOST_DEVICE__
        __hipext_ocp_fp6x32_e3m2(const __amd_bf16x32_storage_t in, const __amd_scale_t scale)
#if HIP_ENABLE_GFX950_OCP_BUILTINS
      : __x(__builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16(in, __amd_scale_to_float(scale))){}
#else
      : __x(fcbx::fp6_cvt_packedx32<__amd_bf16x32_storage_t, __amd_fp6x32_storage_t,
                                    __amd_bf16_storage_t, fcbx::Encoding::E8M7,
                                    fcbx::Encoding::E3M2>(in, scale)) {
  }
#endif

        __OCP_FP_HOST_DEVICE__ __amd_floatx32_storage_t
        get_scaled_floatx32(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(__x, __amd_scale_to_float(scale));
#else
    using namespace fcbx;
    return fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_floatx32_storage_t, float,
                             Encoding::E3M2, Encoding::IEEE754>(__x, scale);
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_fp16x32_storage_t
  get_scaled_fp16x32(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk32_f16_bf6(__x, __amd_scale_to_float(scale));
#else
    using namespace fcbx;
    return fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_fp16x32_storage_t, __amd_fp16_storage_t,
                             Encoding::E3M2, Encoding::E5M10>(__x, scale);
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_bf16x32_storage_t
  get_scaled_bf16x32(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6(__x, __amd_scale_to_float(scale));
#else
    using namespace fcbx;
    return fp6_cvt_packedx32<__amd_fp6x32_storage_t, __amd_bf16x32_storage_t, __amd_bf16_storage_t,
                             Encoding::E3M2, Encoding::E8M7>(__x, scale);
#endif
  }
};

struct __hipext_ocp_fp4x2_e2m1 {
  __amd_fp4x2_storage_t __x;
  static const __amd_fp4_interpretation_t __default_interpret = __AMD_OCP_E2M1;

  static_assert(sizeof(unsigned int) == sizeof(__amd_fp4x2_storage_t[4]));

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp4x2_e2m1(const float a, const float b,
                                                 const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, a, b, __amd_scale_to_float(scale), 0);
    __x = u.fp4x2[0];
#else
    using namespace fcbx;
    auto l = from_float<float, Encoding::E2M1, true>(a, scale);
    auto r = from_float<float, Encoding::E2M1, true>(b, scale);
    __x = r << 4 | l;
#endif
  }

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp4x2_e2m1(const __amd_floatx2_storage_t in,
                                                 const __amd_scale_t scale)
      : __hipext_ocp_fp4x2_e2m1(in[0], in[1], scale) {}

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp4x2_e2m1(const __amd_bf16x2_storage_t in,
                                                 const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u.ui32, in, __amd_scale_to_float(scale), 1);
    __x = u.fp4x2[1];
#else
    using namespace fcbx;
    auto l = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(in[0], scale);
    auto r = from_float<__amd_bf16_storage_t, Encoding::E2M1, true>(in[1], scale);
    __x = r << 4 | l;
#endif
  }

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp4x2_e2m1(const __amd_fp16x2_storage_t in,
                                                 const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(u.ui32, in, __amd_scale_to_float(scale), 1);
    __x = u.fp4x2[1];
#else
    using namespace fcbx;
    auto l = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(in[0], scale);
    auto r = from_float<__amd_fp16_storage_t, Encoding::E2M1, true>(in[1], scale);
    __x = r << 4 | l;
#endif
  }

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp4x2_e2m1(const __amd_floatx2_storage_t in,
                                                 const unsigned int seed,
                                                 const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(u.ui32, in, seed,
                                                         __amd_scale_to_float(scale), 1);
    __x = u.fp4x2[1];
#else
    static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(uint32_t));
    union u {
      uint32_t ui32t;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    using namespace fcbx;
    auto l = from_float_sr<float, Encoding::E2M1, true>(in[0], seed, scale);
    auto r = from_float_sr<float, Encoding::E2M1, true>(in[1], seed, scale);
    r <<= 4;
    l |= r;
    u.ui32t = l;
    __x = u.fp4x2[0];
#endif
  }

  __OCP_FP_HOST_DEVICE__
  __hipext_ocp_fp4x2_e2m1(const __amd_bf16x2_storage_t in, const unsigned int seed,
                          const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16(u.ui32, in, seed,
                                                          __amd_scale_to_float(scale), 1);
    __x = u.fp4x2[1];
#else
    static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(uint32_t));
    union u {
      uint32_t ui32t;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    using namespace fcbx;
    auto l = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(in[0], seed, scale);
    auto r = from_float_sr<__amd_bf16_storage_t, Encoding::E2M1, true>(in[1], seed, scale);
    r <<= 4;
    l |= r;
    u.ui32t = l;
    __x = u.fp4x2[0];
#endif
  }

  __OCP_FP_HOST_DEVICE__ __hipext_ocp_fp4x2_e2m1(const __amd_fp16x2_storage_t in,
                                                 const unsigned int seed,
                                                 const __amd_scale_t scale) {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.ui32 = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16(u.ui32, in, seed,
                                                         __amd_scale_to_float(scale), 1);
    __x = u.fp4x2[1];
#else
    static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(uint32_t));
    union u {
      uint32_t ui32t;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    using namespace fcbx;
    auto l = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(in[0], seed, scale);
    auto r = from_float_sr<__amd_fp16_storage_t, Encoding::E2M1, true>(in[1], seed, scale);
    r <<= 4;
    l |= r;
    u.ui32t = l;
    __x = u.fp4x2[0];
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_floatx2_storage_t
  get_scaled_floatx2(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x, __amd_scale_to_float(scale), 0);
#else
    using namespace fcbx;
    __amd_floatx2_storage_t ret{to_float<float, Encoding::E2M1, true>(__x & 0xFu, scale),
                                to_float<float, Encoding::E2M1, true>(__x >> 4, scale)};
    return ret;
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_fp16x2_storage_t get_scaled_fp16x2(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(__x, __amd_scale_to_float(scale), 0);
#else
    using namespace fcbx;
    __amd_fp16x2_storage_t ret{
        to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(__x & 0xFu, scale),
        to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(__x >> 4, scale)};
    return ret;
#endif
  }

  __OCP_FP_HOST_DEVICE__ __amd_bf16x2_storage_t get_scaled_bf16x2(const __amd_scale_t scale) const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(__x, __amd_scale_to_float(scale), 0);
#else
    using namespace fcbx;
    __amd_bf16x2_storage_t ret{
        to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(__x & 0xFu, scale),
        to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(__x >> 4, scale)};
    return ret;
#endif
  }
};
