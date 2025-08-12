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

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_FUNCTIONS_H

#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_common.h>
#include <hip/amd_detail/device_library_decls.h>
#include <hip/amd_detail/hip_assert.h>
#include "host_defines.h"
#include "math_fwd.h"
#include <hip/hip_runtime_api.h>
#include <stddef.h>
#include <hip/hip_vector_types.h>
#endif // !defined(__HIPCC_RTC__)

#if defined(__clang__) && defined(__HIP__)
extern "C" __device__ int printf(const char *fmt, ...);
#else
template <typename... All>
static inline __device__ void printf(const char* format, All... all) {}
#endif

extern "C" __device__ unsigned long long __ockl_steadyctr_u64();

/*
Integer Intrinsics
*/

// integer intrinsic function __poc __clz __ffs __brev
__device__ static inline unsigned int __popc(unsigned int input) {
    return __builtin_popcount(input);
}
__device__ static inline unsigned int __popcll(unsigned long long int input) {
    return __builtin_popcountll(input);
}

__device__ static inline int __clz(int input) {
    return __ockl_clz_u32((uint)input);
}

__device__ static inline int __clzll(long long int input) {
    return __ockl_clz_u64((uint64_t)input);
}

__device__ static inline unsigned int __ffs(unsigned int input) {
    return ( input == 0 ? -1 : __builtin_ctz(input) ) + 1;
}

__device__ static inline unsigned int __ffsll(unsigned long long int input) {
    return ( input == 0 ? -1 : __builtin_ctzll(input) ) + 1;
}

__device__ static inline unsigned int __ffs(int input) {
    return ( input == 0 ? -1 : __builtin_ctz(input) ) + 1;
}

__device__ static inline unsigned int __ffsll(long long int input) {
    return ( input == 0 ? -1 : __builtin_ctzll(input) ) + 1;
}

// Given a 32/64-bit value exec mask and an integer value base (between 0 and WAVEFRONT_SIZE),
// find the n-th (given by offset) set bit in the exec mask from the base bit, and return the bit position.
// If not found, return -1.
__device__  static int32_t __fns64(uint64_t mask, uint32_t base, int32_t offset) {
  uint64_t temp_mask = mask;
  int32_t temp_offset = offset;

  if (offset == 0) {
    temp_mask &= (1 << base);
    temp_offset = 1;
  }
  else if (offset < 0) {
    temp_mask = __builtin_bitreverse64(mask);
    base = 63 - base;
    temp_offset = -offset;
  }

  temp_mask = temp_mask & ((~0ULL) << base);
  if (__builtin_popcountll(temp_mask) < temp_offset)
    return -1;
  int32_t total = 0;
  for (int i = 0x20; i > 0; i >>= 1) {
    uint64_t temp_mask_lo = temp_mask & ((1ULL << i) - 1);
    int32_t pcnt = __builtin_popcountll(temp_mask_lo);
    if (pcnt < temp_offset) {
      temp_mask = temp_mask >> i;
      temp_offset -= pcnt;
      total += i;
    }
    else {
      temp_mask = temp_mask_lo;
    }
  }
  if (offset < 0)
    return 63 - total;
  else
    return total;
}

__device__ static int32_t __fns32(uint64_t mask, uint32_t base, int32_t offset) {
  uint64_t temp_mask = mask;
  int32_t temp_offset = offset;
  if (offset == 0) {
    temp_mask &= (1 << base);
    temp_offset = 1;
  }
  else if (offset < 0) {
    temp_mask = __builtin_bitreverse64(mask);
    base = 63 - base;
    temp_offset = -offset;
  }
  temp_mask = temp_mask & ((~0ULL) << base);
  if (__builtin_popcountll(temp_mask) < temp_offset)
    return -1;
  int32_t total = 0;
  for (int i = 0x20; i > 0; i >>= 1) {
    uint64_t temp_mask_lo = temp_mask & ((1ULL << i) - 1);
    int32_t pcnt = __builtin_popcountll(temp_mask_lo);
    if (pcnt < temp_offset) {
      temp_mask = temp_mask >> i;
      temp_offset -= pcnt;
      total += i;
    }
    else {
      temp_mask = temp_mask_lo;
    }
  }
  if (offset < 0)
    return 63 - total;
  else
    return total;
}
__device__ static inline unsigned int __brev(unsigned int input) {
    return __builtin_bitreverse32(input);
}

__device__ static inline unsigned long long int __brevll(unsigned long long int input) {
    return __builtin_bitreverse64(input);
}

__device__ static inline unsigned int __lastbit_u32_u64(uint64_t input) {
    return input == 0 ? -1 : __builtin_ctzl(input);
}

__device__ static inline unsigned int __bitextract_u32(unsigned int src0, unsigned int src1, unsigned int src2) {
    uint32_t offset = src1 & 31;
    uint32_t width = src2 & 31;
    return width == 0 ? 0 : (src0 << (32 - offset - width)) >> (32 - width);
}

__device__ static inline uint64_t __bitextract_u64(uint64_t src0, unsigned int src1, unsigned int src2) {
    uint64_t offset = src1 & 63;
    uint64_t width = src2 & 63;
    return width == 0 ? 0 : (src0 << (64 - offset - width)) >> (64 - width);
}

__device__ static inline unsigned int __bitinsert_u32(unsigned int src0, unsigned int src1, unsigned int src2, unsigned int src3) {
    uint32_t offset = src2 & 31;
    uint32_t width = src3 & 31;
    uint32_t mask = (1 << width) - 1;
    return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

__device__ static inline uint64_t __bitinsert_u64(uint64_t src0, uint64_t src1, unsigned int src2, unsigned int src3) {
    uint64_t offset = src2 & 63;
    uint64_t width = src3 & 63;
    uint64_t mask = (1ULL << width) - 1;
    return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

__device__ inline unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)
{
    uint32_t mask_shift = shift & 31;
    return mask_shift == 0 ? hi : __builtin_amdgcn_alignbit(hi, lo, 32 - mask_shift);
}

__device__ inline unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    uint32_t min_shift = shift >= 32 ? 32 : shift;
    return min_shift == 0 ? hi : __builtin_amdgcn_alignbit(hi, lo, 32 - min_shift);
}

__device__ inline unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)
{
    return __builtin_amdgcn_alignbit(hi, lo, shift);
}

__device__ inline unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    return shift >= 32 ? hi : __builtin_amdgcn_alignbit(hi, lo, shift);
}

__device__ static unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);
__device__ static unsigned int __hadd(int x, int y);
__device__ static int __mul24(int x, int y);
__device__ static long long int __mul64hi(long long int x, long long int y);
__device__ static int __mulhi(int x, int y);
__device__ static int __rhadd(int x, int y);
__device__ static unsigned int __sad(int x, int y,unsigned int z);
__device__ static unsigned int __uhadd(unsigned int x, unsigned int y);
__device__ static int __umul24(unsigned int x, unsigned int y);
__device__ static unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);
__device__ static unsigned int __umulhi(unsigned int x, unsigned int y);
__device__ static unsigned int __urhadd(unsigned int x, unsigned int y);
__device__ static unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);

struct ucharHolder {
    union {
        unsigned char c[4];
        unsigned int ui;
    };
} __attribute__((aligned(4)));

struct uchar2Holder {
    union {
        unsigned int ui[2];
        unsigned char c[8];
    };
} __attribute__((aligned(8)));

__device__
static inline unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s) {
    struct uchar2Holder cHoldVal;
    struct ucharHolder cHoldKey;
    cHoldKey.ui = s;
    cHoldVal.ui[0] = x;
    cHoldVal.ui[1] = y;
    unsigned int result;
    result = cHoldVal.c[cHoldKey.c[0] & 0x07];
    result += (cHoldVal.c[(cHoldKey.c[0] & 0x70) >> 4] << 8);
    result += (cHoldVal.c[cHoldKey.c[1] & 0x07] << 16);
    result += (cHoldVal.c[(cHoldKey.c[1] & 0x70) >> 4] << 24);
    return result;
}

__device__ static inline unsigned int __hadd(int x, int y) {
    int z = x + y;
    int sign = z & 0x8000000;
    int value = z & 0x7FFFFFFF;
    return ((value) >> 1 || sign);
}

__device__ static inline int __mul24(int x, int y) {
    return __ockl_mul24_i32(x, y);
}

__device__ static inline long long __mul64hi(long long int x, long long int y) {
    unsigned long long x0 = (unsigned long long)x & 0xffffffffUL;
    long long x1 = x >> 32;
    unsigned long long y0 = (unsigned long long)y & 0xffffffffUL;
    long long y1 = y >> 32;
    unsigned long long z0 = x0*y0;
    long long t = x1*y0 + (z0 >> 32);
    long long z1 = t & 0xffffffffL;
    long long z2 = t >> 32;
    z1 = x0*y1 + z1;
    return x1*y1 + z2 + (z1 >> 32);
}

__device__ static inline int __mulhi(int x, int y) {
    return __ockl_mul_hi_i32(x, y);
}

__device__ static inline int __rhadd(int x, int y) {
    int z = x + y + 1;
    int sign = z & 0x8000000;
    int value = z & 0x7FFFFFFF;
    return ((value) >> 1 || sign);
}
__device__ static inline unsigned int __sad(int x, int y, unsigned int z) {
    return x > y ? x - y + z : y - x + z;
}
__device__ static inline unsigned int __uhadd(unsigned int x, unsigned int y) {
    return (x + y) >> 1;
}
__device__ static inline int __umul24(unsigned int x, unsigned int y) {
    return __ockl_mul24_u32(x, y);
}

__device__
static inline unsigned long long __umul64hi(unsigned long long int x, unsigned long long int y) {
    unsigned long long x0 = x & 0xffffffffUL;
    unsigned long long x1 = x >> 32;
    unsigned long long y0 = y & 0xffffffffUL;
    unsigned long long y1 = y >> 32;
    unsigned long long z0 = x0*y0;
    unsigned long long t = x1*y0 + (z0 >> 32);
    unsigned long long z1 = t & 0xffffffffUL;
    unsigned long long z2 = t >> 32;
    z1 = x0*y1 + z1;
    return x1*y1 + z2 + (z1 >> 32);
}

__device__ static inline unsigned int __umulhi(unsigned int x, unsigned int y) {
    return __ockl_mul_hi_u32(x, y);
}
__device__ static inline unsigned int __urhadd(unsigned int x, unsigned int y) {
    return (x + y + 1) >> 1;
}
__device__ static inline unsigned int __usad(unsigned int x, unsigned int y, unsigned int z) {
    return __ockl_sadd_u32(x, y, z);
}

__device__
static inline unsigned int __mbcnt_lo(unsigned int x, unsigned int y) {return __builtin_amdgcn_mbcnt_lo(x,y);};

__device__
static inline unsigned int __mbcnt_hi(unsigned int x, unsigned int y) {return __builtin_amdgcn_mbcnt_hi(x,y);};

/*
HIP specific device functions
*/

#if !defined(__HIPCC_RTC__)
#include "amd_warp_functions.h"
#include "amd_warp_sync_functions.h"
#endif

#define MASK1 0x00ff00ff
#define MASK2 0xff00ff00

__device__ static inline char4 __hip_hc_add8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.w & MASK1;
    unsigned one2 = in2.w & MASK1;
    out.w = (one1 + one2) & MASK1;
    one1 = in1.w & MASK2;
    one2 = in2.w & MASK2;
    out.w = out.w | ((one1 + one2) & MASK2);
    return out;
}

__device__ static inline char4 __hip_hc_sub8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.w & MASK1;
    unsigned one2 = in2.w & MASK1;
    out.w = (one1 - one2) & MASK1;
    one1 = in1.w & MASK2;
    one2 = in2.w & MASK2;
    out.w = out.w | ((one1 - one2) & MASK2);
    return out;
}

__device__ static inline char4 __hip_hc_mul8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.w & MASK1;
    unsigned one2 = in2.w & MASK1;
    out.w = (one1 * one2) & MASK1;
    one1 = in1.w & MASK2;
    one2 = in2.w & MASK2;
    out.w = out.w | ((one1 * one2) & MASK2);
    return out;
}

__device__ static inline float __double2float_rd(double x) {
    return __ocml_cvtrtn_f32_f64(x);
}
__device__ static inline float __double2float_rn(double x) { return x; }
__device__ static inline float __double2float_ru(double x) {
    return __ocml_cvtrtp_f32_f64(x);
}
__device__ static inline float __double2float_rz(double x) {
    return __ocml_cvtrtz_f32_f64(x);
}

__device__ static inline int __double2hiint(double x) {
    static_assert(sizeof(double) == 2 * sizeof(int), "");

    int tmp[2];
    __builtin_memcpy(tmp, &x, sizeof(tmp));

    return tmp[1];
}
__device__ static inline int __double2loint(double x) {
    static_assert(sizeof(double) == 2 * sizeof(int), "");

    int tmp[2];
    __builtin_memcpy(tmp, &x, sizeof(tmp));

    return tmp[0];
}

__device__ static inline int __double2int_rd(double x) { return (int)__ocml_floor_f64(x); }
__device__ static inline int __double2int_rn(double x) { return (int)__ocml_rint_f64(x); }
__device__ static inline int __double2int_ru(double x) { return (int)__ocml_ceil_f64(x); }
__device__ static inline int __double2int_rz(double x) { return (int)x; }

__device__ static inline long long int __double2ll_rd(double x) {
  return (long long)__ocml_floor_f64(x);
}
__device__ static inline long long int __double2ll_rn(double x) {
  return (long long)__ocml_rint_f64(x);
}
__device__ static inline long long int __double2ll_ru(double x) {
  return (long long)__ocml_ceil_f64(x);
}
__device__ static inline long long int __double2ll_rz(double x) { return (long long)x; }

__device__ static inline unsigned int __double2uint_rd(double x) {
  return (unsigned int)__ocml_floor_f64(x);
}
__device__ static inline unsigned int __double2uint_rn(double x) {
  return (unsigned int)__ocml_rint_f64(x);
}
__device__ static inline unsigned int __double2uint_ru(double x) {
  return (unsigned int)__ocml_ceil_f64(x);
}
__device__ static inline unsigned int __double2uint_rz(double x) { return (unsigned int)x; }

__device__ static inline unsigned long long int __double2ull_rd(double x) {
  return (unsigned long long int)__ocml_floor_f64(x);
}
__device__ static inline unsigned long long int __double2ull_rn(double x) {
  return (unsigned long long int)__ocml_rint_f64(x);
}
__device__ static inline unsigned long long int __double2ull_ru(double x) {
  return (unsigned long long int)__ocml_ceil_f64(x);
}
__device__ static inline unsigned long long int __double2ull_rz(double x) {
  return (unsigned long long int)x;
}
__device__ static inline long long int __double_as_longlong(double x) {
    static_assert(sizeof(long long) == sizeof(double), "");

    long long tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

/*
__device__ unsigned short __float2half_rn(float x);
__device__ float __half2float(unsigned short);

The above device function are not a valid .
Use
__device__ __half __float2half_rn(float x);
__device__ float __half2float(__half);
from hip_fp16.h

CUDA implements half as unsigned short whereas, HIP doesn't.

*/

__device__ static inline int __float2int_rd(float x) { return (int)__ocml_floor_f32(x); }
__device__ static inline int __float2int_rn(float x) { return (int)__ocml_rint_f32(x); }
__device__ static inline int __float2int_ru(float x) { return (int)__ocml_ceil_f32(x); }
__device__ static inline int __float2int_rz(float x) { return (int)__ocml_trunc_f32(x); }

__device__ static inline long long int __float2ll_rd(float x) {
  return (long long int)__ocml_floor_f32(x);
}
__device__ static inline long long int __float2ll_rn(float x) {
  return (long long int)__ocml_rint_f32(x);
}
__device__ static inline long long int __float2ll_ru(float x) {
  return (long long int)__ocml_ceil_f32(x);
}
__device__ static inline long long int __float2ll_rz(float x) { return (long long int)x; }

__device__ static inline unsigned int __float2uint_rd(float x) {
  return (unsigned int)__ocml_floor_f32(x);
}
__device__ static inline unsigned int __float2uint_rn(float x) {
  return (unsigned int)__ocml_rint_f32(x);
}
__device__ static inline unsigned int __float2uint_ru(float x) {
  return (unsigned int)__ocml_ceil_f32(x);
}
__device__ static inline unsigned int __float2uint_rz(float x) { return (unsigned int)x; }

__device__ static inline unsigned long long int __float2ull_rd(float x) {
  return (unsigned long long int)__ocml_floor_f32(x);
}
__device__ static inline unsigned long long int __float2ull_rn(float x) {
  return (unsigned long long int)__ocml_rint_f32(x);
}
__device__ static inline unsigned long long int __float2ull_ru(float x) {
  return (unsigned long long int)__ocml_ceil_f32(x);
}
__device__ static inline unsigned long long int __float2ull_rz(float x) {
  return (unsigned long long int)x;
}

__device__ static inline int __float_as_int(float x) {
    static_assert(sizeof(int) == sizeof(float), "");

    int tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline unsigned int __float_as_uint(float x) {
    static_assert(sizeof(unsigned int) == sizeof(float), "");

    unsigned int tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __hiloint2double(int hi, int lo) {
    static_assert(sizeof(double) == sizeof(uint64_t), "");

    uint64_t tmp0 = (static_cast<uint64_t>(hi) << 32ull) | static_cast<uint32_t>(lo);
    double tmp1;
    __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));

    return tmp1;
}

__device__ static inline double __int2double_rn(int x) { return (double)x; }

__device__ static inline float __int2float_rd(int x) {
    return __ocml_cvtrtn_f32_s32(x);
}
__device__ static inline float __int2float_rn(int x) { return (float)x; }
__device__ static inline float __int2float_ru(int x) {
    return __ocml_cvtrtp_f32_s32(x);
}
__device__ static inline float __int2float_rz(int x) {
    return __ocml_cvtrtz_f32_s32(x);
}

__device__ static inline float __int_as_float(int x) {
    static_assert(sizeof(float) == sizeof(int), "");

    float tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __ll2double_rd(long long int x) {
    return __ocml_cvtrtn_f64_s64(x);
}
__device__ static inline double __ll2double_rn(long long int x) { return (double)x; }
__device__ static inline double __ll2double_ru(long long int x) {
    return __ocml_cvtrtp_f64_s64(x);
}
__device__ static inline double __ll2double_rz(long long int x) {
    return __ocml_cvtrtz_f64_s64(x);
}

__device__ static inline float __ll2float_rd(long long int x) {
    return __ocml_cvtrtn_f32_s64(x);
}
__device__ static inline float __ll2float_rn(long long int x) { return (float)x; }
__device__ static inline float __ll2float_ru(long long int x) {
    return __ocml_cvtrtp_f32_s64(x);
}
__device__ static inline float __ll2float_rz(long long int x) {
    return __ocml_cvtrtz_f32_s64(x);
}

__device__ static inline double __longlong_as_double(long long int x) {
    static_assert(sizeof(double) == sizeof(long long), "");

    double tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __uint2double_rn(unsigned int x) { return (double)x; }

__device__ static inline float __uint2float_rd(unsigned int x) {
    return __ocml_cvtrtn_f32_u32(x);
}
__device__ static inline float __uint2float_rn(unsigned int x) { return (float)x; }
__device__ static inline float __uint2float_ru(unsigned int x) {
    return __ocml_cvtrtp_f32_u32(x);
}
__device__ static inline float __uint2float_rz(unsigned int x) {
    return __ocml_cvtrtz_f32_u32(x);
}

__device__ static inline float __uint_as_float(unsigned int x) {
   static_assert(sizeof(float) == sizeof(unsigned int), "");

    float tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __ull2double_rd(unsigned long long int x) {
    return __ocml_cvtrtn_f64_u64(x);
}
__device__ static inline double __ull2double_rn(unsigned long long int x) { return (double)x; }
__device__ static inline double __ull2double_ru(unsigned long long int x) {
    return __ocml_cvtrtp_f64_u64(x);
}
__device__ static inline double __ull2double_rz(unsigned long long int x) {
    return __ocml_cvtrtz_f64_u64(x);
}

__device__ static inline float __ull2float_rd(unsigned long long int x) {
    return __ocml_cvtrtn_f32_u64(x);
}
__device__ static inline float __ull2float_rn(unsigned long long int x) { return (float)x; }
__device__ static inline float __ull2float_ru(unsigned long long int x) {
    return __ocml_cvtrtp_f32_u64(x);
}
__device__ static inline float __ull2float_rz(unsigned long long int x) {
    return __ocml_cvtrtz_f32_u64(x);
}

#if defined(__clang__) && defined(__HIP__)

// Clock functions
__device__ long long int __clock64();
__device__ long long int __clock();
__device__ long long int clock64();
__device__ long long int clock();
__device__ long long int wall_clock64();
// hip.amdgcn.bc - named sync
__device__ void __named_sync();

#ifdef __HIP_DEVICE_COMPILE__

// Clock function to return GPU core cycle count.
// GPU can change its core clock frequency at runtime. The maximum frequency can be queried
// through hipDeviceAttributeClockRate attribute.
__device__
inline  __attribute((always_inline))
long long int __clock64() {
#if __has_builtin(__builtin_amdgcn_s_memtime)
  // Exists on gfx8, gfx9, gfx10.1, gfx10.2, gfx10.3
  return (long long int) __builtin_amdgcn_s_memtime();
#else
  // Subject to change when better solution available
  return (long long int) __builtin_readcyclecounter();
#endif
}

__device__
inline __attribute((always_inline))
long long int  __clock() { return __clock64(); }

// Clock function to return wall clock count at a constant frequency that can be queried
// through hipDeviceAttributeWallClockRate attribute.
__device__
inline  __attribute__((always_inline))
long long int wall_clock64() {
  return (long long int) __ockl_steadyctr_u64();
}

__device__
inline  __attribute__((always_inline))
long long int clock64() { return __clock64(); }

__device__
inline __attribute__((always_inline))
long long int  clock() { return __clock(); }

// hip.amdgcn.bc - named sync
__device__
inline
void __named_sync() { __builtin_amdgcn_s_barrier(); }

#endif // __HIP_DEVICE_COMPILE__

// hip.amdgcn.bc - lanemask
__device__
inline
uint64_t  __lanemask_gt()
{
    uint32_t lane = __ockl_lane_u32();
    if (lane == 63)
      return 0;
    uint64_t ballot = __ballot64(1);
    uint64_t mask = (~((uint64_t)0)) << (lane + 1);
    return mask & ballot;
}

__device__
inline
uint64_t __lanemask_lt()
{
    uint32_t lane = __ockl_lane_u32();
    int64_t ballot = __ballot64(1);
    uint64_t mask = ((uint64_t)1 << lane) - (uint64_t)1;
    return mask & ballot;
}

__device__
inline
uint64_t  __lanemask_eq()
{
    uint32_t lane = __ockl_lane_u32();
    int64_t mask = ((uint64_t)1 << lane);
    return mask;
}


__device__ inline void* __local_to_generic(void* p) { return p; }

#ifdef __HIP_DEVICE_COMPILE__
__device__
inline
void* __get_dynamicgroupbaseptr()
{
    // Get group segment base pointer.
    return (char*)__local_to_generic((void*)__to_local(__builtin_amdgcn_groupstaticsize()));
}
#else
__device__
void* __get_dynamicgroupbaseptr();
#endif // __HIP_DEVICE_COMPILE__

__device__
inline
void *__amdgcn_get_dynamicgroupbaseptr() {
    return __get_dynamicgroupbaseptr();
}

// Memory Fence Functions
__device__
inline
static void __threadfence()
{
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
}

__device__
inline
static void __threadfence_block()
{
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
}

__device__
inline
static void __threadfence_system()
{
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
}
__device__ inline static void __work_group_barrier(__cl_mem_fence_flags flags) {
    if (flags) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    } else {
        __builtin_amdgcn_s_barrier();
    }
}

__device__
inline
static void __barrier(int n)
{
  __work_group_barrier((__cl_mem_fence_flags)n);
}

__device__
inline
__attribute__((convergent))
void __syncthreads()
{
  __barrier(__CLK_LOCAL_MEM_FENCE);
}

__device__
inline
__attribute__((convergent))
int __syncthreads_count(int predicate)
{
  return __ockl_wgred_add_i32(!!predicate);
}

__device__
inline
__attribute__((convergent))
int __syncthreads_and(int predicate)
{
  return __ockl_wgred_and_i32(!!predicate);
}

__device__
inline
__attribute__((convergent))
int __syncthreads_or(int predicate)
{
  return __ockl_wgred_or_i32(!!predicate);
}

// hip.amdgcn.bc - device routine
/*
  HW_ID Register bit structure for RDNA2 & RDNA3
  WAVE_ID     4:0     Wave id within the SIMD.
  SIMD_ID     9:8     SIMD_ID within the WGP: [0] = row, [1] = column.
  WGP_ID      13:10   Physical WGP ID.
  SA_ID       16      Shader Array ID
  SE_ID       20:18   Shader Engine the wave is assigned to for gfx11
  SE_ID       19:18   Shader Engine the wave is assigned to for gfx10
  DP_RATE     31:29   Number of double-precision float units per SIMD

  HW_ID Register bit structure for GCN and CDNA
  WAVE_ID     3:0     Wave buffer slot number. 0-9.
  SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
  PIPE_ID     7:6     Pipeline from which the wave was dispatched.
  CU_ID       11:8    Compute Unit the wave is assigned to.
  SH_ID       12      Shader Array (within an SE) the wave is assigned to.
  SE_ID       15:13   Shader Engine the wave is assigned to for gfx908, gfx90a, gfx940-942
              14:13   Shader Engine the wave is assigned to for Vega.
  TG_ID       19:16   Thread-group ID
  VM_ID       23:20   Virtual Memory ID
  QUEUE_ID    26:24   Queue from which this wave was dispatched.
  STATE_ID    29:27   State ID (graphics only, not compute).
  ME_ID       31:30   Micro-engine ID.

  XCC_ID Register bit structure for gfx940
  XCC_ID      3:0     XCC the wave is assigned to.
 */

#if (defined (__GFX10__) || defined (__GFX11__))
  #define HW_ID               23
#else
  #define HW_ID               4
#endif

#if (defined(__GFX10__) || defined(__GFX11__))
  #define HW_ID_WGP_ID_SIZE   4
  #define HW_ID_WGP_ID_OFFSET 10
  #if (defined(__AMDGCN_CUMODE__))
    #define HW_ID_CU_ID_SIZE    1
    #define HW_ID_CU_ID_OFFSET  8
  #endif
#else
  #define HW_ID_CU_ID_SIZE    4
  #define HW_ID_CU_ID_OFFSET  8
#endif

#if (defined(__gfx908__) || defined(__gfx90a__) || \
     defined(__GFX11__))
  #define HW_ID_SE_ID_SIZE    3
#else //4 SEs/XCC for gfx940-942
  #define HW_ID_SE_ID_SIZE    2
#endif
#if (defined(__GFX10__) || defined(__GFX11__))
  #define HW_ID_SE_ID_OFFSET  18
  #define HW_ID_SA_ID_OFFSET  16
  #define HW_ID_SA_ID_SIZE    1
#else
  #define HW_ID_SE_ID_OFFSET  13
#endif

#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
  #define XCC_ID                   20
  #define XCC_ID_XCC_ID_SIZE       4
  #define XCC_ID_XCC_ID_OFFSET     0
#endif

#if (!defined(__HIP_NO_IMAGE_SUPPORT) && \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)))
  #define __HIP_NO_IMAGE_SUPPORT   1
#endif

/*
   Encoding of parameter bitmask
   HW_ID        5:0     HW_ID
   OFFSET       10:6    Range: 0..31
   SIZE         15:11   Range: 1..32
 */

#define GETREG_IMMED(SZ,OFF,REG) (((SZ) << 11) | ((OFF) << 6) | (REG))

/*
  __smid returns the wave's assigned Compute Unit and Shader Engine.
  The Compute Unit, CU_ID returned in bits 3:0, and Shader Engine, SE_ID in bits 5:4.
  Note: the results vary over time.
  SZ minus 1 since SIZE is 1-based.
*/
__device__
inline
unsigned __smid(void)
{
    unsigned se_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_SE_ID_SIZE-1, HW_ID_SE_ID_OFFSET, HW_ID));
    #if (defined(__GFX10__) || defined(__GFX11__))
      unsigned wgp_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_WGP_ID_SIZE - 1, HW_ID_WGP_ID_OFFSET, HW_ID));
      unsigned sa_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_SA_ID_SIZE - 1, HW_ID_SA_ID_OFFSET, HW_ID));
      #if (defined(__AMDGCN_CUMODE__))
        unsigned cu_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_CU_ID_SIZE - 1, HW_ID_CU_ID_OFFSET, HW_ID));
      #endif
    #else
      #if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
      unsigned xcc_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(XCC_ID_XCC_ID_SIZE - 1, XCC_ID_XCC_ID_OFFSET, XCC_ID));
      #endif
      unsigned cu_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_CU_ID_SIZE - 1, HW_ID_CU_ID_OFFSET, HW_ID));
    #endif
    #if (defined(__GFX10__) || defined(__GFX11__))
      unsigned temp = se_id;
      temp = (temp << HW_ID_SA_ID_SIZE) | sa_id;
      temp = (temp << HW_ID_WGP_ID_SIZE) | wgp_id;
      #if (defined(__AMDGCN_CUMODE__))
        temp = (temp << HW_ID_CU_ID_SIZE) | cu_id;
      #endif
      return temp;
      //TODO : CU Mode impl
    #elif (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
      unsigned temp = xcc_id;
      temp = (temp << HW_ID_SE_ID_SIZE) | se_id;
      temp = (temp << HW_ID_CU_ID_SIZE) | cu_id;
      return temp;
    #else
      return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
    #endif
}

/**
 * Map HIP_DYNAMIC_SHARED to "extern __shared__" for compatibility with old HIP applications
 * To be removed in a future release.
 */
#define HIP_DYNAMIC_SHARED(type, var) extern __shared__ type var[];
#define HIP_DYNAMIC_SHARED_ATTRIBUTE

#endif //defined(__clang__) && defined(__HIP__)


// loop unrolling
static inline __device__ void* __hip_hc_memcpy(void* dst, const void* src, size_t size) {
    auto dstPtr = static_cast<unsigned char*>(dst);
    auto srcPtr = static_cast<const unsigned char*>(src);

    while (size >= 4u) {
        dstPtr[0] = srcPtr[0];
        dstPtr[1] = srcPtr[1];
        dstPtr[2] = srcPtr[2];
        dstPtr[3] = srcPtr[3];

        size -= 4u;
        srcPtr += 4u;
        dstPtr += 4u;
    }
    switch (size) {
        case 3:
            dstPtr[2] = srcPtr[2];
        case 2:
            dstPtr[1] = srcPtr[1];
        case 1:
            dstPtr[0] = srcPtr[0];
    }

    return dst;
}

static inline __device__ void* __hip_hc_memset(void* dst, unsigned char val, size_t size) {
    auto dstPtr = static_cast<unsigned char*>(dst);

    while (size >= 4u) {
        dstPtr[0] = val;
        dstPtr[1] = val;
        dstPtr[2] = val;
        dstPtr[3] = val;

        size -= 4u;
        dstPtr += 4u;
    }
    switch (size) {
        case 3:
            dstPtr[2] = val;
        case 2:
            dstPtr[1] = val;
        case 1:
            dstPtr[0] = val;
    }

    return dst;
}
#ifndef __OPENMP_AMDGCN__
static inline __device__ void* memcpy(void* dst, const void* src, size_t size) {
    return __hip_hc_memcpy(dst, src, size);
}

static inline __device__ void* memset(void* ptr, int val, size_t size) {
    unsigned char val8 = static_cast<unsigned char>(val);
    return __hip_hc_memset(ptr, val8, size);
}
#endif // !__OPENMP_AMDGCN__

#endif
