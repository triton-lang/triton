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

/**
 *  @file  amd_detail/device_library_decls.h
 *  @brief Contains declarations for types and functions in device library.
 *         Uses int64_t and uint64_t instead of long, long long, unsigned
 *         long and unsigned long long types for device library API
 *         declarations.
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_LIBRARY_DECLS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_LIBRARY_DECLS_H

#if !defined(__HIPCC_RTC__)
#include "hip/amd_detail/host_defines.h"
#endif

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ullong;

extern "C" __device__ __attribute__((const)) bool __ockl_wfany_i32(int);
extern "C" __device__ __attribute__((const)) bool __ockl_wfall_i32(int);
extern "C" __device__ uint __ockl_activelane_u32(void);

extern "C" __device__ __attribute__((const)) uint __ockl_mul24_u32(uint, uint);
extern "C" __device__ __attribute__((const)) int __ockl_mul24_i32(int, int);
extern "C" __device__ __attribute__((const)) uint __ockl_mul_hi_u32(uint, uint);
extern "C" __device__ __attribute__((const)) int __ockl_mul_hi_i32(int, int);
extern "C" __device__ __attribute__((const)) uint __ockl_sadd_u32(uint, uint, uint);

extern "C" __device__ __attribute__((const)) uchar __ockl_clz_u8(uchar);
extern "C" __device__ __attribute__((const)) ushort __ockl_clz_u16(ushort);
extern "C" __device__ __attribute__((const)) uint __ockl_clz_u32(uint);
extern "C" __device__ __attribute__((const)) uint64_t __ockl_clz_u64(uint64_t);

extern "C" __device__ __attribute__((const)) float __ocml_floor_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_rint_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_ceil_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_trunc_f32(float);

extern "C" __device__ __attribute__((const)) float __ocml_fmin_f32(float, float);
extern "C" __device__ __attribute__((const)) float __ocml_fmax_f32(float, float);

extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_f64(double);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_f64(double);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_f64(double);

extern "C" __device__ __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
extern "C" __device__ __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
extern "C" __device__ __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);

extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_s32(int);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_s32(int);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_s32(int);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_u32(uint32_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_u32(uint32_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_u32(uint32_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_s64(int64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_s64(int64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_s64(int64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_u64(uint64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_u64(uint64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_u64(uint64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtn_f64_s64(int64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtp_f64_s64(int64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtz_f64_s64(int64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtn_f64_u64(uint64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtp_f64_u64(uint64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtz_f64_u64(uint64_t);

extern "C" __device__ __attribute__((convergent)) void __ockl_gws_init(uint nwm1, uint rid);
extern "C" __device__ __attribute__((convergent)) void __ockl_gws_barrier(uint nwm1, uint rid);

extern "C" __device__ __attribute__((const)) uint32_t __ockl_lane_u32();
extern "C" __device__ __attribute__((const)) int __ockl_grid_is_valid(void);
extern "C" __device__ __attribute__((convergent)) void __ockl_grid_sync(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_num_grids(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_grid_rank(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_size(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_thread_rank(void);
extern "C" __device__ __attribute__((const)) int __ockl_multi_grid_is_valid(void);
extern "C" __device__ __attribute__((convergent)) void __ockl_multi_grid_sync(void);

extern "C" __device__ void __ockl_atomic_add_noret_f32(float*, float);

extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_add_i32(int a);
extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_and_i32(int a);
extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_or_i32(int a);

extern "C" __device__ uint64_t __ockl_fprintf_stderr_begin();
extern "C" __device__ uint64_t __ockl_fprintf_append_args(uint64_t msg_desc, uint32_t num_args,
                                                          uint64_t value0, uint64_t value1,
                                                          uint64_t value2, uint64_t value3,
                                                          uint64_t value4, uint64_t value5,
                                                          uint64_t value6, uint32_t is_last);
extern "C" __device__ uint64_t __ockl_fprintf_append_string_n(uint64_t msg_desc, const char* data,
                                                              uint64_t length, uint32_t is_last);

// Introduce local address space
#define __local __attribute__((address_space(3)))

#ifdef __HIP_DEVICE_COMPILE__
__device__ inline static __local void* __to_local(unsigned x) { return (__local void*)x; }
#endif //__HIP_DEVICE_COMPILE__

// Using hip.amdgcn.bc - sync threads
#define __CLK_LOCAL_MEM_FENCE    0x01
typedef unsigned __cl_mem_fence_flags;

#endif
