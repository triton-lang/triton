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

#define __OCP_FP_HOST__ __host__
#define __OCP_FP_DEVICE__ __device__
#define __OCP_FP_HOST_DEVICE__ __OCP_FP_HOST__ __OCP_FP_DEVICE__
#define __OCP_FP_DEVICE_STATIC__ __OCP_FP_DEVICE__ static __inline__ __attribute__((always_inline))
#define __OCP_FP_HOST_DEVICE_STATIC__ __OCP_FP_HOST_DEVICE__ static

static_assert(sizeof(unsigned int) == 4);
static_assert(sizeof(float) == 4);
static_assert(sizeof(unsigned short) == 2);

#if (defined(__clang__) && (__clang_major__ > 17) && defined(__HIP__)) ||                          \
    (defined(__GNUC__) && (__GNUC__ > 13))
static_assert(sizeof(__bf16) == 2);
static_assert(sizeof(_Float16) == 2);
#endif

// Although we do have some abstractions of half and bfloat16, since this will be a standalone
// header which will act as a base abstraction, and will be maintained in the future, it makes sense
// to keep these vector types separate from existing implementations. We can add conversion
// functions in a different header using these functions.
typedef uint8_t __amd_fp8_storage_t;
typedef uint16_t __amd_fp8x2_storage_t;
typedef uint8_t __amd_fp4x2_storage_t;
typedef uint32_t __amd_fp4x8_storage_t;
typedef __bf16 __amd_bf16_storage_t;
typedef _Float16 __amd_fp16_storage_t;
typedef int8_t __amd_scale_t;

#if defined(__clang__) && (__clang_major__ > 17) && defined(__HIP__)
typedef unsigned int __attribute__((ext_vector_type(2))) __amd_uintx2_storage_t;
typedef uint8_t __attribute__((ext_vector_type(8))) __amd_fp8x8_storage_t;
typedef __bf16 __attribute__((ext_vector_type(2))) __amd_bf16x2_storage_t;
typedef __bf16 __attribute__((ext_vector_type(8))) __amd_bf16x8_storage_t;
typedef __bf16 __attribute__((ext_vector_type(32))) __amd_bf16x32_storage_t;
typedef float __attribute__((ext_vector_type(2))) __amd_floatx2_storage_t;
typedef float __attribute__((ext_vector_type(8))) __amd_floatx8_storage_t;
typedef float __attribute__((ext_vector_type(16))) __amd_floatx16_storage_t;
typedef float __attribute__((ext_vector_type(32))) __amd_floatx32_storage_t;
typedef _Float16 __attribute__((ext_vector_type(2))) __amd_fp16x2_storage_t;
typedef _Float16 __attribute__((ext_vector_type(8))) __amd_fp16x8_storage_t;
typedef _Float16 __attribute__((ext_vector_type(32))) __amd_fp16x32_storage_t;
typedef uint32_t __attribute__((ext_vector_type(6))) __amd_fp6x32_storage_t;
typedef short __attribute__((ext_vector_type(2))) __amd_shortx2_storage_t;
#elif defined(__GNUC__) && (__GNUC__ > 13)
/* GCC expects vector size in bytes */
typedef unsigned int __attribute__((vector_size(8))) __amd_uintx2_storage_t;
typedef uint8_t __attribute__((vector_size(8))) __amd_fp8x8_storage_t;
typedef __bf16 __attribute__((vector_size(4))) __amd_bf16x2_storage_t;
typedef __bf16 __attribute__((vector_size(16))) __amd_bf16x8_storage_t;
typedef __bf16 __attribute__((vector_size(64))) __amd_bf16x32_storage_t;
typedef float __attribute__((vector_size(8))) __amd_floatx2_storage_t;
typedef float __attribute__((vector_size(32))) __amd_floatx8_storage_t;
typedef float __attribute__((vector_size(64))) __amd_floatx16_storage_t;
typedef float __attribute__((vector_size(128))) __amd_floatx32_storage_t;
typedef _Float16 __attribute__((vector_size(4))) __amd_fp16x2_storage_t;
typedef _Float16 __attribute__((vector_size(16))) __amd_fp16x8_storage_t;
typedef _Float16 __attribute__((vector_size(64))) __amd_fp16x32_storage_t;
typedef uint32_t __attribute__((vector_size(24))) __amd_fp6x32_storage_t;
typedef short __attribute__((vector_size(4))) __amd_shortx2_storage_t;
#else
#error "Only supported by HIPCC or GCC >= 13."
#endif

static_assert(sizeof(__amd_uintx2_storage_t) == sizeof(__amd_fp8x8_storage_t));
