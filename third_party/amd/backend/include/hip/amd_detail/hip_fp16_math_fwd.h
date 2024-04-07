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

// /*
// Half Math Functions
// */
#if !defined(__HIPCC_RTC__)
#include "host_defines.h"
#endif
#ifndef __CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
extern "C"
{
    __device__ __attribute__((const)) _Float16 __ocml_ceil_f16(_Float16);
    __device__ _Float16 __ocml_cos_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_exp_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_exp10_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_exp2_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_floor_f16(_Float16);
    __device__ __attribute__((const))
    _Float16 __ocml_fma_f16(_Float16, _Float16, _Float16);
    __device__ __attribute__((const)) _Float16 __ocml_fabs_f16(_Float16);
    __device__ __attribute__((const)) int __ocml_isinf_f16(_Float16);
    __device__ __attribute__((const)) int __ocml_isnan_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_log_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_log10_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_log2_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_pown_f16(_Float16, int);
    __device__ __attribute__((const)) _Float16 __ocml_rint_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_rsqrt_f16(_Float16);
    __device__ _Float16 __ocml_sin_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_sqrt_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_trunc_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_fmax_f16(_Float16, _Float16);
    __device__ __attribute__((const)) _Float16 __ocml_fmin_f16(_Float16, _Float16);

    typedef _Float16 __2f16 __attribute__((ext_vector_type(2)));
    typedef short __2i16 __attribute__((ext_vector_type(2)));

    #if defined(__clang__) && defined(__HIP__)
    __device__ __attribute__((const)) float __ockl_fdot2(__2f16 a, __2f16 b, float c, bool s);
    #endif

    __device__ __attribute__((const)) __2f16 __ocml_ceil_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_fabs_2f16(__2f16);
    __device__ __2f16 __ocml_cos_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_exp_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_exp10_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_exp2_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_floor_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_fma_2f16(__2f16, __2f16, __2f16);
    __device__ __attribute__((const)) __2i16 __ocml_isinf_2f16(__2f16);
    __device__ __attribute__((const)) __2i16 __ocml_isnan_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_log_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_log10_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_log2_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_rint_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_rsqrt_2f16(__2f16);
    __device__ __2f16 __ocml_sin_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_sqrt_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_trunc_2f16(__2f16);

    __device__ __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
    __device__ __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
    __device__ __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);

}
#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
//TODO: remove these after they get into clang header __clang_hip_libdevice_declares.h'
extern "C" {
    __device__ __attribute__((const)) _Float16 __ocml_fmax_f16(_Float16, _Float16);
    __device__ __attribute__((const)) _Float16 __ocml_fmin_f16(_Float16, _Float16);
    __device__ __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
    __device__ __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
    __device__ __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);
}
