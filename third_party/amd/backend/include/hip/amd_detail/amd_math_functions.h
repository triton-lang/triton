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

#if !defined(__HIPCC_RTC__)
#include "hip_fp16_math_fwd.h"
#include "amd_hip_vector_types.h"
#include "math_fwd.h"

#include <hip/amd_detail/host_defines.h>

#include <algorithm>
// assert.h is only for the host version of assert.
// The device version of assert is implemented in hip/amd_detail/hip_runtime.h.
// Users should include hip_runtime.h for the device version of assert.
#if !__HIP_DEVICE_COMPILE__
#include <assert.h>
#endif
#include <limits.h>
#include <limits>
#include <stdint.h>
#endif // !defined(__HIPCC_RTC__)

#if _LIBCPP_VERSION && __HIP__
namespace std {
template <>
struct __numeric_type<_Float16>
{
   static _Float16 __test(_Float16);

   typedef _Float16 type;
   static const bool value = true;
};
}
#endif // _LIBCPP_VERSION

#pragma push_macro("__DEVICE__")
#pragma push_macro("__RETURN_TYPE")

#define __DEVICE__ static __device__
#define __RETURN_TYPE bool

// DOT FUNCTIONS
#if defined(__clang__) && defined(__HIP__)
__DEVICE__
inline
int amd_mixed_dot(short2 a, short2 b, int c, bool saturate) {
    return __ockl_sdot2(a.data, b.data, c, saturate);
}
__DEVICE__
inline
uint amd_mixed_dot(ushort2 a, ushort2 b, uint c, bool saturate) {
    return __ockl_udot2(a.data, b.data, c, saturate);
}
__DEVICE__
inline
int amd_mixed_dot(char4 a, char4 b, int c, bool saturate) {
    return __ockl_sdot4(a.data, b.data, c, saturate);
}
__DEVICE__
inline
uint amd_mixed_dot(uchar4 a, uchar4 b, uint c, bool saturate) {
    return __ockl_udot4(a.data, b.data, c, saturate);
}
__DEVICE__
inline
int amd_mixed_dot(int a, int b, int c, bool saturate) {
    return __ockl_sdot8(a, b, c, saturate);
}
__DEVICE__
inline
uint amd_mixed_dot(uint a, uint b, uint c, bool saturate) {
    return __ockl_udot8(a, b, c, saturate);
}
#endif

#pragma pop_macro("__DEVICE__")
#pragma pop_macro("__RETURN_TYPE")
// For backward compatibility.
// There are HIP applications e.g. TensorFlow, expecting __HIP_ARCH_* macros
// defined after including math_functions.h.
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_runtime.h>
#endif
