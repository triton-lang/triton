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

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#include <hip/driver_types.h>
#include <hip/amd_detail/amd_hip_vector_types.h>
#endif

#ifdef __cplusplus

extern "C" HIP_PUBLIC_API
hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f);

static inline hipChannelFormatDesc hipCreateChannelDescHalf() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

static inline hipChannelFormatDesc hipCreateChannelDescHalf1() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

static inline hipChannelFormatDesc hipCreateChannelDescHalf2() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindFloat);
}

static inline hipChannelFormatDesc hipCreateChannelDescHalf4() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindFloat);
}

template <typename T>
static inline hipChannelFormatDesc hipCreateChannelDesc() {
    return hipCreateChannelDesc(0, 0, 0, 0, hipChannelFormatKindNone);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<char>() {
    int e = (int)sizeof(char) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<signed char>() {
    int e = (int)sizeof(signed char) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<unsigned char>() {
    int e = (int)sizeof(unsigned char) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uchar1>() {
    int e = (int)sizeof(unsigned char) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<char1>() {
    int e = (int)sizeof(signed char) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uchar2>() {
    int e = (int)sizeof(unsigned char) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<char2>() {
    int e = (int)sizeof(signed char) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__  // vector3 is the same as vector4
template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uchar3>() {
    int e = (int)sizeof(unsigned char) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<char3>() {
    int e = (int)sizeof(signed char) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uchar4>() {
    int e = (int)sizeof(unsigned char) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<char4>() {
    int e = (int)sizeof(signed char) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<unsigned short>() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<signed short>() {
    int e = (int)sizeof(signed short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ushort1>() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<short1>() {
    int e = (int)sizeof(signed short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ushort2>() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<short2>() {
    int e = (int)sizeof(signed short) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__
template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ushort3>() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<short3>() {
    int e = (int)sizeof(signed short) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ushort4>() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<short4>() {
    int e = (int)sizeof(signed short) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<unsigned int>() {
    int e = (int)sizeof(unsigned int) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<signed int>() {
    int e = (int)sizeof(signed int) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uint1>() {
    int e = (int)sizeof(unsigned int) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<int1>() {
    int e = (int)sizeof(signed int) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uint2>() {
    int e = (int)sizeof(unsigned int) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<int2>() {
    int e = (int)sizeof(signed int) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__
template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uint3>() {
    int e = (int)sizeof(unsigned int) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<int3>() {
    int e = (int)sizeof(signed int) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uint4>() {
    int e = (int)sizeof(unsigned int) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<int4>() {
    int e = (int)sizeof(signed int) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<float>() {
    int e = (int)sizeof(float) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<float1>() {
    int e = (int)sizeof(float) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<float2>() {
    int e = (int)sizeof(float) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindFloat);
}

#ifndef __GNUC__
template <>
inline hipChannelFormatDesc hipCreateChannelDesc<float3>() {
    int e = (int)sizeof(float) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindFloat);
}
#endif

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<float4>() {
    int e = (int)sizeof(float) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindFloat);
}

#if !defined(__LP64__)

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<unsigned long>() {
    int e = (int)sizeof(unsigned long) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<signed long>() {
    int e = (int)sizeof(signed long) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ulong1>() {
    int e = (int)sizeof(unsigned long) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<long1>() {
    int e = (int)sizeof(signed long) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ulong2>() {
    int e = (int)sizeof(unsigned long) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<long2>() {
    int e = (int)sizeof(signed long) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__
template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ulong3>() {
    int e = (int)sizeof(unsigned long) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<long3>() {
    int e = (int)sizeof(signed long) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ulong4>() {
    int e = (int)sizeof(unsigned long) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<long4>() {
    int e = (int)sizeof(signed long) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}
#endif /* !__LP64__ */

#else

struct hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                                 enum hipChannelFormatKind f);

#endif /* __cplusplus */

#endif /* !HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H */
