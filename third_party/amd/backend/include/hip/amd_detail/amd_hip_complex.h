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

/* The header defines complex numbers and related functions*/

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMPLEX_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMPLEX_H

#if !defined(__HIPCC_RTC__)
#include "hip/amd_detail/amd_hip_vector_types.h"
#endif

#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#define __HOST_DEVICE__ __host__ __device__
// TODO: Clang has a bug which allows device functions to call std functions
// when std functions are introduced into default namespace by using statement.
// math.h may be included after this bug is fixed.
#if __cplusplus
#include <cmath>
#else
#include "math.h"
#endif
#endif // !defined(__HIPCC_RTC__)

typedef float2 hipFloatComplex;

__HOST_DEVICE__ static inline float hipCrealf(hipFloatComplex z) { return z.x; }

__HOST_DEVICE__ static inline float hipCimagf(hipFloatComplex z) { return z.y; }

__HOST_DEVICE__ static inline hipFloatComplex make_hipFloatComplex(float a, float b) {
    hipFloatComplex z;
    z.x = a;
    z.y = b;
    return z;
}

__HOST_DEVICE__ static inline hipFloatComplex hipConjf(hipFloatComplex z) {
    hipFloatComplex ret;
    ret.x = z.x;
    ret.y = -z.y;
    return ret;
}

__HOST_DEVICE__ static inline float hipCsqabsf(hipFloatComplex z) {
    return z.x * z.x + z.y * z.y;
}

__HOST_DEVICE__ static inline hipFloatComplex hipCaddf(hipFloatComplex p, hipFloatComplex q) {
    return make_hipFloatComplex(p.x + q.x, p.y + q.y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipCsubf(hipFloatComplex p, hipFloatComplex q) {
    return make_hipFloatComplex(p.x - q.x, p.y - q.y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipCmulf(hipFloatComplex p, hipFloatComplex q) {
    return make_hipFloatComplex(p.x * q.x - p.y * q.y, p.y * q.x + p.x * q.y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipCdivf(hipFloatComplex p, hipFloatComplex q) {
    float sqabs = hipCsqabsf(q);
    hipFloatComplex ret;
    ret.x = (p.x * q.x + p.y * q.y) / sqabs;
    ret.y = (p.y * q.x - p.x * q.y) / sqabs;
    return ret;
}

__HOST_DEVICE__ static inline float hipCabsf(hipFloatComplex z) { return sqrtf(hipCsqabsf(z)); }


typedef double2 hipDoubleComplex;

__HOST_DEVICE__ static inline double hipCreal(hipDoubleComplex z) { return z.x; }

__HOST_DEVICE__ static inline double hipCimag(hipDoubleComplex z) { return z.y; }

__HOST_DEVICE__ static inline hipDoubleComplex make_hipDoubleComplex(double a, double b) {
    hipDoubleComplex z;
    z.x = a;
    z.y = b;
    return z;
}

__HOST_DEVICE__ static inline hipDoubleComplex hipConj(hipDoubleComplex z) {
    hipDoubleComplex ret;
    ret.x = z.x;
    ret.y = -z.y;
    return ret;
}

__HOST_DEVICE__ static inline double hipCsqabs(hipDoubleComplex z) {
    return z.x * z.x + z.y * z.y;
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCadd(hipDoubleComplex p, hipDoubleComplex q) {
    return make_hipDoubleComplex(p.x + q.x, p.y + q.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCsub(hipDoubleComplex p, hipDoubleComplex q) {
    return make_hipDoubleComplex(p.x - q.x, p.y - q.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCmul(hipDoubleComplex p, hipDoubleComplex q) {
    return make_hipDoubleComplex(p.x * q.x - p.y * q.y, p.y * q.x + p.x * q.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCdiv(hipDoubleComplex p, hipDoubleComplex q) {
    double sqabs = hipCsqabs(q);
    hipDoubleComplex ret;
    ret.x = (p.x * q.x + p.y * q.y) / sqabs;
    ret.y = (p.y * q.x - p.x * q.y) / sqabs;
    return ret;
}

__HOST_DEVICE__ static inline double hipCabs(hipDoubleComplex z) { return sqrt(hipCsqabs(z)); }

typedef hipFloatComplex hipComplex;

__HOST_DEVICE__ static inline hipComplex make_hipComplex(float x, float y) {
    return make_hipFloatComplex(x, y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipComplexDoubleToFloat(hipDoubleComplex z) {
    return make_hipFloatComplex((float)z.x, (float)z.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipComplexFloatToDouble(hipFloatComplex z) {
    return make_hipDoubleComplex((double)z.x, (double)z.y);
}

__HOST_DEVICE__ static inline hipComplex hipCfmaf(hipComplex p, hipComplex q, hipComplex r) {
    float real = (p.x * q.x) + r.x;
    float imag = (q.x * p.y) + r.y;

    real = -(p.y * q.y) + real;
    imag = (p.x * q.y) + imag;

    return make_hipComplex(real, imag);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCfma(hipDoubleComplex p, hipDoubleComplex q,
                                                           hipDoubleComplex r) {
    double real = (p.x * q.x) + r.x;
    double imag = (q.x * p.y) + r.y;

    real = -(p.y * q.y) + real;
    imag = (p.x * q.y) + imag;

    return make_hipDoubleComplex(real, imag);
}

#endif //HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMPLEX_H
