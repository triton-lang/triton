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

#if __cplusplus
#define COMPLEX_NEG_OP_OVERLOAD(type)                                                              \
    __HOST_DEVICE__ static inline type operator-(const type& op) {                             \
        type ret;                                                                                  \
        ret.x = -op.x;                                                                             \
        ret.y = -op.y;                                                                             \
        return ret;                                                                                \
    }

#define COMPLEX_EQ_OP_OVERLOAD(type)                                                               \
    __HOST_DEVICE__ static inline bool operator==(const type& lhs, const type& rhs) {          \
        return lhs.x == rhs.x && lhs.y == rhs.y;                                                   \
    }

#define COMPLEX_NE_OP_OVERLOAD(type)                                                               \
    __HOST_DEVICE__ static inline bool operator!=(const type& lhs, const type& rhs) {          \
        return !(lhs == rhs);                                                                      \
    }

#define COMPLEX_ADD_OP_OVERLOAD(type)                                                              \
    __HOST_DEVICE__ static inline type operator+(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.x = lhs.x + rhs.x;                                                                     \
        ret.y = lhs.y + rhs.y;                                                                     \
        return ret;                                                                                \
    }

#define COMPLEX_SUB_OP_OVERLOAD(type)                                                              \
    __HOST_DEVICE__ static inline type operator-(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.x = lhs.x - rhs.x;                                                                     \
        ret.y = lhs.y - rhs.y;                                                                     \
        return ret;                                                                                \
    }

#define COMPLEX_MUL_OP_OVERLOAD(type)                                                              \
    __HOST_DEVICE__ static inline type operator*(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.x = lhs.x * rhs.x - lhs.y * rhs.y;                                                     \
        ret.y = lhs.x * rhs.y + lhs.y * rhs.x;                                                     \
        return ret;                                                                                \
    }

#define COMPLEX_DIV_OP_OVERLOAD(type)                                                              \
    __HOST_DEVICE__ static inline type operator/(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.x = (lhs.x * rhs.x + lhs.y * rhs.y);                                                   \
        ret.y = (rhs.x * lhs.y - lhs.x * rhs.y);                                                   \
        ret.x = ret.x / (rhs.x * rhs.x + rhs.y * rhs.y);                                           \
        ret.y = ret.y / (rhs.x * rhs.x + rhs.y * rhs.y);                                           \
        return ret;                                                                                \
    }

#define COMPLEX_ADD_PREOP_OVERLOAD(type)                                                           \
    __HOST_DEVICE__ static inline type& operator+=(type& lhs, const type& rhs) {               \
        lhs.x += rhs.x;                                                                            \
        lhs.y += rhs.y;                                                                            \
        return lhs;                                                                                \
    }

#define COMPLEX_SUB_PREOP_OVERLOAD(type)                                                           \
    __HOST_DEVICE__ static inline type& operator-=(type& lhs, const type& rhs) {               \
        lhs.x -= rhs.x;                                                                            \
        lhs.y -= rhs.y;                                                                            \
        return lhs;                                                                                \
    }

#define COMPLEX_MUL_PREOP_OVERLOAD(type)                                                            \
    __HOST_DEVICE__ static inline type& operator*=(type& lhs, const type& rhs) {                    \
        type temp{lhs};                                                                             \
        lhs.x = rhs.x * temp.x - rhs.y * temp.y;                                                    \
        lhs.y = rhs.y * temp.x + rhs.x * temp.y;                                                    \
        return lhs;                                                                                 \
    }

#define COMPLEX_DIV_PREOP_OVERLOAD(type)                                                           \
    __HOST_DEVICE__ static inline type& operator/=(type& lhs, const type& rhs) {                   \
        type temp;                                                                                 \
        temp.x = (lhs.x*rhs.x + lhs.y * rhs.y) / (rhs.x*rhs.x + rhs.y*rhs.y);                      \
        temp.y = (lhs.y * rhs.x - lhs.x * rhs.y) / (rhs.x*rhs.x + rhs.y*rhs.y);                    \
        lhs = temp;                                                                                \
        return lhs;                                                                                \
    }

#define COMPLEX_SCALAR_PRODUCT(type, type1)                                                        \
    __HOST_DEVICE__ static inline type operator*(const type& lhs, type1 rhs) {                 \
        type ret;                                                                                  \
        ret.x = lhs.x * rhs;                                                                       \
        ret.y = lhs.y * rhs;                                                                       \
        return ret;                                                                                \
    }

#endif

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


#if __cplusplus

COMPLEX_NEG_OP_OVERLOAD(hipFloatComplex)
COMPLEX_EQ_OP_OVERLOAD(hipFloatComplex)
COMPLEX_NE_OP_OVERLOAD(hipFloatComplex)
COMPLEX_ADD_OP_OVERLOAD(hipFloatComplex)
COMPLEX_SUB_OP_OVERLOAD(hipFloatComplex)
COMPLEX_MUL_OP_OVERLOAD(hipFloatComplex)
COMPLEX_DIV_OP_OVERLOAD(hipFloatComplex)
COMPLEX_ADD_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_SUB_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_MUL_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_DIV_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned short)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed short)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned int)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed int)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, float)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned long)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed long)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, double)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed long long)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned long long)

COMPLEX_NEG_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_EQ_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_NE_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_ADD_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_SUB_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_MUL_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_DIV_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_ADD_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_SUB_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_MUL_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_DIV_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned short)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed short)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned int)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed int)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, float)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned long)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed long)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, double)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed long long)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned long long)

#endif


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
