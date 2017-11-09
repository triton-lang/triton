/*
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

#if !defined(__CUDA_FP16_HPP__)
#define __CUDA_FP16_HPP__

/* C++11 header for std::move */
#if __cplusplus >= 201103L
#include <utility>
#endif /* __cplusplus >= 201103L */

/* Set up function decorations */
#if defined(__CUDACC_RTC__)
#define __CUDA_FP16_DECL__ __host__ __device__
#define __VECTOR_FUNCTIONS_DECL__ __host__ __device__
#define __CUDA_HOSTDEVICE__ __host__ __device__
#elif defined(__CUDACC__) /* !__CUDACC_RTC__ but yes __CUDACC__ */
#define __CUDA_FP16_DECL__ static __device__ __inline__
#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__
#define __CUDA_HOSTDEVICE__ __host__ __device__
#else /* !__CUDACC_RTC and !__CUDACC__ (i.e. host non-nvcc compiler */
#define __CUDA_HOSTDEVICE__
#endif /* __CUDACC_RTC__ and __CUDACC__ */

/* Set up structure-alignment attribute */
#if defined(__CUDACC__)
#define __CUDA_ALIGN__(align) __align__(align)
#else
/* Define alignment macro based on compiler type (cannot assume C11 "_Alignas" is available) */
#if __cplusplus >= 201103L
#define __CUDA_ALIGN__(n) alignas(n)    /* C++11 kindly gives us a keyword for this */
#else /* !(__cplusplus >= 201103L)*/
#if defined(__GNUC__) /* || defined(__IBMC__) || defined(__clang__) || defined(__PGI) */
#define __CUDA_ALIGN__(n) __attribute__ ((aligned(n)))
#elif defined(_MSC_VER) /* || defined(__ICC) */
#define __CUDA_ALIGN__(n) __declspec(align(n))
#else
#define __CUDA_ALIGN__(n)
#endif /* defined(__GNUC__) */
#endif /* __cplusplus >= 201103L */
#endif /* defined(__CUDACC__) */


/* Macros to allow half & half2 to be used by inline assembly */
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))


/**
* Types which allow static initialization of "half" and "half2" until
* these become an actual builtin. Note this initialization is as a
* bitfield representation of "half", and not a conversion from short->half.
* Such a representation will be deprecated in a future version of CUDA. 
* (Note these are visible to non-nvcc compilers, including C-only compilation)
*/
typedef struct __CUDA_ALIGN__(2) {
    unsigned short x;
} __half_raw;

typedef struct __CUDA_ALIGN__(4) {
    unsigned short x, y;
} __half2_raw;

/* All other definitions in this file are only visible to C++ compilers */
#if defined(__cplusplus)

/* Hide GCC member initialization list warnings because of host/device in-function init requirement */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Weffc++"
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

struct __CUDA_ALIGN__(2) __half {
protected:
    unsigned short __x;

public:
#if __cplusplus >= 201103L
    __half() = default;
#else
    __CUDA_HOSTDEVICE__ __half() { }
#endif /* __cplusplus >= 201103L */

    /* Convert to/from __half_raw */
    __CUDA_HOSTDEVICE__ __half(const __half_raw &hr) : __x(hr.x) { }
    __CUDA_HOSTDEVICE__ __half &operator=(const __half_raw &hr) { __x = hr.x; return *this; }
    __CUDA_HOSTDEVICE__ operator __half_raw() const { __half_raw ret; ret.x = __x; return ret; }

/* Member functions are only available to nvcc compilation */
#if defined(__CUDACC__)
#if !defined(__CUDA_NO_HALF_CONVERSIONS__)
    /* Allow automatic construction from types supported natively in hardware */
    /* Note we do avoid constructor init-list because of special host/device compilation rules */
    __device__ __half(float f) { __x = __float2half(f).__x;  }
    __device__ __half(double f) { __x = __float2half((float)f).__x;  }
    __device__ __half(short val) { __x = __short2half_rn(val).__x;  }
    __device__ __half(unsigned short val) { __x = __ushort2half_rn(val).__x;  }
    __device__ __half(int val) { __x = __int2half_rn(val).__x;  }
    __device__ __half(unsigned int val) { __x = __uint2half_rn(val).__x;  }
    __device__ __half(long long val) { __x = __ll2half_rn(val).__x;  }
    __device__ __half(unsigned long long val) { __x = __ull2half_rn(val).__x; }

    /* Allow automatic casts to supported builtin types, matching all that are permitted with float */
    __device__ operator float() const { return __half2float(*this); }
    __device__ __half &operator=(float f) { __x = __float2half(f).__x; return *this; }

    /* We omit "cast to double" operator, so as to not be ambiguous about up-cast */
    __device__ __half &operator=(double f) { __x = __float2half((float)f).__x; return *this; }

    __device__ operator short() const { return __half2short_rn(*this); }
    __device__ __half &operator=(short val) { __x = __short2half_rn(val).__x; return *this; }

    __device__ operator unsigned short() const { return __half2ushort_rn(*this); }
    __device__ __half &operator=(unsigned short val) { __x = __ushort2half_rn(val).__x; return *this; }

    __device__ operator int() const { return __half2int_rn(*this); }
    __device__ __half &operator=(int val) { __x = __int2half_rn(val).__x; return *this; }

    __device__ operator unsigned int() const { return __half2uint_rn(*this); }
    __device__ __half &operator=(unsigned int val) { __x = __uint2half_rn(val).__x; return *this; }

    __device__ operator long long() const { return __half2ll_rn(*this); }
    __device__ __half &operator=(long long val) { __x = __ll2half_rn(val).__x; return *this; }

    __device__ operator unsigned long long() const { return __half2ull_rn(*this); }
    __device__ __half &operator=(unsigned long long val) { __x = __ull2half_rn(val).__x; return *this; }

    /* Boolean conversion - note both 0 and -0 must return false */
    __device__ operator bool() const { return (__x & 0x7FFF) != 0; }
#endif /* !defined(__CUDA_NO_HALF_CONVERSIONS__) */
#endif /* defined(__CUDACC__) */
};

/* Global-space operator functions are only available to nvcc compilation */
#if defined(__CUDACC__)

/* Arithmetic FP16 operations only supported on arch >= 5.3 */
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
#if !defined(__CUDA_NO_HALF_OPERATORS__)
/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ __half operator+(const __half &lh, const __half &rh) { return __hadd(lh, rh); }
__device__ __forceinline__ __half operator-(const __half &lh, const __half &rh) { return __hsub(lh, rh); }
__device__ __forceinline__ __half operator*(const __half &lh, const __half &rh) { return __hmul(lh, rh); }
__device__ __forceinline__ __half operator/(const __half &lh, const __half &rh) { return __hdiv(lh, rh); }

__device__ __forceinline__ __half &operator+=(__half &lh, const __half &rh) { lh = __hadd(lh, rh); return lh; }
__device__ __forceinline__ __half &operator-=(__half &lh, const __half &rh) { lh = __hsub(lh, rh); return lh; }
__device__ __forceinline__ __half &operator*=(__half &lh, const __half &rh) { lh = __hmul(lh, rh); return lh; }
__device__ __forceinline__ __half &operator/=(__half &lh, const __half &rh) { lh = __hdiv(lh, rh); return lh; }

/* Note for increment and decrement we use the raw value 0x3C00 equating to half(1.0f), to avoid the extra conversion */
__device__ __forceinline__ __half &operator++(__half &h)      { __half_raw one; one.x = 0x3C00; h += one; return h; }
__device__ __forceinline__ __half &operator--(__half &h)      { __half_raw one; one.x = 0x3C00; h -= one; return h; }
__device__ __forceinline__ __half  operator++(__half &h, int) { __half ret = h; __half_raw one; one.x = 0x3C00; h += one; return ret; }
__device__ __forceinline__ __half  operator--(__half &h, int) { __half ret = h; __half_raw one; one.x = 0x3C00; h -= one; return ret; }

/* Unary plus and inverse operators */
__device__ __forceinline__ __half operator+(const __half &h) { return h; }
__device__ __forceinline__ __half operator-(const __half &h) { return __hneg(h); }

/* Some basic comparison operations to make it look like a builtin */
__device__ __forceinline__ bool operator==(const __half &lh, const __half &rh) { return __heq(lh, rh); }
__device__ __forceinline__ bool operator!=(const __half &lh, const __half &rh) { return __hne(lh, rh); }
__device__ __forceinline__ bool operator> (const __half &lh, const __half &rh) { return __hgt(lh, rh); }
__device__ __forceinline__ bool operator< (const __half &lh, const __half &rh) { return __hlt(lh, rh); }
__device__ __forceinline__ bool operator>=(const __half &lh, const __half &rh) { return __hge(lh, rh); }
__device__ __forceinline__ bool operator<=(const __half &lh, const __half &rh) { return __hle(lh, rh); }
#endif /* !defined(__CUDA_NO_HALF_OPERATORS__) */
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
#endif /* defined(__CUDACC__) */

/* __half2 is visible to non-nvcc host compilers */
struct __CUDA_ALIGN__(4) __half2 {
    __half x, y;

    // All construct/copy/assign/move
public:
#if __cplusplus >= 201103L
    __half2() = default;
    __CUDA_HOSTDEVICE__ __half2(__half2 &&src) { __HALF2_TO_UI(*this) = std::move(__HALF2_TO_CUI(src)); }
    __CUDA_HOSTDEVICE__ __half2 &operator=(__half2 &&src) { __HALF2_TO_UI(*this) = std::move(__HALF2_TO_CUI(src)); return *this; }
#else
    __CUDA_HOSTDEVICE__ __half2() { }
#endif /* __cplusplus >= 201103L */
    __CUDA_HOSTDEVICE__ __half2(const __half &a, const __half &b) : x(a), y(b) { }
    __CUDA_HOSTDEVICE__ __half2(const __half2 &src) { __HALF2_TO_UI(*this) = __HALF2_TO_CUI(src); }
    __CUDA_HOSTDEVICE__ __half2 &operator=(const __half2 &src) { __HALF2_TO_UI(*this) = __HALF2_TO_CUI(src); return *this; }

    /* Convert to/from __half2_raw */
    __CUDA_HOSTDEVICE__ __half2(const __half2_raw &h2r ) { __HALF2_TO_UI(*this) = __HALF2_TO_CUI(h2r); }
    __CUDA_HOSTDEVICE__ __half2 &operator=(const __half2_raw &h2r) { __HALF2_TO_UI(*this) = __HALF2_TO_CUI(h2r); return *this; }
    __CUDA_HOSTDEVICE__ operator __half2_raw() const { __half2_raw ret; __HALF2_TO_UI(ret) = __HALF2_TO_CUI(*this); return ret; }
};

/* Restore -Weffc++ warnings from here on */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

#undef __CUDA_HOSTDEVICE__
#undef __CUDA_ALIGN__

/* All intrinsic functions are only available to nvcc compilers */
#if defined(__CUDACC__)

/* CUDA vector-types compatible vector creation function (note returns __half2, not half2) */
__VECTOR_FUNCTIONS_DECL__ __half2 make_half2(__half x, __half y)
{
    __half2 t; t.x = x; t.y = y; return t;
}
#undef __VECTOR_FUNCTIONS_DECL__


/* Definitions of intrinsics */
__CUDA_FP16_DECL__ int __half2int_rn(__half h)
{
    int i;
    asm("cvt.rni.s32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ int __half2int_rz(__half h)
{
    int i;
    asm("cvt.rzi.s32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ int __half2int_rd(__half h)
{
    int i;
    asm("cvt.rmi.s32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ int __half2int_ru(__half h)
{
    int i;
    asm("cvt.rpi.s32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ __half __int2half_rn(int i)
{
    __half h;
    asm("cvt.rn.f16.s32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __int2half_rz(int i)
{
    __half h;
    asm("cvt.rz.f16.s32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __int2half_rd(int i)
{
    __half h;
    asm("cvt.rm.f16.s32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __int2half_ru(int i)
{
    __half h;
    asm("cvt.rp.f16.s32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
    return h;
}

__CUDA_FP16_DECL__ short int __half2short_rn(__half h)
{
    short int i;
    asm("cvt.rni.s16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ short int __half2short_rz(__half h)
{
    short int i;
    asm("cvt.rzi.s16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ short int __half2short_rd(__half h)
{
    short int i;
    asm("cvt.rmi.s16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ short int __half2short_ru(__half h)
{
    short int i;
    asm("cvt.rpi.s16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ __half __short2half_rn(short int i)
{
    __half h;
    asm("cvt.rn.f16.s16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __short2half_rz(short int i)
{
    __half h;
    asm("cvt.rz.f16.s16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __short2half_rd(short int i)
{
    __half h;
    asm("cvt.rm.f16.s16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __short2half_ru(short int i)
{
    __half h;
    asm("cvt.rp.f16.s16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
    return h;
}

__CUDA_FP16_DECL__ unsigned int __half2uint_rn(__half h)
{
    unsigned int i;
    asm("cvt.rni.u32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned int __half2uint_rz(__half h)
{
    unsigned int i;
    asm("cvt.rzi.u32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned int __half2uint_rd(__half h)
{
    unsigned int i;
    asm("cvt.rmi.u32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned int __half2uint_ru(__half h)
{
    unsigned int i;
    asm("cvt.rpi.u32.f16 %0, %1;" : "=r"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ __half __uint2half_rn(unsigned int i)
{
    __half h;
    asm("cvt.rn.f16.u32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __uint2half_rz(unsigned int i)
{
    __half h;
    asm("cvt.rz.f16.u32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __uint2half_rd(unsigned int i)
{
    __half h;
    asm("cvt.rm.f16.u32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __uint2half_ru(unsigned int i)
{
    __half h;
    asm("cvt.rp.f16.u32 %0, %1;" : "=h"(__HALF_TO_US(h)) : "r"(i));
    return h;
}

__CUDA_FP16_DECL__ unsigned short int __half2ushort_rn(__half h)
{
    unsigned short int i;
    asm("cvt.rni.u16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rz(__half h)
{
    unsigned short int i;
    asm("cvt.rzi.u16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rd(__half h)
{
    unsigned short int i;
    asm("cvt.rmi.u16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned short int __half2ushort_ru(__half h)
{
    unsigned short int i;
    asm("cvt.rpi.u16.f16 %0, %1;" : "=h"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ __half __ushort2half_rn(unsigned short int i)
{
    __half h;
    asm("cvt.rn.f16.u16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __ushort2half_rz(unsigned short int i)
{
    __half h;
    asm("cvt.rz.f16.u16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __ushort2half_rd(unsigned short int i)
{
    __half h;
    asm("cvt.rm.f16.u16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __ushort2half_ru(unsigned short int i)
{
    __half h;
    asm("cvt.rp.f16.u16 %0, %1;" : "=h"(__HALF_TO_US(h)) : "h"(i));
    return h;
}

__CUDA_FP16_DECL__ unsigned long long int __half2ull_rn(__half h)
{
    unsigned long long int i;
    asm("cvt.rni.u64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rz(__half h)
{
    unsigned long long int i;
    asm("cvt.rzi.u64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rd(__half h)
{
    unsigned long long int i;
    asm("cvt.rmi.u64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned long long int __half2ull_ru(__half h)
{
    unsigned long long int i;
    asm("cvt.rpi.u64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ __half __ull2half_rn(unsigned long long int i)
{
    __half h;
    asm("cvt.rn.f16.u64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __ull2half_rz(unsigned long long int i)
{
    __half h;
    asm("cvt.rz.f16.u64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __ull2half_rd(unsigned long long int i)
{
    __half h;
    asm("cvt.rm.f16.u64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __ull2half_ru(unsigned long long int i)
{
    __half h;
    asm("cvt.rp.f16.u64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
    return h;
}

__CUDA_FP16_DECL__ long long int __half2ll_rn(__half h)
{
    long long int i;
    asm("cvt.rni.s64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ long long int __half2ll_rz(__half h)
{
    long long int i;
    asm("cvt.rzi.s64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ long long int __half2ll_rd(__half h)
{
    long long int i;
    asm("cvt.rmi.s64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ long long int __half2ll_ru(__half h)
{
    long long int i;
    asm("cvt.rpi.s64.f16 %0, %1;" : "=l"(i) : "h"(__HALF_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ __half __ll2half_rn(long long int i)
{
    __half h;
    asm("cvt.rn.f16.s64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __ll2half_rz(long long int i)
{
    __half h;
    asm("cvt.rz.f16.s64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __ll2half_rd(long long int i)
{
    __half h;
    asm("cvt.rm.f16.s64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ __half __ll2half_ru(long long int i)
{
    __half h;
    asm("cvt.rp.f16.s64 %0, %1;" : "=h"(__HALF_TO_US(h)) : "l"(i));
    return h;
}

__CUDA_FP16_DECL__ __half htrunc(const __half h)
{
    __half r;
    asm("cvt.rzi.f16.f16 %0, %1;" : "=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(h)));
    return r;
}
__CUDA_FP16_DECL__ __half hceil(const __half h)
{
    __half r;
    asm("cvt.rpi.f16.f16 %0, %1;" : "=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(h)));
    return r;
}
__CUDA_FP16_DECL__ __half hfloor(const __half h)
{
    __half r;
    asm("cvt.rmi.f16.f16 %0, %1;" : "=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(h)));
    return r;
}
__CUDA_FP16_DECL__ __half hrint(const __half h)
{
    __half r;
    asm("cvt.rni.f16.f16 %0, %1;" : "=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(h)));
    return r;
}

__CUDA_FP16_DECL__ __half2 h2trunc(const __half2 h)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rzi.f16.f16 low, low;\n"
        "  cvt.rzi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2ceil(const __half2 h)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rpi.f16.f16 low, low;\n"
        "  cvt.rpi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2floor(const __half2 h)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rmi.f16.f16 low, low;\n"
        "  cvt.rmi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2rint(const __half2 h)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rni.f16.f16 low, low;\n"
        "  cvt.rni.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(h)));
    return val;
}

__CUDA_FP16_DECL__ float2 __half22float2(const __half2 l)
{
    float hi_float;
    float lo_float;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, low;}\n" : "=f"(lo_float) : "r"(__HALF2_TO_CUI(l)));

    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, high;}\n" : "=f"(hi_float) : "r"(__HALF2_TO_CUI(l)));

    return make_float2(lo_float, hi_float);
}
__CUDA_FP16_DECL__ __half __float2half(const float f) 
{ 
    __half val;
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
    return val;
}
__CUDA_FP16_DECL__ __half __float2half_rn(const float f)
{
    __half val;
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
    return val;
}
__CUDA_FP16_DECL__ __half __float2half_rz(const float f)
{
    __half val;
    asm("{  cvt.rz.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
    return val;
}
__CUDA_FP16_DECL__ __half __float2half_rd(const float f)
{
    __half val;
    asm("{  cvt.rm.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
    return val;
}
__CUDA_FP16_DECL__ __half __float2half_ru(const float f)
{
    __half val;
    asm("{  cvt.rp.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
    return val;
}
__CUDA_FP16_DECL__ float __half2float(const __half h)
{
    float val;
    asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
    return val;
}
__CUDA_FP16_DECL__ __half2 __float2half2_rn(const float f)
{
    __half2 val;
    asm("{.reg .f16 low;\n"
        "  cvt.rn.f16.f32 low, %1;\n"
        "  mov.b32 %0, {low,low};}\n" : "=r"(__HALF2_TO_UI(val)) : "f"(f));
    return val;
}
__CUDA_FP16_DECL__ __half2 __floats2half2_rn(const float f1, const float f2)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  cvt.rn.f16.f32 low, %1;\n"
        "  cvt.rn.f16.f32 high, %2;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "f"(f1), "f"(f2));
    return val;
}
__CUDA_FP16_DECL__ __half2 __float22half2_rn(const float2 f)
{
    __half2 val = __floats2half2_rn(f.x, f.y);
    return val;
}
__CUDA_FP16_DECL__ float __low2float(const __half2 l)
{
    float val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, low;}\n" : "=f"(val) : "r"(__HALF2_TO_CUI(l)));
    return val;
}
__CUDA_FP16_DECL__ float __high2float(const __half2 l)
{
    float val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, high;}\n" : "=f"(val) : "r"(__HALF2_TO_CUI(l)));
    return val;
}
__CUDA_FP16_DECL__ __half2 __lows2half2(const __half2 l, const __half2 h)
{
    __half2 val;
    asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
        "  mov.b32 {alow,ahigh}, %1;\n"
        "  mov.b32 {blow,bhigh}, %2;\n"
        "  mov.b32 %0, {alow,blow};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(l)), "r"(__HALF2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ __half2 __highs2half2(const __half2 l, const __half2 h)
{
    __half2 val;
    asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
        "  mov.b32 {alow,ahigh}, %1;\n"
        "  mov.b32 {blow,bhigh}, %2;\n"
        "  mov.b32 %0, {ahigh,bhigh};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(l)), "r"(__HALF2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ __half __low2half(const __half2 h)
{
    __half ret;
    asm("{.reg .f16 low,high;\n"
        " mov.b32 {low,high}, %1;\n"
        " mov.b16 %0, low;}" : "=h"(__HALF_TO_US(ret)) : "r"(__HALF2_TO_CUI(h)));
    return ret;
}
__CUDA_FP16_DECL__ int __hisinf(const __half a)
{
    if (__HALF_TO_CUS(a) == 0xFC00)
        return -1;
    if (__HALF_TO_CUS(a) == 0x7C00)
        return 1;
    return 0;
}
__CUDA_FP16_DECL__ __half2 __low2half2(const __half2 l)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {low,low};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(l)));
    return val;
}
__CUDA_FP16_DECL__ __half2 __high2half2(const __half2 l)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {high,high};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(l)));
    return val;
}
__CUDA_FP16_DECL__ __half __high2half(const __half2 h)
{
    __half ret;
    asm("{.reg .f16 low,high;\n"
        " mov.b32 {low,high}, %1;\n"
        " mov.b16 %0, high;}" : "=h"(__HALF_TO_US(ret)) : "r"(__HALF2_TO_CUI(h)));
    return ret;
}
__CUDA_FP16_DECL__ __half2 __halves2half2(const __half l, const __half h)
{
    __half2 val;
    asm("{  mov.b32 %0, {%1,%2};}\n"
        : "=r"(__HALF2_TO_UI(val)) : "h"(__HALF_TO_CUS(l)), "h"(__HALF_TO_CUS(h)));
    return val;
}
__CUDA_FP16_DECL__ __half2 __half2half2(const __half lh)
{
    __half2 val;
    asm("{  mov.b32 %0, {%1,%1};}\n"
        : "=r"(__HALF2_TO_UI(val)) : "h"(__HALF_TO_CUS(lh)));
    return val;
}
__CUDA_FP16_DECL__ __half2 __lowhigh2highlow(const __half2 lh)
{
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {high,low};}\n" : "=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(lh)));
    return val;
}
__CUDA_FP16_DECL__ short int __half_as_short(const __half h)
{
    return (short int)__HALF_TO_CUS(h);
}
__CUDA_FP16_DECL__ unsigned short int __half_as_ushort(const __half h)
{
    return __HALF_TO_CUS(h);
}
__CUDA_FP16_DECL__ __half __short_as_half(const short int i)
{
    __half h;
    __HALF_TO_US(h) = (unsigned short int)i;
    return h;
}
__CUDA_FP16_DECL__ __half __ushort_as_half(const unsigned short int i)
{
    __half h;
    __HALF_TO_US(h) = i;
    return h;
}

#if __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)
/******************************************************************************
*                           __half, __half2 warp shuffle                     *
******************************************************************************/
#define __SHUFFLE_HALF2_MACRO(name) do {\
   __half2 r; \
   asm("{"#name" %0,%1,%2,%3;\n}" \
       :"=r"(__HALF2_TO_UI(r)): "r"(__HALF2_TO_CUI(var)), "r"(delta), "r"(c)); \
   return r; \
} while(0);

#define __SHUFFLE_SYNC_HALF2_MACRO(name) do {\
   __half2 r; \
   asm("{"#name" %0,%1,%2,%3,%4;\n}" \
       :"=r"(__HALF2_TO_UI(r)): "r"(__HALF2_TO_CUI(var)), "r"(delta), "r"(c), "r"(mask)); \
   return r; \
} while(0);

__CUDA_FP16_DECL__ __half2 __shfl(__half2 var, int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_HALF2_MACRO(shfl.idx.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_up(__half2 var, unsigned int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = (warpSize - width) << 8;
    __SHUFFLE_HALF2_MACRO(shfl.up.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_down(__half2 var, unsigned int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_HALF2_MACRO(shfl.down.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_xor(__half2 var, int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_HALF2_MACRO(shfl.bfly.b32);
}

__CUDA_FP16_DECL__ __half2 __shfl_sync(unsigned mask, __half2 var, int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.idx.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_up_sync(unsigned mask, __half2 var, unsigned int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = (warpSize - width) << 8;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.up.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_down_sync(unsigned mask, __half2 var, unsigned int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.down.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_xor_sync(unsigned mask, __half2 var, int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.bfly.b32);
}

#undef __SHUFFLE_HALF2_MACRO
#undef __SHUFFLE_SYNC_HALF2_MACRO

__CUDA_FP16_DECL__ __half __shfl(__half var, int delta, int width)
{
    __half2 temp1 = __halves2half2(var, var);
    __half2 temp2 = __shfl(temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_up(__half var, unsigned int delta, int width)
{
    __half2 temp1 = __halves2half2(var, var);
    __half2 temp2 = __shfl_up(temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_down(__half var, unsigned int delta, int width)
{
    __half2 temp1 = __halves2half2(var, var);
    __half2 temp2 = __shfl_down(temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_xor(__half var, int delta, int width)
{
    __half2 temp1 = __halves2half2(var, var);
    __half2 temp2 = __shfl_xor(temp1, delta, width);
    return __low2half(temp2);
}

__CUDA_FP16_DECL__ __half __shfl_sync(unsigned mask, __half var, int delta, int width)
{
    __half2 temp1 = __halves2half2(var, var);
    __half2 temp2 = __shfl_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_up_sync(unsigned mask, __half var, unsigned int delta, int width)
{
    __half2 temp1 = __halves2half2(var, var);
    __half2 temp2 = __shfl_up_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_down_sync(unsigned mask, __half var, unsigned int delta, int width)
{
    __half2 temp1 = __halves2half2(var, var);
    __half2 temp2 = __shfl_down_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ __half __shfl_xor_sync(unsigned mask, __half var, int delta, int width)
{
    __half2 temp1 = __halves2half2(var, var);
    __half2 temp2 = __shfl_xor_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}

#endif /*__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)*/
/******************************************************************************
*               __half and __half2 __ldg,__ldcg,__ldca,__ldcs                *
******************************************************************************/

#if defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/
__CUDA_FP16_DECL__ __half2 __ldg(const  __half2 *ptr)
{
    __half2 ret;
    asm ("ld.global.nc.b32 %0, [%1];"  : "=r"(__HALF2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half __ldg(const __half *ptr)
{
    __half ret;
    asm ("ld.global.nc.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half2 __ldcg(const  __half2 *ptr)
{
    __half2 ret;
    asm ("ld.global.cg.b32 %0, [%1];"  : "=r"(__HALF2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half __ldcg(const __half *ptr)
{
    __half ret;
    asm ("ld.global.cg.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half2 __ldca(const  __half2 *ptr)
{
    __half2 ret;
    asm ("ld.global.ca.b32 %0, [%1];"  : "=r"(__HALF2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half __ldca(const __half *ptr)
{
    __half ret;
    asm ("ld.global.ca.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half2 __ldcs(const  __half2 *ptr)
{
    __half2 ret;
    asm ("ld.global.cs.b32 %0, [%1];"  : "=r"(__HALF2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ __half __ldcs(const __half *ptr)
{
    __half ret;
    asm ("ld.global.cs.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
#undef __LDG_PTR
#endif /*defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))*/
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
/******************************************************************************
*                             __half2 comparison                             *
******************************************************************************/
#define __COMPARISON_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{ "#name".f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half2 __heq2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.eq);
}
__CUDA_FP16_DECL__ __half2 __hne2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.ne);
}
__CUDA_FP16_DECL__ __half2 __hle2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.le);
}
__CUDA_FP16_DECL__ __half2 __hge2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.ge);
}
__CUDA_FP16_DECL__ __half2 __hlt2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.lt);
}
__CUDA_FP16_DECL__ __half2 __hgt2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.gt);
}
__CUDA_FP16_DECL__ __half2 __hequ2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.equ);
}
__CUDA_FP16_DECL__ __half2 __hneu2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.neu);
}
__CUDA_FP16_DECL__ __half2 __hleu2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.leu);
}
__CUDA_FP16_DECL__ __half2 __hgeu2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.geu);
}
__CUDA_FP16_DECL__ __half2 __hltu2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.ltu);
}
__CUDA_FP16_DECL__ __half2 __hgtu2(const __half2 a, const __half2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.gtu);
}
#undef __COMPARISON_OP_HALF2_MACRO
#define __BOOL_COMPARISON_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{ "#name".f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
   if (__HALF2_TO_CUI(val) == 0x3C003C00) \
      return true; \
   else  \
      return false; \
} while(0);
__CUDA_FP16_DECL__ bool __hbeq2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.eq);
}
__CUDA_FP16_DECL__ bool __hbne2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.ne);
}
__CUDA_FP16_DECL__ bool __hble2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.le);
}
__CUDA_FP16_DECL__ bool __hbge2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.ge);
}
__CUDA_FP16_DECL__ bool __hblt2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.lt);
}
__CUDA_FP16_DECL__ bool __hbgt2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.gt);
}
__CUDA_FP16_DECL__ bool __hbequ2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.equ);
}
__CUDA_FP16_DECL__ bool __hbneu2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.neu);
}
__CUDA_FP16_DECL__ bool __hbleu2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.leu);
}
__CUDA_FP16_DECL__ bool __hbgeu2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.geu);
}
__CUDA_FP16_DECL__ bool __hbltu2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.ltu);
}
__CUDA_FP16_DECL__ bool __hbgtu2(const __half2 a, const __half2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.gtu);
}
#undef __BOOL_COMPARISON_OP_HALF2_MACRO
/******************************************************************************
*                             __half comparison                              *
******************************************************************************/
#define __COMPARISON_OP_HALF_MACRO(name) do {\
   unsigned short val; \
   asm( "{ .reg .pred __$temp3;\n" \
        "  setp."#name".f16  __$temp3, %1, %2;\n" \
        "  selp.u16 %0, 1, 0, __$temp3;}" \
        : "=h"(val) : "h"(__HALF_TO_CUS(a)), "h"(__HALF_TO_CUS(b))); \
   return val ? true : false; \
} while(0);
__CUDA_FP16_DECL__ bool __heq(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(eq);
}
__CUDA_FP16_DECL__ bool __hne(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(ne);
}
__CUDA_FP16_DECL__ bool __hle(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(le);
}
__CUDA_FP16_DECL__ bool __hge(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(ge);
}
__CUDA_FP16_DECL__ bool __hlt(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(lt);
}
__CUDA_FP16_DECL__ bool __hgt(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(gt);
}
__CUDA_FP16_DECL__ bool __hequ(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(equ);
}
__CUDA_FP16_DECL__ bool __hneu(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(neu);
}
__CUDA_FP16_DECL__ bool __hleu(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(leu);
}
__CUDA_FP16_DECL__ bool __hgeu(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(geu);
}
__CUDA_FP16_DECL__ bool __hltu(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(ltu);
}
__CUDA_FP16_DECL__ bool __hgtu(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(gtu);
}
#undef __COMPARISON_OP_HALF_MACRO
/******************************************************************************
*                            __half2 arithmetic                             *
******************************************************************************/
#define __BINARY_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{"#name".f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
   return val; \
} while(0);

__CUDA_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b)
{
    __BINARY_OP_HALF2_MACRO(add);
}
__CUDA_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b)
{
    __BINARY_OP_HALF2_MACRO(sub);
}
__CUDA_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b)
{
    __BINARY_OP_HALF2_MACRO(mul);
}
__CUDA_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b)
{
    __BINARY_OP_HALF2_MACRO(add.sat);
}
__CUDA_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b)
{
    __BINARY_OP_HALF2_MACRO(sub.sat);
}
__CUDA_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b)
{
    __BINARY_OP_HALF2_MACRO(mul.sat);
}
#undef __BINARY_OP_HALF2_MACRO
#define __TERNARY_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{"#name".f16x2 %0,%1,%2,%3;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b)),"r"(__HALF2_TO_CUI(c))); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c)
{
    __TERNARY_OP_HALF2_MACRO(fma.rn);
}
__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c)
{
    __TERNARY_OP_HALF2_MACRO(fma.rn.sat);
}
#undef __TERNARY_OP_HALF2_MACRO
__CUDA_FP16_DECL__ __half2 __h2div(__half2 a, __half2 b) {
    __half ha, hb;

    ha = __low2half(a);
    hb = __low2half(b);

    __half v1 = __hdiv(ha, hb);

    ha = __high2half(a);
    hb = __high2half(b);

    __half v2 = __hdiv(ha, hb);

    return __halves2half2(v1, v2);
}
/******************************************************************************
*                             __half arithmetic                             *
******************************************************************************/
#define __BINARY_OP_HALF_MACRO(name) do {\
   __half val; \
   asm( "{"#name".f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half __hadd(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(add);
}
__CUDA_FP16_DECL__ __half __hsub(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(sub);
}
__CUDA_FP16_DECL__ __half __hmul(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(mul);
}
__CUDA_FP16_DECL__ __half __hadd_sat(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(add.sat);
}
__CUDA_FP16_DECL__ __half __hsub_sat(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(sub.sat);
}
__CUDA_FP16_DECL__ __half __hmul_sat(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(mul.sat);
}
#undef __BINARY_OP_HALF_MACRO
#define __TERNARY_OP_HALF_MACRO(name) do {\
   __half val; \
   asm( "{"#name".f16 %0,%1,%2,%3;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b)),"h"(__HALF_TO_CUS(c))); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c)
{
    __TERNARY_OP_HALF_MACRO(fma.rn);
}
__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c)
{
    __TERNARY_OP_HALF_MACRO(fma.rn.sat);
}
#undef __TERNARY_OP_HALF2_MACRO
__CUDA_FP16_DECL__ __half __hdiv(__half a, __half b) {
    __half v, abs, den;
    __HALF_TO_US(den) = 0x008F;
    float fa, fb, fv, rcp;

    fa = __half2float(a);
    fb = __half2float(b);

    asm("{rcp.approx.f32 %0, %1;\n}" :"=f"(rcp) : "f"(fb));

    fv = rcp * fa;

    v = __float2half(fv);
    __HALF_TO_US(abs) = (unsigned short)(((unsigned int)__HALF_TO_CUS(v)) & 0x00007FFF);
    if (__hlt(abs, den) && (!(__HALF_TO_CUS(abs) == 0x0000))) {
        float err = __fmaf_rn(-fb, fv, fa);
        fv = __fmaf_rn(rcp, err, fv);
        v = __float2half(fv);
    }
    return v;
}

/******************************************************************************
*                             __half2 functions                  *
******************************************************************************/
#define __SPEC_CASE2(i,r, spc, ulp) \
   "{.reg.b32 spc, ulp, p;\n"\
   "  mov.b32 spc,"#spc";\n"\
   "  mov.b32 ulp,"#ulp";\n"\
   "  set.eq.f16x2.f16x2 p,"#i", spc;\n"\
   "  fma.rn.f16x2 "#r",p,ulp,"#r";\n}\n"
#define __SPEC_CASE(i,r, spc, ulp) \
   "{.reg.b16 spc, ulp, p;\n"\
   "  mov.b16 spc,"#spc";\n"\
   "  mov.b16 ulp,"#ulp";\n"\
   "  set.eq.f16.f16 p,"#i", spc;\n"\
   "  fma.rn.f16 "#r",p,ulp,"#r";\n}\n"
#define __APPROX_FCAST(fun) do {\
   __half val;\
   asm("{.reg.b32         f;        \n"\
                " .reg.b16         r;        \n"\
                "  mov.b16         r,%1;     \n"\
                "  cvt.f32.f16     f,r;      \n"\
                "  "#fun".approx.f32   f,f;  \n"\
                "  cvt.rn.f16.f32      r,f;  \n"\
                "  mov.b16         %0,r;     \n"\
                "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));\
   return val;\
} while(0);
#define __APPROX_FCAST2(fun) do {\
   __half2 val;\
   asm("{.reg.b16         hl, hu;         \n"\
                " .reg.b32         fl, fu;         \n"\
                "  mov.b32         {hl, hu}, %1;   \n"\
                "  cvt.f32.f16     fl, hl;         \n"\
                "  cvt.f32.f16     fu, hu;         \n"\
                "  "#fun".approx.f32   fl, fl;     \n"\
                "  "#fun".approx.f32   fu, fu;     \n"\
                "  cvt.rn.f16.f32      hl, fl;     \n"\
                "  cvt.rn.f16.f32      hu, fu;     \n"\
                "  mov.b32         %0, {hl, hu};   \n"\
                "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));       \
   return val;\
} while(0);
static __device__ __forceinline__ float __float_simpl_sinf(float);
static __device__ __forceinline__ float __float_simpl_cosf(float);
__CUDA_FP16_DECL__ __half __hsin_internal(const __half a) {
    float f = __half2float(a);
    f = __float_simpl_sinf(f);
    return __float2half_rn(f);
}
__CUDA_FP16_DECL__ __half hsin(const __half a) {
    __half r = __hsin_internal(a);
    asm("{\n\t"
        "  .reg.b16 i,r,t;     \n\t"
        "  mov.b16 r, %0;      \n\t"
        "  mov.b16 i, %1;      \n\t"
        "  mov.b16 t, 0x8000;  \n\t"
        "  and.b16 t,r,t;      \n\t"
        __SPEC_CASE(i, r, 0X32B3, 0x0800)
        __SPEC_CASE(i, r, 0X5CB0, 0x1000)
        __SPEC_CASE(i, r, 0XB2B3, 0x8800)
        __SPEC_CASE(i, r, 0XDCB0, 0x9000)
        "  or.b16  r,r,t;      \n\t"
        "  mov.b16 %0, r;      \n"
        "}\n" : "+h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(a)));
    return r;
}
__CUDA_FP16_DECL__ __half2 h2sin(const __half2 a) {
    __half l = __low2half(a);
    __half h = __high2half(a);
    __half2 r = __halves2half2(__hsin_internal(l), __hsin_internal(h));
    asm("{\n\t"
        "  .reg.b32 i,r,t;             \n\t"
        "  mov.b32 r, %0;              \n\t"
        "  mov.b32 i, %1;              \n\t"
        "  and.b32 t, r, 0x80008000;   \n\t"
        __SPEC_CASE2(i, r, 0X32B332B3, 0x08000800)
        __SPEC_CASE2(i, r, 0X5CB05CB0, 0x10001000)
        __SPEC_CASE2(i, r, 0XB2B3B2B3, 0x88008800)
        __SPEC_CASE2(i, r, 0XDCB0DCB0, 0x90009000)
        "  or.b32  r, r, t;            \n\t"
        "  mov.b32 %0, r;              \n"
        "}\n" : "+r"(__HALF2_TO_UI(r)) : "r"(__HALF2_TO_CUI(a)));
    return r;
}
__CUDA_FP16_DECL__ __half __hcos_internal(const __half a) {
    float f = __half2float(a);
    f = __float_simpl_cosf(f);
    return __float2half_rn(f);
}
__CUDA_FP16_DECL__ __half hcos(const __half a) {
    __half r = __hcos_internal(a);
    asm("{\n\t"
        "  .reg.b16 i,r;        \n\t"
        "  mov.b16 r, %0;       \n\t"
        "  mov.b16 i, %1;       \n\t"
        __SPEC_CASE(i, r, 0X2B7C, 0x1000)
        __SPEC_CASE(i, r, 0XAB7C, 0x1000)
        "  mov.b16 %0, r;       \n"
        "}\n" : "+h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(a)));
    return r;
}
__CUDA_FP16_DECL__ __half2 h2cos(const __half2 a) {
    __half l = __low2half(a);
    __half h = __high2half(a);
    __half2 r = __halves2half2(__hcos_internal(l), __hcos_internal(h));
    asm("{\n\t"
        "  .reg.b32 i,r;   \n\t"
        "  mov.b32 r, %0;  \n\t"
        "  mov.b32 i, %1;  \n\t"
        __SPEC_CASE2(i, r, 0X2B7C2B7C, 0x10001000)
        __SPEC_CASE2(i, r, 0XAB7CAB7C, 0x10001000)
        "  mov.b32 %0, r;  \n"
        "}\n" : "+r"(__HALF2_TO_UI(r)) : "r"(__HALF2_TO_CUI(a)));
    return r;
}
static __device__ __forceinline__ float __internal_trig_reduction_kernel(float a, int *quadrant)
{
    float j, t;
    int q;
    q = __float2int_rn(a * 0.636619772f);
    j = (float)q;
    t = __fmaf_rn(-j, 1.5707962512969971e+000f, a);
    t = __fmaf_rn(-j, 7.5497894158615964e-008f, t);
    *quadrant = q;
    return t;
}
static __device__ __forceinline__ float __internal_sin_cos_kernel(float x, int i)
{
    float x2, z;
    x2 = x*x;

    if (i & 1) {
        z = 2.44331571e-5f;
        z = __fmaf_rn(z, x2, -1.38873163e-3f);
    }
    else {
        z = -1.95152959e-4f;
        z = __fmaf_rn(z, x2, 8.33216087e-3f);
    }
    if (i & 1) {
        z = __fmaf_rn(z, x2, 4.16666457e-2f);
        z = __fmaf_rn(z, x2, -5.00000000e-1f);
    }
    else {
        z = __fmaf_rn(z, x2, -1.66666546e-1f);
        z = __fmaf_rn(z, x2, 0.0f);
    }
    x = __fmaf_rn(z, x, x);
    if (i & 1) x = __fmaf_rn(z, x2, 1.0f);
    if (i & 2) x = __fmaf_rn(x, -1.0f, 0.0f);
    return x;
}
static __device__ __forceinline__ float __float_simpl_sinf(float a)
{
    float z;
    int i;
    if (isinf(a)) {
        a = a * 0.0f;
    }
    a = __internal_trig_reduction_kernel(a, &i);
    z = __internal_sin_cos_kernel(a, i);
    return z;
}
static __device__ __forceinline__ float __float_simpl_cosf(float a)
{
    float z;
    int i;
    if (isinf(a)) {
        a = a * 0.0f;
    }
    a = __internal_trig_reduction_kernel(a, &i);
    i++;
    z = __internal_sin_cos_kernel(a, i);
    return z;
}
__CUDA_FP16_DECL__ __half hexp(const __half a) {
    __half val;
    asm("{.reg.b32         f, C;           \n"
        " .reg.b16         h,r;            \n"
        "  mov.b16         h,%1;           \n"
        "  cvt.f32.f16     f,h;            \n"
        "  mov.b32         C, 0x3fb8aa3b;  \n"
        "  mul.f32         f,f,C;          \n"
        "  ex2.approx.f32      f,f;        \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        __SPEC_CASE(h, r, 0X1F79, 0x9400)
        __SPEC_CASE(h, r, 0X25CF, 0x9400)
        __SPEC_CASE(h, r, 0XC13B, 0x0400)
        __SPEC_CASE(h, r, 0XC1EF, 0x0200)
        "  mov.b16         %0,r;           \n"
        "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2exp(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         h,r,fl,fu, C;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         h, %1;          \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  mov.b32         C, 0x3fb8aa3b;  \n"
        "  mul.f32         fl,fl,C;        \n"
        "  mul.f32         fu,fu,C;        \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        __SPEC_CASE2(h, r, 0X1F791F79, 0x94009400)
        __SPEC_CASE2(h, r, 0X25CF25CF, 0x94009400)
        __SPEC_CASE2(h, r, 0XC13BC13B, 0x04000400)
        __SPEC_CASE2(h, r, 0XC1EFC1EF, 0x02000200)
        "  mov.b32         %0, r;  \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ __half hexp2(const __half a) {
    __half val;
    asm("{.reg.b32         f, ULP;         \n"
        " .reg.b16         r;              \n"
        "  mov.b16         r,%1;           \n"
        "  cvt.f32.f16     f,r;            \n"
        "  ex2.approx.f32      f,f;        \n"
        "  mov.b32         ULP, 0x33800000;\n"
        "  fma.rn.f32      f,f,ULP,f;      \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        "  mov.b16         %0,r;           \n"
        "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2exp2(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         fl, fu, ULP;    \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  mov.b32         ULP, 0x33800000;\n"
        "  fma.rn.f32      fl,fl,ULP,fl;   \n"
        "  fma.rn.f32      fu,fu,ULP,fu;   \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         %0, {hl, hu};   \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ __half hexp10(const __half a) {
    __half val;
    asm("{.reg.b16         h,r;            \n"
        " .reg.b32         f, C;           \n"
        "  mov.b16         h, %1;          \n"
        "  cvt.f32.f16     f, h;           \n"
        "  mov.b32         C, 0x40549A78;  \n"
        "  mul.f32         f,f,C;          \n"
        "  ex2.approx.f32      f, f;       \n"
        "  cvt.rn.f16.f32      r, f;       \n"
        __SPEC_CASE(h, r, 0x34DE, 0x9800)
        __SPEC_CASE(h, r, 0x9766, 0x9000)
        __SPEC_CASE(h, r, 0x9972, 0x1000)
        __SPEC_CASE(h, r, 0xA5C4, 0x1000)
        __SPEC_CASE(h, r, 0xBF0A, 0x8100)
        "  mov.b16         %0, r;          \n"
        "}":"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2exp10(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         h,r,fl,fu, C;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         h, %1;          \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  mov.b32         C, 0x40549A78;  \n"
        "  mul.f32         fl,fl,C;        \n"
        "  mul.f32         fu,fu,C;        \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        __SPEC_CASE2(h, r, 0x34DE34DE, 0x98009800)
        __SPEC_CASE2(h, r, 0x97669766, 0x90009000)
        __SPEC_CASE2(h, r, 0x99729972, 0x10001000)
        __SPEC_CASE2(h, r, 0xA5C4A5C4, 0x10001000)
        __SPEC_CASE2(h, r, 0xBF0ABF0A, 0x81008100)
        "  mov.b32         %0, r;  \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ __half hlog2(const __half a) {
    __half val;
    asm("{.reg.b16         h, r;           \n"
        " .reg.b32         f;              \n"
        "  mov.b16         h, %1;          \n"
        "  cvt.f32.f16     f, h;           \n"
        "  lg2.approx.f32      f, f;       \n"
        "  cvt.rn.f16.f32      r, f;       \n"
        __SPEC_CASE(r, r, 0xA2E2, 0x8080)
        __SPEC_CASE(r, r, 0xBF46, 0x9400)
        "  mov.b16         %0, r;          \n"
        "}":"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2log2(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         fl, fu, r, p;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  lg2.approx.f32      fl, fl;     \n"
        "  lg2.approx.f32      fu, fu;     \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        __SPEC_CASE2(r, r, 0xA2E2A2E2, 0x80808080)
        __SPEC_CASE2(r, r, 0xBF46BF46, 0x94009400)
        "  mov.b32         %0, r;          \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ __half hlog(const __half a) {
    __half val;
    asm("{.reg.b32         f, C;           \n"
        " .reg.b16         r,h;            \n"
        "  mov.b16         h,%1;           \n"
        "  cvt.f32.f16     f,h;            \n"
        "  lg2.approx.f32      f,f;        \n"
        "  mov.b32         C, 0x3f317218;  \n"
        "  mul.f32         f,f,C;          \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        __SPEC_CASE(h, r, 0X160D, 0x9C00)
        __SPEC_CASE(h, r, 0X3BFE, 0x8010)
        __SPEC_CASE(h, r, 0X3C0B, 0x8080)
        __SPEC_CASE(h, r, 0X6051, 0x1C00)
        "  mov.b16         %0,r;           \n"
        "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2log(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;             \n"
        " .reg.b32         r, fl, fu, C, h;    \n"
        "  mov.b32         {hl, hu}, %1;       \n"
        "  mov.b32         h, %1;              \n"
        "  cvt.f32.f16     fl, hl;             \n"
        "  cvt.f32.f16     fu, hu;             \n"
        "  lg2.approx.f32      fl, fl;         \n"
        "  lg2.approx.f32      fu, fu;         \n"
        "  mov.b32         C, 0x3f317218;      \n"
        "  mul.f32         fl,fl,C;            \n"
        "  mul.f32         fu,fu,C;            \n"
        "  cvt.rn.f16.f32      hl, fl;         \n"
        "  cvt.rn.f16.f32      hu, fu;         \n"
        "  mov.b32         r, {hl, hu};        \n"
        __SPEC_CASE2(h, r, 0X160D160D, 0x9C009C00)
        __SPEC_CASE2(h, r, 0X3BFE3BFE, 0x80108010)
        __SPEC_CASE2(h, r, 0X3C0B3C0B, 0x80808080)
        __SPEC_CASE2(h, r, 0X60516051, 0x1C001C00)
        "  mov.b32         %0, r;              \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ __half hlog10(const __half a) {
    __half val;
    asm("{.reg.b16         h, r;           \n"
        " .reg.b32         f, C;           \n"
        "  mov.b16         h, %1;          \n"
        "  cvt.f32.f16     f, h;           \n"
        "  lg2.approx.f32      f, f;       \n"
        "  mov.b32         C, 0x3E9A209B;  \n"
        "  mul.f32         f,f,C;          \n"
        "  cvt.rn.f16.f32      r, f;       \n"
        __SPEC_CASE(h, r, 0x338F, 0x1000)
        __SPEC_CASE(h, r, 0x33F8, 0x9000)
        __SPEC_CASE(h, r, 0x57E1, 0x9800)
        __SPEC_CASE(h, r, 0x719D, 0x9C00)
        "  mov.b16         %0, r;          \n"
        "}":"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ __half2 h2log10(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;             \n"
        " .reg.b32         r, fl, fu, C, h;    \n"
        "  mov.b32         {hl, hu}, %1;       \n"
        "  mov.b32         h, %1;              \n"
        "  cvt.f32.f16     fl, hl;             \n"
        "  cvt.f32.f16     fu, hu;             \n"
        "  lg2.approx.f32      fl, fl;         \n"
        "  lg2.approx.f32      fu, fu;         \n"
        "  mov.b32         C, 0x3E9A209B;      \n"
        "  mul.f32         fl,fl,C;            \n"
        "  mul.f32         fu,fu,C;            \n"
        "  cvt.rn.f16.f32      hl, fl;         \n"
        "  cvt.rn.f16.f32      hu, fu;         \n"
        "  mov.b32         r, {hl, hu};        \n"
        __SPEC_CASE2(h, r, 0x338F338F, 0x10001000)
        __SPEC_CASE2(h, r, 0x33F833F8, 0x90009000)
        __SPEC_CASE2(h, r, 0x57E157E1, 0x98009800)
        __SPEC_CASE2(h, r, 0x719D719D, 0x9C009C00)
        "  mov.b32         %0, r;              \n"
        "}":"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)));
    return val;
}
#undef __SPEC_CASE2
#undef __SPEC_CASE
__CUDA_FP16_DECL__ __half2 h2rcp(const __half2 a) {
    __APPROX_FCAST2(rcp);
}
__CUDA_FP16_DECL__ __half hrcp(const __half a) {
    __APPROX_FCAST(rcp);
}
__CUDA_FP16_DECL__ __half2 h2rsqrt(const __half2 a) {
    __APPROX_FCAST2(rsqrt);
}
__CUDA_FP16_DECL__ __half hrsqrt(const __half a) {
    __APPROX_FCAST(rsqrt);
}
__CUDA_FP16_DECL__ __half2 h2sqrt(const __half2 a) {
    __APPROX_FCAST2(sqrt);
}
__CUDA_FP16_DECL__ __half hsqrt(const __half a) {
    __APPROX_FCAST(sqrt);
}
#undef __APPROX_FCAST
#undef __APPROX_FCAST2
__CUDA_FP16_DECL__ __half2 __hisnan2(const __half2 a)
{
    __half2 r;
    asm("{set.nan.f16x2.f16x2 %0,%1,%2;\n}"
        :"=r"(__HALF2_TO_UI(r)) : "r"(__HALF2_TO_CUI(a)), "r"(__HALF2_TO_CUI(a)));
    return r;
}
__CUDA_FP16_DECL__ bool __hisnan(const __half a)
{
    __half r;
    asm("{set.nan.f16.f16 %0,%1,%2;\n}"
        :"=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_CUS(a)), "h"(__HALF_TO_CUS(a)));
    if (__HALF_TO_CUS(r) == 0)
        return false;
    else return true;
}
__CUDA_FP16_DECL__ __half2 __hneg2(const __half2 a)
{
    __half2 zero = __float2half2_rn(0.0);
    return __hsub2(zero, a);
}
__CUDA_FP16_DECL__ __half __hneg(const __half a)
{
    __half zero;
    zero = __float2half(0.0);
    return __hsub(zero, a);
}
#endif /*__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/

#undef __CUDA_FP16_DECL__
#endif /* defined(__CUDACC__) */
#endif /* defined(__cplusplus) */

#undef __HALF_TO_US
#undef __HALF_TO_CUS
#undef __HALF2_TO_UI
#undef __HALF2_TO_CUI


/* Define first-class types "half" and "half2", unless user specifies otherwise via "#define CUDA_NO_HALF" */
/* C cannot ever have these types defined here, because __half and __half2 are C++ classes */
#if defined(__cplusplus) && !defined(CUDA_NO_HALF)
typedef __half half;
typedef __half2 half2;
#endif /* defined(__cplusplus) && !defined(CUDA_NO_HALF) */

#endif /* end of include guard: __CUDA_FP16_HPP__ */
