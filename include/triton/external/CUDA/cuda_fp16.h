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

/**
* \defgroup CUDA_MATH_INTRINSIC_HALF Half Precision Intrinsics
* This section describes half precision intrinsic functions that are
* only supported in device code.
*/

/**
* \defgroup CUDA_MATH__HALF_ARITHMETIC Half Arithmetic Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
*/

/**
* \defgroup CUDA_MATH__HALF2_ARITHMETIC Half2 Arithmetic Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
*/

/**
* \defgroup CUDA_MATH__HALF_COMPARISON Half Comparison Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
*/

/**
* \defgroup CUDA_MATH__HALF2_COMPARISON Half2 Comparison Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
*/

/**
* \defgroup CUDA_MATH__HALF_MISC Half Precision Conversion And Data Movement
* \ingroup CUDA_MATH_INTRINSIC_HALF
*/

/**
* \defgroup CUDA_MATH__HALF_FUNCTIONS Half Math Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
*/

/**
* \defgroup CUDA_MATH__HALF2_FUNCTIONS Half2 Math Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
*/

#ifndef __CUDA_FP16_H__
#define __CUDA_FP16_H__

#if defined(__cplusplus) && defined(__CUDACC__)

#if defined(__CUDACC_RTC__)
#define __CUDA_FP16_DECL__ __host__ __device__
#define __VECTOR_FUNCTIONS_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __CUDA_FP16_DECL__ static __device__ __inline__
#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__
#endif /* __CUDACC_RTC__ */

#define __CUDA_FP16_TYPES_EXIST__
/* Forward-declaration of structures defined in "cuda_fp16.hpp" */
struct __half;
struct __half2;

/* Vector type creation functions, match vector_functions.h */
__VECTOR_FUNCTIONS_DECL__ float2 make_float2(float x, float y);

#undef __VECTOR_FUNCTIONS_DECL__

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-to-nearest-even mode
* and returns \p half with converted value.
*
* Converts float number \p a to half precision in round-to-nearest-even mode.
*
* \return Returns \p half result with converted value.
*/
__CUDA_FP16_DECL__ __half __float2half(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-to-nearest-even mode
* and returns \p half with converted value.
*
* Converts float number \p a to half precision in round-to-nearest-even mode.
*
* \return Returns \p half result with converted value.
*/
__CUDA_FP16_DECL__ __half __float2half_rn(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-towards-zero mode
* and returns \p half with converted value.
*
* Converts float number \p a to half precision in round-towards-zero mode.
*
* \return Returns \p half result with converted value.
*/
__CUDA_FP16_DECL__ __half __float2half_rz(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-down mode
* and returns \p half with converted value.
*
* Converts float number \p a to half precision in round-down mode.
*
* \return Returns \p half result with converted value.
*/
__CUDA_FP16_DECL__ __half __float2half_rd(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-up mode
* and returns \p half with converted value.
*
* Converts float number \p a to half precision in round-up mode.
*
* \return Returns \p half result with converted value.
*/
__CUDA_FP16_DECL__ __half __float2half_ru(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts \p half number to float.
*
* Converts half number \p a to float.
*
* \return Returns float result with converted value.
*/
__CUDA_FP16_DECL__ float __half2float(const __half a);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-to-nearest-even mode.
*
* Convert the half-precision floating point value \p h to a signed integer in
* round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ int __half2int_rn(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-towards-zero mode.
*
* Convert the half-precision floating point value \p h to a signed integer in
* round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ int __half2int_rz(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-down mode.
*
* Convert the half-precision floating point value \p h to a signed integer in
* round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ int __half2int_rd(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-up mode.
*
* Convert the half-precision floating point value \p h to a signed integer in
* round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ int __half2int_ru(__half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-to-nearest-even mode.
*
* Convert the signed integer value \p i to a half-precision floating point
* value in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __int2half_rn(int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-towards-zero mode.
*
* Convert the signed integer value \p i to a half-precision floating point
* value in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __int2half_rz(int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-down mode.
*
* Convert the signed integer value \p i to a half-precision floating point
* value in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __int2half_rd(int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-up mode.
*
* Convert the signed integer value \p i to a half-precision floating point
* value in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __int2half_ru(int i);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-to-nearest-even
* mode.
*
* Convert the half-precision floating point value \p h to a signed short
* integer in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ short int __half2short_rn(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-towards-zero mode.
*
* Convert the half-precision floating point value \p h to a signed short
* integer in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ short int __half2short_rz(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-down mode.
*
* Convert the half-precision floating point value \p h to a signed short
* integer in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ short int __half2short_rd(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-up mode.
*
* Convert the half-precision floating point value \p h to a signed short
* integer in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ short int __half2short_ru(__half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-to-nearest-even
* mode.
*
* Convert the signed short integer value \p i to a half-precision floating
* point value in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __short2half_rn(short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-towards-zero mode.
*
* Convert the signed short integer value \p i to a half-precision floating
* point value in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __short2half_rz(short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-down mode.
*
* Convert the signed short integer value \p i to a half-precision floating
* point value in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __short2half_rd(short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-up mode.
*
* Convert the signed short integer value \p i to a half-precision floating
* point value in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __short2half_ru(short int i);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-to-nearest-even mode.
*
* Convert the half-precision floating point value \p h to an unsigned integer
* in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned int __half2uint_rn(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-towards-zero mode.
*
* Convert the half-precision floating point value \p h to an unsigned integer
* in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned int __half2uint_rz(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-down mode.
*
* Convert the half-precision floating point value \p h to an unsigned integer
* in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned int __half2uint_rd(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-up mode.
*
* Convert the half-precision floating point value \p h to an unsigned integer
* in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned int __half2uint_ru(__half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-to-nearest-even mode.
*
* Convert the unsigned integer value \p i to a half-precision floating point
* value in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __uint2half_rn(unsigned int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-towards-zero mode.
*
* Convert the unsigned integer value \p i to a half-precision floating point
* value in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __uint2half_rz(unsigned int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-down mode.
*
* Convert the unsigned integer value \p i to a half-precision floating point
* value in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __uint2half_rd(unsigned int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-up mode.
*
* Convert the unsigned integer value \p i to a half-precision floating point
* value in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __uint2half_ru(unsigned int i);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-to-nearest-even
* mode.
*
* Convert the half-precision floating point value \p h to an unsigned short
* integer in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rn(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-towards-zero
* mode.
*
* Convert the half-precision floating point value \p h to an unsigned short
* integer in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rz(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-down mode.
*
* Convert the half-precision floating point value \p h to an unsigned short
* integer in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rd(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-up mode.
*
* Convert the half-precision floating point value \p h to an unsigned short
* integer in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned short int __half2ushort_ru(__half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-to-nearest-even
* mode.
*
* Convert the unsigned short integer value \p i to a half-precision floating
* point value in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ushort2half_rn(unsigned short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-towards-zero
* mode.
*
* Convert the unsigned short integer value \p i to a half-precision floating
* point value in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ushort2half_rz(unsigned short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-down mode.
*
* Convert the unsigned short integer value \p i to a half-precision floating
* point value in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ushort2half_rd(unsigned short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-up mode.
*
* Convert the unsigned short integer value \p i to a half-precision floating
* point value in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ushort2half_ru(unsigned short int i);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-to-nearest-even
* mode.
*
* Convert the half-precision floating point value \p h to an unsigned 64-bit
* integer in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rn(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-towards-zero
* mode.
*
* Convert the half-precision floating point value \p h to an unsigned 64-bit
* integer in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rz(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-down mode.
*
* Convert the half-precision floating point value \p h to an unsigned 64-bit
* integer in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rd(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-up mode.
*
* Convert the half-precision floating point value \p h to an unsigned 64-bit
* integer in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ unsigned long long int __half2ull_ru(__half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-to-nearest-even
* mode.
*
* Convert the unsigned 64-bit integer value \p i to a half-precision floating
* point value in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ull2half_rn(unsigned long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-towards-zero
* mode.
*
* Convert the unsigned 64-bit integer value \p i to a half-precision floating
* point value in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ull2half_rz(unsigned long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-down mode.
*
* Convert the unsigned 64-bit integer value \p i to a half-precision floating
* point value in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ull2half_rd(unsigned long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-up mode.
*
* Convert the unsigned 64-bit integer value \p i to a half-precision floating
* point value in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ull2half_ru(unsigned long long int i);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-to-nearest-even
* mode.
*
* Convert the half-precision floating point value \p h to a signed 64-bit
* integer in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ long long int __half2ll_rn(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-towards-zero mode.
*
* Convert the half-precision floating point value \p h to a signed 64-bit
* integer in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ long long int __half2ll_rz(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-down mode.
*
* Convert the half-precision floating point value \p h to a signed 64-bit
* integer in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ long long int __half2ll_rd(__half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-up mode.
*
* Convert the half-precision floating point value \p h to a signed 64-bit
* integer in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ long long int __half2ll_ru(__half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-to-nearest-even
* mode.
*
* Convert the signed 64-bit integer value \p i to a half-precision floating
* point value in round-to-nearest-even mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ll2half_rn(long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-towards-zero mode.
*
* Convert the signed 64-bit integer value \p i to a half-precision floating
* point value in round-towards-zero mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ll2half_rz(long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-down mode.
*
* Convert the signed 64-bit integer value \p i to a half-precision floating
* point value in round-down mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ll2half_rd(long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-up mode.
*
* Convert the signed 64-bit integer value \p i to a half-precision floating
* point value in round-up mode.
*
* \return Returns converted value.
*/
__CUDA_FP16_DECL__ __half __ll2half_ru(long long int i);

/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Truncate input argument to the integral part.
*
* Round \p h to the nearest integer value that does not exceed \p h in
* magnitude.
*
* \return Returns truncated integer value.
*/
__CUDA_FP16_DECL__ __half htrunc(const __half h);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculate ceiling of the input argument.
*
* Compute the smallest integer value not less than \p h.
*
* \return Returns ceiling expressed as a half-precision floating point number.
*/
__CUDA_FP16_DECL__ __half hceil(const __half h);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculate the largest integer less than or equal to \p h.
*
* Calculate the largest integer value which is less than or equal to \p h.
*
* \return Returns floor expressed as half-precision floating point number.
*/
__CUDA_FP16_DECL__ __half hfloor(const __half h);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Round input to nearest integer value in half-precision floating point
* number.
*
* Round \p h to the nearest integer value in half-precision floating point
* format, with halfway cases rounded to the nearest even integer value.
*
* \return Returns rounded integer value expressed as half-precision floating
* point number.
*/
__CUDA_FP16_DECL__ __half hrint(const __half h);

/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Truncate \p half2 vector input argument to the integral part.
*
* Round each component of vector \p h to the nearest integer value that does
* not exceed \p h in magnitude.
*
* \return Returns \p half2 vector truncated integer value.
*/
__CUDA_FP16_DECL__ __half2 h2trunc(const __half2 h);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculate \p half2 vector ceiling of the input argument.
*
* For each component of vector \p h compute the smallest integer value not less
* than \p h.
*
* \return Returns \p half2 vector ceiling expressed as a pair of half-precision
* floating point numbers.
*/
__CUDA_FP16_DECL__ __half2 h2ceil(const __half2 h);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculate the largest integer less than or equal to \p h.
*
* For each component of vector \p h calculate the largest integer value which
* is less than or equal to \p h.
*
* \return Returns \p half2 vector floor expressed as a pair of half-precision
* floating point number.
*/
__CUDA_FP16_DECL__ __half2 h2floor(const __half2 h);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Round input to nearest integer value in half-precision floating point
* number.
*
* Round each component of \p half2 vector \p h to the nearest integer value in
* half-precision floating point format, with halfway cases rounded to the
* nearest even integer value.
*
* \return Returns \p half2 vector of rounded integer values expressed as
* half-precision floating point numbers.
*/
__CUDA_FP16_DECL__ __half2 h2rint(const __half2 h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts input to half precision in round-to-nearest-even mode and
* populates both halves of \p half2 with converted value.
*
* Converts input \p a to half precision in round-to-nearest-even mode and
* populates both halves of \p half2 with converted value.
*
* \return Returns \p half2 with both halves equal to the converted half
* precision number.
*/
__CUDA_FP16_DECL__ __half2 __float2half2_rn(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts both input floats to half precision in round-to-nearest-even
* mode and returns \p half2 with converted values.
*
* Converts both input floats to half precision in round-to-nearest-even mode
* and combines the results into one \p half2 number. Low 16 bits of the return
* value correspond to the input \p a, high 16 bits correspond to the input \p
* b.
*
* \return Returns \p half2 which has corresponding halves equal to the
* converted input floats.
*/
__CUDA_FP16_DECL__ __half2 __floats2half2_rn(const float a, const float b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts both components of float2 number to half precision in
* round-to-nearest-even mode and returns \p half2 with converted values.
*
* Converts both components of float2 to half precision in round-to-nearest
* mode and combines the results into one \p half2 number. Low 16 bits of the
* return value correspond to \p a.x and high 16 bits of the return value
* correspond to \p a.y.
*
* \return Returns \p half2 which has corresponding halves equal to the
* converted float2 components.
*/
__CUDA_FP16_DECL__ __half2 __float22half2_rn(const float2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts both halves of \p half2 to float2 and returns the result.
*
* Converts both halves of \p half2 input \p a to float2 and returns the
* result.
*
* \return Returns converted float2.
*/
__CUDA_FP16_DECL__ float2 __half22float2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts low 16 bits of \p half2 to float and returns the result
*
* Converts low 16 bits of \p half2 input \p a to 32 bit floating point number
* and returns the result.
*
* \return Returns low 16 bits of \p a converted to float.
*/
__CUDA_FP16_DECL__ float __low2float(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Returns \p half2 with both halves equal to the input value.
*
* Returns \p half2 number with both halves equal to the input \p a \p half
* number.
*
* \return Returns \p half2 with both halves equal to the input \p a.
*/
__CUDA_FP16_DECL__ __half2 __half2half2(const __half a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts high 16 bits of \p half2 to float and returns the result
*
* Converts high 16 bits of \p half2 input \p a to 32 bit floating point number
* and returns the result.
*
* \return Returns high 16 bits of \p a converted to float.
*/
__CUDA_FP16_DECL__ float __high2float(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Swaps both halves of the \p half2 input.
*
* Swaps both halves of the \p half2 input and returns a new \p half2 number
* with swapped halves.
*
* \return Returns \p half2 with halves swapped.
*/
__CUDA_FP16_DECL__ __half2 __lowhigh2highlow(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts low 16 bits from each of the two \p half2 inputs and combines
* into one \p half2 number.
*
* Extracts low 16 bits from each of the two \p half2 inputs and combines into
* one \p half2 number. Low 16 bits from input \p a is stored in low 16 bits of
* the return value, low 16 bits from input \p b is stored in high 16 bits of
* the return value.
*
* \return Returns \p half2 which contains low 16 bits from \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __lows2half2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts high 16 bits from each of the two \p half2 inputs and
* combines into one \p half2 number.
*
* Extracts high 16 bits from each of the two \p half2 inputs and combines into
* one \p half2 number. High 16 bits from input \p a is stored in low 16 bits of
* the return value, high 16 bits from input \p b is stored in high 16 bits of
* the return value.
*
* \return Returns \p half2 which contains high 16 bits from \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __highs2half2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Returns high 16 bits of \p half2 input.
*
* Returns high 16 bits of \p half2 input \p a.
*
* \return Returns \p half which contains high 16 bits of the input.
*/
__CUDA_FP16_DECL__ __half __high2half(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Returns low 16 bits of \p half2 input.
*
* Returns low 16 bits of \p half2 input \p a.
*
* \return Returns \p half which contains low 16 bits of the input.
*/
__CUDA_FP16_DECL__ __half __low2half(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Checks if the input \p half number is infinite.
*
* Checks if the input \p half number \p a is infinite.
*
* \return Returns -1 iff \p a is equal to negative infinity, 1 iff \p a is
* equal to positive infinity and 0 otherwise.
*/
__CUDA_FP16_DECL__ int __hisinf(const __half a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Combines two \p half numbers into one \p half2 number.
*
* Combines two input \p half number \p a and \p b into one \p half2 number.
* Input \p a is stored in low 16 bits of the return value, input \p b is stored
* in high 16 bits of the return value.
*
* \return Returns \p half2 number which has one half equal to \p a and the
* other to \p b.
*/
__CUDA_FP16_DECL__ __half2 __halves2half2(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts low 16 bits from \p half2 input.
*
* Extracts low 16 bits from \p half2 input \p a and returns a new \p half2
* number which has both halves equal to the extracted bits.
*
* \return Returns \p half2 with both halves equal to low 16 bits from the
* input.
*/
__CUDA_FP16_DECL__ __half2 __low2half2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts high 16 bits from \p half2 input.
*
* Extracts high 16 bits from \p half2 input \p a and returns a new \p half2
* number which has both halves equal to the extracted bits.
*
* \return Returns \p half2 with both halves equal to high 16 bits from the
* input.
*/
__CUDA_FP16_DECL__ __half2 __high2half2(const __half2 a);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in a \p half as a signed short integer.
*
* Reinterprets the bits in the half-precision floating point value \p h
* as a signed short integer.
*
* \return Returns reinterpreted value.
*/
__CUDA_FP16_DECL__ short int __half_as_short(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in a \p half as an unsigned short integer.
*
* Reinterprets the bits in the half-precision floating point value \p h
* as an unsigned short integer.
*
* \return Returns reinterpreted value.
*/
__CUDA_FP16_DECL__ unsigned short int __half_as_ushort(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in a signed short integer as a \p half.
*
* Reinterprets the bits in the signed short integer value \p i as a
* half-precision floating point value.
*
* \return Returns reinterpreted value.
*/
__CUDA_FP16_DECL__ __half __short_as_half(const short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in an unsigned short integer as a \p half.
*
* Reinterprets the bits in the unsigned short integer value \p i as a
* half-precision floating point value.
*
* \return Returns reinterpreted value.
*/
__CUDA_FP16_DECL__ __half __ushort_as_half(const unsigned short int i);

#if __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)
#if !defined warpSize && !defined __local_warpSize
#define warpSize    32
#define __local_warpSize
#endif

#if defined(_WIN32)
# define __DEPRECATED__(msg) __declspec(deprecated(msg))
#elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
# define __DEPRECATED__(msg) __attribute__((deprecated))
#else
# define __DEPRECATED__(msg) __attribute__((deprecated(msg)))
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define __WSB_DEPRECATION_MESSAGE(x) #x"() is not valid on compute_70 and above, and should be replaced with "#x"_sync()." \
    "To continue using "#x"(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."
#else
#define __WSB_DEPRECATION_MESSAGE(x) #x"() is deprecated in favor of "#x"_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."
#endif

__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl)) __half2 __shfl(__half2 var, int delta, int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_up)) __half2 __shfl_up(__half2 var, unsigned int delta, int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_down))__half2 __shfl_down(__half2 var, unsigned int delta, int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_xor)) __half2 __shfl_xor(__half2 var, int delta, int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl)) __half __shfl(__half var, int delta, int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_up)) __half __shfl_up(__half var, unsigned int delta, int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_down)) __half __shfl_down(__half var, unsigned int delta, int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_xor)) __half __shfl_xor(__half var, int delta, int width = warpSize);

__CUDA_FP16_DECL__ __half2 __shfl_sync(unsigned mask, __half2 var, int delta, int width = warpSize);
__CUDA_FP16_DECL__ __half2 __shfl_up_sync(unsigned mask, __half2 var, unsigned int delta, int width = warpSize);
__CUDA_FP16_DECL__ __half2 __shfl_down_sync(unsigned mask, __half2 var, unsigned int delta, int width = warpSize);
__CUDA_FP16_DECL__ __half2 __shfl_xor_sync(unsigned mask, __half2 var, int delta, int width = warpSize);
__CUDA_FP16_DECL__ __half __shfl_sync(unsigned mask, __half var, int delta, int width = warpSize);
__CUDA_FP16_DECL__ __half __shfl_up_sync(unsigned mask, __half var, unsigned int delta, int width = warpSize);
__CUDA_FP16_DECL__ __half __shfl_down_sync(unsigned mask, __half var, unsigned int delta, int width = warpSize);
__CUDA_FP16_DECL__ __half __shfl_xor_sync(unsigned mask, __half var, int delta, int width = warpSize);

#if defined(__local_warpSize)
#undef warpSize
#undef __local_warpSize
#endif
#endif /*__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__) */

#if defined(__cplusplus) && ( __CUDA_ARCH__ >=320 || !defined(__CUDA_ARCH__) )
__CUDA_FP16_DECL__ __half2 __ldg(const  __half2 *ptr);
__CUDA_FP16_DECL__ __half __ldg(const __half *ptr);
__CUDA_FP16_DECL__ __half2 __ldcg(const  __half2 *ptr);
__CUDA_FP16_DECL__ __half __ldcg(const __half *ptr);
__CUDA_FP16_DECL__ __half2 __ldca(const  __half2 *ptr);
__CUDA_FP16_DECL__ __half __ldca(const __half *ptr);
__CUDA_FP16_DECL__ __half2 __ldcs(const  __half2 *ptr);
__CUDA_FP16_DECL__ __half __ldcs(const __half *ptr);
#endif /*defined(__cplusplus) && ( __CUDA_ARCH__ >=320 || !defined(__CUDA_ARCH__) )*/

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs half2 vector if-equal comparison.
*
* Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
*
* \return Returns the \p half2 vector result of if-equal comparison of vectors
* \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __heq2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector not-equal comparison.
*
* Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
*
* \return Returns the \p half2 vector result of not-equal comparison of vectors
* \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hne2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-equal comparison.
*
* Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
*
* \return Returns the \p half2 vector result of less-equal comparison of
* vectors \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hle2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-equal comparison.
*
* Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
*
* \return Returns the \p half2 vector result of greater-equal comparison of
* vectors \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hge2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-than comparison.
*
* Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
*
* \return Returns the \p half2 vector result of less-than comparison of vectors
* \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hlt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-than comparison.
*
* Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
*
* \return Returns the half2 vector result of greater-than comparison of vectors
* \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hgt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered if-equal comparison.
*
* Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
*
* \return Returns the \p half2 vector result of unordered if-equal comparison
* of vectors \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hequ2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered not-equal comparison.
*
* Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
*
* \return Returns the \p half2 vector result of unordered not-equal comparison
* of vectors \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hneu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-equal comparison.
*
* Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
*
* \return Returns the \p half2 vector result of unordered less-equal comparison
* of vectors \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hleu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-equal comparison.
*
* Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
*
* \return Returns the \p half2 vector result of unordered greater-equal
* comparison of vectors \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hgeu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-than comparison.
*
* Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
*
* \return Returns the \p half2 vector result of unordered less-than comparison
* of vectors \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hltu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-than comparison.
*
* Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
*
* \return Returns the \p half2 vector result of unordered greater-than
* comparison of vectors \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hgtu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Determine whether \p half2 argument is a NaN.
*
* Determine whether each half of input \p half2 number \p a is a NaN.
*
* \return Returns \p half2 which has the corresponding \p half results set to
* 1.0 for true, or 0.0 for false.
*/
__CUDA_FP16_DECL__ __half2 __hisnan2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector addition in round-to-nearest-even mode.
*
* Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest
* mode.
*
* \return Returns the \p half2 vector result of adding vectors \p a and \p b.
*/
__CUDA_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector subtraction in round-to-nearest-even mode.
*
* Subtracts \p half2 input vector \p b from input vector \p a in
* round-to-nearest-even mode.
*
* \return Returns the \p half2 vector result of subtraction vector \p b from \p
* a.
*/
__CUDA_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector multiplication in round-to-nearest-even mode.
*
* Performs \p half2 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode.
*
* \return Returns the \p half2 vector result of multiplying vectors \p a and \p
* b.
*/
__CUDA_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half2 vector division in round-to-nearest-even mode.
*
* Divides \p half2 input vector \p a by input vector \p b in round-to-nearest
* mode.
*
* \return Returns the \p half2 vector result of division \p a by \p b.
*/
__CUDA_FP16_DECL__ __half2 __h2div(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector addition in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest
* mode, and clamps the results to range [0.0, 1.0]. NaN results are flushed to
* +0.0.
*
* \return Returns the \p half2 vector result of adding vectors \p a and \p b
* with saturation.
*/
__CUDA_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector subtraction in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* Subtracts \p half2 input vector \p b from input vector \p a in
* round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
* results are flushed to +0.0.
*
* \return Returns the \p half2 vector result of subtraction vector \p b from \p
* a with saturation.
*/
__CUDA_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector multiplication in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* Performs \p half2 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
* results are flushed to +0.0.
*
* \return Returns the \p half2 vector result of multiplying vectors \p a and \p
* b with saturation.
*/
__CUDA_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
* mode.
*
* Performs \p half2 vector multiply on inputs \p a and \p b,
* then performs a \p half2 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
*
* \return Returns the \p half2 vector result of the fused multiply-add
* operation on vectors \p a, \p b, and \p c.
*/
__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
* mode, with saturation to [0.0, 1.0].
*
* Performs \p half2 vector multiply on inputs \p a and \p b,
* then performs a \p half2 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode, and clamps the
* results to range [0.0, 1.0]. NaN results are flushed to +0.0.
*
* \return Returns the \p half2 vector result of the fused multiply-add
* operation on vectors \p a, \p b, and \p c with saturation.
*/
__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Negates both halves of the input \p half2 number and returns the
* result.
*
* Negates both halves of the input \p half2 number \p a and returns the result.
*
* \return Returns \p half2 number with both halves negated.
*/
__CUDA_FP16_DECL__ __half2 __hneg2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half addition in round-to-nearest-even mode.
*
* Performs \p half addition of inputs \p a and \p b, in round-to-nearest-even
* mode.
*
* \return Returns the \p half result of adding \p a and \p b.
*/
__CUDA_FP16_DECL__ __half __hadd(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half subtraction in round-to-nearest-even mode.
*
* Subtracts \p half input \p b from input \p a in round-to-nearest
* mode.
*
* \return Returns the \p half result of subtraction \p b from \p a.
*/
__CUDA_FP16_DECL__ __half __hsub(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half multiplication in round-to-nearest-even mode.
*
* Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest
* mode.
*
* \return Returns the \p half result of multiplying \p a and \p b.
*/
__CUDA_FP16_DECL__ __half __hmul(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half division in round-to-nearest-even mode.
*
* Divides \p half input \p a by input \p b in round-to-nearest
* mode.
*
* \return Returns the \p half result of division \p a by \p b.
*/
__CUDA_FP16_DECL__  __half __hdiv(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half addition in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* Performs \p half add of inputs \p a and \p b, in round-to-nearest-even mode,
* and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
*
* \return Returns the \p half result of adding \p a and \p b with saturation.
*/
__CUDA_FP16_DECL__ __half __hadd_sat(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half subtraction in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* Subtracts \p half input \p b from input \p a in round-to-nearest
* mode,
* and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
*
* \return Returns the \p half result of subtraction \p b from \p a
* with saturation.
*/
__CUDA_FP16_DECL__ __half __hsub_sat(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half multiplication in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest
* mode, and clamps the result to range [0.0, 1.0]. NaN results are flushed to
* +0.0.
*
* \return Returns the \p half result of multiplying \p a and \p b with
* saturation.
*/
__CUDA_FP16_DECL__ __half __hmul_sat(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half fused multiply-add in round-to-nearest-even mode.
*
* Performs \p half multiply on inputs \p a and \p b,
* then performs a \p half add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
*
* \return Returns the \p half result of the fused multiply-add operation on \p
* a, \p b, and \p c.
*/
__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half fused multiply-add in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* Performs \p half multiply on inputs \p a and \p b,
* then performs a \p half add of the result with \p c,
* rounding the result once in round-to-nearest-even mode, and clamps the result
* to range [0.0, 1.0]. NaN results are flushed to +0.0.
*
* \return Returns the \p half result of the fused multiply-add operation on \p
* a, \p b, and \p c with saturation.
*/
__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Negates input \p half number and returns the result.
*
* Negates input \p half number and returns the result.
*
* \return Returns negated \p half input \p a.
*/
__CUDA_FP16_DECL__ __half __hneg(const __half a);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector if-equal comparison, and returns boolean true
* iff both \p half results are true, boolean false otherwise.
*
* Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half if-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
*
* \return Returns boolean true if both \p half results of if-equal comparison
* of vectors \p a and \p b are true, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hbeq2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector not-equal comparison, and returns boolean
* true iff both \p half results are true, boolean false otherwise.
*
* Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half not-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
*
* \return Returns boolean true if both \p half results of not-equal comparison
* of vectors \p a and \p b are true, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hbne2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-equal comparison, and returns boolean
* true iff both \p half results are true, boolean false otherwise.
*
* Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
*
* \return Returns boolean true if both \p half results of less-equal comparison
* of vectors \p a and \p b are true, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hble2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-equal comparison, and returns boolean
* true iff both \p half results are true, boolean false otherwise.
*
* Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
*
* \return Returns boolean true if both \p half results of greater-equal
* comparison of vectors \p a and \p b are true, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hbge2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-than comparison, and returns boolean
* true iff both \p half results are true, boolean false otherwise.
*
* Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
*
* \return Returns boolean true if both \p half results of less-than comparison
* of vectors \p a and \p b are true, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hblt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-than comparison, and returns boolean
* true iff both \p half results are true, boolean false otherwise.
*
* Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
*
* \return Returns boolean true if both \p half results of greater-than
* comparison of vectors \p a and \p b are true, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hbgt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered if-equal comparison, and returns
* boolean true iff both \p half results are true, boolean false otherwise.
*
* Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half if-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
*
* \return Returns boolean true if both \p half results of unordered if-equal
* comparison of vectors \p a and \p b are true, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hbequ2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered not-equal comparison, and returns
* boolean true iff both \p half results are true, boolean false otherwise.
*
* Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half not-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
*
* \return Returns boolean true if both \p half results of unordered not-equal
* comparison of vectors \p a and \p b are true, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hbneu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-equal comparison, and returns
* boolean true iff both \p half results are true, boolean false otherwise.
*
* Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
*
* \return Returns boolean true if both \p half results of unordered less-equal
* comparison of vectors \p a and \p b are true, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hbleu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-equal comparison, and
* returns boolean true iff both \p half results are true, boolean false
* otherwise.
*
* Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
*
* \return Returns boolean true if both \p half results of unordered
* greater-equal comparison of vectors \p a and \p b are true, boolean false
* otherwise.
*/
__CUDA_FP16_DECL__ bool __hbgeu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-than comparison, and returns
* boolean true iff both \p half results are true, boolean false otherwise.
*
* Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
*
* \return Returns boolean true if both \p half results of unordered less-than
* comparison of vectors \p a and \p b are true, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hbltu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-than comparison, and
* returns boolean true iff both \p half results are true, boolean false
* otherwise.
*
* Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
*
* \return Returns boolean true if both \p half results of unordered
* greater-than comparison of vectors \p a and \p b are true, boolean false
* otherwise.
*/
__CUDA_FP16_DECL__ bool __hbgtu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half if-equal comparison.
*
* Performs \p half if-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
*
* \return Returns boolean result of if-equal comparison of \p a and \p b.
*/
__CUDA_FP16_DECL__ bool __heq(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half not-equal comparison.
*
* Performs \p half not-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
*
* \return Returns boolean result of not-equal comparison of \p a and \p b.
*/
__CUDA_FP16_DECL__ bool __hne(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half less-equal comparison.
*
* Performs \p half less-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
*
* \return Returns boolean result of less-equal comparison of \p a and \p b.
*/
__CUDA_FP16_DECL__ bool __hle(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half greater-equal comparison.
*
* Performs \p half greater-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
*
* \return Returns boolean result of greater-equal comparison of \p a and \p b.
*/
__CUDA_FP16_DECL__ bool __hge(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half less-than comparison.
*
* Performs \p half less-than comparison of inputs \p a and \p b.
* NaN inputs generate false results.
*
* \return Returns boolean result of less-than comparison of \p a and \p b.
*/
__CUDA_FP16_DECL__ bool __hlt(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half greater-than comparison.
*
* Performs \p half greater-than comparison of inputs \p a and \p b.
* NaN inputs generate false results.
*
* \return Returns boolean result of greater-than comparison of \p a and \p b.
*/
__CUDA_FP16_DECL__ bool __hgt(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered if-equal comparison.
*
* Performs \p half if-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
*
* \return Returns boolean result of unordered if-equal comparison of \p a and
* \p b.
*/
__CUDA_FP16_DECL__ bool __hequ(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered not-equal comparison.
*
* Performs \p half not-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
*
* \return Returns boolean result of unordered not-equal comparison of \p a and
* \p b.
*/
__CUDA_FP16_DECL__ bool __hneu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered less-equal comparison.
*
* Performs \p half less-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
*
* \return Returns boolean result of unordered less-equal comparison of \p a and
* \p b.
*/
__CUDA_FP16_DECL__ bool __hleu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered greater-equal comparison.
*
* Performs \p half greater-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
*
* \return Returns boolean result of unordered greater-equal comparison of \p a
* and \p b.
*/
__CUDA_FP16_DECL__ bool __hgeu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered less-than comparison.
*
* Performs \p half less-than comparison of inputs \p a and \p b.
* NaN inputs generate true results.
*
* \return Returns boolean result of unordered less-than comparison of \p a and
* \p b.
*/
__CUDA_FP16_DECL__ bool __hltu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered greater-than comparison.
*
* Performs \p half greater-than comparison of inputs \p a and \p b.
* NaN inputs generate true results.
*
* \return Returns boolean result of unordered greater-than comparison of \p a
* and \p b.
*/
__CUDA_FP16_DECL__ bool __hgtu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Determine whether \p half argument is a NaN.
*
* Determine whether \p half value \p a is a NaN.
*
* \return Returns boolean true iff argument is a NaN, boolean false otherwise.
*/
__CUDA_FP16_DECL__ bool __hisnan(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half square root in round-to-nearest-even mode.
*
* Calculates \p half square root of input \p a in round-to-nearest-even mode.
*
* \return Returns \p half square root of \p a.
*/
__CUDA_FP16_DECL__ __half hsqrt(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half reciprocal square root in round-to-nearest-even
* mode.
*
* Calculates \p half reciprocal square root of input \p a in round-to-nearest
* mode.
*
* \return Returns \p half reciprocal square root of \p a.
*/
__CUDA_FP16_DECL__ __half hrsqrt(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half reciprocal in round-to-nearest-even mode.
*
* Calculates \p half reciprocal of input \p a in round-to-nearest-even mode.
*
* \return Returns \p half reciprocal of \p a.
*/
__CUDA_FP16_DECL__ __half hrcp(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half natural logarithm in round-to-nearest-even mode.
*
* Calculates \p half natural logarithm of input \p a in round-to-nearest-even
* mode.
*
* \return Returns \p half natural logarithm of \p a.
*/
__CUDA_FP16_DECL__ __half hlog(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half binary logarithm in round-to-nearest-even mode.
*
* Calculates \p half binary logarithm of input \p a in round-to-nearest-even
* mode.
*
* \return Returns \p half binary logarithm of \p a.
*/
__CUDA_FP16_DECL__ __half hlog2(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half decimal logarithm in round-to-nearest-even mode.
*
* Calculates \p half decimal logarithm of input \p a in round-to-nearest-even
* mode.
*
* \return Returns \p half decimal logarithm of \p a.
*/
__CUDA_FP16_DECL__ __half hlog10(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half natural exponential function in round-to-nearest
* mode.
*
* Calculates \p half natural exponential function of input \p a in
* round-to-nearest-even mode.
*
* \return Returns \p half natural exponential function of \p a.
*/
__CUDA_FP16_DECL__ __half hexp(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half binary exponential function in round-to-nearest
* mode.
*
* Calculates \p half binary exponential function of input \p a in
* round-to-nearest-even mode.
*
* \return Returns \p half binary exponential function of \p a.
*/
__CUDA_FP16_DECL__ __half hexp2(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half decimal exponential function in round-to-nearest
* mode.
*
* Calculates \p half decimal exponential function of input \p a in
* round-to-nearest-even mode.
*
* \return Returns \p half decimal exponential function of \p a.
*/
__CUDA_FP16_DECL__ __half hexp10(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half cosine in round-to-nearest-even mode.
*
* Calculates \p half cosine of input \p a in round-to-nearest-even mode.
*
* \return Returns \p half cosine of \p a.
*/
__CUDA_FP16_DECL__ __half hcos(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half sine in round-to-nearest-even mode.
*
* Calculates \p half sine of input \p a in round-to-nearest-even mode.
*
* \return Returns \p half sine of \p a.
*/
__CUDA_FP16_DECL__ __half hsin(const __half a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector square root in round-to-nearest-even mode.
*
* Calculates \p half2 square root of input vector \p a in round-to-nearest
* mode.
*
* \return Returns \p half2 square root of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2sqrt(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector reciprocal square root in round-to-nearest
* mode.
*
* Calculates \p half2 reciprocal square root of input vector \p a in
* round-to-nearest-even mode.
*
* \return Returns \p half2 reciprocal square root of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2rsqrt(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector reciprocal in round-to-nearest-even mode.
*
* Calculates \p half2 reciprocal of input vector \p a in round-to-nearest-even
* mode.
*
* \return Returns \p half2 reciprocal of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2rcp(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector natural logarithm in round-to-nearest-even
* mode.
*
* Calculates \p half2 natural logarithm of input vector \p a in
* round-to-nearest-even mode.
*
* \return Returns \p half2 natural logarithm of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2log(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector binary logarithm in round-to-nearest-even
* mode.
*
* Calculates \p half2 binary logarithm of input vector \p a in round-to-nearest
* mode.
*
* \return Returns \p half2 binary logarithm of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2log2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector decimal logarithm in round-to-nearest-even
* mode.
*
* Calculates \p half2 decimal logarithm of input vector \p a in
* round-to-nearest-even mode.
*
* \return Returns \p half2 decimal logarithm of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2log10(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector exponential function in round-to-nearest
* mode.
*
* Calculates \p half2 exponential function of input vector \p a in
* round-to-nearest-even mode.
*
* \return Returns \p half2 exponential function of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2exp(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector binary exponential function in
* round-to-nearest-even mode.
*
* Calculates \p half2 binary exponential function of input vector \p a in
* round-to-nearest-even mode.
*
* \return Returns \p half2 binary exponential function of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2exp2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector decimal exponential function in
* round-to-nearest-even mode.
*
* Calculates \p half2 decimal exponential function of input vector \p a in
* round-to-nearest-even mode.
*
* \return Returns \p half2 decimal exponential function of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2exp10(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector cosine in round-to-nearest-even mode.
*
* Calculates \p half2 cosine of input vector \p a in round-to-nearest-even
* mode.
*
* \return Returns \p half2 cosine of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2cos(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector sine in round-to-nearest-even mode.
*
* Calculates \p half2 sine of input vector \p a in round-to-nearest-even mode.
*
* \return Returns \p half2 sine of vector \p a.
*/
__CUDA_FP16_DECL__ __half2 h2sin(const __half2 a);

#endif /*if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/

#undef __CUDA_FP16_DECL__

#endif /* defined(__cplusplus) && defined(__CUDACC__) */

/* Note the .hpp file is included even for host-side compilation, to capture the "half" & "half2" definitions */
#include "cuda_fp16.hpp"

#endif /* end of include guard: __CUDA_FP16_H__ */
