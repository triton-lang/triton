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

#ifndef CUDA_FP16_H_JNESTUG4
#define CUDA_FP16_H_JNESTUG4

typedef struct __align__(2) {
   unsigned short x;
} __half;

typedef struct __align__(4) {
   unsigned int x;
} __half2;

#ifndef CUDA_NO_HALF
typedef __half half;
typedef __half2 half2;
#endif /*CUDA_NO_HALF*/

#if defined(__CUDACC__)

#if !defined(__cplusplus)
#include <stdbool.h>
#endif /*!defined(__cplusplus)*/

#if defined(__CUDACC_RTC__)
#define __CUDA_FP16_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __CUDA_FP16_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts float number to half precision in round-to-nearest mode and
 * returns \p half with converted value.
 *
 * Converts float number \p a to half precision in round-to-nearest mode.
 *
 * \return Returns \p half result with converted value.
 */
__CUDA_FP16_DECL__ __half __float2half(const float a);
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
 * \brief Converts input to half precision in round-to-nearest mode and
 * populates both halves of \p half2 with converted value.
 *
 * Converts input \p a to half precision in round-to-nearest mode and populates
 * both halves of \p half2 with converted value.
 *
 * \return Returns \p half2 with both halves equal to the converted half
 * precision number.
 */
__CUDA_FP16_DECL__ __half2 __float2half2_rn(const float a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts both input floats to half precision in round-to-nearest mode
 * and returns \p half2 with converted values.
 *
 * Converts both input floats to half precision in round-to-nearest mode and
 * combines the results into one \p half2 number. Low 16 bits of the return
 * value correspond to the input \p a, high 16 bits correspond to the input \p
 * b.
 *
 * \return Returns \p half2 which has corresponding halves equal to the converted
 * input floats.
 */
__CUDA_FP16_DECL__ __half2 __floats2half2_rn(const float a, const float b);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts both components of float2 number to half precision in
 * round-to-nearest mode and returns \p half2 with converted values.
 *
 * Converts both components of float2 to half precision in round-to-nearest mode
 * and combines the results into one \p half2 number. Low 16 bits of the return
 * value correspond to \p a.x and high 16 bits of the return value correspond to
 * \p a.y.
 *
 * \return Returns \p half2 which has corresponding halves equal to the converted
 * float2 components.
 */
__CUDA_FP16_DECL__ __half2 __float22half2_rn(const float2 a);
/**
 * \ingroup CUDA_MATH__HALF_MISC
 * \brief Converts both halves of \p half2 to float2 and returns the result.
 *
 * Converts both halves of \p half2 input \p a to float2 and returns the result.
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
 * \brief Extracts high 16 bits from each of the two \p half2 inputs and combines
 * into one \p half2 number.
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
 * \return Returns \p half2 with both halves equal to low 16 bits from the input.
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

#if __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)
__CUDA_FP16_DECL__ __half2 __shfl(__half2 var, int delta, int width);
__CUDA_FP16_DECL__ __half2 __shfl_up(__half2 var, unsigned int delta, int width);
__CUDA_FP16_DECL__ __half2 __shfl_down(__half2 var, unsigned int delta, int width);
__CUDA_FP16_DECL__ __half2 __shfl_xor(__half2 var, int delta, int width);
#endif /*__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__) */

#if defined(__cplusplus) && ( __CUDA_ARCH__ >=320 || !defined(__CUDA_ARCH__) )
__CUDA_FP16_DECL__ __half2 __ldg(const  __half2 *ptr);
__CUDA_FP16_DECL__ __half __ldg(const __half *ptr);
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
 * \brief Performs \p half2 vector addition in round-to-nearest mode.
 *
 * Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest
 * mode.
 *
 * \return Returns the \p half2 vector result of adding vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector subtraction in round-to-nearest mode.
 *
 * Subtracts \p half2 input vector \p b from input vector \p a in round-to-nearest
 * mode.
 *
 * \return Returns the \p half2 vector result of subtraction vector \p b from \p
 * a.
 */
__CUDA_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector multiplication in round-to-nearest mode.
 *
 * Performs \p half2 vector multiplication of inputs \p a and \p b, in
 * round-to-nearest mode.
 *
 * \return Returns the \p half2 vector result of multiplying vectors \p a and \p b.
 */
__CUDA_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector addition in round-to-nearest mode, with
 * saturation to [0.0, 1.0].
 *
 * Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest mode,
 * and clamps the results to range [0.0, 1.0]. NaN results are flushed to +0.0.
 *
 * \return Returns the \p half2 vector result of adding vectors \p a and \p b
 * with saturation.
 */
__CUDA_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector subtraction in round-to-nearest mode, with
 * saturation to [0.0, 1.0].
 *
 * Subtracts \p half2 input vector \p b from input vector \p a in round-to-nearest
 * mode,
 * and clamps the results to range [0.0, 1.0]. NaN results are flushed to +0.0.
 *
 * \return Returns the \p half2 vector result of subtraction vector \p b from \p a
 * with saturation.
 */
__CUDA_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector multiplication in round-to-nearest mode, with
 * saturation to [0.0, 1.0].
 *
 * Performs \p half2 vector multiplication of inputs \p a and \p b, in
 * round-to-nearest mode, and clamps the results to range [0.0, 1.0]. NaN
 * results are flushed to +0.0.
 *
 * \return Returns the \p half2 vector result of multiplying vectors \p a and \p
 * b with saturation.
 */
__CUDA_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector fused multiply-add in round-to-nearest mode.
 *
 * Performs \p half2 vector multiply on inputs \p a and \p b,
 * then performs a \p half2 vector add of the result with \p c,
 * rounding the result once in round-to-nearest mode.
 *
 * \return Returns the \p half2 vector result of the fused multiply-add
 * operation on vectors \p a, \p b, and \p c.
 */
__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Performs \p half2 vector fused multiply-add in round-to-nearest mode,
 * with saturation to [0.0, 1.0].
 *
 * Performs \p half2 vector multiply on inputs \p a and \p b,
 * then performs a \p half2 vector add of the result with \p c,
 * rounding the result once in round-to-nearest mode, and clamps the results to
 * range [0.0, 1.0]. NaN results are flushed to +0.0.
 *
 * \return Returns the \p half2 vector result of the fused multiply-add
 * operation on vectors \p a, \p b, and \p c with saturation.
 */
__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c);
/**
 * \ingroup CUDA_MATH__HALF2_ARITHMETIC
 * \brief Negates both halves of the input \p half2 number and returns the result.
 *
 * Negates both halves of the input \p half2 number \p a and returns the result.
 *
 * \return Returns \p half2 number with both halves negated.
 */
__CUDA_FP16_DECL__ __half2 __hneg2(const __half2 a);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half addition in round-to-nearest mode.
 *
 * Performs \p half addition of inputs \p a and \p b, in round-to-nearest mode.
 *
 * \return Returns the \p half result of adding \p a and \p b.
 */
__CUDA_FP16_DECL__ __half __hadd(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half subtraction in round-to-nearest mode.
 *
 * Subtracts \p half input \p b from input \p a in round-to-nearest
 * mode.
 *
 * \return Returns the \p half result of subtraction \p b from \p a.
 */
__CUDA_FP16_DECL__ __half __hsub(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half multiplication in round-to-nearest mode.
 *
 * Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest
 * mode.
 *
 * \return Returns the \p half result of multiplying \p a and \p b.
 */
__CUDA_FP16_DECL__ __half __hmul(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half addition in round-to-nearest mode, with saturation to
 * [0.0, 1.0].
 *
 * Performs \p half add of inputs \p a and \p b, in round-to-nearest mode,
 * and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
 *
 * \return Returns the \p half result of adding \p a and \p b with saturation.
 */
__CUDA_FP16_DECL__ __half __hadd_sat(const __half a, const __half b);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half subtraction in round-to-nearest mode, with saturation
 * to [0.0, 1.0].
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
 * \brief Performs \p half multiplication in round-to-nearest mode, with
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
 * \brief Performs \p half fused multiply-add in round-to-nearest mode.
 *
 * Performs \p half multiply on inputs \p a and \p b,
 * then performs a \p half add of the result with \p c,
 * rounding the result once in round-to-nearest mode.
 *
 * \return Returns the \p half result of the fused multiply-add operation on \p
 * a, \p b, and \p c.
 */
__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c);
/**
 * \ingroup CUDA_MATH__HALF_ARITHMETIC
 * \brief Performs \p half fused multiply-add in round-to-nearest mode,
 * with saturation to [0.0, 1.0].
 *
 * Performs \p half multiply on inputs \p a and \p b,
 * then performs a \p half add of the result with \p c,
 * rounding the result once in round-to-nearest mode, and clamps the result to
 * range [0.0, 1.0]. NaN results are flushed to +0.0.
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

#endif /*if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/

__CUDA_FP16_DECL__ float2 __half22float2(const __half2 l)
{
   float hi_float;
   float lo_float;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, low;}\n" : "=f"(lo_float) : "r"(l.x));

   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, high;}\n" : "=f"(hi_float) : "r"(l.x));

   return make_float2(lo_float, hi_float);
}
__CUDA_FP16_DECL__ __half __float2half(const float f)
{
   __half val;
   asm volatile("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val.x) : "f"(f));
   return val;
}
__CUDA_FP16_DECL__ float __half2float(const __half h)
{
   float val;
   asm volatile("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __float2half2_rn(const float f)
{
   __half2 val;
   asm("{.reg .f16 low;\n"
       "  cvt.rn.f16.f32 low, %1;\n"
       "  mov.b32 %0, {low,low};}\n" : "=r"(val.x) : "f"(f));
   return val;
}
__CUDA_FP16_DECL__ __half2 __floats2half2_rn(const float f1, const float f2)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  cvt.rn.f16.f32 low, %1;\n"
       "  cvt.rn.f16.f32 high, %2;\n"
       "  mov.b32 %0, {low,high};}\n" : "=r"(val.x) : "f"(f1), "f"(f2));
   return val;
}
__CUDA_FP16_DECL__ __half2 __float22half2_rn(const float2 f)
{
   __half2 val = __floats2half2_rn(f.x,f.y);
   return val;
}
__CUDA_FP16_DECL__ float __low2float(const __half2 l)
{
   float val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, low;}\n" : "=f"(val) : "r"(l.x));
   return val;
}
__CUDA_FP16_DECL__ float __high2float(const __half2 l)
{
   float val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, high;}\n" : "=f"(val) : "r"(l.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __lows2half2(const __half2 l, const __half2 h)
{
   __half2 val;
   asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
       "  mov.b32 {alow,ahigh}, %1;\n"
       "  mov.b32 {blow,bhigh}, %2;\n"
       "  mov.b32 %0, {alow,blow};}\n" : "=r"(val.x) : "r"(l.x), "r"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __highs2half2(const __half2 l, const __half2 h)
{
   __half2 val;
   asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
       "  mov.b32 {alow,ahigh}, %1;\n"
       "  mov.b32 {blow,bhigh}, %2;\n"
       "  mov.b32 %0, {ahigh,bhigh};}\n" : "=r"(val.x) : "r"(l.x), "r"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half __low2half(const __half2 h)
{
   __half ret;
   asm("{.reg .f16 low,high;\n"
       " mov.b32 {low,high}, %1;\n"
       " mov.b16 %0, low;}" : "=h"(ret.x) : "r"(h.x));
   return ret;
}
__CUDA_FP16_DECL__ int __hisinf(const __half a)
{
   if ( a.x == 0xFC00 )
      return -1;
   if ( a.x == 0x7C00 )
      return 1;
   return 0;
}
__CUDA_FP16_DECL__ __half2 __low2half2(const __half2 l)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high}, %1;\n"
       "  mov.b32 %0, {low,low};}\n" : "=r"(val.x) : "r"(l.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __high2half2(const __half2 l)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high}, %1;\n"
       "  mov.b32 %0, {high,high};}\n" : "=r"(val.x) : "r"(l.x));
   return val;
}
__CUDA_FP16_DECL__ __half __high2half(const __half2 h)
{
   __half ret;
   asm("{.reg .f16 low,high;\n"
       " mov.b32 {low,high}, %1;\n"
       " mov.b16 %0, high;}" : "=h"(ret.x) : "r"(h.x));
   return ret;
}
__CUDA_FP16_DECL__ __half2 __halves2half2(const __half l, const __half h)
{
   __half2 val;
   asm("{  mov.b32 %0, {%1,%2};}\n"
       : "=r"(val.x) : "h"(l.x), "h"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __half2half2(const __half lh)
{
   __half2 val;
   asm("{  mov.b32 %0, {%1,%1};}\n"
       : "=r"(val.x) : "h"(lh.x));
   return val;
}
__CUDA_FP16_DECL__ __half2 __lowhigh2highlow(const __half2 lh)
{
   __half2 val;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high}, %1;\n"
       "  mov.b32 %0, {high,low};}\n" : "=r"(val.x) : "r"(lh.x));
   return val;
}
#if __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)
/******************************************************************************
 *                            __half2 warp shuffle                            *
 ******************************************************************************/
#define SHUFFLE_HALF2_MACRO(name) do {\
   __half2 r; \
   asm("{"#name" %0,%1,%2,%3;\n}" \
       :"=r"(r.x): "r"(var.x), "r"(delta), "r"(c)); \
   return r; \
} while(0);
__CUDA_FP16_DECL__ __half2 __shfl(__half2 var, int delta, int width)
{
   int warpSize;
   asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
   int c = ((warpSize-width) << 8) | 0x1f;
   SHUFFLE_HALF2_MACRO(shfl.idx.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_up(__half2 var, unsigned int delta, int width)
{
   int warpSize;
   asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
   int c = (warpSize-width) << 8;
   SHUFFLE_HALF2_MACRO(shfl.up.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_down(__half2 var, unsigned int delta, int width)
{
   int warpSize;
   asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
   int c = ((warpSize-width) << 8) | 0x1f;
   SHUFFLE_HALF2_MACRO(shfl.down.b32);
}
__CUDA_FP16_DECL__ __half2 __shfl_xor(__half2 var, int delta, int width)
{
   int warpSize;
   asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
   int c = ((warpSize-width) << 8) | 0x1f;
   SHUFFLE_HALF2_MACRO(shfl.bfly.b32);
}
#undef SHUFFLE_HALF2_MACRO
#endif /*__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)*/
/******************************************************************************
 *                          __half and __half2 __ldg                          *
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
   asm volatile ("ld.global.nc.b32 %0, [%1];"  : "=r"(ret.x): __LDG_PTR (ptr));
   return ret;
}
__CUDA_FP16_DECL__ __half __ldg(const __half *ptr)
{
   __half ret;
   asm volatile ("ld.global.nc.b16 %0, [%1];"  : "=h"(ret.x) : __LDG_PTR (ptr));
   return ret;
}
#undef __LDG_PTR
#endif /*defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))*/
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
/******************************************************************************
 *                             __half2 comparison                             *
 ******************************************************************************/
#define COMPARISON_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{ "#name".f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(val.x) : "r"(a.x),"r"(b.x)); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half2 __heq2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.eq);
}
__CUDA_FP16_DECL__ __half2 __hne2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.ne);
}
__CUDA_FP16_DECL__ __half2 __hle2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.le);
}
__CUDA_FP16_DECL__ __half2 __hge2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.ge);
}
__CUDA_FP16_DECL__ __half2 __hlt2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.lt);
}
__CUDA_FP16_DECL__ __half2 __hgt2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.gt);
}
__CUDA_FP16_DECL__ __half2 __hequ2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.equ);
}
__CUDA_FP16_DECL__ __half2 __hneu2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.neu);
}
__CUDA_FP16_DECL__ __half2 __hleu2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.leu);
}
__CUDA_FP16_DECL__ __half2 __hgeu2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.geu);
}
__CUDA_FP16_DECL__ __half2 __hltu2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.ltu);
}
__CUDA_FP16_DECL__ __half2 __hgtu2(const __half2 a, const __half2 b)
{
   COMPARISON_OP_HALF2_MACRO(set.gtu);
}
#undef COMPARISON_OP_HALF2_MACRO
#define BOOL_COMPARISON_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{ "#name".f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(val.x) : "r"(a.x),"r"(b.x)); \
   if (val.x == 0x3C003C00) \
      return true; \
   else  \
      return false; \
} while(0);
__CUDA_FP16_DECL__ bool __hbeq2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.eq);
}
__CUDA_FP16_DECL__ bool __hbne2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.ne);
}
__CUDA_FP16_DECL__ bool __hble2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.le);
}
__CUDA_FP16_DECL__ bool __hbge2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.ge);
}
__CUDA_FP16_DECL__ bool __hblt2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.lt);
}
__CUDA_FP16_DECL__ bool __hbgt2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.gt);
}
__CUDA_FP16_DECL__ bool __hbequ2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.equ);
}
__CUDA_FP16_DECL__ bool __hbneu2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.neu);
}
__CUDA_FP16_DECL__ bool __hbleu2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.leu);
}
__CUDA_FP16_DECL__ bool __hbgeu2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.geu);
}
__CUDA_FP16_DECL__ bool __hbltu2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.ltu);
}
__CUDA_FP16_DECL__ bool __hbgtu2(const __half2 a, const __half2 b)
{
   BOOL_COMPARISON_OP_HALF2_MACRO(set.gtu);
}
#undef BOOL_COMPARISON_OP_HALF2_MACRO
/******************************************************************************
 *                             __half comparison                              *
 ******************************************************************************/
#define COMPARISON_OP_HALF_MACRO(name) do {\
   unsigned short val; \
   asm( "{ .reg .pred __$temp3;\n" \
        "  setp."#name".f16  __$temp3, %1, %2;\n" \
        "  selp.u16 %0, 1, 0, __$temp3;}" \
        : "=h"(val) : "h"(a.x), "h"(b.x)); \
   return val ? true : false; \
} while(0);
__CUDA_FP16_DECL__ bool __heq(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(eq);
}
__CUDA_FP16_DECL__ bool __hne(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(ne);
}
__CUDA_FP16_DECL__ bool __hle(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(le);
}
__CUDA_FP16_DECL__ bool __hge(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(ge);
}
__CUDA_FP16_DECL__ bool __hlt(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(lt);
}
__CUDA_FP16_DECL__ bool __hgt(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(gt);
}
__CUDA_FP16_DECL__ bool __hequ(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(equ);
}
__CUDA_FP16_DECL__ bool __hneu(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(neu);
}
__CUDA_FP16_DECL__ bool __hleu(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(leu);
}
__CUDA_FP16_DECL__ bool __hgeu(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(geu);
}
__CUDA_FP16_DECL__ bool __hltu(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(ltu);
}
__CUDA_FP16_DECL__ bool __hgtu(const __half a, const __half b)
{
   COMPARISON_OP_HALF_MACRO(gtu);
}
#undef COMPARISON_OP_HALF_MACRO
/******************************************************************************
 *                            __half2 arithmetic                             *
 ******************************************************************************/
#define BINARY_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{"#name".f16x2 %0,%1,%2;\n}" \
        :"=r"(val.x) : "r"(a.x),"r"(b.x)); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(add);
}
__CUDA_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(sub);
}
__CUDA_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(mul);
}
__CUDA_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(add.sat);
}
__CUDA_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(sub.sat);
}
__CUDA_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b)
{
   BINARY_OP_HALF2_MACRO(mul.sat);
}
#undef BINARY_OP_HALF2_MACRO
#define TERNARY_OP_HALF2_MACRO(name) do {\
   __half2 val; \
   asm( "{"#name".f16x2 %0,%1,%2,%3;\n}" \
        :"=r"(val.x) : "r"(a.x),"r"(b.x),"r"(c.x)); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c)
{
   TERNARY_OP_HALF2_MACRO(fma.rn);
}
__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c)
{
   TERNARY_OP_HALF2_MACRO(fma.rn.sat);
}
#undef TERNARY_OP_HALF2_MACRO
/******************************************************************************
 *                             __half arithmetic                             *
 ******************************************************************************/
#define BINARY_OP_HALF_MACRO(name) do {\
   __half val; \
   asm( "{"#name".f16 %0,%1,%2;\n}" \
        :"=h"(val.x) : "h"(a.x),"h"(b.x)); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half __hadd(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(add);
}
__CUDA_FP16_DECL__ __half __hsub(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(sub);
}
__CUDA_FP16_DECL__ __half __hmul(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(mul);
}
__CUDA_FP16_DECL__ __half __hadd_sat(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(add.sat);
}
__CUDA_FP16_DECL__ __half __hsub_sat(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(sub.sat);
}
__CUDA_FP16_DECL__ __half __hmul_sat(const __half a, const __half b)
{
   BINARY_OP_HALF_MACRO(mul.sat);
}
#undef BINARY_OP_HALF_MACRO
#define TERNARY_OP_HALF_MACRO(name) do {\
   __half val; \
   asm( "{"#name".f16 %0,%1,%2,%3;\n}" \
        :"=h"(val.x) : "h"(a.x),"h"(b.x),"h"(c.x)); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c)
{
   TERNARY_OP_HALF_MACRO(fma.rn);
}
__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c)
{
   TERNARY_OP_HALF_MACRO(fma.rn.sat);
}
#undef TERNARY_OP_HALF2_MACRO
__CUDA_FP16_DECL__ __half2 __hisnan2(const __half2 a)
{
   __half2 r;
   asm( "{set.nan.f16x2.f16x2 %0,%1,%2;\n}"
        :"=r"(r.x) : "r"(a.x),"r"(a.x));
   return r;
}
__CUDA_FP16_DECL__ bool __hisnan(const __half a)
{
   __half r;
   asm( "{set.nan.f16.f16 %0,%1,%2;\n}"
        :"=h"(r.x) : "h"(a.x),"h"(a.x));
   if (r.x == 0)
      return false;
   else return true;
}
__CUDA_FP16_DECL__ __half2 __hneg2(const __half2 a)
{
   __half2 zero = __float2half2_rn(0.0);
   return __hsub2(zero,a);
}
__CUDA_FP16_DECL__ __half __hneg(const __half a)
{
   __half zero;
   zero = __float2half(0.0);
   return __hsub(zero,a);
}
#endif /*__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
#undef __CUDA_FP16_DECL__
#endif /*defined(__CUDACC__)*/
#endif /* end of include guard: CUDA_FP16_H_JNESTUG4 */
