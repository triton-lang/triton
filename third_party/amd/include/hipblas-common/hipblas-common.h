/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define an extremely thin runtime layer that allows source code to be
//! compiled unmodified through either AMD HCC or NVCC.   Key features tend to
//! be in the spirit and terminology of CUDA, but with a portable path to other
//! accelerators as well.
//!
//!  This is the master include file for hipblas-common, providing shared
//!  functionality between hipBLAS and hipBLASLt.

#ifndef HIPBLAS_COMMON_H
#define HIPBLAS_COMMON_H

/*! \brief hipblas status codes definition */
typedef enum {
  HIPBLAS_STATUS_SUCCESS = 0,         /**< Function succeeds */
  HIPBLAS_STATUS_NOT_INITIALIZED = 1, /**< HIPBLAS library not initialized */
  HIPBLAS_STATUS_ALLOC_FAILED = 2,    /**< resource allocation failed */
  HIPBLAS_STATUS_INVALID_VALUE =
      3, /**< unsupported numerical value was passed to function */
  HIPBLAS_STATUS_MAPPING_ERROR = 4,    /**< access to GPU memory space failed */
  HIPBLAS_STATUS_EXECUTION_FAILED = 5, /**< GPU program failed to execute */
  HIPBLAS_STATUS_INTERNAL_ERROR =
      6,                            /**< an internal HIPBLAS operation failed */
  HIPBLAS_STATUS_NOT_SUPPORTED = 7, /**< function not implemented */
  HIPBLAS_STATUS_ARCH_MISMATCH = 8, /**< architecture mismatch */
  HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9, /**< hipBLAS handle is null pointer */
  HIPBLAS_STATUS_INVALID_ENUM =
      10, /**<  unsupported enum value was passed to function */
  HIPBLAS_STATUS_UNKNOWN =
      11, /**<  back-end returned an unsupported status code */
} hipblasStatus_t;

#ifndef HIPBLAS_OPERATION_DECLARED
#define HIPBLAS_OPERATION_DECLARED
/*! \brief Used to specify whether the matrix is to be transposed or not. */
typedef enum {
  HIPBLAS_OP_N = 111, /**<  Operate with the matrix. */
  HIPBLAS_OP_T = 112, /**<  Operate with the transpose of the matrix. */
  HIPBLAS_OP_C = 113 /**< Operate with the conjugate transpose of the matrix. */
} hipblasOperation_t;

#elif __cplusplus >= 201103L
static_assert(HIPBLAS_OP_N == 111, "Inconsistent declaration of HIPBLAS_OP_N");
static_assert(HIPBLAS_OP_T == 112, "Inconsistent declaration of HIPBLAS_OP_T");
static_assert(HIPBLAS_OP_C == 113, "Inconsistent declaration of HIPBLAS_OP_C");
#endif // HIPBLAS_OPERATION_DECLARED

/*! \brief The compute type to be used. Currently only used with GemmEx with the
 * HIPBLAS_V2 interface. Note that support for compute types is largely
 * dependent on backend. */
typedef enum {
  // Note that these types are taken from cuBLAS. With the rocBLAS backend,
  // currently hipBLAS will convert to rocBLAS types to get equivalent
  // functionality where supported.
  HIPBLAS_COMPUTE_16F = 0, /**< compute will be at least 16-bit precision */
  HIPBLAS_COMPUTE_16F_PEDANTIC =
      1,                   /**< compute will be exactly 16-bit precision */
  HIPBLAS_COMPUTE_32F = 2, /**< compute will be at least 32-bit precision */
  HIPBLAS_COMPUTE_32F_PEDANTIC =
      3, /**< compute will be exactly 32-bit precision */
  HIPBLAS_COMPUTE_32F_FAST_16F = 4,  /**< 32-bit input can use 16-bit compute */
  HIPBLAS_COMPUTE_32F_FAST_16BF = 5, /**< 32-bit input can is bf16 compute */
  HIPBLAS_COMPUTE_32F_FAST_TF32 =
      6, /**< 32-bit input can use tensor cores w/ TF32 compute. Only supported
            with cuBLAS and hipBLASLT backend currently */
  HIPBLAS_COMPUTE_64F = 7, /**< compute will be at least 64-bit precision */
  HIPBLAS_COMPUTE_64F_PEDANTIC =
      8, /**< compute will be exactly 64-bit precision */
  HIPBLAS_COMPUTE_32I =
      9, /**< compute will be at least 32-bit integer precision */
  HIPBLAS_COMPUTE_32I_PEDANTIC =
      10, /**< compute will be exactly 32-bit integer precision */
  HIPBLAS_COMPUTE_32F_FAST_8F_FNUZ =
      100, /**< 32-bit compute using fp8 mfma instruction */
  HIPBLAS_COMPUTE_32F_FAST_8BF_FNUZ =
      101, /**< 32-bit compute using bf8 mfma instruction */
  HIPBLAS_COMPUTE_32F_FAST_8F8BF_FNUZ =
      102, /**< 32-bit compute using f8bf8 mfma instruction */
  HIPBLAS_COMPUTE_32F_FAST_8BF8F_FNUZ =
      103, /**< 32-bit compute using bf8f8 mfma instruction */
} hipblasComputeType_t;

#endif
