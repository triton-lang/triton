/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/



#ifndef CLBLAS_H_
#define CLBLAS_H_

/**
 * @mainpage OpenCL BLAS
 *
 * This is an implementation of
 * <A HREF="http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms">
 * Basic Linear Algebra Subprograms</A>, levels 1, 2 and 3 using
 * <A HREF="http://www.khronos.org/opencl/">OpenCL</A> and optimized for
 * the AMD GPU hardware.
 */

#include "isaac/driver/external/CL/cl.h"
#include "clBLAS-complex.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup OVERVIEW Overview
 *
 * This library provides an implementation of the Basic Linear Algebra Subprograms levels 1, 2 and 3,
 * using OpenCL and optimized for AMD GPU hardware. It provides BLAS-1 functions
 * SWAP, SCAL, COPY, AXPY, DOT, DOTU, DOTC, ROTG, ROTMG, ROT, ROTM, iAMAX, ASUM and NRM2,
 * BLAS-2 functions GEMV, SYMV, TRMV, TRSV, HEMV, SYR, SYR2, HER, HER2, GER, GERU, GERC,
 * TPMV, SPMV, HPMV, TPSV, SPR, SPR2, HPR, HPR2, GBMV, TBMV, SBMV, HBMV and TBSV
 * and BLAS-3 functions GEMM, SYMM, TRMM, TRSM, HEMM, HERK, HER2K, SYRK and SYR2K.
 *
 * This library’s primary goal is to assist the end user to enqueue OpenCL
 * kernels to process BLAS functions in an OpenCL-efficient manner, while
 * keeping interfaces familiar to users who know how to use BLAS. All
 * functions accept matrices through buffer objects.
 *
 * This library is entirely thread-safe with the exception of the following API :
 * clblasSetup and clblasTeardown. 
 * Developers using the library can safely using any blas routine from different thread. 
 *
 * @section deprecated
 * This library provided support for the creation of scratch images to achieve better performance
 * on older <a href="http://developer.amd.com/gpu/AMDAPPSDK/Pages/default.aspx">AMD APP SDK's</a>.
 * However, memory buffers now give the same performance as buffers objects in the current SDK's.
 * Scratch image buffers are being deprecated and users are advised not to use scratch images in
 * new applications.
 */

/**
 * @defgroup TYPES clblas types
 */
/*@{*/


/** Shows how matrices are placed in memory. */
typedef enum clblasOrder_ {
    clblasRowMajor,           /**< Every row is placed sequentially */
    clblasColumnMajor         /**< Every column is placed sequentially */
} clblasOrder;

/** Used to specify whether the matrix is to be transposed or not. */
typedef enum clblasTranspose_ {
    clblasNoTrans,           /**< Operate with the matrix. */
    clblasTrans,             /**< Operate with the transpose of the matrix. */
    clblasConjTrans          /**< Operate with the conjugate transpose of
                                     the matrix. */
} clblasTranspose;

/** Used by the Hermitian, symmetric and triangular matrix
 * routines to specify whether the upper or lower triangle is being referenced.
 */
typedef enum clblasUplo_ {
    clblasUpper,               /**< Upper triangle. */
    clblasLower                /**< Lower triangle. */
} clblasUplo;

/** It is used by the triangular matrix routines to specify whether the
 * matrix is unit triangular.
 */
typedef enum clblasDiag_ {
    clblasUnit,               /**< Unit triangular. */
    clblasNonUnit             /**< Non-unit triangular. */
} clblasDiag;

/** Indicates the side matrix A is located relative to matrix B during multiplication. */
typedef enum clblasSide_ {
    clblasLeft,        /**< Multiply general matrix by symmetric,
                               Hermitian or triangular matrix on the left. */
    clblasRight        /**< Multiply general matrix by symmetric,
                               Hermitian or triangular matrix on the right. */
} clblasSide;

/**
 *   @brief clblas error codes definition, incorporating OpenCL error
 *   definitions.
 *
 *   This enumeration is a subset of the OpenCL error codes extended with some
 *   additional extra codes.  For example, CL_OUT_OF_HOST_MEMORY, which is
 *   defined in cl.h is aliased as clblasOutOfHostMemory.
 */
typedef enum clblasStatus_ {
    clblasSuccess                         = CL_SUCCESS,
    clblasInvalidValue                    = CL_INVALID_VALUE,
    clblasInvalidCommandQueue             = CL_INVALID_COMMAND_QUEUE,
    clblasInvalidContext                  = CL_INVALID_CONTEXT,
    clblasInvalidMemObject                = CL_INVALID_MEM_OBJECT,
    clblasInvalidDevice                   = CL_INVALID_DEVICE,
    clblasInvalidEventWaitList            = CL_INVALID_EVENT_WAIT_LIST,
    clblasOutOfResources                  = CL_OUT_OF_RESOURCES,
    clblasOutOfHostMemory                 = CL_OUT_OF_HOST_MEMORY,
    clblasInvalidOperation                = CL_INVALID_OPERATION,
    clblasCompilerNotAvailable            = CL_COMPILER_NOT_AVAILABLE,
    clblasBuildProgramFailure             = CL_BUILD_PROGRAM_FAILURE,
    /* Extended error codes */
    clblasNotImplemented         = -1024, /**< Functionality is not implemented */
    clblasNotInitialized,                 /**< clblas library is not initialized yet */
    clblasInvalidMatA,                    /**< Matrix A is not a valid memory object */
    clblasInvalidMatB,                    /**< Matrix B is not a valid memory object */
    clblasInvalidMatC,                    /**< Matrix C is not a valid memory object */
    clblasInvalidVecX,                    /**< Vector X is not a valid memory object */
    clblasInvalidVecY,                    /**< Vector Y is not a valid memory object */
    clblasInvalidDim,                     /**< An input dimension (M,N,K) is invalid */
    clblasInvalidLeadDimA,                /**< Leading dimension A must not be less than the size of the first dimension */
    clblasInvalidLeadDimB,                /**< Leading dimension B must not be less than the size of the second dimension */
    clblasInvalidLeadDimC,                /**< Leading dimension C must not be less than the size of the third dimension */
    clblasInvalidIncX,                    /**< The increment for a vector X must not be 0 */
    clblasInvalidIncY,                    /**< The increment for a vector Y must not be 0 */
    clblasInsufficientMemMatA,            /**< The memory object for Matrix A is too small */
    clblasInsufficientMemMatB,            /**< The memory object for Matrix B is too small */
    clblasInsufficientMemMatC,            /**< The memory object for Matrix C is too small */
    clblasInsufficientMemVecX,            /**< The memory object for Vector X is too small */
    clblasInsufficientMemVecY             /**< The memory object for Vector Y is too small */
} clblasStatus;


/*@}*/

/**
 * @defgroup VERSION Version information
 */
/*@{*/

/**
 * @brief Get the clblas library version info.
 *
 * @param[out] major        Location to store library's major version.
 * @param[out] minor        Location to store library's minor version.
 * @param[out] patch        Location to store library's patch version.
 *
 * @returns always \b clblasSuccess.
 *
 * @ingroup VERSION
 */
clblasStatus
clblasGetVersion(cl_uint* major, cl_uint* minor, cl_uint* patch);

/*@}*/

/**
 * @defgroup INIT Initialize library
 */
/*@{*/

/**
 * @brief Initialize the clblas library.
 *
 * Must be called before any other clblas API function is invoked.
 * @note This function is not thread-safe.
 *
 * @return
 *   - \b clblasSucces on success;
 *   - \b clblasOutOfHostMemory if there is not enough of memory to allocate
 *     library's internal structures;
 *   - \b clblasOutOfResources in case of requested resources scarcity.
 *
 * @ingroup INIT
 */
clblasStatus
clblasSetup(void);

/**
 * @brief Finalize the usage of the clblas library.
 *
 * Frees all memory allocated for different computational kernel and other
 * internal data.
 * @note This function is not thread-safe.
 *
 * @ingroup INIT
 */
void
clblasTeardown(void);

/*@}*/

/**
 * @defgroup BLAS1 BLAS-1 functions
 *
 * The Level 1 Basic Linear Algebra Subprograms are functions that perform
 * vector-vector operations.
 */
/*@{*/
/*@}*/

/**
 * @defgroup SWAP SWAP  - Swap elements from 2 vectors
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief interchanges two vectors of float.
 *
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SWAP
 */
clblasStatus
clblasSswap(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sswap.c
 * Example of how to use the @ref clblasSswap function.
 */

 /**
 * @brief interchanges two vectors of double.
 *
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSswap() function otherwise.
 *
 * @ingroup SWAP
 */
clblasStatus
clblasDswap(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief interchanges two vectors of complex-float elements.
 *
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasSwap() function otherwise.
 *
 * @ingroup SWAP
 */
clblasStatus
clblasCswap(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief interchanges two vectors of double-complex elements.
 *
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasDwap() function otherwise.
 *
 * @ingroup SWAP
 */
clblasStatus
clblasZswap(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup SCAL SCAL  - Scales a vector by a constant
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief Scales a half vector by a half constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - \b incx zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SCAL
 */
clblasStatus
clblasHscal(
    size_t N,
    cl_float alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @brief Scales a float vector by a float constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - \b incx zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SCAL
 */
clblasStatus
clblasSscal(
    size_t N,
    cl_float alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_sscal.c
 * Example of how to use the @ref clblasSscal function.
 */

 /**
 * @brief Scales a double vector by a double constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSscal() function otherwise.
 *
 * @ingroup SCAL
 */
clblasStatus
clblasDscal(
    size_t N,
    cl_double alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Scales a complex-float vector by a complex-float constant
 *
  *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasSscal() function otherwise.
 *
 * @ingroup SCAL
 */
clblasStatus
clblasCscal(
    size_t N,
    cl_float2 alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Scales a complex-double vector by a complex-double constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasDscal() function otherwise.
 *
 * @ingroup SCAL
 */
clblasStatus
clblasZscal(
    size_t N,
    cl_double2 alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup SSCAL SSCAL  - Scales a complex vector by a real constant
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief Scales a complex-float vector by a float constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - \b incx zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SSCAL
 */
clblasStatus
clblasCsscal(
    size_t N,
    cl_float alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_csscal.c
 * Example of how to use the @ref clblasCsscal function.
 */

/**
 * @brief Scales a complex-double vector by a double constant
 *
 *   - \f$ X \leftarrow \alpha X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasCsscal() function otherwise.
 *
 * @ingroup SSCAL
 */
clblasStatus
clblasZdscal(
    size_t N,
    cl_double alpha,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

 /*@}*/


/**
 * @defgroup COPY COPY  - Copies elements from vector X to vector Y
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief Copies half elements from vector X to vector Y
 *
 *   - \f$ Y \leftarrow X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup COPY
 */
clblasStatus
clblasHcopy(
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @brief Copies float elements from vector X to vector Y
 *
 *   - \f$ Y \leftarrow X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup COPY
 */
clblasStatus
clblasScopy(
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_scopy.c
 * Example of how to use the @ref clblasScopy function.
 */

 /**
 * @brief Copies double elements from vector X to vector Y
 *
 *   - \f$ Y \leftarrow X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasScopy() function otherwise.
 *
 * @ingroup COPY
 */
clblasStatus
clblasDcopy(
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Copies complex-float elements from vector X to vector Y
 *
 *   - \f$ Y \leftarrow X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasScopy() function otherwise.
 *
 * @ingroup COPY
 */
clblasStatus
clblasCcopy(
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Copies complex-double elements from vector X to vector Y
 *
 *   - \f$ Y \leftarrow X \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasDcopy() function otherwise.
 *
 * @ingroup COPY
 */
clblasStatus
clblasZcopy(
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

 /*@}*/

/**
 * @defgroup AXPY AXPY  - Scale X and add to Y
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief Scale vector X of half elements and add to Y
 *
 *   - \f$ Y \leftarrow \alpha X + Y \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup AXPY
 */
clblasStatus
clblasHaxpy(
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @brief Scale vector X of float elements and add to Y
 *
 *   - \f$ Y \leftarrow \alpha X + Y \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup AXPY
 */
clblasStatus
clblasSaxpy(
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @example example_saxpy.c
 * Example of how to use the @ref clblasSaxpy function.
 */

/**
 * @brief Scale vector X of double elements and add to Y
 *
 *   - \f$ Y \leftarrow \alpha X + Y \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSaxpy() function otherwise.
 *
 * @ingroup AXPY
 */
clblasStatus
clblasDaxpy(
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Scale vector X of complex-float elements and add to Y
 *
 *   - \f$ Y \leftarrow \alpha X + Y \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasSaxpy() function otherwise.
 *
 * @ingroup AXPY
 */
clblasStatus
clblasCaxpy(
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Scale vector X of double-complex elements and add to Y
 *
 *   - \f$ Y \leftarrow \alpha X + Y \f$
 *
 * @param[in] N         Number of elements in vector \b X.
 * @param[in] alpha     The constant factor for vector \b X.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasDaxpy() function otherwise.
 *
 * @ingroup AXPY
 */
clblasStatus
clblasZaxpy(
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup DOT DOT  - Dot product of two vectors
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief dot product of two vectors containing half elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, \b Y or \b dotProduct object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup DOT
 */
clblasStatus
clblasHdot(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief dot product of two vectors containing float elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, \b Y or \b dotProduct object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup DOT
 */
clblasStatus
clblasSdot(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sdot.c
 * Example of how to use the @ref clblasSdot function.
 */

/**
 * @brief dot product of two vectors containing double elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSdot() function otherwise.
 *
 * @ingroup DOT
 */
clblasStatus
clblasDdot(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief dot product of two vectors containing float-complex elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasSdot() function otherwise.
 *
 * @ingroup DOT
 */

clblasStatus
clblasCdotu(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief dot product of two vectors containing double-complex elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSdot() function otherwise.
 *
 * @ingroup DOT
 */

clblasStatus
clblasZdotu(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief dot product of two vectors containing float-complex elements conjugating the first vector
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasSdot() function otherwise.
 *
 * @ingroup DOT
 */

clblasStatus
clblasCdotc(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief dot product of two vectors containing double-complex elements conjugating the first vector
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] dotProduct   Buffer object that will contain the dot-product value
 * @param[in] offDP         Offset to dot-product in \b dotProduct buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] Y             Buffer object storing the vector \b Y.
 * @param[in] offy          Offset of first element of vector \b Y in buffer object.
 *                          Counted in elements.
 * @param[in] incy          Increment for the elements of \b Y. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSdot() function otherwise.
 *
 * @ingroup DOT
 */

clblasStatus
clblasZdotc(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup ROTG ROTG  - Constructs givens plane rotation
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief construct givens plane rotation on float elements
 *
 * @param[out] SA           Buffer object that contains SA
 * @param[in] offSA         Offset to SA in \b SA buffer object.
 *                          Counted in elements.
 * @param[out] SB           Buffer object that contains SB
 * @param[in] offSB         Offset to SB in \b SB buffer object.
 *                          Counted in elements.
 * @param[out] C            Buffer object that contains C
 * @param[in] offC          Offset to C in \b C buffer object.
 *                          Counted in elements.
 * @param[out] S            Buffer object that contains S
 * @param[in] offS          Offset to S in \b S buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidMemObject if either \b SA, \b SB, \b C or \b S object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ROTG
 */
clblasStatus
clblasSrotg(
    cl_mem SA,
    size_t offSA,
    cl_mem SB,
    size_t offSB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_srotg.c
 * Example of how to use the @ref clblasSrotg function.
 */

/**
 * @brief construct givens plane rotation on double elements
 *
 * @param[out] DA           Buffer object that contains DA
 * @param[in] offDA         Offset to DA in \b DA buffer object.
 *                          Counted in elements.
 * @param[out] DB           Buffer object that contains DB
 * @param[in] offDB         Offset to DB in \b DB buffer object.
 *                          Counted in elements.
 * @param[out] C            Buffer object that contains C
 * @param[in] offC          Offset to C in \b C buffer object.
 *                          Counted in elements.
 * @param[out] S            Buffer object that contains S
 * @param[in] offS          Offset to S in \b S buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSrotg() function otherwise.
 *
 * @ingroup ROTG
 */
clblasStatus
clblasDrotg(
    cl_mem DA,
    size_t offDA,
    cl_mem DB,
    size_t offDB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief construct givens plane rotation on float-complex elements
 *
 * @param[out] CA           Buffer object that contains CA
 * @param[in] offCA         Offset to CA in \b CA buffer object.
 *                          Counted in elements.
 * @param[out] CB           Buffer object that contains CB
 * @param[in] offCB         Offset to CB in \b CB buffer object.
 *                          Counted in elements.
 * @param[out] C            Buffer object that contains C. C is real.
 * @param[in] offC          Offset to C in \b C buffer object.
 *                          Counted in elements.
 * @param[out] S            Buffer object that contains S
 * @param[in] offS          Offset to S in \b S buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasSrotg() function otherwise.
 *
 * @ingroup ROTG
 */
clblasStatus
clblasCrotg(
    cl_mem CA,
    size_t offCA,
    cl_mem CB,
    size_t offCB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief construct givens plane rotation on double-complex elements
 *
 * @param[out] CA           Buffer object that contains CA
 * @param[in] offCA         Offset to CA in \b CA buffer object.
 *                          Counted in elements.
 * @param[out] CB           Buffer object that contains CB
 * @param[in] offCB         Offset to CB in \b CB buffer object.
 *                          Counted in elements.
 * @param[out] C            Buffer object that contains C. C is real.
 * @param[in] offC          Offset to C in \b C buffer object.
 *                          Counted in elements.
 * @param[out] S            Buffer object that contains S
 * @param[in] offS          Offset to S in \b S buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasDrotg() function otherwise.
 *
 * @ingroup ROTG
 */
clblasStatus
clblasZrotg(
    cl_mem CA,
    size_t offCA,
    cl_mem CB,
    size_t offCB,
    cl_mem C,
    size_t offC,
    cl_mem S,
    size_t offS,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup ROTMG ROTMG  - Constructs the modified givens rotation
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief construct the modified givens rotation on float elements
 *
 * @param[out] SD1          Buffer object that contains SD1
 * @param[in] offSD1        Offset to SD1 in \b SD1 buffer object.
 *                          Counted in elements.
 * @param[out] SD2          Buffer object that contains SD2
 * @param[in] offSD2        Offset to SD2 in \b SD2 buffer object.
 *                          Counted in elements.
 * @param[out] SX1          Buffer object that contains SX1
 * @param[in] offSX1        Offset to SX1 in \b SX1 buffer object.
 *                          Counted in elements.
 * @param[in] SY1           Buffer object that contains SY1
 * @param[in] offSY1        Offset to SY1 in \b SY1 buffer object.
 *                          Counted in elements.
 * @param[out] SPARAM       Buffer object that contains SPARAM array of minimum length 5
                            SPARAM(0) = SFLAG
                            SPARAM(1) = SH11
                            SPARAM(2) = SH21
                            SPARAM(3) = SH12
                            SPARAM(4) = SH22

 * @param[in] offSparam     Offset to SPARAM in \b SPARAM buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidMemObject if either \b SX1, \b SY1, \b SD1, \b SD2 or \b SPARAM object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ROTMG
 */
clblasStatus
clblasSrotmg(
    cl_mem SD1,
    size_t offSD1,
    cl_mem SD2,
    size_t offSD2,
    cl_mem SX1,
    size_t offSX1,
    const cl_mem SY1,
    size_t offSY1,
    cl_mem SPARAM,
    size_t offSparam,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_srotmg.c
 * Example of how to use the @ref clblasSrotmg function.
 */

/**
 * @brief construct the modified givens rotation on double elements
 *
 * @param[out] DD1          Buffer object that contains DD1
 * @param[in] offDD1        Offset to DD1 in \b DD1 buffer object.
 *                          Counted in elements.
 * @param[out] DD2          Buffer object that contains DD2
 * @param[in] offDD2        Offset to DD2 in \b DD2 buffer object.
 *                          Counted in elements.
 * @param[out] DX1          Buffer object that contains DX1
 * @param[in] offDX1        Offset to DX1 in \b DX1 buffer object.
 *                          Counted in elements.
 * @param[in] DY1           Buffer object that contains DY1
 * @param[in] offDY1        Offset to DY1 in \b DY1 buffer object.
 *                          Counted in elements.
 * @param[out] DPARAM       Buffer object that contains DPARAM array of minimum length 5
                            DPARAM(0) = DFLAG
                            DPARAM(1) = DH11
                            DPARAM(2) = DH21
                            DPARAM(3) = DH12
                            DPARAM(4) = DH22

 * @param[in] offDparam     Offset to DPARAM in \b DPARAM buffer object.
 *                          Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSrotmg() function otherwise.
 *
 * @ingroup ROTMG
 */
clblasStatus
clblasDrotmg(
    cl_mem DD1,
    size_t offDD1,
    cl_mem DD2,
    size_t offDD2,
    cl_mem DX1,
    size_t offDX1,
    const cl_mem DY1,
    size_t offDY1,
    cl_mem DPARAM,
    size_t offDparam,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup ROT ROT  - Apply givens rotation
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief applies a plane rotation for float elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] C         C specifies the cosine, cos.
 * @param[in] S         S specifies the sine, sin.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ROT
 */
clblasStatus
clblasSrot(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_float C,
    cl_float S,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_srot.c
 * Example of how to use the @ref clblasSrot function.
 */

/**
 * @brief applies a plane rotation for double elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] C         C specifies the cosine, cos.
 * @param[in] S         S specifies the sine, sin.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSrot() function otherwise.
 *
 * @ingroup ROT
 */
clblasStatus
clblasDrot(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_double C,
    cl_double S,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief applies a plane rotation for float-complex elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] C         C specifies the cosine, cos. This number is real
 * @param[in] S         S specifies the sine, sin. This number is real
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasSrot() function otherwise.
 *
 * @ingroup ROT
 */
clblasStatus
clblasCsrot(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_float C,
    cl_float S,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief applies a plane rotation for double-complex elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] C         C specifies the cosine, cos. This number is real
 * @param[in] S         S specifies the sine, sin. This number is real
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSrot() function otherwise.
 *
 * @ingroup ROT
 */
clblasStatus
clblasZdrot(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_double C,
    cl_double S,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup ROTM ROTM  - Apply modified givens rotation for points in the plane
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief modified givens rotation for float elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] SPARAM    Buffer object that contains SPARAM array of minimum length 5
 *                      SPARAM(1)=SFLAG
 *                      SPARAM(2)=SH11
 *                      SPARAM(3)=SH21
 *                      SPARAM(4)=SH12
 *                      SPARAM(5)=SH22
 * @param[in] offSparam Offset of first element of array \b SPARAM in buffer object.
 *                      Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b X, \b Y or \b SPARAM object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ROTM
 */
clblasStatus
clblasSrotm(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    const cl_mem SPARAM,
    size_t offSparam,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_srotm.c
 * Example of how to use the @ref clblasSrotm function.
 */

/**
 * @brief modified givens rotation for double elements
 *
 * @param[in] N         Number of elements in vector \b X and \b Y.
 * @param[out] X        Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] Y        Buffer object storing the vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] DPARAM    Buffer object that contains SPARAM array of minimum length 5
 *                      DPARAM(1)=DFLAG
 *                      DPARAM(2)=DH11
 *                      DPARAM(3)=DH21
 *                      DPARAM(4)=DH12
 *                      DPARAM(5)=DH22
 * @param[in] offDparam Offset of first element of array \b DPARAM in buffer object.
 *                      Counted in elements.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
* @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSrotm() function otherwise.
 *
 * @ingroup ROTM
 */
clblasStatus
clblasDrotm(
    size_t N,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    const cl_mem DPARAM,
    size_t offDparam,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup NRM2 NRM2  - Euclidean norm of a vector
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief computes the euclidean norm of vector containing float elements
 *
 *  NRM2 = sqrt( X' * X )
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] NRM2         Buffer object that will contain the NRM2 value
 * @param[in] offNRM2       Offset to NRM2 value in \b NRM2 buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object that can hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if any of \b X or \b NRM2 or \b scratchBuff object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup NRM2
 */
clblasStatus
clblasSnrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_snrm2.c
 * Example of how to use the @ref clblasSnrm2 function.
 */

/**
 * @brief computes the euclidean norm of vector containing double elements
 *
 *  NRM2 = sqrt( X' * X )
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] NRM2         Buffer object that will contain the NRM2 value
 * @param[in] offNRM2       Offset to NRM2 value in \b NRM2 buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object that can hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSnrm2() function otherwise.
 *
 * @ingroup NRM2
 */
clblasStatus
clblasDnrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief computes the euclidean norm of vector containing float-complex elements
 *
 *  NRM2 = sqrt( X**H * X )
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] NRM2         Buffer object that will contain the NRM2 value.
 *                          Note that the answer of Scnrm2 is a real value.
 * @param[in] offNRM2       Offset to NRM2 value in \b NRM2 buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object that can hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasSnrm2() function otherwise.
 *
 * @ingroup NRM2
 */
clblasStatus
clblasScnrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief computes the euclidean norm of vector containing double-complex elements
 *
 *  NRM2 = sqrt( X**H * X )
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] NRM2         Buffer object that will contain the NRM2 value.
 *                          Note that the answer of Dznrm2 is a real value.
 * @param[in] offNRM2       Offset to NRM2 value in \b NRM2 buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff	Temporary cl_mem scratch buffer object that can hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSnrm2() function otherwise.
 *     executable.
 *
 * @ingroup NRM2
 */
clblasStatus
clblasDznrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup iAMAX iAMAX  - Index of max absolute value
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief index of max absolute value in a float array
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] iMax         Buffer object storing the index of first absolute max.
 *                          The index will be of type unsigned int
 * @param[in] offiMax       Offset for storing index in the buffer iMax
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temprory cl_mem object to store intermediate results
                            It should be able to hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if any of \b iMax \b X or \b scratchBuff object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if the context, the passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup iAMAX
 */
clblasStatus
clblasiSamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_isamax.c
 * Example of how to use the @ref clblasiSamax function.
 */


/**
 * @brief index of max absolute value in a double array
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] iMax         Buffer object storing the index of first absolute max.
 *                          The index will be of type unsigned int
 * @param[in] offiMax       Offset for storing index in the buffer iMax
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temprory cl_mem object to store intermediate results
                            It should be able to hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasiSamax() function otherwise.
 *
 * @ingroup iAMAX
 */
clblasStatus
clblasiDamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief index of max absolute value in a complex float array
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] iMax         Buffer object storing the index of first absolute max.
 *                          The index will be of type unsigned int
 * @param[in] offiMax       Offset for storing index in the buffer iMax
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temprory cl_mem object to store intermediate results
                            It should be able to hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasiSamax() function otherwise.
 *
 * @ingroup iAMAX
 */
clblasStatus
clblasiCamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief index of max absolute value in a complex double array
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] iMax         Buffer object storing the index of first absolute max.
 *                          The index will be of type unsigned int
 * @param[in] offiMax       Offset for storing index in the buffer iMax
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temprory cl_mem object to store intermediate results
                            It should be able to hold minimum of (2*N) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasiSamax() function otherwise.
 *
 * @ingroup iAMAX
 */
clblasStatus
clblasiZamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup ASUM ASUM  - Sum of absolute values
 * @ingroup BLAS1
 */
/*@{*/

/**
 * @brief absolute sum of values of a vector containing half elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] asum         Buffer object that will contain the absoule sum value
 * @param[in] offAsum       Offset to absolute sum in \b asum buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if any of \b X or \b asum or \b scratchBuff object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ASUM
 */

clblasStatus
clblasHasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief absolute sum of values of a vector containing float elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] asum         Buffer object that will contain the absoule sum value
 * @param[in] offAsum       Offset to absolute sum in \b asum buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if any of \b X or \b asum or \b scratchBuff object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup ASUM
 */

clblasStatus
clblasSasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sasum.c
 * Example of how to use the @ref clblasSasum function.
 */

/**
 * @brief absolute sum of values of a vector containing double elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] asum         Buffer object that will contain the absoulte sum value
 * @param[in] offAsum       Offset to absoule sum in \b asum buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSasum() function otherwise.
 *
 * @ingroup ASUM
 */

clblasStatus
clblasDasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief absolute sum of values of a vector containing float-complex elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] asum         Buffer object that will contain the absolute sum value
 * @param[in] offAsum       Offset to absolute sum in \b asum buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - the same error codes as the clblasSasum() function otherwise.
 *
 * @ingroup ASUM
 */

clblasStatus
clblasScasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief absolute sum of values of a vector containing double-complex elements
 *
 * @param[in] N             Number of elements in vector \b X.
 * @param[out] asum         Buffer object that will contain the absolute sum value
 * @param[in] offAsum       Offset to absolute sum in \b asum buffer object.
 *                          Counted in elements.
 * @param[in] X             Buffer object storing vector \b X.
 * @param[in] offx          Offset of first element of vector \b X in buffer object.
 *                          Counted in elements.
 * @param[in] incx          Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff   Temporary cl_mem scratch buffer object of minimum size N
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clblasSasum() function otherwise.
 *
 * @ingroup ASUM
 */

clblasStatus
clblasDzasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup BLAS2 BLAS-2 functions
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * matrix-vector operations.
 */
/*@{*/
/*@}*/


/**
 * @defgroup GEMV GEMV  - General matrix-Vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        float elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clblasColumnMajor.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clblasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clblasStatus
clblasSgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sgemv.c
 * This is an example of how to use the @ref clblasSgemvEx function.
 */

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        double elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of \b A in the buffer
 *                      object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clblasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - \b clblasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clblasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clblasStatus
clblasDgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        float complex elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clblasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clblasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clblasStatus
clblasCgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    FloatComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        double complex elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clblasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - \b clblasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clblasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clblasStatus
clblasZgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup SYMV SYMV  - Symmetric matrix-Vector multiplication
 * @ingroup BLAS2
 */

/*@{*/

/**
 * @brief Matrix-vector product with a symmetric matrix and float elements.
 *
 *
 * Matrix-vector products:
 * - \f$ y \leftarrow \alpha A x + \beta y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b x. It cannot be zero.
 * @param[in] beta      The factor of vector \b y.
 * @param[out] y        Buffer object storing vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clblasSgemv() function otherwise.
 *
 * @ingroup SYMV
 */
clblasStatus
clblasSsymv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_ssymv.c
 * This is an example of how to use the @ref clblasSsymv function.
 */

/**
 * @brief Matrix-vector product with a symmetric matrix and double elements.
 *
 *
 * Matrix-vector products:
 * - \f$ y \leftarrow \alpha A x + \beta y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b x. It cannot be zero.
 * @param[in] beta      The factor of vector \b y.
 * @param[out] y        Buffer object storing vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clblasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clblasSsymv() function otherwise.
 *
 * @ingroup SYMV
 */
clblasStatus
clblasDsymv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup HEMV HEMV  - Hermitian matrix-vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a hermitian matrix and float-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HEMV
 */
clblasStatus
clblasChemv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    FloatComplex beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a hermitian matrix and double-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasChemv() function otherwise.
 *
 * @ingroup HEMV
 */
clblasStatus
clblasZhemv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_zhemv.cpp
 * Example of how to use the @ref clblasZhemv function.
 */
/*@}*/



/**
 * @defgroup TRMV TRMV  - Triangular matrix vector multiply
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a triangular matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TRMV
 */
clblasStatus
clblasStrmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_strmv.c
 * Example of how to use the @ref clblasStrmv function.
 */

/**
 * @brief Matrix-vector product with a triangular matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasStrmv() function otherwise.
 *
 * @ingroup TRMV
 */
clblasStatus
clblasDtrmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a triangular matrix and
 * float complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clblasStrmv() function.
 * @ingroup TRMV
 */
clblasStatus
clblasCtrmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a triangular matrix and
 * double complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clblasDtrmv() function.
 * @ingroup TRMV
 */
clblasStatus
clblasZtrmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/*@}*/

/**
 * @defgroup TRSV TRSV  - Triangular matrix vector Solve
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief solving triangular matrix problems with float elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TRSV
 */
clblasStatus
clblasStrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_strsv.c
 * Example of how to use the @ref clblasStrsv function.
 */


/**
 * @brief solving triangular matrix problems with double elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasStrsv() function otherwise.
 *
 * @ingroup TRSV
 */
clblasStatus
clblasDtrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief solving triangular matrix problems with float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clblasStrsv() function.
 *
 * @ingroup TRSV
 */
clblasStatus
clblasCtrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief solving triangular matrix problems with double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clblasDtrsv() function.
 *
 * @ingroup TRSV
 */
clblasStatus
clblasZtrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup GER GER   - General matrix rank 1 operation
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief vector-vector product with float elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clblasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or
 *	   - either \b incx or \b incy is zero, or
 *     - a leading dimension is invalid;
 *   - \b clblasInvalidMemObject if A, X, or Y object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clblasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GER
 */
clblasStatus
clblasSger(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sger.c
 * Example of how to use the @ref clblasSger function.
 */


/**
 * @brief vector-vector product with double elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clblasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasSger() function otherwise.
 *
 * @ingroup GER
 */
clblasStatus
clblasDger(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/

/**
 * @defgroup GERU GERU  - General matrix rank 1 operation
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief vector-vector product with float complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clblasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or
 *	   - either \b incx or \b incy is zero, or
 *     - a leading dimension is invalid;
 *   - \b clblasInvalidMemObject if A, X, or Y object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clblasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GERU
 */
clblasStatus
clblasCgeru(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A ,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief vector-vector product with double complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		   Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clblasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasCgeru() function otherwise.
 *
 * @ingroup GERU
 */
clblasStatus
clblasZgeru(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/

/**
 * @defgroup GERC GERC  - General matrix rank 1 operation
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief vector-vector product with float complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clblasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or
 *	   - either \b incx or \b incy is zero, or
 *     - a leading dimension is invalid;
 *   - \b clblasInvalidMemObject if A, X, or Y object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clblasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GERC
 */

clblasStatus
clblasCgerc(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A ,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief vector-vector product with double complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clblasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasCgerc() function otherwise.
 *
 * @ingroup GERC
 */
clblasStatus
clblasZgerc(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/*@}*/

/**
 * @defgroup SYR SYR   - Symmetric rank 1 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 1 update operations.
  * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 1 operation with a general triangular matrix and
 * float elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha x x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR
 */
clblasStatus
clblasSsyr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);

/**
 * @brief Symmetric rank 1 operation with a general triangular matrix and
 * double elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha x x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasSsyr() function otherwise.
 *
 * @ingroup SYR
 */

clblasStatus
clblasDsyr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/*@}*/


/**
 * @defgroup HER HER   - Hermitian rank 1 operation
 *
 * The Level 2 Basic Linear Algebra Subprogram functions that perform
 * hermitian rank 1 operations.
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief hermitian rank 1 operation with a general triangular matrix and
 * float-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar float value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HER
 */
clblasStatus
clblasCher(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/**
 * @example example_cher.c
 * Example of how to use the @ref clblasCher function.
 */

/**
 * @brief hermitian rank 1 operation with a general triangular matrix and
 * double-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar double value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasCher() function otherwise.
 *
 * @ingroup HER
 */
clblasStatus
clblasZher(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/*@}*/

/**
 * @defgroup SYR2 SYR2  - Symmetric rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 2 update operations.
  * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 2 operation with a general triangular matrix and
 * float elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha x y^T + \alpha y x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR2
 */

clblasStatus
clblasSsyr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int  incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);

/**
 * @brief Symmetric rank 2 operation with a general triangular matrix and
 * double elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha x y^T + \alpha y x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR2
 */

clblasStatus
clblasDsyr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);

/*@}*/

/**
 * @defgroup HER2 HER2  - Hermitian rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * hermitian rank 2 update operations.
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Hermitian rank 2 operation with a general triangular matrix and
 * float-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \overline{ \alpha } Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HER2
 */
clblasStatus
clblasCher2(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);


/**
* @brief Hermitian rank 2 operation with a general triangular matrix and
 * double-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \overline{ \alpha } Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasCher2() function otherwise.
 *
 * @ingroup HER2
 */
clblasStatus
clblasZher2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
	cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);

/**
 * @example example_zher2.c
 * Example of how to use the @ref clblasZher2 function.
 */

/*@}*/

/**
 * @defgroup TPMV TPMV  - Triangular packed matrix-vector multiply
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a packed triangular matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero
 *   - \b clblasInvalidMemObject if either \b AP or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPMV
 */
clblasStatus
clblasStpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_stpmv.c
 * Example of how to use the @ref clblasStpmv function.
 */

/**
 * @brief Matrix-vector product with a packed triangular matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b AP.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasStpmv() function otherwise.
 *
 * @ingroup TPMV
 */
clblasStatus
clblasDtpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
  * @brief Matrix-vector product with a packed triangular matrix and
 * float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b AP.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clblasStpmv() function.
 * @ingroup TPMV
 */
clblasStatus
clblasCtpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a packed triangular matrix and
 * double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b AP.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clblasDtpmv() function.
 * @ingroup TPMV
 */
clblasStatus
clblasZtpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/



/**
 * @defgroup TPSV TPSV  - Triangular packed matrix vector solve
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief solving triangular packed matrix problems with float elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */

clblasStatus
clblasStpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_stpsv.c
 * Example of how to use the @ref clblasStpsv function.
 */

/**
 * @brief solving triangular packed matrix problems with double elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */

clblasStatus
clblasDtpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief solving triangular packed matrix problems with float complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */

clblasStatus
clblasCtpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief solving triangular packed matrix problems with double complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */

clblasStatus
clblasZtpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup SPMV SPMV  - Symmetric packed matrix vector multiply
 * @ingroup BLAS2
 */

/*@{*/

/**
 * @brief Matrix-vector product with a symmetric packed-matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SPMV
 */
clblasStatus
clblasSspmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sspmv.c
 * This is an example of how to use the @ref clblasSspmv function.
 */

/**
 * @brief Matrix-vector product with a symmetric packed-matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasSspmv() function otherwise.
 *
 * @ingroup SPMV
 */
clblasStatus
clblasDspmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/



/**
 * @defgroup HPMV HPMV  - Hermitian packed matrix-vector multiplication
 * @ingroup BLAS2
 */

/*@{*/

/**
 * @brief Matrix-vector product with a packed hermitian matrix and float-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing packed matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HPMV
 */
clblasStatus
clblasChpmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float2 alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_chpmv.c
 * This is an example of how to use the @ref clblasChpmv function.
 */


/**
 * @brief Matrix-vector product with a packed hermitian matrix and double-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing packed matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasChpmv() function otherwise.
 *
 * @ingroup HPMV
 */
clblasStatus
clblasZhpmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double2 alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup SPR SPR   - Symmetric packed matrix rank 1 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 1 update operations on packed matrix
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 1 operation with a general triangular packed-matrix and
 * float elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero
 *   - \b clblasInvalidMemObject if either \b AP, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SPR
 */
clblasStatus
clblasSspr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/**
 * @example example_sspr.c
 * Example of how to use the @ref clblasSspr function.
 */

/**
 * @brief Symmetric rank 1 operation with a general triangular packed-matrix and
 * double elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasSspr() function otherwise.
 *
 * @ingroup SPR
 */

clblasStatus
clblasDspr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/*@}*/

/**
 * @defgroup HPR HPR   - Hermitian packed matrix rank 1 update
 *
 * The Level 2 Basic Linear Algebra Subprogram functions that perform
 * hermitian rank 1 operations on packed matrix
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief hermitian rank 1 operation with a general triangular packed-matrix and
 * float-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar float value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero
 *   - \b clblasInvalidMemObject if either \b AP, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HPR
 */
clblasStatus
clblasChpr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int  incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/**
 * @example example_chpr.c
 * Example of how to use the @ref clblasChpr function.
 */

/**
 * @brief hermitian rank 1 operation with a general triangular packed-matrix and
 * double-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar float value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasChpr() function otherwise.
 *
 * @ingroup HPR
 */
clblasStatus
clblasZhpr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/*@}*/

/**
 * @defgroup SPR2 SPR2  - Symmetric packed matrix rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 2 update operations on packed matrices
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 2 operation with a general triangular packed-matrix and
 * float elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^T + \alpha Y X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero
 *   - \b clblasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SPR2
 */

clblasStatus
clblasSspr2(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/**
 * @example example_sspr2.c
 * Example of how to use the @ref clblasSspr2 function.
 */

/**
 * @brief Symmetric rank 2 operation with a general triangular packed-matrix and
 * double elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^T + \alpha Y X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasSspr2() function otherwise.
 *
 * @ingroup SPR2
 */

clblasStatus
clblasDspr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
	cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/*@}*/

/**
 * @defgroup HPR2 HPR2  - Hermitian packed matrix rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * hermitian rank 2 update operations on packed matrices
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Hermitian rank 2 operation with a general triangular packed-matrix and
 * float-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \conjg( alpha ) Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero
 *   - \b clblasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HPR2
 */
clblasStatus
clblasChpr2(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);


/**
 * @brief Hermitian rank 2 operation with a general triangular packed-matrix and
 * double-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \conjg( alpha ) Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasChpr2() function otherwise.
 *
 * @ingroup HPR2
 */
clblasStatus
clblasZhpr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
	cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);

/**
 * @example example_zhpr2.c
 * Example of how to use the @ref clblasZhpr2 function.
 */
/*@}*/



/**
 * @defgroup GBMV GBMV  - General banded matrix-vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b M or \b N is zero, or
 *     - KL is greater than \b M - 1, or
 *     - KU is greater than \b N - 1, or
 *     - either \b incx or \b incy is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix size or the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GBMV
 */
clblasStatus
clblasSgbmv(
    clblasOrder order,
    clblasTranspose trans,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_sgbmv.c
 * Example of how to use the @ref clblasSgbmv function.
 */


/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasSgbmv() function otherwise.
 *
 * @ingroup GBMV
 */
clblasStatus
clblasDgbmv(
    clblasOrder order,
    clblasTranspose trans,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clblasSgbmv() function.
 *
 * @ingroup GBMV
 */
clblasStatus
clblasCgbmv(
    clblasOrder order,
    clblasTranspose trans,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clblasDgbmv() function.
 *
 * @ingroup GBMV
 */
clblasStatus
clblasZgbmv(
    clblasOrder order,
    clblasTranspose trans,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup TBMV TBMV  - Triangular banded matrix vector multiply
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TBMV
 */
clblasStatus
clblasStbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_stbmv.c
 * Example of how to use the @ref clblasStbmv function.
 */


/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasStbmv() function otherwise.
 *
 * @ingroup TBMV
 */
clblasStatus
clblasDtbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
* @return The same result as the clblasStbmv() function.
 *
 * @ingroup TBMV
 */
clblasStatus
clblasCtbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
* @return The same result as the clblasDtbmv() function.
 *
 * @ingroup TBMV
 */
clblasStatus
clblasZtbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup SBMV SBMV  - Symmetric banded matrix-vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a symmetric banded matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SBMV
 */
clblasStatus
clblasSsbmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_ssbmv.c
 * This is an example of how to use the @ref clblasSsbmv function.
 */


/**
 * @brief Matrix-vector product with a symmetric banded matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasSsbmv() function otherwise.
 *
 * @ingroup SBMV
 */
clblasStatus
clblasDsbmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup HBMV HBMV  - Hermitian banded matrix-vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a hermitian banded matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HBMV
 */
clblasStatus
clblasChbmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_chbmv.c
 * This is an example of how to use the @ref clblasChbmv function.
 */


/**
 * @brief Matrix-vector product with a hermitian banded matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasChbmv() function otherwise.
 *
 * @ingroup HBMV
 */
clblasStatus
clblasZhbmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup TBSV TBSV  - Solving triangular banded matrix
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief solving triangular banded matrix problems with float elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clblasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TBSV
 */
 clblasStatus
clblasStbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_stbsv.c
 * This is an example of how to use the @ref clblasStbsv function.
 */


/**
 * @brief solving triangular banded matrix problems with double elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasStbsv() function otherwise.
 *
 * @ingroup TBSV
 */
clblasStatus
clblasDtbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief solving triangular banded matrix problems with float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clblasStbsv() function.
 *
 * @ingroup TBSV
 */
clblasStatus
clblasCtbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief solving triangular banded matrix problems with double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clblasDtbsv() function.
 *
 * @ingroup TBSV
 */
clblasStatus
clblasZtbsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup BLAS3 BLAS-3 functions
 *
 * The Level 3 Basic Linear Algebra Subprograms are funcions that perform
 * matrix-matrix operations.
 */
/*@{*/
/*@}*/

/**
 * @defgroup GEMM GEMM - General matrix-matrix multiplication
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Matrix-matrix product of general rectangular matrices with float
 *        elements. Extended version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b K when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clblasColumnMajor.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b K
 *                      when it is set to \b clblasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in]  offC     Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when
 *                      it is set to \b clblasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as clblasSgemm() otherwise.
 *
 * @ingroup GEMM
 */
clblasStatus
clblasSgemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sgemm.c
 * This is an example of how to use the @ref clblasSgemmEx function.
 */

/**
 * @brief Matrix-matrix product of general rectangular matrices with double
 *        elements. Extended version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clblasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clblasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offC      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clblasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clblasInvalidValue if either \b offA, \b offB or offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clblasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
clblasStatus
clblasDgemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-matrix product of general rectangular matrices with float
 *        complex elements. Extended version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clblasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clblasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offC      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clblasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if either \b offA, \b offB or offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clblasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
clblasStatus
clblasCgemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-matrix product of general rectangular matrices with double
 *        complex elements. Exteneded version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clblasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clblasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offC      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clblasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clblasInvalidValue if either \b offA, \b offB or offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clblasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
clblasStatus
clblasZgemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup TRMM TRMM - Triangular matrix-matrix multiplication
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Multiplying a matrix by a triangular matrix with float elements.
 *        Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clblasLeft,\n or less than \b N when it is set
 *                      to \b clblasRight.
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or not less than \b M
 *                      when it is set to \b clblasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clblasStrmm() otherwise.
 *
 * @ingroup TRMM
 */
clblasStatus
clblasStrmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_strmm.c
 * This is an example of how to use the @ref clblasStrmmEx function.
 */

/**
 * @brief Multiplying a matrix by a triangular matrix with double elements.
 *        Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clblasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clblasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clblasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clblasStrmm() function otherwise.
 *
 * @ingroup TRMM
 */
clblasStatus
clblasDtrmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Multiplying a matrix by a triangular matrix with float complex
 *        elements. Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clblasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clblasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clblasStrmm() otherwise.
 *
 * @ingroup TRMM
 */
clblasStatus
clblasCtrmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Multiplying a matrix by a triangular matrix with double complex
 *        elements. Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha A B \f$
 *   - \f$ B \leftarrow \alpha A^T B \f$
 *   - \f$ B \leftarrow \alpha B A \f$
 *   - \f$ B \leftarrow \alpha B A^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clblasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clblasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clblasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clblasStrmm() function otherwise.
 *
 * @ingroup TRMM
 */
clblasStatus
clblasZtrmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup TRSM TRSM - Solving triangular systems of equations
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and float elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clblasLeft,\n or less than \b N
 *                      when it is set to \b clblasRight.
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M
 *                      when it is set to \b clblasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clblasStrsm() otherwise.
 *
 * @ingroup TRSM
 */
clblasStatus
clblasStrsm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_strsm.c
 * This is an example of how to use the @ref clblasStrsmEx function.
 */

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and double elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clblasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clblasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clblasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clblasStrsm() function otherwise.
 *
 * @ingroup TRSM
 */
clblasStatus
clblasDtrsm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and float complex elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clblasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clblasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clblasStrsm() otherwise.
 *
 * @ingroup TRSM
 */
clblasStatus
clblasCtrsm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and double complex elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha A^{-1} B \f$
 *   - \f$ B \leftarrow \alpha A^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B A^{-1} \f$
 *   - \f$ B \leftarrow \alpha B A^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrix \b B.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clblasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clblasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clblasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clblasStrsm() function otherwise
 *
 * @ingroup TRSM
 */
clblasStatus
clblasZtrsm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup SYRK SYRK - Symmetric rank-k update of a matrix
 * @ingroup BLAS3
 */

/*@{*/

/**
 * @brief Rank-k update of a symmetric matrix with float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clblasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clblasStatus
clblasSsyrk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_ssyrk.c
 * This is an example of how to use the @ref clblasSsyrkEx function.
 */

/**
 * @brief Rank-k update of a symmetric matrix with double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clblasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clblasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clblasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clblasStatus
clblasDsyrk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-k update of a symmetric matrix with complex float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clblasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - \b clblasInvalidValue if \b transA is set to \ref clblasConjTrans.
 *   - the same error codes as the clblasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clblasStatus
clblasCsyrk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    FloatComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-k update of a symmetric matrix with complex double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clblasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *         point arithmetic with double precision;
 *   - \b clblasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - \b clblasInvalidValue if \b transA is set to \ref clblasConjTrans.
 *   - the same error codes as the clblasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clblasStatus
clblasZsyrk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    DoubleComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup SYR2K SYR2K - Symmetric rank-2k update to a matrix
 * @ingroup BLAS3
 */

/*@{*/

/**
 * @brief Rank-2k update of a symmetric matrix with float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be less
 *                       than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be less
 *                       less than \b K if \b B matches to the op(\b B) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clblasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clblasStatus
clblasSsyr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_ssyr2k.c
 * This is an example of how to use the @ref clblasSsyr2kEx function.
 */

/**
 * @brief Rank-2k update of a symmetric matrix with double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clblasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clblasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clblasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clblasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clblasStatus
clblasDsyr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-2k update of a symmetric matrix with complex float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clblasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clblasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - \b clblasInvalidValue if \b transAB is set to \ref clblasConjTrans.
 *   - the same error codes as the clblasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clblasStatus
clblasCsyr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-2k update of a symmetric matrix with complex double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clblasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clblasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clblasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - \b clblasInvalidValue if \b transAB is set to \ref clblasConjTrans.
 *   - the same error codes as the clblasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clblasStatus
clblasZsyr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup SYMM SYMM  - Symmetric matrix-matrix multiply
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with float
 * elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clblasLeft,\n or less than \b N when the
 *                      parameter is set to \b clblasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M
 *                      when it is set to \b clblasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when
 *                      it is set to \b clblasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b M or \b N is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clblasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clblasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYMM
 */
clblasStatus
clblasSsymm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_ssymm.c
 * This is an example of how to use the @ref clblasSsymm function.
 */


/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with double
 * elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clblasLeft,\n or less than \b N when the
 *                      parameter is set to \b clblasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M
 *                      when it is set to \b clblasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when
 *                      it is set to \b clblasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasSsymm() function otherwise.
 *
 * @ingroup SYMM
 */
clblasStatus
clblasDsymm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with
 * float-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clblasLeft,\n or less than \b N when the
 *                      parameter is set to \b clblasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M
 *                      when it is set to \b clblasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when
 *                      it is set to \b clblasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return The same result as the clblasSsymm() function.
 *
 * @ingroup SYMM
 */
clblasStatus
clblasCsymm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with
 * double-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clblasLeft,\n or less than \b N when the
 *                      parameter is set to \b clblasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M
 *                      when it is set to \b clblasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when
 *                      it is set to \b clblasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return The same result as the clblasDsymm() function.
 *
 * @ingroup SYMM
 */
clblasStatus
clblasZsymm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup HEMM HEMM  - Hermitian matrix-matrix multiplication
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Matrix-matrix product of hermitian rectangular matrices with
 * float-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clblasLeft,\n or less than \b N when the
 *                      parameter is set to \b clblasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M
 *                      when it is set to \b clblasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when
 *                      it is set to \b clblasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - \b M or \b N is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clblasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clblasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clblasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clblasCompilerNotAvailable if a compiler is not available;
 *   - \b clblasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HEMM
 */
clblasStatus
clblasChemm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_chemm.cpp
 * This is an example of how to use the @ref clblasChemm function.
 */


/**
 * @brief Matrix-matrix product of hermitian rectangular matrices with
 * double-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrices \b B and \b C.
 * @param[in] N         Number of columns in matrices \b B and \b C.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clblasLeft,\n or less than \b N when the
 *                      parameter is set to \b clblasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M
 *                      when it is set to \b clblasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clblasRowMajor,\n or less than \b M when
 *                      it is set to \b clblasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasChemm() function otherwise.
 *
 * @ingroup HEMM
 */
clblasStatus
clblasZhemm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup HERK HERK  - Hermitian rank-k update to a matrix
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Rank-k update of a hermitian matrix with float-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b A or \b C object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs to
 *     was released.
 *
 * @ingroup HERK
 */
clblasStatus
clblasCherk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_cherk.cpp
 * This is an example of how to use the @ref clblasCherk function.
 */


/**
 * @brief Rank-k update of a hermitian matrix with double-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasCherk() function otherwise.
 *
 * @ingroup HERK
 */
clblasStatus
clblasZherk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup HER2K HER2K  - Hermitian rank-2k update to a matrix
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Rank-2k update of a hermitian matrix with float-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^H + conj( \alpha ) B A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H B + conj( \alpha ) B^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] trans      How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vice-versa for transpose case.
 * @param[in] B          Buffer object storing the matrix \b B.
 * @param[in] offb       Offset in number of elements for the first element in matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be
 *                       less than \b K if \b B is
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vice-versa for transpose case
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasNotInitialized if clblasSetup() was not called;
 *   - \b clblasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clblasInvalidMemObject if either \b A , \b B or \b C object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clblasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clblasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clblasInvalidContext if a context a passed command queue belongs to
 *     was released.
 *
 * @ingroup HER2K
 */
clblasStatus
clblasCher2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_cher2k.c
 * This is an example of how to use the @ref clblasCher2k function.
 */


/**
 * @brief Rank-2k update of a hermitian matrix with double-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^H + conj( \alpha ) B A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H B + conj( \alpha ) B^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] trans      How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A is
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vice-versa for transpose case.
 * @param[in] B          Buffer object storing the matrix \b B.
 * @param[in] offb       Offset in number of elements for the first element in matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be
 *                       less than \b K if B is
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vice-versa for transpose case.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clblasSuccess on success;
 *   - \b clblasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clblasCher2k() function otherwise.
 *
 * @ingroup HER2K
 */
clblasStatus
clblasZher2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/




#ifdef __cplusplus
}      /* extern "C" { */
#endif

#endif /* CLBLAS_H_ */
