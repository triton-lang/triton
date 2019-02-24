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
 
#if !defined(CUSPARSE_H_)
#define CUSPARSE_H_


#ifndef CUSPARSEAPI
#ifdef _WIN32
#define CUSPARSEAPI __stdcall
#else
#define CUSPARSEAPI 
#endif
#endif

#include "driver_types.h"
#include "cuComplex.h"   /* import complex data type */

#include "cuda_fp16.h"

#include "library_types.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/* CUSPARSE status type returns */
typedef enum{
    CUSPARSE_STATUS_SUCCESS=0,
    CUSPARSE_STATUS_NOT_INITIALIZED=1,
    CUSPARSE_STATUS_ALLOC_FAILED=2,
    CUSPARSE_STATUS_INVALID_VALUE=3,
    CUSPARSE_STATUS_ARCH_MISMATCH=4,
    CUSPARSE_STATUS_MAPPING_ERROR=5,
    CUSPARSE_STATUS_EXECUTION_FAILED=6,
    CUSPARSE_STATUS_INTERNAL_ERROR=7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
    CUSPARSE_STATUS_ZERO_PIVOT=9
} cusparseStatus_t;

/* Opaque structure holding CUSPARSE library context */
struct cusparseContext;
typedef struct cusparseContext *cusparseHandle_t;

/* Opaque structure holding the matrix descriptor */
struct cusparseMatDescr;
typedef struct cusparseMatDescr *cusparseMatDescr_t;

/* Opaque structure holding the sparse triangular solve information */
struct cusparseSolveAnalysisInfo;
typedef struct cusparseSolveAnalysisInfo *cusparseSolveAnalysisInfo_t;

/* Opaque structures holding the sparse triangular solve information */
struct csrsv2Info;
typedef struct csrsv2Info *csrsv2Info_t;

struct bsrsv2Info;
typedef struct bsrsv2Info *bsrsv2Info_t;

struct bsrsm2Info;
typedef struct bsrsm2Info *bsrsm2Info_t;

/* Opaque structures holding incomplete Cholesky information */
struct csric02Info;
typedef struct csric02Info *csric02Info_t;

struct bsric02Info;
typedef struct bsric02Info *bsric02Info_t;

/* Opaque structures holding incomplete LU information */
struct csrilu02Info;
typedef struct csrilu02Info *csrilu02Info_t;

struct bsrilu02Info;
typedef struct bsrilu02Info *bsrilu02Info_t;

/* Opaque structures holding the hybrid (HYB) storage information */
struct cusparseHybMat;
typedef struct cusparseHybMat *cusparseHybMat_t;

/* Opaque structures holding sparse gemm information */
struct csrgemm2Info;
typedef struct csrgemm2Info *csrgemm2Info_t;

/* Opaque structure holding the sorting information */
struct csru2csrInfo;
typedef struct csru2csrInfo *csru2csrInfo_t;

/* Opaque structure holding the coloring information */
struct cusparseColorInfo;
typedef struct cusparseColorInfo *cusparseColorInfo_t;

/* Opaque structure holding the prune information */
struct pruneInfo;
typedef struct pruneInfo *pruneInfo_t;

/* Types definitions */
typedef enum { 
    CUSPARSE_POINTER_MODE_HOST = 0,  
    CUSPARSE_POINTER_MODE_DEVICE = 1        
} cusparsePointerMode_t;

typedef enum { 
    CUSPARSE_ACTION_SYMBOLIC = 0,  
    CUSPARSE_ACTION_NUMERIC = 1        
} cusparseAction_t;

typedef enum {
    CUSPARSE_MATRIX_TYPE_GENERAL = 0, 
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,     
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2, 
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3 
} cusparseMatrixType_t;

typedef enum {
    CUSPARSE_FILL_MODE_LOWER = 0, 
    CUSPARSE_FILL_MODE_UPPER = 1
} cusparseFillMode_t;

typedef enum {
    CUSPARSE_DIAG_TYPE_NON_UNIT = 0, 
    CUSPARSE_DIAG_TYPE_UNIT = 1
} cusparseDiagType_t; 

typedef enum {
    CUSPARSE_INDEX_BASE_ZERO = 0, 
    CUSPARSE_INDEX_BASE_ONE = 1
} cusparseIndexBase_t;

typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,  
    CUSPARSE_OPERATION_TRANSPOSE = 1,  
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2  
} cusparseOperation_t;

typedef enum {
    CUSPARSE_DIRECTION_ROW = 0,  
    CUSPARSE_DIRECTION_COLUMN = 1  
} cusparseDirection_t;

typedef enum {
    CUSPARSE_HYB_PARTITION_AUTO = 0,  // automatically decide how to split the data into regular/irregular part
    CUSPARSE_HYB_PARTITION_USER = 1,  // store data into regular part up to a user specified treshhold
    CUSPARSE_HYB_PARTITION_MAX = 2    // store all data in the regular part
} cusparseHybPartition_t;

// used in csrsv2, csric02, and csrilu02
typedef enum {
    CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0, // no level information is generated, only reports structural zero.
    CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1  
} cusparseSolvePolicy_t;

typedef enum {
    CUSPARSE_SIDE_LEFT =0,
    CUSPARSE_SIDE_RIGHT=1
} cusparseSideMode_t;

typedef enum {
    CUSPARSE_COLOR_ALG0 = 0, // default
    CUSPARSE_COLOR_ALG1 = 1
} cusparseColorAlg_t;

typedef enum {
    CUSPARSE_ALG0 = 0, //default, naive
    CUSPARSE_ALG1 = 1  //merge path
} cusparseAlgMode_t;

/* CUSPARSE initialization and managment routines */
cusparseStatus_t CUSPARSEAPI cusparseCreate(cusparseHandle_t *handle);
cusparseStatus_t CUSPARSEAPI cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t CUSPARSEAPI cusparseGetVersion(cusparseHandle_t handle, int *version);
cusparseStatus_t CUSPARSEAPI cusparseGetProperty(libraryPropertyType type, int *value);
cusparseStatus_t CUSPARSEAPI cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId); 
cusparseStatus_t CUSPARSEAPI cusparseGetStream(cusparseHandle_t handle, cudaStream_t *streamId);


/* CUSPARSE type creation, destruction, set and get routines */
cusparseStatus_t CUSPARSEAPI cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t *mode);
cusparseStatus_t CUSPARSEAPI cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode);

/* sparse matrix descriptor */
/* When the matrix descriptor is created, its fields are initialized to: 
   CUSPARSE_MATRIX_TYPE_GENERAL
   CUSPARSE_INDEX_BASE_ZERO
   All other fields are uninitialized
*/                                   
cusparseStatus_t CUSPARSEAPI cusparseCreateMatDescr(cusparseMatDescr_t *descrA);
cusparseStatus_t CUSPARSEAPI cusparseDestroyMatDescr (cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI cusparseCopyMatDescr(cusparseMatDescr_t dest, const cusparseMatDescr_t src);

cusparseStatus_t CUSPARSEAPI cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type);
cusparseMatrixType_t CUSPARSEAPI cusparseGetMatType(const cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode);
cusparseFillMode_t CUSPARSEAPI cusparseGetMatFillMode(const cusparseMatDescr_t descrA);
 
cusparseStatus_t CUSPARSEAPI cusparseSetMatDiagType(cusparseMatDescr_t  descrA, cusparseDiagType_t diagType);
cusparseDiagType_t CUSPARSEAPI cusparseGetMatDiagType(const cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base);
cusparseIndexBase_t CUSPARSEAPI cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);

/* sparse triangular solve and incomplete-LU and Cholesky (algorithm 1) */
cusparseStatus_t CUSPARSEAPI cusparseCreateSolveAnalysisInfo(cusparseSolveAnalysisInfo_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo_t info);
cusparseStatus_t CUSPARSEAPI cusparseGetLevelInfo(cusparseHandle_t handle, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  int *nlevels, 
                                                  int **levelPtr, 
                                                  int **levelInd);

/* sparse triangular solve (algorithm 2) */
cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsv2Info(csrsv2Info_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsv2Info(csrsv2Info_t info);

/* incomplete Cholesky (algorithm 2)*/
cusparseStatus_t CUSPARSEAPI cusparseCreateCsric02Info(csric02Info_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroyCsric02Info(csric02Info_t info);

cusparseStatus_t CUSPARSEAPI cusparseCreateBsric02Info(bsric02Info_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroyBsric02Info(bsric02Info_t info);

/* incomplete LU (algorithm 2) */
cusparseStatus_t CUSPARSEAPI cusparseCreateCsrilu02Info(csrilu02Info_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrilu02Info(csrilu02Info_t info);

cusparseStatus_t CUSPARSEAPI cusparseCreateBsrilu02Info(bsrilu02Info_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrilu02Info(bsrilu02Info_t info);

/* block-CSR triangular solve (algorithm 2) */
cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsv2Info(bsrsv2Info_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsv2Info(bsrsv2Info_t info);

cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsm2Info(bsrsm2Info_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsm2Info(bsrsm2Info_t info);

/* hybrid (HYB) format */
cusparseStatus_t CUSPARSEAPI cusparseCreateHybMat(cusparseHybMat_t *hybA);
cusparseStatus_t CUSPARSEAPI cusparseDestroyHybMat(cusparseHybMat_t hybA);

/* sorting information */
cusparseStatus_t CUSPARSEAPI cusparseCreateCsru2csrInfo(csru2csrInfo_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroyCsru2csrInfo(csru2csrInfo_t info);

/* coloring info */
cusparseStatus_t CUSPARSEAPI cusparseCreateColorInfo(cusparseColorInfo_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroyColorInfo(cusparseColorInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseSetColorAlgs(cusparseColorInfo_t info, cusparseColorAlg_t alg);
cusparseStatus_t CUSPARSEAPI cusparseGetColorAlgs(cusparseColorInfo_t info, cusparseColorAlg_t *alg);

/* prune information */
cusparseStatus_t CUSPARSEAPI cusparseCreatePruneInfo(pruneInfo_t *info);

cusparseStatus_t CUSPARSEAPI cusparseDestroyPruneInfo(pruneInfo_t info);


/* --- Sparse Level 1 routines --- */

/* Description: Addition of a scalar multiple of a sparse vector x  
   and a dense vector y. */ 
cusparseStatus_t CUSPARSEAPI cusparseSaxpyi(cusparseHandle_t handle, 
                                            int nnz, 
                                            const float *alpha, 
                                            const float *xVal, 
                                            const int *xInd, 
                                            float *y, 
                                            cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDaxpyi(cusparseHandle_t handle, 
                                            int nnz, 
                                            const double *alpha, 
                                            const double *xVal, 
                                            const int *xInd, 
                                            double *y, 
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCaxpyi(cusparseHandle_t handle, 
                                            int nnz, 
                                            const cuComplex *alpha, 
                                            const cuComplex *xVal, 
                                            const int *xInd, 
                                            cuComplex *y, 
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZaxpyi(cusparseHandle_t handle, 
                                            int nnz, 
                                            const cuDoubleComplex *alpha, 
                                            const cuDoubleComplex *xVal, 
                                            const int *xInd, 
                                            cuDoubleComplex *y, 
                                            cusparseIndexBase_t idxBase);

/* Description: dot product of a sparse vector x and a dense vector y. */
cusparseStatus_t CUSPARSEAPI cusparseSdoti(cusparseHandle_t handle,  
                                           int nnz, 
                                           const float *xVal, 
                                           const int *xInd, 
                                           const float *y,
                                           float *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseDdoti(cusparseHandle_t handle, 
                                           int nnz, 
                                           const double *xVal, 
                                           const int *xInd, 
                                           const double *y, 
                                           double *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCdoti(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuComplex *xVal, 
                                           const int *xInd, 
                                           const cuComplex *y, 
                                           cuComplex *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZdoti(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuDoubleComplex *xVal, 
                                           const int *xInd, 
                                           const cuDoubleComplex *y, 
                                           cuDoubleComplex *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

/* Description: dot product of complex conjugate of a sparse vector x
   and a dense vector y. */
cusparseStatus_t CUSPARSEAPI cusparseCdotci(cusparseHandle_t handle, 
                                            int nnz, 
                                            const cuComplex *xVal, 
                                            const int *xInd, 
                                            const cuComplex *y, 
                                            cuComplex *resultDevHostPtr,
                                            cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseZdotci(cusparseHandle_t handle, 
                                            int nnz, 
                                            const cuDoubleComplex *xVal, 
                                            const int *xInd, 
                                            const cuDoubleComplex *y, 
                                            cuDoubleComplex *resultDevHostPtr,
                                            cusparseIndexBase_t idxBase);


/* Description: Gather of non-zero elements from dense vector y into 
   sparse vector x. */
cusparseStatus_t CUSPARSEAPI cusparseSgthr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const float *y, 
                                           float *xVal, 
                                           const int *xInd, 
                                           cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDgthr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const double *y, 
                                           double *xVal, 
                                           const int *xInd, 
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCgthr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuComplex *y, 
                                           cuComplex *xVal, 
                                           const int *xInd, 
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZgthr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuDoubleComplex *y, 
                                           cuDoubleComplex *xVal, 
                                           const int *xInd, 
                                           cusparseIndexBase_t idxBase);

/* Description: Gather of non-zero elements from desne vector y into 
   sparse vector x (also replacing these elements in y by zeros). */
cusparseStatus_t CUSPARSEAPI cusparseSgthrz(cusparseHandle_t handle, 
                                            int nnz, 
                                            float *y, 
                                            float *xVal, 
                                            const int *xInd, 
                                            cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDgthrz(cusparseHandle_t handle, 
                                            int nnz, 
                                            double *y, 
                                            double *xVal, 
                                            const int *xInd, 
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCgthrz(cusparseHandle_t handle, 
                                            int nnz, 
                                            cuComplex *y, 
                                            cuComplex *xVal, 
                                            const int *xInd, 
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZgthrz(cusparseHandle_t handle, 
                                            int nnz, 
                                            cuDoubleComplex *y, 
                                            cuDoubleComplex *xVal, 
                                            const int *xInd, 
                                            cusparseIndexBase_t idxBase);

/* Description: Scatter of elements of the sparse vector x into 
   dense vector y. */
cusparseStatus_t CUSPARSEAPI cusparseSsctr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const float *xVal, 
                                           const int *xInd, 
                                           float *y, 
                                           cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDsctr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const double *xVal, 
                                           const int *xInd, 
                                           double *y, 
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCsctr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuComplex *xVal, 
                                           const int *xInd, 
                                           cuComplex *y, 
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZsctr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuDoubleComplex *xVal, 
                                           const int *xInd, 
                                           cuDoubleComplex *y, 
                                           cusparseIndexBase_t idxBase);

/* Description: Givens rotation, where c and s are cosine and sine, 
   x and y are sparse and dense vectors, respectively. */
cusparseStatus_t CUSPARSEAPI cusparseSroti(cusparseHandle_t handle, 
                                              int nnz, 
                                              float *xVal, 
                                              const int *xInd, 
                                              float *y, 
                                              const float *c, 
                                              const float *s, 
                                              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseDroti(cusparseHandle_t handle, 
                                              int nnz, 
                                              double *xVal, 
                                              const int *xInd, 
                                              double *y, 
                                              const double *c, 
                                              const double *s, 
                                              cusparseIndexBase_t idxBase);


/* --- Sparse Level 2 routines --- */

cusparseStatus_t  CUSPARSEAPI cusparseSgemvi(cusparseHandle_t handle,
                                    cusparseOperation_t transA,
                                    int m,
                                    int n,
                                    const float *alpha, /* host or device pointer */
                                    const float *A,
                                    int lda,
                                    int nnz,
                                    const float *xVal,
                                    const int *xInd,
                                    const float *beta, /* host or device pointer */
                                    float *y,
                                    cusparseIndexBase_t   idxBase,
                                    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseSgemvi_bufferSize( cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int n,
    int nnz,
    int *pBufferSize);

cusparseStatus_t  CUSPARSEAPI cusparseDgemvi(cusparseHandle_t handle,
                                    cusparseOperation_t transA,
                                    int m,
                                    int n,
                                    const double *alpha, /* host or device pointer */
                                    const double *A,
                                    int lda,
                                    int nnz,
                                    const double *xVal,
                                    const int *xInd,
                                    const double *beta, /* host or device pointer */
                                    double *y,
                                    cusparseIndexBase_t   idxBase,
                                    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDgemvi_bufferSize( cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int n,
    int nnz,
    int *pBufferSize);

cusparseStatus_t  CUSPARSEAPI cusparseCgemvi(cusparseHandle_t handle,
                                    cusparseOperation_t transA,
                                    int m,
                                    int n,
                                    const cuComplex *alpha, /* host or device pointer */
                                    const cuComplex *A,
                                    int lda,
                                    int nnz,
                                    const cuComplex *xVal,
                                    const int *xInd,
                                    const cuComplex *beta, /* host or device pointer */
                                    cuComplex *y,
                                    cusparseIndexBase_t   idxBase,
                                    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCgemvi_bufferSize( cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int n,
    int nnz,
    int *pBufferSize);

cusparseStatus_t  CUSPARSEAPI cusparseZgemvi(cusparseHandle_t handle,
                                    cusparseOperation_t transA,
                                    int m,
                                    int n,
                                    const cuDoubleComplex *alpha, /* host or device pointer */
                                    const cuDoubleComplex *A,
                                    int lda,
                                    int nnz,
                                    const cuDoubleComplex *xVal,
                                    const int *xInd,
                                    const cuDoubleComplex *beta, /* host or device pointer */
                                    cuDoubleComplex *y,
                                    cusparseIndexBase_t   idxBase,
                                    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZgemvi_bufferSize( cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int n,
    int nnz,
    int *pBufferSize);


/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in CSR storage format, x and y are dense vectors. */ 
cusparseStatus_t CUSPARSEAPI cusparseScsrmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const float *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const float *x, 
                                            const float *beta, 
                                            float *y);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsrmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const double *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const double *x, 
                                            const double *beta,  
                                            double *y);

cusparseStatus_t CUSPARSEAPI cusparseCcsrmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n,
                                            int nnz,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const cuComplex *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const cuComplex *x, 
                                            const cuComplex *beta, 
                                            cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZcsrmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const cuDoubleComplex *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const cuDoubleComplex *x, 
                                            const cuDoubleComplex *beta, 
                                            cuDoubleComplex *y);  

//Returns number of bytes
cusparseStatus_t CUSPARSEAPI cusparseCsrmvEx_bufferSize(cusparseHandle_t handle, 
                                                        cusparseAlgMode_t alg,
                                                        cusparseOperation_t transA, 
                                                        int m, 
                                                        int n, 
                                                        int nnz,
                                                        const void *alpha,
                                                        cudaDataType alphatype,
                                                        const cusparseMatDescr_t descrA,
                                                        const void *csrValA,
                                                        cudaDataType csrValAtype,
                                                        const int *csrRowPtrA,
                                                        const int *csrColIndA,
                                                        const void *x,
                                                        cudaDataType xtype,
                                                        const void *beta,
                                                        cudaDataType betatype,
                                                        void *y,
                                                        cudaDataType ytype,
                                                        cudaDataType executiontype,
                                                        size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCsrmvEx(cusparseHandle_t handle, 
                                             cusparseAlgMode_t alg,
                                             cusparseOperation_t transA, 
                                             int m, 
                                             int n, 
                                             int nnz,
                                             const void *alpha,
                                             cudaDataType alphatype,
                                             const cusparseMatDescr_t descrA,
                                             const void *csrValA,
                                             cudaDataType csrValAtype,
                                             const int *csrRowPtrA,
                                             const int *csrColIndA,
                                             const void *x,
                                             cudaDataType xtype,
                                             const void *beta,
                                             cudaDataType betatype,
                                             void *y,
                                             cudaDataType ytype,
                                             cudaDataType executiontype,
                                             void* buffer);

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in CSR storage format, x and y are dense vectors
   using a Merge Path load-balancing implementation. */ 
   cusparseStatus_t CUSPARSEAPI cusparseScsrmv_mp(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const float *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const float *x, 
                                            const float *beta, 
                                            float *y);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsrmv_mp(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const double *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const double *x, 
                                            const double *beta,  
                                            double *y);

cusparseStatus_t CUSPARSEAPI cusparseCcsrmv_mp(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n,
                                            int nnz,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const cuComplex *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const cuComplex *x, 
                                            const cuComplex *beta, 
                                            cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZcsrmv_mp(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const cuDoubleComplex *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const cuDoubleComplex *x, 
                                            const cuDoubleComplex *beta, 
                                            cuDoubleComplex *y);  


/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in HYB storage format, x and y are dense vectors. */    
cusparseStatus_t CUSPARSEAPI cusparseShybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const float *x,
                                            const float *beta,
                                            float *y);

cusparseStatus_t CUSPARSEAPI cusparseDhybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const double *x,
                                            const double *beta,
                                            double *y);

cusparseStatus_t CUSPARSEAPI cusparseChybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const cuComplex *x,
                                            const cuComplex *beta,
                                            cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZhybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const cuDoubleComplex *x,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *y);
    
/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in BSR storage format, x and y are dense vectors. */
cusparseStatus_t CUSPARSEAPI cusparseSbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const float *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            int  blockDim,
                                            const float *x,
                                            const float *beta,
                                            float *y);

cusparseStatus_t CUSPARSEAPI cusparseDbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const double *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            int  blockDim,
                                            const double *x,
                                            const double *beta,
                                            double *y);

cusparseStatus_t CUSPARSEAPI cusparseCbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            int  blockDim,
                                            const cuComplex *x,
                                            const cuComplex *beta,
                                            cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuDoubleComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            int  blockDim,
                                            const cuDoubleComplex *x,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *y);

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in extended BSR storage format, x and y are dense 
   vectors. */
cusparseStatus_t CUSPARSEAPI cusparseSbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const float *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const float *bsrSortedValA,
                                             const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA,
                                             const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA,
                                             int  blockDim,
                                             const float *x,
                                             const float *beta,
                                             float *y);


cusparseStatus_t CUSPARSEAPI cusparseDbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const double *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const double *bsrSortedValA,
                                             const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA,
                                             const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA,
                                             int  blockDim,
                                             const double *x,
                                             const double *beta,
                                             double *y);
    
cusparseStatus_t CUSPARSEAPI cusparseCbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const cuComplex *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const cuComplex *bsrSortedValA,
                                             const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA,
                                             const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA,
                                             int  blockDim,
                                             const cuComplex *x,
                                             const cuComplex *beta,
                                             cuComplex *y);


cusparseStatus_t CUSPARSEAPI cusparseZbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const cuDoubleComplex *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const cuDoubleComplex *bsrSortedValA,
                                             const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA,
                                             const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA,
                                             int  blockDim,
                                             const cuDoubleComplex *x,
                                             const cuDoubleComplex *beta,
                                             cuDoubleComplex *y);

/* Description: Solution of triangular linear system op(A) * x = alpha * f, 
   where A is a sparse matrix in CSR storage format, rhs f and solution x 
   are dense vectors. This routine implements algorithm 1 for the solve. */     
cusparseStatus_t CUSPARSEAPI cusparseCsrsv_analysisEx(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const void *csrSortedValA,
                                                     cudaDataType csrSortedValAtype,
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     cusparseSolveAnalysisInfo_t info,
                                                     cudaDataType executiontype);

cusparseStatus_t CUSPARSEAPI cusparseScsrsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const float *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const double *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const cuComplex *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const cuDoubleComplex *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     cusparseSolveAnalysisInfo_t info); 

cusparseStatus_t CUSPARSEAPI cusparseCsrsv_solveEx(cusparseHandle_t handle, 
                                                   cusparseOperation_t transA, 
                                                   int m,
                                                   const void *alpha, 
                                                   cudaDataType alphatype,
                                                   const cusparseMatDescr_t descrA, 
                                                   const void *csrSortedValA, 
                                                   cudaDataType csrSortedValAtype,
                                                   const int *csrSortedRowPtrA, 
                                                   const int *csrSortedColIndA, 
                                                   cusparseSolveAnalysisInfo_t info, 
                                                   const void *f, 
                                                   cudaDataType ftype,
                                                   void *x,
                                                   cudaDataType xtype,
                                                   cudaDataType executiontype);
  
cusparseStatus_t CUSPARSEAPI cusparseScsrsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m,
                                                  const float *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const float *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const float *f, 
                                                  float *x);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  const double *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const double *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const double *f, 
                                                  double *x);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  const cuComplex *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const cuComplex *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const cuComplex *f, 
                                                  cuComplex *x);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  const cuDoubleComplex *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const cuDoubleComplex *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const cuDoubleComplex *f, 
                                                  cuDoubleComplex *x);      

/* Description: Solution of triangular linear system op(A) * x = alpha * f, 
   where A is a sparse matrix in CSR storage format, rhs f and solution y 
   are dense vectors. This routine implements algorithm 1 for this problem. 
   Also, it provides a utility function to query size of buffer used. */
cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle,
                                                       csrsv2Info_t info,
                                                       int *position);

cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        float *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        double *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        cuComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        cuDoubleComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSizeExt(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        float *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSizeExt(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        double *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSizeExt(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        cuComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSizeExt(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        cuDoubleComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        size_t *pBufferSize);


cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const cusparseMatDescr_t descrA,
                                                      const float *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const cusparseMatDescr_t descrA,
                                                      const double *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const cusparseMatDescr_t descrA,
                                                      const cuComplex *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const cusparseMatDescr_t descrA,
                                                      const cuDoubleComplex *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_solve(cusparseHandle_t handle,
                                                   cusparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const float *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const float *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const float *f,
                                                   float *x,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_solve(cusparseHandle_t handle,
                                                   cusparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const double *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const double *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const double *f,
                                                   double *x,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_solve(cusparseHandle_t handle,
                                                   cusparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const cuComplex *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const cuComplex *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const cuComplex *f,
                                                   cuComplex *x,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_solve(cusparseHandle_t handle,
                                                   cusparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const cuDoubleComplex *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const cuDoubleComplex *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const cuDoubleComplex *f,
                                                   cuDoubleComplex *x,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);

/* Description: Solution of triangular linear system op(A) * x = alpha * f, 
   where A is a sparse matrix in block-CSR storage format, rhs f and solution y 
   are dense vectors. This routine implements algorithm 2 for this problem. 
   Also, it provides a utility function to query size of buffer used. */
cusparseStatus_t CUSPARSEAPI cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle,
                                                       bsrsv2Info_t info,
                                                       int *position);


cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        float *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockDim,
                                                        bsrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        double *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockDim,
                                                        bsrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        cuComplex *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockDim,
                                                        bsrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        cuDoubleComplex *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockDim,
                                                        bsrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSizeExt(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        float *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockSize,
                                                        bsrsv2Info_t info,
                                                        size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSizeExt(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        double *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockSize,
                                                        bsrsv2Info_t info,
                                                        size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSizeExt(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        cuComplex *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockSize,
                                                        bsrsv2Info_t info,
                                                        size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSizeExt(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        cuDoubleComplex *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockSize,
                                                        bsrsv2Info_t info,
                                                        size_t *pBufferSize);


cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseDirection_t dirA,
                                                      cusparseOperation_t transA,
                                                      int mb,
                                                      int nnzb,
                                                      const cusparseMatDescr_t descrA,
                                                      const float *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA,
                                                      const int *bsrSortedColIndA,
                                                      int blockDim,
                                                      bsrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseDirection_t dirA,
                                                      cusparseOperation_t transA,
                                                      int mb,
                                                      int nnzb,
                                                      const cusparseMatDescr_t descrA,
                                                      const double *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA,
                                                      const int *bsrSortedColIndA,
                                                      int blockDim,
                                                      bsrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseDirection_t dirA,
                                                      cusparseOperation_t transA,
                                                      int mb,
                                                      int nnzb,
                                                      const cusparseMatDescr_t descrA,
                                                      const cuComplex *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA,
                                                      const int *bsrSortedColIndA,
                                                      int blockDim,
                                                      bsrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseDirection_t dirA,
                                                      cusparseOperation_t transA,
                                                      int mb,
                                                      int nnzb,
                                                      const cusparseMatDescr_t descrA,
                                                      const cuDoubleComplex *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA,
                                                      const int *bsrSortedColIndA,
                                                      int blockDim,
                                                      bsrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_solve(cusparseHandle_t handle,
                                                   cusparseDirection_t dirA,
                                                   cusparseOperation_t transA,
                                                   int mb,
                                                   int nnzb,
                                                   const float *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const float *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA,
                                                   const int *bsrSortedColIndA,
                                                   int blockDim,
                                                   bsrsv2Info_t info,
                                                   const float *f,
                                                   float *x,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_solve(cusparseHandle_t handle,
                                                   cusparseDirection_t dirA,
                                                   cusparseOperation_t transA,
                                                   int mb,
                                                   int nnzb,
                                                   const double *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const double *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA,
                                                   const int *bsrSortedColIndA,
                                                   int blockDim,
                                                   bsrsv2Info_t info,
                                                   const double *f,
                                                   double *x, 
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_solve(cusparseHandle_t handle,
                                                   cusparseDirection_t dirA,
                                                   cusparseOperation_t transA,
                                                   int mb,
                                                   int nnzb,
                                                   const cuComplex *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const cuComplex *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA,
                                                   const int *bsrSortedColIndA,
                                                   int blockDim,
                                                   bsrsv2Info_t info,
                                                   const cuComplex *f,
                                                   cuComplex *x,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_solve(cusparseHandle_t handle,
                                                   cusparseDirection_t dirA,
                                                   cusparseOperation_t transA,
                                                   int mb,
                                                   int nnzb,
                                                   const cuDoubleComplex *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const cuDoubleComplex *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA,
                                                   const int *bsrSortedColIndA,
                                                   int blockDim,
                                                   bsrsv2Info_t info,
                                                   const cuDoubleComplex *f,
                                                   cuDoubleComplex *x,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);

/* Description: Solution of triangular linear system op(A) * x = alpha * f, 
   where A is a sparse matrix in HYB storage format, rhs f and solution x 
   are dense vectors. */
cusparseStatus_t CUSPARSEAPI cusparseShybsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     const cusparseMatDescr_t descrA, 
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDhybsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     const cusparseMatDescr_t descrA, 
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);
    
cusparseStatus_t CUSPARSEAPI cusparseChybsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     const cusparseMatDescr_t descrA, 
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZhybsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     const cusparseMatDescr_t descrA, 
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseShybsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t trans, 
                                                  const float *alpha, 
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const float *f,
                                                  float *x);

cusparseStatus_t CUSPARSEAPI cusparseChybsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t trans,
                                                  const cuComplex *alpha, 
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const cuComplex *f,
                                                  cuComplex *x);

cusparseStatus_t CUSPARSEAPI cusparseDhybsv_solve(cusparseHandle_t handle,
                                                  cusparseOperation_t trans,
                                                  const double *alpha, 
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA, 
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const double *f,
                                                  double *x);

cusparseStatus_t CUSPARSEAPI cusparseZhybsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t trans,
                                                  const cuDoubleComplex *alpha, 
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const cuDoubleComplex *f,
                                                  cuDoubleComplex *x);


/* --- Sparse Level 3 routines --- */           
 
/* Description: sparse - dense matrix multiplication C = alpha * op(A) * B  + beta * C, 
   where A is a sparse matrix in CSR format, B and C are dense tall matrices.  */                 
cusparseStatus_t CUSPARSEAPI cusparseScsrmm(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            int nnz,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const float  *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const float *B, 
                                            int ldb, 
                                            const float *beta, 
                                            float *C, 
                                            int ldc);
                     
cusparseStatus_t CUSPARSEAPI cusparseDcsrmm(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            int nnz,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const double *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const double *B, 
                                            int ldb, 
                                            const double *beta, 
                                            double *C, 
                                            int ldc);
                     
cusparseStatus_t CUSPARSEAPI cusparseCcsrmm(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            int nnz,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const cuComplex  *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const cuComplex *B, 
                                            int ldb, 
                                            const cuComplex *beta, 
                                            cuComplex *C, 
                                            int ldc);
                     
cusparseStatus_t CUSPARSEAPI cusparseZcsrmm(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            int nnz,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const cuDoubleComplex  *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const cuDoubleComplex *B, 
                                            int ldb, 
                                            const cuDoubleComplex *beta, 
                                            cuDoubleComplex *C, 
                                            int ldc);    

/* Description: sparse - dense matrix multiplication C = alpha * op(A) * B  + beta * C, 
   where A is a sparse matrix in CSR format, B and C are dense tall matrices.
   This routine allows transposition of matrix B, which may improve performance. */     
cusparseStatus_t CUSPARSEAPI cusparseScsrmm2(cusparseHandle_t handle,
                                             cusparseOperation_t transA,
                                             cusparseOperation_t transB,
                                             int m,
                                             int n,
                                             int k,
                                             int nnz,
                                             const float *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const float *csrSortedValA,
                                             const int *csrSortedRowPtrA,
                                             const int *csrSortedColIndA,
                                             const float *B,
                                             int ldb,
                                             const float *beta,
                                             float *C,
                                             int ldc);

cusparseStatus_t CUSPARSEAPI cusparseDcsrmm2(cusparseHandle_t handle,
                                             cusparseOperation_t transA,
                                             cusparseOperation_t transB,
                                             int m,
                                             int n,
                                             int k,
                                             int nnz,
                                             const double *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const double *csrSortedValA,
                                             const int *csrSortedRowPtrA,
                                             const int *csrSortedColIndA,
                                             const double *B,
                                             int ldb,
                                             const double *beta,
                                             double *C,
                                             int ldc);

cusparseStatus_t CUSPARSEAPI cusparseCcsrmm2(cusparseHandle_t handle,
                                             cusparseOperation_t transA,
                                             cusparseOperation_t transB,
                                             int m,
                                             int n,
                                             int k,
                                             int nnz,
                                             const cuComplex *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const cuComplex *csrSortedValA,
                                             const int *csrSortedRowPtrA,
                                             const int *csrSortedColIndA,
                                             const cuComplex *B,
                                             int ldb,
                                             const cuComplex *beta,
                                             cuComplex *C,
                                             int ldc);

cusparseStatus_t CUSPARSEAPI cusparseZcsrmm2(cusparseHandle_t handle,
                                             cusparseOperation_t transA,
                                             cusparseOperation_t transB,
                                             int m,
                                             int n,
                                             int k,
                                             int nnz,
                                             const cuDoubleComplex *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const cuDoubleComplex *csrSortedValA,
                                             const int *csrSortedRowPtrA,
                                             const int *csrSortedColIndA,
                                             const cuDoubleComplex *B,
                                             int ldb,
                                             const cuDoubleComplex *beta,
                                             cuDoubleComplex *C,
                                             int ldc);

/* Description: sparse - dense matrix multiplication C = alpha * op(A) * B  + beta * C, 
   where A is a sparse matrix in block-CSR format, B and C are dense tall matrices.
   This routine allows transposition of matrix B, which may improve performance. */ 
cusparseStatus_t CUSPARSEAPI cusparseSbsrmm(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            cusparseOperation_t transB,
                                            int mb,
                                            int n,
                                            int kb,
                                            int nnzb,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const float *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            const int  blockSize,
                                            const float *B,
                                            const int ldb,
                                            const float *beta,
                                            float *C,
                                            int ldc);

cusparseStatus_t CUSPARSEAPI cusparseDbsrmm(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            cusparseOperation_t transB,
                                            int mb,
                                            int n, 
                                            int kb, 
                                            int nnzb,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const double *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            const int  blockSize,
                                            const double *B,
                                            const int ldb,
                                            const double *beta,
                                            double *C,
                                            int ldc);

cusparseStatus_t CUSPARSEAPI cusparseCbsrmm(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            cusparseOperation_t transB,
                                            int mb,
                                            int n, 
                                            int kb, 
                                            int nnzb,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            const int  blockSize,
                                            const cuComplex *B,
                                            const int ldb,
                                            const cuComplex *beta,
                                            cuComplex *C,
                                            int ldc);

cusparseStatus_t CUSPARSEAPI cusparseZbsrmm(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            cusparseOperation_t transB,
                                            int mb,
                                            int n, 
                                            int kb, 
                                            int nnzb,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuDoubleComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            const int  blockSize,
                                            const cuDoubleComplex *B,
                                            const int ldb,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *C,
                                            int ldc);


/* Description: dense - sparse matrix multiplication C = alpha * A * B  + beta * C, 
   where A is column-major dense matrix, B is a sparse matrix in CSC format, 
   and C is column-major dense matrix. */
cusparseStatus_t  CUSPARSEAPI cusparseSgemmi(cusparseHandle_t handle,
                                             int m,
                                             int n,
					     int k,
					     int nnz, 
                                             const float *alpha, /* host or device pointer */
                                             const float *A,
                                             int lda,
                                             const float *cscValB,
					     const int *cscColPtrB, 
					     const int *cscRowIndB, 
                                             const float *beta, /* host or device pointer */
                                             float *C,
                                             int ldc);

cusparseStatus_t  CUSPARSEAPI cusparseDgemmi(cusparseHandle_t handle,
                                             int m,
                                             int n,
					     int k,
					     int nnz, 
                                             const double *alpha, /* host or device pointer */
                                             const double *A,
                                             int lda,
                                             const double *cscValB,
					     const int *cscColPtrB, 
					     const int *cscRowIndB, 
                                             const double *beta, /* host or device pointer */
                                             double *C,
                                             int ldc);

cusparseStatus_t  CUSPARSEAPI cusparseCgemmi(cusparseHandle_t handle,
                                             int m,
                                             int n,
					     int k,
					     int nnz, 
                                             const cuComplex *alpha, /* host or device pointer */
                                             const cuComplex *A,
                                             int lda,
                                             const cuComplex *cscValB,
					     const int *cscColPtrB, 
					     const int *cscRowIndB, 
                                             const cuComplex *beta, /* host or device pointer */
                                             cuComplex *C,
                                             int ldc);

cusparseStatus_t  CUSPARSEAPI cusparseZgemmi(cusparseHandle_t handle,
                                             int m,
                                             int n,
					     int k,
					     int nnz, 
                                             const cuDoubleComplex *alpha, /* host or device pointer */
                                             const cuDoubleComplex *A,
                                             int lda,
                                             const cuDoubleComplex *cscValB,
					     const int *cscColPtrB, 
					     const int *cscRowIndB, 
                                             const cuDoubleComplex *beta, /* host or device pointer */
                                             cuDoubleComplex *C,
                                             int ldc);


/* Description: Solution of triangular linear system op(A) * X = alpha * F, 
   with multiple right-hand-sides, where A is a sparse matrix in CSR storage 
   format, rhs F and solution X are dense tall matrices. 
   This routine implements algorithm 1 for this problem. */
cusparseStatus_t CUSPARSEAPI cusparseScsrsm_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const float *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsm_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const double *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsm_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const cuComplex *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsm_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const cuDoubleComplex *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     cusparseSolveAnalysisInfo_t info); 


cusparseStatus_t CUSPARSEAPI cusparseScsrsm_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m,
                                                  int n,
                                                  const float *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const float *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const float *F, 
                                                  int ldf,
                                                  float *X,
                                                  int ldx);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsm_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  int n,
                                                  const double *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const double *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const double *F, 
                                                  int ldf,
                                                  double *X,
                                                  int ldx);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsm_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  int n,
                                                  const cuComplex *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const cuComplex *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const cuComplex *F,
                                                  int ldf,
                                                  cuComplex *X,
                                                  int ldx);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsm_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  int n,
                                                  const cuDoubleComplex *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const cuDoubleComplex *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const cuDoubleComplex *F,
                                                  int ldf,
                                                  cuDoubleComplex *X,
                                                  int ldx);                                                                 
                    
/* Description: Solution of triangular linear system op(A) * X = alpha * F, 
   with multiple right-hand-sides, where A is a sparse matrix in CSR storage 
   format, rhs F and solution X are dense tall matrices.
   This routine implements algorithm 2 for this problem. */
cusparseStatus_t CUSPARSEAPI cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle,
                                                       bsrsm2Info_t info,
                                                       int *position);

cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_bufferSize(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        cusparseOperation_t transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        float *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockSize,
                                                        bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_bufferSize(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        cusparseOperation_t transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        double *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockSize,
                                                        bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_bufferSize(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        cusparseOperation_t transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        cuComplex *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockSize,
                                                        bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_bufferSize(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        cusparseOperation_t transA,
                                                        cusparseOperation_t transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        cuDoubleComplex *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockSize,
                                                        bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes);


cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_bufferSizeExt(cusparseHandle_t handle,
                                                           cusparseDirection_t dirA,
                                                           cusparseOperation_t transA,
                                                           cusparseOperation_t transB,
                                                           int mb,
                                                           int n,
                                                           int nnzb,
                                                           const cusparseMatDescr_t descrA,
                                                           float *bsrSortedVal,
                                                           const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd,
                                                           int blockSize,
                                                           bsrsm2Info_t info,
                                                           size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_bufferSizeExt(cusparseHandle_t handle,
                                                           cusparseDirection_t dirA,
                                                           cusparseOperation_t transA,
                                                           cusparseOperation_t transB,
                                                           int mb,
                                                           int n,
                                                           int nnzb,
                                                           const cusparseMatDescr_t descrA,
                                                           double *bsrSortedVal,
                                                           const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd,
                                                           int blockSize,
                                                           bsrsm2Info_t info,
                                                           size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_bufferSizeExt(cusparseHandle_t handle,
                                                           cusparseDirection_t dirA,
                                                           cusparseOperation_t transA,
                                                           cusparseOperation_t transB,
                                                           int mb,
                                                           int n,
                                                           int nnzb,
                                                           const cusparseMatDescr_t descrA,
                                                           cuComplex *bsrSortedVal,
                                                           const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd,
                                                           int blockSize,
                                                           bsrsm2Info_t info,
                                                           size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_bufferSizeExt(cusparseHandle_t handle,
                                                           cusparseDirection_t dirA,
                                                           cusparseOperation_t transA,
                                                           cusparseOperation_t transB,
                                                           int mb,
                                                           int n,
                                                           int nnzb,
                                                           const cusparseMatDescr_t descrA,
                                                           cuDoubleComplex *bsrSortedVal,
                                                           const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd,
                                                           int blockSize,
                                                           bsrsm2Info_t info,
                                                           size_t *pBufferSize);


cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_analysis(cusparseHandle_t handle,
                                                      cusparseDirection_t dirA,
                                                      cusparseOperation_t transA,
                                                      cusparseOperation_t transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      const cusparseMatDescr_t descrA,
                                                      const float *bsrSortedVal,
                                                      const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd,
                                                      int blockSize,
                                                      bsrsm2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_analysis(cusparseHandle_t handle,
                                                      cusparseDirection_t dirA,
                                                      cusparseOperation_t transA,
                                                      cusparseOperation_t transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      const cusparseMatDescr_t descrA,
                                                      const double *bsrSortedVal,
                                                      const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd,
                                                      int blockSize,
                                                      bsrsm2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_analysis(cusparseHandle_t handle,
                                                      cusparseDirection_t dirA,
                                                      cusparseOperation_t transA,
                                                      cusparseOperation_t transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      const cusparseMatDescr_t descrA,
                                                      const cuComplex *bsrSortedVal,
                                                      const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd,
                                                      int blockSize,
                                                      bsrsm2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_analysis(cusparseHandle_t handle,
                                                      cusparseDirection_t dirA,
                                                      cusparseOperation_t transA,
                                                      cusparseOperation_t transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      const cusparseMatDescr_t descrA,
                                                      const cuDoubleComplex *bsrSortedVal,
                                                      const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd,
                                                      int blockSize,
                                                      bsrsm2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_solve(cusparseHandle_t handle,
                                                   cusparseDirection_t dirA,
                                                   cusparseOperation_t transA,
                                                   cusparseOperation_t transXY,
                                                   int mb,
                                                   int n,
                                                   int nnzb,
                                                   const float *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const float *bsrSortedVal,
                                                   const int *bsrSortedRowPtr,
                                                   const int *bsrSortedColInd,
                                                   int blockSize,
                                                   bsrsm2Info_t info,
                                                   const float *F,
                                                   int ldf,
                                                   float *X,
                                                   int ldx,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_solve(cusparseHandle_t handle,
                                                   cusparseDirection_t dirA,
                                                   cusparseOperation_t transA,
                                                   cusparseOperation_t transXY,
                                                   int mb,
                                                   int n,
                                                   int nnzb,
                                                   const double *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const double *bsrSortedVal,
                                                   const int *bsrSortedRowPtr,
                                                   const int *bsrSortedColInd,
                                                   int blockSize,
                                                   bsrsm2Info_t info,
                                                   const double *F,
                                                   int ldf,
                                                   double *X,
                                                   int ldx,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_solve(cusparseHandle_t handle,
                                                   cusparseDirection_t dirA,
                                                   cusparseOperation_t transA,
                                                   cusparseOperation_t transXY,
                                                   int mb,
                                                   int n,
                                                   int nnzb,
                                                   const cuComplex *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const cuComplex *bsrSortedVal,
                                                   const int *bsrSortedRowPtr,
                                                   const int *bsrSortedColInd,
                                                   int blockSize,
                                                   bsrsm2Info_t info,
                                                   const cuComplex *F,
                                                   int ldf,
                                                   cuComplex *X,
                                                   int ldx,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_solve(cusparseHandle_t handle,
                                                   cusparseDirection_t dirA,
                                                   cusparseOperation_t transA,
                                                   cusparseOperation_t transXY,
                                                   int mb,
                                                   int n,
                                                   int nnzb,
                                                   const cuDoubleComplex *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const cuDoubleComplex *bsrSortedVal,
                                                   const int *bsrSortedRowPtr,
                                                   const int *bsrSortedColInd,
                                                   int blockSize,
                                                   bsrsm2Info_t info,
                                                   const cuDoubleComplex *F,
                                                   int ldf,
                                                   cuDoubleComplex *X,
                                                   int ldx,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer);


/* --- Preconditioners --- */ 

/* Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
   of the matrix A stored in CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (csrsv_analysis). 
   This routine implements algorithm 1 for this problem. */
cusparseStatus_t CUSPARSEAPI cusparseCsrilu0Ex(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              void *csrSortedValA_ValM, 
                                              cudaDataType csrSortedValA_ValMtype,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              cusparseSolveAnalysisInfo_t info,
                                              cudaDataType executiontype);

cusparseStatus_t CUSPARSEAPI cusparseScsrilu0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              float *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              double *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              cuComplex *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              cuDoubleComplex *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

/* Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
   of the matrix A stored in CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (csrsv2_analysis).
   This routine implements algorithm 2 for this problem. */
cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_numericBoost(cusparseHandle_t handle,
                                                            csrilu02Info_t info,
                                                            int enable_boost,    
                                                            double *tol,
                                                            float *boost_val);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_numericBoost(cusparseHandle_t handle,
                                                            csrilu02Info_t info,
                                                            int enable_boost,    
                                                            double *tol,
                                                            double *boost_val);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_numericBoost(cusparseHandle_t handle,
                                                            csrilu02Info_t info,
                                                            int enable_boost,    
                                                            double *tol,
                                                            cuComplex *boost_val);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_numericBoost(cusparseHandle_t handle,
                                                            csrilu02Info_t info,
                                                            int enable_boost,    
                                                            double *tol,
                                                            cuDoubleComplex *boost_val);

cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle,
                                                         csrilu02Info_t info,
                                                         int *position);

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSize(cusparseHandle_t handle,
                                                          int m,
                                                          int nnz,
                                                          const cusparseMatDescr_t descrA,
                                                          float *csrSortedValA,
                                                          const int *csrSortedRowPtrA,
                                                          const int *csrSortedColIndA,
                                                          csrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSize(cusparseHandle_t handle,
                                                          int m,
                                                          int nnz,
                                                          const cusparseMatDescr_t descrA,
                                                          double *csrSortedValA,
                                                          const int *csrSortedRowPtrA,
                                                          const int *csrSortedColIndA,
                                                          csrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSize(cusparseHandle_t handle,
                                                          int m,
                                                          int nnz,
                                                          const cusparseMatDescr_t descrA,
                                                          cuComplex *csrSortedValA,
                                                          const int *csrSortedRowPtrA,
                                                          const int *csrSortedColIndA,
                                                          csrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSize(cusparseHandle_t handle,
                                                          int m,
                                                          int nnz,
                                                          const cusparseMatDescr_t descrA,
                                                          cuDoubleComplex *csrSortedValA,
                                                          const int *csrSortedRowPtrA,
                                                          const int *csrSortedColIndA,
                                                          csrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int nnz,
                                                             const cusparseMatDescr_t descrA,
                                                             float *csrSortedVal,
                                                             const int *csrSortedRowPtr,
                                                             const int *csrSortedColInd,
                                                             csrilu02Info_t info,
                                                             size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int nnz,
                                                             const cusparseMatDescr_t descrA,
                                                             double *csrSortedVal,
                                                             const int *csrSortedRowPtr,
                                                             const int *csrSortedColInd,
                                                             csrilu02Info_t info,
                                                             size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int nnz,
                                                             const cusparseMatDescr_t descrA,
                                                             cuComplex *csrSortedVal,
                                                             const int *csrSortedRowPtr,
                                                             const int *csrSortedColInd,
                                                             csrilu02Info_t info,
                                                             size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int nnz,
                                                             const cusparseMatDescr_t descrA,
                                                             cuDoubleComplex *csrSortedVal,
                                                             const int *csrSortedRowPtr,
                                                             const int *csrSortedColInd,
                                                             csrilu02Info_t info,
                                                             size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_analysis(cusparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const float *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_analysis(cusparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const double *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_analysis(cusparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const cuComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_analysis(cusparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const cuDoubleComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02(cusparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               float *csrSortedValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02(cusparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               double *csrSortedValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02(cusparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               cuComplex *csrSortedValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02(cusparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               cuDoubleComplex *csrSortedValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer);

/* Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
   of the matrix A stored in block-CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (bsrsv2_analysis).
   This routine implements algorithm 2 for this problem. */
cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_numericBoost(cusparseHandle_t handle,
                                                            bsrilu02Info_t info,
                                                            int enable_boost,
                                                            double *tol,
                                                            float *boost_val);

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_numericBoost(cusparseHandle_t handle,
                                                            bsrilu02Info_t info,
                                                            int enable_boost,
                                                            double *tol,
                                                            double *boost_val);

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_numericBoost(cusparseHandle_t handle,
                                                            bsrilu02Info_t info,
                                                            int enable_boost,
                                                            double *tol,
                                                            cuComplex *boost_val);

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_numericBoost(cusparseHandle_t handle,
                                                            bsrilu02Info_t info,
                                                            int enable_boost,
                                                            double *tol,
                                                            cuDoubleComplex *boost_val);

cusparseStatus_t CUSPARSEAPI cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle,
                                                         bsrilu02Info_t info,
                                                         int *position);

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_bufferSize(cusparseHandle_t handle,
                                                          cusparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const cusparseMatDescr_t descrA,
                                                          float *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockDim,
                                                          bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_bufferSize(cusparseHandle_t handle,
                                                          cusparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const cusparseMatDescr_t descrA,
                                                          double *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockDim,
                                                          bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_bufferSize(cusparseHandle_t handle,
                                                          cusparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const cusparseMatDescr_t descrA,
                                                          cuComplex *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockDim,
                                                          bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_bufferSize(cusparseHandle_t handle,
                                                          cusparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const cusparseMatDescr_t descrA,
                                                          cuDoubleComplex *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockDim,
                                                          bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                                          cusparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const cusparseMatDescr_t descrA,
                                                          float *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockSize,
                                                          bsrilu02Info_t info,
                                                          size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                                          cusparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const cusparseMatDescr_t descrA,
                                                          double *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockSize,
                                                          bsrilu02Info_t info,
                                                          size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                                          cusparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const cusparseMatDescr_t descrA,
                                                          cuComplex *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockSize,
                                                          bsrilu02Info_t info,
                                                          size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                                          cusparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const cusparseMatDescr_t descrA,
                                                          cuDoubleComplex *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockSize,
                                                          bsrilu02Info_t info,
                                                          size_t *pBufferSize);


cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_analysis(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        float *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockDim,
                                                        bsrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_analysis(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        double *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockDim,
                                                        bsrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_analysis(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        cuComplex *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockDim,
                                                        bsrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_analysis(cusparseHandle_t handle,
                                                        cusparseDirection_t dirA,
                                                        int mb,
                                                        int nnzb,
                                                        const cusparseMatDescr_t descrA,
                                                        cuDoubleComplex *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockDim,
                                                        bsrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02(cusparseHandle_t handle,
                                               cusparseDirection_t dirA,
                                               int mb,
                                               int nnzb,
                                               const cusparseMatDescr_t descra,
                                               float *bsrSortedVal,
                                               const int *bsrSortedRowPtr,
                                               const int *bsrSortedColInd,
                                               int blockDim,
                                               bsrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02(cusparseHandle_t handle,
                                               cusparseDirection_t dirA,
                                               int mb,
                                               int nnzb,
                                               const cusparseMatDescr_t descra,
                                               double *bsrSortedVal,
                                               const int *bsrSortedRowPtr,
                                               const int *bsrSortedColInd,
                                               int blockDim,
                                               bsrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02(cusparseHandle_t handle,
                                               cusparseDirection_t dirA,
                                               int mb,
                                               int nnzb,
                                               const cusparseMatDescr_t descra,
                                               cuComplex *bsrSortedVal,
                                               const int *bsrSortedRowPtr,
                                               const int *bsrSortedColInd,
                                               int blockDim,
                                               bsrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02(cusparseHandle_t handle,
                                               cusparseDirection_t dirA,
                                               int mb,
                                               int nnzb,
                                               const cusparseMatDescr_t descra,
                                               cuDoubleComplex *bsrSortedVal,
                                               const int *bsrSortedRowPtr,
                                               const int *bsrSortedColInd,
                                               int blockDim,
                                               bsrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer);

/* Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
   of the matrix A stored in CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (csrsv_analysis). 
   This routine implements algorithm 1 for this problem. */
cusparseStatus_t CUSPARSEAPI cusparseScsric0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA,
                                              float *csrSortedValA_ValM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */ 
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDcsric0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              double *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseCcsric0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              cuComplex *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZcsric0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              cuDoubleComplex *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

/* Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
   of the matrix A stored in CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (csrsv2_analysis). 
   This routine implements algorithm 2 for this problem. */
cusparseStatus_t CUSPARSEAPI cusparseXcsric02_zeroPivot(cusparseHandle_t handle,
                                                        csric02Info_t info,
                                                        int *position);

cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSize(cusparseHandle_t handle,
                                                         int m,
                                                         int nnz,
                                                         const cusparseMatDescr_t descrA,
                                                         float *csrSortedValA,
                                                         const int *csrSortedRowPtrA,
                                                         const int *csrSortedColIndA,
                                                         csric02Info_t info,
                                                         int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSize(cusparseHandle_t handle,
                                                         int m,
                                                         int nnz,
                                                         const cusparseMatDescr_t descrA,
                                                         double *csrSortedValA,
                                                         const int *csrSortedRowPtrA,
                                                         const int *csrSortedColIndA,
                                                         csric02Info_t info,
                                                         int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSize(cusparseHandle_t handle,
                                                         int m,
                                                         int nnz,
                                                         const cusparseMatDescr_t descrA,
                                                         cuComplex *csrSortedValA,
                                                         const int *csrSortedRowPtrA,
                                                         const int *csrSortedColIndA,
                                                         csric02Info_t info,
                                                         int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSize(cusparseHandle_t handle,
                                                         int m,
                                                         int nnz,
                                                         const cusparseMatDescr_t descrA,
                                                         cuDoubleComplex *csrSortedValA,
                                                         const int *csrSortedRowPtrA,
                                                         const int *csrSortedColIndA,
                                                         csric02Info_t info,
                                                         int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSizeExt(cusparseHandle_t handle,
                                                            int m,
                                                            int nnz,
                                                            const cusparseMatDescr_t descrA,
                                                            float *csrSortedVal,
                                                            const int *csrSortedRowPtr,
                                                            const int *csrSortedColInd,
                                                            csric02Info_t info,
                                                            size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSizeExt(cusparseHandle_t handle,
                                                            int m,
                                                            int nnz,
                                                            const cusparseMatDescr_t descrA,
                                                            double *csrSortedVal,
                                                            const int *csrSortedRowPtr,
                                                            const int *csrSortedColInd,
                                                            csric02Info_t info,
                                                            size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSizeExt(cusparseHandle_t handle,
                                                            int m,
                                                            int nnz,
                                                            const cusparseMatDescr_t descrA,
                                                            cuComplex *csrSortedVal,
                                                            const int *csrSortedRowPtr,
                                                            const int *csrSortedColInd,
                                                            csric02Info_t info,
                                                            size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSizeExt(cusparseHandle_t handle,
                                                            int m,
                                                            int nnz,
                                                            const cusparseMatDescr_t descrA,
                                                            cuDoubleComplex *csrSortedVal,
                                                            const int *csrSortedRowPtr,
                                                            const int *csrSortedColInd,
                                                            csric02Info_t info,
                                                            size_t *pBufferSize);


cusparseStatus_t CUSPARSEAPI cusparseScsric02_analysis(cusparseHandle_t handle,
                                                       int m,
                                                       int nnz,
                                                       const cusparseMatDescr_t descrA,
                                                       const float *csrSortedValA,
                                                       const int *csrSortedRowPtrA,
                                                       const int *csrSortedColIndA,
                                                       csric02Info_t info,
                                                       cusparseSolvePolicy_t policy,
                                                       void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseDcsric02_analysis(cusparseHandle_t handle,
                                                       int m,
                                                       int nnz,
                                                       const cusparseMatDescr_t descrA,
                                                       const double *csrSortedValA,
                                                       const int *csrSortedRowPtrA,
                                                       const int *csrSortedColIndA,
                                                       csric02Info_t info,
                                                       cusparseSolvePolicy_t policy,
                                                       void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsric02_analysis(cusparseHandle_t handle,
                                                       int m,
                                                       int nnz,
                                                       const cusparseMatDescr_t descrA,
                                                       const cuComplex *csrSortedValA,
                                                       const int *csrSortedRowPtrA,
                                                       const int *csrSortedColIndA, 
                                                       csric02Info_t info,
                                                       cusparseSolvePolicy_t policy,
                                                       void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsric02_analysis(cusparseHandle_t handle,
                                                       int m,
                                                       int nnz,
                                                       const cusparseMatDescr_t descrA,
                                                       const cuDoubleComplex *csrSortedValA,
                                                       const int *csrSortedRowPtrA,
                                                       const int *csrSortedColIndA,
                                                       csric02Info_t info,
                                                       cusparseSolvePolicy_t policy,
                                                       void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseScsric02(cusparseHandle_t handle,
                                              int m,
                                              int nnz,
                                              const cusparseMatDescr_t descrA,
                                              float *csrSortedValA_valM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              csric02Info_t info,
                                              cusparseSolvePolicy_t policy,
                                              void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsric02(cusparseHandle_t handle,
                                              int m,
                                              int nnz,
                                              const cusparseMatDescr_t descrA,
                                              double *csrSortedValA_valM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              csric02Info_t info,
                                              cusparseSolvePolicy_t policy,
                                              void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsric02(cusparseHandle_t handle,
                                              int m,
                                              int nnz,
                                              const cusparseMatDescr_t descrA,
                                              cuComplex *csrSortedValA_valM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              csric02Info_t info,
                                              cusparseSolvePolicy_t policy,
                                              void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsric02(cusparseHandle_t handle,
                                              int m,
                                              int nnz,
                                              const cusparseMatDescr_t descrA,
                                              cuDoubleComplex *csrSortedValA_valM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              csric02Info_t info,
                                              cusparseSolvePolicy_t policy,
                                              void *pBuffer);

/* Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
   of the matrix A stored in block-CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (bsrsv2_analysis). 
   This routine implements algorithm 1 for this problem. */
cusparseStatus_t CUSPARSEAPI cusparseXbsric02_zeroPivot(cusparseHandle_t handle,
                                                        bsric02Info_t info,
                                                        int *position);

cusparseStatus_t CUSPARSEAPI cusparseSbsric02_bufferSize(cusparseHandle_t handle,
                                                         cusparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const cusparseMatDescr_t descrA,
                                                         float *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockDim,
                                                         bsric02Info_t info,
                                                         int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDbsric02_bufferSize(cusparseHandle_t handle,
                                                         cusparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const cusparseMatDescr_t descrA,
                                                         double *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockDim,
                                                         bsric02Info_t info,
                                                         int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCbsric02_bufferSize(cusparseHandle_t handle,
                                                         cusparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const cusparseMatDescr_t descrA,
                                                         cuComplex *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockDim,
                                                         bsric02Info_t info,
                                                         int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZbsric02_bufferSize(cusparseHandle_t handle,
                                                         cusparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const cusparseMatDescr_t descrA,
                                                         cuDoubleComplex *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockDim,
                                                         bsric02Info_t info,
                                                         int *pBufferSizeInBytes); 

cusparseStatus_t CUSPARSEAPI cusparseSbsric02_bufferSizeExt(cusparseHandle_t handle,
                                                         cusparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const cusparseMatDescr_t descrA,
                                                         float *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockSize,
                                                         bsric02Info_t info,
                                                         size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDbsric02_bufferSizeExt(cusparseHandle_t handle,
                                                         cusparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const cusparseMatDescr_t descrA,
                                                         double *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockSize,
                                                         bsric02Info_t info,
                                                         size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseCbsric02_bufferSizeExt(cusparseHandle_t handle,
                                                         cusparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const cusparseMatDescr_t descrA,
                                                         cuComplex *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockSize,
                                                         bsric02Info_t info,
                                                         size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseZbsric02_bufferSizeExt(cusparseHandle_t handle,
                                                         cusparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const cusparseMatDescr_t descrA,
                                                         cuDoubleComplex *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockSize,
                                                         bsric02Info_t info,
                                                         size_t *pBufferSize);



cusparseStatus_t CUSPARSEAPI cusparseSbsric02_analysis(cusparseHandle_t handle,
                                                       cusparseDirection_t dirA,
                                                       int mb,
                                                       int nnzb,
                                                       const cusparseMatDescr_t descrA,
                                                       const float *bsrSortedVal,
                                                       const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd,
                                                       int blockDim,
                                                       bsric02Info_t info,
                                                       cusparseSolvePolicy_t policy,
                                                       void *pInputBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDbsric02_analysis(cusparseHandle_t handle,
                                                       cusparseDirection_t dirA,
                                                       int mb,
                                                       int nnzb,
                                                       const cusparseMatDescr_t descrA,
                                                       const double *bsrSortedVal,
                                                       const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd,
                                                       int blockDim,
                                                       bsric02Info_t info,
                                                       cusparseSolvePolicy_t policy,
                                                       void *pInputBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCbsric02_analysis(cusparseHandle_t handle,
                                                       cusparseDirection_t dirA,
                                                       int mb,
                                                       int nnzb,
                                                       const cusparseMatDescr_t descrA,
                                                       const cuComplex *bsrSortedVal,
                                                       const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd,
                                                       int blockDim,
                                                       bsric02Info_t info,
                                                       cusparseSolvePolicy_t policy,
                                                       void *pInputBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZbsric02_analysis(cusparseHandle_t handle,
                                                       cusparseDirection_t dirA,
                                                       int mb,
                                                       int nnzb,
                                                       const cusparseMatDescr_t descrA,
                                                       const cuDoubleComplex *bsrSortedVal,
                                                       const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd,
                                                       int blockDim,
                                                       bsric02Info_t info,
                                                       cusparseSolvePolicy_t policy,
                                                       void *pInputBuffer);

cusparseStatus_t CUSPARSEAPI cusparseSbsric02(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nnzb,
                                              const cusparseMatDescr_t descrA,
                                              float *bsrSortedVal,
                                              const int *bsrSortedRowPtr,
                                              const int *bsrSortedColInd,
                                              int blockDim,
                                              bsric02Info_t info,
                                              cusparseSolvePolicy_t policy,
                                              void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDbsric02(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nnzb,
                                              const cusparseMatDescr_t descrA,
                                              double *bsrSortedVal,
                                              const int *bsrSortedRowPtr,
                                              const int *bsrSortedColInd,
                                              int blockDim,
                                              bsric02Info_t info,
                                              cusparseSolvePolicy_t policy,
                                              void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCbsric02(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nnzb,
                                              const cusparseMatDescr_t descrA,
                                              cuComplex *bsrSortedVal,
                                              const int *bsrSortedRowPtr,
                                              const int *bsrSortedColInd,
                                              int blockDim,
                                              bsric02Info_t info,
                                              cusparseSolvePolicy_t policy,
                                              void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZbsric02(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nnzb,
                                              const cusparseMatDescr_t descrA,
                                              cuDoubleComplex *bsrSortedVal,
                                              const int *bsrSortedRowPtr,
                                              const int *bsrSortedColInd,
                                              int blockDim,
                                              bsric02Info_t info,
                                              cusparseSolvePolicy_t policy,
                                              void *pBuffer);


/* Description: Solution of tridiagonal linear system A * X = F, 
   with multiple right-hand-sides. The coefficient matrix A is 
   composed of lower (dl), main (d) and upper (du) diagonals, and 
   the right-hand-sides F are overwritten with the solution X. 
   These routine use pivoting. */
cusparseStatus_t CUSPARSEAPI cusparseSgtsv(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    float *B,
    int ldb);
 
cusparseStatus_t CUSPARSEAPI cusparseDgtsv(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    double *B,
    int ldb);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    cuComplex *B,
    int ldb);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuDoubleComplex *dl,
    const cuDoubleComplex  *d,
    const cuDoubleComplex *du,
    cuDoubleComplex *B,
    int ldb);


cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    const float *B,
    int ldb,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    const double *B,
    int ldb,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    const cuComplex *B,
    int ldb,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuDoubleComplex *dl,
    const cuDoubleComplex  *d,
    const cuDoubleComplex *du,
    const cuDoubleComplex *B,
    int ldb,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    float *B,
    int ldb,
    void* pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    double *B,
    int ldb,
    void* pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    cuComplex *B,
    int ldb,
    void* pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuDoubleComplex *dl,
    const cuDoubleComplex  *d,
    const cuDoubleComplex *du,
    cuDoubleComplex *B,
    int ldb,
    void* pBuffer);


/* Description: Solution of tridiagonal linear system A * X = F, 
   with multiple right-hand-sides. The coefficient matrix A is 
   composed of lower (dl), main (d) and upper (du) diagonals, and 
   the right-hand-sides F are overwritten with the solution X. 
   These routine does not use pivoting. */                               
cusparseStatus_t CUSPARSEAPI cusparseSgtsv_nopivot(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    float *B,
    int ldb);
                                 
cusparseStatus_t CUSPARSEAPI cusparseDgtsv_nopivot(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    double *B,
    int ldb);
                                                                 
cusparseStatus_t CUSPARSEAPI cusparseCgtsv_nopivot(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    cuComplex *B,
    int ldb);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv_nopivot(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuDoubleComplex *dl,  
    const cuDoubleComplex  *d,  
    const cuDoubleComplex *du,
    cuDoubleComplex *B,     
    int ldb);                               
                                  

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    const float *B,
    int ldb,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    const double *B,
    int ldb,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    const cuComplex *B,
    int ldb,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuDoubleComplex *dl,
    const cuDoubleComplex  *d,
    const cuDoubleComplex *du,
    const cuDoubleComplex *B,
    int ldb,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    float *B,
    int ldb,
    void* pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    double *B,
    int ldb,
    void* pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    cuComplex *B,
    int ldb,
    void* pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuDoubleComplex *dl,
    const cuDoubleComplex  *d,
    const cuDoubleComplex *du,
    cuDoubleComplex *B,
    int ldb,
    void* pBuffer);

/* Description: Solution of a set of tridiagonal linear systems 
   A_{i} * x_{i} = f_{i} for i=1,...,batchCount. The coefficient 
   matrices A_{i} are composed of lower (dl), main (d) and upper (du) 
   diagonals and stored separated by a batchStride. Also, the 
   right-hand-sides/solutions f_{i}/x_{i} are separated by a batchStride. */
cusparseStatus_t CUSPARSEAPI cusparseSgtsvStridedBatch(
    cusparseHandle_t handle,
    int m, 
    const float *dl,
    const float  *d,
    const float *du,
    float *x,
    int batchCount,
    int batchStride);

cusparseStatus_t CUSPARSEAPI cusparseDgtsvStridedBatch(
    cusparseHandle_t handle,
    int m, 
    const double *dl,
    const double  *d,
    const double *du,
    double *x,
    int batchCount,
    int batchStride);

cusparseStatus_t CUSPARSEAPI cusparseCgtsvStridedBatch(
    cusparseHandle_t handle,
    int m, 
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    cuComplex *x,
    int batchCount,
    int batchStride);

cusparseStatus_t CUSPARSEAPI cusparseZgtsvStridedBatch(
    cusparseHandle_t handle,
    int m, 
    const cuDoubleComplex *dl,
    const cuDoubleComplex  *d,
    const cuDoubleComplex *du,
    cuDoubleComplex *x,
    int batchCount,
    int batchStride);


cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    const float *dl,
    const float  *d,
    const float *du,
    const float *x,
    int batchCount,
    int batchStride,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    const double *dl,
    const double  *d,
    const double *du,
    const double *x,
    int batchCount,
    int batchStride,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    const cuComplex *x,
    int batchCount,
    int batchStride,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    const cuDoubleComplex *dl,
    const cuDoubleComplex  *d,
    const cuDoubleComplex *du,
    const cuDoubleComplex *x,
    int batchCount,
    int batchStride,
    size_t *bufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch(
    cusparseHandle_t handle,
    int m,
    const float *dl,
    const float  *d,
    const float *du,
    float *x,
    int batchCount,
    int batchStride,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch(
    cusparseHandle_t handle,
    int m,
    const double *dl,
    const double  *d,
    const double *du,
    double *x,
    int batchCount,
    int batchStride,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch(
    cusparseHandle_t handle,
    int m,
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    cuComplex *x,
    int batchCount,
    int batchStride,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch(
    cusparseHandle_t handle,
    int m,
    const cuDoubleComplex *dl,
    const cuDoubleComplex  *d,
    const cuDoubleComplex *du,
    cuDoubleComplex *x,
    int batchCount,
    int batchStride,
    void *pBuffer);

/* --- Sparse Level 4 routines --- */

/* Description: Compute sparse - sparse matrix multiplication for matrices 
   stored in CSR format. */
cusparseStatus_t CUSPARSEAPI cusparseXcsrgemmNnz(cusparseHandle_t handle,
                                                 cusparseOperation_t transA, 
                                                 cusparseOperation_t transB, 
                                                 int m, 
                                                 int n, 
                                                 int k, 
                                                 const cusparseMatDescr_t descrA,
                                                 const int nnzA,
                                                 const int *csrSortedRowPtrA, 
                                                 const int *csrSortedColIndA,     
                                                 const cusparseMatDescr_t descrB,
                                                 const int nnzB,
                                                 const int *csrSortedRowPtrB, 
                                                 const int *csrSortedColIndB,  
                                                 const cusparseMatDescr_t descrC,                                                
                                                 int *csrSortedRowPtrC, 
                                                 int *nnzTotalDevHostPtr);                                              
                                              
cusparseStatus_t CUSPARSEAPI cusparseScsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA, 
                                              cusparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const cusparseMatDescr_t descrA,
                                              const int nnzA,      
                                              const float *csrSortedValA, 
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              const cusparseMatDescr_t descrB,
                                              const int nnzB,                                                    
                                              const float *csrSortedValB, 
                                              const int *csrSortedRowPtrB, 
                                              const int *csrSortedColIndB,
                                              const cusparseMatDescr_t descrC, 
                                              float *csrSortedValC, 
                                              const int *csrSortedRowPtrC, 
                                              int *csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA, 
                                              cusparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,      
                                              const double *csrSortedValA, 
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,                                                    
                                              const double *csrSortedValB, 
                                              const int *csrSortedRowPtrB, 
                                              const int *csrSortedColIndB,
                                              const cusparseMatDescr_t descrC, 
                                              double *csrSortedValC, 
                                              const int *csrSortedRowPtrC, 
                                              int *csrSortedColIndC);
                                              
cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA, 
                                              cusparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,      
                                              const cuComplex *csrSortedValA, 
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,                                                    
                                              const cuComplex *csrSortedValB, 
                                              const int *csrSortedRowPtrB, 
                                              const int *csrSortedColIndB,
                                              const cusparseMatDescr_t descrC, 
                                              cuComplex *csrSortedValC, 
                                              const int *csrSortedRowPtrC, 
                                              int *csrSortedColIndC); 
                                              
cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA, 
                                              cusparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,      
                                              const cuDoubleComplex *csrSortedValA, 
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,                                                    
                                              const cuDoubleComplex *csrSortedValB, 
                                              const int *csrSortedRowPtrB, 
                                              const int *csrSortedColIndB,
                                              const cusparseMatDescr_t descrC, 
                                              cuDoubleComplex *csrSortedValC, 
                                              const int *csrSortedRowPtrC, 
                                              int *csrSortedColIndC);

/* Description: Compute sparse - sparse matrix multiplication for matrices 
   stored in CSR format. */

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrgemm2Info(csrgemm2Info_t *info);

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrgemm2Info(csrgemm2Info_t info);

cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int k,
                                                             const float *alpha,
                                                             const cusparseMatDescr_t descrA,
                                                             int nnzA,
                                                             const int *csrSortedRowPtrA,
                                                             const int *csrSortedColIndA,
                                                             const cusparseMatDescr_t descrB,
                                                             int nnzB,
                                                             const int *csrSortedRowPtrB,
                                                             const int *csrSortedColIndB,
                                                             const float *beta,
                                                             const cusparseMatDescr_t descrD,
                                                             int nnzD,
                                                             const int *csrSortedRowPtrD,
                                                             const int *csrSortedColIndD,
                                                             csrgemm2Info_t info,
                                                             size_t *pBufferSizeInBytes );

cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int k,
                                                             const double *alpha,
                                                             const cusparseMatDescr_t descrA,
                                                             int nnzA,
                                                             const int *csrSortedRowPtrA,
                                                             const int *csrSortedColIndA,
                                                             const cusparseMatDescr_t descrB,
                                                             int nnzB,
                                                             const int *csrSortedRowPtrB,
                                                             const int *csrSortedColIndB,
                                                             const double *beta,
                                                             const cusparseMatDescr_t descrD,
                                                             int nnzD,
                                                             const int *csrSortedRowPtrD,
                                                             const int *csrSortedColIndD,
                                                             csrgemm2Info_t info,
                                                             size_t *pBufferSizeInBytes );

cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int k,
                                                             const cuComplex *alpha,
                                                             const cusparseMatDescr_t descrA,
                                                             int nnzA,
                                                             const int *csrSortedRowPtrA,
                                                             const int *csrSortedColIndA,
                                                             const cusparseMatDescr_t descrB,
                                                             int nnzB,
                                                             const int *csrSortedRowPtrB,
                                                             const int *csrSortedColIndB,
                                                             const cuComplex *beta,
                                                             const cusparseMatDescr_t descrD,
                                                             int nnzD,
                                                             const int *csrSortedRowPtrD,
                                                             const int *csrSortedColIndD,
                                                             csrgemm2Info_t info,
                                                             size_t *pBufferSizeInBytes );

cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int k,
                                                             const cuDoubleComplex *alpha,
                                                             const cusparseMatDescr_t descrA,
                                                             int nnzA,
                                                             const int *csrSortedRowPtrA,
                                                             const int *csrSortedColIndA,
                                                             const cusparseMatDescr_t descrB,
                                                             int nnzB,
                                                             const int *csrSortedRowPtrB,
                                                             const int *csrSortedColIndB,
                                                             const cuDoubleComplex *beta,
                                                             const cusparseMatDescr_t descrD,
                                                             int nnzD,
                                                             const int *csrSortedRowPtrD,
                                                             const int *csrSortedColIndD,
                                                             csrgemm2Info_t info,
                                                             size_t *pBufferSizeInBytes );


cusparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Nnz(cusparseHandle_t handle,
                                                  int m,
                                                  int n, 
                                                  int k,
                                                  const cusparseMatDescr_t descrA,
                                                  int nnzA,
                                                  const int *csrSortedRowPtrA,
                                                  const int *csrSortedColIndA,
                                                  const cusparseMatDescr_t descrB,
                                                  int nnzB,
                                                  const int *csrSortedRowPtrB,
                                                  const int *csrSortedColIndB,
                                                  const cusparseMatDescr_t descrD,
                                                  int nnzD,
                                                  const int *csrSortedRowPtrD,
                                                  const int *csrSortedColIndD,
                                                  const cusparseMatDescr_t descrC,
                                                  int *csrSortedRowPtrC,
                                                  int *nnzTotalDevHostPtr,
                                                  const csrgemm2Info_t info,
                                                  void *pBuffer );


cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int k,
                                               const float *alpha,
                                               const cusparseMatDescr_t descrA,
                                               int nnzA,
                                               const float *csrSortedValA,
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               const cusparseMatDescr_t descrB,
                                               int nnzB,
                                               const float *csrSortedValB,
                                               const int *csrSortedRowPtrB,
                                               const int *csrSortedColIndB,
                                               const float *beta,
                                               const cusparseMatDescr_t descrD,
                                               int nnzD,
                                               const float *csrSortedValD,
                                               const int *csrSortedRowPtrD,
                                               const int *csrSortedColIndD,
                                               const cusparseMatDescr_t descrC,
                                               float *csrSortedValC,
                                               const int *csrSortedRowPtrC,
                                               int *csrSortedColIndC,
                                               const csrgemm2Info_t info,
                                               void *pBuffer );

cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int k,
                                               const double *alpha,
                                               const cusparseMatDescr_t descrA,
                                               int nnzA,
                                               const double *csrSortedValA,
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               const cusparseMatDescr_t descrB,
                                               int nnzB,
                                               const double *csrSortedValB,
                                               const int *csrSortedRowPtrB,
                                               const int *csrSortedColIndB,
                                               const double *beta,
                                               const cusparseMatDescr_t descrD,
                                               int nnzD,
                                               const double *csrSortedValD,
                                               const int *csrSortedRowPtrD,
                                               const int *csrSortedColIndD,
                                               const cusparseMatDescr_t descrC,
                                               double *csrSortedValC,
                                               const int *csrSortedRowPtrC,
                                               int *csrSortedColIndC,
                                               const csrgemm2Info_t info,
                                               void *pBuffer );


cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int k,
                                               const cuComplex *alpha,
                                               const cusparseMatDescr_t descrA,
                                               int nnzA,
                                               const cuComplex *csrSortedValA,
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               const cusparseMatDescr_t descrB,
                                               int nnzB,
                                               const cuComplex *csrSortedValB,
                                               const int *csrSortedRowPtrB,
                                               const int *csrSortedColIndB,
                                               const cuComplex *beta,
                                               const cusparseMatDescr_t descrD,
                                               int nnzD,
                                               const cuComplex *csrSortedValD,
                                               const int *csrSortedRowPtrD,
                                               const int *csrSortedColIndD,
                                               const cusparseMatDescr_t descrC,
                                               cuComplex *csrSortedValC,
                                               const int *csrSortedRowPtrC,
                                               int *csrSortedColIndC,
                                               const csrgemm2Info_t info,
                                               void *pBuffer );


cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int k,
                                               const cuDoubleComplex *alpha,
                                               const cusparseMatDescr_t descrA,
                                               int nnzA,
                                               const cuDoubleComplex *csrSortedValA,
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               const cusparseMatDescr_t descrB,
                                               int nnzB,
                                               const cuDoubleComplex *csrSortedValB,
                                               const int *csrSortedRowPtrB,
                                               const int *csrSortedColIndB,
                                               const cuDoubleComplex *beta,
                                               const cusparseMatDescr_t descrD,
                                               int nnzD,
                                               const cuDoubleComplex *csrSortedValD,
                                               const int *csrSortedRowPtrD,
                                               const int *csrSortedColIndD,
                                               const cusparseMatDescr_t descrC,
                                               cuDoubleComplex *csrSortedValC,
                                               const int *csrSortedRowPtrC,
                                               int *csrSortedColIndC,
                                               const csrgemm2Info_t info,
                                               void *pBuffer );


/* Description: Compute sparse - sparse matrix addition of matrices 
   stored in CSR format */
cusparseStatus_t CUSPARSEAPI cusparseXcsrgeamNnz(cusparseHandle_t handle,
                                                 int m,
                                                 int n,
                                                 const cusparseMatDescr_t descrA,
                                                 int nnzA,
                                                 const int *csrSortedRowPtrA,
                                                 const int *csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int nnzB,
                                                 const int *csrSortedRowPtrB,
                                                 const int *csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 int *csrSortedRowPtrC,
                                                 int *nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseScsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const float *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const float *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              const float *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const float *csrSortedValB,
                                              const int *csrSortedRowPtrB,
                                              const int *csrSortedColIndB,
                                              const cusparseMatDescr_t descrC,
                                              float *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const double *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const double *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              const double *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const double *csrSortedValB,
                                              const int *csrSortedRowPtrB,
                                              const int *csrSortedColIndB,
                                              const cusparseMatDescr_t descrC,
                                              double *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);
    
cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cuComplex *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const cuComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              const cuComplex *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const cuComplex *csrSortedValB,
                                              const int *csrSortedRowPtrB,
                                              const int *csrSortedColIndB,
                                              const cusparseMatDescr_t descrC,
                                              cuComplex *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);
    
cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cuDoubleComplex *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const cuDoubleComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              const cuDoubleComplex *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const cuDoubleComplex *csrSortedValB,
                                              const int *csrSortedRowPtrB,
                                              const int *csrSortedColIndB,
                                              const cusparseMatDescr_t descrC,
                                              cuDoubleComplex *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);


/* --- Sparse Matrix Reorderings --- */

/* Description: Find an approximate coloring of a matrix stored in CSR format. */
cusparseStatus_t CUSPARSEAPI cusparseScsrcolor(cusparseHandle_t handle,
                                               int m, 
                                               int nnz,
                                               const cusparseMatDescr_t descrA, 
                                               const float *csrSortedValA, 
                                               const int *csrSortedRowPtrA, 
                                               const int *csrSortedColIndA,
                                               const float *fractionToColor,
                                               int *ncolors,
                                               int *coloring,
                                               int *reordering,
                                               const cusparseColorInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDcsrcolor(cusparseHandle_t handle,
                                               int m, 
                                               int nnz,
                                               const cusparseMatDescr_t descrA, 
                                               const double *csrSortedValA, 
                                               const int *csrSortedRowPtrA, 
                                               const int *csrSortedColIndA,
                                               const double *fractionToColor,
                                               int *ncolors,
                                               int *coloring,
                                               int *reordering,
                                               const cusparseColorInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseCcsrcolor(cusparseHandle_t handle,
                                               int m, 
                                               int nnz,
                                               const cusparseMatDescr_t descrA, 
                                               const cuComplex *csrSortedValA, 
                                               const int *csrSortedRowPtrA, 
                                               const int *csrSortedColIndA,
                                               const float *fractionToColor,
                                               int *ncolors,
                                               int *coloring,
                                               int *reordering,
                                               const cusparseColorInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZcsrcolor(cusparseHandle_t handle,
                                               int m, 
                                               int nnz,
                                               const cusparseMatDescr_t descrA, 
                                               const cuDoubleComplex *csrSortedValA, 
                                               const int *csrSortedRowPtrA, 
                                               const int *csrSortedColIndA,
                                               const double *fractionToColor,
                                               int *ncolors,
                                               int *coloring,
                                               int *reordering,
                                               const cusparseColorInfo_t info);

/* --- Sparse Format Conversion --- */

/* Description: This routine finds the total number of non-zero elements and 
   the number of non-zero elements per row or column in the dense matrix A. */
cusparseStatus_t CUSPARSEAPI cusparseSnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA, 
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const float *A, 
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseDnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const double *A, 
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseCnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const cuComplex *A,
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseZnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const cuDoubleComplex *A,
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);

/* --- Sparse Format Conversion --- */

/* Description: This routine finds the total number of non-zero elements and 
   the number of non-zero elements per row in a noncompressed csr matrix A. */
cusparseStatus_t CUSPARSEAPI cusparseSnnz_compress(cusparseHandle_t handle, 
                                          int m, 
                                          const cusparseMatDescr_t descr,
                                          const float *values, 
                                          const int *rowPtr, 
                                          int *nnzPerRow, 
                                          int *nnzTotal,
                                          float tol);

cusparseStatus_t CUSPARSEAPI cusparseDnnz_compress(cusparseHandle_t handle, 
                                          int m, 
                                          const cusparseMatDescr_t descr,
                                          const double *values, 
                                          const int *rowPtr, 
                                          int *nnzPerRow, 
                                          int *nnzTotal,
                                          double tol);

cusparseStatus_t CUSPARSEAPI cusparseCnnz_compress(cusparseHandle_t handle, 
                                          int m, 
                                          const cusparseMatDescr_t descr,
                                          const cuComplex *values, 
                                          const int *rowPtr, 
                                          int *nnzPerRow, 
                                          int *nnzTotal,
                                          cuComplex tol);

cusparseStatus_t CUSPARSEAPI cusparseZnnz_compress(cusparseHandle_t handle, 
                                          int m, 
                                          const cusparseMatDescr_t descr,
                                          const cuDoubleComplex *values, 
                                          const int *rowPtr, 
                                          int *nnzPerRow, 
                                          int *nnzTotal,
                                          cuDoubleComplex tol);
/* Description: This routine takes as input a csr form where the values may have 0 elements
   and compresses it to return a csr form with no zeros. */

cusparseStatus_t CUSPARSEAPI cusparseScsr2csr_compress(cusparseHandle_t handle,
                                                      int m, 
                                                      int n,
                                                      const cusparseMatDescr_t descra,
                                                      const float *inVal,
                                                      const int *inColInd,
                                                      const int * inRowPtr, 
                                                      int inNnz,
                                                      int *nnzPerRow, 
                                                      float *outVal,
                                                      int *outColInd,
                                                      int *outRowPtr,
                                                      float tol);        

cusparseStatus_t CUSPARSEAPI cusparseDcsr2csr_compress(cusparseHandle_t handle,
                                                      int m, //number of rows
                                                      int n,
                                                      const cusparseMatDescr_t descra,
                                                      const double *inVal, //csr values array-the elements which are below a certain tolerance will be remvoed
                                                      const int *inColInd,
                                                      const int * inRowPtr,  //corresponding input noncompressed row pointer
                                                      int inNnz,
                                                      int *nnzPerRow, //output: returns number of nonzeros per row 
                                                      double *outVal,
                                                      int *outColInd,
                                                      int *outRowPtr,
                                                      double tol);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2csr_compress(cusparseHandle_t handle,
                                                        int m, //number of rows
                                                        int n,
                                                        const cusparseMatDescr_t descra,
                                                        const cuComplex *inVal, //csr values array-the elements which are below a certain tolerance will be remvoed
                                                        const int *inColInd,
                                                        const int * inRowPtr,  //corresponding input noncompressed row pointer
                                                        int inNnz,
                                                        int *nnzPerRow, //output: returns number of nonzeros per row 
                                                        cuComplex *outVal,
                                                        int *outColInd,
                                                        int *outRowPtr,
                                                        cuComplex tol);                       

cusparseStatus_t CUSPARSEAPI cusparseZcsr2csr_compress(cusparseHandle_t handle,
                                                      int m, //number of rows
                                                      int n,
                                                      const cusparseMatDescr_t descra,
                                                      const cuDoubleComplex *inVal, //csr values array-the elements which are below a certain tolerance will be remvoed
                                                      const int *inColInd,
                                                      const int * inRowPtr,  //corresponding input noncompressed row pointer
                                                      int inNnz,
                                                      int *nnzPerRow, //output: returns number of nonzeros per row 
                                                      cuDoubleComplex *outVal,
                                                      int *outColInd,
                                                      int *outRowPtr,
                                                      cuDoubleComplex tol);                        
                                                                                                        
/* Description: This routine converts a dense matrix to a sparse matrix 
   in the CSR storage format, using the information computed by the 
   nnz routine. */
cusparseStatus_t CUSPARSEAPI cusparseSdense2csr(cusparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const cusparseMatDescr_t descrA,                            
                                                const float *A, 
                                                int lda,
                                                const int *nnzPerRow,                                                 
                                                float *csrSortedValA, 
                                                int *csrSortedRowPtrA, 
                                                int *csrSortedColIndA);
 
cusparseStatus_t CUSPARSEAPI cusparseDdense2csr(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,                                     
                                                const double *A, 
                                                int lda, 
                                                const int *nnzPerRow,                                                 
                                                double *csrSortedValA, 
                                                int *csrSortedRowPtrA, 
                                                int *csrSortedColIndA);
    
cusparseStatus_t CUSPARSEAPI cusparseCdense2csr(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,                                     
                                                const cuComplex *A, 
                                                int lda, 
                                                const int *nnzPerRow,                                                 
                                                cuComplex *csrSortedValA, 
                                                int *csrSortedRowPtrA, 
                                                int *csrSortedColIndA);
 
cusparseStatus_t CUSPARSEAPI cusparseZdense2csr(cusparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const cusparseMatDescr_t descrA,                                    
                                                const cuDoubleComplex *A, 
                                                int lda, 
                                                const int *nnzPerRow,                                                 
                                                cuDoubleComplex *csrSortedValA, 
                                                int *csrSortedRowPtrA, 
                                                int *csrSortedColIndA);

/* Description: This routine converts a sparse matrix in CSR storage format
   to a dense matrix. */
cusparseStatus_t CUSPARSEAPI cusparseScsr2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,  
                                                const float *csrSortedValA, 
                                                const int *csrSortedRowPtrA, 
                                                const int *csrSortedColIndA,
                                                float *A, 
                                                int lda);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsr2dense(cusparseHandle_t handle, 
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const double *csrSortedValA, 
                                                const int *csrSortedRowPtrA, 
                                                const int *csrSortedColIndA,
                                                double *A, 
                                                int lda);
    
cusparseStatus_t CUSPARSEAPI cusparseCcsr2dense(cusparseHandle_t handle, 
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const cuComplex *csrSortedValA, 
                                                const int *csrSortedRowPtrA, 
                                                const int *csrSortedColIndA,
                                                cuComplex *A, 
                                                int lda);
    
cusparseStatus_t CUSPARSEAPI cusparseZcsr2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const cuDoubleComplex *csrSortedValA, 
                                                const int *csrSortedRowPtrA, 
                                                const int *csrSortedColIndA,
                                                cuDoubleComplex *A, 
                                                int lda); 
                                 
/* Description: This routine converts a dense matrix to a sparse matrix 
   in the CSC storage format, using the information computed by the 
   nnz routine. */
cusparseStatus_t CUSPARSEAPI cusparseSdense2csc(cusparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const cusparseMatDescr_t descrA,                            
                                                const float *A, 
                                                int lda,
                                                const int *nnzPerCol,                                                 
                                                float *cscSortedValA, 
                                                int *cscSortedRowIndA, 
                                                int *cscSortedColPtrA);
 
cusparseStatus_t CUSPARSEAPI cusparseDdense2csc(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,                                     
                                                const double *A, 
                                                int lda,
                                                const int *nnzPerCol,                                                
                                                double *cscSortedValA, 
                                                int *cscSortedRowIndA, 
                                                int *cscSortedColPtrA); 

cusparseStatus_t CUSPARSEAPI cusparseCdense2csc(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,                                     
                                                const cuComplex *A, 
                                                int lda, 
                                                const int *nnzPerCol,                                                 
                                                cuComplex *cscSortedValA, 
                                                int *cscSortedRowIndA, 
                                                int *cscSortedColPtrA);
    
cusparseStatus_t CUSPARSEAPI cusparseZdense2csc(cusparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const cusparseMatDescr_t descrA,                                    
                                                const cuDoubleComplex *A, 
                                                int lda, 
                                                const int *nnzPerCol,
                                                cuDoubleComplex *cscSortedValA, 
                                                int *cscSortedRowIndA, 
                                                int *cscSortedColPtrA);

/* Description: This routine converts a sparse matrix in CSC storage format
   to a dense matrix. */
cusparseStatus_t CUSPARSEAPI cusparseScsc2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,  
                                                const float *cscSortedValA, 
                                                const int *cscSortedRowIndA, 
                                                const int *cscSortedColPtrA,
                                                float *A, 
                                                int lda);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsc2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const double *cscSortedValA, 
                                                const int *cscSortedRowIndA, 
                                                const int *cscSortedColPtrA,
                                                double *A, 
                                                int lda);

cusparseStatus_t CUSPARSEAPI cusparseCcsc2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const cuComplex *cscSortedValA, 
                                                const int *cscSortedRowIndA, 
                                                const int *cscSortedColPtrA,
                                                cuComplex *A, 
                                                int lda);

cusparseStatus_t CUSPARSEAPI cusparseZcsc2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const cuDoubleComplex *cscSortedValA, 
                                                const int *cscSortedRowIndA, 
                                                const int *cscSortedColPtrA,
                                                cuDoubleComplex *A, 
                                                int lda);
    
/* Description: This routine compresses the indecis of rows or columns.
   It can be interpreted as a conversion from COO to CSR sparse storage
   format. */
cusparseStatus_t CUSPARSEAPI cusparseXcoo2csr(cusparseHandle_t handle,
                                              const int *cooRowInd, 
                                              int nnz, 
                                              int m, 
                                              int *csrSortedRowPtr, 
                                              cusparseIndexBase_t idxBase);
    
/* Description: This routine uncompresses the indecis of rows or columns.
   It can be interpreted as a conversion from CSR to COO sparse storage
   format. */
cusparseStatus_t CUSPARSEAPI cusparseXcsr2coo(cusparseHandle_t handle,
                                              const int *csrSortedRowPtr, 
                                              int nnz, 
                                              int m, 
                                              int *cooRowInd, 
                                              cusparseIndexBase_t idxBase);     
    
/* Description: This routine converts a matrix from CSR to CSC sparse 
   storage format. The resulting matrix can be re-interpreted as a 
   transpose of the original matrix in CSR storage format. */
cusparseStatus_t CUSPARSEAPI cusparseCsr2cscEx(cusparseHandle_t handle,
                                              int m, 
                                              int n, 
                                              int nnz,
                                              const void  *csrSortedVal, 
                                              cudaDataType csrSortedValtype,
                                              const int *csrSortedRowPtr, 
                                              const int *csrSortedColInd, 
                                              void *cscSortedVal, 
                                              cudaDataType cscSortedValtype,
                                              int *cscSortedRowInd, 
                                              int *cscSortedColPtr, 
                                              cusparseAction_t copyValues, 
                                              cusparseIndexBase_t idxBase,
                                              cudaDataType executiontype);
    
cusparseStatus_t CUSPARSEAPI cusparseScsr2csc(cusparseHandle_t handle,
                                              int m, 
                                              int n, 
                                              int nnz,
                                              const float  *csrSortedVal, 
                                              const int *csrSortedRowPtr, 
                                              const int *csrSortedColInd, 
                                              float *cscSortedVal, 
                                              int *cscSortedRowInd, 
                                              int *cscSortedColPtr, 
                                              cusparseAction_t copyValues, 
                                              cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsr2csc(cusparseHandle_t handle,
                                              int m, 
                                              int n,
                                              int nnz,
                                              const double  *csrSortedVal, 
                                              const int *csrSortedRowPtr, 
                                              const int *csrSortedColInd,
                                              double *cscSortedVal, 
                                              int *cscSortedRowInd, 
                                              int *cscSortedColPtr,
                                              cusparseAction_t copyValues, 
                                              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2csc(cusparseHandle_t handle,
                                              int m, 
                                              int n,
                                              int nnz,
                                              const cuComplex  *csrSortedVal, 
                                              const int *csrSortedRowPtr, 
                                              const int *csrSortedColInd,
                                              cuComplex *cscSortedVal, 
                                              int *cscSortedRowInd, 
                                              int *cscSortedColPtr, 
                                              cusparseAction_t copyValues, 
                                              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2csc(cusparseHandle_t handle,
                                              int m, 
                                              int n, 
                                              int nnz,
                                              const cuDoubleComplex *csrSortedVal, 
                                              const int *csrSortedRowPtr, 
                                              const int *csrSortedColInd, 
                                              cuDoubleComplex *cscSortedVal, 
                                              int *cscSortedRowInd, 
                                              int *cscSortedColPtr,
                                              cusparseAction_t copyValues, 
                                              cusparseIndexBase_t idxBase);

/* Description: This routine converts a dense matrix to a sparse matrix 
   in HYB storage format. */
cusparseStatus_t CUSPARSEAPI cusparseSdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const float *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseDdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const double *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseCdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuComplex *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseZdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

/* Description: This routine converts a sparse matrix in HYB storage format
   to a dense matrix. */
cusparseStatus_t CUSPARSEAPI cusparseShyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                float *A,
                                                int lda);

cusparseStatus_t CUSPARSEAPI cusparseDhyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                double *A,
                                                int lda);
    
cusparseStatus_t CUSPARSEAPI cusparseChyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                cuComplex *A,
                                                int lda);

cusparseStatus_t CUSPARSEAPI cusparseZhyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                cuDoubleComplex *A,
                                                int lda);

/* Description: This routine converts a sparse matrix in CSR storage format
   to a sparse matrix in HYB storage format. */
cusparseStatus_t CUSPARSEAPI cusparseScsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const float *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const double *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

/* Description: This routine converts a sparse matrix in HYB storage format
   to a sparse matrix in CSR storage format. */
cusparseStatus_t CUSPARSEAPI cusparseShyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              float *csrSortedValA,
                                              int *csrSortedRowPtrA,
                                              int *csrSortedColIndA);

cusparseStatus_t CUSPARSEAPI cusparseDhyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              double *csrSortedValA,
                                              int *csrSortedRowPtrA,
                                              int *csrSortedColIndA);              

cusparseStatus_t CUSPARSEAPI cusparseChyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuComplex *csrSortedValA,
                                              int *csrSortedRowPtrA,
                                              int *csrSortedColIndA);

cusparseStatus_t CUSPARSEAPI cusparseZhyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuDoubleComplex *csrSortedValA,
                                              int *csrSortedRowPtrA,
                                              int *csrSortedColIndA);

/* Description: This routine converts a sparse matrix in CSC storage format
   to a sparse matrix in HYB storage format. */
cusparseStatus_t CUSPARSEAPI cusparseScsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const float *cscSortedValA,
                                              const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseDcsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const double *cscSortedValA,
                                              const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseCcsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *cscSortedValA,
                                              const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseZcsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *cscSortedValA,
                                              const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

/* Description: This routine converts a sparse matrix in HYB storage format
   to a sparse matrix in CSC storage format. */
cusparseStatus_t CUSPARSEAPI cusparseShyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              float *cscSortedVal,
                                              int *cscSortedRowInd,
                                              int *cscSortedColPtr);

cusparseStatus_t CUSPARSEAPI cusparseDhyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              double *cscSortedVal,
                                              int *cscSortedRowInd,
                                              int *cscSortedColPtr);

cusparseStatus_t CUSPARSEAPI cusparseChyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuComplex *cscSortedVal,
                                              int *cscSortedRowInd,
                                              int *cscSortedColPtr);

cusparseStatus_t CUSPARSEAPI cusparseZhyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuDoubleComplex *cscSortedVal,
                                              int *cscSortedRowInd,
                                              int *cscSortedColPtr);

/* Description: This routine converts a sparse matrix in CSR storage format
   to a sparse matrix in block-CSR storage format. */
cusparseStatus_t CUSPARSEAPI cusparseXcsr2bsrNnz(cusparseHandle_t handle,
                                                 cusparseDirection_t dirA,
                                                 int m,
                                                 int n,
                                                 const cusparseMatDescr_t descrA,
                                                 const int *csrSortedRowPtrA,
                                                 const int *csrSortedColIndA,
                                                 int blockDim,
                                                 const cusparseMatDescr_t descrC,
                                                 int *bsrSortedRowPtrC,
                                                 int *nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseScsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const float *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              float *bsrSortedValC,
                                              int *bsrSortedRowPtrC,
                                              int *bsrSortedColIndC);

cusparseStatus_t CUSPARSEAPI cusparseDcsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const double *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              double *bsrSortedValC,
                                              int *bsrSortedRowPtrC,
                                              int *bsrSortedColIndC);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuComplex *bsrSortedValC,
                                              int *bsrSortedRowPtrC,
                                              int *bsrSortedColIndC);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuDoubleComplex *bsrSortedValC,
                                              int *bsrSortedRowPtrC,
                                              int *bsrSortedColIndC);

/* Description: This routine converts a sparse matrix in block-CSR storage format
   to a sparse matrix in CSR storage format. */
cusparseStatus_t CUSPARSEAPI cusparseSbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const float *bsrSortedValA,
                                              const int *bsrSortedRowPtrA,
                                              const int *bsrSortedColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              float *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI cusparseDbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const double *bsrSortedValA,
                                              const int *bsrSortedRowPtrA,
                                              const int *bsrSortedColIndA,
                                              int   blockDim,
                                              const cusparseMatDescr_t descrC,
                                              double *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI cusparseCbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *bsrSortedValA,
                                              const int *bsrSortedRowPtrA,
                                              const int *bsrSortedColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuComplex *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

cusparseStatus_t CUSPARSEAPI cusparseZbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *bsrSortedValA,
                                              const int *bsrSortedRowPtrA,
                                              const int *bsrSortedColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuDoubleComplex *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

/* Description: This routine converts a sparse matrix in general block-CSR storage format
   to a sparse matrix in general block-CSC storage format. */
cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const float *bsrSortedVal,
                                                             const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd,
                                                             int rowBlockDim,
                                                             int colBlockDim,
                                                             int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const double *bsrSortedVal,
                                                             const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd,
                                                             int rowBlockDim,
                                                             int colBlockDim,
                                                             int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const cuComplex *bsrSortedVal,
                                                             const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd,
                                                             int rowBlockDim,
                                                             int colBlockDim,
                                                             int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const cuDoubleComplex *bsrSortedVal,
                                                             const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd,
                                                             int rowBlockDim,
                                                             int colBlockDim,
                                                             int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const float *bsrSortedVal,
                                                                const int *bsrSortedRowPtr,
                                                                const int *bsrSortedColInd,
                                                                int rowBlockDim,
                                                                int colBlockDim,
                                                                size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const double *bsrSortedVal,
                                                                const int *bsrSortedRowPtr,
                                                                const int *bsrSortedColInd,
                                                                int rowBlockDim,
                                                                int colBlockDim,
                                                                size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const cuComplex *bsrSortedVal,
                                                                const int *bsrSortedRowPtr,
                                                                const int *bsrSortedColInd,
                                                                int rowBlockDim,
                                                                int colBlockDim,
                                                                size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const cuDoubleComplex *bsrSortedVal,
                                                                const int *bsrSortedRowPtr,
                                                                const int *bsrSortedColInd,
                                                                int rowBlockDim,
                                                                int colBlockDim,
                                                                size_t *pBufferSize);


cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc(cusparseHandle_t handle,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const float *bsrSortedVal,
                                                  const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd,
                                                  int rowBlockDim,
                                                  int colBlockDim,
                                                  float *bscVal,
                                                  int *bscRowInd,
                                                  int *bscColPtr,
                                                  cusparseAction_t copyValues,
                                                  cusparseIndexBase_t baseIdx, 
                                                  void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc(cusparseHandle_t handle,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const double *bsrSortedVal,
                                                  const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd,
                                                  int rowBlockDim,
                                                  int colBlockDim,
                                                  double *bscVal,
                                                  int *bscRowInd,
                                                  int *bscColPtr,
                                                  cusparseAction_t copyValues,
                                                  cusparseIndexBase_t baseIdx,
                                                  void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc(cusparseHandle_t handle,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const cuComplex *bsrSortedVal,
                                                  const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd,
                                                  int rowBlockDim,
                                                  int colBlockDim,
                                                  cuComplex *bscVal,
                                                  int *bscRowInd,
                                                  int *bscColPtr,
                                                  cusparseAction_t copyValues,
                                                  cusparseIndexBase_t baseIdx,
                                                  void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc(cusparseHandle_t handle,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const cuDoubleComplex *bsrSortedVal,
                                                  const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd,
                                                  int rowBlockDim,
                                                  int colBlockDim,
                                                  cuDoubleComplex *bscVal,
                                                  int *bscRowInd,
                                                  int *bscColPtr,
                                                  cusparseAction_t copyValues,
                                                  cusparseIndexBase_t baseIdx,
                                                  void *pBuffer);

/* Description: This routine converts a sparse matrix in general block-CSR storage format
   to a sparse matrix in CSR storage format. */
cusparseStatus_t CUSPARSEAPI cusparseXgebsr2csr(cusparseHandle_t handle,
                                                cusparseDirection_t dirA,
                                                int mb,
                                                int nb,
                                                const cusparseMatDescr_t descrA,
                                                const int    *bsrSortedRowPtrA,
                                                const int    *bsrSortedColIndA,
                                                int   rowBlockDim,
                                                int   colBlockDim,
                                                const cusparseMatDescr_t descrC,
                                                int    *csrSortedRowPtrC,
                                                int    *csrSortedColIndC );

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2csr(cusparseHandle_t handle,
                                                cusparseDirection_t dirA,
                                                int mb,
                                                int nb,
                                                const cusparseMatDescr_t descrA,
                                                const float *bsrSortedValA,
                                                const int    *bsrSortedRowPtrA,
                                                const int    *bsrSortedColIndA,
                                                int   rowBlockDim,
                                                int   colBlockDim,
                                                const cusparseMatDescr_t descrC,
                                                float  *csrSortedValC,
                                                int    *csrSortedRowPtrC,
                                                int    *csrSortedColIndC );


cusparseStatus_t CUSPARSEAPI cusparseDgebsr2csr(cusparseHandle_t handle,
                                                cusparseDirection_t dirA,
                                                int mb,
                                                int nb,
                                                const cusparseMatDescr_t descrA,
                                                const double *bsrSortedValA,
                                                const int    *bsrSortedRowPtrA,
                                                const int    *bsrSortedColIndA,
                                                int   rowBlockDim,
                                                int   colBlockDim,
                                                const cusparseMatDescr_t descrC,
                                                double  *csrSortedValC,
                                                int    *csrSortedRowPtrC,
                                                int    *csrSortedColIndC );


cusparseStatus_t CUSPARSEAPI cusparseCgebsr2csr(cusparseHandle_t handle,
                                                cusparseDirection_t dirA,
                                                int mb,
                                                int nb,
                                                const cusparseMatDescr_t descrA,
                                                const cuComplex *bsrSortedValA,
                                                const int    *bsrSortedRowPtrA,
                                                const int    *bsrSortedColIndA,
                                                int   rowBlockDim,
                                                int   colBlockDim,
                                                const cusparseMatDescr_t descrC,
                                                cuComplex  *csrSortedValC,
                                                int    *csrSortedRowPtrC,
                                                int    *csrSortedColIndC );


cusparseStatus_t CUSPARSEAPI cusparseZgebsr2csr(cusparseHandle_t handle,
                                                cusparseDirection_t dirA,
                                                int mb,
                                                int nb,
                                                const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *bsrSortedValA,
                                                const int    *bsrSortedRowPtrA,
                                                const int    *bsrSortedColIndA,
                                                int   rowBlockDim,
                                                int   colBlockDim,
                                                const cusparseMatDescr_t descrC,
                                                cuDoubleComplex  *csrSortedValC,
                                                int    *csrSortedRowPtrC,
                                                int    *csrSortedColIndC );

/* Description: This routine converts a sparse matrix in CSR storage format
   to a sparse matrix in general block-CSR storage format. */
cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr_bufferSize(cusparseHandle_t handle,
                                                           cusparseDirection_t dirA,
                                                           int m,
                                                           int n,
                                                           const cusparseMatDescr_t descrA,
                                                           const float *csrSortedValA,
                                                           const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA,
                                                           int rowBlockDim,
                                                           int colBlockDim,
                                                           int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr_bufferSize(cusparseHandle_t handle,
                                                           cusparseDirection_t dirA,
                                                           int m,
                                                           int n,
                                                           const cusparseMatDescr_t descrA,
                                                           const double *csrSortedValA,
                                                           const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA,
                                                           int rowBlockDim,
                                                           int colBlockDim,
                                                           int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr_bufferSize(cusparseHandle_t handle,
                                                           cusparseDirection_t dirA,
                                                           int m,
                                                           int n,
                                                           const cusparseMatDescr_t descrA,
                                                           const cuComplex *csrSortedValA,
                                                           const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA,
                                                           int rowBlockDim,
                                                           int colBlockDim,
                                                           int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr_bufferSize(cusparseHandle_t handle,
                                                           cusparseDirection_t dirA,
                                                           int m,
                                                           int n,
                                                           const cusparseMatDescr_t descrA,
                                                           const cuDoubleComplex *csrSortedValA,
                                                           const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA,
                                                           int rowBlockDim,
                                                           int colBlockDim,
                                                           int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                                              cusparseDirection_t dirA,
                                                              int m,
                                                              int n,
                                                              const cusparseMatDescr_t descrA,
                                                              const float *csrSortedValA,
                                                              const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA,
                                                              int rowBlockDim,
                                                              int colBlockDim,
                                                              size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                                              cusparseDirection_t dirA,
                                                              int m,
                                                              int n,
                                                              const cusparseMatDescr_t descrA,
                                                              const double *csrSortedValA,
                                                              const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA,
                                                              int rowBlockDim,
                                                              int colBlockDim,
                                                              size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                                              cusparseDirection_t dirA,
                                                              int m,
                                                              int n,
                                                              const cusparseMatDescr_t descrA,
                                                              const cuComplex *csrSortedValA,
                                                              const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA,
                                                              int rowBlockDim,
                                                              int colBlockDim,
                                                              size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                                              cusparseDirection_t dirA,
                                                              int m,
                                                              int n,
                                                              const cusparseMatDescr_t descrA,
                                                              const cuDoubleComplex *csrSortedValA,
                                                              const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA,
                                                              int rowBlockDim,
                                                              int colBlockDim,
                                                              size_t *pBufferSize);



cusparseStatus_t CUSPARSEAPI cusparseXcsr2gebsrNnz(cusparseHandle_t handle,
                                                   cusparseDirection_t dirA,
                                                   int m,
                                                   int n,
                                                   const cusparseMatDescr_t descrA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   const cusparseMatDescr_t descrC,
                                                   int *bsrSortedRowPtrC,
                                                   int rowBlockDim,
                                                   int colBlockDim,
                                                   int *nnzTotalDevHostPtr,
                                                   void *pBuffer );

cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr(cusparseHandle_t handle,
                                                cusparseDirection_t dirA,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const float *csrSortedValA,
                                                const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA,
                                                const cusparseMatDescr_t descrC,
                                                float *bsrSortedValC,
                                                int *bsrSortedRowPtrC,
                                                int *bsrSortedColIndC,
                                                int rowBlockDim,
                                                int colBlockDim,
                                                void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr(cusparseHandle_t handle,
                                                cusparseDirection_t dirA,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const double *csrSortedValA,
                                                const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA,
                                                const cusparseMatDescr_t descrC,
                                                double *bsrSortedValC,
                                                int *bsrSortedRowPtrC,
                                                int *bsrSortedColIndC,
                                                int rowBlockDim,
                                                int colBlockDim,
                                                void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr(cusparseHandle_t handle,
                                                cusparseDirection_t dirA,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuComplex *csrSortedValA,
                                                const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA,
                                                const cusparseMatDescr_t descrC,
                                                cuComplex *bsrSortedValC,
                                                int *bsrSortedRowPtrC,
                                                int *bsrSortedColIndC,
                                                int rowBlockDim,
                                                int colBlockDim,
                                                void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr(cusparseHandle_t handle,
                                                cusparseDirection_t dirA,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *csrSortedValA,
                                                const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA,
                                                const cusparseMatDescr_t descrC,
                                                cuDoubleComplex *bsrSortedValC,
                                                int *bsrSortedRowPtrC,
                                                int *bsrSortedColIndC,
                                                int rowBlockDim,
                                                int colBlockDim,
                                                void *pBuffer);

/* Description: This routine converts a sparse matrix in general block-CSR storage format
   to a sparse matrix in general block-CSR storage format with different block size. */
cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr_bufferSize(cusparseHandle_t handle,
                                                             cusparseDirection_t dirA,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const cusparseMatDescr_t descrA,
                                                             const float *bsrSortedValA,
                                                             const int *bsrSortedRowPtrA,
                                                             const int *bsrSortedColIndA,
                                                             int rowBlockDimA,
                                                             int colBlockDimA,
                                                             int rowBlockDimC,
                                                             int colBlockDimC,
                                                             int *pBufferSizeInBytes );

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr_bufferSize(cusparseHandle_t handle,
                                                             cusparseDirection_t dirA,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const cusparseMatDescr_t descrA,
                                                             const double *bsrSortedValA,
                                                             const int *bsrSortedRowPtrA,
                                                             const int *bsrSortedColIndA,
                                                             int rowBlockDimA,
                                                             int colBlockDimA,
                                                             int rowBlockDimC,
                                                             int colBlockDimC,
                                                             int *pBufferSizeInBytes );

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr_bufferSize(cusparseHandle_t handle,
                                                             cusparseDirection_t dirA,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const cusparseMatDescr_t descrA,
                                                             const cuComplex *bsrSortedValA,
                                                             const int *bsrSortedRowPtrA,
                                                             const int *bsrSortedColIndA,
                                                             int rowBlockDimA,
                                                             int colBlockDimA,
                                                             int rowBlockDimC,
                                                             int colBlockDimC,
                                                             int *pBufferSizeInBytes );

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr_bufferSize(cusparseHandle_t handle,
                                                             cusparseDirection_t dirA,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const cusparseMatDescr_t descrA,
                                                             const cuDoubleComplex *bsrSortedValA,
                                                             const int *bsrSortedRowPtrA,
                                                             const int *bsrSortedColIndA,
                                                             int rowBlockDimA,
                                                             int colBlockDimA,
                                                             int rowBlockDimC,
                                                             int colBlockDimC,
                                                             int *pBufferSizeInBytes );


cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                                                cusparseDirection_t dirA,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const cusparseMatDescr_t descrA,
                                                                const float *bsrSortedValA,
                                                                const int    *bsrSortedRowPtrA,
                                                                const int    *bsrSortedColIndA,
                                                                int   rowBlockDimA,
                                                                int   colBlockDimA,
                                                                int   rowBlockDimC,
                                                                int   colBlockDimC,
                                                                size_t  *pBufferSize );

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                                                cusparseDirection_t dirA,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const cusparseMatDescr_t descrA,
                                                                const double *bsrSortedValA,
                                                                const int    *bsrSortedRowPtrA,
                                                                const int    *bsrSortedColIndA,
                                                                int   rowBlockDimA,
                                                                int   colBlockDimA,
                                                                int   rowBlockDimC,
                                                                int   colBlockDimC,
                                                                size_t  *pBufferSize );

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                                                cusparseDirection_t dirA,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const cusparseMatDescr_t descrA,
                                                                const cuComplex *bsrSortedValA,
                                                                const int    *bsrSortedRowPtrA,
                                                                const int    *bsrSortedColIndA,
                                                                int   rowBlockDimA,
                                                                int   colBlockDimA,
                                                                int   rowBlockDimC,
                                                                int   colBlockDimC,
                                                                size_t  *pBufferSize );

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                                                cusparseDirection_t dirA,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const cusparseMatDescr_t descrA,
                                                                const cuDoubleComplex *bsrSortedValA,
                                                                const int    *bsrSortedRowPtrA,
                                                                const int    *bsrSortedColIndA,
                                                                int   rowBlockDimA,
                                                                int   colBlockDimA,
                                                                int   rowBlockDimC,
                                                                int   colBlockDimC,
                                                                size_t  *pBufferSize );



cusparseStatus_t CUSPARSEAPI cusparseXgebsr2gebsrNnz(cusparseHandle_t handle,
                                                     cusparseDirection_t dirA,
                                                     int mb,
                                                     int nb,
                                                     int nnzb,
                                                     const cusparseMatDescr_t descrA,
                                                     const int *bsrSortedRowPtrA,
                                                     const int *bsrSortedColIndA,
                                                     int rowBlockDimA,
                                                     int colBlockDimA,
                                                     const cusparseMatDescr_t descrC,
                                                     int *bsrSortedRowPtrC,
                                                     int rowBlockDimC,
                                                     int colBlockDimC,
                                                     int *nnzTotalDevHostPtr,
                                                     void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr(cusparseHandle_t handle,
                                                  cusparseDirection_t dirA,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const cusparseMatDescr_t descrA,
                                                  const float *bsrSortedValA,
                                                  const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA,
                                                  int rowBlockDimA,
                                                  int colBlockDimA,
                                                  const cusparseMatDescr_t descrC,
                                                  float *bsrSortedValC,
                                                  int *bsrSortedRowPtrC,
                                                  int *bsrSortedColIndC,
                                                  int rowBlockDimC,
                                                  int colBlockDimC,
                                                  void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr(cusparseHandle_t handle,
                                                  cusparseDirection_t dirA,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const cusparseMatDescr_t descrA,
                                                  const double *bsrSortedValA,
                                                  const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA,
                                                  int rowBlockDimA,
                                                  int colBlockDimA,
                                                  const cusparseMatDescr_t descrC,
                                                  double *bsrSortedValC,
                                                  int *bsrSortedRowPtrC,
                                                  int *bsrSortedColIndC,
                                                  int rowBlockDimC,
                                                  int colBlockDimC,
                                                  void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr(cusparseHandle_t handle,
                                                  cusparseDirection_t dirA,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const cusparseMatDescr_t descrA,
                                                  const cuComplex *bsrSortedValA,
                                                  const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA,
                                                  int rowBlockDimA,
                                                  int colBlockDimA,
                                                  const cusparseMatDescr_t descrC,
                                                  cuComplex *bsrSortedValC,
                                                  int *bsrSortedRowPtrC,
                                                  int *bsrSortedColIndC,
                                                  int rowBlockDimC,
                                                  int colBlockDimC,
                                                  void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr(cusparseHandle_t handle,
                                                  cusparseDirection_t dirA,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const cusparseMatDescr_t descrA,
                                                  const cuDoubleComplex *bsrSortedValA,
                                                  const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA,
                                                  int rowBlockDimA,
                                                  int colBlockDimA,
                                                  const cusparseMatDescr_t descrC,
                                                  cuDoubleComplex *bsrSortedValC,
                                                  int *bsrSortedRowPtrC,
                                                  int *bsrSortedColIndC,
                                                  int rowBlockDimC,
                                                  int colBlockDimC,
                                                  void *pBuffer);

/* --- Sparse Matrix Sorting --- */

/* Description: Create a identity sequence p=[0,1,...,n-1]. */
cusparseStatus_t CUSPARSEAPI cusparseCreateIdentityPermutation(cusparseHandle_t handle,
                                                               int n,
                                                               int *p);

/* Description: Sort sparse matrix stored in COO format */
cusparseStatus_t CUSPARSEAPI cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle,
                                                            int m,
                                                            int n,
                                                            int nnz,
                                                            const int *cooRowsA,
                                                            const int *cooColsA,
                                                            size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseXcoosortByRow(cusparseHandle_t handle,
                                                   int m,
                                                   int n,
                                                   int nnz,
                                                   int *cooRowsA,
                                                   int *cooColsA,
                                                   int *P,
                                                   void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseXcoosortByColumn(cusparseHandle_t handle,
                                                      int m,
                                                      int n, 
                                                      int nnz,
                                                      int *cooRowsA,
                                                      int *cooColsA,
                                                      int *P,
                                                      void *pBuffer);

/* Description: Sort sparse matrix stored in CSR format */
cusparseStatus_t CUSPARSEAPI cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle,
                                                            int m,
                                                            int n,
                                                            int nnz,
                                                            const int *csrRowPtrA,
                                                            const int *csrColIndA,
                                                            size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseXcsrsort(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              int nnz,
                                              const cusparseMatDescr_t descrA,
                                              const int *csrRowPtrA,
                                              int *csrColIndA,
                                              int *P,
                                              void *pBuffer);
    
/* Description: Sort sparse matrix stored in CSC format */
cusparseStatus_t CUSPARSEAPI cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle,
                                                            int m,
                                                            int n,
                                                            int nnz,
                                                            const int *cscColPtrA,
                                                            const int *cscRowIndA,
                                                            size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseXcscsort(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              int nnz,
                                              const cusparseMatDescr_t descrA,
                                              const int *cscColPtrA,
                                              int *cscRowIndA,
                                              int *P,
                                              void *pBuffer);

/* Description: Wrapper that sorts sparse matrix stored in CSR format 
   (without exposing the permutation). */
cusparseStatus_t CUSPARSEAPI cusparseScsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             float *csrVal,
                                                             const int *csrRowPtr,
                                                             int *csrColInd,
                                                             csru2csrInfo_t  info,
                                                             size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             double *csrVal,
                                                             const int *csrRowPtr,
                                                             int *csrColInd,
                                                             csru2csrInfo_t  info,
                                                             size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             cuComplex *csrVal,
                                                             const int *csrRowPtr,
                                                             int *csrColInd,
                                                             csru2csrInfo_t  info,
                                                             size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             cuDoubleComplex *csrVal,
                                                             const int *csrRowPtr,
                                                             int *csrColInd,
                                                             csru2csrInfo_t  info,
                                                             size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseScsru2csr(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               float *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsru2csr(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               double *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsru2csr(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               cuComplex *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsru2csr(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               cuDoubleComplex *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

/* Description: Wrapper that un-sorts sparse matrix stored in CSR format 
   (without exposing the permutation). */
cusparseStatus_t CUSPARSEAPI cusparseScsr2csru(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               float *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsr2csru(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               double *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2csru(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               cuComplex *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2csru(cusparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               cuDoubleComplex *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

/* Description: prune dense matrix to a sparse matrix with CSR format */
#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    const __half *threshold,
    const cusparseMatDescr_t descrC,
    const __half *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    size_t *pBufferSizeInBytes);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    const float *threshold,
    const cusparseMatDescr_t descrC,
    const float *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    const double *threshold,
    const cusparseMatDescr_t descrC,
    const double *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    size_t *pBufferSizeInBytes);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csrNnz(
    cusparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    const __half *threshold,
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr,
    void *pBuffer);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrNnz(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    const float *threshold,
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrNnz(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    const double *threshold,
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr,
    void *pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csr(
    cusparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    const __half *threshold,
    const cusparseMatDescr_t descrC,
    __half *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    void *pBuffer);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csr(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    const float *threshold,
    const cusparseMatDescr_t descrC,
    float *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csr(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    const double *threshold,
    const cusparseMatDescr_t descrC,
    double *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    void *pBuffer);

/* Description: prune sparse matrix with CSR format to another sparse matrix with CSR format */
#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const __half *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const __half *threshold,
    const cusparseMatDescr_t descrC,
    const __half *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    size_t *pBufferSizeInBytes);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *threshold,
    const cusparseMatDescr_t descrC,
    const float *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *threshold,
    const cusparseMatDescr_t descrC,
    const double *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    size_t *pBufferSizeInBytes);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csrNnz(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const __half *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const __half *threshold,
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    void *pBuffer);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrNnz(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *threshold,
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrNnz(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *threshold,
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    void *pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csr(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const __half *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const __half *threshold,
    const cusparseMatDescr_t descrC,
    __half *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    void *pBuffer);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csr(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *threshold,
    const cusparseMatDescr_t descrC,
    float *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csr(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *threshold,
    const cusparseMatDescr_t descrC,
    double *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    void *pBuffer);

/* Description: prune dense matrix to a sparse matrix with CSR format by percentage */
#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    const __half *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    const float *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    const double *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneDense2csrByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    __half *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    pruneInfo_t info,
    void *pBuffer);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    float *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    pruneInfo_t info,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    double *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    pruneInfo_t info,
    void *pBuffer);


/* Description: prune sparse matrix to a sparse matrix with CSR format by percentage*/
#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const __half *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    const __half *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    const float *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    const double *csrValC,
    const int *csrRowPtrC,
    const int *csrColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const __half *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHpruneCsr2csrByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const __half *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    __half *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    pruneInfo_t info,
    void *pBuffer);
#endif

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    float *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    pruneInfo_t info,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float percentage, /* between 0 to 100 */
    const cusparseMatDescr_t descrC,
    double *csrValC,
    const int *csrRowPtrC,
    int *csrColIndC,
    pruneInfo_t info,
    void *pBuffer);




#if defined(__cplusplus)
}
#endif /* __cplusplus */                         

#endif /* !defined(CUSPARSE_H_) */

