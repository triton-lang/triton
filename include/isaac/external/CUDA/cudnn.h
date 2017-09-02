/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
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

 /*   cudnn : Neural Networks Library

 */

#if !defined(CUDNN_H_)
#define CUDNN_H_

#define CUDNN_MAJOR      7
#define CUDNN_MINOR      0
#define CUDNN_PATCHLEVEL 1

#define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#include "driver_types.h"
#include "cuda_runtime.h"

#ifndef CUDNNWINAPI
#ifdef _WIN32
#define CUDNNWINAPI __stdcall
#else
#define CUDNNWINAPI
#endif
#endif

#if defined (__cplusplus)
extern "C" {
#endif

struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;

size_t CUDNNWINAPI cudnnGetVersion(void);

/* Returns CUDA Runtime version statically linked against cudnn */
size_t CUDNNWINAPI cudnnGetCudartVersion(void);

/*
 * CUDNN return codes
 */
typedef enum
{
    CUDNN_STATUS_SUCCESS                      = 0,
    CUDNN_STATUS_NOT_INITIALIZED              = 1,
    CUDNN_STATUS_ALLOC_FAILED                 = 2,
    CUDNN_STATUS_BAD_PARAM                    = 3,
    CUDNN_STATUS_INTERNAL_ERROR               = 4,
    CUDNN_STATUS_INVALID_VALUE                = 5,
    CUDNN_STATUS_ARCH_MISMATCH                = 6,
    CUDNN_STATUS_MAPPING_ERROR                = 7,
    CUDNN_STATUS_EXECUTION_FAILED             = 8,
    CUDNN_STATUS_NOT_SUPPORTED                = 9,
    CUDNN_STATUS_LICENSE_ERROR                = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS          = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW          = 13,
} cudnnStatus_t;

/* human-readable error messages */
const char *  CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status);

/* Forward definition in this version only */
typedef struct cudnnRuntimeTag_t cudnnRuntimeTag_t;

typedef enum
{
    CUDNN_ERRQUERY_RAWCODE      = 0,
    CUDNN_ERRQUERY_NONBLOCKING  = 1,
    CUDNN_ERRQUERY_BLOCKING     = 2,
} cudnnErrQueryMode_t;

cudnnStatus_t CUDNNWINAPI cudnnQueryRuntimeError(
                                cudnnHandle_t                       handle,
                                cudnnStatus_t                      *rstatus,
                                cudnnErrQueryMode_t                 mode,
                                cudnnRuntimeTag_t                  *tag );

#ifndef __LIBRARY_TYPES_H__

typedef enum libraryPropertyType_t
{
    MAJOR_VERSION,
    MINOR_VERSION,
    PATCH_LEVEL
} libraryPropertyType;

#endif

cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType type, int *value);

cudnnStatus_t CUDNNWINAPI cudnnCreate        (cudnnHandle_t *handle);
cudnnStatus_t CUDNNWINAPI cudnnDestroy       (cudnnHandle_t handle);
cudnnStatus_t CUDNNWINAPI cudnnSetStream     (cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t CUDNNWINAPI cudnnGetStream     (cudnnHandle_t handle, cudaStream_t *streamId);

/* Data structures to represent Image/Filter and the Neural Network Layer */
typedef struct cudnnTensorStruct*          cudnnTensorDescriptor_t;
typedef struct cudnnConvolutionStruct*     cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct*         cudnnPoolingDescriptor_t;
typedef struct cudnnFilterStruct*          cudnnFilterDescriptor_t;
typedef struct cudnnLRNStruct*             cudnnLRNDescriptor_t;
typedef struct cudnnActivationStruct*      cudnnActivationDescriptor_t;
typedef struct cudnnSpatialTransformerStruct* cudnnSpatialTransformerDescriptor_t;
typedef struct cudnnOpTensorStruct*        cudnnOpTensorDescriptor_t;
typedef struct cudnnReduceTensorStruct*    cudnnReduceTensorDescriptor_t;
typedef struct cudnnCTCLossStruct*         cudnnCTCLossDescriptor_t;
/*
* CUDNN data type
*/
typedef enum
{
    CUDNN_DATA_FLOAT  = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF   = 2,
    CUDNN_DATA_INT8   = 3,
    CUDNN_DATA_INT32  = 4,
    CUDNN_DATA_INT8x4 = 5
} cudnnDataType_t;

/*
* CUDNN math type
*/
typedef enum { 
    CUDNN_DEFAULT_MATH   = 0,
    CUDNN_TENSOR_OP_MATH = 1,
} cudnnMathType_t; 

/*
 * CUDNN propagate Nan
 */
typedef enum{
    CUDNN_NOT_PROPAGATE_NAN  = 0,
    CUDNN_PROPAGATE_NAN      = 1,
} cudnnNanPropagation_t;

/* 
 * CUDNN Determinism
 */
typedef enum
{
    CUDNN_NON_DETERMINISTIC = 0,
    CUDNN_DETERMINISTIC     = 1,
} cudnnDeterminism_t;

/* Maximum supported number of tensor dimensions */
#define CUDNN_DIM_MAX 8

/* Create an instance of a generic Tensor descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(
                                cudnnTensorDescriptor_t            *tensorDesc );

typedef enum
{
    CUDNN_TENSOR_NCHW = 0,          /* row major (wStride = 1, hStride = w) */
    CUDNN_TENSOR_NHWC = 1,          /* feature maps interleaved ( cStride = 1 )*/
    CUDNN_TENSOR_NCHW_VECT_C = 2    /* each image point is vector of element of C : the length of the vector is carried by the data type*/
} cudnnTensorFormat_t;

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(
                                cudnnTensorDescriptor_t             tensorDesc,
                                cudnnTensorFormat_t                 format,
                                cudnnDataType_t                     dataType, /* image data type */
                                int                                 n,        /* number of inputs (batch size) */
                                int                                 c,        /* number of input feature maps */
                                int                                 h,        /* height of input section */
                                int                                 w );      /* width of input section */

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(
                                cudnnTensorDescriptor_t             tensorDesc,
                                cudnnDataType_t                     dataType, /* image data type */
                                int                                 n,        /* number of inputs (batch size) */
                                int                                 c,        /* number of input feature maps */
                                int                                 h,        /* height of input section */
                                int                                 w,        /* width of input section */
                                int                                 nStride,
                                int                                 cStride,
                                int                                 hStride,
                                int                                 wStride );

cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(
                                const cudnnTensorDescriptor_t       tensorDesc,
                                cudnnDataType_t                    *dataType, /* image data type */
                                int                                *n,        /* number of inputs (batch size) */
                                int                                *c,        /* number of input feature maps  */
                                int                                *h,        /* height of input section */
                                int                                *w,        /* width of input section */
                                int                                *nStride,
                                int                                *cStride,
                                int                                *hStride,
                                int                                *wStride );
                                
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(
                                cudnnTensorDescriptor_t             tensorDesc,
                                cudnnDataType_t                     dataType,
                                int                                 nbDims,
                                const int                           dimA[],
                                const int                           strideA[] );     
                                
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptorEx(
                                cudnnTensorDescriptor_t             tensorDesc,
                                cudnnTensorFormat_t                 format,
                                cudnnDataType_t                     dataType,
                                int                                 nbDims,
                                const int                           dimA[] );                                                           

cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(
                                const cudnnTensorDescriptor_t       tensorDesc,
                                int                                 nbDimsRequested,
                                cudnnDataType_t                    *dataType,
                                int                                *nbDims,
                                int                                 dimA[],
                                int                                 strideA[] );
                                
cudnnStatus_t CUDNNWINAPI cudnnGetTensorSizeInBytes(
                                const cudnnTensorDescriptor_t       tensorDesc,
                                size_t                              *size);                                

/* PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h * h_stride + w * w_stride

   1)Example of all images in row major order one batch of features after the other (with an optional padding on row)
   input_stride :  c x h x h_stride
   feature_stride : h x h_stride
   h_stride  :  >= w  ( h_stride = w if no padding)
   w_stride  : 1


   2)Example of all images in row major with features maps interleaved
   input_stride :  c x h x h_stride
   feature_stride : 1
   h_stride  :  w x c
   w_stride  : c

   3)Example of all images in column major order one batch of features after the other (with optional padding on column)
   input_stride :  c x w x w_stride
   feature_stride : w x w_stride
   h_stride  :  1
   w_stride  :  >= h

*/

/* Destroy an instance of Tensor4d descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(
                                cudnnTensorDescriptor_t             tensorDesc );


/* Tensor layout conversion helper (y = alpha * x + beta * y) */
cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );


/* Tensor Bias addition : C = alpha * A + beta * C  */
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       cDesc,
                                void                               *C );

/*
* CUDNN OpTensor op type
*/
typedef enum
{
    CUDNN_OP_TENSOR_ADD  = 0,
    CUDNN_OP_TENSOR_MUL  = 1,
    CUDNN_OP_TENSOR_MIN  = 2,
    CUDNN_OP_TENSOR_MAX  = 3,
    CUDNN_OP_TENSOR_SQRT = 4,
    CUDNN_OP_TENSOR_NOT  = 5,
} cudnnOpTensorOp_t;

cudnnStatus_t CUDNNWINAPI cudnnCreateOpTensorDescriptor(
                                cudnnOpTensorDescriptor_t          *opTensorDesc );

cudnnStatus_t CUDNNWINAPI cudnnSetOpTensorDescriptor(
                                cudnnOpTensorDescriptor_t           opTensorDesc,
                                cudnnOpTensorOp_t                   opTensorOp,
                                cudnnDataType_t                     opTensorCompType,
                                cudnnNanPropagation_t               opTensorNanOpt );

cudnnStatus_t CUDNNWINAPI cudnnGetOpTensorDescriptor(
                                const cudnnOpTensorDescriptor_t     opTensorDesc,
                                cudnnOpTensorOp_t                  *opTensorOp,
                                cudnnDataType_t                    *opTensorCompType,
                                cudnnNanPropagation_t              *opTensorNanOpt );

cudnnStatus_t CUDNNWINAPI cudnnDestroyOpTensorDescriptor(
                                cudnnOpTensorDescriptor_t           opTensorDesc );

/* Tensor operation : C = op( alpha1 * A, alpha2 * B ) + beta * C */
/* B tensor is ignored for CUDNN_OP_TENSOR_SQRT, CUDNN_OP_TENSOR_NOT. */
cudnnStatus_t CUDNNWINAPI cudnnOpTensor(
                                cudnnHandle_t                       handle,
                                const cudnnOpTensorDescriptor_t     opTensorDesc,
                                const void                         *alpha1,
                                const cudnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *alpha2,
                                const cudnnTensorDescriptor_t       bDesc,
                                const void                         *B,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       cDesc,
                                void                               *C );

/*
* CUDNN ReduceTensor op type
*/
typedef enum
{
    CUDNN_REDUCE_TENSOR_ADD          = 0,
    CUDNN_REDUCE_TENSOR_MUL          = 1,
    CUDNN_REDUCE_TENSOR_MIN          = 2,
    CUDNN_REDUCE_TENSOR_MAX          = 3,
    CUDNN_REDUCE_TENSOR_AMAX         = 4,
    CUDNN_REDUCE_TENSOR_AVG          = 5,
    CUDNN_REDUCE_TENSOR_NORM1        = 6,
    CUDNN_REDUCE_TENSOR_NORM2        = 7,
    CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
} cudnnReduceTensorOp_t;

/*
* CUDNN ReduceTensor indices type
*/
typedef enum
{
    CUDNN_REDUCE_TENSOR_NO_INDICES        = 0,
    CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1,
} cudnnReduceTensorIndices_t;

/*
* CUDNN tensor indices type size (all unsigned)
* Currently not supported, default is 32 bit unsigned.
*/
typedef enum
{
    CUDNN_32BIT_INDICES = 0,
    CUDNN_64BIT_INDICES = 1,
    CUDNN_16BIT_INDICES = 2,
    CUDNN_8BIT_INDICES  = 3,
} cudnnIndicesType_t;

cudnnStatus_t CUDNNWINAPI cudnnCreateReduceTensorDescriptor(
                                cudnnReduceTensorDescriptor_t          *reduceTensorDesc );

cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(
                                cudnnReduceTensorDescriptor_t           reduceTensorDesc,
                                cudnnReduceTensorOp_t                   reduceTensorOp,
                                cudnnDataType_t                     reduceTensorCompType,
                                cudnnNanPropagation_t               reduceTensorNanOpt,
                                cudnnReduceTensorIndices_t          reduceTensorIndices,
                                cudnnIndicesType_t                  reduceTensorIndicesType );

cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(
                                const cudnnReduceTensorDescriptor_t     reduceTensorDesc,
                                cudnnReduceTensorOp_t                  *reduceTensorOp,
                                cudnnDataType_t                    *reduceTensorCompType,
                                cudnnNanPropagation_t              *reduceTensorNanOpt,
                                cudnnReduceTensorIndices_t         *reduceTensorIndices,
                                cudnnIndicesType_t                 *reduceTensorIndicesType );

cudnnStatus_t CUDNNWINAPI cudnnDestroyReduceTensorDescriptor(
                                cudnnReduceTensorDescriptor_t           reduceTensorDesc );

 /* Helper function to return the minimum size of the index space to be passed to the reduction given the input and output tensors */
cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(
                                cudnnHandle_t                       handle,
                                const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                const cudnnTensorDescriptor_t       aDesc,
                                const cudnnTensorDescriptor_t       cDesc,
                                size_t                             *sizeInBytes );

 /* Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors */
cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                const cudnnTensorDescriptor_t       aDesc,
                                const cudnnTensorDescriptor_t       cDesc,
                                size_t                             *sizeInBytes );

/* Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(
                                cudnnHandle_t                       handle,
                                const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                void                               *indices,
                                size_t                              indicesSizeInBytes,
                                void                               *workspace,
                                size_t                              workspaceSizeInBytes,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       cDesc,
                                void                               *C );

/* Set all values of a tensor to a given value : y[i] = value[0] */
cudnnStatus_t CUDNNWINAPI cudnnSetTensor(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y,
                                const void                         *valuePtr );

/* Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y,
                                const void                         *alpha );

/*
 *  convolution mode
 */
typedef enum
{
    CUDNN_CONVOLUTION       = 0,
    CUDNN_CROSS_CORRELATION = 1
} cudnnConvolutionMode_t;


/* Create an instance of FilterStruct */
cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(
                                cudnnFilterDescriptor_t            *filterDesc );


cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor(
                                cudnnFilterDescriptor_t             filterDesc,
                                cudnnDataType_t                     dataType, /* image data type */
                                cudnnTensorFormat_t                 format,
                                int                                 k,        /* number of output feature maps */
                                int                                 c,        /* number of input feature maps */
                                int                                 h,        /* height of each input filter */
                                int                                 w );      /* width of  each input filter */


cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor(
                                const cudnnFilterDescriptor_t       filterDesc,
                                cudnnDataType_t                    *dataType, /* image data type */
                                cudnnTensorFormat_t                *format,
                                int                                *k,        /* number of output feature maps */
                                int                                *c,        /* number of input feature maps */
                                int                                *h,        /* height of each input filter */
                                int                                *w );      /* width of  each input filter */


cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(
                                cudnnFilterDescriptor_t             filterDesc,
                                cudnnDataType_t                     dataType, /* image data type */
                                cudnnTensorFormat_t                 format,
                                int                                 nbDims,
                                const int                           filterDimA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor(
                                const cudnnFilterDescriptor_t       filterDesc,
                                int                                 nbDimsRequested,
                                cudnnDataType_t                    *dataType, /* image data type */
                                cudnnTensorFormat_t                *format,
                                int                                *nbDims,
                                int                                 filterDimA[] );


cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(
                                cudnnFilterDescriptor_t             filterDesc );

/* Create an instance of convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(
                                cudnnConvolutionDescriptor_t       *convDesc );

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType( cudnnConvolutionDescriptor_t convDesc,
                                                       cudnnMathType_t mathType );

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType( cudnnConvolutionDescriptor_t convDesc,
                                                       cudnnMathType_t *mathType );

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionGroupCount( cudnnConvolutionDescriptor_t convDesc,
                                                         int groupCount );

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount( cudnnConvolutionDescriptor_t convDesc,
                                                         int *groupCount );

cudnnStatus_t CUDNNWINAPI cudnnSetConvolution2dDescriptor( cudnnConvolutionDescriptor_t convDesc,
                                                             int pad_h,    /* zero-padding height */
                                                             int pad_w,    /* zero-padding width */
                                                             int u,   /* vertical filter stride */
                                                             int v,   /* horizontal filter stride */
                                                             int dilation_h, /* filter dilation in the vertical dimension */
                                                             int dilation_w, /* filter dilation in the horizontal dimension */
                                                             cudnnConvolutionMode_t mode,
                                                             cudnnDataType_t computeType
                                                           );
                                                           
cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dDescriptor(  const cudnnConvolutionDescriptor_t convDesc,
                                                            int* pad_h,    /* zero-padding height */
                                                            int* pad_w,    /* zero-padding width */
                                                            int* u,        /* vertical filter stride */
                                                            int* v,        /* horizontal filter stride */
                                                            int* dilation_h, /* filter dilation in the vertical dimension */
                                                            int* dilation_w, /* filter dilation in the horizontal dimension */
                                                            cudnnConvolutionMode_t* mode,
                                                            cudnnDataType_t *computeType
                                                         );
                                                         
/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim(
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       inputTensorDesc,
                                const cudnnFilterDescriptor_t       filterDesc,
                                int                                *n,
                                int                                *c,
                                int                                *h,
                                int                                *w );


cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(
                                cudnnConvolutionDescriptor_t        convDesc,
                                int                                 arrayLength,             /* nbDims-2 size */
                                const int                           padA[],
                                const int                           filterStrideA[],
                                const int                           dilationA[],
                                cudnnConvolutionMode_t              mode,
                                cudnnDataType_t                     computeType );  /* convolution data type */

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(
                                const cudnnConvolutionDescriptor_t  convDesc,
                                int                                 arrayLengthRequested,
                                int                                *arrayLength,
                                int                                 padA[],
                                int                                 strideA[],
                                int                                 dilationA[],
                                cudnnConvolutionMode_t             *mode,
                                cudnnDataType_t                    *computeType );   /* convolution data type */


/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       inputTensorDesc,
                                const cudnnFilterDescriptor_t       filterDesc,
                                int                                 nbDims,
                                int                                 tensorOuputDimA[] );

/* Destroy an instance of convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor(
                                cudnnConvolutionDescriptor_t        convDesc );


/* helper function to provide the convolution algo that fit best the requirement */
typedef enum
{
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionFwdPreference_t;


typedef enum
{
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT                 = 8
} cudnnConvolutionFwdAlgo_t;

typedef struct {
    cudnnConvolutionFwdAlgo_t   algo;
    cudnnStatus_t               status;
    float                       time;
    size_t                      memory;
    cudnnDeterminism_t          determinism;
    cudnnMathType_t             mathType;
    int                         reserved[3];
} cudnnConvolutionFwdAlgoPerf_t;

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithmMaxCount( cudnnHandle_t     handle,
                                                                       int              *count);

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                cudnnConvolutionFwdAlgoPerf_t      *perfResults );

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                cudnnConvolutionFwdAlgoPerf_t      *perfResults,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes );


cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                cudnnConvolutionFwdPreference_t     preference,
                                size_t                              memoryLimitInBytes,
                                cudnnConvolutionFwdAlgo_t          *algo );


cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(
                                cudnnHandle_t                      handle,
                                const cudnnTensorDescriptor_t      srcDesc,
                                const cudnnFilterDescriptor_t      filterDesc,
                                const cudnnConvolutionDescriptor_t convDesc,
                                const cudnnTensorDescriptor_t      destDesc,
                                const int                          requestedAlgoCount,
                                int                               *returnedAlgoCount,
                                cudnnConvolutionFwdAlgoPerf_t     *perfResults);

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

 /* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                cudnnConvolutionFwdAlgo_t           algo,
                                size_t                             *sizeInBytes );


/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform the forward pass for batch convolution */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionFwdAlgo_t           algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(
                                cudnnHandle_t                       handle,
                                const void                         *alpha1,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionFwdAlgo_t           algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *alpha2,
                                const cudnnTensorDescriptor_t       zDesc,
                                const void                         *z,
                                const cudnnTensorDescriptor_t       biasDesc,
                                const void                         *bias,
                                const cudnnActivationDescriptor_t   activationDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to compute the bias gradient for batch convolution */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dbDesc,
                                void                               *db );


/* helper function to provide the convolution algo that fit best the requirement */
typedef enum
{
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionBwdFilterPreference_t;

typedef enum
{
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0                 = 0,  /* non-deterministic */
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1                 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT               = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3                 = 3,  /* non-deterministic */
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD          = 4,  /* not implemented */
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING        = 6,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT             = 7
} cudnnConvolutionBwdFilterAlgo_t;


typedef struct {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t                   status;
    float                           time;
    size_t                          memory;
    cudnnDeterminism_t              determinism;
    cudnnMathType_t                 mathType;
    int                             reserved[3];
} cudnnConvolutionBwdFilterAlgoPerf_t;

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithmMaxCount( cudnnHandle_t     handle,
                                                                              int              *count);
                                                                              
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnFilterDescriptor_t       dwDesc,
                                const int                           requestedAlgoCount,
                                int                                 *returnedAlgoCount,
                                cudnnConvolutionBwdFilterAlgoPerf_t *perfResults );

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithmEx(
                                cudnnHandle_t                        handle,
                                const cudnnTensorDescriptor_t        xDesc,
                                const void                          *x,
                                const cudnnTensorDescriptor_t        dyDesc,
                                const void                          *y,
                                const cudnnConvolutionDescriptor_t   convDesc,
                                const cudnnFilterDescriptor_t        dwDesc,
                                void                                *dw,
                                const int                            requestedAlgoCount,
                                int                                 *returnedAlgoCount,
                                cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
                                void                                *workSpace,
                                size_t                               workSpaceSizeInBytes );

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(
                                cudnnHandle_t                         handle,
                                const cudnnTensorDescriptor_t         xDesc,
                                const cudnnTensorDescriptor_t         dyDesc,
                                const cudnnConvolutionDescriptor_t    convDesc,
                                const cudnnFilterDescriptor_t         dwDesc,
                                cudnnConvolutionBwdFilterPreference_t preference,
                                size_t                                memoryLimitInBytes,
                                cudnnConvolutionBwdFilterAlgo_t      *algo );

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                                cudnnHandle_t                         handle,
                                const cudnnTensorDescriptor_t         srcDesc,
                                const cudnnTensorDescriptor_t         diffDesc,
                                const cudnnConvolutionDescriptor_t    convDesc,
                                const cudnnFilterDescriptor_t         gradDesc,
                                const int                             requestedAlgoCount,
                                int                                  *returnedAlgoCount,
                                cudnnConvolutionBwdFilterAlgoPerf_t  *perfResults);

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

 /* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnFilterDescriptor_t       gradDesc,
                                cudnnConvolutionBwdFilterAlgo_t     algo,
                                size_t                             *sizeInBytes );

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdFilterAlgo_t     algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnFilterDescriptor_t       dwDesc,
                                void                               *dw );
                                                           
/*********************************************************/
/* helper function to provide the convolution algo that fit best the requirement */
typedef enum
{
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE             = 0,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST           = 1,
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT  = 2,
} cudnnConvolutionBwdDataPreference_t;

typedef enum
{
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0                 = 0, /* non-deterministic */
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1                 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT               = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING        = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD          = 4,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT             = 6
} cudnnConvolutionBwdDataAlgo_t;

typedef struct {
    cudnnConvolutionBwdDataAlgo_t   algo;
    cudnnStatus_t                   status;
    float                           time;
    size_t                          memory;
    cudnnDeterminism_t              determinism;
    cudnnMathType_t                 mathType;
    int                             reserved[3];
} cudnnConvolutionBwdDataAlgoPerf_t;

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithmMaxCount( cudnnHandle_t     handle,
                                                                            int              *count);

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                cudnnConvolutionBwdDataAlgoPerf_t  *perfResults );

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithmEx(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                cudnnConvolutionBwdDataAlgoPerf_t  *perfResults,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes );

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                cudnnConvolutionBwdDataPreference_t preference,
                                size_t                              memoryLimitInBytes,
                                cudnnConvolutionBwdDataAlgo_t      *algo );

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm_v7(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       filterDesc,
                                const cudnnTensorDescriptor_t       diffDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       gradDesc,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                cudnnConvolutionBwdDataAlgoPerf_t  *perfResults);

 /* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                size_t                             *sizeInBytes );


cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );


cudnnStatus_t CUDNNWINAPI cudnnIm2Col(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                void                               *colBuffer );


/*
 *  softmax algorithm
 */
typedef enum
{
    CUDNN_SOFTMAX_FAST     = 0,         /* straightforward implementation */
    CUDNN_SOFTMAX_ACCURATE = 1,         /* subtract max from every point to avoid overflow */
    CUDNN_SOFTMAX_LOG      = 2
} cudnnSoftmaxAlgorithm_t;

typedef enum
{
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,   /* compute the softmax over all C, H, W for each N */
    CUDNN_SOFTMAX_MODE_CHANNEL = 1     /* compute the softmax over all C for each H, W, N */
} cudnnSoftmaxMode_t;

/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward softmax */
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(
                                cudnnHandle_t                       handle,
                                cudnnSoftmaxAlgorithm_t             algo,
                                cudnnSoftmaxMode_t                  mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to perform backward softmax */
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(
                                cudnnHandle_t                       handle,
                                cudnnSoftmaxAlgorithm_t             algo,
                                cudnnSoftmaxMode_t                  mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

/*
 *  pooling mode
 */
typedef enum
{
    CUDNN_POOLING_MAX     = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1, /* count for average includes padded values */
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2, /* count for average does not include padded values */
    CUDNN_POOLING_MAX_DETERMINISTIC     = 3
} cudnnPoolingMode_t;

/* Create an instance of pooling descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(
                                cudnnPoolingDescriptor_t           *poolingDesc );

cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(
                                cudnnPoolingDescriptor_t            poolingDesc,
                                cudnnPoolingMode_t                  mode,
                                cudnnNanPropagation_t               maxpoolingNanOpt,
                                int                                 windowHeight,
                                int                                 windowWidth,
                                int                                 verticalPadding,
                                int                                 horizontalPadding,
                                int                                 verticalStride,
                                int                                 horizontalStride );

cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescriptor(
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                cudnnPoolingMode_t                 *mode,
                                cudnnNanPropagation_t              *maxpoolingNanOpt,
                                int                                *windowHeight,
                                int                                *windowWidth,
                                int                                *verticalPadding,
                                int                                *horizontalPadding,
                                int                                *verticalStride,
                                int                                *horizontalStride );

cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(
                                cudnnPoolingDescriptor_t            poolingDesc,
                                const cudnnPoolingMode_t            mode,
                                const cudnnNanPropagation_t         maxpoolingNanOpt,
                                int                                 nbDims,
                                const int                           windowDimA[],
                                const int                           paddingA[],
                                const int                           strideA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                int                                 nbDimsRequested,
                                cudnnPoolingMode_t                 *mode,
                                cudnnNanPropagation_t              *maxpoolingNanOpt,
                                int                                *nbDims,
                                int                                 windowDimA[],
                                int                                 paddingA[],
                                int                                 strideA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdForwardOutputDim(
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                const cudnnTensorDescriptor_t       inputTensorDesc,
                                int                                 nbDims,
                                int                                 outputTensorDimA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dForwardOutputDim(
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                const cudnnTensorDescriptor_t       inputTensorDesc,
                                int                                *n,
                                int                                *c,
                                int                                *h,
                                int                                *w );


/* Destroy an instance of pooling descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor(
                                cudnnPoolingDescriptor_t            poolingDesc );

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward pooling */
cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(
                                cudnnHandle_t                       handle,
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to perform backward pooling */
cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(
                                cudnnHandle_t                       handle,
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                const void                          *alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

/*
 * activation mode
 */
typedef enum
{
    CUDNN_ACTIVATION_SIGMOID      = 0,
    CUDNN_ACTIVATION_RELU         = 1,
    CUDNN_ACTIVATION_TANH         = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU          = 4
} cudnnActivationMode_t;

/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(
                                cudnnActivationDescriptor_t        *activationDesc);

cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(
                                cudnnActivationDescriptor_t         activationDesc,
                                cudnnActivationMode_t               mode,
                                cudnnNanPropagation_t               reluNanOpt,
                                double                              coef ); /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t CUDNNWINAPI cudnnGetActivationDescriptor(
                                const cudnnActivationDescriptor_t   activationDesc,
                                cudnnActivationMode_t              *mode,
                                cudnnNanPropagation_t              *reluNanOpt,
                                double*                             coef ); /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(
                                cudnnActivationDescriptor_t activationDesc);

/* Function to perform forward activation  */
cudnnStatus_t CUDNNWINAPI cudnnActivationForward(
                                cudnnHandle_t                       handle,
                                cudnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to perform backward activation  */
cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(
                                cudnnHandle_t                       handle,
                                cudnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

/* 
* Create an instance of LRN (Local Response Normalization) descriptor
* Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
*/
cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(
                                cudnnLRNDescriptor_t               *normDesc );

#define CUDNN_LRN_MIN_N     1       /* minimum allowed lrnN */
#define CUDNN_LRN_MAX_N     16      /* maximum allowed lrnN */
#define CUDNN_LRN_MIN_K     1e-5    /* minimum allowed lrnK */
#define CUDNN_LRN_MIN_BETA  0.01    /* minimum allowed lrnBeta */

/* LRN layer mode */
typedef enum
{
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0,/* Normalize across tensor's dimA[1] dimension */
} cudnnLRNMode_t;

/*
* Uses a window [center-lookBehind, center+lookAhead], where
* lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
* Values of double parameters cast to tensor data type.
*/
cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(
                                cudnnLRNDescriptor_t                normDesc,
                                unsigned                            lrnN,
                                double                              lrnAlpha,
                                double                              lrnBeta,
                                double                              lrnK );
/*
* Retrieve the settings currently stored in an LRN layer descriptor
* Any of the provided pointers can be NULL (no corresponding value will be returned)
*/
cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(
                                cudnnLRNDescriptor_t                normDesc,
                                unsigned*                           lrnN,
                                double*                             lrnAlpha,
                                double*                             lrnBeta,
                                double*                             lrnK );

/* Destroy an instance of LRN descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor( cudnnLRNDescriptor_t lrnDesc );

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

/* LRN cross-channel forward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(
                                cudnnHandle_t                       handle,
                                cudnnLRNDescriptor_t                normDesc,
                                cudnnLRNMode_t                      lrnMode,
                                const void*                         alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* LRN cross-channel backward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelBackward(
                                cudnnHandle_t                       handle,
                                cudnnLRNDescriptor_t                normDesc,
                                cudnnLRNMode_t                      lrnMode,
                                const void*                         alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx);

typedef enum
{
    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0,
} cudnnDivNormMode_t;

/* LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(
                                cudnnHandle_t                       handle,
                                cudnnLRNDescriptor_t                normDesc,
                                cudnnDivNormMode_t                  mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc, /* same desc for means, temp, temp2 */
                                const void                         *x,
                                const void                         *means, /* if NULL, means are assumed to be zero */
                                void                               *temp,
                                void                               *temp2,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(
                                cudnnHandle_t                       handle,
                                cudnnLRNDescriptor_t                normDesc,
                                cudnnDivNormMode_t                  mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc, /* same desc for x, means, dy, temp, temp2 */
                                const void                         *x,
                                const void                         *means, /* if NULL, means are assumed to be zero */
                                const void                         *dy,
                                void                               *temp,
                                void                               *temp2,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dXdMeansDesc, /* same desc for dx, dMeans */
                                void                               *dx, /* output x differential */
                                void                               *dMeans ); /* output means differential, can be NULL */

typedef enum
{
    /* bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice) */
    CUDNN_BATCHNORM_PER_ACTIVATION     = 0,

    /* bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors) */
    CUDNN_BATCHNORM_SPATIAL            = 1,

    /* 
     * bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors). 
     * May be faster than CUDNN_BATCHNORM_SPATIAL but imposes some limits on the range of values 
     */
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,
} cudnnBatchNormMode_t;

#define CUDNN_BN_MIN_EPSILON 1e-5 /* Minimum epsilon allowed to be used in the Batch Normalization formula */

/*
* Derives a tensor descriptor from layer data descriptor for BatchNormalization 
* scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for 
* bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
*/
cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(
                                cudnnTensorDescriptor_t             derivedBnDesc,
                                const cudnnTensorDescriptor_t       xDesc,
                                cudnnBatchNormMode_t                mode );

/* Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
                                cudnnHandle_t                       handle,
                                cudnnBatchNormMode_t                mode,

                                const void                         *alpha, /* alpha[0] = result blend factor */
                                const void                         *beta,  /* beta[0] = dest layer blend factor */

                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,     /* NxCxHxW */
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y,     /* NxCxHxW */

                                /* Shared desc for the next 6 tensors in the argument list.
                                   Data type to be set as follows:
                                   type = (typeOf(x) == double) ? double : float
                                   Dimensions for this descriptor depend on normalization mode
                                   - Spatial Normalization : tensors are expected to have dims 1xCx1x1
                                    (normalization is performed across NxHxW)
                                   - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW 
                                    (normalization is performed across N) */
                                const cudnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,

                                /* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation */
                                const void                         *bnScale,
                                const void                         *bnBias,

                                /* MUST use factor=1 in the very first call of a complete training cycle.
                                   Use a factor=1/(1+n) at N-th call to the function to get
                                   Cumulative Moving Average (CMA) behavior
                                   CMA[n] = (x[1]+...+x[n])/n
                                   Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
                                   ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
                                   CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
                                double                              exponentialAverageFactor,

                                /* Used in Training phase only. 
                                   runningMean = newMean*factor + runningMean*(1-factor) */
                                void                               *resultRunningMean,
                                /* Output in training mode, input in inference. Is the moving average
                                   of  variance[x] (factor is applied in the same way as for runningMean) */
                                void                               *resultRunningVariance,

                                /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                                double                              epsilon,

                                /* Optionally save intermediate results from the forward pass here
                                   - can be reused to speed up backward pass. NULL if unused */
                                void                               *resultSaveMean,
                                void                               *resultSaveInvVariance );

/*
* Performs Batch Normalization during Inference: 
* y[i] = bnScale[k]*(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + bnBias[k]
* with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
* according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
* above for notes on function arguments.
*/
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
                                cudnnHandle_t                       handle,
                                cudnnBatchNormMode_t                mode,
                                const void                         *alpha, /* alpha[0] = result blend factor */
                                const void                         *beta,  /* beta[0] = dest layer blend factor */
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,     /* NxCxHxW */
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y,     /* NxCxHxW */
                                const cudnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,
                                const void                         *bnScale,
                                const void                         *bnBias,
                                const void                         *estimatedMean,
                                const void                         *estimatedVariance,
                                double                              epsilon );

/* Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(
                                cudnnHandle_t                       handle,
                                cudnnBatchNormMode_t                mode,
                                const void                         *alphaDataDiff,
                                const void                         *betaDataDiff,
                                const void                         *alphaParamDiff,
                                const void                         *betaParamDiff,
                                const cudnnTensorDescriptor_t       xDesc, /* same desc for x, dx, dy */
                                const void                         *x,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx,
                                /* Shared tensor desc for the 4 tensors below */
                                const cudnnTensorDescriptor_t       dBnScaleBiasDesc,
                                const void                         *bnScale, /* bnBias doesn't affect backpropagation */
                                /* scale and bias diff are not backpropagated below this layer */
                                void                               *dBnScaleResult,
                                void                               *dBnBiasResult,
                                /* Same epsilon as forward pass */
                                double                              epsilon,

                                /* Optionally cached intermediate results from
                                   forward pass */
                                const void                         *savedMean,
                                const void                         *savedInvVariance );

                                
/* APIs for spatial transformer network*/
typedef enum {
    CUDNN_SAMPLER_BILINEAR=0,
} cudnnSamplerType_t;
 
cudnnStatus_t CUDNNWINAPI cudnnCreateSpatialTransformerDescriptor(
                               cudnnSpatialTransformerDescriptor_t        *stDesc);
 
cudnnStatus_t CUDNNWINAPI cudnnSetSpatialTransformerNdDescriptor(
                                cudnnSpatialTransformerDescriptor_t         stDesc,
                                cudnnSamplerType_t                          samplerType, 
                                cudnnDataType_t                             dataType,
                                const int                                   nbDims,
                                const int                                   dimA[]);

cudnnStatus_t CUDNNWINAPI cudnnDestroySpatialTransformerDescriptor(
                                 cudnnSpatialTransformerDescriptor_t        stDesc);

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorForward(
                                 cudnnHandle_t                              handle,
                                 const cudnnSpatialTransformerDescriptor_t  stDesc,
                                 const void                                *theta,
                                 void                                      *grid);

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorBackward(
                                 cudnnHandle_t                              handle,
                                 const cudnnSpatialTransformerDescriptor_t  stDesc,
                                 const void                                *dgrid,
                                 void                                      *dtheta);

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerForward(
                                 cudnnHandle_t                              handle,
                                 cudnnSpatialTransformerDescriptor_t        stDesc,
                                 const void                                *alpha,                                    
                                 const cudnnTensorDescriptor_t              xDesc,
                                 const void                                *x,
                                 const void                                *grid,
                                 const void                                *beta,
                                 cudnnTensorDescriptor_t                    yDesc,
                                 void                                      *y);

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerBackward(
                                 cudnnHandle_t                              handle,
                                 cudnnSpatialTransformerDescriptor_t        stDesc,
                                 const void                                *alpha,
                                 const cudnnTensorDescriptor_t              xDesc,
                                 const void                                *x,
                                 const void                                *beta,
                                 const cudnnTensorDescriptor_t              dxDesc,
                                 void                                      *dx,
                                 const void                                *alphaDgrid,
                                 const cudnnTensorDescriptor_t              dyDesc,
                                 const void                                *dy,
                                 const void                                *grid,
                                 const void                                *betaDgrid,
                                 void                                      *dgrid);
                              
typedef struct cudnnDropoutStruct * cudnnDropoutDescriptor_t;

cudnnStatus_t CUDNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t * dropoutDesc);

cudnnStatus_t CUDNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc);

/*helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor */
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t * sizeInBytes);

/*helper function to determine size of the reserve space to be passed to dropout forward/backward calls */
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, size_t * sizeInBytes);

cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, 
                                                    cudnnHandle_t            handle,
                                                    float                    dropout, 
                                                    void *                   states, 
                                                    size_t                   stateSizeInBytes, 
                                                    unsigned long long       seed);

// Restores the dropout descriptor to a previously saved-off state
cudnnStatus_t CUDNNWINAPI cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, 
                                                        cudnnHandle_t            handle,
                                                        float                    dropout, 
                                                        void *                   states, 
                                                        size_t                   stateSizeInBytes, 
                                                        unsigned long long       seed);

cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, 
                                                    cudnnHandle_t            handle,
                                                    float *                  dropout, 
                                                    void **                  states,
                                                    unsigned long long *     seed);
                                                    
cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(cudnnHandle_t                  handle, 
                                              const cudnnDropoutDescriptor_t dropoutDesc,
                                              const cudnnTensorDescriptor_t  xdesc, 
                                              const void *                   x,
                                              const cudnnTensorDescriptor_t  ydesc,
                                              void *                         y,
                                              void *                         reserveSpace,
                                              size_t                         reserveSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(cudnnHandle_t                  handle, 
                                               const cudnnDropoutDescriptor_t dropoutDesc,
                                               const cudnnTensorDescriptor_t  dydesc, 
                                               const void *                   dy,
                                               const cudnnTensorDescriptor_t  dxdesc,
                                               void *                         dx,
                                               void *                         reserveSpace,
                                               size_t                         reserveSpaceSizeInBytes);

/* RNN API */
typedef enum 
  {
    CUDNN_RNN_RELU = 0, /* Stock RNN with ReLu activation */
    CUDNN_RNN_TANH = 1, /* Stock RNN with tanh activation */
    CUDNN_LSTM = 2,     /* LSTM with no peephole connections */
    CUDNN_GRU = 3       /* Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1); */
  } cudnnRNNMode_t;

typedef enum
  {
   CUDNN_UNIDIRECTIONAL = 0,
   CUDNN_BIDIRECTIONAL = 1      /* Using output concatination at each step. Do we also want to support output sum? */
  } cudnnDirectionMode_t;

typedef enum
  {
   CUDNN_LINEAR_INPUT = 0,
   CUDNN_SKIP_INPUT = 1    
  } cudnnRNNInputMode_t;  
    
  
typedef enum 
  {
    CUDNN_RNN_ALGO_STANDARD = 0, 
    CUDNN_RNN_ALGO_PERSIST_STATIC = 1,
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
  } cudnnRNNAlgo_t;  
  
struct cudnnRNNStruct;
typedef struct cudnnRNNStruct*        cudnnRNNDescriptor_t;

cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t * rnnDesc);
cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc);

struct cudnnPersistentRNNPlan;
typedef struct cudnnPersistentRNNPlan *cudnnPersistentRNNPlan_t;

                   
/* Expensive. Creates the plan for the specific settings. */
cudnnStatus_t CUDNNWINAPI cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t       rnnDesc,
                                                       const int                  minibatch,
                                                       const cudnnDataType_t      dataType,
                                                       cudnnPersistentRNNPlan_t * plan);

/* Attaches the plan to the descriptor. */
cudnnStatus_t CUDNNWINAPI cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                                    cudnnPersistentRNNPlan_t plan);

cudnnStatus_t CUDNNWINAPI cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan);

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor(cudnnHandle_t            handle,
                                                   cudnnRNNDescriptor_t     rnnDesc,
                                                   const int                hiddenSize,
                                                   const int                numLayers,
                                                   cudnnDropoutDescriptor_t dropoutDesc, /* Between layers, not between recurrent steps. */
                                                   cudnnRNNInputMode_t      inputMode,          
                                                   cudnnDirectionMode_t     direction,
                                                   cudnnRNNMode_t           mode,
                                                   cudnnRNNAlgo_t           algo,
                                                   cudnnDataType_t          dataType);

cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor(cudnnHandle_t              cudnnHandle,
                                                cudnnRNNDescriptor_t       rnnDesc,
                                                int *                      hiddenSize, 
                                                int *                      numLayers, 
                                                cudnnDropoutDescriptor_t * dropoutDesc,
                                                cudnnRNNInputMode_t *      inputMode, 
                                                cudnnDirectionMode_t *     direction, 
                                                cudnnRNNMode_t *           mode, 
                                                cudnnRNNAlgo_t *           algo, 
                                                cudnnDataType_t *          dataType);

cudnnStatus_t CUDNNWINAPI cudnnSetRNNMatrixMathType (cudnnRNNDescriptor_t desc, cudnnMathType_t math);
                                                
/* dataType in the RNN descriptor is used to determine math precision */
/* dataType in weight descriptors and input descriptors is used to describe storage */
cudnnStatus_t CUDNNWINAPI cudnnGetRNNWorkspaceSize( cudnnHandle_t              handle,
                                                    const cudnnRNNDescriptor_t rnnDesc,  
                                                    const int seqLength, 
                                                    const cudnnTensorDescriptor_t    *xDesc,
                                                    size_t                     *sizeInBytes);
                                                      
cudnnStatus_t CUDNNWINAPI cudnnGetRNNTrainingReserveSize( cudnnHandle_t              handle,
                                                          const cudnnRNNDescriptor_t rnnDesc,  
                                                          const int                  seqLength,
                                                          const cudnnTensorDescriptor_t    *xDesc,
                                                          size_t                   *sizeInBytes);

                                                    
cudnnStatus_t CUDNNWINAPI cudnnGetRNNParamsSize( cudnnHandle_t                    handle,
                                                 const cudnnRNNDescriptor_t       rnnDesc,  
                                                 const cudnnTensorDescriptor_t    xDesc,
                                                 size_t                          *sizeInBytes,
                                                 cudnnDataType_t dataType);

cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerMatrixParams( cudnnHandle_t              handle,
                                                           const cudnnRNNDescriptor_t rnnDesc, 
                                                           const int layer,
                                                           const cudnnTensorDescriptor_t xDesc,
                                                           const cudnnFilterDescriptor_t wDesc,
                                                           const void * w, 
                                                           const int linLayerID,  
                                                           cudnnFilterDescriptor_t linLayerMatDesc,
                                                           void ** linLayerMat);

cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerBiasParams( cudnnHandle_t              handle,
                                                         const cudnnRNNDescriptor_t rnnDesc, 
                                                         const int layer,
                                                         const cudnnTensorDescriptor_t xDesc, 
                                                         const cudnnFilterDescriptor_t wDesc,
                                                         const void * w,
                                                         const int linLayerID,
                                                         cudnnFilterDescriptor_t linLayerBiasDesc,
                                                         void ** linLayerBias);

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInference( cudnnHandle_t handle,
                                                    const cudnnRNNDescriptor_t rnnDesc,
                                                    const int seqLength,
                                                    const cudnnTensorDescriptor_t * xDesc,
                                                    const void * x,
                                                    const cudnnTensorDescriptor_t hxDesc,
                                                    const void * hx,
                                                    const cudnnTensorDescriptor_t cxDesc,
                                                    const void * cx,
                                                    const cudnnFilterDescriptor_t wDesc,
                                                    const void * w,
                                                    const cudnnTensorDescriptor_t *yDesc,
                                                    void * y,
                                                    const cudnnTensorDescriptor_t hyDesc,
                                                    void * hy,
                                                    const cudnnTensorDescriptor_t cyDesc,
                                                    void * cy,
                                                    void * workspace,
                                                    size_t workSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining( cudnnHandle_t handle,
                                                   const cudnnRNNDescriptor_t rnnDesc,
                                                   const int seqLength,
                                                   const cudnnTensorDescriptor_t *xDesc,
                                                   const void * x,
                                                   const cudnnTensorDescriptor_t hxDesc,
                                                   const void * hx,
                                                   const cudnnTensorDescriptor_t cxDesc,
                                                   const void * cx,
                                                   const cudnnFilterDescriptor_t wDesc,
                                                   const void * w,
                                                   const cudnnTensorDescriptor_t *yDesc,
                                                   void * y,
                                                   const cudnnTensorDescriptor_t hyDesc,
                                                   void * hy,
                                                   const cudnnTensorDescriptor_t cyDesc,
                                                   void * cy,
                                                   void * workspace,
                                                   size_t workSpaceSizeInBytes,
                                                   void * reserveSpace,
                                                   size_t reserveSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData( cudnnHandle_t handle,
                                                const cudnnRNNDescriptor_t rnnDesc,
                                                const int seqLength,
                                                const cudnnTensorDescriptor_t * yDesc,
                                                const void * y,
                                                const cudnnTensorDescriptor_t * dyDesc,
                                                const void * dy,
                                                const cudnnTensorDescriptor_t dhyDesc,
                                                const void * dhy,
                                                const cudnnTensorDescriptor_t dcyDesc,
                                                const void * dcy,
                                                const cudnnFilterDescriptor_t wDesc,
                                                const void * w,
                                                const cudnnTensorDescriptor_t hxDesc,
                                                const void * hx,
                                                const cudnnTensorDescriptor_t cxDesc,
                                                const void * cx,
                                                const cudnnTensorDescriptor_t * dxDesc,
                                                void * dx,
                                                const cudnnTensorDescriptor_t dhxDesc,
                                                void * dhx,
                                                const cudnnTensorDescriptor_t dcxDesc,
                                                void * dcx,
                                                void * workspace,
                                                size_t workSpaceSizeInBytes,
                                                void * reserveSpace,
                                                size_t reserveSpaceSizeInBytes );


cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights( cudnnHandle_t handle,
                                                   const cudnnRNNDescriptor_t rnnDesc,
                                                   const int seqLength,
                                                   const cudnnTensorDescriptor_t * xDesc,
                                                   const void * x,
                                                   const cudnnTensorDescriptor_t hxDesc,
                                                   const void * hx,
                                                   const cudnnTensorDescriptor_t * yDesc, 
                                                   const void * y,
                                                   const void * workspace, 
                                                   size_t workSpaceSizeInBytes, 
                                                   const cudnnFilterDescriptor_t dwDesc, 
                                                   void * dw,
                                                   const void * reserveSpace, 
                                                   size_t reserveSpaceSizeInBytes );

typedef enum
{
    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0,
    CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1
}cudnnCTCLossAlgo_t;

/* 
* Create an instance of a CTC (Connectionist Temporal Classification) loss descriptor
*/
cudnnStatus_t CUDNNWINAPI cudnnCreateCTCLossDescriptor( cudnnCTCLossDescriptor_t* ctcLossDesc );

cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor(
                                cudnnCTCLossDescriptor_t         ctcLossDesc,
                                cudnnDataType_t                  compType );

cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptor(
                                cudnnCTCLossDescriptor_t         ctcLossDesc,
                                cudnnDataType_t*                 compType );

cudnnStatus_t CUDNNWINAPI cudnnDestroyCTCLossDescriptor( cudnnCTCLossDescriptor_t ctcLossDesc );

/* return the ctc costs and gradients, given the probabilities and labels */
cudnnStatus_t CUDNNWINAPI cudnnCTCLoss( cudnnHandle_t handle, 
                                        const cudnnTensorDescriptor_t probsDesc,     /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)  */
                                        const void * probs,                          /* probabilities after softmax, in GPU memory */
                                        const int * labels,                          /* labels, in CPU memory */
                                        const int * labelLengths,                    /* the length of each label, in CPU memory */
                                        const int * inputLengths,                    /* the lengths of timing steps in each batch, in CPU memory */
                                        void * costs,                                /* the returned costs of CTC, in GPU memory */
                                        const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
                                        const void * gradients,                      /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
                                        cudnnCTCLossAlgo_t algo,                     /* algorithm selected, supported now 0 and 1 */
                                        cudnnCTCLossDescriptor_t ctcLossDesc,
                                        void * workspace,                            /* pointer to the workspace, in GPU memory */
                                        size_t workSpaceSizeInBytes);                /* the workspace size needed */

/* return the workspace size needed for ctc */
cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       probsDesc,       /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size) */
                                const cudnnTensorDescriptor_t       gradientsDesc,   /* Tensor descriptor for gradients, the dimensions are T,N,A. To compute costs only, set it to NULL */
                                const int                          * labels,         /* labels, in CPU memory */
                                const int                          * labelLengths,   /* the length of each label, in CPU memory */
                                const int                          * inputLengths,   /* the lengths of timing steps in each batch, in CPU memory */
                                cudnnCTCLossAlgo_t                  algo,            /* algorithm selected, supported now 0 and 1 */
                                cudnnCTCLossDescriptor_t            ctcLossDesc,
                                size_t                             *sizeInBytes );   /* pointer to the returned workspace size */


/* DEPRECATED routines to be removed next release : 
   User should use the non-suffixed version (which has the API and functionality of _v6 version)
   Routines with _v5 suffix has the functionality of the non-suffixed routines in the CUDNN V6
 */

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(cudnnHandle_t            handle,
                                                   cudnnRNNDescriptor_t     rnnDesc,
                                                   const int                hiddenSize,
                                                   const int                numLayers,
                                                   cudnnDropoutDescriptor_t dropoutDesc, /* Between layers, not between recurrent steps. */
                                                   cudnnRNNInputMode_t      inputMode,          
                                                   cudnnDirectionMode_t     direction,
                                                   cudnnRNNMode_t           mode,
                                                   cudnnRNNAlgo_t           algo,
                                                   cudnnDataType_t          dataType);

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t     rnnDesc,
                                                int                      hiddenSize,
                                                int                      numLayers,
                                                cudnnDropoutDescriptor_t dropoutDesc, /* Between layers, not between recurrent steps. */
                                                cudnnRNNInputMode_t      inputMode,
                                                cudnnDirectionMode_t     direction,
                                                cudnnRNNMode_t           mode,
                                                cudnnDataType_t          dataType);
#if defined (__cplusplus)
}
#endif

#endif /* CUDNN_H_ */

