/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef ISAAC_DRIVER_CUBLAS_H
#define ISAAC_DRIVER_CUBLAS_H

#include "isaac/templates/common.hpp"
#include "isaac/driver/dispatch.h"
#include "isaac/driver/buffer.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/backend.h"

namespace isaac
{
namespace driver
{

template<typename... Args> void cublasGemm_impl(half, Args... args){ driver::dispatch::cublasHgemm(args...); }
template<typename... Args> void cublasGemm_impl(float, Args... args){ driver::dispatch::cublasSgemm_v2(args...); }
template<typename... Args> void cublasGemm_impl(double, Args... args){ driver::dispatch::cublasDgemm_v2(args...); }


template<class cuType>
inline void cublasGemm_dispatch(Stream& stream, char AT, char BT, int32_t M, int32_t N, int32_t K, void* alpha, Buffer const & A, int32_t lda, Buffer const & B, int32_t ldb, void* beta, Buffer& C, int32_t ldc){
  auto cu_trans = [](char xt) { return (xt=='N')?CUBLAS_OP_N:CUBLAS_OP_T; };
  cublasHandle_t handle = dispatch::cublasHandle(stream.context());
  dispatch::cublasSetStream_v2(handle, (CUstream)stream);
  CUdeviceptr cuA = A, cuB = B, cuC = C;
  cublasGemm_impl(cuType(), handle, cu_trans(AT), cu_trans(BT), M, N, K, (cuType*)alpha, (const cuType*)cuA, lda, (const cuType*)cuB, ldb, (cuType*)beta, (cuType*)cuC, ldc);
}

inline void cublasGemm(DType dtype, Stream& stream, char AT, char BT, int32_t M, int32_t N, int32_t K, scalar alpha, Buffer const & A, int32_t lda, Buffer const & B, int32_t ldb, scalar beta, Buffer& C, int32_t ldc){
  ContextSwitcher ctx_switch(stream.context());
  switch(dtype){
    case HALF_TYPE: return cublasGemm_dispatch<half>(stream, AT, BT, M, N, K, alpha.data(), A, lda, B, ldb, beta.data(), C, ldc);
    case FLOAT_TYPE: return cublasGemm_dispatch<float>(stream, AT, BT, M, N, K, alpha.data(), A, lda, B, ldb, beta.data(), C, ldc);
    case DOUBLE_TYPE: return cublasGemm_dispatch<double>(stream, AT, BT, M, N, K, alpha.data(), A, lda, B, ldb, beta.data(), C, ldc);
    default: throw;
  }
}

inline cudnnDataType_t cudnnDtype(DType dtype){
  switch(dtype){
    case HALF_TYPE: return CUDNN_DATA_HALF;
    case FLOAT_TYPE: return CUDNN_DATA_FLOAT;
    case DOUBLE_TYPE: return CUDNN_DATA_DOUBLE;
  }
  throw;
}

inline void cudnnConv(DType dtype, Stream& stream, int32_t H, int32_t W, int32_t N, int32_t K, int32_t P, int32_t Q, int32_t C, int32_t R, int32_t S,
                      int32_t pad_h, int32_t pad_w, int32_t stride_h, int32_t stride_w, scalar alpha, Buffer const & I, Buffer const & F, scalar beta, Buffer const & O){
  driver::Context const & ctx = stream.context();

//  ContextSwitcher switch_ctx(ctx);
//  CUcontext cuctx;
  dispatch::cuCtxSetCurrent(ctx);
//  std::cout << cuctx << " " << CUcontext(ctx) << std::endl;

  cudnnHandle_t handle = dispatch::cudnnHandle(ctx);
  cudnnDataType_t cutype = cudnnDtype(dtype);

  dispatch::cudnnSetStream(handle, (CUstream)stream);
  cudnnTensorDescriptor_t tO, tI;
  cudnnFilterDescriptor_t tF;
  cudnnConvolutionDescriptor_t conv;
  cudnnConvolutionFwdAlgo_t algo;
  dispatch::cudnnCreateTensorDescriptor(&tO);
  dispatch::cudnnCreateTensorDescriptor(&tI);
  dispatch::cudnnCreateFilterDescriptor(&tF);

  dispatch::cudnnSetTensor4dDescriptor(tO, CUDNN_TENSOR_NCHW, cutype, N, K, P, Q);
  dispatch::cudnnSetFilter4dDescriptor(tF, cutype, CUDNN_TENSOR_NCHW, K, C, R, S);
  dispatch::cudnnSetTensor4dDescriptor(tI, CUDNN_TENSOR_NCHW, cutype, N, C, H, W);

  dispatch::cudnnCreateConvolutionDescriptor(&conv);
  int pad[] = {pad_h, pad_w};
  int stride[] = {stride_h, stride_w};
  int upscale[] = {1, 1};
  dispatch::cudnnSetConvolutionNdDescriptor(conv, 2, pad, stride, upscale, CUDNN_CROSS_CORRELATION, cutype);
//  dispatch::cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION);

//  dispatch::cudnnGetConvolutionForwardAlgorithm(handle, tI, tF, conv, tO, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 1024*1024, &algo);
  algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  size_t workspace_size;
  dispatch::cudnnGetConvolutionForwardWorkspaceSize(handle, tI, tF, conv, tO, algo, &workspace_size);
  Buffer work(ctx, std::max((size_t)1,workspace_size));
  CUdeviceptr twork = work;
  CUdeviceptr pI = I, pF = F, pO = O;
  dispatch::cudnnConvolutionForward(handle, alpha.data(), tI, (void*)pI, tF, (void*)pF, conv, algo, (void*)twork, workspace_size, beta.data(), tO, (void*)pO);
}






}
}



#endif
