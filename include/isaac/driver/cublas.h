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
#include "isaac/driver/error.h"
#include "isaac/tools/bench.hpp"
#include "isaac/tools/collections.hpp"

namespace isaac
{
namespace driver
{

enum cublasStrategy_t{
    CUBLAS_PREFER_FASTEST,
    CUBLAS_HEURISTICS
};


static const std::vector<cublasGemmAlgo_t> cublasAlgorithms = {
  CUBLAS_GEMM_DFALT, CUBLAS_GEMM_ALGO0, CUBLAS_GEMM_ALGO1, CUBLAS_GEMM_ALGO2, CUBLAS_GEMM_ALGO3,
  CUBLAS_GEMM_ALGO4, CUBLAS_GEMM_ALGO5, CUBLAS_GEMM_ALGO6, CUBLAS_GEMM_ALGO7
};

static const std::map<DType, cudaDataType> cudtype = {{HALF_TYPE, CUDA_R_16F}, {FLOAT_TYPE, CUDA_R_32F}, {DOUBLE_TYPE,CUDA_R_64F}};
static const std::map<char, cublasOperation_t> cuop = {{'N', CUBLAS_OP_N}, {'T', CUBLAS_OP_T}};

cublasGemmAlgo_t cublasGemmFastest(Stream& stream, cublasHandle_t handle, cudaDataType cudt, cublasOperation_t AT, cublasOperation_t BT, int32_t M, int32_t N, int32_t K,
                         void* alpha, CUdeviceptr A, int32_t lda, CUdeviceptr B, int32_t ldb,
                         void* beta, CUdeviceptr C, int32_t ldc){

  typedef std::tuple<cudaDataType_t, cublasOperation_t, cublasOperation_t, int32_t, int32_t, int32_t> key_t;
  // Benchmark fastest algorithm in cublasGemmEx
  auto benchmark_fastest = [&](key_t const &){
    std::vector<double> times;
    for(cublasGemmAlgo_t a: cublasAlgorithms){
      try{
        times.push_back(bench([&](){ dispatch::cublasGemmEx(handle, AT, BT, M, N, K, alpha, (const void*)A, cudt, lda, (const void*)B, cudt, ldb, beta, (void*)C, cudt, ldc, cudt, a); },
        [&](){ stream.synchronize(); },
        stream.context().device()));
      }catch(driver::exception::cublas::base const &){
        times.push_back(INFINITY);
      }
    }
    size_t argmin = std::min_element(times.begin(), times.end()) - times.begin();
    return cublasAlgorithms[argmin];
  };
  // Cache result
  static cpp::CachedMap<key_t, cublasGemmAlgo_t> cache(benchmark_fastest);
  return cache.get(std::make_tuple(cudt, AT, BT, M, N, K));
}

/* Wrapper for cublasGemmEx */
inline void cublasGemmEx(cublasHandle_t handle, cudaDataType cudt, cublasOperation_t AT, cublasOperation_t BT, int32_t M, int32_t N, int32_t K,
                         void* alpha, CUdeviceptr A, int32_t lda, CUdeviceptr B, int32_t ldb,
                         void* beta, CUdeviceptr C, int32_t ldc, cublasGemmAlgo_t algo)
{ dispatch::cublasGemmEx(handle, AT, BT, M, N, K, alpha, (const void*)A, cudt, lda, (const void*)B, cudt, ldb, beta, (void*)C, cudt, ldc, cudt, algo); }


/* Simplified API for default GEMM */
inline void cublasGemm(DType dtype, Stream& stream, char cAT, char cBT, int32_t M, int32_t N, int32_t K, scalar alpha, Buffer const & A, int32_t lda, Buffer const & B, int32_t ldb, scalar beta, Buffer& C, int32_t ldc, cublasGemmAlgo_t* fastest = NULL, cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT){
  ContextSwitcher ctx_switch(stream.context());
  cublasHandle_t handle = dispatch::cublasHandle(stream.context());
  dispatch::cublasSetStream_v2(handle, (CUstream)stream);
  if(fastest)
    *fastest = cublasGemmFastest(stream, handle, cudtype.at(dtype), cuop.at(cAT), cuop.at(cBT), M, N, K, alpha.data(), A, lda, B, ldb, beta.data(), C, ldc);
  else
    cublasGemmEx(handle, cudtype.at(dtype), cuop.at(cAT), cuop.at(cBT), M, N, K, alpha.data(), A, lda, B, ldb, beta.data(), C, ldc, algo);
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
  ContextSwitcher switch_ctx(ctx);

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

//  dispatch::cudnnGetConvolutionForwardAlgorithm(handle, tI, tF, conv, tO, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 1024*1024, &algo);
  algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  size_t workspace_size;
  dispatch::cudnnGetConvolutionForwardWorkspaceSize(handle, tI, tF, conv, tO, algo, &workspace_size);
//  Buffer work(ctx, std::max((size_t)1,workspace_size));
  static Buffer work(ctx, 1024*1024*16);
  CUdeviceptr twork = work;
  CUdeviceptr pI = I, pF = F, pO = O;
  dispatch::cudnnConvolutionForward(handle, alpha.data(), tI, (void*)pI, tF, (void*)pF, conv, algo, (void*)twork, workspace_size, beta.data(), tO, (void*)pO);
}






}
}



#endif
