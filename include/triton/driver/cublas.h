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

#ifndef TDL_INCLUDE_DRIVER_CUBLAS_H
#define TDL_INCLUDE_DRIVER_CUBLAS_H

#include "isaac/templates/common.hpp"
#include "triton/driver/dispatch.h"
#include "triton/driver/buffer.h"
#include "triton/driver/stream.h"
#include "triton/driver/backend.h"
#include "triton/driver/error.h"
#include "triton/tools/bench.hpp"
#include "triton/tools/collections.hpp"

namespace tdl
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

static const std::map<DType, cudaDataType> cudtype = {{FLOAT_TYPE, CUDA_R_32F}, {DOUBLE_TYPE,CUDA_R_64F}};
static const std::map<char, cublasOperation_t> cuop = {{'N', CUBLAS_OP_N}, {'T', CUBLAS_OP_T}};

inline cublasGemmAlgo_t cublasGemmFastest(Stream& stream, cublasHandle_t handle, cudaDataType cudt, cublasOperation_t AT, cublasOperation_t BT, int32_t M, int32_t N, int32_t K,
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
    case INT8X4_TYPE: return CUDNN_DATA_INT8x4;
    case INT32_TYPE: return CUDNN_DATA_INT32;
    case FLOAT_TYPE: return CUDNN_DATA_FLOAT;
    case DOUBLE_TYPE: return CUDNN_DATA_DOUBLE;
  }
  throw;
}

inline cudnnTensorFormat_t format(cudnnDataType_t cutype){
  switch(cutype){
    case CUDNN_DATA_INT8x4: return CUDNN_TENSOR_NCHW_VECT_C;
    default: return CUDNN_TENSOR_NCHW;
  }
}

inline void cudnnConv(DType dtype, Stream& stream, int32_t D, int32_t H, int32_t W, int32_t N, int32_t K, int32_t M, int32_t P, int32_t Q, int32_t C, int32_t T, int32_t R, int32_t S,
                      int32_t pad_d, int32_t pad_h, int32_t pad_w, int32_t stride_d, int32_t stride_h, int32_t stride_w, scalar alpha, Buffer const & I, Buffer const & F, scalar beta, Buffer const & O){
  driver::Context const & ctx = stream.context();
  ContextSwitcher switch_ctx(ctx);

  std::vector<int> pad = {pad_d, pad_h, pad_w};
  std::vector<int> stride = {stride_d, stride_h, stride_w};
  std::vector<int> upscale = {1, 1, 1};
  std::vector<int> Oshapes = {N, K, M, P, Q};
  std::vector<int> Fshapes = {K, C, T, R, S};
  std::vector<int> Ishapes = {N, C, D, H, W};
  if(M == 1 && T == 1 && D == 1){
    pad.erase(pad.begin());
    stride.erase(stride.begin());
    upscale.erase(upscale.begin());
    Oshapes.erase(Oshapes.begin() + 2);
    Ishapes.erase(Ishapes.begin() + 2);
    Fshapes.erase(Fshapes.begin() + 2);
  }

  cudnnHandle_t handle = dispatch::cudnnHandle(ctx);
  cudnnDataType_t in_cutype = cudnnDtype(dtype);
  cudnnDataType_t conv_cutype = (dtype == INT8X4_TYPE)?CUDNN_DATA_INT32:in_cutype;

  dispatch::cudnnSetStream(handle, (CUstream)stream);
  cudnnTensorDescriptor_t tO, tI;
  cudnnFilterDescriptor_t tF;
  cudnnConvolutionDescriptor_t conv;
  cudnnConvolutionFwdAlgo_t algo;
  dispatch::cudnnCreateTensorDescriptor(&tO);
  dispatch::cudnnCreateTensorDescriptor(&tI);
  dispatch::cudnnCreateFilterDescriptor(&tF);

  dispatch::cudnnSetTensorNdDescriptorEx(tO, format(in_cutype), in_cutype, Oshapes.size(), Oshapes.data());
  dispatch::cudnnSetFilterNdDescriptor(tF, in_cutype, format(in_cutype), Fshapes.size(), Fshapes.data());
  dispatch::cudnnSetTensorNdDescriptorEx(tI, format(in_cutype), in_cutype, Ishapes.size(), Ishapes.data());

  dispatch::cudnnCreateConvolutionDescriptor(&conv);
  dispatch::cudnnSetConvolutionNdDescriptor(conv, pad.size(), pad.data(), stride.data(), upscale.data(), CUDNN_CROSS_CORRELATION, conv_cutype);
  dispatch::cudnnGetConvolutionForwardAlgorithm(handle, tI, tF, conv, tO, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 1024*1024*64, &algo);

  size_t workspace_size;
  dispatch::cudnnGetConvolutionForwardWorkspaceSize(handle, tI, tF, conv, tO, algo, &workspace_size);
  static Buffer work(ctx, 1024*1024*64);
  CUdeviceptr twork = work;
  CUdeviceptr pI = I, pF = F, pO = O;
  dispatch::cudnnConvolutionForward(handle, alpha.data(), tI, (void*)pI, tF, (void*)pF, conv, algo, (void*)twork, workspace_size, beta.data(), tO, (void*)pO);
}


inline void cudnnPool(DType dtype, Stream& stream, int32_t D, int32_t H, int32_t W, int32_t N, int32_t K, int32_t M, int32_t P, int32_t Q, int32_t T, int32_t R, int32_t S,
                      int32_t pad_d, int32_t pad_h, int32_t pad_w, int32_t stride_d, int32_t stride_h, int32_t stride_w, scalar alpha, Buffer const & I, scalar beta, Buffer const & O){
  driver::Context const & ctx = stream.context();
  ContextSwitcher switch_ctx(ctx);

  std::vector<int> pad = {pad_d, pad_h, pad_w};
  std::vector<int> stride = {stride_d, stride_h, stride_w};
  std::vector<int> upscale = {1, 1, 1};
  std::vector<int> Oshapes = {N, K, M, P, Q};
  std::vector<int> Ishapes = {N, K, D, H, W};
  std::vector<int> window = {T, R, S};
  if(M == 1 && T == 1 && D == 1){
    window.erase(window.begin());
    pad.erase(pad.begin());
    stride.erase(stride.begin());
    upscale.erase(upscale.begin());
    Oshapes.erase(Oshapes.begin() + 2);
    Ishapes.erase(Ishapes.begin() + 2);
  }

  cudnnHandle_t handle = dispatch::cudnnHandle(ctx);
  cudnnDataType_t cutype = cudnnDtype(dtype);

  dispatch::cudnnSetStream(handle, (CUstream)stream);
  cudnnTensorDescriptor_t tO, tI;
  cudnnPoolingDescriptor_t desc;
  dispatch::cudnnCreateTensorDescriptor(&tO);
  dispatch::cudnnCreateTensorDescriptor(&tI);

  dispatch::cudnnSetTensorNdDescriptorEx(tO, CUDNN_TENSOR_NCHW, cutype, Oshapes.size(), Oshapes.data());
  dispatch::cudnnSetTensorNdDescriptorEx(tI, CUDNN_TENSOR_NCHW, cutype, Ishapes.size(), Ishapes.data());

  dispatch::cudnnCreatePoolingDescriptor(&desc);
  dispatch::cudnnSetPoolingNdDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, window.size(), window.data(), pad.data(), stride.data());

  CUdeviceptr pI = I, pO = O;
  dispatch::cudnnPoolingForward(handle, desc, alpha.data(), tI, (void*)pI, beta.data(), tO, (void*)pO);
}

inline void cudnnTransformTensor(driver::Stream & stream,
               DType in_dtype, DType out_dtype,
               cudnnTensorFormat_t in_layout, cudnnTensorFormat_t out_layout,
               int32_t N, int32_t C, int32_t D, int32_t H, int32_t W,
               scalar alpha, driver::Buffer const & I, scalar beta, driver::Buffer& O)
{
  cudnnHandle_t handle = dispatch::cudnnHandle(stream.context());
  dispatch::cudnnSetStream(handle, (CUstream)stream);

  cudnnTensorDescriptor_t tO, tI;
  std::vector<int> shapes = {N, C, D, H, W};
  dispatch::cudnnCreateTensorDescriptor(&tI);
  dispatch::cudnnSetTensorNdDescriptorEx(tI, in_layout, cudnnDtype(in_dtype), shapes.size(), shapes.data());
  dispatch::cudnnCreateTensorDescriptor(&tO);
  dispatch::cudnnSetTensorNdDescriptorEx(tO, out_layout, cudnnDtype(out_dtype), shapes.size(), shapes.data());

  CUdeviceptr pI = I, pO = O;
  dispatch::cudnnTransformTensor(handle, alpha.data(), tI, (void*)pI, beta.data(), tO, (void*)pO);
}


}
}



#endif
