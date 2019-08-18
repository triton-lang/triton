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

#include <algorithm>
#include <vector>
#include <cassert>
#include "cublas_v2.h"
#include "triton/driver/buffer.h"
#include "triton/driver/stream.h"
#include "triton/driver/context.h"
#include "triton/tools/bench.hpp"

enum cublasStrategy_t{
    CUBLAS_PREFER_FASTEST,
    CUBLAS_HEURISTICS
};

enum DType{
  HALF_TYPE,
  FLOAT_TYPE,
  DOUBLE_TYPE,
};

inline size_t size_of(DType dtype){
  switch (dtype) {
  case HALF_TYPE: return 2;
  case FLOAT_TYPE: return 4;
  case DOUBLE_TYPE: return 8;
  default: throw;
  }
}

std::vector<cublasGemmAlgo_t> gather_all_algos() {
  std::vector<cublasGemmAlgo_t> result;
  // non-tensor ops
  for(int i = -1; i < 24; i++)
    result.push_back((cublasGemmAlgo_t)i);
  // tensor ops
  for(int i = 99; i < 116; i++)
    result.push_back((cublasGemmAlgo_t)i);
  return result;
}

static const std::vector<cublasGemmAlgo_t> algorithms = gather_all_algos();

static const std::map<DType, cudaDataType> cu_dtype = {
  {HALF_TYPE, CUDA_R_16F},
  {FLOAT_TYPE, CUDA_R_32F},
  {DOUBLE_TYPE, CUDA_R_64F}
};

static const std::map<char, cublasOperation_t> cu_op = {
  {false, CUBLAS_OP_N},
  {true, CUBLAS_OP_T}
};

inline cublasGemmAlgo_t cublasGemmFastest(
                         triton::driver::stream* stream,
                         cublasHandle_t handle, cudaDataType cudt,
                         cublasOperation_t AT, cublasOperation_t BT,
                         int32_t M, int32_t N, int32_t K,
                         void* alpha, CUdeviceptr A, int32_t lda, CUdeviceptr B, int32_t ldb,
                         void* beta, CUdeviceptr C, int32_t ldc) {

  // cache to avoid re-benchmarking
  typedef std::tuple<cudaDataType_t,
                     cublasOperation_t, cublasOperation_t,
                     int32_t, int32_t, int32_t> key_t;
  static std::map<key_t, cublasGemmAlgo_t> cache;
  key_t key(cudt, AT, BT, M, N, K);
  // benchmark algorithms if necessary
  if(cache.find(key) == cache.end()){
    std::vector<double> times;
    for(cublasGemmAlgo_t a: algorithms) {
      cublasStatus_t status;
      double nanosec = triton::tools::bench([&](){ status = cublasGemmEx(handle, AT, BT,
                                                                 M, N, K,
                                                                 alpha, (const void*)A, cudt, lda,
                                                                 (const void*)B, cudt, ldb,
                                                                 beta, (void*)C, cudt, ldc, cudt,
                                                                 a); }, stream);
      if(status != CUBLAS_STATUS_SUCCESS)
        nanosec = INFINITY;
    }
    size_t argmin = std::min_element(times.begin(), times.end()) - times.begin();
    assert(times[argmin] != INFINITY);
    cache.insert({key, algorithms[argmin]});
  }

  // return best algorithm
  return cache.at(key);
}

/* Wrapper for cublasGemmEx */
inline cublasStatus_t cublasGemmEx(cublasHandle_t handle, cudaDataType cudt, cublasOperation_t AT, cublasOperation_t BT, int32_t M, int32_t N, int32_t K,
                         void* alpha, CUdeviceptr A, int32_t lda, CUdeviceptr B, int32_t ldb,
                         void* beta, CUdeviceptr C, int32_t ldc, cublasGemmAlgo_t algo)
{
  cublasStatus_t status = cublasGemmEx(handle, AT, BT, M, N, K, alpha, (const void*)A, cudt, lda, (const void*)B, cudt, ldb, beta, (void*)C, cudt, ldc, cudt, algo);
  if(status != CUBLAS_STATUS_SUCCESS){
    std::cout << status;
    exit(EXIT_FAILURE);
  }
}


/* Get cuBLAS handle */
cublasHandle_t cublasGetHandle(triton::driver::stream* stream) {
  static std::map<CUstream, cublasHandle_t> cache;
  CUstream key = *stream->cu();

  // create handle if necessary
  if(cache.find(key) == cache.end()) {
    cublasHandle_t handle;
    if(cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Error: could not create cuBLAS handle");
    cublasSetStream_v2(handle, key);
    cache.insert({key, handle});
  }

  // return handle for the stream
  return cache.at(key);
}

/* Simplified API for default GEMM */
inline void cublasGemm(DType dtype, triton::driver::stream* stream, bool AT, bool BT,
                       int32_t M, int32_t N, int32_t K,
                       void* alpha, triton::driver::buffer* A, int32_t lda,
                       triton::driver::buffer* B, int32_t ldb,
                       void* beta, triton::driver::buffer* C, int32_t ldc,
                       cublasGemmAlgo_t* fastest = NULL, cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT) {
  triton::driver::cu_context::context_switcher scope(*stream->context());
  static cublasHandle_t handle = cublasGetHandle(stream);
  if(dtype == HALF_TYPE)
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  cublasStatus_t status;
  if(fastest)
    *fastest = cublasGemmFastest(stream, handle, cu_dtype.at(dtype), cu_op.at(AT), cu_op.at(BT), M, N, K, alpha, *A->cu(), lda, *B->cu(), ldb, beta, *C->cu(), ldc);
  else
    status = cublasGemmEx(handle, cu_dtype.at(dtype), cu_op.at(AT), cu_op.at(BT), M, N, K, alpha, *A->cu(), lda, *B->cu(), ldb, beta, *C->cu(), ldc, algo);
}
