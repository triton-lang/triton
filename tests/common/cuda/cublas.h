/* Copyright 2019 Philippe Tillet
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
#include "forward.h"
#include "triton/driver/buffer.h"
#include "triton/driver/stream.h"
#include "triton/driver/context.h"
#include "triton/driver/error.h"
#include "triton/tools/bench.hpp"


class cublas {
private:
  template <class F>
  struct return_type;

  template <class R, class... A>
  struct return_type<R (*)(A...)>
  { typedef R type; };

  typedef bool (*f_init_t)();

  template<f_init_t initializer, typename FunPtrT, typename... Args>
  static typename return_type<FunPtrT>::type f_impl(void*& lib_h, FunPtrT, void*& cache, const char * name, Args... args)
  {
    initializer();
    if(cache == nullptr){
      cache = dlsym(lib_h, name);
      if(cache == 0)
        throw std::runtime_error("dlsym unable to load function");
    }
    FunPtrT fptr;
    *reinterpret_cast<void **>(&fptr) = cache;
    typename return_type<FunPtrT>::type res = (*fptr)(args...);
    triton::driver::check(res);
    return res;
  }

public:
  static bool cublasinit();
  static cublasStatus_t cublasSetMathMode(cublasHandle_t h, cublasMath_t m);
  static cublasStatus_t cublasCreate_v2(cublasHandle_t* h);
  static cublasStatus_t cublasGetStream_v2(cublasHandle_t h, cudaStream_t *streamId);
  static cublasStatus_t cublasSetStream_v2(cublasHandle_t h, cudaStream_t streamId);
  static cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                     int m, int n, int k,
                                     const void *alpha, const void *A, cudaDataType Atype, int lda,
                                     const void *B, cudaDataType Btype, int ldb, const void *beta,
                                     void *C, cudaDataType Ctype, int ldc,
                                     cudaDataType computeType, cublasGemmAlgo_t algo);

private:
  static void* so_;
  static void* cublasGetStream_v2_;
  static void* cublasSetStream_v2_;
  static void* cublasCreate_v2_;
  static void* cublasGemmEx_;
  static void* cublasSetMathMode_;
};

void* cublas::so_;
void* cublas::cublasGetStream_v2_;
void* cublas::cublasSetStream_v2_;
void* cublas::cublasCreate_v2_;
void* cublas::cublasGemmEx_;
void* cublas::cublasSetMathMode_;


bool cublas::cublasinit() {
  if(so_==nullptr)
    so_ = dlopen("libcublas.so", RTLD_LAZY);
  return so_ != nullptr;
}

cublasStatus_t cublas::cublasGetStream_v2(cublasHandle_t h, cudaStream_t *a)
{ return f_impl<cublas::cublasinit>(so_, cublasGetStream_v2, cublasGetStream_v2_, "cublasGetStream_v2", h, a); }
cublasStatus_t cublas::cublasSetStream_v2(cublasHandle_t h, cudaStream_t a)
{ return f_impl<cublas::cublasinit>(so_, cublasSetStream_v2, cublasSetStream_v2_, "cublasSetStream_v2", h, a); }
cublasStatus_t cublas::cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                    const void *alpha, const void *A, cudaDataType Atype, int lda,
                                    const void *B, cudaDataType Btype, int ldb, const void *beta,
                                    void *C, cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo) {
  return f_impl<cublas::cublasinit>(so_, cublasGemmEx, cublasGemmEx_, "cublasGemmEx", handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}
cublasStatus_t cublas::cublasCreate_v2(cublasHandle_t *h) {
  return f_impl<cublas::cublasinit>(so_, cublasCreate_v2, cublasCreate_v2_, "cublasCreate_v2", h);
}
cublasStatus_t cublas::cublasSetMathMode(cublasHandle_t h, cublasMath_t m) {
  return f_impl<cublas::cublasinit>(so_, cublasSetMathMode, cublasSetMathMode_, "cublasSetMathMode", h, m);
}



inline cublasGemmAlgo_t cublasGemmFastest(
                         triton::driver::stream* stream,
                         cublasHandle_t handle, cudaDataType cudt,
                         cublasOperation_t AT, cublasOperation_t BT,
                         int32_t M, int32_t N, int32_t K,
                         void* alpha, CUdeviceptr A, int32_t lda, CUdeviceptr B, int32_t ldb,
                         void* beta, CUdeviceptr C, int32_t ldc) {

  // initialize list of cublas algorithms
  static std::vector<cublasGemmAlgo_t> algorithms;
  if(algorithms.empty()) {
    // non-tensor ops
    for(int i = -1; i < 24; i++)
      algorithms.push_back((cublasGemmAlgo_t)i);
    // tensor ops
    for(int i = 99; i < 116; i++)
      algorithms.push_back((cublasGemmAlgo_t)i);
  }

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
      double nanosec = triton::tools::bench([&](){ status = cublas::cublasGemmEx(handle, AT, BT,
                                                                 M, N, K,
                                                                 alpha, (const void*)A, cudt, lda,
                                                                 (const void*)B, cudt, ldb,
                                                                 beta, (void*)C, cudt, ldc, CUDA_R_32F,
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




/* Get cuBLAS handle */
inline cublasHandle_t cublasGetHandle(triton::driver::stream* stream) {
  static std::map<CUstream, cublasHandle_t> cache;
  CUstream key = *stream->cu();

  // create handle if necessary
  if(cache.find(key) == cache.end()) {
    cublasHandle_t handle;
    if(cublas::cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Error: could not create cuBLAS handle");
    cublas::cublasSetStream_v2(handle, key);
    cache.insert({key, handle});
  }

  // return handle for the stream
  return cache.at(key);
}



/* Simplified API for default GEMM */
inline void cublasGemm(cublasDataType_t dtype,
                       triton::driver::stream* stream,
                       bool AT, bool BT,
                       int32_t M, int32_t N, int32_t K,
                       void* alpha, triton::driver::buffer* A, int32_t lda,
                       triton::driver::buffer* B, int32_t ldb,
                       void* beta, triton::driver::buffer* C, int32_t ldc,
                       cublasGemmAlgo_t* fastest = NULL, cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT) {
  // get handle
  static cublasHandle_t handle = cublasGetHandle(stream);
  // set math mode
  if(dtype == CUDA_R_16F)
    cublas::cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  // cuda types
  static const std::map<char, cublasOperation_t> cu_op = {
    {false, CUBLAS_OP_N},
    {true, CUBLAS_OP_T}
  };
  cublasOperation_t opa = cu_op.at(AT);
  cublasOperation_t opb = cu_op.at(BT);
  // benchmark fastest
  if(fastest)
    *fastest = cublasGemmFastest(stream, handle, dtype, opa, opb, M, N, K, alpha, *A->cu(), lda, *B->cu(), ldb, beta, *C->cu(), ldc);
  else {
    // execute supplied algo
    cublasStatus_t status = cublas::cublasGemmEx(handle, opa, opb, M, N, K,
                                                 alpha, (const void*)*A->cu(), dtype, lda,
                                                 (const void*)*B->cu(), dtype, ldb,
                                                 beta, (void*)*C->cu(), dtype, ldc, CUDA_R_32F, algo);
  }
}
