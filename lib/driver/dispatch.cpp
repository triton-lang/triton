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

#include <map>
#include "triton/driver/dispatch.h"
#include "triton/driver/context.h"

namespace triton
{
namespace driver
{

//Helpers for function definition
#define DEFINE0(init, hlib, ret, fname) ret dispatch::fname()\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname); }

#define DEFINE1(init, hlib, ret, fname, t1) ret dispatch::fname(t1 a)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a); }

#define DEFINE2(init, hlib, ret, fname, t1, t2) ret dispatch::fname(t1 a, t2 b)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b); }

#define DEFINE3(init, hlib, ret, fname, t1, t2, t3) ret dispatch::fname(t1 a, t2 b, t3 c)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c); }

#define DEFINE4(init, hlib, ret, fname, t1, t2, t3, t4) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d); }

#define DEFINE5(init, hlib, ret, fname, t1, t2, t3, t4, t5) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e); }

#define DEFINE6(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f); }

#define DEFINE7(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g); }

#define DEFINE8(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h); }

#define DEFINE9(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h, t9 i)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h, i); }

#define DEFINE10(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h, t9 i, t10 j)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h, i, j); }

#define DEFINE11(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h, t9 i, t10 j, t11 k)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h, i, j, k); }

#define DEFINE13(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h, t9 i, t10 j, t11 k, t12 l, t13 m)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h, i, j, k, l, m); }

#define DEFINE19(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h, t9 i, t10 j, t11 k, t12 l, t13 m, t14 n, t15 o, t16 p, t17 q, t18 r, t19 s)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s); }

//Specialized helpers for CUDA
#define CUDA_DEFINE1(ret, fname, t1) DEFINE1(cuinit, cuda_, ret, fname, t1)
#define CUDA_DEFINE2(ret, fname, t1, t2) DEFINE2(cuinit, cuda_, ret, fname, t1, t2)
#define CUDA_DEFINE3(ret, fname, t1, t2, t3) DEFINE3(cuinit, cuda_, ret, fname, t1, t2, t3)
#define CUDA_DEFINE4(ret, fname, t1, t2, t3, t4) DEFINE4(cuinit, cuda_, ret, fname, t1, t2, t3, t4)
#define CUDA_DEFINE5(ret, fname, t1, t2, t3, t4, t5) DEFINE5(cuinit, cuda_, ret, fname, t1, t2, t3, t4, t5)
#define CUDA_DEFINE6(ret, fname, t1, t2, t3, t4, t5, t6) DEFINE6(cuinit, cuda_, ret, fname, t1, t2, t3, t4, t5, t6)
#define CUDA_DEFINE7(ret, fname, t1, t2, t3, t4, t5, t6, t7) DEFINE7(cuinit, cuda_, ret, fname, t1, t2, t3, t4, t5, t6, t7)
#define CUDA_DEFINE8(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8) DEFINE8(cuinit, cuda_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8)
#define CUDA_DEFINE9(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9) DEFINE9(cuinit, cuda_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9)
#define CUDA_DEFINE10(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10) DEFINE10(cuinit, cuda_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
#define CUDA_DEFINE11(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11) DEFINE11(cuinit, cuda_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11)

#define NVML_DEFINE0(ret, fname) DEFINE0(nvmlinit, nvml_, ret, fname)
#define NVML_DEFINE1(ret, fname, t1) DEFINE1(nvmlinit, nvml_, ret, fname, t1)
#define NVML_DEFINE2(ret, fname, t1, t2) DEFINE2(nvmlinit, nvml_, ret, fname, t1, t2)
#define NVML_DEFINE3(ret, fname, t1, t2, t3) DEFINE3(nvmlinit, nvml_, ret, fname, t1, t2, t3)

#define CUBLAS_DEFINE1(ret, fname, t1) DEFINE1(cublasinit, cublas_, ret, fname, t1)
#define CUBLAS_DEFINE13(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13) DEFINE13(cublasinit, cublas_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13)
#define CUBLAS_DEFINE19(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19) DEFINE19(cublasinit, cublas_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19)

#define CUDNN_DEFINE1(ret, fname, t1) DEFINE1(cudnninit, cudnn_, ret, fname, t1)
#define CUDNN_DEFINE2(ret, fname, t1, t2) DEFINE2(cudnninit, cudnn_, ret, fname, t1, t2)
#define CUDNN_DEFINE3(ret, fname, t1, t2, t3) DEFINE3(cudnninit, cudnn_, ret, fname, t1, t2, t3)
#define CUDNN_DEFINE5(ret, fname, t1, t2, t3, t4, t5) DEFINE5(cudnninit, cudnn_, ret, fname, t1, t2, t3, t4, t5)
#define CUDNN_DEFINE6(ret, fname, t1, t2, t3, t4, t5, t6) DEFINE6(cudnninit, cudnn_, ret, fname, t1, t2, t3, t4, t5, t6)
#define CUDNN_DEFINE7(ret, fname, t1, t2, t3, t4, t5, t6, t7) DEFINE7(cudnninit, cudnn_, ret, fname, t1, t2, t3, t4, t5, t6, t7)
#define CUDNN_DEFINE8(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8) DEFINE8(cudnninit, cudnn_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8)
#define CUDNN_DEFINE13(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13) DEFINE13(cudnninit, cudnn_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13)


bool dispatch::cuinit(){
  if(cuda_==nullptr)
    cuda_ = dlopen("libcuda.so", RTLD_LAZY);
  CUresult (*fptr)(unsigned int);
	cuInit_ = dlsym(cuda_, "cuInit");
	*reinterpret_cast<void **>(&fptr) = cuInit_;
	CUresult res = (*fptr)(0);
	check(res);
  return cuda_ != nullptr;
}

bool dispatch::nvmlinit(){
  if(nvml_==nullptr)
    nvml_ = dlopen("libnvidia-ml.so", RTLD_LAZY);
  nvmlReturn_t (*fptr)();
  nvmlInit_v2_ = dlsym(nvml_, "nvmlInit_v2");
  *reinterpret_cast<void **>(&fptr) = nvmlInit_v2_;
  nvmlReturn_t res = (*fptr)();
  check(res);
  return res;
}

bool dispatch::cublasinit(){
  if(cublas_==nullptr)
    cublas_ = dlopen("libcublas.so", RTLD_LAZY);
  return cublas_ != nullptr;
}

bool dispatch::cudnninit(){
  if(cudnn_==nullptr)
    cudnn_ = dlopen("libcudnn.so", RTLD_LAZY);
  return cudnn_ != nullptr;
}

//CUDA
CUDA_DEFINE1(CUresult, cuCtxDestroy_v2, CUcontext)
CUDA_DEFINE2(CUresult, cuEventCreate, CUevent *, unsigned int)
CUDA_DEFINE2(CUresult, cuDeviceGet, CUdevice *, int)
CUDA_DEFINE3(CUresult, cuMemcpyDtoH_v2, void *, CUdeviceptr, size_t)
CUDA_DEFINE2(CUresult, cuStreamCreate, CUstream *, unsigned int)
CUDA_DEFINE3(CUresult, cuEventElapsedTime, float *, CUevent, CUevent)
CUDA_DEFINE1(CUresult, cuMemFree_v2, CUdeviceptr)
CUDA_DEFINE4(CUresult, cuMemcpyDtoHAsync_v2, void *, CUdeviceptr, size_t, CUstream)
CUDA_DEFINE1(CUresult, cuDriverGetVersion, int *)
CUDA_DEFINE3(CUresult, cuDeviceGetName, char *, int, CUdevice)
CUDA_DEFINE3(CUresult, cuDeviceGetPCIBusId, char *, int, CUdevice)
CUDA_DEFINE4(CUresult, cuModuleGetGlobal_v2, CUdeviceptr*, size_t*, CUmodule, const char*)

CUDA_DEFINE4(CUresult, cuMemcpyHtoDAsync_v2, CUdeviceptr, const void *, size_t, CUstream)
CUDA_DEFINE2(CUresult, cuModuleLoad, CUmodule *, const char *)
CUDA_DEFINE11(CUresult, cuLaunchKernel, CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **)
CUDA_DEFINE1(CUresult, cuModuleUnload, CUmodule)
CUDA_DEFINE5(CUresult, cuModuleLoadDataEx, CUmodule *, const void *, unsigned int, CUjit_option *, void **)
CUDA_DEFINE3(CUresult, cuDeviceGetAttribute, int *, CUdevice_attribute, CUdevice)
CUDA_DEFINE1(CUresult, cuDeviceGetCount, int *)
CUDA_DEFINE3(CUresult, cuMemcpyHtoD_v2, CUdeviceptr, const void *, size_t )
CUDA_DEFINE1(CUresult, cuInit, unsigned int)
CUDA_DEFINE2(CUresult, cuEventRecord, CUevent, CUstream)
CUDA_DEFINE3(CUresult, cuCtxCreate_v2, CUcontext *, unsigned int, CUdevice)
CUDA_DEFINE3(CUresult, cuModuleGetFunction, CUfunction *, CUmodule, const char *)
CUDA_DEFINE1(CUresult, cuStreamSynchronize, CUstream)
CUDA_DEFINE1(CUresult, cuStreamDestroy_v2, CUstream)
CUDA_DEFINE1(CUresult, cuEventDestroy_v2, CUevent)
CUDA_DEFINE2(CUresult, cuMemAlloc_v2, CUdeviceptr*, size_t)
CUDA_DEFINE3(CUresult, cuPointerGetAttribute, void*, CUpointer_attribute, CUdeviceptr)
CUDA_DEFINE1(CUresult, cuCtxGetDevice, CUdevice*)
CUDA_DEFINE1(CUresult, cuCtxGetCurrent, CUcontext*)
CUDA_DEFINE1(CUresult, cuCtxSetCurrent, CUcontext)
CUDA_DEFINE4(CUresult, cuMemsetD8Async, CUdeviceptr, unsigned char, size_t, CUstream)
CUDA_DEFINE1(CUresult, cuCtxPushCurrent_v2, CUcontext)
CUDA_DEFINE1(CUresult, cuCtxPopCurrent_v2, CUcontext*)

NVML_DEFINE2(nvmlReturn_t, nvmlDeviceGetHandleByPciBusId_v2, const char *, nvmlDevice_t*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceGetClockInfo, nvmlDevice_t, nvmlClockType_t, unsigned int*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceGetMaxClockInfo, nvmlDevice_t, nvmlClockType_t, unsigned int*)

cublasHandle_t dispatch::cublasHandle(driver::context const & ctx){
  static std::map<context, cublasHandle_t> handles;
  auto pr = handles.insert({ctx, cublasHandle_t()});
  if(pr.second)
    cublasCreate_v2(&pr.first->second);
  return pr.first->second;
}

cudnnHandle_t dispatch::cudnnHandle(driver::context const & ctx){
  static std::map<context, cudnnHandle_t> handles;
  auto pr = handles.insert({ctx, cudnnHandle_t()});
  if(pr.second)
    cudnnCreate(&pr.first->second);
  return pr.first->second;
}

CUBLAS_DEFINE1(cublasStatus_t, cublasCreate_v2, cublasHandle_t*)
cublasStatus_t dispatch::cublasGetStream_v2(cublasHandle_t h, cudaStream_t *a)
{ return f_impl<dispatch::cublasinit>(cublas_, cublasGetStream_v2, cublasGetStream_v2_, "cublasGetStream_v2", h, a); }
cublasStatus_t dispatch::cublasSetStream_v2(cublasHandle_t h, cudaStream_t a)
{ return f_impl<dispatch::cublasinit>(cublas_, cublasSetStream_v2, cublasSetStream_v2_, "cublasSetStream_v2", h, a); }
cublasStatus_t dispatch::cublasSgemm_v2(cublasHandle_t h, cublasOperation_t at, cublasOperation_t bt, int m, int n, int k, float* alpha, const float *A, int lda, const float *B, int ldb, float* beta, float *C, int ldc)
{ return f_impl<dispatch::cublasinit>(cublas_, cublasSgemm_v2, cublasSgemm_v2_, "cublasSgemm_v2", h, at, bt, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);}
cublasStatus_t dispatch::cublasDgemm_v2(cublasHandle_t h, cublasOperation_t at, cublasOperation_t bt, int m, int n, int k, double* alpha, const double *A, int lda, const double *B, int ldb, double* beta, double *C, int ldc)
{ return f_impl<dispatch::cublasinit>(cublas_, cublasDgemm_v2, cublasDgemm_v2_, "cublasDgemm_v2", h, at, bt, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);}
cublasStatus_t dispatch::cublasHgemm(cublasHandle_t h, cublasOperation_t at, cublasOperation_t bt, int m, int n, int k, half* alpha, const half *A, int lda, const half *B, int ldb, half* beta, half *C, int ldc)
{ return f_impl<dispatch::cublasinit>(cublas_, cublasHgemm, cublasHgemm_, "cublasHgemm", h, at, bt, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);}
CUBLAS_DEFINE19(cublasStatus_t, cublasGemmEx, cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType, int, const void*, cudaDataType, int, const void*, void*, cudaDataType, int, cudaDataType, cublasGemmAlgo_t)

//cuDNN
CUDNN_DEFINE1(cudnnStatus_t, cudnnCreateConvolutionDescriptor, cudnnConvolutionDescriptor_t*)
CUDNN_DEFINE1(cudnnStatus_t, cudnnCreateTensorDescriptor, cudnnTensorDescriptor_t*)
CUDNN_DEFINE1(cudnnStatus_t, cudnnCreateFilterDescriptor, cudnnFilterDescriptor_t*)
CUDNN_DEFINE1(cudnnStatus_t, cudnnCreate, cudnnHandle_t*)
CUDNN_DEFINE7(cudnnStatus_t, cudnnSetTensor4dDescriptor, cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int, int, int, int)
CUDNN_DEFINE7(cudnnStatus_t, cudnnSetFilter4dDescriptor, cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, int, int, int, int)
CUDNN_DEFINE5(cudnnStatus_t, cudnnSetTensorNdDescriptorEx, cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int, const int*)
CUDNN_DEFINE5(cudnnStatus_t, cudnnSetFilterNdDescriptor, cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, int, const int*)
CUDNN_DEFINE1(cudnnStatus_t, cudnnCreatePoolingDescriptor, cudnnPoolingDescriptor_t*)
CUDNN_DEFINE7(cudnnStatus_t, cudnnSetPoolingNdDescriptor, cudnnPoolingDescriptor_t, const cudnnPoolingMode_t, const cudnnNanPropagation_t, int, const int*, const int*, const int*)
CUDNN_DEFINE8(cudnnStatus_t, cudnnPoolingForward, cudnnHandle_t, const cudnnPoolingDescriptor_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*, const cudnnTensorDescriptor_t, void*)


CUDNN_DEFINE8(cudnnStatus_t, cudnnSetConvolution2dDescriptor, cudnnConvolutionDescriptor_t, int, int, int, int, int, int, cudnnConvolutionMode_t)
CUDNN_DEFINE7(cudnnStatus_t, cudnnSetConvolutionNdDescriptor, cudnnConvolutionDescriptor_t, int, const int*, const int*, const int*, cudnnConvolutionMode_t, cudnnDataType_t)
CUDNN_DEFINE8(cudnnStatus_t, cudnnGetConvolutionForwardAlgorithm, cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, cudnnConvolutionFwdPreference_t, size_t, cudnnConvolutionFwdAlgo_t *)
CUDNN_DEFINE7(cudnnStatus_t, cudnnGetConvolutionForwardWorkspaceSize, cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, cudnnConvolutionFwdAlgo_t, size_t*)
CUDNN_DEFINE13(cudnnStatus_t, cudnnConvolutionForward, cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *, const cudnnFilterDescriptor_t, const void *, const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void *, size_t, const void *, const cudnnTensorDescriptor_t, void *)
CUDNN_DEFINE2(cudnnStatus_t, cudnnSetStream, cudnnHandle_t, cudaStream_t)
CUDNN_DEFINE7(cudnnStatus_t, cudnnTransformTensor, cudnnHandle_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*, const cudnnTensorDescriptor_t, void*)


void dispatch::release(){
  if(cuda_){
    dlclose(cuda_);
    cuda_ = nullptr;
  }
  if(nvrtc_){
    dlclose(nvrtc_);
    nvrtc_ = nullptr;
  }
  if(cublas_){
    dlclose(cublas_);
    cublas_ = nullptr;
  }
  if(cudnn_){
    dlclose(cudnn_);
    cudnn_ = nullptr;
  }
}

void* dispatch::cuda_;
void* dispatch::nvrtc_;
void* dispatch::nvml_;
void* dispatch::cublas_;
void* dispatch::cudnn_;

//CUDA
void* dispatch::cuCtxGetCurrent_;
void* dispatch::cuCtxSetCurrent_;
void* dispatch::cuCtxDestroy_v2_;
void* dispatch::cuEventCreate_;
void* dispatch::cuDeviceGet_;
void* dispatch::cuMemcpyDtoH_v2_;
void* dispatch::cuStreamCreate_;
void* dispatch::cuEventElapsedTime_;
void* dispatch::cuMemFree_v2_;
void* dispatch::cuMemcpyDtoHAsync_v2_;
void* dispatch::cuDriverGetVersion_;
void* dispatch::cuDeviceGetName_;
void* dispatch::cuDeviceGetPCIBusId_;
void* dispatch::cuModuleGetGlobal_v2_;

void* dispatch::cuMemcpyHtoDAsync_v2_;
void* dispatch::cuModuleLoad_;
void* dispatch::cuLaunchKernel_;
void* dispatch::cuModuleUnload_;
void* dispatch::cuModuleLoadDataEx_;
void* dispatch::cuDeviceGetAttribute_;
void* dispatch::cuDeviceGetCount_;
void* dispatch::cuMemcpyHtoD_v2_;
void* dispatch::cuInit_;
void* dispatch::cuEventRecord_;
void* dispatch::cuCtxCreate_v2_;
void* dispatch::cuModuleGetFunction_;
void* dispatch::cuStreamSynchronize_;
void* dispatch::cuStreamDestroy_v2_;
void* dispatch::cuEventDestroy_v2_;
void* dispatch::cuMemAlloc_v2_;
void* dispatch::cuPointerGetAttribute_;
void* dispatch::cuCtxGetDevice_;
void* dispatch::cuMemsetD8Async_;
void* dispatch::cuCtxPushCurrent_v2_;
void* dispatch::cuCtxPopCurrent_v2_;

void* dispatch::nvrtcCompileProgram_;
void* dispatch::nvrtcGetProgramLogSize_;
void* dispatch::nvrtcGetPTX_;
void* dispatch::nvrtcGetPTXSize_;
void* dispatch::nvrtcCreateProgram_;
void* dispatch::nvrtcGetProgramLog_;

void* dispatch::nvmlInit_v2_;
void* dispatch::nvmlDeviceGetHandleByPciBusId_v2_;
void* dispatch::nvmlDeviceGetClockInfo_;
void* dispatch::nvmlDeviceGetMaxClockInfo_;

void* dispatch::cublasCreate_v2_;
void* dispatch::cublasGetStream_v2_;
void* dispatch::cublasSetStream_v2_;
void* dispatch::cublasHgemm_;
void* dispatch::cublasSgemm_v2_;
void* dispatch::cublasDgemm_v2_;
void* dispatch::cublasGemmEx_;

void* dispatch::cudnnCreateConvolutionDescriptor_;
void* dispatch::cudnnCreatePoolingDescriptor_;
void* dispatch::cudnnCreateTensorDescriptor_;
void* dispatch::cudnnCreateFilterDescriptor_;
void* dispatch::cudnnCreate_;
void* dispatch::cudnnSetTensor4dDescriptor_;
void* dispatch::cudnnSetFilter4dDescriptor_;
void* dispatch::cudnnSetTensorNdDescriptorEx_;
void* dispatch::cudnnSetFilterNdDescriptor_;
void* dispatch::cudnnSetPoolingNdDescriptor_;
void* dispatch::cudnnSetConvolution2dDescriptor_;
void* dispatch::cudnnSetConvolutionNdDescriptor_;
void* dispatch::cudnnGetConvolutionForwardAlgorithm_;
void* dispatch::cudnnGetConvolutionForwardWorkspaceSize_;
void* dispatch::cudnnConvolutionForward_;
void* dispatch::cudnnPoolingForward_;
void* dispatch::cudnnSetStream_;
void* dispatch::cudnnTransformTensor_;

}
}
