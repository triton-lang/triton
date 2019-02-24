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

#ifndef TDL_INCLUDE_DRIVER_DISPATCHER_H
#define TDL_INCLUDE_DRIVER_DISPATCHER_H

#include <type_traits>
#include <dlfcn.h>

//CUDA Backend
#include "triton/external/CUDA/cuda.h"
#include "triton/external/CUDA/nvrtc.h"
#include "triton/external/CUDA/cublas_v2.h"
#include "triton/external/CUDA/cudnn.h"
#include "triton/external/CUDA/nvml.h"

//Exceptions
#include <iostream>
#include <stdexcept>

namespace tdl
{
namespace driver
{

class Context;

template<class T> void check(T){}
void check(nvrtcResult err);
void check(CUresult err);
void check(cublasStatus_t err);
void check(cudnnStatus_t err);

class dispatch
{
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
    check(res);
    return res;
  }

public:
  static bool nvrtcinit();
  static bool nvmlinit();
  static bool cuinit();
  static bool cublasinit();
  static bool cudnninit();

  static void release();

  //CUDA
  static CUresult cuCtxGetCurrent(CUcontext *pctx);
  static CUresult cuCtxSetCurrent(CUcontext ctx);

  static CUresult cuCtxDestroy_v2(CUcontext ctx);
  static CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags);
  static CUresult cuDeviceGet(CUdevice *device, int ordinal);
  static CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
  static CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags);
  static CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
  static CUresult cuMemFree_v2(CUdeviceptr dptr);
  static CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
  static CUresult cuDriverGetVersion(int *driverVersion);
  static CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
  static CUresult cuDeviceGetPCIBusId(char *id, int len, CUdevice dev);
  static CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t* bytes, CUmodule hmod, const char *name);

  static CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
  static CUresult cuModuleLoad(CUmodule *module, const char *fname);
  static CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
  static CUresult cuModuleUnload(CUmodule hmod);
  static CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
  static CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
  static CUresult cuDeviceGetCount(int *count);
  static CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
  static CUresult cuInit(unsigned int Flags);
  static CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
  static CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
  static CUresult cuCtxPushCurrent_v2(CUcontext ctx);
  static CUresult cuCtxPopCurrent_v2(CUcontext *pctx);
  static CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
  static CUresult cuStreamSynchronize(CUstream hStream);
  static CUresult cuStreamDestroy_v2(CUstream hStream);
  static CUresult cuEventDestroy_v2(CUevent hEvent);
  static CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
  static CUresult cuPointerGetAttribute(void * data, CUpointer_attribute attribute, CUdeviceptr ptr);
  static CUresult cuCtxGetDevice(CUdevice* result);
  static CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char x, size_t N, CUstream stream);

  static nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2( const char* pciBusId, nvmlDevice_t* device);
  static nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);
  static nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);

  static nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char **options);
  static nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet);
  static nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx);
  static nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet);
  static nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char *src, const char *name, int numHeaders, const char **headers, const char **includeNames);
  static nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log);

  static cublasHandle_t cublasHandle(Context const & ctx);
  static cublasStatus_t cublasCreate_v2(cublasHandle_t* h);
  static cublasStatus_t cublasGetStream_v2(cublasHandle_t h, cudaStream_t *streamId);
  static cublasStatus_t cublasSetStream_v2(cublasHandle_t h, cudaStream_t streamId);
  static cublasStatus_t cublasSgemm_v2 (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float* alpha, const float *A, int lda, const float *B, int ldb, float* beta, float *C, int ldc);
  static cublasStatus_t cublasDgemm_v2 (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double* alpha, const double *A, int lda, const double *B, int ldb, double* beta, double *C, int ldc);
  static cublasStatus_t cublasHgemm (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, half* alpha, const half *A, int lda, const half *B, int ldb, half* beta, half *C, int ldc);
  static cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const void *beta, void *C, cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo);

  static cudnnHandle_t cudnnHandle(Context const & ctx);
  static cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc);
  static cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc);
  static cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc);
  static cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc);
  static cudnnStatus_t cudnnCreate(cudnnHandle_t *handle);
  static cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w);
  static cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int k, int c, int h, int w);
  static cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int nbDims, const int dimA[]);
  static cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims, const int filterDimA[]);
  static cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v, int upscalex, int upscaley, cudnnConvolutionMode_t mode);
  static cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc, int arrayLength, const int padA[], const int filterStrideA[], const int upscaleA[], cudnnConvolutionMode_t mode, cudnnDataType_t dataType);
  static cudnnStatus_t cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode, const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims, const int windowDimA[], const int paddingA[], const int strideA[]);
  static cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionFwdAlgo_t *algo);
  static cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo, size_t *sizeInBytes);
  static cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);
  static cudnnStatus_t cudnnPoolingForward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);
  static cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
  static cudnnStatus_t cudnnTransformTensor(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);

private:
  static void* cuda_;
  static void* nvrtc_;
  static void* nvml_;
  static void* cublas_;
  static void* cudnn_;

  //CUDA
  static void* cuCtxGetCurrent_;
  static void* cuCtxSetCurrent_;
  static void* cuCtxDestroy_v2_;
  static void* cuEventCreate_;
  static void* cuDeviceGet_;
  static void* cuMemcpyDtoH_v2_;
  static void* cuStreamCreate_;
  static void* cuEventElapsedTime_;
  static void* cuMemFree_v2_;
  static void* cuMemcpyDtoHAsync_v2_;
  static void* cuDriverGetVersion_;
  static void* cuDeviceGetName_;
  static void* cuDeviceGetPCIBusId_;
  static void* cuModuleGetGlobal_v2_;

  static void* cuMemcpyHtoDAsync_v2_;
  static void* cuModuleLoad_;
  static void* cuLaunchKernel_;
  static void* cuModuleUnload_;
  static void* cuModuleLoadDataEx_;
  static void* cuDeviceGetAttribute_;
  static void* cuDeviceGetCount_;
  static void* cuMemcpyHtoD_v2_;
  static void* cuInit_;
  static void* cuEventRecord_;
  static void* cuCtxCreate_v2_;
  static void* cuModuleGetFunction_;
  static void* cuStreamSynchronize_;
  static void* cuStreamDestroy_v2_;
  static void* cuEventDestroy_v2_;
  static void* cuMemAlloc_v2_;
  static void* cuPointerGetAttribute_;
  static void* cuCtxGetDevice_;
  static void* cuMemsetD8Async_;
  static void* cuCtxPushCurrent_v2_;
  static void* cuCtxPopCurrent_v2_;

  static void* nvmlInit_v2_;
  static void* nvmlDeviceGetHandleByPciBusId_v2_;
  static void* nvmlDeviceGetClockInfo_;
  static void* nvmlDeviceGetMaxClockInfo_;

  static void* nvrtcCompileProgram_;
  static void* nvrtcGetProgramLogSize_;
  static void* nvrtcGetPTX_;
  static void* nvrtcGetPTXSize_;
  static void* nvrtcCreateProgram_;
  static void* nvrtcGetProgramLog_;

  static void* cublasCreate_v2_;
  static void* cublasGetStream_v2_;
  static void* cublasSetStream_v2_;
  static void* cublasHgemm_;
  static void* cublasSgemm_v2_;
  static void* cublasDgemm_v2_;
  static void* cublasGemmEx_;

  static void* cudnnCreateConvolutionDescriptor_;
  static void* cudnnCreatePoolingDescriptor_;
  static void* cudnnCreateTensorDescriptor_;
  static void* cudnnCreateFilterDescriptor_;
  static void* cudnnCreate_;
  static void* cudnnSetTensor4dDescriptor_;
  static void* cudnnSetFilter4dDescriptor_;
  static void* cudnnSetTensorNdDescriptorEx_;
  static void* cudnnSetFilterNdDescriptor_;
  static void* cudnnSetConvolution2dDescriptor_;
  static void* cudnnSetConvolutionNdDescriptor_;
  static void* cudnnSetPoolingNdDescriptor_;
  static void* cudnnGetConvolutionForwardAlgorithm_;
  static void* cudnnGetConvolutionForwardWorkspaceSize_;
  static void* cudnnConvolutionForward_;
  static void* cudnnPoolingForward_;
  static void* cudnnSetStream_;
  static void* cudnnTransformTensor_;
};

}
}


#endif
