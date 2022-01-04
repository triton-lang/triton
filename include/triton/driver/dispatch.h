#pragma once

#ifndef _TRITON_DRIVER_DISPATCH_H_
#define _TRITON_DRIVER_DISPATCH_H_

#include <type_traits>
#include <dlfcn.h>

//CUDA Backend
#include "triton/external/CUDA/cuda.h"
#include "triton/external/CUDA/nvml.h"

//// HIP backend
//#define __HIP_PLATFORM_AMD__
#include "triton/external/hip.h"

//Exceptions
#include <iostream>
#include <stdexcept>

namespace llvm {
class PassRegistry;
class Module;
}

namespace triton
{
namespace driver
{

class cu_context;

template<class T> void check(T){}
void check(CUresult err);
void check(hipError_t err);

class dispatch
{
protected:
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
  static void release();
  // Nvidia
  static bool nvmlinit();
  static bool cuinit();
  // AMD
  static bool hipinit();

  /* ------------------- *
   * CUDA
   * ------------------- */
  // context management
  static CUresult cuInit(unsigned int Flags);
  static CUresult cuCtxDestroy_v2(CUcontext ctx);
  static CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
  static CUresult cuCtxPushCurrent_v2(CUcontext ctx);
  static CUresult cuCtxPopCurrent_v2(CUcontext *pctx);
  static CUresult cuCtxGetDevice(CUdevice* result);
  static CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags);
  static CUresult cuDriverGetVersion(int *driverVersion);
  // device management
  static CUresult cuDeviceGet(CUdevice *device, int ordinal);
  static CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
  static CUresult cuDeviceGetPCIBusId(char *id, int len, CUdevice dev);
  static CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
  static CUresult cuDeviceGetCount(int *count);
  // link management
  static CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues);
  static CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues);
  static CUresult cuLinkCreate_v2(unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut);
  static CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut);
  static CUresult cuLinkDestroy(CUlinkState state);
  // module management
  static CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t* bytes, CUmodule hmod, const char *name);
  static CUresult cuModuleLoad(CUmodule *module, const char *fname);
  static CUresult cuModuleLoadData(CUmodule* module, const void* image);
  static CUresult cuModuleUnload(CUmodule hmod);
  static CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
  static CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
  // stream management
  static CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags);
  static CUresult cuStreamSynchronize(CUstream hStream);
  static CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx);
  static CUresult cuStreamDestroy_v2(CUstream hStream);
  static CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
  // function management
  static CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc);
  static CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
  static CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
  // memory management
  static CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
  static CUresult cuPointerGetAttribute(void * data, CUpointer_attribute attribute, CUdeviceptr ptr);
  static CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char x, size_t N, CUstream stream);
  static CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
  static CUresult cuMemFree_v2(CUdeviceptr dptr);
  static CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
  static CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
  static CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
  // event management
  static CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags);
  static CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
  static CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
  static CUresult cuEventDestroy_v2(CUevent hEvent);


  /* ------------------- *
   * NVML
   * ------------------- */
  static nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2( const char* pciBusId, nvmlDevice_t* device);
  static nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);
  static nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);
  static nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int mem_clock, unsigned int sm_clock);

  /* ------------------- *
   * HIP
   * ------------------- */
  // context management
  static hipError_t hipInit(unsigned int Flags);
  static hipError_t hipCtxDestroy(hipCtx_t ctx);
  static hipError_t hipCtxCreate(hipCtx_t *pctx, unsigned int flags, hipDevice_t dev);
  static hipError_t hipCtxPushCurrent(hipCtx_t ctx);
  static hipError_t hipCtxPopCurrent(hipCtx_t *pctx);
  static hipError_t hipCtxGetDevice(hipDevice_t* result);
  static hipError_t hipCtxEnablePeerAccess(hipCtx_t peerContext, unsigned int flags);
  static hipError_t hipDriverGetVersion(int *driverVersion);
  // device management
  static hipError_t hipGetDevice(hipDevice_t *device, int ordinal);
  static hipError_t hipDeviceGetName(char *name, int len, hipDevice_t dev);
  static hipError_t hipDeviceGetPCIBusId(char *id, int len, hipDevice_t dev);
  static hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attrib, hipDevice_t dev);
  static hipError_t hipGetDeviceCount(int *count);
  // module management
  static hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t* bytes, hipModule_t hmod, const char *name);
  static hipError_t hipModuleLoad(hipModule_t *module, const char *fname);
  static hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
  static hipError_t hipModuleUnload(hipModule_t hmod);
  static hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image, unsigned int numOptions, hipJitOption *options, void **optionValues);
  static hipError_t hipModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod, const char *name);
  // stream management
  static hipError_t hipStreamCreate(hipStream_t *phStream, unsigned int Flags);
  static hipError_t hipStreamSynchronize(hipStream_t hStream);
  static hipError_t hipStreamDestroy(hipStream_t hStream);
  static hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t hStream, void **kernelParams, void **extra);
  // function management
  static hipError_t hipFuncGetAttributes(hipFuncAttributes* attrib, void* hfunc);
  static hipError_t hipFuncSetAttribute(hipFunction_t hfunc, hipFuncAttribute attrib, int value);
  static hipError_t hipFuncSetCacheConfig(hipFunction_t hfunc, hipFuncCache_t config);
  // memory management
  static hipError_t hipMalloc(hipDeviceptr_t *dptr, size_t bytesize);
  static hipError_t hipPointerGetAttribute(void * data, CUpointer_attribute attribute, hipDeviceptr_t ptr);
  static hipError_t hipMemsetD8Async(hipDeviceptr_t dst, unsigned char x, size_t N, hipStream_t stream);
  static hipError_t hipMemcpyDtoH(void *dstHost, hipDeviceptr_t srcDevice, size_t ByteCount);
  static hipError_t hipFree(hipDeviceptr_t dptr);
  static hipError_t hipMemcpyDtoHAsync(void *dstHost, hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hStream);
  static hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dstDevice, const void *srcHost, size_t ByteCount, hipStream_t hStream);
  static hipError_t hipMemcpyHtoD(hipDeviceptr_t dstDevice, const void *srcHost, size_t ByteCount);
  // event management
  static hipError_t hipEventCreate(hipEvent_t *phEvent, unsigned int Flags);
  static hipError_t hipEventElapsedTime(float *pMilliseconds, hipEvent_t hStart, hipEvent_t hEnd);
  static hipError_t hipEventRecord(hipEvent_t hEvent, hipStream_t hStream);
  static hipError_t hipEventDestroy(hipEvent_t hEvent);



private:

  // Libraries
  static void* cuda_;
  static void* nvml_;
  static void* hip_;


  /* ------------------- *
   * CUDA
   * ------------------- */
  // context management
  static void* cuCtxGetCurrent_;
  static void* cuCtxSetCurrent_;
  static void* cuCtxDestroy_v2_;
  static void* cuCtxCreate_v2_;
  static void* cuCtxGetDevice_;
  static void* cuCtxPushCurrent_v2_;
  static void* cuCtxPopCurrent_v2_;
  static void* cuCtxEnablePeerAccess_;
  static void* cuDriverGetVersion_;
  static void* cuInit_;
  // device management
  static void* cuDeviceGet_;
  static void* cuDeviceGetName_;
  static void* cuDeviceGetPCIBusId_;
  static void* cuDeviceGetAttribute_;
  static void* cuDeviceGetCount_;
  // link management
  static void* cuLinkAddFile_v2_;
  static void* cuLinkAddData_v2_;
  static void* cuLinkCreate_v2_;
  static void* cuLinkDestroy_;
  static void* cuLinkComplete_;
  // module management
  static void* cuModuleGetGlobal_v2_;
  static void* cuModuleLoad_;
  static void* cuModuleUnload_;
  static void* cuModuleLoadDataEx_;
  static void* cuModuleLoadData_;
  static void* cuModuleGetFunction_;
  // stream management
  static void* cuStreamCreate_;
  static void* cuStreamSynchronize_;
  static void* cuStreamDestroy_v2_;
  static void* cuStreamGetCtx_;
  static void* cuLaunchKernel_;
  // function management
  static void* cuFuncGetAttribute_;
  static void* cuFuncSetAttribute_;
  static void* cuFuncSetCacheConfig_;
  // memory management
  static void* cuMemcpyDtoH_v2_;
  static void* cuMemFree_v2_;
  static void* cuMemcpyDtoHAsync_v2_;
  static void* cuMemcpyHtoDAsync_v2_;
  static void* cuMemcpyHtoD_v2_;
  static void* cuMemAlloc_v2_;
  static void* cuMemsetD8Async_;
  static void* cuPointerGetAttribute_;
  // event management
  static void* cuEventCreate_;
  static void* cuEventElapsedTime_;
  static void* cuEventRecord_;
  static void* cuEventDestroy_v2_;

  /* ------------------- *
   * NVML
   * ------------------- */
  static void* nvmlInit_v2_;
  static void* nvmlDeviceGetHandleByPciBusId_v2_;
  static void* nvmlDeviceGetClockInfo_;
  static void* nvmlDeviceGetMaxClockInfo_;
  static void* nvmlDeviceSetApplicationsClocks_;

  /* ------------------- *
   * HIP
   * ------------------- */
  // context management
  static void* hipInit_;
  static void* hipCtxDestroy_;
  static void* hipCtxCreate_;
  static void* hipCtxPushCurrent_;
  static void* hipCtxPopCurrent_;
  static void* hipCtxGetDevice_;
  static void* hipCtxEnablePeerAccess_;
  static void* hipDriverGetVersion_;
  // device management
  static void* hipGetDevice_;
  static void* hipDeviceGetName_;
  static void* hipDeviceGetPCIBusId_;
  static void* hipDeviceGetAttribute_;
  static void* hipGetDeviceCount_;
  // module management
  static void* hipModuleGetGlobal_;
  static void* hipModuleLoad_;
  static void* hipModuleLoadData_;
  static void* hipModuleUnload_;
  static void* hipModuleLoadDataEx_;
  static void* hipModuleGetFunction_;
  // stream management
  static void* hipStreamCreate_;
  static void* hipStreamSynchronize_;
  static void* hipStreamDestroy_;
  static void* hipModuleLaunchKernel_;;
  // function management
  static void* hipFuncGetAttributes_;
  static void* hipFuncSetAttribute_;
  static void* hipFuncSetCacheConfig_;
  // memory management
  static void* hipMalloc_;
  static void* hipPointerGetAttribute_;
  static void* hipMemsetD8Async_;
  static void* hipMemcpyDtoH_;
  static void* hipFree_;
  static void* hipMemcpyDtoHAsync_;
  static void* hipMemcpyHtoDAsync_;
  static void* hipMemcpyHtoD_;
  // event management
  static void* hipEventCreate_;
  static void* hipEventElapsedTime_;
  static void* hipEventRecord_;
  static void* hipEventDestroy_;
};

}
}


#endif
