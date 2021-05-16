#pragma once

#ifndef _TRITON_DRIVER_DISPATCH_H_
#define _TRITON_DRIVER_DISPATCH_H_

#include <type_traits>
#include <dlfcn.h>

//CUDA Backend
#include "triton/external/CUDA/cuda.h"
#include "triton/external/CUDA/nvml.h"

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
  static bool nvmlinit();
  static bool cuinit();
  static bool spvllvminit();
  static void release();

  // CUDA
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
  static CUresult cuModuleLoadData(CUmodule* module, const void* image);
  static CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
  static CUresult cuModuleUnload(CUmodule hmod);
  static CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);

  static CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues);
  static CUresult cuLinkCreate_v2(unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut);
  static CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut);
  static CUresult cuLinkDestroy(CUlinkState state);

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
  static CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx);
  static CUresult cuStreamDestroy_v2(CUstream hStream);
  static CUresult cuEventDestroy_v2(CUevent hEvent);
  static CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
  static CUresult cuPointerGetAttribute(void * data, CUpointer_attribute attribute, CUdeviceptr ptr);
  static CUresult cuCtxGetDevice(CUdevice* result);
  static CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char x, size_t N, CUstream stream);
  static CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc);
  static CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int  value);
  static CUresult cuFuncSetCacheConfig (CUfunction hfunc, CUfunc_cache config);
  // NVML
  static nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2( const char* pciBusId, nvmlDevice_t* device);
  static nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);
  static nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);
  static nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int mem_clock, unsigned int sm_clock);


  // SPIR-V libraries
  static int initializeLLVMToSPIRVPass(llvm::PassRegistry &);
  static bool writeSpirv(llvm::Module *M, std::ostream &OS, std::string &ErrMsg);


private:

  // Libraries
  static void* cuda_;
  static void* nvml_;
  static void* vulkan_;
  static void* spvllvm_;
  static void* spvcross_;
  static void* opengl_;


  // CUDA functions
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
  static void* cuLinkAddData_v2_;
  static void* cuLinkCreate_v2_;
  static void* cuLinkDestroy_;
  static void* cuModuleLoadData_;
  static void* cuLinkComplete_;
  static void* cuDeviceGetAttribute_;
  static void* cuDeviceGetCount_;
  static void* cuMemcpyHtoD_v2_;
  static void* cuInit_;
  static void* cuEventRecord_;
  static void* cuCtxCreate_v2_;
  static void* cuModuleGetFunction_;
  static void* cuStreamSynchronize_;
  static void* cuStreamDestroy_v2_;
  static void* cuStreamGetCtx_;
  static void* cuEventDestroy_v2_;
  static void* cuMemAlloc_v2_;
  static void* cuPointerGetAttribute_;
  static void* cuCtxGetDevice_;
  static void* cuMemsetD8Async_;
  static void* cuCtxPushCurrent_v2_;
  static void* cuCtxPopCurrent_v2_;
  static void* cuFuncGetAttribute_;
  static void* cuFuncSetAttribute_;
  static void* cuFuncSetCacheConfig_;
  // NVML
  static void* nvmlInit_v2_;
  static void* nvmlDeviceGetHandleByPciBusId_v2_;
  static void* nvmlDeviceGetClockInfo_;
  static void* nvmlDeviceGetMaxClockInfo_;
  static void* nvmlDeviceSetApplicationsClocks_;

  // LLVM to SPIR-V
  static void* initializeLLVMToSPIRVPass_;
  static void* writeSpirv_;
};

}
}


#endif
