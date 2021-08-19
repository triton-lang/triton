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

#ifdef __HIP_PLATFORM_AMD__
#include "triton/driver/dispatch_hip.h"
#include "triton/driver/context_hip.h"
#else
#include "triton/driver/dispatch.h"
#include "triton/driver/context.h"
#endif
#include "triton/tools/sys/getenv.hpp"

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


bool dispatch::cuinit(){
  if(cuda_==nullptr){
    putenv((char*)"CUDA_CACHE_DISABLE=1");
    std::string libcuda = tools::getenv("TRITON_LIBCUDA");
    if(libcuda.empty()){
#ifdef __HIP_PLATFORM_AMD__
     cuda_ = dlopen("libamdhip64.so", RTLD_LAZY);
#else
      cuda_ = dlopen("libcuda.so", RTLD_LAZY);
      if(!cuda_)
        cuda_ = dlopen("libcuda.so.1", RTLD_LAZY);
      if(!cuda_)
        throw std::runtime_error("Could not find `libcuda.so`. Make sure it is in your LD_LIBRARY_PATH.");
#endif
    }
    else
      cuda_ = dlopen(libcuda.c_str(), RTLD_LAZY);
  }
  if(cuda_ == nullptr)
    return false;
  CUresult (*fptr)(unsigned int);
#ifdef __HIP_PLATFORM_AMD__
  hipInit_ = dlsym(cuda_, "hipInit");
  *reinterpret_cast<void **>(&fptr) = hipInit_;
#else
  cuInit_ = dlsym(cuda_, "cuInit");
  *reinterpret_cast<void **>(&fptr) = cuInit_;
#endif
  CUresult res = (*fptr)(0);
  check(res);
  return true;
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

bool dispatch::spvllvminit(){
  if(spvllvm_==nullptr)
    spvllvm_ = dlopen("libLLVMSPIRVLib.so", RTLD_LAZY);
  return spvllvm_ != nullptr;
}

#ifdef __HIP_PLATFORM_AMD__
//HIP
CUDA_DEFINE1(hipError_t, hipCtxDestroy, hipCtx_t)
CUDA_DEFINE2(hipError_t, hipEventCreate, hipEvent_t *, unsigned int)
CUDA_DEFINE2(hipError_t, hipGetDevice, hipDevice_t *, int)
CUDA_DEFINE3(hipError_t, hipMemcpyDtoH, void *, hipDeviceptr_t, size_t)
CUDA_DEFINE2(hipError_t, hipStreamCreate__, hipStream_t *, unsigned int)
CUDA_DEFINE3(hipError_t, hipEventElapsedTime, float *, hipEvent_t, hipEvent_t)
CUDA_DEFINE1(hipError_t, hipFree, hipDeviceptr_t)
CUDA_DEFINE4(hipError_t, hipMemcpyDtoHAsync, void *, hipDeviceptr_t, size_t, hipStream_t)
CUDA_DEFINE1(hipError_t, hipDriverGetVersion, int *)
CUDA_DEFINE3(hipError_t, hipDeviceGetName, char *, int, int)
CUDA_DEFINE3(hipError_t, hipDeviceGetPCIBusId, char *, int, int)
CUDA_DEFINE4(hipError_t, hipModuleGetGlobal, hipDeviceptr_t*, size_t*, hipModule_t, const char*)

CUDA_DEFINE4(hipError_t, hipMemcpyHtoDAsync, hipDeviceptr_t, const void *, size_t, hipStream_t)
CUDA_DEFINE2(hipError_t, hipModuleLoad, hipModule_t *, const char *)
CUDA_DEFINE11(hipError_t, hipModuleLaunchKernel, hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, hipStream_t, void **, void **)
CUDA_DEFINE1(hipError_t, hipModuleUnload, hipModule_t)
CUDA_DEFINE5(hipError_t, hipModuleLoadDataEx, hipModule_t *, const void *, unsigned int, hipJitOption *, void **)
CUDA_DEFINE3(hipError_t, hipDeviceGetAttribute, int *, hipDeviceAttribute_t, int)
CUDA_DEFINE1(hipError_t, hipGetDeviceCount, int *)
CUDA_DEFINE3(hipError_t, hipMemcpyHtoD, hipDeviceptr_t, const void *, size_t )
CUDA_DEFINE1(hipError_t, hipInit, unsigned int)
CUDA_DEFINE2(hipError_t, hipEventRecord, hipEvent_t, hipStream_t)
CUDA_DEFINE3(hipError_t, hipCtxCreate, hipCtx_t *, unsigned int, hipDevice_t)
CUDA_DEFINE3(hipError_t, hipModuleGetFunction, hipFunction_t *, hipModule_t, const char *)
CUDA_DEFINE1(hipError_t, hipStreamSynchronize, hipStream_t)
CUDA_DEFINE1(hipError_t, hipStreamDestroy, hipStream_t)
CUDA_DEFINE2(hipError_t, cuStreamGetCtx, hipStream_t, hipCtx_t*)
CUDA_DEFINE1(hipError_t, hipEventDestroy, hipEvent_t)
CUDA_DEFINE2(hipError_t, hipMalloc, hipDeviceptr_t*, size_t)
CUDA_DEFINE3(hipError_t, hipPointerGetAttribute, void*, hipPointerAttribute_t, hipDeviceptr_t)
CUDA_DEFINE1(hipError_t, hipCtxGetDevice, hipDevice_t*)
CUDA_DEFINE1(hipError_t, hipCtxGetCurrent, hipCtx_t*)
CUDA_DEFINE1(hipError_t, hipCtxSetCurrent, hipCtx_t)
CUDA_DEFINE4(hipError_t, hipMemsetD8Async, hipDeviceptr_t, unsigned char, size_t, hipStream_t)
CUDA_DEFINE1(hipError_t, hipCtxPushCurrent, hipCtx_t)
CUDA_DEFINE1(hipError_t, hipCtxPopCurrent, hipCtx_t*)
// CUDA_DEFINE3(hipError_t, hipFuncGetAttribute, int*, hipFuncAttribute_t, hipFunction_t)
// CUDA_DEFINE3(hipError_t, hipFuncSetAttribute, hipFunction_t, hipFuncAttribute_t, int)
// CUDA_DEFINE2(hipError_t, hipFuncSetCacheConfig, hipFunction_t, hipFuncCache)
#else
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
CUDA_DEFINE8(CUresult, cuLinkAddData_v2, CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**);
CUDA_DEFINE4(CUresult, cuLinkCreate_v2, unsigned int, CUjit_option*, void**, CUlinkState*);
CUDA_DEFINE1(CUresult, cuLinkDestroy, CUlinkState);

CUDA_DEFINE3(CUresult, cuLinkComplete, CUlinkState, void**, size_t*);
CUDA_DEFINE4(CUresult, cuMemcpyHtoDAsync_v2, CUdeviceptr, const void *, size_t, CUstream)
CUDA_DEFINE2(CUresult, cuModuleLoad, CUmodule *, const char *)
CUDA_DEFINE11(CUresult, cuLaunchKernel, CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **)
CUDA_DEFINE1(CUresult, cuModuleUnload, CUmodule)
CUDA_DEFINE2(CUresult, cuModuleLoadData, CUmodule *, const void *)
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
CUDA_DEFINE2(CUresult, cuStreamGetCtx, CUstream, CUcontext*)
CUDA_DEFINE1(CUresult, cuEventDestroy_v2, CUevent)
CUDA_DEFINE2(CUresult, cuMemAlloc_v2, CUdeviceptr*, size_t)
CUDA_DEFINE3(CUresult, cuPointerGetAttribute, void*, CUpointer_attribute, CUdeviceptr)
CUDA_DEFINE1(CUresult, cuCtxGetDevice, CUdevice*)
CUDA_DEFINE1(CUresult, cuCtxGetCurrent, CUcontext*)
CUDA_DEFINE1(CUresult, cuCtxSetCurrent, CUcontext)
CUDA_DEFINE4(CUresult, cuMemsetD8Async, CUdeviceptr, unsigned char, size_t, CUstream)
CUDA_DEFINE1(CUresult, cuCtxPushCurrent_v2, CUcontext)
CUDA_DEFINE1(CUresult, cuCtxPopCurrent_v2, CUcontext*)
CUDA_DEFINE3(CUresult, cuFuncGetAttribute, int*, CUfunction_attribute, CUfunction)
CUDA_DEFINE3(CUresult, cuFuncSetAttribute, CUfunction, CUfunction_attribute, int)
CUDA_DEFINE2(CUresult, cuFuncSetCacheConfig, CUfunction, CUfunc_cache)
#endif

NVML_DEFINE2(nvmlReturn_t, nvmlDeviceGetHandleByPciBusId_v2, const char *, nvmlDevice_t*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceGetClockInfo, nvmlDevice_t, nvmlClockType_t, unsigned int*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceGetMaxClockInfo, nvmlDevice_t, nvmlClockType_t, unsigned int*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceSetApplicationsClocks, nvmlDevice_t, unsigned int, unsigned int)

// LLVM to SPIR-V
int dispatch::initializeLLVMToSPIRVPass(llvm::PassRegistry &registry){
  return f_impl<dispatch::spvllvminit>(spvllvm_, initializeLLVMToSPIRVPass, initializeLLVMToSPIRVPass_, "initializeLLVMToSPIRVPass", std::ref(registry));
}

bool dispatch::writeSpirv(llvm::Module *M, std::ostream &OS, std::string &ErrMsg){
  return f_impl<dispatch::spvllvminit>(spvllvm_, writeSpirv, writeSpirv_, "writeSpirv", M, std::ref(OS), std::ref(ErrMsg));
}

// Release
void dispatch::release(){
  if(cuda_){
    dlclose(cuda_);
    cuda_ = nullptr;
  }
}

void* dispatch::cuda_;
void* dispatch::nvml_;
void* dispatch::spvllvm_;


#ifdef __HIP_PLATFORM_AMD__
void* dispatch::hipCtxGetCurrent_;
void* dispatch::hipCtxSetCurrent_;
void* dispatch::hipCtxDestroy_;
void* dispatch::hipEventCreate_;
void* dispatch::hipGetDevice_;
void* dispatch::hipMemcpyDtoH_;
void* dispatch::hipStreamCreate___;
void* dispatch::hipEventElapsedTime_;
void* dispatch::hipFree_;
void* dispatch::hipMemcpyDtoHAsync_;
void* dispatch::hipDriverGetVersion_;
void* dispatch::hipDeviceGetName_;
void* dispatch::hipDeviceGetPCIBusId_;
void* dispatch::hipModuleGetGlobal_;

void* dispatch::hipMemcpyHtoDAsync_;
void* dispatch::hipModuleLoad_;
void* dispatch::hipModuleLaunchKernel_;
void* dispatch::hipModuleUnload_;
void* dispatch::hipModuleLoadDataEx_;
void* dispatch::hipDeviceGetAttribute_;
void* dispatch::hipGetDeviceCount_;
void* dispatch::hipMemcpyHtoD_;
void* dispatch::hipInit_;
void* dispatch::hipEventRecord_;
void* dispatch::hipCtxCreate_;
void* dispatch::hipModuleGetFunction_;
void* dispatch::hipStreamSynchronize_;
void* dispatch::hipStreamDestroy_;
void* dispatch::cuStreamGetCtx_;
void* dispatch::hipEventDestroy_;
void* dispatch::hipMalloc_;
void* dispatch::hipPointerGetAttribute_;
void* dispatch::hipCtxGetDevice_;
void* dispatch::hipMemsetD8Async_;
void* dispatch::hipCtxPushCurrent_;
void* dispatch::hipCtxPopCurrent_;
void* dispatch::hipFuncGetAttribute_;
void* dispatch::cuFuncSetAttribute_;
void* dispatch::hipFuncSetCacheConfig_;
#else
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

void* dispatch::cuLinkAddData_v2_;
void* dispatch::cuLinkCreate_v2_;
void* dispatch::cuLinkDestroy_;
void* dispatch::cuModuleLoadData_;
void* dispatch::cuLinkComplete_;

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
void* dispatch::cuStreamGetCtx_;
void* dispatch::cuEventDestroy_v2_;
void* dispatch::cuMemAlloc_v2_;
void* dispatch::cuPointerGetAttribute_;
void* dispatch::cuCtxGetDevice_;
void* dispatch::cuMemsetD8Async_;
void* dispatch::cuCtxPushCurrent_v2_;
void* dispatch::cuCtxPopCurrent_v2_;
void* dispatch::cuFuncGetAttribute_;
void* dispatch::cuFuncSetAttribute_;
void* dispatch::cuFuncSetCacheConfig_;
#endif

void* dispatch::nvmlInit_v2_;
void* dispatch::nvmlDeviceGetHandleByPciBusId_v2_;
void* dispatch::nvmlDeviceGetClockInfo_;
void* dispatch::nvmlDeviceGetMaxClockInfo_;
void* dispatch::nvmlDeviceSetApplicationsClocks_;

// SPIR-V
void* dispatch::initializeLLVMToSPIRVPass_;
void* dispatch::writeSpirv_;

}
}
