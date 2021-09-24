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

#include "triton/driver/dispatch.h"

namespace triton
{
namespace driver
{

//Helpers for function definition
#define DEFINE0(init, hlib, ret, fname) ret dispatch::fname()\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname); }\
void* dispatch::fname ## _;

#define DEFINE1(init, hlib, ret, fname, t1) ret dispatch::fname(t1 a)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a); }\
void* dispatch::fname ## _;

#define DEFINE2(init, hlib, ret, fname, t1, t2) ret dispatch::fname(t1 a, t2 b)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b); }\
void* dispatch::fname ## _;

#define DEFINE3(init, hlib, ret, fname, t1, t2, t3) ret dispatch::fname(t1 a, t2 b, t3 c)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c); }\
void* dispatch::fname ## _;

#define DEFINE4(init, hlib, ret, fname, t1, t2, t3, t4) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d); }\
void* dispatch::fname ## _;

#define DEFINE5(init, hlib, ret, fname, t1, t2, t3, t4, t5) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e); }\
void* dispatch::fname ## _;

#define DEFINE6(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f); }\
void* dispatch::fname ## _;

#define DEFINE7(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g); }\
void* dispatch::fname ## _;

#define DEFINE8(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h); }\
void* dispatch::fname ## _;

#define DEFINE9(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h, t9 i)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h, i); }\
void* dispatch::fname ## _;

#define DEFINE10(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h, t9 i, t10 j)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h, i, j); }\
void* dispatch::fname ## _;

#define DEFINE11(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h, t9 i, t10 j, t11 k)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h, i, j, k); }\
void* dispatch::fname ## _;

#define DEFINE13(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h, t9 i, t10 j, t11 k, t12 l, t13 m)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h, i, j, k, l, m); }\
void* dispatch::fname ## _;

#define DEFINE19(init, hlib, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19) ret dispatch::fname(t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g, t8 h, t9 i, t10 j, t11 k, t12 l, t13 m, t14 n, t15 o, t16 p, t17 q, t18 r, t19 s)\
{return f_impl<dispatch::init>(hlib, fname, fname ## _, #fname, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s); }\
void* dispatch::fname ## _;


/* ------------------- *
 * CUDA
 * ------------------- */

bool dispatch::cuinit(){
  if(cuda_==nullptr){
    cuda_ = dlopen("libcuda.so", RTLD_LAZY);
    if(!cuda_)
      cuda_ = dlopen("libcuda.so.1", RTLD_LAZY);
    if(!cuda_)
      throw std::runtime_error("Could not find `libcuda.so`. Make sure it is in your LD_LIBRARY_PATH.");
  }
  if(cuda_ == nullptr)
    return false;
  CUresult (*fptr)(unsigned int);
  cuInit_ = dlsym(cuda_, "cuInit");
  *reinterpret_cast<void **>(&fptr) = cuInit_;
  CUresult res = (*fptr)(0);
  check(res);
  return true;
}

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

// context management
CUDA_DEFINE1(CUresult, cuCtxDestroy_v2, CUcontext)
CUDA_DEFINE3(CUresult, cuCtxCreate_v2, CUcontext *, unsigned int, CUdevice)
CUDA_DEFINE1(CUresult, cuCtxGetDevice, CUdevice*)
CUDA_DEFINE2(CUresult, cuCtxEnablePeerAccess, CUcontext, unsigned int)
CUDA_DEFINE1(CUresult, cuInit, unsigned int)
CUDA_DEFINE1(CUresult, cuDriverGetVersion, int *)
// device management
CUDA_DEFINE2(CUresult, cuDeviceGet, CUdevice *, int)
CUDA_DEFINE3(CUresult, cuDeviceGetName, char *, int, CUdevice)
CUDA_DEFINE3(CUresult, cuDeviceGetPCIBusId, char *, int, CUdevice)
CUDA_DEFINE3(CUresult, cuDeviceGetAttribute, int *, CUdevice_attribute, CUdevice)
CUDA_DEFINE1(CUresult, cuDeviceGetCount, int*)

// link management
CUDA_DEFINE8(CUresult, cuLinkAddData_v2, CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**);
CUDA_DEFINE4(CUresult, cuLinkCreate_v2, unsigned int, CUjit_option*, void**, CUlinkState*);
CUDA_DEFINE1(CUresult, cuLinkDestroy, CUlinkState);
CUDA_DEFINE3(CUresult, cuLinkComplete, CUlinkState, void**, size_t*);
// module management
CUDA_DEFINE4(CUresult, cuModuleGetGlobal_v2, CUdeviceptr*, size_t*, CUmodule, const char*)
CUDA_DEFINE2(CUresult, cuModuleLoad, CUmodule *, const char *)
CUDA_DEFINE1(CUresult, cuModuleUnload, CUmodule)
CUDA_DEFINE2(CUresult, cuModuleLoadData, CUmodule *, const void *)
CUDA_DEFINE5(CUresult, cuModuleLoadDataEx, CUmodule *, const void *, unsigned int, CUjit_option *, void **)
CUDA_DEFINE3(CUresult, cuModuleGetFunction, CUfunction *, CUmodule, const char *)
// stream management
CUDA_DEFINE2(CUresult, cuStreamCreate, CUstream *, unsigned int)
CUDA_DEFINE1(CUresult, cuStreamSynchronize, CUstream)
CUDA_DEFINE1(CUresult, cuStreamDestroy_v2, CUstream)
CUDA_DEFINE2(CUresult, cuStreamGetCtx, CUstream, CUcontext*)
CUDA_DEFINE11(CUresult, cuLaunchKernel, CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **)
// function management
CUDA_DEFINE3(CUresult, cuFuncGetAttribute, int*, CUfunction_attribute, CUfunction)
CUDA_DEFINE3(CUresult, cuFuncSetAttribute, CUfunction, CUfunction_attribute, int)
CUDA_DEFINE2(CUresult, cuFuncSetCacheConfig, CUfunction, CUfunc_cache)
// memory management
CUDA_DEFINE3(CUresult, cuMemcpyDtoH_v2, void *, CUdeviceptr, size_t)
CUDA_DEFINE1(CUresult, cuMemFree_v2, CUdeviceptr)
CUDA_DEFINE4(CUresult, cuMemcpyDtoHAsync_v2, void *, CUdeviceptr, size_t, CUstream)
CUDA_DEFINE4(CUresult, cuMemcpyHtoDAsync_v2, CUdeviceptr, const void *, size_t, CUstream)
CUDA_DEFINE3(CUresult, cuMemcpyHtoD_v2, CUdeviceptr, const void *, size_t )
CUDA_DEFINE2(CUresult, cuMemAlloc_v2, CUdeviceptr*, size_t)
CUDA_DEFINE3(CUresult, cuPointerGetAttribute, void*, CUpointer_attribute, CUdeviceptr)
CUDA_DEFINE4(CUresult, cuMemsetD8Async, CUdeviceptr, unsigned char, size_t, CUstream)
// event management
CUDA_DEFINE2(CUresult, cuEventCreate, CUevent *, unsigned int)
CUDA_DEFINE3(CUresult, cuEventElapsedTime, float *, CUevent, CUevent)
CUDA_DEFINE2(CUresult, cuEventRecord, CUevent, CUstream)
CUDA_DEFINE1(CUresult, cuEventDestroy_v2, CUevent)



/* ------------------- *
 * NVML
 * ------------------- */
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

#define NVML_DEFINE0(ret, fname) DEFINE0(nvmlinit, nvml_, ret, fname)
#define NVML_DEFINE1(ret, fname, t1) DEFINE1(nvmlinit, nvml_, ret, fname, t1)
#define NVML_DEFINE2(ret, fname, t1, t2) DEFINE2(nvmlinit, nvml_, ret, fname, t1, t2)
#define NVML_DEFINE3(ret, fname, t1, t2, t3) DEFINE3(nvmlinit, nvml_, ret, fname, t1, t2, t3)

NVML_DEFINE2(nvmlReturn_t, nvmlDeviceGetHandleByPciBusId_v2, const char *, nvmlDevice_t*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceGetClockInfo, nvmlDevice_t, nvmlClockType_t, unsigned int*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceGetMaxClockInfo, nvmlDevice_t, nvmlClockType_t, unsigned int*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceSetApplicationsClocks, nvmlDevice_t, unsigned int, unsigned int)

/* ------------------- *
 * HIP
 * ------------------- */
bool dispatch::hipinit(){
  if(hip_==nullptr)
    hip_ = dlopen("libamdhip64.so", RTLD_LAZY);
  if(hip_ == nullptr)
    return false;
  hipError_t (*fptr)();
  hipInit_ = dlsym(hip_, "hipInit");
  *reinterpret_cast<void **>(&fptr) = hipInit_;
  hipError_t res = (*fptr)();
  check(res);
  return res;
}

#define HIP_DEFINE1(ret, fname, t1) DEFINE1(hipinit, hip_, ret, fname, t1)
#define HIP_DEFINE2(ret, fname, t1, t2) DEFINE2(hipinit, hip_, ret, fname, t1, t2)
#define HIP_DEFINE3(ret, fname, t1, t2, t3) DEFINE3(hipinit, hip_, ret, fname, t1, t2, t3)
#define HIP_DEFINE4(ret, fname, t1, t2, t3, t4) DEFINE4(hipinit, hip_, ret, fname, t1, t2, t3, t4)
#define HIP_DEFINE5(ret, fname, t1, t2, t3, t4, t5) DEFINE5(hipinit, hip_, ret, fname, t1, t2, t3, t4, t5)
#define HIP_DEFINE6(ret, fname, t1, t2, t3, t4, t5, t6) DEFINE6(hipinit, hip_, ret, fname, t1, t2, t3, t4, t5, t6)
#define HIP_DEFINE7(ret, fname, t1, t2, t3, t4, t5, t6, t7) DEFINE7(hipinit, hip_, ret, fname, t1, t2, t3, t4, t5, t6, t7)
#define HIP_DEFINE8(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8) DEFINE8(hipinit, hip_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8)
#define HIP_DEFINE9(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9) DEFINE9(hipinit, hip_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9)
#define HIP_DEFINE10(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10) DEFINE10(hipinit, hip_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
#define HIP_DEFINE11(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11) DEFINE11(hipinit, hip_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11)

// context management
HIP_DEFINE1(hipError_t, hipCtxDestroy, hipCtx_t)
HIP_DEFINE3(hipError_t, hipCtxCreate, hipCtx_t *, unsigned int, hipDevice_t)
HIP_DEFINE1(hipError_t, hipCtxGetDevice, hipDevice_t*)
HIP_DEFINE1(hipError_t, hipCtxPushCurrent, hipCtx_t)
HIP_DEFINE1(hipError_t, hipCtxPopCurrent, hipCtx_t*)
HIP_DEFINE2(hipError_t, hipCtxEnablePeerAccess, hipCtx_t, unsigned int)
HIP_DEFINE1(hipError_t, hipInit, unsigned int)
HIP_DEFINE1(hipError_t, hipDriverGetVersion, int *)
// device management
HIP_DEFINE2(hipError_t, hipGetDevice, hipDevice_t *, int)
HIP_DEFINE3(hipError_t, hipDeviceGetName, char *, int, hipDevice_t)
HIP_DEFINE3(hipError_t, hipDeviceGetPCIBusId, char *, int, hipDevice_t)
HIP_DEFINE3(hipError_t, hipDeviceGetAttribute, int *, hipDeviceAttribute_t, hipDevice_t)
HIP_DEFINE1(hipError_t, hipGetDeviceCount, int *)
// module management
HIP_DEFINE4(hipError_t, hipModuleGetGlobal, hipDeviceptr_t*, size_t*, hipModule_t, const char*)
HIP_DEFINE2(hipError_t, hipModuleLoad, hipModule_t *, const char *)
HIP_DEFINE1(hipError_t, hipModuleUnload, hipModule_t)
HIP_DEFINE2(hipError_t, hipModuleLoadData, hipModule_t *, const void *)
HIP_DEFINE5(hipError_t, hipModuleLoadDataEx, hipModule_t *, const void *, unsigned int, hipJitOption *, void **)
HIP_DEFINE3(hipError_t, hipModuleGetFunction, hipFunction_t *, hipModule_t, const char *)
// stream management
HIP_DEFINE2(hipError_t, hipStreamCreate, hipStream_t *, unsigned int)
HIP_DEFINE1(hipError_t, hipStreamSynchronize, hipStream_t)
HIP_DEFINE1(hipError_t, hipStreamDestroy, hipStream_t)
HIP_DEFINE11(hipError_t, hipModuleLaunchKernel, hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, hipStream_t, void **, void **)
// function management
HIP_DEFINE2(hipError_t, hipFuncGetAttributes, hipFuncAttributes*, void*)
HIP_DEFINE2(hipError_t, hipFuncSetCacheConfig, hipFunction_t, hipFuncCache_t)
// memory management
HIP_DEFINE3(hipError_t, hipMemcpyDtoH, void *, hipDeviceptr_t, size_t)
HIP_DEFINE1(hipError_t, hipFree, hipDeviceptr_t)
HIP_DEFINE4(hipError_t, hipMemcpyDtoHAsync, void *, hipDeviceptr_t, size_t, hipStream_t)
HIP_DEFINE4(hipError_t, hipMemcpyHtoDAsync, hipDeviceptr_t, const void *, size_t, hipStream_t)
HIP_DEFINE3(hipError_t, hipMemcpyHtoD, hipDeviceptr_t, const void *, size_t )
HIP_DEFINE2(hipError_t, hipMalloc, hipDeviceptr_t*, size_t)
HIP_DEFINE3(hipError_t, hipPointerGetAttribute, void*, CUpointer_attribute, hipDeviceptr_t)
HIP_DEFINE4(hipError_t, hipMemsetD8Async, hipDeviceptr_t, unsigned char, size_t, hipStream_t)
// event management
HIP_DEFINE2(hipError_t, hipEventCreate, hipEvent_t *, unsigned int)
HIP_DEFINE3(hipError_t, hipEventElapsedTime, float *, hipEvent_t, hipEvent_t)
HIP_DEFINE2(hipError_t, hipEventRecord, hipEvent_t, hipStream_t)
HIP_DEFINE1(hipError_t, hipEventDestroy, hipEvent_t)


/* ------------------- *
 * COMMON
 * ------------------- */

// Release
void dispatch::release(){
  if(cuda_){
    dlclose(cuda_);
    cuda_ = nullptr;
  }
}

void* dispatch::cuda_;
void* dispatch::nvml_;
void* dispatch::nvmlInit_v2_;
void* dispatch::hip_;


}
}
