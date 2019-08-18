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

//Specialized helpers for OpenCL
#define OCL_DEFINE1(ret, fname, t1) DEFINE1(clinit, opencl_, ret, fname, t1)
#define OCL_DEFINE2(ret, fname, t1, t2) DEFINE2(clinit, opencl_, ret, fname, t1, t2)
#define OCL_DEFINE3(ret, fname, t1, t2, t3) DEFINE3(clinit, opencl_, ret, fname, t1, t2, t3)
#define OCL_DEFINE4(ret, fname, t1, t2, t3, t4) DEFINE4(clinit, opencl_, ret, fname, t1, t2, t3, t4)
#define OCL_DEFINE5(ret, fname, t1, t2, t3, t4, t5) DEFINE5(clinit, opencl_, ret, fname, t1, t2, t3, t4, t5)
#define OCL_DEFINE6(ret, fname, t1, t2, t3, t4, t5, t6) DEFINE6(clinit, opencl_, ret, fname, t1, t2, t3, t4, t5, t6)
#define OCL_DEFINE7(ret, fname, t1, t2, t3, t4, t5, t6, t7) DEFINE7(clinit, opencl_, ret, fname, t1, t2, t3, t4, t5, t6, t7)
#define OCL_DEFINE8(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8) DEFINE8(clinit, opencl_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8)
#define OCL_DEFINE9(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9) DEFINE9(clinit, opencl_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9)

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

bool dispatch::clinit()
{
    if(opencl_==nullptr)
        opencl_ = dlopen("libOpenCL.so", RTLD_LAZY);
    return opencl_ != nullptr;
}

bool dispatch::cuinit(){
  if(cuda_==nullptr)
    cuda_ = dlopen("libcuda.so", RTLD_LAZY);
  if(cuda_ == nullptr)
    return false;
  CUresult (*fptr)(unsigned int);
  cuInit_ = dlsym(cuda_, "cuInit");
  *reinterpret_cast<void **>(&fptr) = cuInit_;
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
CUDA_DEFINE3(CUresult, cuFuncGetAttribute, int*, CUfunction_attribute, CUfunction)
CUDA_DEFINE3(CUresult, cuFuncSetAttribute, CUfunction, CUfunction_attribute, int)
CUDA_DEFINE2(CUresult, cuFuncSetCacheConfig, CUfunction, CUfunc_cache)

NVML_DEFINE2(nvmlReturn_t, nvmlDeviceGetHandleByPciBusId_v2, const char *, nvmlDevice_t*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceGetClockInfo, nvmlDevice_t, nvmlClockType_t, unsigned int*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceGetMaxClockInfo, nvmlDevice_t, nvmlClockType_t, unsigned int*)
NVML_DEFINE3(nvmlReturn_t, nvmlDeviceSetApplicationsClocks, nvmlDevice_t, unsigned int, unsigned int)

// OpenCL
cl_int dispatch::clBuildProgram(cl_program a, cl_uint b, const cl_device_id * c, const char * d, void (*e)(cl_program, void *), void * f)
{ return f_impl<dispatch::clinit>(opencl_, clBuildProgram, clBuildProgram_, "clBuildProgram", a, b, c, d, e, f); }

cl_context dispatch::clCreateContext(const cl_context_properties * a, cl_uint b, const cl_device_id * c, void (*d)(const char *, const void *, size_t, void *), void * e, cl_int * f)
{ return f_impl<dispatch::clinit>(opencl_, dispatch::clCreateContext, dispatch::clCreateContext_, "clCreateContext", a, b, c, d, e, f); }

OCL_DEFINE9(cl_int, clEnqueueNDRangeKernel, cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*,  cl_uint, const cl_event*, cl_event*)
OCL_DEFINE4(cl_int, clSetKernelArg, cl_kernel, cl_uint, size_t, const void *)
OCL_DEFINE1(cl_int, clReleaseMemObject, cl_mem)
OCL_DEFINE1(cl_int, clFinish, cl_command_queue)
OCL_DEFINE5(cl_int, clGetMemObjectInfo, cl_mem, cl_mem_info, size_t, void *, size_t *)
OCL_DEFINE5(cl_int, clGetCommandQueueInfo, cl_command_queue, cl_command_queue_info, size_t, void *, size_t *)
OCL_DEFINE1(cl_int, clReleaseContext, cl_context)
OCL_DEFINE1(cl_int, clReleaseEvent, cl_event)
OCL_DEFINE9(cl_int, clEnqueueWriteBuffer, cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *)
OCL_DEFINE9(cl_int, clEnqueueReadBuffer, cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *)
OCL_DEFINE6(cl_int, clGetProgramBuildInfo, cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *)
OCL_DEFINE1(cl_int, clReleaseDevice, cl_device_id)
OCL_DEFINE5(cl_int, clGetDeviceIDs, cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *)
OCL_DEFINE5(cl_int, clGetContextInfo, cl_context, cl_context_info, size_t, void *, size_t *)
OCL_DEFINE5(cl_int, clGetDeviceInfo, cl_device_id, cl_device_info, size_t, void *, size_t *)
OCL_DEFINE1(cl_int, clReleaseCommandQueue, cl_command_queue)
OCL_DEFINE3(cl_int, clGetPlatformIDs, cl_uint, cl_platform_id *, cl_uint *)
OCL_DEFINE5(cl_int, clGetPlatformInfo, cl_platform_id, cl_platform_info, size_t, void *, size_t *)
OCL_DEFINE5(cl_int, clGetEventProfilingInfo, cl_event, cl_profiling_info, size_t, void *, size_t *)
OCL_DEFINE7(cl_program, clCreateProgramWithBinary, cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *)
OCL_DEFINE4(cl_command_queue, clCreateCommandQueue, cl_context, cl_device_id, cl_command_queue_properties, cl_int *)
OCL_DEFINE1(cl_int, clRetainEvent, cl_event)
OCL_DEFINE1(cl_int, clReleaseProgram, cl_program)
OCL_DEFINE1(cl_int, clFlush, cl_command_queue)
OCL_DEFINE5(cl_int, clGetProgramInfo, cl_program, cl_program_info, size_t, void *, size_t *)
OCL_DEFINE5(cl_int, clGetKernelInfo, cl_kernel, cl_kernel_info, size_t, void *, size_t *)
OCL_DEFINE6(cl_int, clGetKernelWorkGroupInfo, cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *)
OCL_DEFINE3(cl_kernel, clCreateKernel, cl_program, const char *, cl_int *)
OCL_DEFINE4(cl_int, clCreateKernelsInProgram, cl_program, cl_uint, cl_kernel*, cl_uint*)
OCL_DEFINE5(cl_mem, clCreateBuffer, cl_context, cl_mem_flags, size_t, void *, cl_int *)
OCL_DEFINE5(cl_program, clCreateProgramWithSource, cl_context, cl_uint, const char **, const size_t *, cl_int *)
OCL_DEFINE1(cl_int, clReleaseKernel, cl_kernel)

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

void * dispatch::opencl_;
void* dispatch::cuda_;
void* dispatch::nvml_;
void* dispatch::spvllvm_;

//OpenCL
void* dispatch::clBuildProgram_;
void* dispatch::clEnqueueNDRangeKernel_;
void* dispatch::clSetKernelArg_;
void* dispatch::clReleaseMemObject_;
void* dispatch::clFinish_;
void* dispatch::clGetMemObjectInfo_;
void* dispatch::clGetCommandQueueInfo_;
void* dispatch::clReleaseContext_;
void* dispatch::clReleaseEvent_;
void* dispatch::clEnqueueWriteBuffer_;
void* dispatch::clEnqueueReadBuffer_;
void* dispatch::clGetProgramBuildInfo_;
void* dispatch::clReleaseDevice_;
void* dispatch::clCreateContext_;
void* dispatch::clGetDeviceIDs_;
void* dispatch::clGetContextInfo_;
void* dispatch::clGetDeviceInfo_;
void* dispatch::clReleaseCommandQueue_;
void* dispatch::clGetPlatformIDs_;
void* dispatch::clGetPlatformInfo_;
void* dispatch::clGetEventProfilingInfo_;
void* dispatch::clCreateProgramWithBinary_;
void* dispatch::clCreateCommandQueue_;
void* dispatch::clRetainEvent_;
void* dispatch::clReleaseProgram_;
void* dispatch::clFlush_;
void* dispatch::clGetProgramInfo_;
void* dispatch::clGetKernelInfo_;
void* dispatch::clGetKernelWorkGroupInfo_;
void* dispatch::clCreateKernel_;
void* dispatch::clCreateKernelsInProgram_;
void* dispatch::clCreateBuffer_;
void* dispatch::clCreateProgramWithSource_;
void* dispatch::clReleaseKernel_;

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
void* dispatch::cuFuncGetAttribute_;
void* dispatch::cuFuncSetAttribute_;
void* dispatch::cuFuncSetCacheConfig_;

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
