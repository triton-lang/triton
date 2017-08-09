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

#include "isaac/driver/dispatch.h"
#include "isaac/driver/context.h"

namespace isaac
{
namespace driver
{

//Helpers for function definition
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

#define NVRTC_DEFINE1(ret, fname, t1) DEFINE1(nvrtcinit, nvrtc_, ret, fname, t1)
#define NVRTC_DEFINE2(ret, fname, t1, t2) DEFINE2(nvrtcinit, nvrtc_, ret, fname, t1, t2)
#define NVRTC_DEFINE3(ret, fname, t1, t2, t3) DEFINE3(nvrtcinit, nvrtc_, ret, fname, t1, t2, t3)
#define NVRTC_DEFINE4(ret, fname, t1, t2, t3, t4) DEFINE4(nvrtcinit, nvrtc_, ret, fname, t1, t2, t3, t4)
#define NVRTC_DEFINE5(ret, fname, t1, t2, t3, t4, t5) DEFINE5(nvrtcinit, nvrtc_, ret, fname, t1, t2, t3, t4, t5)
#define NVRTC_DEFINE6(ret, fname, t1, t2, t3, t4, t5, t6) DEFINE6(nvrtcinit, nvrtc_, ret, fname, t1, t2, t3, t4, t5, t6)
#define NVRTC_DEFINE7(ret, fname, t1, t2, t3, t4, t5, t6, t7) DEFINE7(nvrtcinit, nvrtc_, ret, fname, t1, t2, t3, t4, t5, t6, t7)
#define NVRTC_DEFINE8(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8) DEFINE8(nvrtcinit, nvrtc_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8)
#define NVRTC_DEFINE9(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9) DEFINE9(nvrtcinit, nvrtc_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9)
#define NVRTC_DEFINE10(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10) DEFINE10(nvrtcinit, nvrtc_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
#define NVRTC_DEFINE11(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11) DEFINE11(nvrtcinit, nvrtc_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11)

#define CUBLAS_DEFINE1(ret, fname, t1) DEFINE1(cublasinit, cublas_, ret, fname, t1)
#define CUBLAS_DEFINE13(ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13) DEFINE13(cublasinit, cublas_, ret, fname, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13)

bool dispatch::clinit()
{
  if(opencl_==nullptr)
#ifdef _MSC_VER
    opencl_ = dlopen("OpenCL.dll", RTLD_LAZY);
#else
    opencl_ = dlopen("libOpenCL.so", RTLD_LAZY);
#endif
  return opencl_ != nullptr;
}

bool dispatch::cuinit()
{
  if(cuda_==nullptr)
    cuda_ = dlopen("libcuda.so", RTLD_LAZY);
  return cuda_ != nullptr;
}

bool dispatch::nvrtcinit()
{
  if(nvrtc_==nullptr)
    nvrtc_ = dlopen("libnvrtc.so", RTLD_LAZY);
  return nvrtc_ != nullptr;
}

bool dispatch::cublasinit()
{
  if(cublas_==nullptr)
    cublas_ = dlopen("libcublas.so", RTLD_LAZY);
  return cublas_ != nullptr;
}


//OpenCL

cl_int dispatch::clBuildProgram(cl_program a, cl_uint b, const cl_device_id * c, const char * d, void (*e)(cl_program, void *), void * f)
{ return f_impl<dispatch::clinit>(opencl_, clBuildProgram, clBuildProgram_, "clBuildProgram", a, b, c, d, e, f); }

cl_context dispatch::clCreateContext(const cl_context_properties * a, cl_uint b, const cl_device_id * c, void (*d)(const char *, const void *, size_t, void *), void * e, cl_int * f)
{ return f_impl<dispatch::clinit>(opencl_, dispatch::clCreateContext, dispatch::clCreateContext_, "clCreateContext", a, b, c, d, e, f); }

cl_int dispatch::clSetEventCallback(cl_event event, cl_int a, void(CL_CALLBACK *pfn_notify)(cl_event, cl_int, void *), void * arg)
{ return f_impl<dispatch::clinit>(opencl_, dispatch::clSetEventCallback, dispatch::clSetEventCallback_, "clSetEventCallback", event, a, pfn_notify, arg); }

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
OCL_DEFINE5(cl_mem, clCreateBuffer, cl_context, cl_mem_flags, size_t, void *, cl_int *)
OCL_DEFINE6(cl_mem, clCreateImage, cl_context, cl_mem_flags, const cl_image_format *, const cl_image_desc *, void *, cl_int *) 
OCL_DEFINE5(cl_program, clCreateProgramWithSource, cl_context, cl_uint, const char **, const size_t *, cl_int *)
OCL_DEFINE1(cl_int, clReleaseKernel, cl_kernel)
OCL_DEFINE9(cl_int, clEnqueueCopyBufferToImage, cl_command_queue, cl_mem, cl_mem, size_t, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *)

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

NVRTC_DEFINE3(nvrtcResult, nvrtcCompileProgram, nvrtcProgram, int, const char **)
NVRTC_DEFINE2(nvrtcResult, nvrtcGetProgramLogSize, nvrtcProgram, size_t *)
NVRTC_DEFINE2(nvrtcResult, nvrtcGetPTX, nvrtcProgram, char *)
NVRTC_DEFINE2(nvrtcResult, nvrtcGetPTXSize, nvrtcProgram, size_t *)
NVRTC_DEFINE6(nvrtcResult, nvrtcCreateProgram, nvrtcProgram *, const char *, const char *, int, const char **, const char **)
NVRTC_DEFINE2(nvrtcResult, nvrtcGetProgramLog, nvrtcProgram, char *)

cublasHandle_t dispatch::cublasHandle(Context const & ctx)
{
  static std::map<Context, cublasHandle_t> handles;
  auto pr = handles.insert({ctx, cublasHandle_t()});
  if(pr.second)
    cublasCreate_v2(&pr.first->second);
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

void dispatch::release()
{
  if(opencl_){
    dlclose(opencl_);
    opencl_ = nullptr;
  }
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
}

void * dispatch::opencl_;
void * dispatch::cuda_;
void * dispatch::nvrtc_;
void * dispatch::cublas_;

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
void* dispatch::clCreateBuffer_;
void* dispatch::clCreateImage_;
void* dispatch::clCreateProgramWithSource_;
void* dispatch::clReleaseKernel_;
void* dispatch::clEnqueueCopyBufferToImage_;
void* dispatch::clSetEventCallback_;

//CUDA
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

void* dispatch::nvrtcCompileProgram_;
void* dispatch::nvrtcGetProgramLogSize_;
void* dispatch::nvrtcGetPTX_;
void* dispatch::nvrtcGetPTXSize_;
void* dispatch::nvrtcCreateProgram_;
void* dispatch::nvrtcGetProgramLog_;

void* dispatch::cublasCreate_v2_;
void* dispatch::cublasGetStream_v2_;
void* dispatch::cublasSetStream_v2_;
void* dispatch::cublasSgemm_v2_;
void* dispatch::cublasDgemm_v2_;

}
}
