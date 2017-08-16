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

#ifndef ISAAC_DRIVER_DISPATCHER_H
#define ISAAC_DRIVER_DISPATCHER_H

#include <type_traits>
#include <dlfcn.h>

//OpenCL Backend
#include "isaac/driver/external/CL/cl.h"
#include "isaac/driver/external/CL/cl_ext.h"
//CUDA Backend
#include "isaac/driver/external/CUDA/cuda.h"
#include "isaac/driver/external/CUDA/nvrtc.h"
#include "isaac/driver/external/CUDA/cublas.h"
//Exceptions
#include "isaac/driver/common.h"
#include <iostream>
#include "string.h"

#if _MSC_VER
#include <Windows.h>
#endif

namespace isaac
{
namespace driver
{

class Context;

template<class T> void check(T){}
void check(nvrtcResult err);
void check(CUresult err);
void check(cublasStatus_t err);
void check(cl_int err);
void check_destruction(CUresult);

class dispatch
{
private:
    template <class F>
    struct return_type;

    template <class R, class... A>
    struct return_type<R (*)(A...)>
    { typedef R type; };

#ifdef _MSC_VER
    template <typename F>
    struct convert_type;

    template <typename R, typename... Args>
    struct convert_type<R (*)(Args... args)>
    { typedef R (__stdcall *fun_ptr)(Args...); };
#endif

    typedef bool (*f_init_t)();

    template<f_init_t initializer, typename FunPtrT, typename... Args>
    static typename return_type<FunPtrT>::type f_impl(void*& lib_h, FunPtrT, void*& cache, const char * name, Args... args)
    {
        initializer();
        if(cache == nullptr)
            cache = dlsym(lib_h, name);
#ifdef _MSC_VER
        typename convert_type<FunPtrT>::fun_ptr fptr;
#else
        FunPtrT fptr;
#endif
        *reinterpret_cast<void **>(&fptr) = cache;
        typename return_type<FunPtrT>::type res = (*fptr)(args...);

        if (strncmp(name, "clGetDeviceIDs", 15)) {
            check(res);
        }
        return res;
    }

public:
    static bool clinit();
    static bool cublasinit();
    static bool nvrtcinit();
    static bool cuinit();

    static void release();

    //OpenCL
    static cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id *, const char *, void (*)(cl_program, void *), void *);
    static cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
    static cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void *);
    static cl_int clReleaseMemObject(cl_mem);
    static cl_int clFinish(cl_command_queue);
    static cl_int clGetMemObjectInfo(cl_mem, cl_mem_info, size_t, void *, size_t *);
    static cl_int clGetCommandQueueInfo(cl_command_queue, cl_command_queue_info, size_t, void *, size_t *);
    static cl_int clReleaseContext(cl_context);
    static cl_int clReleaseEvent(cl_event);
    static cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
    static cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
    static cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
    static cl_int clReleaseDevice(cl_device_id);
    static cl_context clCreateContext(const cl_context_properties *, cl_uint, const cl_device_id *, void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
    static cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    static cl_int clGetContextInfo(cl_context, cl_context_info, size_t, void *, size_t *);
    static cl_int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void *, size_t *);
    static cl_int clReleaseCommandQueue(cl_command_queue);
    static cl_int clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *);
    static cl_int clGetPlatformInfo(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
    static cl_int clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t, void *, size_t *);
    static cl_program clCreateProgramWithBinary(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *);
    static cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
    static cl_int clRetainEvent(cl_event);
    static cl_int clReleaseProgram(cl_program);
    static cl_int clFlush(cl_command_queue);
    static cl_int clGetProgramInfo(cl_program, cl_program_info, size_t, void *, size_t *);
    static cl_int clGetKernelInfo(cl_kernel, cl_kernel_info, size_t, void *, size_t *);
    static cl_int clGetKernelWorkGroupInfo(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);
    static cl_kernel clCreateKernel(cl_program, const char *, cl_int *);
    static cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    static cl_mem clCreateImage(cl_context, cl_mem_flags, const cl_image_format *, const cl_image_desc *, void *, cl_int *);
    static cl_program clCreateProgramWithSource(cl_context, cl_uint, const char **, const size_t *, cl_int *);
    static cl_int clReleaseKernel(cl_kernel);
    static cl_int clEnqueueCopyBufferToImage(cl_command_queue, cl_mem, cl_mem, size_t, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
    static cl_int clSetEventCallback(cl_event, cl_int, void (CL_CALLBACK * /* pfn_notify */)(cl_event, cl_int, void *), void *);

    //CUDA
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
    static CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
    static CUresult cuStreamSynchronize(CUstream hStream);
    static CUresult cuStreamDestroy_v2(CUstream hStream);
    static CUresult cuEventDestroy_v2(CUevent hEvent);
    static CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
    static CUresult cuPointerGetAttribute(void * data, CUpointer_attribute attribute, CUdeviceptr ptr);
    static CUresult cuCtxGetDevice(CUdevice* result);

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

private:
    static void* opencl_;
    static void* cuda_;
    static void* nvrtc_;
    static void* cublas_;


    //OpenCL
    static void* clBuildProgram_;
    static void* clEnqueueNDRangeKernel_;
    static void* clSetKernelArg_;
    static void* clReleaseMemObject_;
    static void* clFinish_;
    static void* clGetMemObjectInfo_;
    static void* clGetCommandQueueInfo_;
    static void* clReleaseContext_;
    static void* clReleaseEvent_;
    static void* clEnqueueWriteBuffer_;
    static void* clEnqueueReadBuffer_;
    static void* clGetProgramBuildInfo_;
    static void* clReleaseDevice_;
    static void* clCreateContext_;
    static void* clGetDeviceIDs_;
    static void* clGetContextInfo_;
    static void* clGetDeviceInfo_;
    static void* clReleaseCommandQueue_;
    static void* clGetPlatformIDs_;
    static void* clGetPlatformInfo_;
    static void* clGetEventProfilingInfo_;
    static void* clCreateProgramWithBinary_;
    static void* clCreateCommandQueue_;
    static void* clRetainEvent_;
    static void* clReleaseProgram_;
    static void* clFlush_;
    static void* clGetProgramInfo_;
    static void* clGetKernelInfo_;
    static void* clGetKernelWorkGroupInfo_;
    static void* clCreateKernel_;
    static void* clCreateBuffer_;
    static void* clCreateImage_;
    static void* clCreateProgramWithSource_;
    static void* clReleaseKernel_;
    static void* clEnqueueCopyBufferToImage_;
    static void* clSetEventCallback_;

    //CUDA
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

    static void* nvrtcCompileProgram_;
    static void* nvrtcGetProgramLogSize_;
    static void* nvrtcGetPTX_;
    static void* nvrtcGetPTXSize_;
    static void* nvrtcCreateProgram_;
    static void* nvrtcGetProgramLog_;

    static void* cublasCreate_v2_;
    static void* cublasGetStream_v2_;
    static void* cublasSetStream_v2_;
    static void* cublasSgemm_v2_;
    static void* cublasDgemm_v2_;
};

}
}



#endif
