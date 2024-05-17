/*
Copyright (c) 2022 - Present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_PT_API_H
#define HIP_INCLUDE_HIP_HIP_RUNTIME_PT_API_H

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

/// hipStreamPerThread implementation
#if defined(HIP_API_PER_THREAD_DEFAULT_STREAM)
    #define __HIP_STREAM_PER_THREAD
    #define __HIP_API_SPT(api) api ## _spt
#else
    #define __HIP_API_SPT(api) api
#endif

#if defined(__HIP_STREAM_PER_THREAD)
    // Memory APIs
    #define hipMemcpy                     __HIP_API_SPT(hipMemcpy)
    #define hipMemcpyToSymbol             __HIP_API_SPT(hipMemcpyToSymbol)
    #define hipMemcpyFromSymbol           __HIP_API_SPT(hipMemcpyFromSymbol)
    #define hipMemcpy2D                   __HIP_API_SPT(hipMemcpy2D)
    #define hipMemcpy2DFromArray          __HIP_API_SPT(hipMemcpy2DFromArray)
    #define hipMemcpy3D                   __HIP_API_SPT(hipMemcpy3D)
    #define hipMemset                     __HIP_API_SPT(hipMemset)
    #define hipMemset2D                   __HIP_API_SPT(hipMemset2D)
    #define hipMemset3D                   __HIP_API_SPT(hipMemset3D)
    #define hipMemcpyAsync                __HIP_API_SPT(hipMemcpyAsync)
    #define hipMemset3DAsync              __HIP_API_SPT(hipMemset3DAsync)
    #define hipMemset2DAsync              __HIP_API_SPT(hipMemset2DAsync)
    #define hipMemsetAsync                __HIP_API_SPT(hipMemsetAsync)
    #define hipMemcpy3DAsync              __HIP_API_SPT(hipMemcpy3DAsync)
    #define hipMemcpy2DAsync              __HIP_API_SPT(hipMemcpy2DAsync)
    #define hipMemcpyFromSymbolAsync      __HIP_API_SPT(hipMemcpyFromSymbolAsync)
    #define hipMemcpyToSymbolAsync        __HIP_API_SPT(hipMemcpyToSymbolAsync)
    #define hipMemcpyFromArray            __HIP_API_SPT(hipMemcpyFromArray)
    #define hipMemcpy2DToArray            __HIP_API_SPT(hipMemcpy2DToArray)
    #define hipMemcpy2DFromArrayAsync     __HIP_API_SPT(hipMemcpy2DFromArrayAsync)
    #define hipMemcpy2DToArrayAsync       __HIP_API_SPT(hipMemcpy2DToArrayAsync)

    // Stream APIs
    #define hipStreamSynchronize          __HIP_API_SPT(hipStreamSynchronize)
    #define hipStreamQuery                __HIP_API_SPT(hipStreamQuery)
    #define hipStreamGetFlags             __HIP_API_SPT(hipStreamGetFlags)
    #define hipStreamGetPriority          __HIP_API_SPT(hipStreamGetPriority)
    #define hipStreamWaitEvent            __HIP_API_SPT(hipStreamWaitEvent)
    #define hipStreamAddCallback          __HIP_API_SPT(hipStreamAddCallback)
    #define hipLaunchHostFunc             __HIP_API_SPT(hipLaunchHostFunc)

    // Event APIs
    #define hipEventRecord               __HIP_API_SPT(hipEventRecord)

    // Launch APIs
    #define hipLaunchKernel               __HIP_API_SPT(hipLaunchKernel)
    #define hipLaunchCooperativeKernel    __HIP_API_SPT(hipLaunchCooperativeKernel)

    // Graph APIs
    #define hipGraphLaunch                __HIP_API_SPT(hipGraphLaunch)
    #define hipStreamBeginCapture         __HIP_API_SPT(hipStreamBeginCapture)
    #define hipStreamEndCapture           __HIP_API_SPT(hipStreamEndCapture)
    #define hipStreamIsCapturing          __HIP_API_SPT(hipStreamIsCapturing)
    #define hipStreamGetCaptureInfo       __HIP_API_SPT(hipStreamGetCaptureInfo)
    #define hipStreamGetCaptureInfo_v2    __HIP_API_SPT(hipStreamGetCaptureInfo_v2)
#endif

#ifdef __cplusplus
extern "C" {
#endif

hipError_t hipMemcpy_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

hipError_t hipMemcpyToSymbol_spt(const void* symbol, const void* src, size_t sizeBytes,
                                 size_t offset __dparm(0),
                                 hipMemcpyKind kind __dparm(hipMemcpyHostToDevice));

hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol,size_t sizeBytes,
                                   size_t offset __dparm(0),
                                   hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost));

hipError_t hipMemcpy2D_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                        size_t height, hipMemcpyKind kind);

hipError_t hipMemcpy2DFromArray_spt( void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                        size_t hOffset, size_t width, size_t height, hipMemcpyKind kind);

hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms* p);

hipError_t hipMemset_spt(void* dst, int value, size_t sizeBytes);

hipError_t hipMemsetAsync_spt(void* dst, int value, size_t sizeBytes, hipStream_t stream);

hipError_t hipMemset2D_spt(void* dst, size_t pitch, int value, size_t width, size_t height);

hipError_t hipMemset2DAsync_spt(void* dst, size_t pitch, int value,
                            size_t width, size_t height, hipStream_t stream);

hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent, hipStream_t stream);

hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent );

hipError_t hipMemcpyAsync_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream);

hipError_t hipMemcpy3DAsync_spt(const hipMemcpy3DParms* p, hipStream_t stream);

hipError_t hipMemcpy2DAsync_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream);

hipError_t hipMemcpyFromSymbolAsync_spt(void* dst, const void* symbol, size_t sizeBytes,
                                    size_t offset, hipMemcpyKind kind, hipStream_t stream);

hipError_t hipMemcpyToSymbolAsync_spt(const void* symbol, const void* src, size_t sizeBytes,
                                  size_t offset, hipMemcpyKind kind, hipStream_t stream);

hipError_t hipMemcpyFromArray_spt(void* dst, hipArray_const_t src, size_t wOffsetSrc, size_t hOffset,
                                  size_t count, hipMemcpyKind kind);

hipError_t hipMemcpy2DToArray_spt(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                  size_t spitch, size_t width, size_t height, hipMemcpyKind kind);

hipError_t hipMemcpy2DFromArrayAsync_spt(void* dst, size_t dpitch, hipArray_const_t src,
                                  size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height,
                                  hipMemcpyKind kind, hipStream_t stream);

hipError_t hipMemcpy2DToArrayAsync_spt(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                  size_t spitch, size_t width, size_t height, hipMemcpyKind kind,
                                  hipStream_t stream);

hipError_t hipStreamQuery_spt(hipStream_t stream);

hipError_t hipStreamSynchronize_spt(hipStream_t stream);

hipError_t hipStreamGetPriority_spt(hipStream_t stream, int* priority);

hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags __dparm(0));

hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int* flags);

hipError_t hipStreamAddCallback_spt(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags);
#ifdef __cplusplus
hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream = NULL);
#else
hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream);
#endif

hipError_t hipLaunchCooperativeKernel_spt(const void* f,
                                      dim3 gridDim, dim3 blockDim,
                                      void **kernelParams, uint32_t sharedMemBytes, hipStream_t hStream);

hipError_t hipLaunchKernel_spt(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes, hipStream_t stream);

hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec, hipStream_t stream);
hipError_t hipStreamBeginCapture_spt(hipStream_t stream, hipStreamCaptureMode mode);
hipError_t hipStreamEndCapture_spt(hipStream_t stream, hipGraph_t* pGraph);
hipError_t hipStreamIsCapturing_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);
hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId);
hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out);
hipError_t hipLaunchHostFunc_spt(hipStream_t stream, hipHostFn_t fn, void* userData);


#ifdef __cplusplus
}
#endif // extern "C"

#endif //defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#endif //HIP_INCLUDE_HIP_HIP_RUNTIME_PT_API_H
