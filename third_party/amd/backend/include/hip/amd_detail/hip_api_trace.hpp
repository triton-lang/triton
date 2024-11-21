/*
    Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime.h>

// Define some version macros for the API table. Use similar naming conventions to HSA-runtime
// (MAJOR and STEP versions). Three groups at this time:
//
// (A) HIP_API_TABLE_* defines for versioning for API table structure
// (B) HIP_RUNTIME_API_TABLE_* defines for versioning the HipDispatchTable struct
// (C) HIP_COMPILER_API_TABLE_* defines for versioning the HipCompilerDispatchTable struct
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     IMPORTANT    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//    1. When new functions are added to the API table, always add the new function pointer to the
//       end of the table and increment the dispatch table's step version number. NEVER re-arrange
//       the order of the member variables in a dispatch table. This will break the ABI.
//    2. In dire circumstances, if the type of an existing member variable in a dispatch
//       table has be changed because a data type has been changed/removed, increment the dispatch
//       table's major version number. If the function pointer type can no longer be declared, DO
//       NOT REMOVE IT! Make the function pointer type void* and have it always be set to a nullptr.
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// The major version number should (ideally) never need to be incremented.
// - Increment the HIP_API_TABLE_MAJOR_VERSION for fundamental changes to the API table structs.
// - Increment the HIP_RUNTIME_API_TABLE_MAJOR_VERSION for fundamental changes to the
//   HipDispatchTable struct, such as a *change* to type/name an existing member variable. DO NOT
//   REMOVE IT.
// - Increment the HIP_COMPILER_API_TABLE_MAJOR_VERSION for fundamental changes to the
//   HipCompilerDispatchTable struct, such as a *change* to type/name an existing member variable.
//   DO NOT REMOVE IT.
#define HIP_API_TABLE_MAJOR_VERSION 0
#define HIP_COMPILER_API_TABLE_MAJOR_VERSION 0
#define HIP_RUNTIME_API_TABLE_MAJOR_VERSION 0

// The step version number should be changed whenever the size of the API table struct(s) change.
// - Increment the HIP_API_TABLE_STEP_VERSION when/if new API table structs are added
// - Increment the HIP_RUNTIME_API_TABLE_STEP_VERSION when new runtime API functions are added
// - Increment the HIP_COMPILER_API_TABLE_STEP_VERSION when new compiler API functions are added
// - Reset any of the *_STEP_VERSION defines to zero if the corresponding *_MAJOR_VERSION increases
#define HIP_API_TABLE_STEP_VERSION 0
#define HIP_COMPILER_API_TABLE_STEP_VERSION 0
#define HIP_RUNTIME_API_TABLE_STEP_VERSION 3

// HIP API interface
typedef hipError_t (*t___hipPopCallConfiguration)(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                                                  hipStream_t* stream);
typedef hipError_t (*t___hipPushCallConfiguration)(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                                   hipStream_t stream);
typedef void** (*t___hipRegisterFatBinary)(const void* data);
typedef void (*t___hipRegisterFunction)(void** modules, const void* hostFunction,
                                        char* deviceFunction, const char* deviceName,
                                        unsigned int threadLimit, uint3* tid, uint3* bid,
                                        dim3* blockDim, dim3* gridDim, int* wSize);
typedef void (*t___hipRegisterManagedVar)(void* hipModule, void** pointer, void* init_value,
                                          const char* name, size_t size, unsigned align);
typedef void (*t___hipRegisterSurface)(void** modules, void* var, char* hostVar,
                                       char* deviceVar, int type, int ext);
typedef void (*t___hipRegisterTexture)(void** modules, void* var, char* hostVar,
                                       char* deviceVar, int type, int norm, int ext);
typedef void (*t___hipRegisterVar)(void** modules, void* var, char* hostVar,
                                   char* deviceVar, int ext, size_t size, int constant, int global);
typedef void (*t___hipUnregisterFatBinary)(void** modules);

typedef const char* (*t_hipApiName)(uint32_t id);
typedef hipError_t (*t_hipArray3DCreate)(hipArray_t* array,
                                         const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray);
typedef hipError_t (*t_hipArray3DGetDescriptor)(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor,
                                                hipArray_t array);
typedef hipError_t (*t_hipArrayCreate)(hipArray_t* pHandle,
                                       const HIP_ARRAY_DESCRIPTOR* pAllocateArray);
typedef hipError_t (*t_hipArrayDestroy)(hipArray_t array);
typedef hipError_t (*t_hipArrayGetDescriptor)(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor,
                                              hipArray_t array);
typedef hipError_t (*t_hipArrayGetInfo)(hipChannelFormatDesc* desc, hipExtent* extent,
                                        unsigned int* flags, hipArray_t array);
typedef hipError_t (*t_hipBindTexture)(size_t* offset, const textureReference* tex,
                                       const void* devPtr, const hipChannelFormatDesc* desc,
                                       size_t size);
typedef hipError_t (*t_hipBindTexture2D)(size_t* offset, const textureReference* tex,
                                         const void* devPtr, const hipChannelFormatDesc* desc,
                                         size_t width, size_t height, size_t pitch);
typedef hipError_t (*t_hipBindTextureToArray)(const textureReference* tex, hipArray_const_t array,
                                              const hipChannelFormatDesc* desc);
typedef hipError_t (*t_hipBindTextureToMipmappedArray)(const textureReference* tex,
                                                       hipMipmappedArray_const_t mipmappedArray,
                                                       const hipChannelFormatDesc* desc);
typedef hipError_t (*t_hipChooseDevice)(int* device, const hipDeviceProp_t* prop);
typedef hipError_t (*t_hipChooseDeviceR0000)(int* device, const hipDeviceProp_tR0000* properties);
typedef hipError_t (*t_hipConfigureCall)(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                         hipStream_t stream);
typedef hipError_t (*t_hipCreateSurfaceObject)(hipSurfaceObject_t* pSurfObject,
                                               const hipResourceDesc* pResDesc);
typedef hipError_t (*t_hipCreateTextureObject)(hipTextureObject_t* pTexObject,
                                               const hipResourceDesc* pResDesc,
                                               const hipTextureDesc* pTexDesc,
                                               const struct hipResourceViewDesc* pResViewDesc);
typedef hipError_t (*t_hipCtxCreate)(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
typedef hipError_t (*t_hipCtxDestroy)(hipCtx_t ctx);
typedef hipError_t (*t_hipCtxDisablePeerAccess)(hipCtx_t peerCtx);
typedef hipError_t (*t_hipCtxEnablePeerAccess)(hipCtx_t peerCtx, unsigned int flags);
typedef hipError_t (*t_hipCtxGetApiVersion)(hipCtx_t ctx, int* apiVersion);
typedef hipError_t (*t_hipCtxGetCacheConfig)(hipFuncCache_t* cacheConfig);
typedef hipError_t (*t_hipCtxGetCurrent)(hipCtx_t* ctx);
typedef hipError_t (*t_hipCtxGetDevice)(hipDevice_t* device);
typedef hipError_t (*t_hipCtxGetFlags)(unsigned int* flags);
typedef hipError_t (*t_hipCtxGetSharedMemConfig)(hipSharedMemConfig* pConfig);
typedef hipError_t (*t_hipCtxPopCurrent)(hipCtx_t* ctx);
typedef hipError_t (*t_hipCtxPushCurrent)(hipCtx_t ctx);
typedef hipError_t (*t_hipCtxSetCacheConfig)(hipFuncCache_t cacheConfig);
typedef hipError_t (*t_hipCtxSetCurrent)(hipCtx_t ctx);
typedef hipError_t (*t_hipCtxSetSharedMemConfig)(hipSharedMemConfig config);
typedef hipError_t (*t_hipCtxSynchronize)(void);
typedef hipError_t (*t_hipDestroyExternalMemory)(hipExternalMemory_t extMem);
typedef hipError_t (*t_hipDestroyExternalSemaphore)(hipExternalSemaphore_t extSem);
typedef hipError_t (*t_hipDestroySurfaceObject)(hipSurfaceObject_t surfaceObject);
typedef hipError_t (*t_hipDestroyTextureObject)(hipTextureObject_t textureObject);
typedef hipError_t (*t_hipDeviceCanAccessPeer)(int* canAccessPeer, int deviceId, int peerDeviceId);
typedef hipError_t (*t_hipDeviceComputeCapability)(int* major, int* minor, hipDevice_t device);
typedef hipError_t (*t_hipDeviceDisablePeerAccess)(int peerDeviceId);
typedef hipError_t (*t_hipDeviceEnablePeerAccess)(int peerDeviceId, unsigned int flags);
typedef hipError_t (*t_hipDeviceGet)(hipDevice_t* device, int ordinal);
typedef hipError_t (*t_hipDeviceGetAttribute)(int* pi, hipDeviceAttribute_t attr, int deviceId);
typedef hipError_t (*t_hipDeviceGetByPCIBusId)(int* device, const char* pciBusId);
typedef hipError_t (*t_hipDeviceGetCacheConfig)(hipFuncCache_t* cacheConfig);
typedef hipError_t (*t_hipDeviceGetDefaultMemPool)(hipMemPool_t* mem_pool, int device);
typedef hipError_t (*t_hipDeviceGetGraphMemAttribute)(int device, hipGraphMemAttributeType attr,
                                                      void* value);
typedef hipError_t (*t_hipDeviceGetLimit)(size_t* pValue, enum hipLimit_t limit);
typedef hipError_t (*t_hipDeviceGetMemPool)(hipMemPool_t* mem_pool, int device);
typedef hipError_t (*t_hipDeviceGetName)(char* name, int len, hipDevice_t device);
typedef hipError_t (*t_hipDeviceGetP2PAttribute)(int* value, hipDeviceP2PAttr attr, int srcDevice,
                                                 int dstDevice);
typedef hipError_t (*t_hipDeviceGetPCIBusId)(char* pciBusId, int len, int device);
typedef hipError_t (*t_hipDeviceGetSharedMemConfig)(hipSharedMemConfig* pConfig);
typedef hipError_t (*t_hipDeviceGetStreamPriorityRange)(int* leastPriority, int* greatestPriority);
typedef hipError_t (*t_hipDeviceGetUuid)(hipUUID* uuid, hipDevice_t device);
typedef hipError_t (*t_hipDeviceGraphMemTrim)(int device);
typedef hipError_t (*t_hipDevicePrimaryCtxGetState)(hipDevice_t dev, unsigned int* flags,
                                                    int* active);
typedef hipError_t (*t_hipDevicePrimaryCtxRelease)(hipDevice_t dev);
typedef hipError_t (*t_hipDevicePrimaryCtxReset)(hipDevice_t dev);
typedef hipError_t (*t_hipDevicePrimaryCtxRetain)(hipCtx_t* pctx, hipDevice_t dev);
typedef hipError_t (*t_hipDevicePrimaryCtxSetFlags)(hipDevice_t dev, unsigned int flags);
typedef hipError_t (*t_hipDeviceReset)(void);
typedef hipError_t (*t_hipDeviceSetCacheConfig)(hipFuncCache_t cacheConfig);
typedef hipError_t (*t_hipDeviceSetGraphMemAttribute)(int device, hipGraphMemAttributeType attr,
                                                      void* value);
typedef hipError_t (*t_hipDeviceSetLimit)(enum hipLimit_t limit, size_t value);
typedef hipError_t (*t_hipDeviceSetMemPool)(int device, hipMemPool_t mem_pool);
typedef hipError_t (*t_hipDeviceSetSharedMemConfig)(hipSharedMemConfig config);
typedef hipError_t (*t_hipDeviceSynchronize)(void);
typedef hipError_t (*t_hipDeviceTotalMem)(size_t* bytes, hipDevice_t device);
typedef hipError_t (*t_hipDriverGetVersion)(int* driverVersion);
typedef hipError_t (*t_hipDrvGetErrorName)(hipError_t hipError, const char** errorString);
typedef hipError_t (*t_hipDrvGetErrorString)(hipError_t hipError, const char** errorString);
typedef hipError_t (*t_hipDrvGraphAddMemcpyNode)(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                                 const hipGraphNode_t* dependencies,
                                                 size_t numDependencies,
                                                 const HIP_MEMCPY3D* copyParams, hipCtx_t ctx);
typedef hipError_t (*t_hipDrvMemcpy2DUnaligned)(const hip_Memcpy2D* pCopy);
typedef hipError_t (*t_hipDrvMemcpy3D)(const HIP_MEMCPY3D* pCopy);
typedef hipError_t (*t_hipDrvMemcpy3DAsync)(const HIP_MEMCPY3D* pCopy, hipStream_t stream);
typedef hipError_t (*t_hipDrvPointerGetAttributes)(unsigned int numAttributes,
                                                   hipPointer_attribute* attributes, void** data,
                                                   hipDeviceptr_t ptr);
typedef hipError_t (*t_hipEventCreate)(hipEvent_t* event);
typedef hipError_t (*t_hipEventCreateWithFlags)(hipEvent_t* event, unsigned flags);
typedef hipError_t (*t_hipEventDestroy)(hipEvent_t event);
typedef hipError_t (*t_hipEventElapsedTime)(float* ms, hipEvent_t start, hipEvent_t stop);
typedef hipError_t (*t_hipEventQuery)(hipEvent_t event);
typedef hipError_t (*t_hipEventRecord)(hipEvent_t event, hipStream_t stream);
typedef hipError_t (*t_hipEventSynchronize)(hipEvent_t event);
typedef hipError_t (*t_hipExtGetLinkTypeAndHopCount)(int device1, int device2, uint32_t* linktype,
                                                     uint32_t* hopcount);
typedef hipError_t (*t_hipExtLaunchKernel)(const void* function_address, dim3 numBlocks,
                                           dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                           hipStream_t stream, hipEvent_t startEvent,
                                           hipEvent_t stopEvent, int flags);
typedef hipError_t (*t_hipExtLaunchMultiKernelMultiDevice)(hipLaunchParams* launchParamsList,
                                                           int numDevices, unsigned int flags);
typedef hipError_t (*t_hipExtMallocWithFlags)(void** ptr, size_t sizeBytes, unsigned int flags);
typedef hipError_t (*t_hipExtStreamCreateWithCUMask)(hipStream_t* stream, uint32_t cuMaskSize,
                                                     const uint32_t* cuMask);
typedef hipError_t (*t_hipExtStreamGetCUMask)(hipStream_t stream, uint32_t cuMaskSize,
                                              uint32_t* cuMask);
typedef hipError_t (*t_hipExternalMemoryGetMappedBuffer)(
    void** devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc* bufferDesc);
typedef hipError_t (*t_hipFree)(void* ptr);
typedef hipError_t (*t_hipFreeArray)(hipArray_t array);
typedef hipError_t (*t_hipFreeAsync)(void* dev_ptr, hipStream_t stream);
typedef hipError_t (*t_hipFreeHost)(void* ptr);
typedef hipError_t (*t_hipFreeMipmappedArray)(hipMipmappedArray_t mipmappedArray);
typedef hipError_t (*t_hipFuncGetAttribute)(int* value, hipFunction_attribute attrib,
                                            hipFunction_t hfunc);
typedef hipError_t (*t_hipFuncGetAttributes)(struct hipFuncAttributes* attr, const void* func);
typedef hipError_t (*t_hipFuncSetAttribute)(const void* func, hipFuncAttribute attr, int value);
typedef hipError_t (*t_hipFuncSetCacheConfig)(const void* func, hipFuncCache_t config);
typedef hipError_t (*t_hipFuncSetSharedMemConfig)(const void* func, hipSharedMemConfig config);
typedef hipError_t (*t_hipGLGetDevices)(unsigned int* pHipDeviceCount, int* pHipDevices,
                                        unsigned int hipDeviceCount, hipGLDeviceList deviceList);
typedef hipError_t (*t_hipGetChannelDesc)(hipChannelFormatDesc* desc, hipArray_const_t array);
typedef hipError_t (*t_hipGetDevice)(int* deviceId);
typedef hipError_t (*t_hipGetDeviceCount)(int* count);
typedef hipError_t (*t_hipGetDeviceFlags)(unsigned int* flags);
typedef hipError_t (*t_hipGetDevicePropertiesR0600)(hipDeviceProp_tR0600* prop, int device);
typedef hipError_t (*t_hipGetDevicePropertiesR0000)(hipDeviceProp_tR0000* prop, int device);
typedef const char* (*t_hipGetErrorName)(hipError_t hip_error);
typedef const char* (*t_hipGetErrorString)(hipError_t hipError);
typedef hipError_t (*t_hipGetLastError)(void);
typedef hipError_t (*t_hipGetMipmappedArrayLevel)(hipArray_t* levelArray,
                                                  hipMipmappedArray_const_t mipmappedArray,
                                                  unsigned int level);
typedef hipError_t (*t_hipGetSymbolAddress)(void** devPtr, const void* symbol);
typedef hipError_t (*t_hipGetSymbolSize)(size_t* size, const void* symbol);
typedef hipError_t (*t_hipGetTextureAlignmentOffset)(size_t* offset,
                                                     const textureReference* texref);
typedef hipError_t (*t_hipGetTextureObjectResourceDesc)(hipResourceDesc* pResDesc,
                                                        hipTextureObject_t textureObject);
typedef hipError_t (*t_hipGetTextureObjectResourceViewDesc)(
    struct hipResourceViewDesc* pResViewDesc, hipTextureObject_t textureObject);
typedef hipError_t (*t_hipGetTextureObjectTextureDesc)(hipTextureDesc* pTexDesc,
                                                       hipTextureObject_t textureObject);
typedef hipError_t (*t_hipGetTextureReference)(const textureReference** texref, const void* symbol);
typedef hipError_t (*t_hipGraphAddChildGraphNode)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                  const hipGraphNode_t* pDependencies,
                                                  size_t numDependencies, hipGraph_t childGraph);
typedef hipError_t (*t_hipGraphAddDependencies)(hipGraph_t graph, const hipGraphNode_t* from,
                                                const hipGraphNode_t* to, size_t numDependencies);
typedef hipError_t (*t_hipGraphAddEmptyNode)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                             const hipGraphNode_t* pDependencies,
                                             size_t numDependencies);
typedef hipError_t (*t_hipGraphAddEventRecordNode)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                   const hipGraphNode_t* pDependencies,
                                                   size_t numDependencies, hipEvent_t event);
typedef hipError_t (*t_hipGraphAddEventWaitNode)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                 const hipGraphNode_t* pDependencies,
                                                 size_t numDependencies, hipEvent_t event);
typedef hipError_t (*t_hipGraphAddHostNode)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                            const hipGraphNode_t* pDependencies,
                                            size_t numDependencies,
                                            const hipHostNodeParams* pNodeParams);
typedef hipError_t (*t_hipGraphAddKernelNode)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                              const hipGraphNode_t* pDependencies,
                                              size_t numDependencies,
                                              const hipKernelNodeParams* pNodeParams);
typedef hipError_t (*t_hipGraphAddMemAllocNode)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                const hipGraphNode_t* pDependencies,
                                                size_t numDependencies,
                                                hipMemAllocNodeParams* pNodeParams);
typedef hipError_t (*t_hipGraphAddMemFreeNode)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                               const hipGraphNode_t* pDependencies,
                                               size_t numDependencies, void* dev_ptr);
typedef hipError_t (*t_hipGraphAddMemcpyNode)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                              const hipGraphNode_t* pDependencies,
                                              size_t numDependencies,
                                              const hipMemcpy3DParms* pCopyParams);
typedef hipError_t (*t_hipGraphAddMemcpyNode1D)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                const hipGraphNode_t* pDependencies,
                                                size_t numDependencies, void* dst, const void* src,
                                                size_t count, hipMemcpyKind kind);
typedef hipError_t (*t_hipGraphAddMemcpyNodeFromSymbol)(hipGraphNode_t* pGraphNode,
                                                        hipGraph_t graph,
                                                        const hipGraphNode_t* pDependencies,
                                                        size_t numDependencies, void* dst,
                                                        const void* symbol, size_t count,
                                                        size_t offset, hipMemcpyKind kind);
typedef hipError_t (*t_hipGraphAddMemcpyNodeToSymbol)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                      const hipGraphNode_t* pDependencies,
                                                      size_t numDependencies, const void* symbol,
                                                      const void* src, size_t count, size_t offset,
                                                      hipMemcpyKind kind);
typedef hipError_t (*t_hipGraphAddMemsetNode)(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                              const hipGraphNode_t* pDependencies,
                                              size_t numDependencies,
                                              const hipMemsetParams* pMemsetParams);

typedef hipError_t (*t_hipGraphChildGraphNodeGetGraph)(hipGraphNode_t node, hipGraph_t* pGraph);
typedef hipError_t (*t_hipGraphClone)(hipGraph_t* pGraphClone, hipGraph_t originalGraph);
typedef hipError_t (*t_hipGraphCreate)(hipGraph_t* pGraph, unsigned int flags);
typedef hipError_t (*t_hipGraphDebugDotPrint)(hipGraph_t graph, const char* path,
                                              unsigned int flags);
typedef hipError_t (*t_hipGraphDestroy)(hipGraph_t graph);
typedef hipError_t (*t_hipGraphDestroyNode)(hipGraphNode_t node);
typedef hipError_t (*t_hipGraphEventRecordNodeGetEvent)(hipGraphNode_t node, hipEvent_t* event_out);
typedef hipError_t (*t_hipGraphEventRecordNodeSetEvent)(hipGraphNode_t node, hipEvent_t event);
typedef hipError_t (*t_hipGraphEventWaitNodeGetEvent)(hipGraphNode_t node, hipEvent_t* event_out);
typedef hipError_t (*t_hipGraphEventWaitNodeSetEvent)(hipGraphNode_t node, hipEvent_t event);
typedef hipError_t (*t_hipGraphExecChildGraphNodeSetParams)(hipGraphExec_t hGraphExec,
                                                            hipGraphNode_t node,
                                                            hipGraph_t childGraph);
typedef hipError_t (*t_hipGraphExecDestroy)(hipGraphExec_t graphExec);
typedef hipError_t (*t_hipGraphExecEventRecordNodeSetEvent)(hipGraphExec_t hGraphExec,
                                                            hipGraphNode_t hNode, hipEvent_t event);
typedef hipError_t (*t_hipGraphExecEventWaitNodeSetEvent)(hipGraphExec_t hGraphExec,
                                                          hipGraphNode_t hNode, hipEvent_t event);
typedef hipError_t (*t_hipGraphExecHostNodeSetParams)(hipGraphExec_t hGraphExec,
                                                      hipGraphNode_t node,
                                                      const hipHostNodeParams* pNodeParams);
typedef hipError_t (*t_hipGraphExecKernelNodeSetParams)(hipGraphExec_t hGraphExec,
                                                        hipGraphNode_t node,
                                                        const hipKernelNodeParams* pNodeParams);
typedef hipError_t (*t_hipGraphExecMemcpyNodeSetParams)(hipGraphExec_t hGraphExec,
                                                        hipGraphNode_t node,
                                                        hipMemcpy3DParms* pNodeParams);
typedef hipError_t (*t_hipGraphExecMemcpyNodeSetParams1D)(hipGraphExec_t hGraphExec,
                                                          hipGraphNode_t node, void* dst,
                                                          const void* src, size_t count,
                                                          hipMemcpyKind kind);
typedef hipError_t (*t_hipGraphExecMemcpyNodeSetParamsFromSymbol)(hipGraphExec_t hGraphExec,
                                                                  hipGraphNode_t node, void* dst,
                                                                  const void* symbol, size_t count,
                                                                  size_t offset,
                                                                  hipMemcpyKind kind);
typedef hipError_t (*t_hipGraphExecMemcpyNodeSetParamsToSymbol)(hipGraphExec_t hGraphExec,
                                                                hipGraphNode_t node,
                                                                const void* symbol, const void* src,
                                                                size_t count, size_t offset,
                                                                hipMemcpyKind kind);
typedef hipError_t (*t_hipGraphExecMemsetNodeSetParams)(hipGraphExec_t hGraphExec,
                                                        hipGraphNode_t node,
                                                        const hipMemsetParams* pNodeParams);
typedef hipError_t (*t_hipGraphExecUpdate)(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                                           hipGraphNode_t* hErrorNode_out,
                                           hipGraphExecUpdateResult* updateResult_out);
typedef hipError_t (*t_hipGraphGetEdges)(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                                         size_t* numEdges);
typedef hipError_t (*t_hipGraphGetNodes)(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes);
typedef hipError_t (*t_hipGraphGetRootNodes)(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                             size_t* pNumRootNodes);
typedef hipError_t (*t_hipGraphHostNodeGetParams)(hipGraphNode_t node,
                                                  hipHostNodeParams* pNodeParams);
typedef hipError_t (*t_hipGraphHostNodeSetParams)(hipGraphNode_t node,
                                                  const hipHostNodeParams* pNodeParams);
typedef hipError_t (*t_hipGraphInstantiate)(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                            hipGraphNode_t* pErrorNode, char* pLogBuffer,
                                            size_t bufferSize);
typedef hipError_t (*t_hipGraphInstantiateWithFlags)(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                                     unsigned long long flags);
typedef hipError_t (*t_hipGraphKernelNodeCopyAttributes)(hipGraphNode_t hSrc, hipGraphNode_t hDst);
typedef hipError_t (*t_hipGraphKernelNodeGetAttribute)(hipGraphNode_t hNode,
                                                       hipKernelNodeAttrID attr,
                                                       hipKernelNodeAttrValue* value);
typedef hipError_t (*t_hipGraphKernelNodeGetParams)(hipGraphNode_t node,
                                                    hipKernelNodeParams* pNodeParams);
typedef hipError_t (*t_hipGraphKernelNodeSetAttribute)(hipGraphNode_t hNode,
                                                       hipKernelNodeAttrID attr,
                                                       const hipKernelNodeAttrValue* value);
typedef hipError_t (*t_hipGraphKernelNodeSetParams)(hipGraphNode_t node,
                                                    const hipKernelNodeParams* pNodeParams);
typedef hipError_t (*t_hipGraphLaunch)(hipGraphExec_t graphExec, hipStream_t stream);
typedef hipError_t (*t_hipGraphMemAllocNodeGetParams)(hipGraphNode_t node,
                                                      hipMemAllocNodeParams* pNodeParams);
typedef hipError_t (*t_hipGraphMemFreeNodeGetParams)(hipGraphNode_t node, void* dev_ptr);
typedef hipError_t (*t_hipGraphMemcpyNodeGetParams)(hipGraphNode_t node,
                                                    hipMemcpy3DParms* pNodeParams);
typedef hipError_t (*t_hipGraphMemcpyNodeSetParams)(hipGraphNode_t node,
                                                    const hipMemcpy3DParms* pNodeParams);
typedef hipError_t (*t_hipGraphMemcpyNodeSetParams1D)(hipGraphNode_t node, void* dst,
                                                      const void* src, size_t count,
                                                      hipMemcpyKind kind);
typedef hipError_t (*t_hipGraphMemcpyNodeSetParamsFromSymbol)(hipGraphNode_t node, void* dst,
                                                              const void* symbol, size_t count,
                                                              size_t offset, hipMemcpyKind kind);
typedef hipError_t (*t_hipGraphMemcpyNodeSetParamsToSymbol)(hipGraphNode_t node, const void* symbol,
                                                            const void* src, size_t count,
                                                            size_t offset, hipMemcpyKind kind);
typedef hipError_t (*t_hipGraphMemsetNodeGetParams)(hipGraphNode_t node,
                                                    hipMemsetParams* pNodeParams);
typedef hipError_t (*t_hipGraphMemsetNodeSetParams)(hipGraphNode_t node,
                                                    const hipMemsetParams* pNodeParams);
typedef hipError_t (*t_hipGraphNodeFindInClone)(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                                hipGraph_t clonedGraph);
typedef hipError_t (*t_hipGraphNodeGetDependencies)(hipGraphNode_t node,
                                                    hipGraphNode_t* pDependencies,
                                                    size_t* pNumDependencies);
typedef hipError_t (*t_hipGraphNodeGetDependentNodes)(hipGraphNode_t node,
                                                      hipGraphNode_t* pDependentNodes,
                                                      size_t* pNumDependentNodes);
typedef hipError_t (*t_hipGraphNodeGetEnabled)(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               unsigned int* isEnabled);
typedef hipError_t (*t_hipGraphNodeGetType)(hipGraphNode_t node, hipGraphNodeType* pType);
typedef hipError_t (*t_hipGraphNodeSetEnabled)(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               unsigned int isEnabled);
typedef hipError_t (*t_hipGraphReleaseUserObject)(hipGraph_t graph, hipUserObject_t object,
                                                  unsigned int count);
typedef hipError_t (*t_hipGraphRemoveDependencies)(hipGraph_t graph, const hipGraphNode_t* from,
                                                   const hipGraphNode_t* to,
                                                   size_t numDependencies);
typedef hipError_t (*t_hipGraphRetainUserObject)(hipGraph_t graph, hipUserObject_t object,
                                                 unsigned int count, unsigned int flags);
typedef hipError_t (*t_hipGraphUpload)(hipGraphExec_t graphExec, hipStream_t stream);
typedef hipError_t (*t_hipGraphicsGLRegisterBuffer)(hipGraphicsResource** resource, GLuint buffer,
                                                    unsigned int flags);
typedef hipError_t (*t_hipGraphicsGLRegisterImage)(hipGraphicsResource** resource, GLuint image,
                                                   GLenum target, unsigned int flags);
typedef hipError_t (*t_hipGraphicsMapResources)(int count, hipGraphicsResource_t* resources,
                                                hipStream_t stream);
typedef hipError_t (*t_hipGraphicsResourceGetMappedPointer)(void** devPtr, size_t* size,
                                                            hipGraphicsResource_t resource);
typedef hipError_t (*t_hipGraphicsSubResourceGetMappedArray)(hipArray_t* array,
                                                             hipGraphicsResource_t resource,
                                                             unsigned int arrayIndex,
                                                             unsigned int mipLevel);
typedef hipError_t (*t_hipGraphicsUnmapResources)(int count, hipGraphicsResource_t* resources,
                                                  hipStream_t stream);
typedef hipError_t (*t_hipGraphicsUnregisterResource)(hipGraphicsResource_t resource);
typedef hipError_t (*t_hipHostAlloc)(void** ptr, size_t size, unsigned int flags);
typedef hipError_t (*t_hipHostFree)(void* ptr);
typedef hipError_t (*t_hipHostGetDevicePointer)(void** devPtr, void* hstPtr, unsigned int flags);
typedef hipError_t (*t_hipHostGetFlags)(unsigned int* flagsPtr, void* hostPtr);
typedef hipError_t (*t_hipHostMalloc)(void** ptr, size_t size, unsigned int flags);
typedef hipError_t (*t_hipHostRegister)(void* hostPtr, size_t sizeBytes, unsigned int flags);
typedef hipError_t (*t_hipHostUnregister)(void* hostPtr);
typedef hipError_t (*t_hipImportExternalMemory)(hipExternalMemory_t* extMem_out,
                                                const hipExternalMemoryHandleDesc* memHandleDesc);
typedef hipError_t (*t_hipImportExternalSemaphore)(
    hipExternalSemaphore_t* extSem_out, const hipExternalSemaphoreHandleDesc* semHandleDesc);
typedef hipError_t (*t_hipInit)(unsigned int flags);
typedef hipError_t (*t_hipIpcCloseMemHandle)(void* devPtr);
typedef hipError_t (*t_hipIpcGetEventHandle)(hipIpcEventHandle_t* handle, hipEvent_t event);
typedef hipError_t (*t_hipIpcGetMemHandle)(hipIpcMemHandle_t* handle, void* devPtr);
typedef hipError_t (*t_hipIpcOpenEventHandle)(hipEvent_t* event, hipIpcEventHandle_t handle);
typedef hipError_t (*t_hipIpcOpenMemHandle)(void** devPtr, hipIpcMemHandle_t handle,
                                            unsigned int flags);
typedef const char* (*t_hipKernelNameRef)(const hipFunction_t f);
typedef const char* (*t_hipKernelNameRefByPtr)(const void* hostFunction, hipStream_t stream);
typedef hipError_t (*t_hipLaunchByPtr)(const void* func);
typedef hipError_t (*t_hipLaunchCooperativeKernel)(const void* f, dim3 gridDim, dim3 blockDimX,
                                                   void** kernelParams, unsigned int sharedMemBytes,
                                                   hipStream_t stream);
typedef hipError_t (*t_hipLaunchCooperativeKernelMultiDevice)(hipLaunchParams* launchParamsList,
                                                              int numDevices, unsigned int flags);
typedef hipError_t (*t_hipLaunchHostFunc)(hipStream_t stream, hipHostFn_t fn, void* userData);
typedef hipError_t (*t_hipLaunchKernel)(const void* function_address, dim3 numBlocks,
                                        dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                        hipStream_t stream);
typedef hipError_t (*t_hipMalloc)(void** ptr, size_t size);
typedef hipError_t (*t_hipMalloc3D)(hipPitchedPtr* pitchedDevPtr, hipExtent extent);
typedef hipError_t (*t_hipMalloc3DArray)(hipArray_t* array, const struct hipChannelFormatDesc* desc,
                                         struct hipExtent extent, unsigned int flags);
typedef hipError_t (*t_hipMallocArray)(hipArray_t* array, const hipChannelFormatDesc* desc,
                                       size_t width, size_t height, unsigned int flags);
typedef hipError_t (*t_hipMallocAsync)(void** dev_ptr, size_t size, hipStream_t stream);
typedef hipError_t (*t_hipMallocFromPoolAsync)(void** dev_ptr, size_t size, hipMemPool_t mem_pool,
                                               hipStream_t stream);
typedef hipError_t (*t_hipMallocHost)(void** ptr, size_t size);
typedef hipError_t (*t_hipMallocManaged)(void** dev_ptr, size_t size, unsigned int flags);
typedef hipError_t (*t_hipMallocMipmappedArray)(hipMipmappedArray_t* mipmappedArray,
                                                const struct hipChannelFormatDesc* desc,
                                                struct hipExtent extent, unsigned int numLevels,
                                                unsigned int flags);
typedef hipError_t (*t_hipMallocPitch)(void** ptr, size_t* pitch, size_t width, size_t height);
typedef hipError_t (*t_hipMemAddressFree)(void* devPtr, size_t size);
typedef hipError_t (*t_hipMemAddressReserve)(void** ptr, size_t size, size_t alignment, void* addr,
                                             unsigned long long flags);
typedef hipError_t (*t_hipMemAdvise)(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                                     int device);
typedef hipError_t (*t_hipMemAllocHost)(void** ptr, size_t size);
typedef hipError_t (*t_hipMemAllocPitch)(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes,
                                         size_t height, unsigned int elementSizeBytes);
typedef hipError_t (*t_hipMemCreate)(hipMemGenericAllocationHandle_t* handle, size_t size,
                                     const hipMemAllocationProp* prop, unsigned long long flags);
typedef hipError_t (*t_hipMemExportToShareableHandle)(void* shareableHandle,
                                                      hipMemGenericAllocationHandle_t handle,
                                                      hipMemAllocationHandleType handleType,
                                                      unsigned long long flags);
typedef hipError_t (*t_hipMemGetAccess)(unsigned long long* flags, const hipMemLocation* location,
                                        void* ptr);
typedef hipError_t (*t_hipMemGetAddressRange)(hipDeviceptr_t* pbase, size_t* psize,
                                              hipDeviceptr_t dptr);
typedef hipError_t (*t_hipMemGetAllocationGranularity)(size_t* granularity,
                                                       const hipMemAllocationProp* prop,
                                                       hipMemAllocationGranularity_flags option);
typedef hipError_t (*t_hipMemGetAllocationPropertiesFromHandle)(
    hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle);
typedef hipError_t (*t_hipMemGetInfo)(size_t* free, size_t* total);
typedef hipError_t (*t_hipMemImportFromShareableHandle)(hipMemGenericAllocationHandle_t* handle,
                                                        void* osHandle,
                                                        hipMemAllocationHandleType shHandleType);
typedef hipError_t (*t_hipMemMap)(void* ptr, size_t size, size_t offset,
                                  hipMemGenericAllocationHandle_t handle, unsigned long long flags);
typedef hipError_t (*t_hipMemMapArrayAsync)(hipArrayMapInfo* mapInfoList, unsigned int count,
                                            hipStream_t stream);
typedef hipError_t (*t_hipMemPoolCreate)(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props);
typedef hipError_t (*t_hipMemPoolDestroy)(hipMemPool_t mem_pool);
typedef hipError_t (*t_hipMemPoolExportPointer)(hipMemPoolPtrExportData* export_data,
                                                void* dev_ptr);
typedef hipError_t (*t_hipMemPoolExportToShareableHandle)(void* shared_handle,
                                                          hipMemPool_t mem_pool,
                                                          hipMemAllocationHandleType handle_type,
                                                          unsigned int flags);
typedef hipError_t (*t_hipMemPoolGetAccess)(hipMemAccessFlags* flags, hipMemPool_t mem_pool,
                                            hipMemLocation* location);
typedef hipError_t (*t_hipMemPoolGetAttribute)(hipMemPool_t mem_pool, hipMemPoolAttr attr,
                                               void* value);
typedef hipError_t (*t_hipMemPoolImportFromShareableHandle)(hipMemPool_t* mem_pool,
                                                            void* shared_handle,
                                                            hipMemAllocationHandleType handle_type,
                                                            unsigned int flags);
typedef hipError_t (*t_hipMemPoolImportPointer)(void** dev_ptr, hipMemPool_t mem_pool,
                                                hipMemPoolPtrExportData* export_data);
typedef hipError_t (*t_hipMemPoolSetAccess)(hipMemPool_t mem_pool,
                                            const hipMemAccessDesc* desc_list, size_t count);
typedef hipError_t (*t_hipMemPoolSetAttribute)(hipMemPool_t mem_pool, hipMemPoolAttr attr,
                                               void* value);
typedef hipError_t (*t_hipMemPoolTrimTo)(hipMemPool_t mem_pool, size_t min_bytes_to_hold);
typedef hipError_t (*t_hipMemPrefetchAsync)(const void* dev_ptr, size_t count, int device,
                                            hipStream_t stream);
typedef hipError_t (*t_hipMemPtrGetInfo)(void* ptr, size_t* size);
typedef hipError_t (*t_hipMemRangeGetAttribute)(void* data, size_t data_size,
                                                hipMemRangeAttribute attribute, const void* dev_ptr,
                                                size_t count);
typedef hipError_t (*t_hipMemRangeGetAttributes)(void** data, size_t* data_sizes,
                                                 hipMemRangeAttribute* attributes,
                                                 size_t num_attributes, const void* dev_ptr,
                                                 size_t count);
typedef hipError_t (*t_hipMemRelease)(hipMemGenericAllocationHandle_t handle);
typedef hipError_t (*t_hipMemRetainAllocationHandle)(hipMemGenericAllocationHandle_t* handle,
                                                     void* addr);
typedef hipError_t (*t_hipMemSetAccess)(void* ptr, size_t size, const hipMemAccessDesc* desc,
                                        size_t count);
typedef hipError_t (*t_hipMemUnmap)(void* ptr, size_t size);
typedef hipError_t (*t_hipMemcpy)(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
typedef hipError_t (*t_hipMemcpy2D)(void* dst, size_t dpitch, const void* src, size_t spitch,
                                    size_t width, size_t height, hipMemcpyKind kind);
typedef hipError_t (*t_hipMemcpy2DAsync)(void* dst, size_t dpitch, const void* src, size_t spitch,
                                         size_t width, size_t height, hipMemcpyKind kind,
                                         hipStream_t stream);
typedef hipError_t (*t_hipMemcpy2DFromArray)(void* dst, size_t dpitch, hipArray_const_t src,
                                             size_t wOffset, size_t hOffset, size_t width,
                                             size_t height, hipMemcpyKind kind);
typedef hipError_t (*t_hipMemcpy2DFromArrayAsync)(void* dst, size_t dpitch, hipArray_const_t src,
                                                  size_t wOffset, size_t hOffset, size_t width,
                                                  size_t height, hipMemcpyKind kind,
                                                  hipStream_t stream);
typedef hipError_t (*t_hipMemcpy2DToArray)(hipArray_t dst, size_t wOffset, size_t hOffset,
                                           const void* src, size_t spitch, size_t width,
                                           size_t height, hipMemcpyKind kind);
typedef hipError_t (*t_hipMemcpy2DToArrayAsync)(hipArray_t dst, size_t wOffset, size_t hOffset,
                                                const void* src, size_t spitch, size_t width,
                                                size_t height, hipMemcpyKind kind,
                                                hipStream_t stream);
typedef hipError_t (*t_hipMemcpy3D)(const struct hipMemcpy3DParms* p);
typedef hipError_t (*t_hipMemcpy3DAsync)(const struct hipMemcpy3DParms* p, hipStream_t stream);
typedef hipError_t (*t_hipMemcpyAsync)(void* dst, const void* src, size_t sizeBytes,
                                       hipMemcpyKind kind, hipStream_t stream);
typedef hipError_t (*t_hipMemcpyAtoH)(void* dst, hipArray_t srcArray, size_t srcOffset,
                                      size_t count);
typedef hipError_t (*t_hipMemcpyDtoD)(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes);
typedef hipError_t (*t_hipMemcpyDtoDAsync)(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                                           hipStream_t stream);
typedef hipError_t (*t_hipMemcpyDtoH)(void* dst, hipDeviceptr_t src, size_t sizeBytes);
typedef hipError_t (*t_hipMemcpyDtoHAsync)(void* dst, hipDeviceptr_t src, size_t sizeBytes,
                                           hipStream_t stream);
typedef hipError_t (*t_hipMemcpyFromArray)(void* dst, hipArray_const_t srcArray, size_t wOffset,
                                           size_t hOffset, size_t count, hipMemcpyKind kind);
typedef hipError_t (*t_hipMemcpyFromSymbol)(void* dst, const void* symbol, size_t sizeBytes,
                                            size_t offset, hipMemcpyKind kind);
typedef hipError_t (*t_hipMemcpyFromSymbolAsync)(void* dst, const void* symbol, size_t sizeBytes,
                                                 size_t offset, hipMemcpyKind kind,
                                                 hipStream_t stream);
typedef hipError_t (*t_hipMemcpyHtoA)(hipArray_t dstArray, size_t dstOffset, const void* srcHost,
                                      size_t count);
typedef hipError_t (*t_hipMemcpyHtoD)(hipDeviceptr_t dst, void* src, size_t sizeBytes);
typedef hipError_t (*t_hipMemcpyHtoDAsync)(hipDeviceptr_t dst, void* src, size_t sizeBytes,
                                           hipStream_t stream);
typedef hipError_t (*t_hipMemcpyParam2D)(const hip_Memcpy2D* pCopy);
typedef hipError_t (*t_hipMemcpyParam2DAsync)(const hip_Memcpy2D* pCopy, hipStream_t stream);
typedef hipError_t (*t_hipMemcpyPeer)(void* dst, int dstDeviceId, const void* src, int srcDeviceId,
                                      size_t sizeBytes);
typedef hipError_t (*t_hipMemcpyPeerAsync)(void* dst, int dstDeviceId, const void* src,
                                           int srcDevice, size_t sizeBytes, hipStream_t stream);
typedef hipError_t (*t_hipMemcpyToArray)(hipArray_t dst, size_t wOffset, size_t hOffset,
                                         const void* src, size_t count, hipMemcpyKind kind);
typedef hipError_t (*t_hipMemcpyToSymbol)(const void* symbol, const void* src, size_t sizeBytes,
                                          size_t offset, hipMemcpyKind kind);
typedef hipError_t (*t_hipMemcpyToSymbolAsync)(const void* symbol, const void* src,
                                               size_t sizeBytes, size_t offset, hipMemcpyKind kind,
                                               hipStream_t stream);
typedef hipError_t (*t_hipMemcpyWithStream)(void* dst, const void* src, size_t sizeBytes,
                                            hipMemcpyKind kind, hipStream_t stream);
typedef hipError_t (*t_hipMemset)(void* dst, int value, size_t sizeBytes);
typedef hipError_t (*t_hipMemset2D)(void* dst, size_t pitch, int value, size_t width,
                                    size_t height);
typedef hipError_t (*t_hipMemset2DAsync)(void* dst, size_t pitch, int value, size_t width,
                                         size_t height, hipStream_t stream);
typedef hipError_t (*t_hipMemset3D)(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent);
typedef hipError_t (*t_hipMemset3DAsync)(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                         hipStream_t stream);
typedef hipError_t (*t_hipMemsetAsync)(void* dst, int value, size_t sizeBytes, hipStream_t stream);
typedef hipError_t (*t_hipMemsetD16)(hipDeviceptr_t dest, unsigned short value, size_t count);
typedef hipError_t (*t_hipMemsetD16Async)(hipDeviceptr_t dest, unsigned short value, size_t count,
                                          hipStream_t stream);
typedef hipError_t (*t_hipMemsetD32)(hipDeviceptr_t dest, int value, size_t count);
typedef hipError_t (*t_hipMemsetD32Async)(hipDeviceptr_t dst, int value, size_t count,
                                          hipStream_t stream);
typedef hipError_t (*t_hipMemsetD8)(hipDeviceptr_t dest, unsigned char value, size_t count);
typedef hipError_t (*t_hipMemsetD8Async)(hipDeviceptr_t dest, unsigned char value, size_t count,
                                         hipStream_t stream);
typedef hipError_t (*t_hipMipmappedArrayCreate)(hipMipmappedArray_t* pHandle,
                                                HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                                unsigned int numMipmapLevels);
typedef hipError_t (*t_hipMipmappedArrayDestroy)(hipMipmappedArray_t hMipmappedArray);
typedef hipError_t (*t_hipMipmappedArrayGetLevel)(hipArray_t* pLevelArray,
                                                  hipMipmappedArray_t hMipMappedArray,
                                                  unsigned int level);
typedef hipError_t (*t_hipModuleGetFunction)(hipFunction_t* function, hipModule_t module,
                                             const char* kname);
typedef hipError_t (*t_hipModuleGetGlobal)(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                                           const char* name);
typedef hipError_t (*t_hipModuleGetTexRef)(textureReference** texRef, hipModule_t hmod,
                                           const char* name);
typedef hipError_t (*t_hipModuleLaunchCooperativeKernel)(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams);
typedef hipError_t (*t_hipModuleLaunchCooperativeKernelMultiDevice)(
    hipFunctionLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags);
typedef hipError_t (*t_hipModuleLaunchKernel)(hipFunction_t f, unsigned int gridDimX,
                                              unsigned int gridDimY, unsigned int gridDimZ,
                                              unsigned int blockDimX, unsigned int blockDimY,
                                              unsigned int blockDimZ, unsigned int sharedMemBytes,
                                              hipStream_t stream, void** kernelParams,
                                              void** extra);
typedef hipError_t (*t_hipModuleLoad)(hipModule_t* module, const char* fname);
typedef hipError_t (*t_hipModuleLoadData)(hipModule_t* module, const void* image);
typedef hipError_t (*t_hipModuleLoadDataEx)(hipModule_t* module, const void* image,
                                            unsigned int numOptions, hipJitOption* options,
                                            void** optionValues);
typedef hipError_t (*t_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor)(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk);
typedef hipError_t (*t_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags);
typedef hipError_t (*t_hipModuleOccupancyMaxPotentialBlockSize)(int* gridSize, int* blockSize,
                                                                hipFunction_t f,
                                                                size_t dynSharedMemPerBlk,
                                                                int blockSizeLimit);
typedef hipError_t (*t_hipModuleOccupancyMaxPotentialBlockSizeWithFlags)(
    int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit,
    unsigned int flags);
typedef hipError_t (*t_hipModuleUnload)(hipModule_t module);
typedef hipError_t (*t_hipOccupancyMaxActiveBlocksPerMultiprocessor)(int* numBlocks, const void* f,
                                                                     int blockSize,
                                                                     size_t dynSharedMemPerBlk);
typedef hipError_t (*t_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(
    int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags);
typedef hipError_t (*t_hipOccupancyMaxPotentialBlockSize)(int* gridSize, int* blockSize,
                                                          const void* f, size_t dynSharedMemPerBlk,
                                                          int blockSizeLimit);
typedef hipError_t (*t_hipPeekAtLastError)(void);
typedef hipError_t (*t_hipPointerGetAttribute)(void* data, hipPointer_attribute attribute,
                                               hipDeviceptr_t ptr);
typedef hipError_t (*t_hipPointerGetAttributes)(hipPointerAttribute_t* attributes, const void* ptr);
typedef hipError_t (*t_hipPointerSetAttribute)(const void* value, hipPointer_attribute attribute,
                                               hipDeviceptr_t ptr);
typedef hipError_t (*t_hipProfilerStart)();
typedef hipError_t (*t_hipProfilerStop)();
typedef hipError_t (*t_hipRuntimeGetVersion)(int* runtimeVersion);
typedef hipError_t (*t_hipSetDevice)(int deviceId);
typedef hipError_t (*t_hipSetDeviceFlags)(unsigned flags);
typedef hipError_t (*t_hipSetupArgument)(const void* arg, size_t size, size_t offset);
typedef hipError_t (*t_hipSignalExternalSemaphoresAsync)(
    const hipExternalSemaphore_t* extSemArray, const hipExternalSemaphoreSignalParams* paramsArray,
    unsigned int numExtSems, hipStream_t stream);
typedef hipError_t (*t_hipStreamAddCallback)(hipStream_t stream, hipStreamCallback_t callback,
                                             void* userData, unsigned int flags);
typedef hipError_t (*t_hipStreamAttachMemAsync)(hipStream_t stream, void* dev_ptr, size_t length,
                                                unsigned int flags);
typedef hipError_t (*t_hipStreamBeginCapture)(hipStream_t stream, hipStreamCaptureMode mode);
typedef hipError_t (*t_hipStreamCreate)(hipStream_t* stream);
typedef hipError_t (*t_hipStreamCreateWithFlags)(hipStream_t* stream, unsigned int flags);
typedef hipError_t (*t_hipStreamCreateWithPriority)(hipStream_t* stream, unsigned int flags,
                                                    int priority);
typedef hipError_t (*t_hipStreamDestroy)(hipStream_t stream);
typedef hipError_t (*t_hipStreamEndCapture)(hipStream_t stream, hipGraph_t* pGraph);
typedef hipError_t (*t_hipStreamGetCaptureInfo)(hipStream_t stream,
                                                hipStreamCaptureStatus* pCaptureStatus,
                                                unsigned long long* pId);
typedef hipError_t (*t_hipStreamGetCaptureInfo_v2)(
    hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out,
    hipGraph_t* graph_out, const hipGraphNode_t** dependencies_out, size_t* numDependencies_out);
typedef hipError_t (*t_hipStreamGetDevice)(hipStream_t stream, hipDevice_t* device);
typedef hipError_t (*t_hipStreamGetFlags)(hipStream_t stream, unsigned int* flags);
typedef hipError_t (*t_hipStreamGetPriority)(hipStream_t stream, int* priority);
typedef hipError_t (*t_hipStreamIsCapturing)(hipStream_t stream,
                                             hipStreamCaptureStatus* pCaptureStatus);
typedef hipError_t (*t_hipStreamQuery)(hipStream_t stream);
typedef hipError_t (*t_hipStreamSynchronize)(hipStream_t stream);
typedef hipError_t (*t_hipStreamUpdateCaptureDependencies)(hipStream_t stream,
                                                           hipGraphNode_t* dependencies,
                                                           size_t numDependencies,
                                                           unsigned int flags);
typedef hipError_t (*t_hipStreamWaitEvent)(hipStream_t stream, hipEvent_t event,
                                           unsigned int flags);
typedef hipError_t (*t_hipStreamWaitValue32)(hipStream_t stream, void* ptr, uint32_t value,
                                             unsigned int flags, uint32_t mask);
typedef hipError_t (*t_hipStreamWaitValue64)(hipStream_t stream, void* ptr, uint64_t value,
                                             unsigned int flags, uint64_t mask);
typedef hipError_t (*t_hipStreamWriteValue32)(hipStream_t stream, void* ptr, uint32_t value,
                                              unsigned int flags);
typedef hipError_t (*t_hipStreamWriteValue64)(hipStream_t stream, void* ptr, uint64_t value,
                                              unsigned int flags);
typedef hipError_t (*t_hipTexObjectCreate)(hipTextureObject_t* pTexObject,
                                           const HIP_RESOURCE_DESC* pResDesc,
                                           const HIP_TEXTURE_DESC* pTexDesc,
                                           const HIP_RESOURCE_VIEW_DESC* pResViewDesc);
typedef hipError_t (*t_hipTexObjectDestroy)(hipTextureObject_t texObject);
typedef hipError_t (*t_hipTexObjectGetResourceDesc)(HIP_RESOURCE_DESC* pResDesc,
                                                    hipTextureObject_t texObject);
typedef hipError_t (*t_hipTexObjectGetResourceViewDesc)(HIP_RESOURCE_VIEW_DESC* pResViewDesc,
                                                        hipTextureObject_t texObject);
typedef hipError_t (*t_hipTexObjectGetTextureDesc)(HIP_TEXTURE_DESC* pTexDesc,
                                                   hipTextureObject_t texObject);
typedef hipError_t (*t_hipTexRefGetAddress)(hipDeviceptr_t* dev_ptr,
                                            const textureReference* texRef);
typedef hipError_t (*t_hipTexRefGetAddressMode)(enum hipTextureAddressMode* pam,
                                                const textureReference* texRef, int dim);
typedef hipError_t (*t_hipTexRefGetFilterMode)(enum hipTextureFilterMode* pfm,
                                               const textureReference* texRef);
typedef hipError_t (*t_hipTexRefGetFlags)(unsigned int* pFlags, const textureReference* texRef);
typedef hipError_t (*t_hipTexRefGetFormat)(hipArray_Format* pFormat, int* pNumChannels,
                                           const textureReference* texRef);
typedef hipError_t (*t_hipTexRefGetMaxAnisotropy)(int* pmaxAnsio, const textureReference* texRef);
typedef hipError_t (*t_hipTexRefGetMipMappedArray)(hipMipmappedArray_t* pArray,
                                                   const textureReference* texRef);
typedef hipError_t (*t_hipTexRefGetMipmapFilterMode)(enum hipTextureFilterMode* pfm,
                                                     const textureReference* texRef);
typedef hipError_t (*t_hipTexRefGetMipmapLevelBias)(float* pbias, const textureReference* texRef);
typedef hipError_t (*t_hipTexRefGetMipmapLevelClamp)(float* pminMipmapLevelClamp,
                                                     float* pmaxMipmapLevelClamp,
                                                     const textureReference* texRef);
typedef hipError_t (*t_hipTexRefSetAddress)(size_t* ByteOffset, textureReference* texRef,
                                            hipDeviceptr_t dptr, size_t bytes);
typedef hipError_t (*t_hipTexRefSetAddress2D)(textureReference* texRef,
                                              const HIP_ARRAY_DESCRIPTOR* desc, hipDeviceptr_t dptr,
                                              size_t Pitch);
typedef hipError_t (*t_hipTexRefSetAddressMode)(textureReference* texRef, int dim,
                                                enum hipTextureAddressMode am);
typedef hipError_t (*t_hipTexRefSetArray)(textureReference* tex, hipArray_const_t array,
                                          unsigned int flags);
typedef hipError_t (*t_hipTexRefSetBorderColor)(textureReference* texRef, float* pBorderColor);
typedef hipError_t (*t_hipTexRefSetFilterMode)(textureReference* texRef,
                                               enum hipTextureFilterMode fm);
typedef hipError_t (*t_hipTexRefSetFlags)(textureReference* texRef, unsigned int Flags);
typedef hipError_t (*t_hipTexRefSetFormat)(textureReference* texRef, hipArray_Format fmt,
                                           int NumPackedComponents);
typedef hipError_t (*t_hipTexRefSetMaxAnisotropy)(textureReference* texRef, unsigned int maxAniso);
typedef hipError_t (*t_hipTexRefSetMipmapFilterMode)(textureReference* texRef,
                                                     enum hipTextureFilterMode fm);
typedef hipError_t (*t_hipTexRefSetMipmapLevelBias)(textureReference* texRef, float bias);
typedef hipError_t (*t_hipTexRefSetMipmapLevelClamp)(textureReference* texRef,
                                                     float minMipMapLevelClamp,
                                                     float maxMipMapLevelClamp);
typedef hipError_t (*t_hipTexRefSetMipmappedArray)(textureReference* texRef,
                                                   struct hipMipmappedArray* mipmappedArray,
                                                   unsigned int Flags);
typedef hipError_t (*t_hipThreadExchangeStreamCaptureMode)(hipStreamCaptureMode* mode);
typedef hipError_t (*t_hipUnbindTexture)(const textureReference* tex);
typedef hipError_t (*t_hipUserObjectCreate)(hipUserObject_t* object_out, void* ptr,
                                            hipHostFn_t destroy, unsigned int initialRefcount,
                                            unsigned int flags);
typedef hipError_t (*t_hipUserObjectRelease)(hipUserObject_t object, unsigned int count);
typedef hipError_t (*t_hipUserObjectRetain)(hipUserObject_t object, unsigned int count);
typedef hipError_t (*t_hipWaitExternalSemaphoresAsync)(
    const hipExternalSemaphore_t* extSemArray, const hipExternalSemaphoreWaitParams* paramsArray,
    unsigned int numExtSems, hipStream_t stream);

typedef hipError_t (*t_hipMemcpy_spt)(void* dst, const void* src, size_t sizeBytes,
                                      hipMemcpyKind kind);

typedef hipError_t (*t_hipMemcpyToSymbol_spt)(const void* symbol, const void* src, size_t sizeBytes,
                                              size_t offset, hipMemcpyKind kind);

typedef hipError_t (*t_hipMemcpyFromSymbol_spt)(void* dst, const void* symbol, size_t sizeBytes,
                                                size_t offset, hipMemcpyKind kind);

typedef hipError_t (*t_hipMemcpy2D_spt)(void* dst, size_t dpitch, const void* src, size_t spitch,
                                        size_t width, size_t height, hipMemcpyKind kind);

typedef hipError_t (*t_hipMemcpy2DFromArray_spt)(void* dst, size_t dpitch, hipArray_const_t src,
                                                 size_t wOffset, size_t hOffset, size_t width,
                                                 size_t height, hipMemcpyKind kind);

typedef hipError_t (*t_hipMemcpy3D_spt)(const struct hipMemcpy3DParms* p);

typedef hipError_t (*t_hipMemset_spt)(void* dst, int value, size_t sizeBytes);

typedef hipError_t (*t_hipMemsetAsync_spt)(void* dst, int value, size_t sizeBytes,
                                           hipStream_t stream);

typedef hipError_t (*t_hipMemset2D_spt)(void* dst, size_t pitch, int value, size_t width,
                                        size_t height);

typedef hipError_t (*t_hipMemset2DAsync_spt)(void* dst, size_t pitch, int value, size_t width,
                                             size_t height, hipStream_t stream);

typedef hipError_t (*t_hipMemset3DAsync_spt)(hipPitchedPtr pitchedDevPtr, int value,
                                             hipExtent extent, hipStream_t stream);

typedef hipError_t (*t_hipMemset3D_spt)(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent);

typedef hipError_t (*t_hipMemcpyAsync_spt)(void* dst, const void* src, size_t sizeBytes,
                                           hipMemcpyKind kind, hipStream_t stream);

typedef hipError_t (*t_hipMemcpy3DAsync_spt)(const hipMemcpy3DParms* p, hipStream_t stream);

typedef hipError_t (*t_hipMemcpy2DAsync_spt)(void* dst, size_t dpitch, const void* src,
                                             size_t spitch, size_t width, size_t height,
                                             hipMemcpyKind kind, hipStream_t stream);

typedef hipError_t (*t_hipMemcpyFromSymbolAsync_spt)(void* dst, const void* symbol,
                                                     size_t sizeBytes, size_t offset,
                                                     hipMemcpyKind kind, hipStream_t stream);

typedef hipError_t (*t_hipMemcpyToSymbolAsync_spt)(const void* symbol, const void* src,
                                                   size_t sizeBytes, size_t offset,
                                                   hipMemcpyKind kind, hipStream_t stream);

typedef hipError_t (*t_hipMemcpyFromArray_spt)(void* dst, hipArray_const_t src, size_t wOffsetSrc,
                                               size_t hOffset, size_t count, hipMemcpyKind kind);

typedef hipError_t (*t_hipMemcpy2DToArray_spt)(hipArray_t dst, size_t wOffset, size_t hOffset,
                                               const void* src, size_t spitch, size_t width,
                                               size_t height, hipMemcpyKind kind);

typedef hipError_t (*t_hipMemcpy2DFromArrayAsync_spt)(void* dst, size_t dpitch,
                                                      hipArray_const_t src, size_t wOffsetSrc,
                                                      size_t hOffsetSrc, size_t width,
                                                      size_t height, hipMemcpyKind kind,
                                                      hipStream_t stream);

typedef hipError_t (*t_hipMemcpy2DToArrayAsync_spt)(hipArray_t dst, size_t wOffset, size_t hOffset,
                                                    const void* src, size_t spitch, size_t width,
                                                    size_t height, hipMemcpyKind kind,
                                                    hipStream_t stream);

typedef hipError_t (*t_hipStreamQuery_spt)(hipStream_t stream);

typedef hipError_t (*t_hipStreamSynchronize_spt)(hipStream_t stream);

typedef hipError_t (*t_hipStreamGetPriority_spt)(hipStream_t stream, int* priority);

typedef hipError_t (*t_hipStreamWaitEvent_spt)(hipStream_t stream, hipEvent_t event,
                                               unsigned int flags);

typedef hipError_t (*t_hipStreamGetFlags_spt)(hipStream_t stream, unsigned int* flags);

typedef hipError_t (*t_hipStreamAddCallback_spt)(hipStream_t stream, hipStreamCallback_t callback,
                                                 void* userData, unsigned int flags);
typedef hipError_t (*t_hipEventRecord_spt)(hipEvent_t event, hipStream_t stream);
typedef hipError_t (*t_hipLaunchCooperativeKernel_spt)(const void* f, dim3 gridDim, dim3 blockDim,
                                                       void** kernelParams, uint32_t sharedMemBytes,
                                                       hipStream_t hStream);

typedef hipError_t (*t_hipLaunchKernel_spt)(const void* function_address, dim3 numBlocks,
                                            dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                            hipStream_t stream);

typedef hipError_t (*t_hipGraphLaunch_spt)(hipGraphExec_t graphExec, hipStream_t stream);
typedef hipError_t (*t_hipStreamBeginCapture_spt)(hipStream_t stream, hipStreamCaptureMode mode);
typedef hipError_t (*t_hipStreamEndCapture_spt)(hipStream_t stream, hipGraph_t* pGraph);
typedef hipError_t (*t_hipStreamIsCapturing_spt)(hipStream_t stream,
                                                 hipStreamCaptureStatus* pCaptureStatus);
typedef hipError_t (*t_hipStreamGetCaptureInfo_spt)(hipStream_t stream,
                                                    hipStreamCaptureStatus* pCaptureStatus,
                                                    unsigned long long* pId);
typedef hipError_t (*t_hipStreamGetCaptureInfo_v2_spt)(
    hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out,
    hipGraph_t* graph_out, const hipGraphNode_t** dependencies_out, size_t* numDependencies_out);
typedef hipError_t (*t_hipLaunchHostFunc_spt)(hipStream_t stream, hipHostFn_t fn, void* userData);
typedef hipChannelFormatDesc (*t_hipCreateChannelDesc)(int x, int y, int z, int w,
                                                       hipChannelFormatKind f);
typedef hipError_t (*t_hipExtModuleLaunchKernel)(hipFunction_t f, uint32_t globalWorkSizeX,
                                                 uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                                 uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                                 uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                                 hipStream_t hStream, void** kernelParams,
                                                 void** extra, hipEvent_t startEvent,
                                                 hipEvent_t stopEvent, uint32_t flags);
typedef hipError_t (*t_hipHccModuleLaunchKernel)(hipFunction_t f, uint32_t globalWorkSizeX,
                                                 uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                                 uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                                 uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                                 hipStream_t hStream, void** kernelParams,
                                                 void** extra, hipEvent_t startEvent,
                                                 hipEvent_t stopEvent);
typedef int (*t_hipGetStreamDeviceId)(hipStream_t stream);
typedef hipError_t (*t_hipDrvGraphAddMemsetNode)(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                 const hipGraphNode_t* dependencies, size_t numDependencies,
                                 const HIP_MEMSET_NODE_PARAMS* memsetParams, hipCtx_t ctx);
typedef hipError_t (*t_hipGraphAddExternalSemaphoresWaitNode)(hipGraphNode_t* pGraphNode,
                               hipGraph_t graph, const hipGraphNode_t* pDependencies,
                               size_t numDependencies,
                               const hipExternalSemaphoreWaitNodeParams* nodeParams);
typedef hipError_t (*t_hipGraphAddExternalSemaphoresSignalNode)(hipGraphNode_t* pGraphNode,
                               hipGraph_t graph, const hipGraphNode_t* pDependencies,
                               size_t numDependencies,
                               const hipExternalSemaphoreSignalNodeParams* nodeParams);
typedef hipError_t (*t_hipGraphExternalSemaphoresSignalNodeSetParams)(hipGraphNode_t hNode,
                                            const hipExternalSemaphoreSignalNodeParams* nodeParams);
typedef hipError_t (*t_hipGraphExternalSemaphoresWaitNodeSetParams)(hipGraphNode_t hNode,
                                            const hipExternalSemaphoreWaitNodeParams* nodeParams);
typedef hipError_t (*t_hipGraphExternalSemaphoresSignalNodeGetParams)(hipGraphNode_t hNode,
                                            hipExternalSemaphoreSignalNodeParams* params_out);
typedef hipError_t (*t_hipGraphExternalSemaphoresWaitNodeGetParams)(hipGraphNode_t hNode,
                                            hipExternalSemaphoreWaitNodeParams* params_out);
typedef hipError_t (*t_hipGraphExecExternalSemaphoresSignalNodeSetParams)(hipGraphExec_t hGraphExec,
                                            hipGraphNode_t hNode,
                                            const hipExternalSemaphoreSignalNodeParams* nodeParams);
typedef hipError_t (*t_hipGraphExecExternalSemaphoresWaitNodeSetParams)(hipGraphExec_t hGraphExec,
                                            hipGraphNode_t hNode,
                                            const hipExternalSemaphoreWaitNodeParams* nodeParams);
typedef hipError_t (*t_hipGraphAddNode)(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                           const hipGraphNode_t *pDependencies, size_t numDependencies,
                           hipGraphNodeParams *nodeParams);
typedef hipError_t (*t_hipGraphInstantiateWithParams)(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                                     hipGraphInstantiateParams* instantiateParams);
typedef hipError_t (*t_hipExtGetLastError)();
typedef hipError_t (*t_hipTexRefGetBorderColor)(float* pBorderColor,
                                                const textureReference* texRef);
typedef hipError_t (*t_hipTexRefGetArray)(hipArray_t* pArray, const textureReference* texRef);

typedef hipError_t (*t_hipTexRefGetBorderColor)(float* pBorderColor,
                                                const textureReference* texRef);
typedef hipError_t (*t_hipTexRefGetArray)(hipArray_t* pArray, const textureReference* texRef);
typedef hipError_t (*t_hipGetProcAddress)(const char* symbol, void** pfn, int  hipVersion, uint64_t flags,
                                          hipDriverProcAddressQueryResult* symbolStatus);
typedef hipError_t (*t_hipStreamBeginCaptureToGraph)(hipStream_t stream, hipGraph_t graph,
                                                     const hipGraphNode_t* dependencies,
                                                     const hipGraphEdgeData* dependencyData,
                                                     size_t numDependencies,
                                                     hipStreamCaptureMode mode);
typedef hipError_t (*t_hipGetFuncBySymbol)(hipFunction_t* functionPtr, const void* symbolPtr);
typedef hipError_t (*t_hipSetValidDevices)(int* device_arr, int len);
typedef hipError_t (*t_hipMemcpyAtoD)(hipDeviceptr_t dstDevice, hipArray_t srcArray,
                                      size_t srcOffset, size_t ByteCount);
typedef hipError_t (*t_hipMemcpyDtoA)(hipArray_t dstArray, size_t dstOffset,
                                      hipDeviceptr_t srcDevice, size_t ByteCount);
typedef hipError_t (*t_hipMemcpyAtoA)(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray,
                                      size_t srcOffset, size_t ByteCount);
typedef hipError_t (*t_hipMemcpyAtoHAsync)(void* dstHost, hipArray_t srcArray, size_t srcOffset,
                                           size_t ByteCount, hipStream_t stream);
typedef hipError_t (*t_hipMemcpyHtoAAsync)(hipArray_t dstArray, size_t dstOffset,
                                           const void* srcHost, size_t ByteCount,
                                           hipStream_t stream);
typedef hipError_t (*t_hipMemcpy2DArrayToArray)(hipArray_t dst, size_t wOffsetDst,
                                                size_t hOffsetDst, hipArray_const_t src,
                                                size_t wOffsetSrc, size_t hOffsetSrc, size_t width,
                                                size_t height, hipMemcpyKind kind);

// HIP Compiler dispatch table
struct HipCompilerDispatchTable {
  size_t size;
  t___hipPopCallConfiguration __hipPopCallConfiguration_fn;
  t___hipPushCallConfiguration __hipPushCallConfiguration_fn;
  t___hipRegisterFatBinary __hipRegisterFatBinary_fn;
  t___hipRegisterFunction __hipRegisterFunction_fn;
  t___hipRegisterManagedVar __hipRegisterManagedVar_fn;
  t___hipRegisterSurface __hipRegisterSurface_fn;
  t___hipRegisterTexture __hipRegisterTexture_fn;
  t___hipRegisterVar __hipRegisterVar_fn;
  t___hipUnregisterFatBinary __hipUnregisterFatBinary_fn;
};

// HIP API dispatch table
struct HipDispatchTable {
  size_t size;
  t_hipApiName hipApiName_fn;
  t_hipArray3DCreate hipArray3DCreate_fn;
  t_hipArray3DGetDescriptor hipArray3DGetDescriptor_fn;
  t_hipArrayCreate hipArrayCreate_fn;
  t_hipArrayDestroy hipArrayDestroy_fn;
  t_hipArrayGetDescriptor hipArrayGetDescriptor_fn;
  t_hipArrayGetInfo hipArrayGetInfo_fn;
  t_hipBindTexture hipBindTexture_fn;
  t_hipBindTexture2D hipBindTexture2D_fn;
  t_hipBindTextureToArray hipBindTextureToArray_fn;
  t_hipBindTextureToMipmappedArray hipBindTextureToMipmappedArray_fn;
  t_hipChooseDevice hipChooseDevice_fn;
  t_hipChooseDeviceR0000 hipChooseDeviceR0000_fn;
  t_hipConfigureCall hipConfigureCall_fn;
  t_hipCreateSurfaceObject hipCreateSurfaceObject_fn;
  t_hipCreateTextureObject hipCreateTextureObject_fn;
  t_hipCtxCreate hipCtxCreate_fn;
  t_hipCtxDestroy hipCtxDestroy_fn;
  t_hipCtxDisablePeerAccess hipCtxDisablePeerAccess_fn;
  t_hipCtxEnablePeerAccess hipCtxEnablePeerAccess_fn;
  t_hipCtxGetApiVersion hipCtxGetApiVersion_fn;
  t_hipCtxGetCacheConfig hipCtxGetCacheConfig_fn;
  t_hipCtxGetCurrent hipCtxGetCurrent_fn;
  t_hipCtxGetDevice hipCtxGetDevice_fn;
  t_hipCtxGetFlags hipCtxGetFlags_fn;
  t_hipCtxGetSharedMemConfig hipCtxGetSharedMemConfig_fn;
  t_hipCtxPopCurrent hipCtxPopCurrent_fn;
  t_hipCtxPushCurrent hipCtxPushCurrent_fn;
  t_hipCtxSetCacheConfig hipCtxSetCacheConfig_fn;
  t_hipCtxSetCurrent hipCtxSetCurrent_fn;
  t_hipCtxSetSharedMemConfig hipCtxSetSharedMemConfig_fn;
  t_hipCtxSynchronize hipCtxSynchronize_fn;
  t_hipDestroyExternalMemory hipDestroyExternalMemory_fn;
  t_hipDestroyExternalSemaphore hipDestroyExternalSemaphore_fn;
  t_hipDestroySurfaceObject hipDestroySurfaceObject_fn;
  t_hipDestroyTextureObject hipDestroyTextureObject_fn;
  t_hipDeviceCanAccessPeer hipDeviceCanAccessPeer_fn;
  t_hipDeviceComputeCapability hipDeviceComputeCapability_fn;
  t_hipDeviceDisablePeerAccess hipDeviceDisablePeerAccess_fn;
  t_hipDeviceEnablePeerAccess hipDeviceEnablePeerAccess_fn;
  t_hipDeviceGet hipDeviceGet_fn;
  t_hipDeviceGetAttribute hipDeviceGetAttribute_fn;
  t_hipDeviceGetByPCIBusId hipDeviceGetByPCIBusId_fn;
  t_hipDeviceGetCacheConfig hipDeviceGetCacheConfig_fn;
  t_hipDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool_fn;
  t_hipDeviceGetGraphMemAttribute hipDeviceGetGraphMemAttribute_fn;
  t_hipDeviceGetLimit hipDeviceGetLimit_fn;
  t_hipDeviceGetMemPool hipDeviceGetMemPool_fn;
  t_hipDeviceGetName hipDeviceGetName_fn;
  t_hipDeviceGetP2PAttribute hipDeviceGetP2PAttribute_fn;
  t_hipDeviceGetPCIBusId hipDeviceGetPCIBusId_fn;
  t_hipDeviceGetSharedMemConfig hipDeviceGetSharedMemConfig_fn;
  t_hipDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange_fn;
  t_hipDeviceGetUuid hipDeviceGetUuid_fn;
  t_hipDeviceGraphMemTrim hipDeviceGraphMemTrim_fn;
  t_hipDevicePrimaryCtxGetState hipDevicePrimaryCtxGetState_fn;
  t_hipDevicePrimaryCtxRelease hipDevicePrimaryCtxRelease_fn;
  t_hipDevicePrimaryCtxReset hipDevicePrimaryCtxReset_fn;
  t_hipDevicePrimaryCtxRetain hipDevicePrimaryCtxRetain_fn;
  t_hipDevicePrimaryCtxSetFlags hipDevicePrimaryCtxSetFlags_fn;
  t_hipDeviceReset hipDeviceReset_fn;
  t_hipDeviceSetCacheConfig hipDeviceSetCacheConfig_fn;
  t_hipDeviceSetGraphMemAttribute hipDeviceSetGraphMemAttribute_fn;
  t_hipDeviceSetLimit hipDeviceSetLimit_fn;
  t_hipDeviceSetMemPool hipDeviceSetMemPool_fn;
  t_hipDeviceSetSharedMemConfig hipDeviceSetSharedMemConfig_fn;
  t_hipDeviceSynchronize hipDeviceSynchronize_fn;
  t_hipDeviceTotalMem hipDeviceTotalMem_fn;
  t_hipDriverGetVersion hipDriverGetVersion_fn;
  t_hipDrvGetErrorName hipDrvGetErrorName_fn;
  t_hipDrvGetErrorString hipDrvGetErrorString_fn;
  t_hipDrvGraphAddMemcpyNode hipDrvGraphAddMemcpyNode_fn;
  t_hipDrvMemcpy2DUnaligned hipDrvMemcpy2DUnaligned_fn;
  t_hipDrvMemcpy3D hipDrvMemcpy3D_fn;
  t_hipDrvMemcpy3DAsync hipDrvMemcpy3DAsync_fn;
  t_hipDrvPointerGetAttributes hipDrvPointerGetAttributes_fn;
  t_hipEventCreate hipEventCreate_fn;
  t_hipEventCreateWithFlags hipEventCreateWithFlags_fn;
  t_hipEventDestroy hipEventDestroy_fn;
  t_hipEventElapsedTime hipEventElapsedTime_fn;
  t_hipEventQuery hipEventQuery_fn;
  t_hipEventRecord hipEventRecord_fn;
  t_hipEventSynchronize hipEventSynchronize_fn;
  t_hipExtGetLinkTypeAndHopCount hipExtGetLinkTypeAndHopCount_fn;
  t_hipExtLaunchKernel hipExtLaunchKernel_fn;
  t_hipExtLaunchMultiKernelMultiDevice hipExtLaunchMultiKernelMultiDevice_fn;
  t_hipExtMallocWithFlags hipExtMallocWithFlags_fn;
  t_hipExtStreamCreateWithCUMask hipExtStreamCreateWithCUMask_fn;
  t_hipExtStreamGetCUMask hipExtStreamGetCUMask_fn;
  t_hipExternalMemoryGetMappedBuffer hipExternalMemoryGetMappedBuffer_fn;
  t_hipFree hipFree_fn;
  t_hipFreeArray hipFreeArray_fn;
  t_hipFreeAsync hipFreeAsync_fn;
  t_hipFreeHost hipFreeHost_fn;
  t_hipFreeMipmappedArray hipFreeMipmappedArray_fn;
  t_hipFuncGetAttribute hipFuncGetAttribute_fn;
  t_hipFuncGetAttributes hipFuncGetAttributes_fn;
  t_hipFuncSetAttribute hipFuncSetAttribute_fn;
  t_hipFuncSetCacheConfig hipFuncSetCacheConfig_fn;
  t_hipFuncSetSharedMemConfig hipFuncSetSharedMemConfig_fn;
  t_hipGLGetDevices hipGLGetDevices_fn;
  t_hipGetChannelDesc hipGetChannelDesc_fn;
  t_hipGetDevice hipGetDevice_fn;
  t_hipGetDeviceCount hipGetDeviceCount_fn;
  t_hipGetDeviceFlags hipGetDeviceFlags_fn;
  t_hipGetDevicePropertiesR0600 hipGetDevicePropertiesR0600_fn;
  t_hipGetDevicePropertiesR0000 hipGetDevicePropertiesR0000_fn;
  t_hipGetErrorName hipGetErrorName_fn;
  t_hipGetErrorString hipGetErrorString_fn;
  t_hipGetLastError hipGetLastError_fn;
  t_hipGetMipmappedArrayLevel hipGetMipmappedArrayLevel_fn;
  t_hipGetSymbolAddress hipGetSymbolAddress_fn;
  t_hipGetSymbolSize hipGetSymbolSize_fn;
  t_hipGetTextureAlignmentOffset hipGetTextureAlignmentOffset_fn;
  t_hipGetTextureObjectResourceDesc hipGetTextureObjectResourceDesc_fn;
  t_hipGetTextureObjectResourceViewDesc hipGetTextureObjectResourceViewDesc_fn;
  t_hipGetTextureObjectTextureDesc hipGetTextureObjectTextureDesc_fn;
  t_hipGetTextureReference hipGetTextureReference_fn;
  t_hipGraphAddChildGraphNode hipGraphAddChildGraphNode_fn;
  t_hipGraphAddDependencies hipGraphAddDependencies_fn;
  t_hipGraphAddEmptyNode hipGraphAddEmptyNode_fn;
  t_hipGraphAddEventRecordNode hipGraphAddEventRecordNode_fn;
  t_hipGraphAddEventWaitNode hipGraphAddEventWaitNode_fn;
  t_hipGraphAddHostNode hipGraphAddHostNode_fn;
  t_hipGraphAddKernelNode hipGraphAddKernelNode_fn;
  t_hipGraphAddMemAllocNode hipGraphAddMemAllocNode_fn;
  t_hipGraphAddMemFreeNode hipGraphAddMemFreeNode_fn;
  t_hipGraphAddMemcpyNode hipGraphAddMemcpyNode_fn;
  t_hipGraphAddMemcpyNode1D hipGraphAddMemcpyNode1D_fn;
  t_hipGraphAddMemcpyNodeFromSymbol hipGraphAddMemcpyNodeFromSymbol_fn;
  t_hipGraphAddMemcpyNodeToSymbol hipGraphAddMemcpyNodeToSymbol_fn;
  t_hipGraphAddMemsetNode hipGraphAddMemsetNode_fn;
  t_hipGraphChildGraphNodeGetGraph hipGraphChildGraphNodeGetGraph_fn;
  t_hipGraphClone hipGraphClone_fn;
  t_hipGraphCreate hipGraphCreate_fn;
  t_hipGraphDebugDotPrint hipGraphDebugDotPrint_fn;
  t_hipGraphDestroy hipGraphDestroy_fn;
  t_hipGraphDestroyNode hipGraphDestroyNode_fn;
  t_hipGraphEventRecordNodeGetEvent hipGraphEventRecordNodeGetEvent_fn;
  t_hipGraphEventRecordNodeSetEvent hipGraphEventRecordNodeSetEvent_fn;
  t_hipGraphEventWaitNodeGetEvent hipGraphEventWaitNodeGetEvent_fn;
  t_hipGraphEventWaitNodeSetEvent hipGraphEventWaitNodeSetEvent_fn;
  t_hipGraphExecChildGraphNodeSetParams hipGraphExecChildGraphNodeSetParams_fn;
  t_hipGraphExecDestroy hipGraphExecDestroy_fn;
  t_hipGraphExecEventRecordNodeSetEvent hipGraphExecEventRecordNodeSetEvent_fn;
  t_hipGraphExecEventWaitNodeSetEvent hipGraphExecEventWaitNodeSetEvent_fn;
  t_hipGraphExecHostNodeSetParams hipGraphExecHostNodeSetParams_fn;
  t_hipGraphExecKernelNodeSetParams hipGraphExecKernelNodeSetParams_fn;
  t_hipGraphExecMemcpyNodeSetParams hipGraphExecMemcpyNodeSetParams_fn;
  t_hipGraphExecMemcpyNodeSetParams1D hipGraphExecMemcpyNodeSetParams1D_fn;
  t_hipGraphExecMemcpyNodeSetParamsFromSymbol hipGraphExecMemcpyNodeSetParamsFromSymbol_fn;
  t_hipGraphExecMemcpyNodeSetParamsToSymbol hipGraphExecMemcpyNodeSetParamsToSymbol_fn;
  t_hipGraphExecMemsetNodeSetParams hipGraphExecMemsetNodeSetParams_fn;
  t_hipGraphExecUpdate hipGraphExecUpdate_fn;
  t_hipGraphGetEdges hipGraphGetEdges_fn;
  t_hipGraphGetNodes hipGraphGetNodes_fn;
  t_hipGraphGetRootNodes hipGraphGetRootNodes_fn;
  t_hipGraphHostNodeGetParams hipGraphHostNodeGetParams_fn;
  t_hipGraphHostNodeSetParams hipGraphHostNodeSetParams_fn;
  t_hipGraphInstantiate hipGraphInstantiate_fn;
  t_hipGraphInstantiateWithFlags hipGraphInstantiateWithFlags_fn;
  t_hipGraphKernelNodeCopyAttributes hipGraphKernelNodeCopyAttributes_fn;
  t_hipGraphKernelNodeGetAttribute hipGraphKernelNodeGetAttribute_fn;
  t_hipGraphKernelNodeGetParams hipGraphKernelNodeGetParams_fn;
  t_hipGraphKernelNodeSetAttribute hipGraphKernelNodeSetAttribute_fn;
  t_hipGraphKernelNodeSetParams hipGraphKernelNodeSetParams_fn;
  t_hipGraphLaunch hipGraphLaunch_fn;
  t_hipGraphMemAllocNodeGetParams hipGraphMemAllocNodeGetParams_fn;
  t_hipGraphMemFreeNodeGetParams hipGraphMemFreeNodeGetParams_fn;
  t_hipGraphMemcpyNodeGetParams hipGraphMemcpyNodeGetParams_fn;
  t_hipGraphMemcpyNodeSetParams hipGraphMemcpyNodeSetParams_fn;
  t_hipGraphMemcpyNodeSetParams1D hipGraphMemcpyNodeSetParams1D_fn;
  t_hipGraphMemcpyNodeSetParamsFromSymbol hipGraphMemcpyNodeSetParamsFromSymbol_fn;
  t_hipGraphMemcpyNodeSetParamsToSymbol hipGraphMemcpyNodeSetParamsToSymbol_fn;
  t_hipGraphMemsetNodeGetParams hipGraphMemsetNodeGetParams_fn;
  t_hipGraphMemsetNodeSetParams hipGraphMemsetNodeSetParams_fn;
  t_hipGraphNodeFindInClone hipGraphNodeFindInClone_fn;
  t_hipGraphNodeGetDependencies hipGraphNodeGetDependencies_fn;
  t_hipGraphNodeGetDependentNodes hipGraphNodeGetDependentNodes_fn;
  t_hipGraphNodeGetEnabled hipGraphNodeGetEnabled_fn;
  t_hipGraphNodeGetType hipGraphNodeGetType_fn;
  t_hipGraphNodeSetEnabled hipGraphNodeSetEnabled_fn;
  t_hipGraphReleaseUserObject hipGraphReleaseUserObject_fn;
  t_hipGraphRemoveDependencies hipGraphRemoveDependencies_fn;
  t_hipGraphRetainUserObject hipGraphRetainUserObject_fn;
  t_hipGraphUpload hipGraphUpload_fn;
  t_hipGraphicsGLRegisterBuffer hipGraphicsGLRegisterBuffer_fn;
  t_hipGraphicsGLRegisterImage hipGraphicsGLRegisterImage_fn;
  t_hipGraphicsMapResources hipGraphicsMapResources_fn;
  t_hipGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer_fn;
  t_hipGraphicsSubResourceGetMappedArray hipGraphicsSubResourceGetMappedArray_fn;
  t_hipGraphicsUnmapResources hipGraphicsUnmapResources_fn;
  t_hipGraphicsUnregisterResource hipGraphicsUnregisterResource_fn;
  t_hipHostAlloc hipHostAlloc_fn;
  t_hipHostFree hipHostFree_fn;
  t_hipHostGetDevicePointer hipHostGetDevicePointer_fn;
  t_hipHostGetFlags hipHostGetFlags_fn;
  t_hipHostMalloc hipHostMalloc_fn;
  t_hipHostRegister hipHostRegister_fn;
  t_hipHostUnregister hipHostUnregister_fn;
  t_hipImportExternalMemory hipImportExternalMemory_fn;
  t_hipImportExternalSemaphore hipImportExternalSemaphore_fn;
  t_hipInit hipInit_fn;
  t_hipIpcCloseMemHandle hipIpcCloseMemHandle_fn;
  t_hipIpcGetEventHandle hipIpcGetEventHandle_fn;
  t_hipIpcGetMemHandle hipIpcGetMemHandle_fn;
  t_hipIpcOpenEventHandle hipIpcOpenEventHandle_fn;
  t_hipIpcOpenMemHandle hipIpcOpenMemHandle_fn;
  t_hipKernelNameRef hipKernelNameRef_fn;
  t_hipKernelNameRefByPtr hipKernelNameRefByPtr_fn;
  t_hipLaunchByPtr hipLaunchByPtr_fn;
  t_hipLaunchCooperativeKernel hipLaunchCooperativeKernel_fn;
  t_hipLaunchCooperativeKernelMultiDevice hipLaunchCooperativeKernelMultiDevice_fn;
  t_hipLaunchHostFunc hipLaunchHostFunc_fn;
  t_hipLaunchKernel hipLaunchKernel_fn;
  t_hipMalloc hipMalloc_fn;
  t_hipMalloc3D hipMalloc3D_fn;
  t_hipMalloc3DArray hipMalloc3DArray_fn;
  t_hipMallocArray hipMallocArray_fn;
  t_hipMallocAsync hipMallocAsync_fn;
  t_hipMallocFromPoolAsync hipMallocFromPoolAsync_fn;
  t_hipMallocHost hipMallocHost_fn;
  t_hipMallocManaged hipMallocManaged_fn;
  t_hipMallocMipmappedArray hipMallocMipmappedArray_fn;
  t_hipMallocPitch hipMallocPitch_fn;
  t_hipMemAddressFree hipMemAddressFree_fn;
  t_hipMemAddressReserve hipMemAddressReserve_fn;
  t_hipMemAdvise hipMemAdvise_fn;
  t_hipMemAllocHost hipMemAllocHost_fn;
  t_hipMemAllocPitch hipMemAllocPitch_fn;
  t_hipMemCreate hipMemCreate_fn;
  t_hipMemExportToShareableHandle hipMemExportToShareableHandle_fn;
  t_hipMemGetAccess hipMemGetAccess_fn;
  t_hipMemGetAddressRange hipMemGetAddressRange_fn;
  t_hipMemGetAllocationGranularity hipMemGetAllocationGranularity_fn;
  t_hipMemGetAllocationPropertiesFromHandle hipMemGetAllocationPropertiesFromHandle_fn;
  t_hipMemGetInfo hipMemGetInfo_fn;
  t_hipMemImportFromShareableHandle hipMemImportFromShareableHandle_fn;
  t_hipMemMap hipMemMap_fn;
  t_hipMemMapArrayAsync hipMemMapArrayAsync_fn;
  t_hipMemPoolCreate hipMemPoolCreate_fn;
  t_hipMemPoolDestroy hipMemPoolDestroy_fn;
  t_hipMemPoolExportPointer hipMemPoolExportPointer_fn;
  t_hipMemPoolExportToShareableHandle hipMemPoolExportToShareableHandle_fn;
  t_hipMemPoolGetAccess hipMemPoolGetAccess_fn;
  t_hipMemPoolGetAttribute hipMemPoolGetAttribute_fn;
  t_hipMemPoolImportFromShareableHandle hipMemPoolImportFromShareableHandle_fn;
  t_hipMemPoolImportPointer hipMemPoolImportPointer_fn;
  t_hipMemPoolSetAccess hipMemPoolSetAccess_fn;
  t_hipMemPoolSetAttribute hipMemPoolSetAttribute_fn;
  t_hipMemPoolTrimTo hipMemPoolTrimTo_fn;
  t_hipMemPrefetchAsync hipMemPrefetchAsync_fn;
  t_hipMemPtrGetInfo hipMemPtrGetInfo_fn;
  t_hipMemRangeGetAttribute hipMemRangeGetAttribute_fn;
  t_hipMemRangeGetAttributes hipMemRangeGetAttributes_fn;
  t_hipMemRelease hipMemRelease_fn;
  t_hipMemRetainAllocationHandle hipMemRetainAllocationHandle_fn;
  t_hipMemSetAccess hipMemSetAccess_fn;
  t_hipMemUnmap hipMemUnmap_fn;
  t_hipMemcpy hipMemcpy_fn;
  t_hipMemcpy2D hipMemcpy2D_fn;
  t_hipMemcpy2DAsync hipMemcpy2DAsync_fn;
  t_hipMemcpy2DFromArray hipMemcpy2DFromArray_fn;
  t_hipMemcpy2DFromArrayAsync hipMemcpy2DFromArrayAsync_fn;
  t_hipMemcpy2DToArray hipMemcpy2DToArray_fn;
  t_hipMemcpy2DToArrayAsync hipMemcpy2DToArrayAsync_fn;
  t_hipMemcpy3D hipMemcpy3D_fn;
  t_hipMemcpy3DAsync hipMemcpy3DAsync_fn;
  t_hipMemcpyAsync hipMemcpyAsync_fn;
  t_hipMemcpyAtoH hipMemcpyAtoH_fn;
  t_hipMemcpyDtoD hipMemcpyDtoD_fn;
  t_hipMemcpyDtoDAsync hipMemcpyDtoDAsync_fn;
  t_hipMemcpyDtoH hipMemcpyDtoH_fn;
  t_hipMemcpyDtoHAsync hipMemcpyDtoHAsync_fn;
  t_hipMemcpyFromArray hipMemcpyFromArray_fn;
  t_hipMemcpyFromSymbol hipMemcpyFromSymbol_fn;
  t_hipMemcpyFromSymbolAsync hipMemcpyFromSymbolAsync_fn;
  t_hipMemcpyHtoA hipMemcpyHtoA_fn;
  t_hipMemcpyHtoD hipMemcpyHtoD_fn;
  t_hipMemcpyHtoDAsync hipMemcpyHtoDAsync_fn;
  t_hipMemcpyParam2D hipMemcpyParam2D_fn;
  t_hipMemcpyParam2DAsync hipMemcpyParam2DAsync_fn;
  t_hipMemcpyPeer hipMemcpyPeer_fn;
  t_hipMemcpyPeerAsync hipMemcpyPeerAsync_fn;
  t_hipMemcpyToArray hipMemcpyToArray_fn;
  t_hipMemcpyToSymbol hipMemcpyToSymbol_fn;
  t_hipMemcpyToSymbolAsync hipMemcpyToSymbolAsync_fn;
  t_hipMemcpyWithStream hipMemcpyWithStream_fn;
  t_hipMemset hipMemset_fn;
  t_hipMemset2D hipMemset2D_fn;
  t_hipMemset2DAsync hipMemset2DAsync_fn;
  t_hipMemset3D hipMemset3D_fn;
  t_hipMemset3DAsync hipMemset3DAsync_fn;
  t_hipMemsetAsync hipMemsetAsync_fn;
  t_hipMemsetD16 hipMemsetD16_fn;
  t_hipMemsetD16Async hipMemsetD16Async_fn;
  t_hipMemsetD32 hipMemsetD32_fn;
  t_hipMemsetD32Async hipMemsetD32Async_fn;
  t_hipMemsetD8 hipMemsetD8_fn;
  t_hipMemsetD8Async hipMemsetD8Async_fn;
  t_hipMipmappedArrayCreate hipMipmappedArrayCreate_fn;
  t_hipMipmappedArrayDestroy hipMipmappedArrayDestroy_fn;
  t_hipMipmappedArrayGetLevel hipMipmappedArrayGetLevel_fn;
  t_hipModuleGetFunction hipModuleGetFunction_fn;
  t_hipModuleGetGlobal hipModuleGetGlobal_fn;
  t_hipModuleGetTexRef hipModuleGetTexRef_fn;
  t_hipModuleLaunchCooperativeKernel hipModuleLaunchCooperativeKernel_fn;
  t_hipModuleLaunchCooperativeKernelMultiDevice hipModuleLaunchCooperativeKernelMultiDevice_fn;
  t_hipModuleLaunchKernel hipModuleLaunchKernel_fn;
  t_hipModuleLoad hipModuleLoad_fn;
  t_hipModuleLoadData hipModuleLoadData_fn;
  t_hipModuleLoadDataEx hipModuleLoadDataEx_fn;
  t_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
      hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_fn;
  t_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
      hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn;
  t_hipModuleOccupancyMaxPotentialBlockSize hipModuleOccupancyMaxPotentialBlockSize_fn;
  t_hipModuleOccupancyMaxPotentialBlockSizeWithFlags
      hipModuleOccupancyMaxPotentialBlockSizeWithFlags_fn;
  t_hipModuleUnload hipModuleUnload_fn;
  t_hipOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor_fn;
  t_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
      hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn;
  t_hipOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize_fn;
  t_hipPeekAtLastError hipPeekAtLastError_fn;
  t_hipPointerGetAttribute hipPointerGetAttribute_fn;
  t_hipPointerGetAttributes hipPointerGetAttributes_fn;
  t_hipPointerSetAttribute hipPointerSetAttribute_fn;
  t_hipProfilerStart hipProfilerStart_fn;
  t_hipProfilerStop hipProfilerStop_fn;
  t_hipRuntimeGetVersion hipRuntimeGetVersion_fn;
  t_hipSetDevice hipSetDevice_fn;
  t_hipSetDeviceFlags hipSetDeviceFlags_fn;
  t_hipSetupArgument hipSetupArgument_fn;
  t_hipSignalExternalSemaphoresAsync hipSignalExternalSemaphoresAsync_fn;
  t_hipStreamAddCallback hipStreamAddCallback_fn;
  t_hipStreamAttachMemAsync hipStreamAttachMemAsync_fn;
  t_hipStreamBeginCapture hipStreamBeginCapture_fn;
  t_hipStreamCreate hipStreamCreate_fn;
  t_hipStreamCreateWithFlags hipStreamCreateWithFlags_fn;
  t_hipStreamCreateWithPriority hipStreamCreateWithPriority_fn;
  t_hipStreamDestroy hipStreamDestroy_fn;
  t_hipStreamEndCapture hipStreamEndCapture_fn;
  t_hipStreamGetCaptureInfo hipStreamGetCaptureInfo_fn;
  t_hipStreamGetCaptureInfo_v2 hipStreamGetCaptureInfo_v2_fn;
  t_hipStreamGetDevice hipStreamGetDevice_fn;
  t_hipStreamGetFlags hipStreamGetFlags_fn;
  t_hipStreamGetPriority hipStreamGetPriority_fn;
  t_hipStreamIsCapturing hipStreamIsCapturing_fn;
  t_hipStreamQuery hipStreamQuery_fn;
  t_hipStreamSynchronize hipStreamSynchronize_fn;
  t_hipStreamUpdateCaptureDependencies hipStreamUpdateCaptureDependencies_fn;
  t_hipStreamWaitEvent hipStreamWaitEvent_fn;
  t_hipStreamWaitValue32 hipStreamWaitValue32_fn;
  t_hipStreamWaitValue64 hipStreamWaitValue64_fn;
  t_hipStreamWriteValue32 hipStreamWriteValue32_fn;
  t_hipStreamWriteValue64 hipStreamWriteValue64_fn;
  t_hipTexObjectCreate hipTexObjectCreate_fn;
  t_hipTexObjectDestroy hipTexObjectDestroy_fn;
  t_hipTexObjectGetResourceDesc hipTexObjectGetResourceDesc_fn;
  t_hipTexObjectGetResourceViewDesc hipTexObjectGetResourceViewDesc_fn;
  t_hipTexObjectGetTextureDesc hipTexObjectGetTextureDesc_fn;
  t_hipTexRefGetAddress hipTexRefGetAddress_fn;
  t_hipTexRefGetAddressMode hipTexRefGetAddressMode_fn;
  t_hipTexRefGetFilterMode hipTexRefGetFilterMode_fn;
  t_hipTexRefGetFlags hipTexRefGetFlags_fn;
  t_hipTexRefGetFormat hipTexRefGetFormat_fn;
  t_hipTexRefGetMaxAnisotropy hipTexRefGetMaxAnisotropy_fn;
  t_hipTexRefGetMipMappedArray hipTexRefGetMipMappedArray_fn;
  t_hipTexRefGetMipmapFilterMode hipTexRefGetMipmapFilterMode_fn;
  t_hipTexRefGetMipmapLevelBias hipTexRefGetMipmapLevelBias_fn;
  t_hipTexRefGetMipmapLevelClamp hipTexRefGetMipmapLevelClamp_fn;
  t_hipTexRefSetAddress hipTexRefSetAddress_fn;
  t_hipTexRefSetAddress2D hipTexRefSetAddress2D_fn;
  t_hipTexRefSetAddressMode hipTexRefSetAddressMode_fn;
  t_hipTexRefSetArray hipTexRefSetArray_fn;
  t_hipTexRefSetBorderColor hipTexRefSetBorderColor_fn;
  t_hipTexRefSetFilterMode hipTexRefSetFilterMode_fn;
  t_hipTexRefSetFlags hipTexRefSetFlags_fn;
  t_hipTexRefSetFormat hipTexRefSetFormat_fn;
  t_hipTexRefSetMaxAnisotropy hipTexRefSetMaxAnisotropy_fn;
  t_hipTexRefSetMipmapFilterMode hipTexRefSetMipmapFilterMode_fn;
  t_hipTexRefSetMipmapLevelBias hipTexRefSetMipmapLevelBias_fn;
  t_hipTexRefSetMipmapLevelClamp hipTexRefSetMipmapLevelClamp_fn;
  t_hipTexRefSetMipmappedArray hipTexRefSetMipmappedArray_fn;
  t_hipThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode_fn;
  t_hipUnbindTexture hipUnbindTexture_fn;
  t_hipUserObjectCreate hipUserObjectCreate_fn;
  t_hipUserObjectRelease hipUserObjectRelease_fn;
  t_hipUserObjectRetain hipUserObjectRetain_fn;
  t_hipWaitExternalSemaphoresAsync hipWaitExternalSemaphoresAsync_fn;
  t_hipCreateChannelDesc hipCreateChannelDesc_fn;
  t_hipExtModuleLaunchKernel hipExtModuleLaunchKernel_fn;
  t_hipHccModuleLaunchKernel hipHccModuleLaunchKernel_fn;
  t_hipMemcpy_spt hipMemcpy_spt_fn;
  t_hipMemcpyToSymbol_spt hipMemcpyToSymbol_spt_fn;
  t_hipMemcpyFromSymbol_spt hipMemcpyFromSymbol_spt_fn;
  t_hipMemcpy2D_spt hipMemcpy2D_spt_fn;
  t_hipMemcpy2DFromArray_spt hipMemcpy2DFromArray_spt_fn;
  t_hipMemcpy3D_spt hipMemcpy3D_spt_fn;
  t_hipMemset_spt hipMemset_spt_fn;
  t_hipMemsetAsync_spt hipMemsetAsync_spt_fn;
  t_hipMemset2D_spt hipMemset2D_spt_fn;
  t_hipMemset2DAsync_spt hipMemset2DAsync_spt_fn;
  t_hipMemset3DAsync_spt hipMemset3DAsync_spt_fn;
  t_hipMemset3D_spt hipMemset3D_spt_fn;
  t_hipMemcpyAsync_spt hipMemcpyAsync_spt_fn;
  t_hipMemcpy3DAsync_spt hipMemcpy3DAsync_spt_fn;
  t_hipMemcpy2DAsync_spt hipMemcpy2DAsync_spt_fn;
  t_hipMemcpyFromSymbolAsync_spt hipMemcpyFromSymbolAsync_spt_fn;
  t_hipMemcpyToSymbolAsync_spt hipMemcpyToSymbolAsync_spt_fn;
  t_hipMemcpyFromArray_spt hipMemcpyFromArray_spt_fn;
  t_hipMemcpy2DToArray_spt hipMemcpy2DToArray_spt_fn;
  t_hipMemcpy2DFromArrayAsync_spt hipMemcpy2DFromArrayAsync_spt_fn;
  t_hipMemcpy2DToArrayAsync_spt hipMemcpy2DToArrayAsync_spt_fn;
  t_hipStreamQuery_spt hipStreamQuery_spt_fn;
  t_hipStreamSynchronize_spt hipStreamSynchronize_spt_fn;
  t_hipStreamGetPriority_spt hipStreamGetPriority_spt_fn;
  t_hipStreamWaitEvent_spt hipStreamWaitEvent_spt_fn;
  t_hipStreamGetFlags_spt hipStreamGetFlags_spt_fn;
  t_hipStreamAddCallback_spt hipStreamAddCallback_spt_fn;
  t_hipEventRecord_spt hipEventRecord_spt_fn;
  t_hipLaunchCooperativeKernel_spt hipLaunchCooperativeKernel_spt_fn;
  t_hipLaunchKernel_spt hipLaunchKernel_spt_fn;
  t_hipGraphLaunch_spt hipGraphLaunch_spt_fn;
  t_hipStreamBeginCapture_spt hipStreamBeginCapture_spt_fn;
  t_hipStreamEndCapture_spt hipStreamEndCapture_spt_fn;
  t_hipStreamIsCapturing_spt hipStreamIsCapturing_spt_fn;
  t_hipStreamGetCaptureInfo_spt hipStreamGetCaptureInfo_spt_fn;
  t_hipStreamGetCaptureInfo_v2_spt hipStreamGetCaptureInfo_v2_spt_fn;
  t_hipLaunchHostFunc_spt hipLaunchHostFunc_spt_fn;
  t_hipGetStreamDeviceId hipGetStreamDeviceId_fn;
  t_hipDrvGraphAddMemsetNode hipDrvGraphAddMemsetNode_fn;
  t_hipGraphAddExternalSemaphoresWaitNode hipGraphAddExternalSemaphoresWaitNode_fn;
  t_hipGraphAddExternalSemaphoresSignalNode hipGraphAddExternalSemaphoresSignalNode_fn;
  t_hipGraphExternalSemaphoresSignalNodeSetParams hipGraphExternalSemaphoresSignalNodeSetParams_fn;
  t_hipGraphExternalSemaphoresWaitNodeSetParams hipGraphExternalSemaphoresWaitNodeSetParams_fn;
  t_hipGraphExternalSemaphoresSignalNodeGetParams hipGraphExternalSemaphoresSignalNodeGetParams_fn;
  t_hipGraphExternalSemaphoresWaitNodeGetParams hipGraphExternalSemaphoresWaitNodeGetParams_fn;
  t_hipGraphExecExternalSemaphoresSignalNodeSetParams hipGraphExecExternalSemaphoresSignalNodeSetParams_fn;
  t_hipGraphExecExternalSemaphoresWaitNodeSetParams hipGraphExecExternalSemaphoresWaitNodeSetParams_fn;
  t_hipGraphAddNode hipGraphAddNode_fn;
  t_hipGraphInstantiateWithParams hipGraphInstantiateWithParams_fn;
  t_hipExtGetLastError hipExtGetLastError_fn;
  t_hipTexRefGetBorderColor hipTexRefGetBorderColor_fn;
  t_hipTexRefGetArray hipTexRefGetArray_fn;
  t_hipGetProcAddress hipGetProcAddress_fn;
  t_hipStreamBeginCaptureToGraph hipStreamBeginCaptureToGraph_fn;
  t_hipGetFuncBySymbol hipGetFuncBySymbol_fn;
  t_hipSetValidDevices hipSetValidDevices_fn;
  t_hipMemcpyAtoD hipMemcpyAtoD_fn;
  t_hipMemcpyDtoA hipMemcpyDtoA_fn;
  t_hipMemcpyAtoA hipMemcpyAtoA_fn;
  t_hipMemcpyAtoHAsync hipMemcpyAtoHAsync_fn;
  t_hipMemcpyHtoAAsync hipMemcpyHtoAAsync_fn;
  t_hipMemcpy2DArrayToArray hipMemcpy2DArrayToArray_fn;
};
