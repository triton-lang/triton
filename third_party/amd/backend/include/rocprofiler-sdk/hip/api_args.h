// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_version.h>
// must be included after
#include <hip/hip_deprecated.h>

#include <hip/amd_detail/amd_hip_gl_interop.h>
#include <hip/amd_detail/hip_api_trace.hpp>

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

// Empty struct has a size of 0 in C but size of 1 in C++.
// This struct is added to the union members which represent
// functions with no arguments to ensure ABI compatibility
typedef struct rocprofiler_hip_api_no_args {
  char empty;
} rocprofiler_hip_api_no_args;

typedef union rocprofiler_hip_api_retval_t {
#ifdef __cplusplus
  rocprofiler_hip_api_retval_t() = default;
  ~rocprofiler_hip_api_retval_t() = default;
#endif

  uint64_t uint64_t_retval;
  int int_retval;
  const char *const_charp_retval;
  hipError_t hipError_t_retval;
  hipChannelFormatDesc hipChannelFormatDesc_retval;
  void **voidpp_retval;
} rocprofiler_hip_api_retval_t;

// NOTE: dim3 value arguments replaced with rocprofiler_dim3_t because dim3 has
// a non-trivial destructor
typedef union rocprofiler_hip_api_args_t {
#ifdef __cplusplus
  rocprofiler_hip_api_args_t() = default;
  ~rocprofiler_hip_api_args_t() = default;
#endif

  // compiler
  struct {
    dim3 *gridDim;
    dim3 *blockDim;
    size_t *sharedMem;
    hipStream_t *stream;
  } __hipPopCallConfiguration;
  struct {
    rocprofiler_dim3_t gridDim;
    rocprofiler_dim3_t blockDim;
    size_t sharedMem;
    hipStream_t stream;
  } __hipPushCallConfiguration;
  struct {
    const void *data;
  } __hipRegisterFatBinary;
  struct {
    void **modules;
    const void *hostFunction;
    char *deviceFunction;
    const char *deviceName;
    unsigned int threadLimit;
    uint3 *tid;
    uint3 *bid;
    dim3 *blockDim;
    dim3 *gridDim;
    int *wSize;
  } __hipRegisterFunction;
  struct {
    void *hipModule;
    void **pointer;
    void *init_value;
    const char *name;
    size_t size;
    unsigned align;
  } __hipRegisterManagedVar;
  struct {
    void **modules;
    void *var;
    char *hostVar;
    char *deviceVar;
    int type;
    int ext;
  } __hipRegisterSurface;
  struct {
    void **modules;
    void *var;
    char *hostVar;
    char *deviceVar;
    int type;
    int norm;
    int ext;
  } __hipRegisterTexture;
  struct {
    void **modules;
    void *var;
    char *hostVar;
    char *deviceVar;
    int ext;
    size_t size;
    int constant;
    int global;
  } __hipRegisterVar;
  struct {
    void **modules;
  } __hipUnregisterFatBinary;
  // runtime
  struct {
    uint32_t id;
  } hipApiName;
  struct {
    hipArray_t *array;
    const HIP_ARRAY3D_DESCRIPTOR *pAllocateArray;
  } hipArray3DCreate;
  struct {
    HIP_ARRAY3D_DESCRIPTOR *pArrayDescriptor;
    hipArray_t array;
  } hipArray3DGetDescriptor;
  struct {
    hipArray_t *pHandle;
    const HIP_ARRAY_DESCRIPTOR *pAllocateArray;
  } hipArrayCreate;
  struct {
    hipArray_t array;
  } hipArrayDestroy;
  struct {
    HIP_ARRAY_DESCRIPTOR *pArrayDescriptor;
    hipArray_t array;
  } hipArrayGetDescriptor;
  struct {
    hipChannelFormatDesc *desc;
    hipExtent *extent;
    unsigned int *flags;
    hipArray_t array;
  } hipArrayGetInfo;
  struct {
    size_t *offset;
    const textureReference *tex;
    const void *devPtr;
    const hipChannelFormatDesc *desc;
    size_t size;
  } hipBindTexture;
  struct {
    size_t *offset;
    const textureReference *tex;
    const void *devPtr;
    const hipChannelFormatDesc *desc;
    size_t width;
    size_t height;
    size_t pitch;
  } hipBindTexture2D;
  struct {
    const textureReference *tex;
    hipArray_const_t array;
    const hipChannelFormatDesc *desc;
  } hipBindTextureToArray;
  struct {
    const textureReference *tex;
    hipMipmappedArray_const_t mipmappedArray;
    const hipChannelFormatDesc *desc;
  } hipBindTextureToMipmappedArray;
  struct {
    int *device;
    const hipDeviceProp_tR0600 *prop;
  } hipChooseDevice;
  struct {
    int *device;
    const hipDeviceProp_tR0000 *prop;
  } hipChooseDeviceR0000;
  struct {
    rocprofiler_dim3_t gridDim;
    rocprofiler_dim3_t blockDim;
    size_t sharedMem;
    hipStream_t stream;
  } hipConfigureCall;
  struct {
    hipSurfaceObject_t *pSurfObject;
    const hipResourceDesc *pResDesc;
  } hipCreateSurfaceObject;
  struct {
    hipTextureObject_t *pTexObject;
    const hipResourceDesc *pResDesc;
    const hipTextureDesc *pTexDesc;
    const struct hipResourceViewDesc *pResViewDesc;
  } hipCreateTextureObject;
  struct {
    hipCtx_t *ctx;
    unsigned int flags;
    hipDevice_t device;
  } hipCtxCreate;
  struct {
    hipCtx_t ctx;
  } hipCtxDestroy;
  struct {
    hipCtx_t peerCtx;
  } hipCtxDisablePeerAccess;
  struct {
    hipCtx_t peerCtx;
    unsigned int flags;
  } hipCtxEnablePeerAccess;
  struct {
    // In HIP v7.0, apiVersion was changed from int* to unsigned int* to match
    // CUDA signature. If rocprofiler-sdk is compiled with HIP >= 7.0 and HIP is
    // < 7.0 at runtime, there is expectation that this will NOT cause issues:
    // apiVersion should never be negative and should never be >= INT_MAX
    hipCtx_t ctx;
#if ROCPROFILER_SDK_COMPUTE_VERSION(HIP_VERSION_MAJOR, HIP_VERSION_MINOR,      \
                                    0) >=                                      \
    ROCPROFILER_SDK_COMPUTE_VERSION(7, 0, 0)
    unsigned int *apiVersion;
#else
    int *apiVersion; // HIP version < 7.0
#endif
  } hipCtxGetApiVersion;
  struct {
    hipFuncCache_t *cacheConfig;
  } hipCtxGetCacheConfig;
  struct {
    hipCtx_t *ctx;
  } hipCtxGetCurrent;
  struct {
    hipDevice_t *device;
  } hipCtxGetDevice;
  struct {
    unsigned int *flags;
  } hipCtxGetFlags;
  struct {
    hipSharedMemConfig *pConfig;
  } hipCtxGetSharedMemConfig;
  struct {
    hipCtx_t *ctx;
  } hipCtxPopCurrent;
  struct {
    hipCtx_t ctx;
  } hipCtxPushCurrent;
  struct {
    hipFuncCache_t cacheConfig;
  } hipCtxSetCacheConfig;
  struct {
    hipCtx_t ctx;
  } hipCtxSetCurrent;
  struct {
    hipSharedMemConfig config;
  } hipCtxSetSharedMemConfig;
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_hip_api_no_args struct to fix this
    rocprofiler_hip_api_no_args no_args;
  } hipCtxSynchronize;
  struct {
    hipExternalMemory_t extMem;
  } hipDestroyExternalMemory;
  struct {
    hipExternalSemaphore_t extSem;
  } hipDestroyExternalSemaphore;
  struct {
    hipSurfaceObject_t surfaceObject;
  } hipDestroySurfaceObject;
  struct {
    hipTextureObject_t textureObject;
  } hipDestroyTextureObject;
  struct {
    int *canAccessPeer;
    int deviceId;
    int peerDeviceId;
  } hipDeviceCanAccessPeer;
  struct {
    int *major;
    int *minor;
    hipDevice_t device;
  } hipDeviceComputeCapability;
  struct {
    int peerDeviceId;
  } hipDeviceDisablePeerAccess;
  struct {
    int peerDeviceId;
    unsigned int flags;
  } hipDeviceEnablePeerAccess;
  struct {
    hipDevice_t *device;
    int ordinal;
  } hipDeviceGet;
  struct {
    int *pi;
    hipDeviceAttribute_t attr;
    int deviceId;
  } hipDeviceGetAttribute;
  struct {
    int *device;
    const char *pciBusId;
  } hipDeviceGetByPCIBusId;
  struct {
    hipFuncCache_t *cacheConfig;
  } hipDeviceGetCacheConfig;
  struct {
    hipMemPool_t *mem_pool;
    int device;
  } hipDeviceGetDefaultMemPool;
  struct {
    int device;
    hipGraphMemAttributeType attr;
    void *value;
  } hipDeviceGetGraphMemAttribute;
  struct {
    size_t *pValue;
    enum hipLimit_t limit;
  } hipDeviceGetLimit;
  struct {
    hipMemPool_t *mem_pool;
    int device;
  } hipDeviceGetMemPool;
  struct {
    void *name; // changed to void* (real: char*) to avoid stringify on
                // stack-allocated output parameter
    int len;
    hipDevice_t device;
  } hipDeviceGetName;
  struct {
    int *value;
    hipDeviceP2PAttr attr;
    int srcDevice;
    int dstDevice;
  } hipDeviceGetP2PAttribute;
  struct {
    void *pciBusId; // changed to void* (real: char*) to avoid stringify on
                    // stack-allocated output parameter
    int len;
    int device;
  } hipDeviceGetPCIBusId;
  struct {
    hipSharedMemConfig *pConfig;
  } hipDeviceGetSharedMemConfig;
  struct {
    int *leastPriority;
    int *greatestPriority;
  } hipDeviceGetStreamPriorityRange;
  struct {
    hipUUID *uuid;
    hipDevice_t device;
  } hipDeviceGetUuid;
  struct {
    int device;
  } hipDeviceGraphMemTrim;
  struct {
    hipDevice_t dev;
    unsigned int *flags;
    int *active;
  } hipDevicePrimaryCtxGetState;
  struct {
    hipDevice_t dev;
  } hipDevicePrimaryCtxRelease;
  struct {
    hipDevice_t dev;
  } hipDevicePrimaryCtxReset;
  struct {
    hipCtx_t *pctx;
    hipDevice_t dev;
  } hipDevicePrimaryCtxRetain;
  struct {
    hipDevice_t dev;
    unsigned int flags;
  } hipDevicePrimaryCtxSetFlags;
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_hip_api_no_args struct to fix this
    rocprofiler_hip_api_no_args no_args;
  } hipDeviceReset;
  struct {
    hipFuncCache_t cacheConfig;
  } hipDeviceSetCacheConfig;
  struct {
    int device;
    hipGraphMemAttributeType attr;
    void *value;
  } hipDeviceSetGraphMemAttribute;
  struct {
    enum hipLimit_t limit;
    size_t value;
  } hipDeviceSetLimit;
  struct {
    int device;
    hipMemPool_t mem_pool;
  } hipDeviceSetMemPool;
  struct {
    hipSharedMemConfig config;
  } hipDeviceSetSharedMemConfig;
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_hip_api_no_args struct to fix this
    rocprofiler_hip_api_no_args no_args;
  } hipDeviceSynchronize;
  struct {
    size_t *bytes;
    hipDevice_t device;
  } hipDeviceTotalMem;
  struct {
    int *driverVersion;
  } hipDriverGetVersion;
  struct {
    hipError_t hipError;
    const char **errorString;
  } hipDrvGetErrorName;
  struct {
    hipError_t hipError;
    const char **errorString;
  } hipDrvGetErrorString;
  struct {
    hipGraphNode_t *phGraphNode;
    hipGraph_t hGraph;
    const hipGraphNode_t *dependencies;
    size_t numDependencies;
    const HIP_MEMCPY3D *copyParams;
    hipCtx_t ctx;
  } hipDrvGraphAddMemcpyNode;
  struct {
    const hip_Memcpy2D *pCopy;
  } hipDrvMemcpy2DUnaligned;
  struct {
    const HIP_MEMCPY3D *pCopy;
  } hipDrvMemcpy3D;
  struct {
    const HIP_MEMCPY3D *pCopy;
    hipStream_t stream;
  } hipDrvMemcpy3DAsync;
  struct {
    unsigned int numAttributes;
    hipPointer_attribute *attributes;
    void **data;
    hipDeviceptr_t ptr;
  } hipDrvPointerGetAttributes;
  struct {
    hipEvent_t *event;
  } hipEventCreate;
  struct {
    hipEvent_t *event;
    unsigned flags;
  } hipEventCreateWithFlags;
  struct {
    hipEvent_t event;
  } hipEventDestroy;
  struct {
    float *ms;
    hipEvent_t start;
    hipEvent_t stop;
  } hipEventElapsedTime;
  struct {
    hipEvent_t event;
  } hipEventQuery;
  struct {
    hipEvent_t event;
    hipStream_t stream;
  } hipEventRecord;
  struct {
    hipEvent_t event;
  } hipEventSynchronize;
  struct {
    int device1;
    int device2;
    uint32_t *linktype;
    uint32_t *hopcount;
  } hipExtGetLinkTypeAndHopCount;
  struct {
    const void *function_address;
    rocprofiler_dim3_t numBlocks;
    rocprofiler_dim3_t dimBlocks;
    void **args;
    size_t sharedMemBytes;
    hipStream_t stream;
    hipEvent_t startEvent;
    hipEvent_t stopEvent;
    int flags;
  } hipExtLaunchKernel;
  struct {
    hipLaunchParams *launchParamsList;
    int numDevices;
    unsigned int flags;
  } hipExtLaunchMultiKernelMultiDevice;
  struct {
    void **ptr;
    size_t sizeBytes;
    unsigned int flags;
  } hipExtMallocWithFlags;
  struct {
    hipStream_t *stream;
    uint32_t cuMaskSize;
    const uint32_t *cuMask;
  } hipExtStreamCreateWithCUMask;
  struct {
    hipStream_t stream;
    uint32_t cuMaskSize;
    uint32_t *cuMask;
  } hipExtStreamGetCUMask;
  struct {
    void **devPtr;
    hipExternalMemory_t extMem;
    const hipExternalMemoryBufferDesc *bufferDesc;
  } hipExternalMemoryGetMappedBuffer;
  struct {
    void *ptr;
  } hipFree;
  struct {
    hipArray_t array;
  } hipFreeArray;
  struct {
    void *dev_ptr;
    hipStream_t stream;
  } hipFreeAsync;
  struct {
    void *ptr;
  } hipFreeHost;
  struct {
    hipMipmappedArray_t mipmappedArray;
  } hipFreeMipmappedArray;
  struct {
    int *value;
    hipFunction_attribute attrib;
    hipFunction_t hfunc;
  } hipFuncGetAttribute;
  struct {
    struct hipFuncAttributes *attr;
    const void *func;
  } hipFuncGetAttributes;
  struct {
    const void *func;
    hipFuncAttribute attr;
    int value;
  } hipFuncSetAttribute;
  struct {
    const void *func;
    hipFuncCache_t config;
  } hipFuncSetCacheConfig;
  struct {
    const void *func;
    hipSharedMemConfig config;
  } hipFuncSetSharedMemConfig;
  struct {
    unsigned int *pHipDeviceCount;
    int *pHipDevices;
    unsigned int hipDeviceCount;
    hipGLDeviceList deviceList;
  } hipGLGetDevices;
  struct {
    hipChannelFormatDesc *desc;
    hipArray_const_t array;
  } hipGetChannelDesc;
  struct {
    int *deviceId;
  } hipGetDevice;
  struct {
    int *count;
  } hipGetDeviceCount;
  struct {
    unsigned int *flags;
  } hipGetDeviceFlags;
  struct {
    hipDeviceProp_tR0600 *prop;
    int deviceId;
  } hipGetDevicePropertiesR0600;
  struct {
    hipDeviceProp_tR0000 *prop;
    int deviceId;
  } hipGetDevicePropertiesR0000;
  struct {
    hipError_t hip_error;
  } hipGetErrorName;
  struct {
    hipError_t hipError;
  } hipGetErrorString;
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_hip_api_no_args struct to fix this
    rocprofiler_hip_api_no_args no_args;
  } hipGetLastError;
  struct {
    hipArray_t *levelArray;
    hipMipmappedArray_const_t mipmappedArray;
    unsigned int level;
  } hipGetMipmappedArrayLevel;
  struct {
    void **devPtr;
    const void *symbol;
  } hipGetSymbolAddress;
  struct {
    size_t *size;
    const void *symbol;
  } hipGetSymbolSize;
  struct {
    size_t *offset;
    const textureReference *texref;
  } hipGetTextureAlignmentOffset;
  struct {
    hipResourceDesc *pResDesc;
    hipTextureObject_t textureObject;
  } hipGetTextureObjectResourceDesc;
  struct {
    struct hipResourceViewDesc *pResViewDesc;
    hipTextureObject_t textureObject;
  } hipGetTextureObjectResourceViewDesc;
  struct {
    hipTextureDesc *pTexDesc;
    hipTextureObject_t textureObject;
  } hipGetTextureObjectTextureDesc;
  struct {
    const textureReference **texref;
    const void *symbol;
  } hipGetTextureReference;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    hipGraph_t childGraph;
  } hipGraphAddChildGraphNode;
  struct {
    hipGraph_t graph;
    const hipGraphNode_t *from;
    const hipGraphNode_t *to;
    size_t numDependencies;
  } hipGraphAddDependencies;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
  } hipGraphAddEmptyNode;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    hipEvent_t event;
  } hipGraphAddEventRecordNode;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    hipEvent_t event;
  } hipGraphAddEventWaitNode;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    const hipHostNodeParams *pNodeParams;
  } hipGraphAddHostNode;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    const hipKernelNodeParams *pNodeParams;
  } hipGraphAddKernelNode;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    hipMemAllocNodeParams *pNodeParams;
  } hipGraphAddMemAllocNode;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    void *dev_ptr;
  } hipGraphAddMemFreeNode;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    const hipMemcpy3DParms *pCopyParams;
  } hipGraphAddMemcpyNode;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    void *dst;
    const void *src;
    size_t count;
    hipMemcpyKind kind;
  } hipGraphAddMemcpyNode1D;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
  } hipGraphAddMemcpyNodeFromSymbol;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
  } hipGraphAddMemcpyNodeToSymbol;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    const hipMemsetParams *pMemsetParams;
  } hipGraphAddMemsetNode;
  struct {
    hipGraphNode_t node;
    hipGraph_t *pGraph;
  } hipGraphChildGraphNodeGetGraph;
  struct {
    hipGraph_t *pGraphClone;
    hipGraph_t originalGraph;
  } hipGraphClone;
  struct {
    hipGraph_t *pGraph;
    unsigned int flags;
  } hipGraphCreate;
  struct {
    hipGraph_t graph;
    const char *path;
    unsigned int flags;
  } hipGraphDebugDotPrint;
  struct {
    hipGraph_t graph;
  } hipGraphDestroy;
  struct {
    hipGraphNode_t node;
  } hipGraphDestroyNode;
  struct {
    hipGraphNode_t node;
    hipEvent_t *event_out;
  } hipGraphEventRecordNodeGetEvent;
  struct {
    hipGraphNode_t node;
    hipEvent_t event;
  } hipGraphEventRecordNodeSetEvent;
  struct {
    hipGraphNode_t node;
    hipEvent_t *event_out;
  } hipGraphEventWaitNodeGetEvent;
  struct {
    hipGraphNode_t node;
    hipEvent_t event;
  } hipGraphEventWaitNodeSetEvent;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    hipGraph_t childGraph;
  } hipGraphExecChildGraphNodeSetParams;
  struct {
    hipGraphExec_t graphExec;
  } hipGraphExecDestroy;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    hipEvent_t event;
  } hipGraphExecEventRecordNodeSetEvent;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    hipEvent_t event;
  } hipGraphExecEventWaitNodeSetEvent;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    const hipHostNodeParams *pNodeParams;
  } hipGraphExecHostNodeSetParams;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    const hipKernelNodeParams *pNodeParams;
  } hipGraphExecKernelNodeSetParams;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    hipMemcpy3DParms *pNodeParams;
  } hipGraphExecMemcpyNodeSetParams;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    void *dst;
    const void *src;
    size_t count;
    hipMemcpyKind kind;
  } hipGraphExecMemcpyNodeSetParams1D;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
  } hipGraphExecMemcpyNodeSetParamsFromSymbol;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
  } hipGraphExecMemcpyNodeSetParamsToSymbol;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    const hipMemsetParams *pNodeParams;
  } hipGraphExecMemsetNodeSetParams;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraph_t hGraph;
    hipGraphNode_t *hErrorNode_out;
    hipGraphExecUpdateResult *updateResult_out;
  } hipGraphExecUpdate;
  struct {
    hipGraph_t graph;
    hipGraphNode_t *from;
    hipGraphNode_t *to;
    size_t *numEdges;
  } hipGraphGetEdges;
  struct {
    hipGraph_t graph;
    hipGraphNode_t *nodes;
    size_t *numNodes;
  } hipGraphGetNodes;
  struct {
    hipGraph_t graph;
    hipGraphNode_t *pRootNodes;
    size_t *pNumRootNodes;
  } hipGraphGetRootNodes;
  struct {
    hipGraphNode_t node;
    hipHostNodeParams *pNodeParams;
  } hipGraphHostNodeGetParams;
  struct {
    hipGraphNode_t node;
    const hipHostNodeParams *pNodeParams;
  } hipGraphHostNodeSetParams;
  struct {
    hipGraphExec_t *pGraphExec;
    hipGraph_t graph;
    hipGraphNode_t *pErrorNode;
    void *pLogBuffer; // changed to void* (real: char*) to avoid stringify on
                      // stack-allocated output parameter
    size_t bufferSize;
  } hipGraphInstantiate;
  struct {
    hipGraphExec_t *pGraphExec;
    hipGraph_t graph;
    unsigned long long flags;
  } hipGraphInstantiateWithFlags;
  struct {
    hipGraphNode_t hSrc;
    hipGraphNode_t hDst;
  } hipGraphKernelNodeCopyAttributes;
  struct {
    hipGraphNode_t hNode;
    hipKernelNodeAttrID attr;
    hipKernelNodeAttrValue *value;
  } hipGraphKernelNodeGetAttribute;
  struct {
    hipGraphNode_t node;
    hipKernelNodeParams *pNodeParams;
  } hipGraphKernelNodeGetParams;
  struct {
    hipGraphNode_t hNode;
    hipKernelNodeAttrID attr;
    const hipKernelNodeAttrValue *value;
  } hipGraphKernelNodeSetAttribute;
  struct {
    hipGraphNode_t node;
    const hipKernelNodeParams *pNodeParams;
  } hipGraphKernelNodeSetParams;
  struct {
    hipGraphExec_t graphExec;
    hipStream_t stream;
  } hipGraphLaunch;
  struct {
    hipGraphNode_t node;
    hipMemAllocNodeParams *pNodeParams;
  } hipGraphMemAllocNodeGetParams;
  struct {
    hipGraphNode_t node;
    void *dev_ptr;
  } hipGraphMemFreeNodeGetParams;
  struct {
    hipGraphNode_t node;
    hipMemcpy3DParms *pNodeParams;
  } hipGraphMemcpyNodeGetParams;
  struct {
    hipGraphNode_t node;
    const hipMemcpy3DParms *pNodeParams;
  } hipGraphMemcpyNodeSetParams;
  struct {
    hipGraphNode_t node;
    void *dst;
    const void *src;
    size_t count;
    hipMemcpyKind kind;
  } hipGraphMemcpyNodeSetParams1D;
  struct {
    hipGraphNode_t node;
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
  } hipGraphMemcpyNodeSetParamsFromSymbol;
  struct {
    hipGraphNode_t node;
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
  } hipGraphMemcpyNodeSetParamsToSymbol;
  struct {
    hipGraphNode_t node;
    hipMemsetParams *pNodeParams;
  } hipGraphMemsetNodeGetParams;
  struct {
    hipGraphNode_t node;
    const hipMemsetParams *pNodeParams;
  } hipGraphMemsetNodeSetParams;
  struct {
    hipGraphNode_t *pNode;
    hipGraphNode_t originalNode;
    hipGraph_t clonedGraph;
  } hipGraphNodeFindInClone;
  struct {
    hipGraphNode_t node;
    hipGraphNode_t *pDependencies;
    size_t *pNumDependencies;
  } hipGraphNodeGetDependencies;
  struct {
    hipGraphNode_t node;
    hipGraphNode_t *pDependentNodes;
    size_t *pNumDependentNodes;
  } hipGraphNodeGetDependentNodes;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    unsigned int *isEnabled;
  } hipGraphNodeGetEnabled;
  struct {
    hipGraphNode_t node;
    hipGraphNodeType *pType;
  } hipGraphNodeGetType;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    unsigned int isEnabled;
  } hipGraphNodeSetEnabled;
  struct {
    hipGraph_t graph;
    hipUserObject_t object;
    unsigned int count;
  } hipGraphReleaseUserObject;
  struct {
    hipGraph_t graph;
    const hipGraphNode_t *from;
    const hipGraphNode_t *to;
    size_t numDependencies;
  } hipGraphRemoveDependencies;
  struct {
    hipGraph_t graph;
    hipUserObject_t object;
    unsigned int count;
    unsigned int flags;
  } hipGraphRetainUserObject;
  struct {
    hipGraphExec_t graphExec;
    hipStream_t stream;
  } hipGraphUpload;
  struct {
    hipGraphicsResource **resource;
    GLuint buffer;
    unsigned int flags;
  } hipGraphicsGLRegisterBuffer;
  struct {
    hipGraphicsResource **resource;
    GLuint image;
    GLenum target;
    unsigned int flags;
  } hipGraphicsGLRegisterImage;
  struct {
    int count;
    hipGraphicsResource_t *resources;
    hipStream_t stream;
  } hipGraphicsMapResources;
  struct {
    void **devPtr;
    size_t *size;
    hipGraphicsResource_t resource;
  } hipGraphicsResourceGetMappedPointer;
  struct {
    hipArray_t *array;
    hipGraphicsResource_t resource;
    unsigned int arrayIndex;
    unsigned int mipLevel;
  } hipGraphicsSubResourceGetMappedArray;
  struct {
    int count;
    hipGraphicsResource_t *resources;
    hipStream_t stream;
  } hipGraphicsUnmapResources;
  struct {
    hipGraphicsResource_t resource;
  } hipGraphicsUnregisterResource;
  struct {
    void **ptr;
    size_t size;
    unsigned int flags;
  } hipHostAlloc;
  struct {
    void *ptr;
  } hipHostFree;
  struct {
    void **devPtr;
    void *hstPtr;
    unsigned int flags;
  } hipHostGetDevicePointer;
  struct {
    unsigned int *flagsPtr;
    void *hostPtr;
  } hipHostGetFlags;
  struct {
    void **ptr;
    size_t size;
    unsigned int flags;
  } hipHostMalloc;
  struct {
    void *hostPtr;
    size_t sizeBytes;
    unsigned int flags;
  } hipHostRegister;
  struct {
    void *hostPtr;
  } hipHostUnregister;
  struct {
    hipExternalMemory_t *extMem_out;
    const hipExternalMemoryHandleDesc *memHandleDesc;
  } hipImportExternalMemory;
  struct {
    hipExternalSemaphore_t *extSem_out;
    const hipExternalSemaphoreHandleDesc *semHandleDesc;
  } hipImportExternalSemaphore;
  struct {
    unsigned int flags;
  } hipInit;
  struct {
    void *devPtr;
  } hipIpcCloseMemHandle;
  struct {
    hipIpcEventHandle_t *handle;
    hipEvent_t event;
  } hipIpcGetEventHandle;
  struct {
    hipIpcMemHandle_t *handle;
    void *devPtr;
  } hipIpcGetMemHandle;
  struct {
    hipEvent_t *event;
    hipIpcEventHandle_t handle;
  } hipIpcOpenEventHandle;
  struct {
    void **devPtr;
    hipIpcMemHandle_t handle;
    unsigned int flags;
  } hipIpcOpenMemHandle;
  struct {
    hipFunction_t func;
  } hipKernelNameRef;
  struct {
    const void *hostFunction;
    hipStream_t stream;
  } hipKernelNameRefByPtr;
  struct {
    const void *func;
  } hipLaunchByPtr;
  struct {
    const void *func;
    rocprofiler_dim3_t gridDim;
    rocprofiler_dim3_t blockDimX;
    void **kernelParams;
    unsigned int sharedMemBytes;
    hipStream_t stream;
  } hipLaunchCooperativeKernel;
  struct {
    hipLaunchParams *launchParamsList;
    int numDevices;
    unsigned int flags;
  } hipLaunchCooperativeKernelMultiDevice;
  struct {
    hipStream_t stream;
    hipHostFn_t fn;
    void *userData;
  } hipLaunchHostFunc;
  struct {
    const void *function_address;
    rocprofiler_dim3_t numBlocks;
    rocprofiler_dim3_t dimBlocks;
    void **args;
    size_t sharedMemBytes;
    hipStream_t stream;
  } hipLaunchKernel;
  struct {
    void **ptr;
    size_t size;
  } hipMalloc;
  struct {
    hipPitchedPtr *pitchedDevPtr;
    hipExtent extent;
  } hipMalloc3D;
  struct {
    hipArray_t *array;
    const struct hipChannelFormatDesc *desc;
    struct hipExtent extent;
    unsigned int flags;
  } hipMalloc3DArray;
  struct {
    hipArray_t *array;
    const hipChannelFormatDesc *desc;
    size_t width;
    size_t height;
    unsigned int flags;
  } hipMallocArray;
  struct {
    void **dev_ptr;
    size_t size;
    hipStream_t stream;
  } hipMallocAsync;
  struct {
    void **dev_ptr;
    size_t size;
    hipMemPool_t mem_pool;
    hipStream_t stream;
  } hipMallocFromPoolAsync;
  struct {
    void **ptr;
    size_t size;
  } hipMallocHost;
  struct {
    void **dev_ptr;
    size_t size;
    unsigned int flags;
  } hipMallocManaged;
  struct {
    hipMipmappedArray_t *mipmappedArray;
    const struct hipChannelFormatDesc *desc;
    struct hipExtent extent;
    unsigned int numLevels;
    unsigned int flags;
  } hipMallocMipmappedArray;
  struct {
    void **ptr;
    size_t *pitch;
    size_t width;
    size_t height;
  } hipMallocPitch;
  struct {
    void *devPtr;
    size_t size;
  } hipMemAddressFree;
  struct {
    void **ptr;
    size_t size;
    size_t alignment;
    void *addr;
    unsigned long long flags;
  } hipMemAddressReserve;
  struct {
    const void *dev_ptr;
    size_t count;
    hipMemoryAdvise advice;
    int device;
  } hipMemAdvise;
  struct {
    void **ptr;
    size_t size;
  } hipMemAllocHost;
  struct {
    hipDeviceptr_t *dptr;
    size_t *pitch;
    size_t widthInBytes;
    size_t height;
    unsigned int elementSizeBytes;
  } hipMemAllocPitch;
  struct {
    hipMemGenericAllocationHandle_t *handle;
    size_t size;
    const hipMemAllocationProp *prop;
    unsigned long long flags;
  } hipMemCreate;
  struct {
    void *shareableHandle;
    hipMemGenericAllocationHandle_t handle;
    hipMemAllocationHandleType handleType;
    unsigned long long flags;
  } hipMemExportToShareableHandle;
  struct {
    unsigned long long *flags;
    const hipMemLocation *location;
    void *ptr;
  } hipMemGetAccess;
  struct {
    hipDeviceptr_t *pbase;
    size_t *psize;
    hipDeviceptr_t dptr;
  } hipMemGetAddressRange;
  struct {
    size_t *granularity;
    const hipMemAllocationProp *prop;
    hipMemAllocationGranularity_flags option;
  } hipMemGetAllocationGranularity;
  struct {
    hipMemAllocationProp *prop;
    hipMemGenericAllocationHandle_t handle;
  } hipMemGetAllocationPropertiesFromHandle;
  struct {
    size_t *free;
    size_t *total;
  } hipMemGetInfo;
  struct {
    hipMemGenericAllocationHandle_t *handle;
    void *osHandle;
    hipMemAllocationHandleType shHandleType;
  } hipMemImportFromShareableHandle;
  struct {
    void *ptr;
    size_t size;
    size_t offset;
    hipMemGenericAllocationHandle_t handle;
    unsigned long long flags;
  } hipMemMap;
  struct {
    hipArrayMapInfo *mapInfoList;
    unsigned int count;
    hipStream_t stream;
  } hipMemMapArrayAsync;
  struct {
    hipMemPool_t *mem_pool;
    const hipMemPoolProps *pool_props;
  } hipMemPoolCreate;
  struct {
    hipMemPool_t mem_pool;
  } hipMemPoolDestroy;
  struct {
    hipMemPoolPtrExportData *export_data;
    void *dev_ptr;
  } hipMemPoolExportPointer;
  struct {
    void *shared_handle;
    hipMemPool_t mem_pool;
    hipMemAllocationHandleType handle_type;
    unsigned int flags;
  } hipMemPoolExportToShareableHandle;
  struct {
    hipMemAccessFlags *flags;
    hipMemPool_t mem_pool;
    hipMemLocation *location;
  } hipMemPoolGetAccess;
  struct {
    hipMemPool_t mem_pool;
    hipMemPoolAttr attr;
    void *value;
  } hipMemPoolGetAttribute;
  struct {
    hipMemPool_t *mem_pool;
    void *shared_handle;
    hipMemAllocationHandleType handle_type;
    unsigned int flags;
  } hipMemPoolImportFromShareableHandle;
  struct {
    void **dev_ptr;
    hipMemPool_t mem_pool;
    hipMemPoolPtrExportData *export_data;
  } hipMemPoolImportPointer;
  struct {
    hipMemPool_t mem_pool;
    const hipMemAccessDesc *desc_list;
    size_t count;
  } hipMemPoolSetAccess;
  struct {
    hipMemPool_t mem_pool;
    hipMemPoolAttr attr;
    void *value;
  } hipMemPoolSetAttribute;
  struct {
    hipMemPool_t mem_pool;
    size_t min_bytes_to_hold;
  } hipMemPoolTrimTo;
  struct {
    const void *dev_ptr;
    size_t count;
    int device;
    hipStream_t stream;
  } hipMemPrefetchAsync;
  struct {
    void *ptr;
    size_t *size;
  } hipMemPtrGetInfo;
  struct {
    void *data;
    size_t data_size;
    hipMemRangeAttribute attribute;
    const void *dev_ptr;
    size_t count;
  } hipMemRangeGetAttribute;
  struct {
    void **data;
    size_t *data_sizes;
    hipMemRangeAttribute *attributes;
    size_t num_attributes;
    const void *dev_ptr;
    size_t count;
  } hipMemRangeGetAttributes;
  struct {
    hipMemGenericAllocationHandle_t handle;
  } hipMemRelease;
  struct {
    hipMemGenericAllocationHandle_t *handle;
    void *addr;
  } hipMemRetainAllocationHandle;
  struct {
    void *ptr;
    size_t size;
    const hipMemAccessDesc *desc;
    size_t count;
  } hipMemSetAccess;
  struct {
    void *ptr;
    size_t size;
  } hipMemUnmap;
  struct {
    void *dst;
    const void *src;
    size_t sizeBytes;
    hipMemcpyKind kind;
  } hipMemcpy;
  struct {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
  } hipMemcpy2D;
  struct {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpy2DAsync;
  struct {
    void *dst;
    size_t dpitch;
    hipArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
  } hipMemcpy2DFromArray;
  struct {
    void *dst;
    size_t dpitch;
    hipArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpy2DFromArrayAsync;
  struct {
    hipArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
  } hipMemcpy2DToArray;
  struct {
    hipArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpy2DToArrayAsync;
  struct {
    const struct hipMemcpy3DParms *p;
  } hipMemcpy3D;
  struct {
    const struct hipMemcpy3DParms *p;
    hipStream_t stream;
  } hipMemcpy3DAsync;
  struct {
    void *dst;
    const void *src;
    size_t sizeBytes;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpyAsync;
  struct {
    void *dst;
    hipArray_t srcArray;
    size_t srcOffset;
    size_t count;
  } hipMemcpyAtoH;
  struct {
    hipDeviceptr_t dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
  } hipMemcpyDtoD;
  struct {
    hipDeviceptr_t dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
    hipStream_t stream;
  } hipMemcpyDtoDAsync;
  struct {
    void *dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
  } hipMemcpyDtoH;
  struct {
    void *dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
    hipStream_t stream;
  } hipMemcpyDtoHAsync;
  struct {
    void *dst;
    hipArray_const_t srcArray;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    hipMemcpyKind kind;
  } hipMemcpyFromArray;
  struct {
    void *dst;
    const void *symbol;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
  } hipMemcpyFromSymbol;
  struct {
    void *dst;
    const void *symbol;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpyFromSymbolAsync;
  struct {
    hipArray_t dstArray;
    size_t dstOffset;
    const void *srcHost;
    size_t count;
  } hipMemcpyHtoA;
  struct {
    hipDeviceptr_t dst;
    const void *src;
    size_t sizeBytes;
  } hipMemcpyHtoD;
  struct {
    hipDeviceptr_t dst;
    const void *src;
    size_t sizeBytes;
    hipStream_t stream;
  } hipMemcpyHtoDAsync;
  struct {
    const hip_Memcpy2D *pCopy;
  } hipMemcpyParam2D;
  struct {
    const hip_Memcpy2D *pCopy;
    hipStream_t stream;
  } hipMemcpyParam2DAsync;
  struct {
    void *dst;
    int dstDeviceId;
    const void *src;
    int srcDeviceId;
    size_t sizeBytes;
  } hipMemcpyPeer;
  struct {
    void *dst;
    int dstDeviceId;
    const void *src;
    int srcDevice;
    size_t sizeBytes;
    hipStream_t stream;
  } hipMemcpyPeerAsync;
  struct {
    hipArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    hipMemcpyKind kind;
  } hipMemcpyToArray;
  struct {
    const void *symbol;
    const void *src;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
  } hipMemcpyToSymbol;
  struct {
    const void *symbol;
    const void *src;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpyToSymbolAsync;
  struct {
    void *dst;
    const void *src;
    size_t sizeBytes;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpyWithStream;
  struct {
    void *dst;
    int value;
    size_t sizeBytes;
  } hipMemset;
  struct {
    void *dst;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
  } hipMemset2D;
  struct {
    void *dst;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    hipStream_t stream;
  } hipMemset2DAsync;
  struct {
    hipPitchedPtr pitchedDevPtr;
    int value;
    hipExtent extent;
  } hipMemset3D;
  struct {
    hipPitchedPtr pitchedDevPtr;
    int value;
    hipExtent extent;
    hipStream_t stream;
  } hipMemset3DAsync;
  struct {
    void *dst;
    int value;
    size_t sizeBytes;
    hipStream_t stream;
  } hipMemsetAsync;
  struct {
    hipDeviceptr_t dest;
    unsigned short value;
    size_t count;
  } hipMemsetD16;
  struct {
    hipDeviceptr_t dest;
    unsigned short value;
    size_t count;
    hipStream_t stream;
  } hipMemsetD16Async;
  struct {
    hipDeviceptr_t dest;
    int value;
    size_t count;
  } hipMemsetD32;
  struct {
    hipDeviceptr_t dst;
    int value;
    size_t count;
    hipStream_t stream;
  } hipMemsetD32Async;
  struct {
    hipDeviceptr_t dest;
    unsigned char value;
    size_t count;
  } hipMemsetD8;
  struct {
    hipDeviceptr_t dest;
    unsigned char value;
    size_t count;
    hipStream_t stream;
  } hipMemsetD8Async;
  struct {
    hipMipmappedArray_t *pHandle;
    HIP_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc;
    unsigned int numMipmapLevels;
  } hipMipmappedArrayCreate;
  struct {
    hipMipmappedArray_t hMipmappedArray;
  } hipMipmappedArrayDestroy;
  struct {
    hipArray_t *pLevelArray;
    hipMipmappedArray_t hMipMappedArray;
    unsigned int level;
  } hipMipmappedArrayGetLevel;
  struct {
    hipFunction_t *function;
    hipModule_t module;
    const char *kname;
  } hipModuleGetFunction;
  struct {
    hipDeviceptr_t *dptr;
    size_t *bytes;
    hipModule_t hmod;
    const char *name;
  } hipModuleGetGlobal;
  struct {
    textureReference **texRef;
    hipModule_t hmod;
    const char *name;
  } hipModuleGetTexRef;
  struct {
    hipFunction_t func;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    hipStream_t stream;
    void **kernelParams;
  } hipModuleLaunchCooperativeKernel;
  struct {
    hipFunctionLaunchParams *launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
  } hipModuleLaunchCooperativeKernelMultiDevice;
  struct {
    hipFunction_t func;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    hipStream_t stream;
    void **kernelParams;
    void **extra;
  } hipModuleLaunchKernel;
  struct {
    hipModule_t *module;
    const char *fname;
  } hipModuleLoad;
  struct {
    hipModule_t *module;
    const void *image;
  } hipModuleLoadData;
  struct {
    hipModule_t *module;
    const void *image;
    unsigned int numOptions;
    hipJitOption *options;
    void **optionValues;
  } hipModuleLoadDataEx;
  struct {
    int *numBlocks;
    hipFunction_t func;
    int blockSize;
    size_t dynSharedMemPerBlk;
  } hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
  struct {
    int *numBlocks;
    hipFunction_t func;
    int blockSize;
    size_t dynSharedMemPerBlk;
    unsigned int flags;
  } hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
  struct {
    int *gridSize;
    int *blockSize;
    hipFunction_t func;
    size_t dynSharedMemPerBlk;
    int blockSizeLimit;
  } hipModuleOccupancyMaxPotentialBlockSize;
  struct {
    int *gridSize;
    int *blockSize;
    hipFunction_t func;
    size_t dynSharedMemPerBlk;
    int blockSizeLimit;
    unsigned int flags;
  } hipModuleOccupancyMaxPotentialBlockSizeWithFlags;
  struct {
    hipModule_t module;
  } hipModuleUnload;
  struct {
    int *numBlocks;
    const void *func;
    int blockSize;
    size_t dynSharedMemPerBlk;
  } hipOccupancyMaxActiveBlocksPerMultiprocessor;
  struct {
    int *numBlocks;
    const void *func;
    int blockSize;
    size_t dynSharedMemPerBlk;
    unsigned int flags;
  } hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
  struct {
    int *gridSize;
    int *blockSize;
    const void *func;
    size_t dynSharedMemPerBlk;
    int blockSizeLimit;
  } hipOccupancyMaxPotentialBlockSize;
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_hip_api_no_args struct to fix this
    rocprofiler_hip_api_no_args no_args;
  } hipPeekAtLastError;
  struct {
    void *data;
    hipPointer_attribute attribute;
    hipDeviceptr_t ptr;
  } hipPointerGetAttribute;
  struct {
    hipPointerAttribute_t *attributes;
    const void *ptr;
  } hipPointerGetAttributes;
  struct {
    const void *value;
    hipPointer_attribute attribute;
    hipDeviceptr_t ptr;
  } hipPointerSetAttribute;
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_hip_api_no_args struct to fix this
    rocprofiler_hip_api_no_args no_args;
  } hipProfilerStart;
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_hip_api_no_args struct to fix this
    rocprofiler_hip_api_no_args no_args;
  } hipProfilerStop;
  struct {
    int *runtimeVersion;
  } hipRuntimeGetVersion;
  struct {
    int deviceId;
  } hipSetDevice;
  struct {
    unsigned flags;
  } hipSetDeviceFlags;
  struct {
    const void *arg;
    size_t size;
    size_t offset;
  } hipSetupArgument;
  struct {
    const hipExternalSemaphore_t *extSemArray;
    const hipExternalSemaphoreSignalParams *paramsArray;
    unsigned int numExtSems;
    hipStream_t stream;
  } hipSignalExternalSemaphoresAsync;
  struct {
    hipStream_t stream;
    hipStreamCallback_t callback;
    void *userData;
    unsigned int flags;
  } hipStreamAddCallback;
  struct {
    hipStream_t stream;
    void *dev_ptr;
    size_t length;
    unsigned int flags;
  } hipStreamAttachMemAsync;
  struct {
    hipStream_t stream;
    hipStreamCaptureMode mode;
  } hipStreamBeginCapture;
  struct {
    hipStream_t *stream;
  } hipStreamCreate;
  struct {
    hipStream_t *stream;
    unsigned int flags;
  } hipStreamCreateWithFlags;
  struct {
    hipStream_t *stream;
    unsigned int flags;
    int priority;
  } hipStreamCreateWithPriority;
  struct {
    hipStream_t stream;
  } hipStreamDestroy;
  struct {
    hipStream_t stream;
    hipGraph_t *pGraph;
  } hipStreamEndCapture;
  struct {
    hipStream_t stream;
    hipStreamCaptureStatus *pCaptureStatus;
    unsigned long long *pId;
  } hipStreamGetCaptureInfo;
  struct {
    hipStream_t stream;
    hipStreamCaptureStatus *captureStatus_out;
    unsigned long long *id_out;
    hipGraph_t *graph_out;
    const hipGraphNode_t **dependencies_out;
    size_t *numDependencies_out;
  } hipStreamGetCaptureInfo_v2;
  struct {
    hipStream_t stream;
    hipDevice_t *device;
  } hipStreamGetDevice;
  struct {
    hipStream_t stream;
    unsigned int *flags;
  } hipStreamGetFlags;
  struct {
    hipStream_t stream;
    int *priority;
  } hipStreamGetPriority;
  struct {
    hipStream_t stream;
    hipStreamCaptureStatus *pCaptureStatus;
  } hipStreamIsCapturing;
  struct {
    hipStream_t stream;
  } hipStreamQuery;
  struct {
    hipStream_t stream;
  } hipStreamSynchronize;
  struct {
    hipStream_t stream;
    hipGraphNode_t *dependencies;
    size_t numDependencies;
    unsigned int flags;
  } hipStreamUpdateCaptureDependencies;
  struct {
    hipStream_t stream;
    hipEvent_t event;
    unsigned int flags;
  } hipStreamWaitEvent;
  struct {
    hipStream_t stream;
    void *ptr;
    uint32_t value;
    unsigned int flags;
    uint32_t mask;
  } hipStreamWaitValue32;
  struct {
    hipStream_t stream;
    void *ptr;
    uint64_t value;
    unsigned int flags;
    uint64_t mask;
  } hipStreamWaitValue64;
  struct {
    hipStream_t stream;
    void *ptr;
    uint32_t value;
    unsigned int flags;
  } hipStreamWriteValue32;
  struct {
    hipStream_t stream;
    void *ptr;
    uint64_t value;
    unsigned int flags;
  } hipStreamWriteValue64;
  struct {
    hipTextureObject_t *pTexObject;
    const HIP_RESOURCE_DESC *pResDesc;
    const HIP_TEXTURE_DESC *pTexDesc;
    const HIP_RESOURCE_VIEW_DESC *pResViewDesc;
  } hipTexObjectCreate;
  struct {
    hipTextureObject_t texObject;
  } hipTexObjectDestroy;
  struct {
    HIP_RESOURCE_DESC *pResDesc;
    hipTextureObject_t texObject;
  } hipTexObjectGetResourceDesc;
  struct {
    HIP_RESOURCE_VIEW_DESC *pResViewDesc;
    hipTextureObject_t texObject;
  } hipTexObjectGetResourceViewDesc;
  struct {
    HIP_TEXTURE_DESC *pTexDesc;
    hipTextureObject_t texObject;
  } hipTexObjectGetTextureDesc;
  struct {
    hipDeviceptr_t *dev_ptr;
    const textureReference *texRef;
  } hipTexRefGetAddress;
  struct {
    enum hipTextureAddressMode *pam;
    const textureReference *texRef;
    int dim;
  } hipTexRefGetAddressMode;
  struct {
    enum hipTextureFilterMode *pfm;
    const textureReference *texRef;
  } hipTexRefGetFilterMode;
  struct {
    unsigned int *pFlags;
    const textureReference *texRef;
  } hipTexRefGetFlags;
  struct {
    hipArray_Format *pFormat;
    int *pNumChannels;
    const textureReference *texRef;
  } hipTexRefGetFormat;
  struct {
    int *pmaxAnsio;
    const textureReference *texRef;
  } hipTexRefGetMaxAnisotropy;
  struct {
    hipMipmappedArray_t *pArray;
    const textureReference *texRef;
  } hipTexRefGetMipMappedArray;
  struct {
    enum hipTextureFilterMode *pfm;
    const textureReference *texRef;
  } hipTexRefGetMipmapFilterMode;
  struct {
    float *pbias;
    const textureReference *texRef;
  } hipTexRefGetMipmapLevelBias;
  struct {
    float *pminMipmapLevelClamp;
    float *pmaxMipmapLevelClamp;
    const textureReference *texRef;
  } hipTexRefGetMipmapLevelClamp;
  struct {
    size_t *ByteOffset;
    textureReference *texRef;
    hipDeviceptr_t dptr;
    size_t bytes;
  } hipTexRefSetAddress;
  struct {
    textureReference *texRef;
    const HIP_ARRAY_DESCRIPTOR *desc;
    hipDeviceptr_t dptr;
    size_t Pitch;
  } hipTexRefSetAddress2D;
  struct {
    textureReference *texRef;
    int dim;
    enum hipTextureAddressMode am;
  } hipTexRefSetAddressMode;
  struct {
    textureReference *tex;
    hipArray_const_t array;
    unsigned int flags;
  } hipTexRefSetArray;
  struct {
    textureReference *texRef;
    float *pBorderColor;
  } hipTexRefSetBorderColor;
  struct {
    textureReference *texRef;
    enum hipTextureFilterMode fm;
  } hipTexRefSetFilterMode;
  struct {
    textureReference *texRef;
    unsigned int Flags;
  } hipTexRefSetFlags;
  struct {
    textureReference *texRef;
    hipArray_Format fmt;
    int NumPackedComponents;
  } hipTexRefSetFormat;
  struct {
    textureReference *texRef;
    unsigned int maxAniso;
  } hipTexRefSetMaxAnisotropy;
  struct {
    textureReference *texRef;
    enum hipTextureFilterMode fm;
  } hipTexRefSetMipmapFilterMode;
  struct {
    textureReference *texRef;
    float bias;
  } hipTexRefSetMipmapLevelBias;
  struct {
    textureReference *texRef;
    float minMipMapLevelClamp;
    float maxMipMapLevelClamp;
  } hipTexRefSetMipmapLevelClamp;
  struct {
    textureReference *texRef;
    struct hipMipmappedArray *mipmappedArray;
    unsigned int Flags;
  } hipTexRefSetMipmappedArray;
  struct {
    hipStreamCaptureMode *mode;
  } hipThreadExchangeStreamCaptureMode;
  struct {
    const textureReference *tex;
  } hipUnbindTexture;
  struct {
    hipUserObject_t *object_out;
    void *ptr;
    hipHostFn_t destroy;
    unsigned int initialRefcount;
    unsigned int flags;
  } hipUserObjectCreate;
  struct {
    hipUserObject_t object;
    unsigned int count;
  } hipUserObjectRelease;
  struct {
    hipUserObject_t object;
    unsigned int count;
  } hipUserObjectRetain;
  struct {
    const hipExternalSemaphore_t *extSemArray;
    const hipExternalSemaphoreWaitParams *paramsArray;
    unsigned int numExtSems;
    hipStream_t stream;
  } hipWaitExternalSemaphoresAsync;
  struct {
    int x;
    int y;
    int z;
    int w;
    hipChannelFormatKind f;
  } hipCreateChannelDesc;
  struct {
    hipFunction_t func;
    uint32_t globalWorkSizeX;
    uint32_t globalWorkSizeY;
    uint32_t globalWorkSizeZ;
    uint32_t localWorkSizeX;
    uint32_t localWorkSizeY;
    uint32_t localWorkSizeZ;
    size_t sharedMemBytes;
    hipStream_t stream;
    void **kernelParams;
    void **extra;
    hipEvent_t startEvent;
    hipEvent_t stopEvent;
    uint32_t flags;
  } hipExtModuleLaunchKernel;
  struct {
    hipFunction_t func;
    uint32_t globalWorkSizeX;
    uint32_t globalWorkSizeY;
    uint32_t globalWorkSizeZ;
    uint32_t localWorkSizeX;
    uint32_t localWorkSizeY;
    uint32_t localWorkSizeZ;
    size_t sharedMemBytes;
    hipStream_t stream;
    void **kernelParams;
    void **extra;
    hipEvent_t startEvent;
    hipEvent_t stopEvent;
  } hipHccModuleLaunchKernel;
  struct {
    void *dst;
    const void *src;
    size_t sizeBytes;
    hipMemcpyKind kind;
  } hipMemcpy_spt;
  struct {
    const void *symbol;
    const void *src;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
  } hipMemcpyToSymbol_spt;
  struct {
    void *dst;
    const void *symbol;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
  } hipMemcpyFromSymbol_spt;
  struct {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
  } hipMemcpy2D_spt;
  struct {
    void *dst;
    size_t dpitch;
    hipArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
  } hipMemcpy2DFromArray_spt;
  struct {
    const struct hipMemcpy3DParms *p;
  } hipMemcpy3D_spt;
  struct {
    void *dst;
    int value;
    size_t sizeBytes;
  } hipMemset_spt;
  struct {
    void *dst;
    int value;
    size_t sizeBytes;
    hipStream_t stream;
  } hipMemsetAsync_spt;
  struct {
    void *dst;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
  } hipMemset2D_spt;
  struct {
    void *dst;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    hipStream_t stream;
  } hipMemset2DAsync_spt;
  struct {
    hipPitchedPtr pitchedDevPtr;
    int value;
    hipExtent extent;
    hipStream_t stream;
  } hipMemset3DAsync_spt;
  struct {
    hipPitchedPtr pitchedDevPtr;
    int value;
    hipExtent extent;
  } hipMemset3D_spt;
  struct {
    void *dst;
    const void *src;
    size_t sizeBytes;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpyAsync_spt;
  struct {
    const hipMemcpy3DParms *p;
    hipStream_t stream;
  } hipMemcpy3DAsync_spt;
  struct {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpy2DAsync_spt;
  struct {
    void *dst;
    const void *symbol;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpyFromSymbolAsync_spt;
  struct {
    const void *symbol;
    const void *src;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpyToSymbolAsync_spt;
  struct {
    void *dst;
    hipArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffset;
    size_t count;
    hipMemcpyKind kind;
  } hipMemcpyFromArray_spt;
  struct {
    hipArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
  } hipMemcpy2DToArray_spt;
  struct {
    void *dst;
    size_t dpitch;
    hipArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpy2DFromArrayAsync_spt;
  struct {
    hipArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
    hipStream_t stream;
  } hipMemcpy2DToArrayAsync_spt;
  struct {
    hipStream_t stream;
  } hipStreamQuery_spt;
  struct {
    hipStream_t stream;
  } hipStreamSynchronize_spt;
  struct {
    hipStream_t stream;
    int *priority;
  } hipStreamGetPriority_spt;
  struct {
    hipStream_t stream;
    hipEvent_t event;
    unsigned int flags;
  } hipStreamWaitEvent_spt;
  struct {
    hipStream_t stream;
    unsigned int *flags;
  } hipStreamGetFlags_spt;
  struct {
    hipStream_t stream;
    hipStreamCallback_t callback;
    void *userData;
    unsigned int flags;
  } hipStreamAddCallback_spt;
  struct {
    hipEvent_t event;
    hipStream_t stream;
  } hipEventRecord_spt;
  struct {
    const void *func;
    rocprofiler_dim3_t gridDim;
    rocprofiler_dim3_t blockDim;
    void **kernelParams;
    uint32_t sharedMemBytes;
    hipStream_t stream;
  } hipLaunchCooperativeKernel_spt;
  struct {
    const void *function_address;
    rocprofiler_dim3_t numBlocks;
    rocprofiler_dim3_t dimBlocks;
    void **args;
    size_t sharedMemBytes;
    hipStream_t stream;
  } hipLaunchKernel_spt;
  struct {
    hipGraphExec_t graphExec;
    hipStream_t stream;
  } hipGraphLaunch_spt;
  struct {
    hipStream_t stream;
    hipStreamCaptureMode mode;
  } hipStreamBeginCapture_spt;
  struct {
    hipStream_t stream;
    hipGraph_t *pGraph;
  } hipStreamEndCapture_spt;
  struct {
    hipStream_t stream;
    hipStreamCaptureStatus *pCaptureStatus;
  } hipStreamIsCapturing_spt;
  struct {
    hipStream_t stream;
    hipStreamCaptureStatus *pCaptureStatus;
    unsigned long long *pId;
  } hipStreamGetCaptureInfo_spt;
  struct {
    hipStream_t stream;
    hipStreamCaptureStatus *captureStatus_out;
    unsigned long long *id_out;
    hipGraph_t *graph_out;
    const hipGraphNode_t **dependencies_out;
    size_t *numDependencies_out;
  } hipStreamGetCaptureInfo_v2_spt;
  struct {
    hipStream_t stream;
    hipHostFn_t fn;
    void *userData;
  } hipLaunchHostFunc_spt;
  struct {
    hipStream_t stream;
  } hipGetStreamDeviceId;
  struct {
    hipGraphNode_t *phGraphNode;
    hipGraph_t hGraph;
    const hipGraphNode_t *dependencies;
    size_t numDependencies;
#if HIP_RUNTIME_API_TABLE_STEP_VERSION < 13
    const HIP_MEMSET_NODE_PARAMS *memsetParams;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 13
    const hipMemsetParams *memsetParams;
#endif
    hipCtx_t ctx;
  } hipDrvGraphAddMemsetNode;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    const hipExternalSemaphoreWaitNodeParams *nodeParams;
  } hipGraphAddExternalSemaphoresWaitNode;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    const hipExternalSemaphoreSignalNodeParams *nodeParams;
  } hipGraphAddExternalSemaphoresSignalNode;
  struct {
    hipGraphNode_t hNode;
    const hipExternalSemaphoreSignalNodeParams *nodeParams;
  } hipGraphExternalSemaphoresSignalNodeSetParams;
  struct {
    hipGraphNode_t hNode;
    const hipExternalSemaphoreWaitNodeParams *nodeParams;
  } hipGraphExternalSemaphoresWaitNodeSetParams;
  struct {
    hipGraphNode_t hNode;
    hipExternalSemaphoreSignalNodeParams *params_out;
  } hipGraphExternalSemaphoresSignalNodeGetParams;
  struct {
    hipGraphNode_t hNode;
    hipExternalSemaphoreWaitNodeParams *params_out;
  } hipGraphExternalSemaphoresWaitNodeGetParams;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    const hipExternalSemaphoreSignalNodeParams *nodeParams;
  } hipGraphExecExternalSemaphoresSignalNodeSetParams;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    const hipExternalSemaphoreWaitNodeParams *nodeParams;
  } hipGraphExecExternalSemaphoresWaitNodeSetParams;
  struct {
    hipGraphNode_t *pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t *pDependencies;
    size_t numDependencies;
    hipGraphNodeParams *nodeParams;
  } hipGraphAddNode;
  struct {
    hipGraphExec_t *pGraphExec;
    hipGraph_t graph;
    hipGraphInstantiateParams *instantiateParams;
  } hipGraphInstantiateWithParams;
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_hip_api_no_args struct to fix this
    rocprofiler_hip_api_no_args no_args;
  } hipExtGetLastError;
  struct {
    float *pBorderColor;
    const textureReference *texRef;
  } hipTexRefGetBorderColor;
  struct {
    hipArray_t *pArray;
    const textureReference *texRef;
  } hipTexRefGetArray;
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 1
  struct {
    const char *symbol;
    void **pfn;
    int hipVersion;
    uint64_t flags;
    hipDriverProcAddressQueryResult *symbolStatus;
  } hipGetProcAddress;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 2
  struct {
    hipStream_t stream;
    hipGraph_t graph;
    const hipGraphNode_t *dependencies;
    const hipGraphEdgeData *dependencyData;
    size_t numDependencies;
    hipStreamCaptureMode mode;
  } hipStreamBeginCaptureToGraph;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 3
  struct {
    hipFunction_t *functionPtr;
    const void *symbolPtr;
  } hipGetFuncBySymbol;
  struct {
    int *device_arr;
    int len;
  } hipSetValidDevices;
  struct {
    hipDeviceptr_t dstDevice;
    hipArray_t srcArray;
    size_t srcOffset;
    size_t ByteCount;
  } hipMemcpyAtoD;
  struct {
    hipArray_t dstArray;
    size_t dstOffset;
    hipDeviceptr_t srcDevice;
    size_t ByteCount;
  } hipMemcpyDtoA;
  struct {
    hipArray_t dstArray;
    size_t dstOffset;
    hipArray_t srcArray;
    size_t srcOffset;
    size_t ByteCount;
  } hipMemcpyAtoA;
  struct {
    void *dstHost;
    hipArray_t srcArray;
    size_t srcOffset;
    size_t ByteCount;
    hipStream_t stream;
  } hipMemcpyAtoHAsync;
  struct {
    hipArray_t dstArray;
    size_t dstOffset;
    const void *srcHost;
    size_t ByteCount;
    hipStream_t stream;
  } hipMemcpyHtoAAsync;
  struct {
    hipArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    hipArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
  } hipMemcpy2DArrayToArray;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 4
  struct {
    hipGraphNode_t *phGraphNode;
    hipGraph_t hGraph;
    const hipGraphNode_t *dependencies;
    size_t numDependencies;
    hipDeviceptr_t dptr;
  } hipDrvGraphAddMemFreeNode;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    const HIP_MEMCPY3D *copyParams;
    hipCtx_t ctx;
  } hipDrvGraphExecMemcpyNodeSetParams;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
#if HIP_RUNTIME_API_TABLE_STEP_VERSION < 13
    const HIP_MEMSET_NODE_PARAMS *memsetParams;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 13
    const hipMemsetParams *memsetParams;
#endif
    hipCtx_t ctx;
  } hipDrvGraphExecMemsetNodeSetParams;
  struct {
    hipGraphExec_t graphExec;
    unsigned long long *flags;
  } hipGraphExecGetFlags;
  struct {
    hipGraphNode_t node;
    hipGraphNodeParams *nodeParams;
  } hipGraphNodeSetParams;
  struct {
    hipGraphExec_t graphExec;
    hipGraphNode_t node;
    hipGraphNodeParams *nodeParams;
  } hipGraphExecNodeSetParams;
  struct {
    hipMipmappedArray_t *mipmap;
    hipExternalMemory_t extMem;
    const hipExternalMemoryMipmappedArrayDesc *mipmapDesc;
  } hipExternalMemoryGetMappedMipmappedArray;
  struct {
    hipGraphNode_t hNode;
    HIP_MEMCPY3D *nodeParams;
  } hipDrvGraphMemcpyNodeGetParams;
  struct {
    hipGraphNode_t hNode;
    const HIP_MEMCPY3D *nodeParams;
  } hipDrvGraphMemcpyNodeSetParams;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 5
  struct {
    void **ptr;
    size_t size;
    unsigned int flags;
  } hipExtHostAlloc;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 6
  struct {
    size_t *maxWidthInElements;
    const hipChannelFormatDesc *fmtDesc;
    int device;
  } hipDeviceGetTexture1DLinearMaxWidth;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 7
  struct {
    hipStream_t stream;
    unsigned int count;
    hipStreamBatchMemOpParams *paramArray;
    unsigned int flags;
  } hipStreamBatchMemOp;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 8
  struct {
    hipGraphNode_t *phGraphNode;
    hipGraph_t hGraph;
    const hipGraphNode_t *dependencies;
    size_t numDependencies;
    const hipBatchMemOpNodeParams *nodeParams;
  } hipGraphAddBatchMemOpNode;
  struct {
    hipGraphNode_t hNode;
    hipBatchMemOpNodeParams *nodeParams_out;
  } hipGraphBatchMemOpNodeGetParams;
  struct {
    hipGraphNode_t hNode;
    hipBatchMemOpNodeParams *nodeParams;
  } hipGraphBatchMemOpNodeSetParams;
  struct {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    const hipBatchMemOpNodeParams *nodeParams;
  } hipGraphExecBatchMemOpNodeSetParams;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 9
  struct {
    hipLinkState_t state;
    hipJitInputType type;
    void *data;
    size_t size;
    const char *name;
    unsigned int numOptions;
    hipJitOption *options;
    void **optionValues;
  } hipLinkAddData;
  struct {
    hipLinkState_t state;
    hipJitInputType type;
    const char *path;
    unsigned int numOptions;
    hipJitOption *options;
    void **optionValues;
  } hipLinkAddFile;
  struct {
    hipLinkState_t state;
    void **hipBinOut;
    size_t *sizeOut;
  } hipLinkComplete;
  struct {
    unsigned int numOptions;
    hipJitOption *options;
    void **optionValues;
    hipLinkState_t *stateOut;
  } hipLinkCreate;
  struct {
    hipLinkState_t state;
  } hipLinkDestroy;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 10
  struct {
    hipEvent_t event;
    hipStream_t stream;
    unsigned int flags;
  } hipEventRecordWithFlags;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 11
  struct {
    const hipLaunchConfig_t *config;
    const void *fPtr;
    void **args;
  } hipLaunchKernelExC;
  struct {
    const HIP_LAUNCH_CONFIG *config;
    hipFunction_t f;
    void **params;
    void **extra;
  } hipDrvLaunchKernelEx;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 12
  struct {
    void *handle;
    hipDeviceptr_t dptr;
    size_t size;
    hipMemRangeHandleType handleType;
    unsigned long long flags;
  } hipMemGetHandleForAddressRange;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 14
  struct {
    unsigned int *count;
    hipModule_t mod;
  } hipModuleGetFunctionCount;
  struct {
    hipDeviceptr_t dst;
    size_t dstPitch;
    unsigned char value;
    size_t width;
    size_t height;
  } hipMemsetD2D8;
  struct {
    hipDeviceptr_t dst;
    size_t dstPitch;
    unsigned char value;
    size_t width;
    size_t height;
    hipStream_t stream;
  } hipMemsetD2D8Async;
  struct {
    hipDeviceptr_t dst;
    size_t dstPitch;
    unsigned short value;
    size_t width;
    size_t height;
  } hipMemsetD2D16;
  struct {
    hipDeviceptr_t dst;
    size_t dstPitch;
    unsigned short value;
    size_t width;
    size_t height;
    hipStream_t stream;
  } hipMemsetD2D16Async;
  struct {
    hipDeviceptr_t dst;
    size_t dstPitch;
    unsigned int value;
    size_t width;
    size_t height;
  } hipMemsetD2D32;
  struct {
    hipDeviceptr_t dst;
    size_t dstPitch;
    unsigned int value;
    size_t width;
    size_t height;
    hipStream_t stream;
  } hipMemsetD2D32Async;
  struct {
    hipStream_t stream;
    hipLaunchAttributeID attr;
    const hipLaunchAttributeValue *value_out;
  } hipStreamGetAttribute;
  struct {
    hipStream_t stream;
    hipLaunchAttributeID attr;
    const hipLaunchAttributeValue *value;
  } hipStreamSetAttribute;
  struct {
    hipModule_t *module;
    const void *fatbin;
  } hipModuleLoadFatBinary;
  struct {
    void **dsts;
    void **srcs;
    size_t *sizes;
    size_t count;
    hipMemcpyAttributes *attrs;
    size_t *attrsIdxs;
    size_t numAttrs;
    size_t *failIdx;
    hipStream_t stream;
  } hipMemcpyBatchAsync;
  struct {
    size_t numOps;
    hipMemcpy3DBatchOp *opList;
    size_t *failIdx;
    unsigned long long flags;
    hipStream_t stream;
  } hipMemcpy3DBatchAsync;
  struct {
    hipMemcpy3DPeerParms *p;
  } hipMemcpy3DPeer;
  struct {
    hipMemcpy3DPeerParms *p;
    hipStream_t stream;
  } hipMemcpy3DPeerAsync;
  struct {
    const char *symbol;
    void **funcPtr;
    unsigned long long flags;
    hipDriverEntryPointQueryResult *driverStatus;
  } hipGetDriverEntryPoint;
  struct {
    const char *symbol;
    void **funcPtr;
    unsigned long long flags;
    hipDriverEntryPointQueryResult *driverStatus;
  } hipGetDriverEntryPoint_spt;
  struct {
    const void *dev_ptr;
    size_t count;
    hipMemLocation location;
    unsigned int flags;
    hipStream_t stream;
  } hipMemPrefetchAsync_v2;
  struct {
    const void *dev_ptr;
    size_t count;
    hipMemoryAdvise advice;
    hipMemLocation location;
  } hipMemAdvise_v2;
  struct {
    hipStream_t stream;
    unsigned long long *streamId;
  } hipStreamGetId;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 15
  struct {
    hipLibrary_t *library;
    const void *code;
    hipJitOption *jitOptions;
    void **jitOptionsValues;
    unsigned int numJitOptions;
    hipLibraryOption *libraryOptions;
    void **libraryOptionValues;
    unsigned int numLibraryOptions;
  } hipLibraryLoadData;
  struct {
    hipLibrary_t *library;
    const char *fileName;
    hipJitOption *jitOptions;
    void **jitOptionsValues;
    unsigned int numJitOptions;
    hipLibraryOption *libraryOptions;
    void **libraryOptionValues;
    unsigned int numLibraryOptions;
  } hipLibraryLoadFromFile;
  struct {
    hipLibrary_t library;
  } hipLibraryUnload;
  struct {
    hipKernel_t *pKernel;
    hipLibrary_t library;
    const char *name;
  } hipLibraryGetKernel;
  struct {
    unsigned int *count;
    hipLibrary_t library;
  } hipLibraryGetKernelCount;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 16
  struct {
    hipStream_t dst;
    hipStream_t src;
  } hipStreamCopyAttributes;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 17
  struct {
    hipKernel_t *kernels;
    unsigned int numKernels;
    hipLibrary_t library;
  } hipLibraryEnumerateKernels;
  struct {
    hipLibrary_t *library;
    hipKernel_t kernel;
  } hipKernelGetLibrary;
  struct {
    const char **name;
    hipKernel_t kernel;
  } hipKernelGetName;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 18
  struct {
    size_t *dynamicSmemSize;
    const void *f;
    int numBlocks;
    int blockSize;
  } hipOccupancyAvailableDynamicSMemPerBlock;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 19
  struct {
    const char *symbol;
    void **pfn;
    int hipVersion;
    uint64_t flags;
    hipDriverProcAddressQueryResult *symbolStatus;
  } hipGetProcAddress_spt;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 20
  struct {
    hipKernel_t kernel;
    size_t paramIndex;
    size_t *paramOffset;
    size_t *paramSize;
  } hipKernelGetParamInfo;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 21
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_hip_api_no_args struct to fix this
    rocprofiler_hip_api_no_args no_args;
  } hipExtDisableLogging;
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_hip_api_no_args struct to fix this
    rocprofiler_hip_api_no_args no_args;
  } hipExtEnableLogging;
  struct {
    size_t log_level;
    size_t log_size;
    size_t log_mask;
  } hipExtSetLoggingParams;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 22
  struct {
    hipMemLocation *location;
    hipMemAllocationType type;
    hipMemPool_t pool;
  } hipMemSetMemPool;
  struct {
    hipMemPool_t *pool;
    hipMemLocation *location;
    hipMemAllocationType type;
  } hipMemGetMemPool;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 23
  struct {
    hipArrayMemoryRequirements *memoryRequirements;
    hipMipmappedArray_t mipmap;
    hipDevice_t device;
  } hipMipmappedArrayGetMemoryRequirements;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 24
  struct {
    int *pi;
    hipFunction_attribute attrib;
    hipKernel_t kernel;
    hipDevice_t dev;
  } hipKernelGetAttribute;
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 25
  struct {
    hipFunction_attribute attrib;
    int value;
    hipKernel_t kernel;
    hipDevice_t dev;
  } hipKernelSetAttribute;
  struct {
    hipFunction_t *pFunc;
    hipKernel_t kernel;
  } hipKernelGetFunction;
#endif
} rocprofiler_hip_api_args_t;

ROCPROFILER_EXTERN_C_FINI
