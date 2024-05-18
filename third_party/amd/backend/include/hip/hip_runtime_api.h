/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * @file hip_runtime_api.h
 *
 * @brief Defines the API signatures for HIP runtime.
 * This file can be compiled with a standard compiler.
 */

#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_HIP_RUNTIME_API_H

#include <string.h>  // for getDeviceProp
#include <hip/hip_version.h>
#include <hip/hip_common.h>

enum {
    HIP_SUCCESS = 0,
    HIP_ERROR_INVALID_VALUE,
    HIP_ERROR_NOT_INITIALIZED,
    HIP_ERROR_LAUNCH_OUT_OF_RESOURCES
};
// hack to get these to show up in Doxygen:
/**
 * @defgroup GlobalDefs Global enum and defines
 * @{
 *
 */
/**
 * hipDeviceArch_t
 *
 */
typedef struct {
    // 32-bit Atomics
    unsigned hasGlobalInt32Atomics : 1;     ///< 32-bit integer atomics for global memory.
    unsigned hasGlobalFloatAtomicExch : 1;  ///< 32-bit float atomic exch for global memory.
    unsigned hasSharedInt32Atomics : 1;     ///< 32-bit integer atomics for shared memory.
    unsigned hasSharedFloatAtomicExch : 1;  ///< 32-bit float atomic exch for shared memory.
    unsigned hasFloatAtomicAdd : 1;  ///< 32-bit float atomic add in global and shared memory.

    // 64-bit Atomics
    unsigned hasGlobalInt64Atomics : 1;  ///< 64-bit integer atomics for global memory.
    unsigned hasSharedInt64Atomics : 1;  ///< 64-bit integer atomics for shared memory.

    // Doubles
    unsigned hasDoubles : 1;  ///< Double-precision floating point.

    // Warp cross-lane operations
    unsigned hasWarpVote : 1;     ///< Warp vote instructions (__any, __all).
    unsigned hasWarpBallot : 1;   ///< Warp ballot instructions (__ballot).
    unsigned hasWarpShuffle : 1;  ///< Warp shuffle operations. (__shfl_*).
    unsigned hasFunnelShift : 1;  ///< Funnel two words into one with shift&mask caps.

    // Sync
    unsigned hasThreadFenceSystem : 1;  ///< __threadfence_system.
    unsigned hasSyncThreadsExt : 1;     ///< __syncthreads_count, syncthreads_and, syncthreads_or.

    // Misc
    unsigned hasSurfaceFuncs : 1;        ///< Surface functions.
    unsigned has3dGrid : 1;              ///< Grid and group dims are 3D (rather than 2D).
    unsigned hasDynamicParallelism : 1;  ///< Dynamic parallelism.
} hipDeviceArch_t;

typedef struct hipUUID_t {
    char bytes[16];
} hipUUID;

//---
// Common headers for both NVCC and HCC paths:

#define hipGetDeviceProperties hipGetDevicePropertiesR0600
#define hipDeviceProp_t hipDeviceProp_tR0600
#define hipChooseDevice hipChooseDeviceR0600

/**
 * hipDeviceProp
 *
 */
typedef struct hipDeviceProp_t {
    char name[256];                   ///< Device name.
    hipUUID uuid;                     ///< UUID of a device
    char luid[8];                     ///< 8-byte unique identifier. Only valid on windows
    unsigned int luidDeviceNodeMask;  ///< LUID node mask
    size_t totalGlobalMem;            ///< Size of global memory region (in bytes).
    size_t sharedMemPerBlock;         ///< Size of shared memory region (in bytes).
    int regsPerBlock;                 ///< Registers per block.
    int warpSize;                     ///< Warp size.
    size_t memPitch;                  ///< Maximum pitch in bytes allowed by memory copies
                                      ///< pitched memory
    int maxThreadsPerBlock;           ///< Max work items per work group or workgroup max size.
    int maxThreadsDim[3];             ///< Max number of threads in each dimension (XYZ) of a block.
    int maxGridSize[3];               ///< Max grid dimensions (XYZ).
    int clockRate;                    ///< Max clock frequency of the multiProcessors in khz.
    size_t totalConstMem;             ///< Size of shared memory region (in bytes).
    int major;  ///< Major compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    int minor;  ///< Minor compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    size_t textureAlignment;       ///< Alignment requirement for textures
    size_t texturePitchAlignment;  ///< Pitch alignment requirement for texture references bound to
    int deviceOverlap;             ///< Deprecated. Use asyncEngineCount instead
    int multiProcessorCount;       ///< Number of multi-processors (compute units).
    int kernelExecTimeoutEnabled;  ///< Run time limit for kernels executed on the device
    int integrated;                ///< APU vs dGPU
    int canMapHostMemory;          ///< Check whether HIP can map host memory
    int computeMode;               ///< Compute mode.
    int maxTexture1D;              ///< Maximum number of elements in 1D images
    int maxTexture1DMipmap;        ///< Maximum 1D mipmap texture size
    int maxTexture1DLinear;        ///< Maximum size for 1D textures bound to linear memory
    int maxTexture2D[2];  ///< Maximum dimensions (width, height) of 2D images, in image elements
    int maxTexture2DMipmap[2];  ///< Maximum number of elements in 2D array mipmap of images
    int maxTexture2DLinear[3];  ///< Maximum 2D tex dimensions if tex are bound to pitched memory
    int maxTexture2DGather[2];  ///< Maximum 2D tex dimensions if gather has to be performed
    int maxTexture3D[3];  ///< Maximum dimensions (width, height, depth) of 3D images, in image
                          ///< elements
    int maxTexture3DAlt[3];           ///< Maximum alternate 3D texture dims
    int maxTextureCubemap;            ///< Maximum cubemap texture dims
    int maxTexture1DLayered[2];       ///< Maximum number of elements in 1D array images
    int maxTexture2DLayered[3];       ///< Maximum number of elements in 2D array images
    int maxTextureCubemapLayered[2];  ///< Maximum cubemaps layered texture dims
    int maxSurface1D;                 ///< Maximum 1D surface size
    int maxSurface2D[2];              ///< Maximum 2D surface size
    int maxSurface3D[3];              ///< Maximum 3D surface size
    int maxSurface1DLayered[2];       ///< Maximum 1D layered surface size
    int maxSurface2DLayered[3];       ///< Maximum 2D layared surface size
    int maxSurfaceCubemap;            ///< Maximum cubemap surface size
    int maxSurfaceCubemapLayered[2];  ///< Maximum cubemap layered surface size
    size_t surfaceAlignment;          ///< Alignment requirement for surface
    int concurrentKernels;         ///< Device can possibly execute multiple kernels concurrently.
    int ECCEnabled;                ///< Device has ECC support enabled
    int pciBusID;                  ///< PCI Bus ID.
    int pciDeviceID;               ///< PCI Device ID.
    int pciDomainID;               ///< PCI Domain ID
    int tccDriver;                 ///< 1:If device is Tesla device using TCC driver, else 0
    int asyncEngineCount;          ///< Number of async engines
    int unifiedAddressing;         ///< Does device and host share unified address space
    int memoryClockRate;           ///< Max global memory clock frequency in khz.
    int memoryBusWidth;            ///< Global memory bus width in bits.
    int l2CacheSize;               ///< L2 cache size.
    int persistingL2CacheMaxSize;  ///< Device's max L2 persisting lines in bytes
    int maxThreadsPerMultiProcessor;    ///< Maximum resident threads per multi-processor.
    int streamPrioritiesSupported;      ///< Device supports stream priority
    int globalL1CacheSupported;         ///< Indicates globals are cached in L1
    int localL1CacheSupported;          ///< Locals are cahced in L1
    size_t sharedMemPerMultiprocessor;  ///< Amount of shared memory available per multiprocessor.
    int regsPerMultiprocessor;          ///< registers available per multiprocessor
    int managedMemory;         ///< Device supports allocating managed memory on this system
    int isMultiGpuBoard;       ///< 1 if device is on a multi-GPU board, 0 if not.
    int multiGpuBoardGroupID;  ///< Unique identifier for a group of devices on same multiboard GPU
    int hostNativeAtomicSupported;         ///< Link between host and device supports native atomics
    int singleToDoublePrecisionPerfRatio;  ///< Deprecated. CUDA only.
    int pageableMemoryAccess;              ///< Device supports coherently accessing pageable memory
                                           ///< without calling hipHostRegister on it
    int concurrentManagedAccess;  ///< Device can coherently access managed memory concurrently with
                                  ///< the CPU
    int computePreemptionSupported;         ///< Is compute preemption supported on the device
    int canUseHostPointerForRegisteredMem;  ///< Device can access host registered memory with same
                                            ///< address as the host
    int cooperativeLaunch;                  ///< HIP device supports cooperative launch
    int cooperativeMultiDeviceLaunch;       ///< HIP device supports cooperative launch on multiple
                                            ///< devices
    size_t
        sharedMemPerBlockOptin;  ///< Per device m ax shared mem per block usable by special opt in
    int pageableMemoryAccessUsesHostPageTables;  ///< Device accesses pageable memory via the host's
                                                 ///< page tables
    int directManagedMemAccessFromHost;  ///< Host can directly access managed memory on the device
                                         ///< without migration
    int maxBlocksPerMultiProcessor;      ///< Max number of blocks on CU
    int accessPolicyMaxWindowSize;       ///< Max value of access policy window
    size_t reservedSharedMemPerBlock;    ///< Shared memory reserved by driver per block
    int hostRegisterSupported;           ///< Device supports hipHostRegister
    int sparseHipArraySupported;         ///< Indicates if device supports sparse hip arrays
    int hostRegisterReadOnlySupported;   ///< Device supports using the hipHostRegisterReadOnly flag
                                         ///< with hipHostRegistger
    int timelineSemaphoreInteropSupported;  ///< Indicates external timeline semaphore support
    int memoryPoolsSupported;  ///< Indicates if device supports hipMallocAsync and hipMemPool APIs
    int gpuDirectRDMASupported;                    ///< Indicates device support of RDMA APIs
    unsigned int gpuDirectRDMAFlushWritesOptions;  ///< Bitmask to be interpreted according to
                                                   ///< hipFlushGPUDirectRDMAWritesOptions
    int gpuDirectRDMAWritesOrdering;               ///< value of hipGPUDirectRDMAWritesOrdering
    unsigned int
        memoryPoolSupportedHandleTypes;  ///< Bitmask of handle types support with mempool based IPC
    int deferredMappingHipArraySupported;  ///< Device supports deferred mapping HIP arrays and HIP
                                           ///< mipmapped arrays
    int ipcEventSupported;                 ///< Device supports IPC events
    int clusterLaunch;                     ///< Device supports cluster launch
    int unifiedFunctionPointers;           ///< Indicates device supports unified function pointers
    int reserved[63];                      ///< CUDA Reserved.

    int hipReserved[32];  ///< Reserved for adding new entries for HIP/CUDA.

    /* HIP Only struct members */
    char gcnArchName[256];                    ///< AMD GCN Arch Name. HIP Only.
    size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per CU. HIP Only.
    int clockInstructionRate;  ///< Frequency in khz of the timer used by the device-side "clock*"
                               ///< instructions.  New for HIP.
    hipDeviceArch_t arch;      ///< Architectural feature flags.  New for HIP.
    unsigned int* hdpMemFlushCntl;            ///< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    unsigned int* hdpRegFlushCntl;            ///< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    int cooperativeMultiDeviceUnmatchedFunc;  ///< HIP device supports cooperative launch on
                                              ///< multiple
                                              /// devices with unmatched functions
    int cooperativeMultiDeviceUnmatchedGridDim;    ///< HIP device supports cooperative launch on
                                                   ///< multiple
                                                   /// devices with unmatched grid dimensions
    int cooperativeMultiDeviceUnmatchedBlockDim;   ///< HIP device supports cooperative launch on
                                                   ///< multiple
                                                   /// devices with unmatched block dimensions
    int cooperativeMultiDeviceUnmatchedSharedMem;  ///< HIP device supports cooperative launch on
                                                   ///< multiple
                                                   /// devices with unmatched shared memories
    int isLargeBar;                                ///< 1: if it is a large PCI bar device, else 0
    int asicRevision;                              ///< Revision of the GPU in this device
} hipDeviceProp_t;

 /**
 * hipMemoryType (for pointer attributes)
 *
 * @note hipMemoryType enum values are combination of cudaMemoryType and cuMemoryType and AMD specific enum values.
 *
 */
typedef enum hipMemoryType {
    hipMemoryTypeUnregistered = 0,  ///< Unregistered memory
    hipMemoryTypeHost         = 1,  ///< Memory is physically located on host
    hipMemoryTypeDevice       = 2,  ///< Memory is physically located on device. (see deviceId for
                                    ///< specific device)
    hipMemoryTypeManaged      = 3,  ///< Managed memory, automaticallly managed by the unified
                                    ///< memory system
                                    ///< place holder for new values.
    hipMemoryTypeArray        = 10, ///< Array memory, physically located on device. (see deviceId for
                                    ///< specific device)
    hipMemoryTypeUnified      = 11  ///< unified address space

} hipMemoryType;

/**
 * Pointer attributes
 */
typedef struct hipPointerAttribute_t {
    enum hipMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    unsigned allocationFlags; /* flags specified when memory was allocated*/
    /* peers? */
} hipPointerAttribute_t;

// Ignoring error-code return values from hip APIs is discouraged. On C++17,
// we can make that yield a warning
#if __cplusplus >= 201703L
#define __HIP_NODISCARD [[nodiscard]]
#else
#define __HIP_NODISCARD
#endif

/**
 * HIP error type
 *
 */
// Developer note - when updating these, update the hipErrorName and hipErrorString functions in
// NVCC and HCC paths Also update the hipCUDAErrorTohipError function in NVCC path.

typedef enum __HIP_NODISCARD hipError_t {
    hipSuccess = 0,  ///< Successful completion.
    hipErrorInvalidValue = 1,  ///< One or more of the parameters passed to the API call is NULL
                               ///< or not in an acceptable range.
    hipErrorOutOfMemory = 2,   ///< out of memory range.
    // Deprecated
    hipErrorMemoryAllocation = 2,  ///< Memory allocation error.
    hipErrorNotInitialized = 3,    ///< Invalid not initialized
    // Deprecated
    hipErrorInitializationError = 3,
    hipErrorDeinitialized = 4,      ///< Deinitialized
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,  ///< Invalide configuration
    hipErrorInvalidPitchValue = 12,   ///< Invalid pitch value
    hipErrorInvalidSymbol = 13,   ///< Invalid symbol
    hipErrorInvalidDevicePointer = 17,  ///< Invalid Device Pointer
    hipErrorInvalidMemcpyDirection = 21,  ///< Invalid memory copy direction
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,  ///< Invalid device function
    hipErrorNoDevice = 100,  ///< Call to hipGetDeviceCount returned 0 devices
    hipErrorInvalidDevice = 101,  ///< DeviceID must be in range from 0 to compute-devices.
    hipErrorInvalidImage = 200,   ///< Invalid image
    hipErrorInvalidContext = 201,  ///< Produced when input context is invalid.
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    // Deprecated
    hipErrorMapBufferObjectFailed = 205,  ///< Produced when the IPC memory attach failed from ROCr.
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,   ///< Unsupported limit
    hipErrorContextAlreadyInUse = 216,   ///< The context is already in use
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,  ///< In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,   ///< Invalid source.
    hipErrorFileNotFound = 301,   ///< the file is not found.
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,   ///< Failed to initialize shared object.
    hipErrorOperatingSystem = 304,   ///< Not the correct operating system
    hipErrorInvalidHandle = 400,  ///< Invalide handle
    // Deprecated
    hipErrorInvalidResourceHandle = 400,  ///< Resource handle (hipEvent_t or hipStream_t) invalid.
    hipErrorIllegalState = 401, ///< Resource required is not in a valid state to perform operation.
    hipErrorNotFound = 500,   ///< Not found
    hipErrorNotReady = 600,  ///< Indicates that asynchronous operations enqueued earlier are not
                             ///< ready.  This is not actually an error, but is used to distinguish
                             ///< from hipSuccess (which indicates completion).  APIs that return
                             ///< this error include hipEventQuery and hipStreamQuery.
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,  ///< Out of resources error.
    hipErrorLaunchTimeOut = 702,   ///< Timeout for the launch.
    hipErrorPeerAccessAlreadyEnabled = 704,  ///< Peer access was already enabled from the current
                                             ///< device.
    hipErrorPeerAccessNotEnabled = 705,  ///< Peer access was never enabled from the current device.
    hipErrorSetOnActiveProcess = 708,   ///< The process is active.
    hipErrorContextIsDestroyed = 709,   ///< The context is already destroyed
    hipErrorAssert = 710,  ///< Produced when the kernel calls assert.
    hipErrorHostMemoryAlreadyRegistered = 712,  ///< Produced when trying to lock a page-locked
                                                ///< memory.
    hipErrorHostMemoryNotRegistered = 713,  ///< Produced when trying to unlock a non-page-locked
                                            ///< memory.
    hipErrorLaunchFailure = 719,  ///< An exception occurred on the device while executing a kernel.
    hipErrorCooperativeLaunchTooLarge = 720,  ///< This error indicates that the number of blocks
                                              ///< launched per grid for a kernel that was launched
                                              ///< via cooperative launch APIs exceeds the maximum
                                              ///< number of allowed blocks for the current device.
    hipErrorNotSupported = 801,  ///< Produced when the hip API is not supported/implemented
    hipErrorStreamCaptureUnsupported = 900,  ///< The operation is not permitted when the stream
                                             ///< is capturing.
    hipErrorStreamCaptureInvalidated = 901,  ///< The current capture sequence on the stream
                                             ///< has been invalidated due to a previous error.
    hipErrorStreamCaptureMerge = 902,  ///< The operation would have resulted in a merge of
                                       ///< two independent capture sequences.
    hipErrorStreamCaptureUnmatched = 903,  ///< The capture was not initiated in this stream.
    hipErrorStreamCaptureUnjoined = 904,  ///< The capture sequence contains a fork that was not
                                          ///< joined to the primary stream.
    hipErrorStreamCaptureIsolation = 905,  ///< A dependency would have been created which crosses
                                           ///< the capture sequence boundary. Only implicit
                                           ///< in-stream ordering dependencies  are allowed
                                           ///< to cross the boundary
    hipErrorStreamCaptureImplicit = 906,  ///< The operation would have resulted in a disallowed
                                          ///< implicit dependency on a current capture sequence
                                          ///< from hipStreamLegacy.
    hipErrorCapturedEvent = 907,  ///< The operation is not permitted on an event which was last
                                  ///< recorded in a capturing stream.
    hipErrorStreamCaptureWrongThread = 908,  ///< A stream capture sequence not initiated with
                                             ///< the hipStreamCaptureModeRelaxed argument to
                                             ///< hipStreamBeginCapture was passed to
                                             ///< hipStreamEndCapture in a different thread.
    hipErrorGraphExecUpdateFailure = 910,  ///< This error indicates that the graph update
                                           ///< not performed because it included changes which
                                           ///< violated constraintsspecific to instantiated graph
                                           ///< update.
    hipErrorUnknown = 999,  ///< Unknown error.
    // HSA Runtime Error Codes start here.
    hipErrorRuntimeMemory = 1052,  ///< HSA runtime memory call returned error.  Typically not seen
                                   ///< in production systems.
    hipErrorRuntimeOther = 1053,  ///< HSA runtime call other than memory returned error.  Typically
                                  ///< not seen in production systems.
    hipErrorTbd  ///< Marker that more error codes are needed.
} hipError_t;

#undef __HIP_NODISCARD

/**
 * hipDeviceAttribute_t
 * hipDeviceAttributeUnused number: 5
 */
typedef enum hipDeviceAttribute_t {
    hipDeviceAttributeCudaCompatibleBegin = 0,

    hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin, ///< Whether ECC support is enabled.
    hipDeviceAttributeAccessPolicyMaxWindowSize,        ///< Cuda only. The maximum size of the window policy in bytes.
    hipDeviceAttributeAsyncEngineCount,                 ///< Asynchronous engines number.
    hipDeviceAttributeCanMapHostMemory,                 ///< Whether host memory can be mapped into device address space
    hipDeviceAttributeCanUseHostPointerForRegisteredMem,///< Device can access host registered memory
                                                        ///< at the same virtual address as the CPU
    hipDeviceAttributeClockRate,                        ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeComputeMode,                      ///< Compute mode that device is currently in.
    hipDeviceAttributeComputePreemptionSupported,       ///< Device supports Compute Preemption.
    hipDeviceAttributeConcurrentKernels,                ///< Device can possibly execute multiple kernels concurrently.
    hipDeviceAttributeConcurrentManagedAccess,          ///< Device can coherently access managed memory concurrently with the CPU
    hipDeviceAttributeCooperativeLaunch,                ///< Support cooperative launch
    hipDeviceAttributeCooperativeMultiDeviceLaunch,     ///< Support cooperative launch on multiple devices
    hipDeviceAttributeDeviceOverlap,                    ///< Device can concurrently copy memory and execute a kernel.
                                                        ///< Deprecated. Use instead asyncEngineCount.
    hipDeviceAttributeDirectManagedMemAccessFromHost,   ///< Host can directly access managed memory on
                                                        ///< the device without migration
    hipDeviceAttributeGlobalL1CacheSupported,           ///< Device supports caching globals in L1
    hipDeviceAttributeHostNativeAtomicSupported,        ///< Link between the device and the host supports native atomic operations
    hipDeviceAttributeIntegrated,                       ///< Device is integrated GPU
    hipDeviceAttributeIsMultiGpuBoard,                  ///< Multiple GPU devices.
    hipDeviceAttributeKernelExecTimeout,                ///< Run time limit for kernels executed on the device
    hipDeviceAttributeL2CacheSize,                      ///< Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
    hipDeviceAttributeLocalL1CacheSupported,            ///< caching locals in L1 is supported
    hipDeviceAttributeLuid,                             ///< 8-byte locally unique identifier in 8 bytes. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeLuidDeviceNodeMask,               ///< Luid device node mask. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeComputeCapabilityMajor,           ///< Major compute capability version number.
    hipDeviceAttributeManagedMemory,                    ///< Device supports allocating managed memory on this system
    hipDeviceAttributeMaxBlocksPerMultiProcessor,       ///< Max block size per multiprocessor
    hipDeviceAttributeMaxBlockDimX,                     ///< Max block size in width.
    hipDeviceAttributeMaxBlockDimY,                     ///< Max block size in height.
    hipDeviceAttributeMaxBlockDimZ,                     ///< Max block size in depth.
    hipDeviceAttributeMaxGridDimX,                      ///< Max grid size  in width.
    hipDeviceAttributeMaxGridDimY,                      ///< Max grid size  in height.
    hipDeviceAttributeMaxGridDimZ,                      ///< Max grid size  in depth.
    hipDeviceAttributeMaxSurface1D,                     ///< Maximum size of 1D surface.
    hipDeviceAttributeMaxSurface1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered surface.
    hipDeviceAttributeMaxSurface2D,                     ///< Maximum dimension (width, height) of 2D surface.
    hipDeviceAttributeMaxSurface2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered surface.
    hipDeviceAttributeMaxSurface3D,                     ///< Maximum dimension (width, height, depth) of 3D surface.
    hipDeviceAttributeMaxSurfaceCubemap,                ///< Cuda only. Maximum dimensions of Cubemap surface.
    hipDeviceAttributeMaxSurfaceCubemapLayered,         ///< Cuda only. Maximum dimension of Cubemap layered surface.
    hipDeviceAttributeMaxTexture1DWidth,                ///< Maximum size of 1D texture.
    hipDeviceAttributeMaxTexture1DLayered,              ///< Maximum dimensions of 1D layered texture.
    hipDeviceAttributeMaxTexture1DLinear,               ///< Maximum number of elements allocatable in a 1D linear texture.
                                                        ///< Use cudaDeviceGetTexture1DLinearMaxWidth() instead on Cuda.
    hipDeviceAttributeMaxTexture1DMipmap,               ///< Maximum size of 1D mipmapped texture.
    hipDeviceAttributeMaxTexture2DWidth,                ///< Maximum dimension width of 2D texture.
    hipDeviceAttributeMaxTexture2DHeight,               ///< Maximum dimension hight of 2D texture.
    hipDeviceAttributeMaxTexture2DGather,               ///< Maximum dimensions of 2D texture if gather operations  performed.
    hipDeviceAttributeMaxTexture2DLayered,              ///< Maximum dimensions of 2D layered texture.
    hipDeviceAttributeMaxTexture2DLinear,               ///< Maximum dimensions (width, height, pitch) of 2D textures bound to pitched memory.
    hipDeviceAttributeMaxTexture2DMipmap,               ///< Maximum dimensions of 2D mipmapped texture.
    hipDeviceAttributeMaxTexture3DWidth,                ///< Maximum dimension width of 3D texture.
    hipDeviceAttributeMaxTexture3DHeight,               ///< Maximum dimension height of 3D texture.
    hipDeviceAttributeMaxTexture3DDepth,                ///< Maximum dimension depth of 3D texture.
    hipDeviceAttributeMaxTexture3DAlt,                  ///< Maximum dimensions of alternate 3D texture.
    hipDeviceAttributeMaxTextureCubemap,                ///< Maximum dimensions of Cubemap texture
    hipDeviceAttributeMaxTextureCubemapLayered,         ///< Maximum dimensions of Cubemap layered texture.
    hipDeviceAttributeMaxThreadsDim,                    ///< Maximum dimension of a block
    hipDeviceAttributeMaxThreadsPerBlock,               ///< Maximum number of threads per block.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,      ///< Maximum resident threads per multiprocessor.
    hipDeviceAttributeMaxPitch,                         ///< Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeMemoryBusWidth,                   ///< Global memory bus width in bits.
    hipDeviceAttributeMemoryClockRate,                  ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeComputeCapabilityMinor,           ///< Minor compute capability version number.
    hipDeviceAttributeMultiGpuBoardGroupID,             ///< Unique ID of device group on the same multi-GPU board
    hipDeviceAttributeMultiprocessorCount,              ///< Number of multiprocessors on the device.
    hipDeviceAttributeUnused1,                          ///< Previously hipDeviceAttributeName
    hipDeviceAttributePageableMemoryAccess,             ///< Device supports coherently accessing pageable memory
                                                        ///< without calling hipHostRegister on it
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables, ///< Device accesses pageable memory via the host's page tables
    hipDeviceAttributePciBusId,                         ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,                      ///< PCI Device ID.
    hipDeviceAttributePciDomainID,                      ///< PCI Domain ID.
    hipDeviceAttributePersistingL2CacheMaxSize,         ///< Maximum l2 persisting lines capacity in bytes
    hipDeviceAttributeMaxRegistersPerBlock,             ///< 32-bit registers available to a thread block. This number is shared
                                                        ///< by all thread blocks simultaneously resident on a multiprocessor.
    hipDeviceAttributeMaxRegistersPerMultiprocessor,    ///< 32-bit registers available per block.
    hipDeviceAttributeReservedSharedMemPerBlock,        ///< Shared memory reserved by CUDA driver per block.
    hipDeviceAttributeMaxSharedMemoryPerBlock,          ///< Maximum shared memory available per block in bytes.
    hipDeviceAttributeSharedMemPerBlockOptin,           ///< Maximum shared memory per block usable by special opt in.
    hipDeviceAttributeSharedMemPerMultiprocessor,       ///< Shared memory available per multiprocessor.
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio, ///< Cuda only. Performance ratio of single precision to double precision.
    hipDeviceAttributeStreamPrioritiesSupported,        ///< Whether to support stream priorities.
    hipDeviceAttributeSurfaceAlignment,                 ///< Alignment requirement for surfaces
    hipDeviceAttributeTccDriver,                        ///< Cuda only. Whether device is a Tesla device using TCC driver
    hipDeviceAttributeTextureAlignment,                 ///< Alignment requirement for textures
    hipDeviceAttributeTexturePitchAlignment,            ///< Pitch alignment requirement for 2D texture references bound to pitched memory;
    hipDeviceAttributeTotalConstantMemory,              ///< Constant memory size in bytes.
    hipDeviceAttributeTotalGlobalMem,                   ///< Global memory available on devicice.
    hipDeviceAttributeUnifiedAddressing,                ///< Cuda only. An unified address space shared with the host.
    hipDeviceAttributeUnused2,                          ///< Previously hipDeviceAttributeUuid
    hipDeviceAttributeWarpSize,                         ///< Warp size in threads.
    hipDeviceAttributeMemoryPoolsSupported,             ///< Device supports HIP Stream Ordered Memory Allocator
    hipDeviceAttributeVirtualMemoryManagementSupported, ///< Device supports HIP virtual memory management
    hipDeviceAttributeHostRegisterSupported,            ///< Can device support host memory registration via hipHostRegister
    hipDeviceAttributeMemoryPoolSupportedHandleTypes,   ///< Supported handle mask for HIP Stream Ordered Memory Allocator

    hipDeviceAttributeCudaCompatibleEnd = 9999,
    hipDeviceAttributeAmdSpecificBegin = 10000,

    hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin,  ///< Frequency in khz of the timer used by the device-side "clock*"
    hipDeviceAttributeUnused3,                                  ///< Previously hipDeviceAttributeArch
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,         ///< Maximum Shared Memory PerMultiprocessor.
    hipDeviceAttributeUnused4,                                  ///< Previously hipDeviceAttributeGcnArch
    hipDeviceAttributeUnused5,                                  ///< Previously hipDeviceAttributeGcnArchName
    hipDeviceAttributeHdpMemFlushCntl,                          ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl,                          ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,      ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched functions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,   ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched grid dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,  ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched block dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem, ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched shared memories
    hipDeviceAttributeIsLargeBar,                               ///< Whether it is LargeBar
    hipDeviceAttributeAsicRevision,                             ///< Revision of the GPU in this device
    hipDeviceAttributeCanUseStreamWaitValue,                    ///< '1' if Device supports hipStreamWaitValue32() and
                                                                ///< hipStreamWaitValue64(), '0' otherwise.
    hipDeviceAttributeImageSupport,                             ///< '1' if Device supports image, '0' otherwise.
    hipDeviceAttributePhysicalMultiProcessorCount,              ///< All available physical compute
                                                                ///< units for the device
    hipDeviceAttributeFineGrainSupport,                         ///< '1' if Device supports fine grain, '0' otherwise
    hipDeviceAttributeWallClockRate,                            ///< Constant frequency of wall clock in kilohertz.

    hipDeviceAttributeAmdSpecificEnd = 19999,
    hipDeviceAttributeVendorSpecificBegin = 20000,
    // Extended attributes for vendors
} hipDeviceAttribute_t;

enum hipComputeMode {
    hipComputeModeDefault = 0,
    hipComputeModeExclusive = 1,
    hipComputeModeProhibited = 2,
    hipComputeModeExclusiveProcess = 3
};

enum hipFlushGPUDirectRDMAWritesOptions {
  hipFlushGPUDirectRDMAWritesOptionHost = 1 << 0,
  hipFlushGPUDirectRDMAWritesOptionMemOps = 1 << 1
};

enum hipGPUDirectRDMAWritesOrdering {
  hipGPUDirectRDMAWritesOrderingNone = 0,
  hipGPUDirectRDMAWritesOrderingOwner = 100,
  hipGPUDirectRDMAWritesOrderingAllDevices = 200
};

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

#include <stdint.h>
#include <stddef.h>
#ifndef GENERIC_GRID_LAUNCH
#define GENERIC_GRID_LAUNCH 1
#endif
#include <hip/amd_detail/host_defines.h>
#include <hip/driver_types.h>
#include <hip/texture_types.h>
#include <hip/surface_types.h>
#if defined(_MSC_VER)
#define DEPRECATED(msg) __declspec(deprecated(msg))
#else // !defined(_MSC_VER)
#define DEPRECATED(msg) __attribute__ ((deprecated(msg)))
#endif // !defined(_MSC_VER)
#define DEPRECATED_MSG "This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define HIP_LAUNCH_PARAM_END ((void*)0x03)
#ifdef __cplusplus
  #define __dparm(x) \
          = x
#else
  #define __dparm(x)
#endif
#ifdef __GNUC__
#pragma GCC visibility push (default)
#endif
#ifdef __cplusplus
namespace hip_impl {
hipError_t hip_init();
}  // namespace hip_impl
#endif
// Structure definitions:
#ifdef __cplusplus
extern "C" {
#endif
//---
// API-visible structures
typedef struct ihipCtx_t* hipCtx_t;
// Note many APIs also use integer deviceIds as an alternative to the device pointer:
typedef int hipDevice_t;
typedef enum hipDeviceP2PAttr {
  hipDevP2PAttrPerformanceRank = 0,
  hipDevP2PAttrAccessSupported,
  hipDevP2PAttrNativeAtomicSupported,
  hipDevP2PAttrHipArrayAccessSupported
} hipDeviceP2PAttr;
typedef struct ihipStream_t* hipStream_t;
#define hipIpcMemLazyEnablePeerAccess 0x01
#define HIP_IPC_HANDLE_SIZE 64
typedef struct hipIpcMemHandle_st {
    char reserved[HIP_IPC_HANDLE_SIZE];
} hipIpcMemHandle_t;
typedef struct hipIpcEventHandle_st {
    char reserved[HIP_IPC_HANDLE_SIZE];
} hipIpcEventHandle_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
/**
 * HIP memory pool
 */
typedef struct ihipMemPoolHandle_t* hipMemPool_t;

typedef struct hipFuncAttributes {
    int binaryVersion;
    int cacheModeCA;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxDynamicSharedSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int preferredShmemCarveout;
    int ptxVersion;
    size_t sharedSizeBytes;
} hipFuncAttributes;
typedef struct ihipEvent_t* hipEvent_t;

/**
 * hipLimit
 *
 * @note In HIP device limit-related APIs, any input limit value other than those defined in the
 * enum is treated as "UnsupportedLimit" by default.
 */
enum hipLimit_t {
    hipLimitStackSize = 0x0,        ///< Limit of stack size in bytes on the current device, per
                                    ///< thread. The size is in units of 256 dwords, up to the
                                    ///< limit of (128K - 16)
    hipLimitPrintfFifoSize = 0x01,  ///< Size limit in bytes of fifo used by printf call on the
                                    ///< device. Currently not supported
    hipLimitMallocHeapSize = 0x02,  ///< Limit of heap size in bytes on the current device, should
                                    ///< be less than the global memory size on the device
    hipLimitRange                   ///< Supported limit range
};
/**
 * Flags that can be used with hipStreamCreateWithFlags.
 */
//Flags that can be used with hipStreamCreateWithFlags.
/** Default stream creation flags. These are used with hipStreamCreate().*/
#define hipStreamDefault  0x00

/** Stream does not implicitly synchronize with null stream.*/
#define hipStreamNonBlocking 0x01

//Flags that can be used with hipEventCreateWithFlags.
/** Default flags.*/
#define hipEventDefault 0x0

/** Waiting will yield CPU. Power-friendly and usage-friendly but may increase latency.*/
#define hipEventBlockingSync 0x1

/** Disable event's capability to record timing information. May improve performance.*/
#define hipEventDisableTiming  0x2

/** Event can support IPC. hipEventDisableTiming also must be set.*/
#define hipEventInterprocess 0x4

/** Disable performing a system scope sequentially consistent memory fence when the event
 * transitions from recording to recorded.  This can be used for events that are only being
 * used to measure timing, and do not require the event inspection operations
 * (see ::hipEventSynchronize, ::hipEventQuery, and ::hipEventElapsedTime) to synchronize-with
 * the work on which the recorded event (see ::hipEventRecord) is waiting.
 * On some AMD GPU devices this can improve the accuracy of timing measurements by avoiding the
 * cost of cache writeback and invalidation, and the performance impact of those actions on the
 * execution of following work. */
#define hipEventDisableSystemFence 0x20000000

/** Use a device-scope release when recording this event. This flag is useful to obtain more
 * precise timings of commands between events.  The flag is a no-op on CUDA platforms.*/
#define hipEventReleaseToDevice  0x40000000

/** Use a system-scope release when recording this event. This flag is useful to make
 * non-coherent host memory visible to the host. The flag is a no-op on CUDA platforms.*/
#define hipEventReleaseToSystem  0x80000000

//Flags that can be used with hipHostMalloc.
/** Default pinned memory allocation on the host.*/
#define hipHostMallocDefault 0x0

/** Memory is considered allocated by all contexts.*/
#define hipHostMallocPortable 0x1

/** Map the allocation into the address space for the current device. The device pointer
 * can be obtained with #hipHostGetDevicePointer.*/
#define hipHostMallocMapped  0x2

/** Allocates the memory as write-combined. On some system configurations, write-combined allocation
 * may be transferred faster across the PCI Express bus, however, could have low read efficiency by
 * most CPUs. It's a good option for data tranfer from host to device via mapped pinned memory.*/
#define hipHostMallocWriteCombined 0x4

/**
* Host memory allocation will follow numa policy set by user.
* @note  This numa allocation flag is applicable on Linux, under development on Windows.
*/
#define hipHostMallocNumaUser  0x20000000

/** Allocate coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific allocation.*/
#define hipHostMallocCoherent  0x40000000

/** Allocate non-coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific allocation.*/
#define hipHostMallocNonCoherent  0x80000000

/** Memory can be accessed by any stream on any device*/
#define hipMemAttachGlobal  0x01

/** Memory cannot be accessed by any stream on any device.*/
#define hipMemAttachHost    0x02

/** Memory can only be accessed by a single stream on the associated device.*/
#define hipMemAttachSingle  0x04

#define hipDeviceMallocDefault 0x0

/** Memory is allocated in fine grained region of device.*/
#define hipDeviceMallocFinegrained 0x1

/** Memory represents a HSA signal.*/
#define hipMallocSignalMemory 0x2

/** Memory allocated will be uncached. */
#define hipDeviceMallocUncached 0x3

//Flags that can be used with hipHostRegister.
/** Memory is Mapped and Portable.*/
#define hipHostRegisterDefault 0x0

/** Memory is considered registered by all contexts.*/
#define hipHostRegisterPortable 0x1

/** Map the allocation into the address space for the current device. The device pointer
 * can be obtained with #hipHostGetDevicePointer.*/
#define hipHostRegisterMapped  0x2

/** Not supported.*/
#define hipHostRegisterIoMemory 0x4

/** This flag is ignored On AMD devices.*/
#define hipHostRegisterReadOnly 0x08

/** Coarse Grained host memory lock.*/
#define hipExtHostRegisterCoarseGrained 0x8

/** Automatically select between Spin and Yield.*/
#define hipDeviceScheduleAuto 0x0

/** Dedicate a CPU core to spin-wait. Provides lowest latency, but burns a CPU core and may
 * consume more power.*/
#define hipDeviceScheduleSpin  0x1

/** Yield the CPU to the operating system when waiting. May increase latency, but lowers power
 * and is friendlier to other threads in the system.*/
#define hipDeviceScheduleYield  0x2
#define hipDeviceScheduleBlockingSync 0x4
#define hipDeviceScheduleMask 0x7
#define hipDeviceMapHost 0x8
#define hipDeviceLmemResizeToMax 0x10
/** Default HIP array allocation flag.*/
#define hipArrayDefault 0x00
#define hipArrayLayered 0x01
#define hipArraySurfaceLoadStore 0x02
#define hipArrayCubemap 0x04
#define hipArrayTextureGather 0x08
#define hipOccupancyDefault 0x00
#define hipOccupancyDisableCachingOverride 0x01
#define hipCooperativeLaunchMultiDeviceNoPreSync 0x01
#define hipCooperativeLaunchMultiDeviceNoPostSync 0x02
#define hipCpuDeviceId ((int)-1)
#define hipInvalidDeviceId ((int)-2)
//Flags that can be used with hipExtLaunch Set of APIs.
/** AnyOrderLaunch of kernels.*/
#define hipExtAnyOrderLaunch 0x01
// Flags to be used with hipStreamWaitValue32 and hipStreamWaitValue64.
#define hipStreamWaitValueGte 0x0
#define hipStreamWaitValueEq 0x1
#define hipStreamWaitValueAnd 0x2
#define hipStreamWaitValueNor 0x3
// Stream per thread
/** Implicit stream per application thread.*/
#define hipStreamPerThread ((hipStream_t)2)

// Indicates that the external memory object is a dedicated resource
#define hipExternalMemoryDedicated 0x1
/**
 * HIP Memory Advise values
 *
 * @note This memory advise enumeration is used on Linux, not Windows.
 */
typedef enum hipMemoryAdvise {
    hipMemAdviseSetReadMostly = 1,          ///< Data will mostly be read and only occassionally
                                            ///< be written to
    hipMemAdviseUnsetReadMostly = 2,        ///< Undo the effect of hipMemAdviseSetReadMostly
    hipMemAdviseSetPreferredLocation = 3,   ///< Set the preferred location for the data as
                                            ///< the specified device
    hipMemAdviseUnsetPreferredLocation = 4, ///< Clear the preferred location for the data
    hipMemAdviseSetAccessedBy = 5,          ///< Data will be accessed by the specified device
                                            ///< so prevent page faults as much as possible
    hipMemAdviseUnsetAccessedBy = 6,        ///< Let HIP to decide on the page faulting policy
                                            ///< for the specified device
    hipMemAdviseSetCoarseGrain = 100,       ///< The default memory model is fine-grain. That allows
                                            ///< coherent operations between host and device, while
                                            ///< executing kernels. The coarse-grain can be used
                                            ///< for data that only needs to be coherent at dispatch
                                            ///< boundaries for better performance
    hipMemAdviseUnsetCoarseGrain = 101      ///< Restores cache coherency policy back to fine-grain
} hipMemoryAdvise;
/**
 * HIP Coherency Mode
 */
typedef enum hipMemRangeCoherencyMode {
    hipMemRangeCoherencyModeFineGrain = 0,      ///< Updates to memory with this attribute can be
                                                ///< done coherently from all devices
    hipMemRangeCoherencyModeCoarseGrain = 1,    ///< Writes to memory with this attribute can be
                                                ///< performed by a single device at a time
    hipMemRangeCoherencyModeIndeterminate = 2   ///< Memory region queried contains subregions with
                                                ///< both hipMemRangeCoherencyModeFineGrain and
                                                ///< hipMemRangeCoherencyModeCoarseGrain attributes
} hipMemRangeCoherencyMode;
/**
 * HIP range attributes
 */
typedef enum hipMemRangeAttribute {
    hipMemRangeAttributeReadMostly = 1,         ///< Whether the range will mostly be read and
                                                ///< only occassionally be written to
    hipMemRangeAttributePreferredLocation = 2,  ///< The preferred location of the range
    hipMemRangeAttributeAccessedBy = 3,         ///< Memory range has hipMemAdviseSetAccessedBy
                                                ///< set for the specified device
    hipMemRangeAttributeLastPrefetchLocation = 4,///< The last location to where the range was
                                                ///< prefetched
    hipMemRangeAttributeCoherencyMode = 100,    ///< Returns coherency mode
                                                ///< @ref hipMemRangeCoherencyMode for the range
} hipMemRangeAttribute;

/**
 * HIP memory pool attributes
 */
typedef enum hipMemPoolAttr
{
    /**
     * (value type = int)
     * Allow @p hipMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * hip events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     */
    hipMemPoolReuseFollowEventDependencies   = 0x1,
    /**
     * (value type = int)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     */
    hipMemPoolReuseAllowOpportunistic        = 0x2,
    /**
     * (value type = int)
     * Allow @p hipMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by cuFreeAsync (default enabled).
     */
    hipMemPoolReuseAllowInternalDependencies = 0x3,
    /**
     * (value type = uint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     */
    hipMemPoolAttrReleaseThreshold           = 0x4,
    /**
     * (value type = uint64_t)
     * Amount of backing memory currently allocated for the mempool.
     */
    hipMemPoolAttrReservedMemCurrent         = 0x5,
    /**
     * (value type = uint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     */
    hipMemPoolAttrReservedMemHigh            = 0x6,
    /**
     * (value type = uint64_t)
     * Amount of memory from the pool that is currently in use by the application.
     */
    hipMemPoolAttrUsedMemCurrent             = 0x7,
    /**
     * (value type = uint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     */
    hipMemPoolAttrUsedMemHigh                = 0x8
} hipMemPoolAttr;
/**
 * Specifies the type of location
 */
 typedef enum hipMemLocationType {
    hipMemLocationTypeInvalid = 0,
    hipMemLocationTypeDevice = 1    ///< Device location, thus it's HIP device ID
} hipMemLocationType;
/**
 * Specifies a memory location.
 *
 * To specify a gpu, set type = @p hipMemLocationTypeDevice and set id = the gpu's device ID
 */
typedef struct hipMemLocation {
    hipMemLocationType type;  ///< Specifies the location type, which describes the meaning of id
    int id;                   ///< Identifier for the provided location type @p hipMemLocationType
} hipMemLocation;
/**
 * Specifies the memory protection flags for mapping
 *
 */
typedef enum hipMemAccessFlags {
    hipMemAccessFlagsProtNone      = 0,  ///< Default, make the address range not accessible
    hipMemAccessFlagsProtRead      = 1,  ///< Set the address range read accessible
    hipMemAccessFlagsProtReadWrite = 3   ///< Set the address range read-write accessible
} hipMemAccessFlags;
/**
 * Memory access descriptor
 */
typedef struct hipMemAccessDesc {
    hipMemLocation      location; ///< Location on which the accessibility has to change
    hipMemAccessFlags   flags;    ///< Accessibility flags to set
} hipMemAccessDesc;
/**
 * Defines the allocation types
 */
typedef enum hipMemAllocationType {
    hipMemAllocationTypeInvalid = 0x0,
    /** This allocation type is 'pinned', i.e. cannot migrate from its current
      * location while the application is actively using it
      */
    hipMemAllocationTypePinned  = 0x1,
    hipMemAllocationTypeMax     = 0x7FFFFFFF
} hipMemAllocationType;
/**
 * Flags for specifying handle types for memory pool allocations
 *
 */
typedef enum hipMemAllocationHandleType {
    hipMemHandleTypeNone                    = 0x0,  ///< Does not allow any export mechanism
    hipMemHandleTypePosixFileDescriptor     = 0x1,  ///< Allows a file descriptor for exporting. Permitted only on POSIX systems
    hipMemHandleTypeWin32                   = 0x2,  ///< Allows a Win32 NT handle for exporting. (HANDLE)
    hipMemHandleTypeWin32Kmt                = 0x4   ///< Allows a Win32 KMT handle for exporting. (D3DKMT_HANDLE)
} hipMemAllocationHandleType;
/**
 * Specifies the properties of allocations made from the pool.
 */
typedef struct hipMemPoolProps {
    hipMemAllocationType       allocType;   ///< Allocation type. Currently must be specified as @p hipMemAllocationTypePinned
    hipMemAllocationHandleType handleTypes; ///< Handle types that will be supported by allocations from the pool
    hipMemLocation             location;    ///< Location where allocations should reside
    /**
     * Windows-specific LPSECURITYATTRIBUTES required when @p hipMemHandleTypeWin32 is specified
     */
    void*                       win32SecurityAttributes;
    unsigned char               reserved[64]; ///< Reserved for future use, must be 0
} hipMemPoolProps;
/**
 * Opaque data structure for exporting a pool allocation
 */
typedef struct hipMemPoolPtrExportData {
    unsigned char reserved[64];
} hipMemPoolPtrExportData;

/**
 * hipJitOption
 */
typedef enum hipJitOption {
    hipJitOptionMaxRegisters = 0,
    hipJitOptionThreadsPerBlock,
    hipJitOptionWallTime,
    hipJitOptionInfoLogBuffer,
    hipJitOptionInfoLogBufferSizeBytes,
    hipJitOptionErrorLogBuffer,
    hipJitOptionErrorLogBufferSizeBytes,
    hipJitOptionOptimizationLevel,
    hipJitOptionTargetFromContext,
    hipJitOptionTarget,
    hipJitOptionFallbackStrategy,
    hipJitOptionGenerateDebugInfo,
    hipJitOptionLogVerbose,
    hipJitOptionGenerateLineInfo,
    hipJitOptionCacheMode,
    hipJitOptionSm3xOpt,
    hipJitOptionFastCompile,
    hipJitOptionNumOptions
} hipJitOption;
/**
 * @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
 */
typedef enum hipFuncAttribute {
    hipFuncAttributeMaxDynamicSharedMemorySize = 8,
    hipFuncAttributePreferredSharedMemoryCarveout = 9,
    hipFuncAttributeMax
} hipFuncAttribute;
/**
 * @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
 */
typedef enum hipFuncCache_t {
    hipFuncCachePreferNone,    ///< no preference for shared memory or L1 (default)
    hipFuncCachePreferShared,  ///< prefer larger shared memory and smaller L1 cache
    hipFuncCachePreferL1,      ///< prefer larger L1 cache and smaller shared memory
    hipFuncCachePreferEqual,   ///< prefer equal size L1 cache and shared memory
} hipFuncCache_t;
/**
 * @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
 */
typedef enum hipSharedMemConfig {
    hipSharedMemBankSizeDefault,  ///< The compiler selects a device-specific value for the banking.
    hipSharedMemBankSizeFourByte,  ///< Shared mem is banked at 4-bytes intervals and performs best
                                   ///< when adjacent threads access data 4 bytes apart.
    hipSharedMemBankSizeEightByte  ///< Shared mem is banked at 8-byte intervals and performs best
                                   ///< when adjacent threads access data 4 bytes apart.
} hipSharedMemConfig;
/**
 * Struct for data in 3D
 */
typedef struct dim3 {
    uint32_t x;  ///< x
    uint32_t y;  ///< y
    uint32_t z;  ///< z
#ifdef __cplusplus
    constexpr __host__ __device__ dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z){};
#endif
} dim3;
/**
 * struct hipLaunchParams_t
 */
typedef struct hipLaunchParams_t {
    void* func;             ///< Device function symbol
    dim3 gridDim;           ///< Grid dimentions
    dim3 blockDim;          ///< Block dimentions
    void **args;            ///< Arguments
    size_t sharedMem;       ///< Shared memory
    hipStream_t stream;     ///< Stream identifier
} hipLaunchParams;
/**
 * struct hipFunctionLaunchParams_t
 */
typedef struct hipFunctionLaunchParams_t {
    hipFunction_t function;      ///< Kernel to launch
    unsigned int gridDimX;       ///< Width(X) of grid in blocks
    unsigned int gridDimY;       ///< Height(Y) of grid in blocks
    unsigned int gridDimZ;       ///< Depth(Z) of grid in blocks
    unsigned int blockDimX;      ///< X dimension of each thread block
    unsigned int blockDimY;      ///< Y dimension of each thread block
    unsigned int blockDimZ;      ///< Z dimension of each thread block
    unsigned int sharedMemBytes; ///< Shared memory
    hipStream_t hStream;         ///< Stream identifier
    void **kernelParams;         ///< Kernel parameters
} hipFunctionLaunchParams;
typedef enum hipExternalMemoryHandleType_enum {
  hipExternalMemoryHandleTypeOpaqueFd = 1,
  hipExternalMemoryHandleTypeOpaqueWin32 = 2,
  hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
  hipExternalMemoryHandleTypeD3D12Heap = 4,
  hipExternalMemoryHandleTypeD3D12Resource = 5,
  hipExternalMemoryHandleTypeD3D11Resource = 6,
  hipExternalMemoryHandleTypeD3D11ResourceKmt = 7,
  hipExternalMemoryHandleTypeNvSciBuf         = 8
} hipExternalMemoryHandleType;
typedef struct hipExternalMemoryHandleDesc_st {
  hipExternalMemoryHandleType type;
  union {
    int fd;
    struct {
      void *handle;
      const void *name;
    } win32;
    const void *nvSciBufObject;
  } handle;
  unsigned long long size;
  unsigned int flags;
  unsigned int reserved[16];
} hipExternalMemoryHandleDesc;
typedef struct hipExternalMemoryBufferDesc_st {
  unsigned long long offset;
  unsigned long long size;
  unsigned int flags;
  unsigned int reserved[16];
} hipExternalMemoryBufferDesc;
typedef struct hipExternalMemoryMipmappedArrayDesc_st {
  unsigned long long offset;
  hipChannelFormatDesc formatDesc;
  hipExtent extent;
  unsigned int flags;
  unsigned int numLevels;
} hipExternalMemoryMipmappedArrayDesc;
typedef void* hipExternalMemory_t;
typedef enum hipExternalSemaphoreHandleType_enum {
  hipExternalSemaphoreHandleTypeOpaqueFd = 1,
  hipExternalSemaphoreHandleTypeOpaqueWin32 = 2,
  hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
  hipExternalSemaphoreHandleTypeD3D12Fence = 4,
  hipExternalSemaphoreHandleTypeD3D11Fence = 5,
  hipExternalSemaphoreHandleTypeNvSciSync = 6,
  hipExternalSemaphoreHandleTypeKeyedMutex = 7,
  hipExternalSemaphoreHandleTypeKeyedMutexKmt = 8,
  hipExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9,
  hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10
} hipExternalSemaphoreHandleType;
typedef struct hipExternalSemaphoreHandleDesc_st {
  hipExternalSemaphoreHandleType type;
  union {
    int fd;
    struct {
      void* handle;
      const void* name;
    } win32;
    const void* NvSciSyncObj;
  } handle;
  unsigned int flags;
  unsigned int reserved[16];
} hipExternalSemaphoreHandleDesc;
typedef void* hipExternalSemaphore_t;
typedef struct hipExternalSemaphoreSignalParams_st {
  struct {
    struct {
      unsigned long long value;
    } fence;
    union {
      void *fence;
      unsigned long long reserved;
    } nvSciSync;
    struct {
      unsigned long long key;
    } keyedMutex;
    unsigned int reserved[12];
  } params;
  unsigned int flags;
  unsigned int reserved[16];
} hipExternalSemaphoreSignalParams;
/**
 * External semaphore wait parameters, compatible with driver type
 */
typedef struct hipExternalSemaphoreWaitParams_st {
  struct {
    struct {
      unsigned long long value;
    } fence;
    union {
      void *fence;
      unsigned long long reserved;
    } nvSciSync;
    struct {
      unsigned long long key;
      unsigned int timeoutMs;
    } keyedMutex;
    unsigned int reserved[10];
  } params;
  unsigned int flags;
  unsigned int reserved[16];
} hipExternalSemaphoreWaitParams;

#if __HIP_HAS_GET_PCH
/**
 * Internal use only. This API may change in the future
 * Pre-Compiled header for online compilation
 */
    void __hipGetPCH(const char** pch, unsigned int*size);
#endif

/**
 * HIP Access falgs for Interop resources.
 */
typedef enum hipGraphicsRegisterFlags {
    hipGraphicsRegisterFlagsNone = 0,
    hipGraphicsRegisterFlagsReadOnly = 1,  ///< HIP will not write to this registered resource
    hipGraphicsRegisterFlagsWriteDiscard =
        2,  ///< HIP will only write and will not read from this registered resource
    hipGraphicsRegisterFlagsSurfaceLoadStore = 4,  ///< HIP will bind this resource to a surface
    hipGraphicsRegisterFlagsTextureGather =
        8  ///< HIP will perform texture gather operations on this registered resource
} hipGraphicsRegisterFlags;

typedef struct _hipGraphicsResource hipGraphicsResource;

typedef hipGraphicsResource* hipGraphicsResource_t;

/**
 * An opaque value that represents a hip graph
 */
typedef struct ihipGraph* hipGraph_t;
/**
 * An opaque value that represents a hip graph node
 */
typedef struct hipGraphNode* hipGraphNode_t;
/**
 * An opaque value that represents a hip graph Exec
 */
typedef struct hipGraphExec* hipGraphExec_t;

/**
 * An opaque value that represents a user obj
 */
typedef struct hipUserObject* hipUserObject_t;


/**
 * hipGraphNodeType
 */
typedef enum hipGraphNodeType {
  hipGraphNodeTypeKernel = 0,             ///< GPU kernel node
  hipGraphNodeTypeMemcpy = 1,             ///< Memcpy node
  hipGraphNodeTypeMemset = 2,             ///< Memset node
  hipGraphNodeTypeHost = 3,               ///< Host (executable) node
  hipGraphNodeTypeGraph = 4,              ///< Node which executes an embedded graph
  hipGraphNodeTypeEmpty = 5,              ///< Empty (no-op) node
  hipGraphNodeTypeWaitEvent = 6,          ///< External event wait node
  hipGraphNodeTypeEventRecord = 7,        ///< External event record node
  hipGraphNodeTypeExtSemaphoreSignal = 8, ///< External Semaphore signal node
  hipGraphNodeTypeExtSemaphoreWait = 9,   ///< External Semaphore wait node
  hipGraphNodeTypeMemAlloc = 10,          ///< Memory alloc node
  hipGraphNodeTypeMemFree = 11,           ///< Memory free node
  hipGraphNodeTypeMemcpyFromSymbol = 12,  ///< MemcpyFromSymbol node
  hipGraphNodeTypeMemcpyToSymbol = 13,    ///< MemcpyToSymbol node
  hipGraphNodeTypeCount
} hipGraphNodeType;

typedef void (*hipHostFn_t)(void* userData);
typedef struct hipHostNodeParams {
  hipHostFn_t fn;
  void* userData;
} hipHostNodeParams;
typedef struct hipKernelNodeParams {
  dim3 blockDim;
  void** extra;
  void* func;
  dim3 gridDim;
  void** kernelParams;
  unsigned int sharedMemBytes;
} hipKernelNodeParams;
typedef struct hipMemsetParams {
  void* dst;
  unsigned int elementSize;
  size_t height;
  size_t pitch;
  unsigned int value;
  size_t width;
} hipMemsetParams;

typedef struct hipMemAllocNodeParams {
    hipMemPoolProps poolProps;          ///< Pool properties, which contain where
                                        ///< the location should reside
    const hipMemAccessDesc* accessDescs;///< The number of memory access descriptors.
                                        ///< Must not be bigger than the number of GPUs
    size_t  accessDescCount;            ///< The number of access descriptors
    size_t  bytesize;                   ///< The size of the requested allocation in bytes
    void*   dptr;                       ///< Returned device address of the allocation
} hipMemAllocNodeParams;

/**
 * Kernel node attributeID
 */
typedef enum hipKernelNodeAttrID {
    hipKernelNodeAttributeAccessPolicyWindow = 1,
    hipKernelNodeAttributeCooperative = 2,
} hipKernelNodeAttrID;
typedef enum hipAccessProperty {
    hipAccessPropertyNormal = 0,
    hipAccessPropertyStreaming  = 1,
    hipAccessPropertyPersisting = 2,
} hipAccessProperty;
typedef struct hipAccessPolicyWindow {
    void* base_ptr;
    hipAccessProperty hitProp;
    float hitRatio;
    hipAccessProperty missProp;
    size_t num_bytes;
} hipAccessPolicyWindow;
typedef union hipKernelNodeAttrValue {
    hipAccessPolicyWindow accessPolicyWindow;
    int cooperative;
} hipKernelNodeAttrValue;

/**
 * Memset node params
 */
typedef struct HIP_MEMSET_NODE_PARAMS {
    hipDeviceptr_t dst;                  ///< Destination pointer on device
    size_t pitch;                        ///< Destination device pointer pitch. Unused if height equals 1
    unsigned int value;                  ///< Value of memset to be set
    unsigned int elementSize;            ///< Element in bytes. Must be 1, 2, or 4.
    size_t width;                        ///< Width of a row
    size_t height;                       ///< Number of rows
} HIP_MEMSET_NODE_PARAMS;

/**
 * Graph execution update result
 */
typedef enum hipGraphExecUpdateResult {
  hipGraphExecUpdateSuccess = 0x0,  ///< The update succeeded
  hipGraphExecUpdateError = 0x1,  ///< The update failed for an unexpected reason which is described
                                  ///< in the return value of the function
  hipGraphExecUpdateErrorTopologyChanged = 0x2,  ///< The update failed because the topology changed
  hipGraphExecUpdateErrorNodeTypeChanged = 0x3,  ///< The update failed because a node type changed
  hipGraphExecUpdateErrorFunctionChanged =
      0x4,  ///< The update failed because the function of a kernel node changed
  hipGraphExecUpdateErrorParametersChanged =
      0x5,  ///< The update failed because the parameters changed in a way that is not supported
  hipGraphExecUpdateErrorNotSupported =
      0x6,  ///< The update failed because something about the node is not supported
  hipGraphExecUpdateErrorUnsupportedFunctionChange = 0x7
} hipGraphExecUpdateResult;

typedef enum hipStreamCaptureMode {
  hipStreamCaptureModeGlobal = 0,
  hipStreamCaptureModeThreadLocal,
  hipStreamCaptureModeRelaxed
} hipStreamCaptureMode;
typedef enum hipStreamCaptureStatus {
  hipStreamCaptureStatusNone = 0,    ///< Stream is not capturing
  hipStreamCaptureStatusActive,      ///< Stream is actively capturing
  hipStreamCaptureStatusInvalidated  ///< Stream is part of a capture sequence that has been
                                     ///< invalidated, but not terminated
} hipStreamCaptureStatus;

typedef enum hipStreamUpdateCaptureDependenciesFlags {
  hipStreamAddCaptureDependencies = 0,  ///< Add new nodes to the dependency set
  hipStreamSetCaptureDependencies,      ///< Replace the dependency set with the new nodes
} hipStreamUpdateCaptureDependenciesFlags;

typedef enum hipGraphMemAttributeType {
  hipGraphMemAttrUsedMemCurrent = 0, ///< Amount of memory, in bytes, currently associated with graphs
  hipGraphMemAttrUsedMemHigh,        ///< High watermark of memory, in bytes, associated with graphs since the last time.
  hipGraphMemAttrReservedMemCurrent, ///< Amount of memory, in bytes, currently allocated for graphs.
  hipGraphMemAttrReservedMemHigh,    ///< High watermark of memory, in bytes, currently allocated for graphs
}hipGraphMemAttributeType;
typedef enum hipUserObjectFlags {
  hipUserObjectNoDestructorSync = 0x1, ///< Destructor execution is not synchronized.
} hipUserObjectFlags;

typedef enum hipUserObjectRetainFlags {
  hipGraphUserObjectMove = 0x1, ///< Add new reference or retain.
} hipUserObjectRetainFlags;

typedef enum hipGraphInstantiateFlags {
  hipGraphInstantiateFlagAutoFreeOnLaunch =
      1,  ///< Automatically free memory allocated in a graph before relaunching.
  hipGraphInstantiateFlagUpload =
      2, ///< Automatically upload the graph after instantiaton.
  hipGraphInstantiateFlagDeviceLaunch  =
      4, ///< Instantiate the graph to be launchable from the device.
  hipGraphInstantiateFlagUseNodePriority =
      8, ///< Run the graph using the per-node priority attributes rather than the priority of the stream it is launched into.
} hipGraphInstantiateFlags;

enum hipGraphDebugDotFlags {
  hipGraphDebugDotFlagsVerbose = 1
      << 0, /**< Output all debug data as if every debug flag is enabled */
  hipGraphDebugDotFlagsKernelNodeParams = 1 << 2, /**< Adds hipKernelNodeParams to output */
  hipGraphDebugDotFlagsMemcpyNodeParams = 1 << 3, /**< Adds hipMemcpy3DParms to output */
  hipGraphDebugDotFlagsMemsetNodeParams = 1 << 4, /**< Adds hipMemsetParams to output */
  hipGraphDebugDotFlagsHostNodeParams = 1 << 5,   /**< Adds hipHostNodeParams to output */
  hipGraphDebugDotFlagsEventNodeParams = 1
      << 6, /**< Adds hipEvent_t handle from record and wait nodes to output */
  hipGraphDebugDotFlagsExtSemasSignalNodeParams = 1
      << 7, /**< Adds hipExternalSemaphoreSignalNodeParams values to output */
  hipGraphDebugDotFlagsExtSemasWaitNodeParams = 1
      << 8, /**< Adds hipExternalSemaphoreWaitNodeParams to output */
  hipGraphDebugDotFlagsKernelNodeAttributes = 1
      << 9, /**< Adds hipKernelNodeAttrID values to output */
  hipGraphDebugDotFlagsHandles = 1
      << 10 /**< Adds node handles and every kernel function handle to output */
};
/**
 * Memory allocation properties
 */
typedef struct hipMemAllocationProp {
    hipMemAllocationType type;                       ///< Memory allocation type
    hipMemAllocationHandleType requestedHandleType;  ///< Requested handle type
    hipMemLocation location;                         ///< Memory location
    void* win32HandleMetaData;                       ///< Metadata for Win32 handles
    struct {
        unsigned char compressionType;               ///< Compression type
        unsigned char gpuDirectRDMACapable;          ///< RDMA capable
        unsigned short usage;                        ///< Usage
    } allocFlags;
} hipMemAllocationProp;

/**
 * External semaphore signal node parameters
 */
typedef struct hipExternalSemaphoreSignalNodeParams {
    ///< Array containing external semaphore handles.
    hipExternalSemaphore_t* extSemArray;
    ///< Array containing parameters of external signal semaphore.
    const hipExternalSemaphoreSignalParams* paramsArray;
    ///< Total number of handles and parameters contained in extSemArray and paramsArray.
    unsigned int numExtSems;
} hipExternalSemaphoreSignalNodeParams;

/**
 * External semaphore wait node parameters
 */
typedef struct hipExternalSemaphoreWaitNodeParams {
    ///< Array containing external semaphore handles.
    hipExternalSemaphore_t* extSemArray;
    ///< Array containing parameters of external wait semaphore.
    const hipExternalSemaphoreWaitParams* paramsArray;
    ///< Total number of handles and parameters contained in extSemArray and paramsArray.
    unsigned int numExtSems;
} hipExternalSemaphoreWaitNodeParams;

/**
 * Generic handle for memory allocation
 */
typedef struct ihipMemGenericAllocationHandle* hipMemGenericAllocationHandle_t;

/**
 * Flags for granularity
 */
typedef enum hipMemAllocationGranularity_flags {
    hipMemAllocationGranularityMinimum     = 0x0,  ///< Minimum granularity
    hipMemAllocationGranularityRecommended = 0x1   ///< Recommended granularity for performance
} hipMemAllocationGranularity_flags;

/**
 * Memory handle type
 */
typedef enum hipMemHandleType {
    hipMemHandleTypeGeneric = 0x0  ///< Generic handle type
} hipMemHandleType;

/**
 * Memory operation types
 */
typedef enum hipMemOperationType {
    hipMemOperationTypeMap   = 0x1,   ///< Map operation
    hipMemOperationTypeUnmap = 0x2    ///< Unmap operation
} hipMemOperationType;

/**
 * Subresource types for sparse arrays
 */
typedef enum hipArraySparseSubresourceType {
    hipArraySparseSubresourceTypeSparseLevel = 0x0, ///< Sparse level
    hipArraySparseSubresourceTypeMiptail     = 0x1  ///< Miptail
} hipArraySparseSubresourceType;

/**
 * Map info for arrays
 */
typedef struct hipArrayMapInfo {
     hipResourceType resourceType;                   ///< Resource type
     union {
         hipMipmappedArray mipmap;
         hipArray_t array;
     } resource;
     hipArraySparseSubresourceType subresourceType;  ///< Sparse subresource type
     union {
         struct {
             unsigned int level;   ///< For mipmapped arrays must be a valid mipmap level. For arrays must be zero
             unsigned int layer;   ///< For layered arrays must be a valid layer index. Otherwise, must be zero
             unsigned int offsetX;                   ///< X offset in elements
             unsigned int offsetY;                   ///< Y offset in elements
             unsigned int offsetZ;                   ///< Z offset in elements
             unsigned int extentWidth;               ///< Width in elements
             unsigned int extentHeight;              ///< Height in elements
             unsigned int extentDepth;               ///< Depth in elements
         } sparseLevel;
         struct {
             unsigned int layer;   ///< For layered arrays must be a valid layer index. Otherwise, must be zero
             unsigned long long offset;              ///< Offset within mip tail
             unsigned long long size;                ///< Extent in bytes
         } miptail;
     } subresource;
     hipMemOperationType memOperationType;           ///< Memory operation type
     hipMemHandleType memHandleType;                 ///< Memory handle type
     union {
         hipMemGenericAllocationHandle_t memHandle;
     } memHandle;
     unsigned long long offset;                      ///< Offset within the memory
     unsigned int deviceBitMask;                     ///< Device ordinal bit mask
     unsigned int flags;                             ///< flags for future use, must be zero now.
     unsigned int reserved[2];                       ///< Reserved for future use, must be zero now.
} hipArrayMapInfo;

/**
 * Memcpy node params
 */
typedef struct hipMemcpyNodeParams {
    int flags;                     ///< Must be zero.
    int reserved[3];               ///< Must be zero.
    hipMemcpy3DParms copyParams;   ///< Params set for the memory copy.
} hipMemcpyNodeParams;

/**
 * Child graph node params
 */
typedef struct hipChildGraphNodeParams {
    hipGraph_t graph; ///< Either the child graph to clone into the node, or
                      ///< a handle to the graph possesed by the node used during query
} hipChildGraphNodeParams;

/**
 * Event record node params
 */
typedef struct hipEventWaitNodeParams {
    hipEvent_t event; ///< Event to wait on
} hipEventWaitNodeParams;

/**
 * Event record node params
 */
typedef struct hipEventRecordNodeParams {
    hipEvent_t event; ///< The event to be recorded when node executes
} hipEventRecordNodeParams;

/**
 * Memory free node params
 */
typedef struct hipMemFreeNodeParams {
    void *dptr; ///< the pointer to be freed
} hipMemFreeNodeParams;

/**
 * Params for different graph nodes
 */
typedef struct hipGraphNodeParams {
    hipGraphNodeType type;
    int reserved0[3];
    union {
        long long                            reserved1[29];
        hipKernelNodeParams                  kernel;
        hipMemcpyNodeParams                  memcpy;
        hipMemsetParams                      memset;
        hipHostNodeParams                    host;
        hipChildGraphNodeParams              graph;
        hipEventWaitNodeParams               eventWait;
        hipEventRecordNodeParams             eventRecord;
        hipExternalSemaphoreSignalNodeParams extSemSignal;
        hipExternalSemaphoreWaitNodeParams   extSemWait;
        hipMemAllocNodeParams                alloc;
        hipMemFreeNodeParams                 free;
    };

    long long reserved2;
} hipGraphNodeParams;
// Doxygen end group GlobalDefs
/**
* @}
*/
/**
 *  @defgroup API HIP API
 *  @{
 *
 *  Defines the HIP API.  See the individual sections for more information.
 */
/**
 *  @defgroup Driver Initialization and Version
 *  @{
 *  This section describes the initializtion and version functions of HIP runtime API.
 *
 */
/**
 * @brief Explicitly initializes the HIP runtime.
 *
 * @param [in] flags  Initialization flag, should be zero.
 *
 * Most HIP APIs implicitly initialize the HIP runtime.
 * This API provides control over the timing of the initialization.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
// TODO-ctx - more description on error codes.
hipError_t hipInit(unsigned int flags);
/**
 * @brief Returns the approximate HIP driver version.
 *
 * @param [out] driverVersion driver version
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning The HIP feature set does not correspond to an exact CUDA SDK driver revision.
 * This function always set *driverVersion to 4 as an approximation though HIP supports
 * some features which were introduced in later CUDA SDK revisions.
 * HIP apps code should not rely on the driver revision number here and should
 * use arch feature flags to test device capabilities or conditional compilation.
 *
 * @see hipRuntimeGetVersion
 */
hipError_t hipDriverGetVersion(int* driverVersion);
/**
 * @brief Returns the approximate HIP Runtime version.
 *
 * @param [out] runtimeVersion HIP runtime version
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning The version definition of HIP runtime is different from CUDA.
 * On AMD platform, the function returns HIP runtime version,
 * while on NVIDIA platform, it returns CUDA runtime version.
 * And there is no mapping/correlation between HIP version and CUDA version.
 *
 * @see hipDriverGetVersion
 */
hipError_t hipRuntimeGetVersion(int* runtimeVersion);
/**
 * @brief Returns a handle to a compute device
 * @param [out] device Handle of device
 * @param [in] ordinal Device ordinal
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice
 */
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);

/**
 * @brief Returns the compute capability of the device
 * @param [out] major Major compute capability version number
 * @param [out] minor Minor compute capability version number
 * @param [in] device Device ordinal
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice
 */
hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);
/**
 * @brief Returns an identifer string for the device.
 * @param [out] name String of the device name
 * @param [in] len Maximum length of string to store in device name
 * @param [in] device Device ordinal
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice
 */
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
/**
 * @brief Returns an UUID for the device.[BETA]
 * @param [out] uuid UUID for the device
 * @param [in] device device ordinal
 *
 * @warning This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotInitialized,
 * #hipErrorDeinitialized
 */
hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device);
/**
 * @brief Returns a value for attribute of link between two devices
 * @param [out] value Pointer of the value for the attrubute
 * @param [in] attr enum of hipDeviceP2PAttr to query
 * @param [in] srcDevice The source device of the link
 * @param [in] dstDevice The destination device of the link
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice
 */
hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr,
                                    int srcDevice, int dstDevice);
/**
 * @brief Returns a PCI Bus Id string for the device, overloaded to take int device ID.
 * @param [out] pciBusId The string of PCI Bus Id format for the device
 * @param [in] len Maximum length of string
 * @param [in] device The device ordinal
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice
 */
hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device);
/**
 * @brief Returns a handle to a compute device.
 * @param [out] device The handle of the device
 * @param [in] pciBusId The string of PCI Bus Id for the device
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 */
hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId);
/**
 * @brief Returns the total amount of memory on the device.
 * @param [out] bytes The size of memory in bytes, on the device
 * @param [in] device The ordinal of the device
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice
 */
hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);
// doxygen end initialization
/**
 * @}
 */
/**
 *  @defgroup Device Device Management
 *  @{
 *  This section describes the device management functions of HIP runtime API.
 */
/**
 * @brief Waits on all active streams on current device
 *
 * When this command is invoked, the host thread gets blocked until all the commands associated
 * with streams associated with the device. HIP does not support multiple blocking modes (yet!).
 *
 * @returns #hipSuccess
 *
 * @see hipSetDevice, hipDeviceReset
 */
hipError_t hipDeviceSynchronize(void);
/**
 * @brief The state of current device is discarded and updated to a fresh state.
 *
 * Calling this function deletes all streams created, memory allocated, kernels running, events
 * created. Make sure that no other thread is using the device or streams, memory, kernels, events
 * associated with the current device.
 *
 * @returns #hipSuccess
 *
 * @see hipDeviceSynchronize
 */
hipError_t hipDeviceReset(void);
/**
 * @brief Set default device to be used for subsequent hip API calls from this thread.
 *
 * @param[in] deviceId Valid device in range 0...hipGetDeviceCount().
 *
 * Sets @p device as the default device for the calling host thread.  Valid device id's are 0...
 * (hipGetDeviceCount()-1).
 *
 * Many HIP APIs implicitly use the "default device" :
 *
 * - Any device memory subsequently allocated from this host thread (using hipMalloc) will be
 * allocated on device.
 * - Any streams or events created from this host thread will be associated with device.
 * - Any kernels launched from this host thread (using hipLaunchKernel) will be executed on device
 * (unless a specific stream is specified, in which case the device associated with that stream will
 * be used).
 *
 * This function may be called from any host thread.  Multiple host threads may use the same device.
 * This function does no synchronization with the previous or new device, and has very little
 * runtime overhead. Applications can use hipSetDevice to quickly switch the default device before
 * making a HIP runtime call which uses the default device.
 *
 * The default device is stored in thread-local-storage for each thread.
 * Thread-pool implementations may inherit the default device of the previous thread.  A good
 * practice is to always call hipSetDevice at the start of HIP coding sequency to establish a known
 * standard device.
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorNoDevice
 *
 * @see #hipGetDevice, #hipGetDeviceCount
 */
hipError_t hipSetDevice(int deviceId);
/**
 * @brief Return the default device id for the calling host thread.
 *
 * @param [out] deviceId *device is written with the default device
 *
 * HIP maintains an default device for each thread using thread-local-storage.
 * This device is used implicitly for HIP runtime APIs called by this thread.
 * hipGetDevice returns in * @p device the default device for the calling host thread.
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 * @see hipSetDevice, hipGetDevicesizeBytes
 */
hipError_t hipGetDevice(int* deviceId);
/**
 * @brief Return number of compute-capable devices.
 *
 * @param [out] count Returns number of compute-capable devices.
 *
 * @returns #hipSuccess, #hipErrorNoDevice
 *
 *
 * Returns in @p *count the number of devices that have ability to run compute commands.  If there
 * are no such devices, then @ref hipGetDeviceCount will return #hipErrorNoDevice. If 1 or more
 * devices can be found, then hipGetDeviceCount returns #hipSuccess.
 */
hipError_t hipGetDeviceCount(int* count);
/**
 * @brief Query for a specific device attribute.
 *
 * @param [out] pi pointer to value to return
 * @param [in] attr attribute to query
 * @param [in] deviceId which device to query for information
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 */
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);
/**
 * @brief Returns the default memory pool of the specified device
 *
 * @param [out] mem_pool Default memory pool to return
 * @param [in] device    Device index for query the default memory pool
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
 * hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device);
/**
 * @brief Sets the current memory pool of a device
 *
 * The memory pool must be local to the specified device.
 * @p hipMallocAsync allocates from the current mempool of the provided stream's device.
 * By default, a device's current memory pool is its default memory pool.
 *
 * @note Use @p hipMallocFromPoolAsync for asynchronous memory allocations from a device
 * different than the one the stream runs on.
 *
 * @param [in] device   Device index for the update
 * @param [in] mem_pool Memory pool for update as the current on the specified device
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice, #hipErrorNotSupported
 *
 * @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
 * hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool);
/**
 * @brief Gets the current memory pool for the specified device
 *
 * Returns the last pool provided to @p hipDeviceSetMemPool for this device
 * or the device's default memory pool if @p hipDeviceSetMemPool has never been called.
 * By default the current mempool is the default mempool for a device,
 * otherwise the returned pool must have been set with @p hipDeviceSetMemPool.
 *
 * @param [out] mem_pool Current memory pool on the specified device
 * @param [in] device    Device index to query the current memory pool
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @see hipDeviceGetDefaultMemPool, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
 * hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device);
/**
 * @brief Returns device properties.
 *
 * @param [out] prop written with device properties
 * @param [in]  deviceId which device to query for information
 *
 * @return #hipSuccess, #hipErrorInvalidDevice
 * @bug HCC always returns 0 for maxThreadsPerMultiProcessor
 * @bug HCC always returns 0 for regsPerBlock
 * @bug HCC always returns 0 for l2CacheSize
 *
 * Populates hipGetDeviceProperties with information for the specified device.
 */
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);
/**
 * @brief Set L1/Shared cache partition.
 *
 * @param [in] cacheConfig Cache configuration
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorNotSupported
 *
 * Note: AMD devices do not support reconfigurable cache. This API is not implemented
 * on AMD platform. If the function is called, it will return hipErrorNotSupported.
 *
 */
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig);
/**
 * @brief Get Cache configuration for a specific Device
 *
 * @param [out] cacheConfig Pointer of cache configuration
 *
 * @returns #hipSuccess, #hipErrorNotInitialized
 * Note: AMD devices do not support reconfigurable cache. This hint is ignored
 * on these architectures.
 *
 */
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig);
/**
 * @brief Gets resource limits of current device
 *
 * The function queries the size of limit value, as required by the input enum value hipLimit_t,
 * which can be either #hipLimitStackSize, or #hipLimitMallocHeapSize. Any other input as
 * default, the function will return #hipErrorUnsupportedLimit.
 *
 * @param [out] pValue Returns the size of the limit in bytes
 * @param [in]  limit The limit to query
 *
 * @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
 *
 */
hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit);
/**
 * @brief Sets resource limits of current device.
 *
 * As the input enum limit,
 * #hipLimitStackSize sets the limit value of the stack size on the current GPU device, per thread.
 * The limit size can get via hipDeviceGetLimit. The size is in units of 256 dwords, up to the limit
 * (128K - 16).
 *
 * #hipLimitMallocHeapSize sets the limit value of the heap used by the malloc()/free()
 * calls. For limit size, use the #hipDeviceGetLimit API.
 *
 * Any other input as default, the funtion will return hipErrorUnsupportedLimit.
 *
 * @param [in] limit Enum of hipLimit_t to set
 * @param [in] value The size of limit value in bytes
 *
 * @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
 *
 */
hipError_t hipDeviceSetLimit ( enum hipLimit_t limit, size_t value );
/**
 * @brief Returns bank width of shared memory for current device
 *
 * @param [out] pConfig The pointer of the bank width for shared memory
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 *
 * Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 */
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig);
/**
 * @brief Gets the flags set for current device
 *
 * @param [out] flags Pointer of the flags
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 */
hipError_t hipGetDeviceFlags(unsigned int* flags);
/**
 * @brief The bank width of shared memory on current device is set
 *
 * @param [in] config Configuration for the bank width of shared memory
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 *
 * Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 */
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config);
/**
 * @brief The current device behavior is changed according the flags passed.
 *
 * @param [in] flags Flag to set on the current device
 *
 * The schedule flags impact how HIP waits for the completion of a command running on a device.
 * hipDeviceScheduleSpin         : HIP runtime will actively spin in the thread which submitted the
 * work until the command completes.  This offers the lowest latency, but will consume a CPU core
 * and may increase power. hipDeviceScheduleYield        : The HIP runtime will yield the CPU to
 * system so that other tasks can use it.  This may increase latency to detect the completion but
 * will consume less power and is friendlier to other tasks in the system.
 * hipDeviceScheduleBlockingSync : On ROCm platform, this is a synonym for hipDeviceScheduleYield.
 * hipDeviceScheduleAuto         : Use a hueristic to select between Spin and Yield modes.  If the
 * number of HIP contexts is greater than the number of logical processors in the system, use Spin
 * scheduling.  Else use Yield scheduling.
 *
 *
 * hipDeviceMapHost              : Allow mapping host memory.  On ROCM, this is always allowed and
 * the flag is ignored. hipDeviceLmemResizeToMax      : @warning ROCm silently ignores this flag.
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorSetOnActiveProcess
 *
 *
 */
hipError_t hipSetDeviceFlags(unsigned flags);
/**
 * @brief Device which matches hipDeviceProp_t is returned
 *
 * @param [out] device Pointer of the device
 * @param [in]  prop Pointer of the properties
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop);
/**
 * @brief Returns the link type and hop count between two devices
 *
 * @param [in] device1 Ordinal for device1
 * @param [in] device2 Ordinal for device2
 * @param [out] linktype Returns the link type (See hsa_amd_link_info_type_t) between the two devices
 * @param [out] hopcount Returns the hop count between the two devices
 *
 * Queries and returns the HSA link type and the hop count between the two specified devices.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype, uint32_t* hopcount);
// TODO: implement IPC apis
/**
 * @brief Gets an interprocess memory handle for an existing device memory
 *          allocation
 *
 * Takes a pointer to the base of an existing device memory allocation created
 * with hipMalloc and exports it for use in another process. This is a
 * lightweight operation and may be called multiple times on an allocation
 * without adverse effects.
 *
 * If a region of memory is freed with hipFree and a subsequent call
 * to hipMalloc returns memory with the same device address,
 * hipIpcGetMemHandle will return a unique handle for the
 * new memory.
 *
 * @param handle - Pointer to user allocated hipIpcMemHandle to return
 *                    the handle in.
 * @param devPtr - Base pointer to previously allocated device memory
 *
 * @returns #hipSuccess, #hipErrorInvalidHandle, #hipErrorOutOfMemory, #hipErrorMapFailed
 *
 * @note This IPC memory related feature API on Windows may behave differently from Linux.
 *
 */
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
/**
 * @brief Opens an interprocess memory handle exported from another process
 *          and returns a device pointer usable in the local process.
 *
 * Maps memory exported from another process with hipIpcGetMemHandle into
 * the current device address space. For contexts on different devices
 * hipIpcOpenMemHandle can attempt to enable peer access between the
 * devices as if the user called hipDeviceEnablePeerAccess. This behavior is
 * controlled by the hipIpcMemLazyEnablePeerAccess flag.
 * hipDeviceCanAccessPeer can determine if a mapping is possible.
 *
 * Contexts that may open hipIpcMemHandles are restricted in the following way.
 * hipIpcMemHandles from each device in a given process may only be opened
 * by one context per device per other process.
 *
 * Memory returned from hipIpcOpenMemHandle must be freed with
 * hipIpcCloseMemHandle.
 *
 * Calling hipFree on an exported memory region before calling
 * hipIpcCloseMemHandle in the importing context will result in undefined
 * behavior.
 *
 * @param devPtr - Returned device pointer
 * @param handle - hipIpcMemHandle to open
 * @param flags  - Flags for this operation. Must be specified as hipIpcMemLazyEnablePeerAccess
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidContext,
 *  #hipErrorInvalidDevicePointer
 *
 * @note During multiple processes, using the same memory handle opened by the current context,
 * there is no guarantee that the same device poiter will be returned in @p *devPtr.
 * This is diffrent from CUDA.
 * @note This IPC memory related feature API on Windows may behave differently from Linux.
 *
 */
hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags);
/**
 * @brief Close memory mapped with hipIpcOpenMemHandle
 *
 * Unmaps memory returnd by hipIpcOpenMemHandle. The original allocation
 * in the exporting process as well as imported mappings in other processes
 * will be unaffected.
 *
 * Any resources used to enable peer access will be freed if this is the
 * last mapping using them.
 *
 * @param devPtr - Device pointer returned by hipIpcOpenMemHandle
 *
 * @returns #hipSuccess, #hipErrorMapFailed, #hipErrorInvalidHandle
 *
 * @note This IPC memory related feature API on Windows may behave differently from Linux.
 *
 */
hipError_t hipIpcCloseMemHandle(void* devPtr);

/**
 * @brief Gets an opaque interprocess handle for an event.
 *
 * This opaque handle may be copied into other processes and opened with hipIpcOpenEventHandle.
 * Then hipEventRecord, hipEventSynchronize, hipStreamWaitEvent and hipEventQuery may be used in
 * either process. Operations on the imported event after the exported event has been freed with hipEventDestroy
 * will result in undefined behavior.
 *
 * @param[out]  handle Pointer to hipIpcEventHandle to return the opaque event handle
 * @param[in]   event  Event allocated with hipEventInterprocess and hipEventDisableTiming flags
 *
 * @returns #hipSuccess, #hipErrorInvalidConfiguration, #hipErrorInvalidValue
 *
 * @note This IPC event related feature API is currently applicable on Linux.
 *
 */
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event);

/**
 * @brief Opens an interprocess event handles.
 *
 * Opens an interprocess event handle exported from another process with cudaIpcGetEventHandle. The returned
 * hipEvent_t behaves like a locally created event with the hipEventDisableTiming flag specified. This event
 * need be freed with hipEventDestroy. Operations on the imported event after the exported event has been freed
 * with hipEventDestroy will result in undefined behavior. If the function is called within the same process where
 * handle is returned by hipIpcGetEventHandle, it will return hipErrorInvalidContext.
 *
 * @param[out]  event  Pointer to hipEvent_t to return the event
 * @param[in]   handle The opaque interprocess handle to open
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidContext
 *
 * @note This IPC event related feature API is currently applicable on Linux.
 *
 */
hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle);

// end doxygen Device
/**
 * @}
 */
/**
 *
 *  @defgroup Execution Execution Control
 *  @{
 *  This section describes the execution control functions of HIP runtime API.
 *
 */
/**
 * @brief Set attribute for a specific function
 *
 * @param [in] func Pointer of the function
 * @param [in] attr Attribute to set
 * @param [in] value Value to set
 *
 * @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
 *
 * Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 */
hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value);
/**
 * @brief Set Cache configuration for a specific function
 *
 * @param [in] func Pointer of the function.
 * @param [in] config Configuration to set.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized
 * Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
 * on those architectures.
 *
 */
hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config);
/**
 * @brief Set shared memory configuation for a specific function
 *
 * @param [in] func Pointer of the function
 * @param [in] config Configuration
 *
 * @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
 *
 * Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 */
hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config);
//doxygen end execution
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Error Error Handling
 *  @{
 *  This section describes the error handling functions of HIP runtime API.
 */
/**
 * @brief Return last error returned by any HIP runtime API call and resets the stored error code to
 * #hipSuccess
 *
 * @returns return code from last HIP called from the active host thread
 *
 * Returns the last error that has been returned by any of the runtime calls in the same host
 * thread, and then resets the saved error to #hipSuccess.
 *
 * @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
 */
hipError_t hipGetLastError(void);

/**
 * @brief Return last error returned by any HIP runtime API call and resets the stored error code to
 * #hipSuccess
 *
 * @returns return code from last HIP called from the active host thread
 *
 * Returns the last error that has been returned by any of the runtime calls in the same host
 * thread, and then resets the saved error to #hipSuccess.
 *
 * @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
 */
hipError_t hipExtGetLastError(void);

/**
 * @brief Return last error returned by any HIP runtime API call.
 *
 * @return #hipSuccess
 *
 * Returns the last error that has been returned by any of the runtime calls in the same host
 * thread. Unlike hipGetLastError, this function does not reset the saved error code.
 *
 * @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
 */
hipError_t hipPeekAtLastError(void);
/**
 * @brief Return hip error as text string form.
 *
 * @param hip_error Error code to convert to name.
 * @return const char pointer to the NULL-terminated error name
 *
 * @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
 */
const char* hipGetErrorName(hipError_t hip_error);
/**
 * @brief Return handy text string message to explain the error which occurred
 *
 * @param hipError Error code to convert to string.
 * @return const char pointer to the NULL-terminated error string
 *
 * @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
 */
const char* hipGetErrorString(hipError_t hipError);
/**
 * @brief Return hip error as text string form.
 *
 * @param [in] hipError Error code to convert to string.
 * @param [out] errorString char pointer to the NULL-terminated error string
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
 */
hipError_t hipDrvGetErrorName(hipError_t hipError, const char** errorString);
/**
 * @brief Return handy text string message to explain the error which occurred
 *
 * @param [in] hipError Error code to convert to string.
 * @param [out] errorString char pointer to the NULL-terminated error string
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
 */
hipError_t hipDrvGetErrorString(hipError_t hipError, const char** errorString);
// end doxygen Error
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Stream Stream Management
 *  @{
 *  This section describes the stream management functions of HIP runtime API.
 *  The following Stream APIs are not (yet) supported in HIP:
 *  - hipStreamAttachMemAsync is a nop
 */

/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
 * newly created stream.
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
 * reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
 * the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
 * used by the stream, applicaiton must call hipStreamDestroy.
 *
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipStreamCreate(hipStream_t* stream);
/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] flags to control stream creation.
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
 * reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
 * the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
 * used by the stream, applicaiton must call hipStreamDestroy. Flags controls behavior of the
 * stream.  See #hipStreamDefault, #hipStreamNonBlocking.
 *
 *
 * @see hipStreamCreate, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
/**
 * @brief Create an asynchronous stream with the specified priority.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] flags to control stream creation.
 * @param[in ] priority of the stream. Lower numbers represent higher priorities.
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream with the specified priority.  @p stream returns an opaque handle
 * that can be used to reference the newly created stream in subsequent hipStream* commands.  The
 * stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
 * To release the memory used by the stream, applicaiton must call hipStreamDestroy. Flags controls
 * behavior of the stream.  See #hipStreamDefault, #hipStreamNonBlocking.
 *
 *
 * @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority);
/**
 * @brief Returns numerical values that correspond to the least and greatest stream priority.
 *
 * @param[in, out] leastPriority pointer in which value corresponding to least priority is returned.
 * @param[in, out] greatestPriority pointer in which value corresponding to greatest priority is returned.
 * @returns #hipSuccess
 *
 * Returns in *leastPriority and *greatestPriority the numerical values that correspond to the least
 * and greatest stream priority respectively. Stream priorities follow a convention where lower numbers
 * imply greater priorities. The range of meaningful stream priorities is given by
 * [*greatestPriority, *leastPriority]. If the user attempts to create a stream with a priority value
 * that is outside the the meaningful range as specified by this API, the priority is automatically
 * clamped to within the valid range.
 */
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
/**
 * @brief Destroys the specified stream.
 *
 * @param[in] stream stream identifier.
 * @return #hipSuccess #hipErrorInvalidHandle
 *
 * Destroys the specified stream.
 *
 * If commands are still executing on the specified stream, some may complete execution before the
 * queue is deleted.
 *
 * The queue may be destroyed while some commands are still inflight, or may wait for all commands
 * queued to the stream before destroying it.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamQuery,
 * hipStreamWaitEvent, hipStreamSynchronize
 */
hipError_t hipStreamDestroy(hipStream_t stream);
/**
 * @brief Return #hipSuccess if all of the operations in the specified @p stream have completed, or
 * #hipErrorNotReady if not.
 *
 * @param[in] stream stream to query
 *
 * @return #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle
 *
 * This is thread-safe and returns a snapshot of the current state of the queue.  However, if other
 * host threads are sending work to the stream, the status may change immediately after the function
 * is called.  It is typically used for debug.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent,
 * hipStreamSynchronize, hipStreamDestroy
 */
hipError_t hipStreamQuery(hipStream_t stream);
/**
 * @brief Wait for all commands in stream to complete.
 *
 * @param[in] stream stream identifier.
 *
 * @return #hipSuccess, #hipErrorInvalidHandle
 *
 * This command is host-synchronous : the host will block until the specified stream is empty.
 *
 * This command follows standard null-stream semantics.  Specifically, specifying the null stream
 * will cause the command to wait for other streams on the same device to complete all pending
 * operations.
 *
 * This command honors the hipDeviceLaunchBlocking flag, which controls whether the wait is active
 * or blocking.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent,
 * hipStreamDestroy
 *
 */
hipError_t hipStreamSynchronize(hipStream_t stream);
/**
 * @brief Make the specified compute stream wait for an event
 *
 * @param[in] stream stream to make wait.
 * @param[in] event event to wait on
 * @param[in] flags control operation [must be 0]
 *
 * @return #hipSuccess, #hipErrorInvalidHandle
 *
 * This function inserts a wait operation into the specified stream.
 * All future work submitted to @p stream will wait until @p event reports completion before
 * beginning execution.
 *
 * This function only waits for commands in the current stream to complete.  Notably,, this function
 * does not impliciy wait for commands in the default stream to complete, even if the specified
 * stream is created with hipStreamNonBlocking = 0.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamDestroy
 */
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags __dparm(0));
/**
 * @brief Return flags associated with this stream.
 *
 * @param[in] stream stream to be queried
 * @param[in,out] flags Pointer to an unsigned integer in which the stream's flags are returned
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
 *
 * @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
 *
 * Return flags associated with this stream in *@p flags.
 *
 * @see hipStreamCreateWithFlags
 */
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags);
/**
 * @brief Query the priority of a stream.
 *
 * @param[in] stream stream to be queried
 * @param[in,out] priority Pointer to an unsigned integer in which the stream's priority is returned
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
 *
 * @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
 *
 * Query the priority of a stream. The priority is returned in in priority.
 *
 * @see hipStreamCreateWithFlags
 */
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority);
/**
 * @brief Get the device assocaited with the stream
 *
 * @param[in] stream stream to be queried
 * @param[out] device device associated with the stream
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorContextIsDestroyed, #hipErrorInvalidHandle,
 * #hipErrorNotInitialized, #hipErrorDeinitialized, #hipErrorInvalidContext
 *
 * @see hipStreamCreate, hipStreamDestroy, hipDeviceGetStreamPriorityRange
 */
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device);
/**
 * @brief Create an asynchronous stream with the specified CU mask.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] cuMaskSize Size of CU mask bit array passed in.
 * @param[in ] cuMask Bit-vector representing the CU mask. Each active bit represents using one CU.
 * The first 32 bits represent the first 32 CUs, and so on. If its size is greater than physical
 * CU number (i.e., multiProcessorCount member of hipDeviceProp_t), the extra elements are ignored.
 * It is user's responsibility to make sure the input is meaningful.
 * @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream with the specified CU mask.  @p stream returns an opaque handle
 * that can be used to reference the newly created stream in subsequent hipStream* commands.  The
 * stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
 * To release the memory used by the stream, application must call hipStreamDestroy.
 *
 *
 * @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize, const uint32_t* cuMask);
/**
 * @brief Get CU mask associated with an asynchronous stream
 *
 * @param[in] stream stream to be queried
 * @param[in] cuMaskSize number of the block of memories (uint32_t *) allocated by user
 * @param[out] cuMask Pointer to a pre-allocated block of memories (uint32_t *) in which
 * the stream's CU mask is returned. The CU mask is returned in a chunck of 32 bits where
 * each active bit represents one active CU
 * @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
 *
 * @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask);
/**
 * Stream CallBack struct
 */
typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* userData);
/**
 * @brief Adds a callback to be called on the host after all currently enqueued
 * items in the stream have completed.  For each
 * hipStreamAddCallback call, a callback will be executed exactly once.
 * The callback will block later work in the stream until it is finished.
 * @param[in] stream   - Stream to add callback to
 * @param[in] callback - The function to call once preceding stream operations are complete
 * @param[in] userData - User specified data to be passed to the callback function
 * @param[in] flags    - Reserved for future use, must be 0
 * @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorNotSupported
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamQuery, hipStreamSynchronize,
 * hipStreamWaitEvent, hipStreamDestroy, hipStreamCreateWithPriority
 *
 */
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags);
// end doxygen Stream
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup StreamM Stream Memory Operations
 *  @{
 *  This section describes Stream Memory Wait and Write functions of HIP runtime API.
 */
/**
 * @brief Enqueues a wait command to the stream.[BETA]
 *
 * @param [in] stream - Stream identifier
 * @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
 * @param [in] value  - Value to be used in compare operation
 * @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
 * hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor
 * @param [in] mask   - Mask to be applied on value at memory before it is compared with value,
 * default value is set to enable every bit
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
 * not execute until the defined wait condition is true.
 *
 * hipStreamWaitValueGte: waits until *ptr&mask >= value
 * hipStreamWaitValueEq : waits until *ptr&mask == value
 * hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
 * hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
 *
 * @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
 *
 * @note Support for hipStreamWaitValue32 can be queried using 'hipDeviceGetAttribute()' and
 * 'hipDeviceAttributeCanUseStreamWaitValue' flag.
 *
 * @warning This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue64, hipStreamWriteValue64,
 * hipStreamWriteValue32, hipDeviceGetAttribute
 */
hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags,
                                uint32_t mask __dparm(0xFFFFFFFF));
/**
 * @brief Enqueues a wait command to the stream.[BETA]
 *
 * @param [in] stream - Stream identifier
 * @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
 * @param [in] value  - Value to be used in compare operation
 * @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
 * hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor.
 * @param [in] mask   - Mask to be applied on value at memory before it is compared with value
 * default value is set to enable every bit
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
 * not execute until the defined wait condition is true.
 *
 * hipStreamWaitValueGte: waits until *ptr&mask >= value
 * hipStreamWaitValueEq : waits until *ptr&mask == value
 * hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
 * hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
 *
 * @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
 *
 * @note Support for hipStreamWaitValue64 can be queried using 'hipDeviceGetAttribute()' and
 * 'hipDeviceAttributeCanUseStreamWaitValue' flag.
 *
 * @warning This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue32, hipStreamWriteValue64,
 * hipStreamWriteValue32, hipDeviceGetAttribute
 */
hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags,
                                uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
/**
 * @brief Enqueues a write command to the stream.[BETA]
 *
 * @param [in] stream - Stream identifier
 * @param [in] ptr    - Pointer to a GPU accessible memory object
 * @param [in] value  - Value to be written
 * @param [in] flags  - reserved, ignored for now, will be used in future releases
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * Enqueues a write command to the stream, write operation is performed after all earlier commands
 * on this stream have completed the execution.
 *
 * @warning This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
 * hipStreamWaitValue64
 */
hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags);
/**
 * @brief Enqueues a write command to the stream.[BETA]
 *
 * @param [in] stream - Stream identifier
 * @param [in] ptr    - Pointer to a GPU accessible memory object
 * @param [in] value  - Value to be written
 * @param [in] flags  - reserved, ignored for now, will be used in future releases
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * Enqueues a write command to the stream, write operation is performed after all earlier commands
 * on this stream have completed the execution.
 *
 * @warning This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
 * hipStreamWaitValue64
 */
hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags);
// end doxygen Stream Memory Operations
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Event Event Management
 *  @{
 *  This section describes the event management functions of HIP runtime API.
 */
/**
 * @brief Create an event with the specified flags
 *
 * @param[in,out] event Returns the newly created event.
 * @param[in] flags     Flags to control event behavior.  Valid values are #hipEventDefault,
 #hipEventBlockingSync, #hipEventDisableTiming, #hipEventInterprocess
 * #hipEventDefault : Default flag.  The event will use active synchronization and will support
 timing.  Blocking synchronization provides lowest possible latency at the expense of dedicating a
 CPU to poll on the event.
 * #hipEventBlockingSync : The event will use blocking synchronization : if hipEventSynchronize is
 called on this event, the thread will block until the event completes.  This can increase latency
 for the synchroniation but can result in lower power and more resources for other CPU threads.
 * #hipEventDisableTiming : Disable recording of timing information. Events created with this flag
 would not record profiling data and provide best performance if used for synchronization.
 * #hipEventInterprocess : The event can be used as an interprocess event. hipEventDisableTiming
 flag also must be set when hipEventInterprocess flag is set.
 * #hipEventDisableSystemFence : Disable acquire and release system scope fence. This may
 improve performance but device memory may not be visible to the host and other devices
 if this flag is set.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
 #hipErrorLaunchFailure, #hipErrorOutOfMemory
 *
 * @see hipEventCreate, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
 */
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
/**
 *  Create an event
 *
 * @param[in,out] event Returns the newly created event.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
 * #hipErrorLaunchFailure, #hipErrorOutOfMemory
 *
 * @see hipEventCreateWithFlags, hipEventRecord, hipEventQuery, hipEventSynchronize,
 * hipEventDestroy, hipEventElapsedTime
 */
hipError_t hipEventCreate(hipEvent_t* event);
/**
 * @brief Record an event in the specified stream.
 *
 * @param[in] event event to record.
 * @param[in] stream stream in which to record event.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized,
 * #hipErrorInvalidHandle, #hipErrorLaunchFailure
 *
 * hipEventQuery() or hipEventSynchronize() must be used to determine when the event
 * transitions from "recording" (after hipEventRecord() is called) to "recorded"
 * (when timestamps are set, if requested).
 *
 * Events which are recorded in a non-NULL stream will transition to
 * from recording to "recorded" state when they reach the head of
 * the specified stream, after all previous
 * commands in that stream have completed executing.
 *
 * If hipEventRecord() has been previously called on this event, then this call will overwrite any
 * existing state in event.
 * 
 * If this function is called on an event that is currently being recorded, results are undefined
 * - either outstanding recording may save state into the event, and the order is not guaranteed.
 *
 * @note: If this function is not called before use hipEventQuery() or hipEventSynchronize(),
 * #hipSuccess is returned, meaning no pending event in the stream.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize,
 * hipEventDestroy, hipEventElapsedTime
 *
 */
#ifdef __cplusplus
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = NULL);
#else
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
#endif
/**
 *  @brief Destroy the specified event.
 *
 *  @param[in] event Event to destroy.
 *  @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
 * #hipErrorLaunchFailure
 *
 *  Releases memory associated with the event.  If the event is recording but has not completed
 * recording when hipEventDestroy() is called, the function will return immediately and the
 * completion_future resources will be released later, when the hipDevice is synchronized.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize, hipEventRecord,
 * hipEventElapsedTime
 *
 * @returns #hipSuccess
 */
hipError_t hipEventDestroy(hipEvent_t event);
/**
 *  @brief Wait for an event to complete.
 *
 *  This function will block until the event is ready, waiting for all previous work in the stream
 * specified when event was recorded with hipEventRecord().
 *
 *  If hipEventRecord() has not been called on @p event, this function returns #hipSuccess when no
 *  event is captured.
 *
 *  This function needs to support hipEventBlockingSync parameter.
 *
 *  @param[in] event Event on which to wait.
 *
 *  @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized,
 * #hipErrorInvalidHandle, #hipErrorLaunchFailure
 *
 *  @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
 * hipEventElapsedTime
 */
hipError_t hipEventSynchronize(hipEvent_t event);
/**
 * @brief Return the elapsed time between two events.
 *
 * @param[out] ms : Return time between start and stop in ms.
 * @param[in]   start : Start event.
 * @param[in]   stop  : Stop event.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotReady, #hipErrorInvalidHandle,
 * #hipErrorNotInitialized, #hipErrorLaunchFailure
 *
 * Computes the elapsed time between two events. Time is computed in ms, with
 * a resolution of approximately 1 us.
 *
 * Events which are recorded in a NULL stream will block until all commands
 * on all other streams complete execution, and then record the timestamp.
 *
 * Events which are recorded in a non-NULL stream will record their timestamp
 * when they reach the head of the specified stream, after all previous
 * commands in that stream have completed executing.  Thus the time that
 * the event recorded may be significantly after the host calls hipEventRecord().
 *
 * If hipEventRecord() has not been called on either event, then #hipErrorInvalidHandle is
 * returned. If hipEventRecord() has been called on both events, but the timestamp has not yet been
 * recorded on one or both events (that is, hipEventQuery() would return #hipErrorNotReady on at
 * least one of the events), then #hipErrorNotReady is returned.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
 * hipEventSynchronize
 */
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
/**
 * @brief Query event status
 *
 * @param[in] event Event to query.
 * @returns #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle, #hipErrorInvalidValue,
 * #hipErrorNotInitialized, #hipErrorLaunchFailure
 *
 * Query the status of the specified event.  This function will return #hipSuccess if all
 * commands in the appropriate stream (specified to hipEventRecord()) have completed.  If any execution
 * has not completed, then #hipErrorNotReady is returned.
 *
 * @note: This API returns #hipSuccess, if hipEventRecord() is not called before this API.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventRecord, hipEventDestroy,
 * hipEventSynchronize, hipEventElapsedTime
 */
hipError_t hipEventQuery(hipEvent_t event);
// end doxygen Events
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Memory Memory Management
 *  @{
 *  This section describes the memory management functions of HIP runtime API.
 *  The following CUDA APIs are not currently supported:
 *  - cudaMalloc3D
 *  - cudaMalloc3DArray
 *  - TODO - more 2D, 3D, array APIs here.
 *
 *
 */

/**
 *  @brief Sets information on the specified pointer.[BETA]
 *
 *  @param [in]      value     Sets pointer attribute value
 *  @param [in]      attribute  Attribute to set
 *  @param [in]      ptr      Pointer to set attributes for
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @warning This API is marked as beta, meaning, while this is feature complete,
 *  it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipPointerSetAttribute(const void* value, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr);


/**
 *  @brief Returns attributes for the specified pointer
 *
 *  @param [out]  attributes  attributes for the specified pointer
 *  @param [in]   ptr         pointer to get attributes for
 *
 *  The output parameter 'attributes' has a member named 'type' that describes what memory the
 *  pointer is associated with, such as device memory, host memory, managed memory, and others.
 *  Otherwise, the API cannot handle the pointer and returns #hipErrorInvalidValue.
 *
 *  @note  The unrecognized memory type is unsupported to keep the HIP functionality backward
 *  compatibility due to #hipMemoryType enum values.
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @note  The current behavior of this HIP API corresponds to the CUDA API before version 11.0.
 *
 *  @see hipPointerGetAttribute
 */
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);
/**
 *  @brief Returns information about the specified pointer.[BETA]
 *
 *  @param [in, out] data     Returned pointer attribute value
 *  @param [in]      attribute  Attribute to query for
 *  @param [in]      ptr      Pointer to get attributes for
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @warning This API is marked as beta, meaning, while this is feature complete,
 *  it is still open to changes and may have outstanding issues.
 *
 *  @see hipPointerGetAttributes
 */
hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr);
/**
 *  @brief Returns information about the specified pointer.[BETA]
 *
 *  @param [in]  numAttributes   number of attributes to query for
 *  @param [in]  attributes      attributes to query for
 *  @param [in, out] data        a two-dimensional containing pointers to memory locations
 *                               where the result of each attribute query will be written to
 *  @param [in]  ptr             pointer to get attributes for
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @warning This API is marked as beta, meaning, while this is feature complete,
 *  it is still open to changes and may have outstanding issues.
 *
 *  @see hipPointerGetAttribute
 */
hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute* attributes,
                                      void** data, hipDeviceptr_t ptr);
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup External External Resource Interoperability
 *  @{
 *  @ingroup API
 *
 *  This section describes the external resource interoperability functions of HIP runtime API.
 *
 */
/**
 *  @brief Imports an external semaphore.
 *
 *  @param[out] extSem_out  External semaphores to be waited on
 *  @param[in] semHandleDesc Semaphore import handle descriptor
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @see
 */
hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out,
                                      const hipExternalSemaphoreHandleDesc* semHandleDesc);
/**
 *  @brief Signals a set of external semaphore objects.
 *
 *  @param[in] extSemArray  External semaphores to be waited on
 *  @param[in] paramsArray Array of semaphore parameters
 *  @param[in] numExtSems Number of semaphores to wait on
 *  @param[in] stream Stream to enqueue the wait operations in
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @see
 */
hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                            const hipExternalSemaphoreSignalParams* paramsArray,
                                            unsigned int numExtSems, hipStream_t stream);
/**
 *  @brief Waits on a set of external semaphore objects
 *
 *  @param[in] extSemArray  External semaphores to be waited on
 *  @param[in] paramsArray Array of semaphore parameters
 *  @param[in] numExtSems Number of semaphores to wait on
 *  @param[in] stream Stream to enqueue the wait operations in
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @see
 */
hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                              const hipExternalSemaphoreWaitParams* paramsArray,
                                              unsigned int numExtSems, hipStream_t stream);
/**
 *  @brief Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.
 *
 *  @param[in] extSem handle to an external memory object
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @see
 */
hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem);

/**
*  @brief Imports an external memory object.
*
*  @param[out] extMem_out  Returned handle to an external memory object
*  @param[in]  memHandleDesc Memory import handle descriptor
*
*  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
*
*  @see
*/
hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out, const hipExternalMemoryHandleDesc* memHandleDesc);
/**
*  @brief Maps a buffer onto an imported memory object.
*
*  @param[out] devPtr Returned device pointer to buffer
*  @param[in]  extMem  Handle to external memory object
*  @param[in]  bufferDesc  Buffer descriptor
*
*  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
*
*  @see
*/
hipError_t hipExternalMemoryGetMappedBuffer(void **devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc *bufferDesc);
/**
*  @brief Destroys an external memory object.
*
*  @param[in] extMem  External memory object to be destroyed
*
*  @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
*
*  @see
*/
hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem);
/**
 *  @brief Maps a mipmapped array onto an external memory object.
 *
 *  @param[out] mipmap mipmapped array to return
 *  @param[in]  extMem external memory object handle
 *  @param[in]  mipmapDesc external mipmapped array descriptor
 *
 *  Returned mipmapped array must be freed using hipFreeMipmappedArray.
 *
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidResourceHandle
 *
 *  @see hipImportExternalMemory, hipDestroyExternalMemory, hipExternalMemoryGetMappedBuffer, hipFreeMipmappedArray
 */
hipError_t hipExternalMemoryGetMappedMipmappedArray(hipMipmappedArray_t* mipmap, hipExternalMemory_t extMem,
    const hipExternalMemoryMipmappedArrayDesc* mipmapDesc);
 // end of external resource
 /**
 * @}
 */
/**
 *  @brief Allocate memory on the default accelerator
 *
 *  @param[out] ptr Pointer to the allocated memory
 *  @param[in]  size Requested memory size
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
 *
 *  @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
 * hipHostFree, hipHostMalloc
 */
hipError_t hipMalloc(void** ptr, size_t size);
/**
 *  @brief Allocate memory on the default accelerator
 *
 *  @param[out] ptr  Pointer to the allocated memory
 *  @param[in]  sizeBytes  Requested memory size
 *  @param[in]  flags  Type of memory allocation
 *
 *  If requested memory size is 0, no memory is allocated, *ptr returns nullptr, and #hipSuccess
 *  is returned.
 *
 *  The memory allocation flag should be either #hipDeviceMallocDefault,
 *  #hipDeviceMallocFinegrained, #hipDeviceMallocUncached, or #hipMallocSignalMemory.
 *  If the flag is any other value, the API returns #hipErrorInvalidValue.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
 *
 *  @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
 * hipHostFree, hipHostMalloc
 */
hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags);
/**
 *  @brief Allocate pinned host memory [Deprecated]
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @warning  This API is deprecated, use hipHostMalloc() instead
 */
DEPRECATED("use hipHostMalloc instead")
hipError_t hipMallocHost(void** ptr, size_t size);
/**
 *  @brief Allocate pinned host memory [Deprecated]
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @warning  This API is deprecated, use hipHostMalloc() instead
 */
DEPRECATED("use hipHostMalloc instead")
hipError_t hipMemAllocHost(void** ptr, size_t size);
/**
 *  @brief Allocates device accessible page locked (pinned) host memory
 *
 *  This API allocates pinned host memory which is mapped into the address space of all GPUs
 *  in the system, the memory can be accessed directly by the GPU device, and can be read or
 *  written with much higher bandwidth than pageable memory obtained with functions such as
 *  malloc().
 *
 *  Using the pinned host memory, applications can implement faster data transfers for HostToDevice
 *  and DeviceToHost. The runtime tracks the hipHostMalloc allocations and can avoid some of the
 *  setup required for regular unpinned memory.
 *
 *  When the memory accesses are infrequent, zero-copy memory can be a good choice, for coherent
 *  allocation. GPU can directly access the host memory over the CPU/GPU interconnect, without need
 *  to copy the data.
 *
 *  Currently the allocation granularity is 4KB for the API.
 *
 *  Developers need to choose proper allocation flag with consideration of synchronization.
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size in bytes
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *  @param[in]  flags Type of host memory allocation
 *
 *  If no input for flags, it will be the default pinned memory allocation on the host.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @see hipSetDeviceFlags, hipHostFree
 */
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup MemoryM Managed Memory
 *
 *  @ingroup Memory
 * @{
 *  This section describes the managed memory management functions of HIP runtime API.
 *
 *  @note  The managed memory management APIs are implemented on Linux, under developement
 *  on Windows.
 *
 */
/**
 * @brief Allocates memory that will be automatically managed by HIP.
 *
 * This API is used for managed memory, allows data be shared and accessible to both CPU and
 * GPU using a single pointer.
 *
 * The API returns the allocation pointer, managed by HMM, can be used further to execute kernels
 * on device and fetch data between the host and device as needed.
 *
 * @note   It is recommend to do the capability check before call this API.
 *
 * @param [out] dev_ptr - pointer to allocated device memory
 * @param [in]  size    - requested allocation size in bytes, it should be granularity of 4KB
 * @param [in]  flags   - must be either hipMemAttachGlobal or hipMemAttachHost
 *                        (defaults to hipMemAttachGlobal)
 *
 * @returns #hipSuccess, #hipErrorMemoryAllocation, #hipErrorNotSupported, #hipErrorInvalidValue
 *
 */
hipError_t hipMallocManaged(void** dev_ptr,
                            size_t size,
                            unsigned int flags __dparm(hipMemAttachGlobal));
/**
 * @brief Prefetches memory to the specified destination device using HIP.
 *
 * @param [in] dev_ptr  pointer to be prefetched
 * @param [in] count    size in bytes for prefetching
 * @param [in] device   destination device to prefetch to
 * @param [in] stream   stream to enqueue prefetch operation
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPrefetchAsync(const void* dev_ptr,
                               size_t count,
                               int device,
                               hipStream_t stream __dparm(0));
/**
 * @brief Advise about the usage of a given memory range to HIP.
 *
 * @param [in] dev_ptr  pointer to memory to set the advice for
 * @param [in] count    size in bytes of the memory range, it should be CPU page size alligned.
 * @param [in] advice   advice to be applied for the specified memory range
 * @param [in] device   device to apply the advice for
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * This HIP API advises about the usage to be applied on unified memory allocation in the
 * range starting from the pointer address devPtr, with the size of count bytes.
 * The memory range must refer to managed memory allocated via the API hipMallocManaged, and the
 * range will be handled with proper round down and round up respectively in the driver to
 * be aligned to CPU page size, the same way as corresponding CUDA API behaves in CUDA version 8.0
 * and afterwards.
 *
 * @note  This API is implemented on Linux and is under development on Windows.
 */
hipError_t hipMemAdvise(const void* dev_ptr,
                        size_t count,
                        hipMemoryAdvise advice,
                        int device);
/**
 * @brief Query an attribute of a given memory range in HIP.
 *
 * @param [in,out] data   a pointer to a memory location where the result of each
 *                        attribute query will be written to
 * @param [in] data_size  the size of data
 * @param [in] attribute  the attribute to query
 * @param [in] dev_ptr    start of the range to query
 * @param [in] count      size of the range to query
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemRangeGetAttribute(void* data,
                                   size_t data_size,
                                   hipMemRangeAttribute attribute,
                                   const void* dev_ptr,
                                   size_t count);
/**
 * @brief Query attributes of a given memory range in HIP.
 *
 * @param [in,out] data     a two-dimensional array containing pointers to memory locations
 *                          where the result of each attribute query will be written to
 * @param [in] data_sizes   an array, containing the sizes of each result
 * @param [in] attributes   the attribute to query
 * @param [in] num_attributes  an array of attributes to query (numAttributes and the number
 *                          of attributes in this array should match)
 * @param [in] dev_ptr      start of the range to query
 * @param [in] count        size of the range to query
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemRangeGetAttributes(void** data,
                                    size_t* data_sizes,
                                    hipMemRangeAttribute* attributes,
                                    size_t num_attributes,
                                    const void* dev_ptr,
                                    size_t count);
/**
 * @brief Attach memory to a stream asynchronously in HIP.
 *
 * @param [in] stream     - stream in which to enqueue the attach operation
 * @param [in] dev_ptr    - pointer to memory (must be a pointer to managed memory or
 *                          to a valid host-accessible region of system-allocated memory)
 * @param [in] length     - length of memory (defaults to zero)
 * @param [in] flags      - must be one of hipMemAttachGlobal, hipMemAttachHost or
 *                          hipMemAttachSingle (defaults to hipMemAttachSingle)
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipStreamAttachMemAsync(hipStream_t stream,
                                   void* dev_ptr,
                                   size_t length __dparm(0),
                                   unsigned int flags __dparm(hipMemAttachSingle));
// end doxygen Managed Memory
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 * @defgroup StreamO Stream Ordered Memory Allocator
 * @{
 * @ingroup Memory
 * This section describes Stream Ordered Memory Allocator functions of HIP runtime API.
 *
 * The asynchronous allocator allows the user to allocate and free in stream order.
 * All asynchronous accesses of the allocation must happen between the stream executions of
 * the allocation and the free. If the memory is accessed outside of the promised stream order,
 * a use before allocation / use after free error  will cause undefined behavior.
 *
 * The allocator is free to reallocate the memory as long as it can guarantee that compliant memory
 * accesses will not overlap temporally. The allocator may refer to internal stream ordering as well
 * as inter-stream dependencies (such as HIP events and null stream dependencies) when establishing
 * the temporal guarantee. The allocator may also insert inter-stream dependencies to establish
 * the temporal guarantee.  Whether or not a device supports the integrated stream ordered memory
 * allocator may be queried by calling @p hipDeviceGetAttribute with the device attribute
 * @p hipDeviceAttributeMemoryPoolsSupported
 *
 * @note  APIs in this section are implemented on Linux, under development on Windows.
 */

/**
 * @brief Allocates memory with stream ordered semantics
 *
 * Inserts a memory allocation operation into @p stream.
 * A pointer to the allocated memory is returned immediately in *dptr.
 * The allocation must not be accessed until the the allocation operation completes.
 * The allocation comes from the memory pool associated with the stream's device.
 *
 * @note The default memory pool of a device contains device memory from that device.
 * @note Basic stream ordering allows future work submitted into the same stream to use the
 *  allocation. Stream query, stream synchronize, and HIP events can be used to guarantee that
 *  the allocation operation completes before work submitted in a separate stream runs.
 * @note During stream capture, this function results in the creation of an allocation node.
 *  In this case, the allocation is owned by the graph instead of the memory pool. The memory
 *  pool's properties are used to set the node's creation parameters.
 *
 * @param [out] dev_ptr  Returned device pointer of memory allocation
 * @param [in] size      Number of bytes to allocate
 * @param [in] stream    The stream establishing the stream ordering contract and
 *                       the memory pool to allocate from
 *
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
 *
 * @see hipMallocFromPoolAsync, hipFreeAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
 * hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream);
/**
 * @brief Frees memory with stream ordered semantics
 *
 * Inserts a free operation into @p stream.
 * The allocation must not be used after stream execution reaches the free.
 * After this API returns, accessing the memory from any subsequent work launched on the GPU
 * or querying its pointer attributes results in undefined behavior.
 *
 * @note During stream capture, this function results in the creation of a free node and
 * must therefore be passed the address of a graph allocation.
 *
 * @param [in] dev_ptr Pointer to device memory to free
 * @param [in] stream  The stream, where the destruciton will occur according to the execution order
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @see hipMallocFromPoolAsync, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute,
 * hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream);
/**
 * @brief Releases freed memory back to the OS
 *
 * Releases memory back to the OS until the pool contains fewer than @p min_bytes_to_keep
 * reserved bytes, or there is no more memory that the allocator can safely release.
 * The allocator cannot release OS allocations that back outstanding asynchronous allocations.
 * The OS allocations may happen at different granularity from the user allocations.
 *
 * @note: Allocations that have not been freed count as outstanding.
 * @note: Allocations that have been asynchronously freed but whose completion has
 * not been observed on the host (eg. by a synchronize) can count as outstanding.
 *
 * @param[in] mem_pool          The memory pool to trim allocations
 * @param[in] min_bytes_to_hold If the pool has less than min_bytes_to_hold reserved,
 * then the TrimTo operation is a no-op.  Otherwise the memory pool will contain
 * at least min_bytes_to_hold bytes reserved after the operation.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
 * hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold);
/**
 * @brief Sets attributes of a memory pool
 *
 * Supported attributes are:
 * - @p hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
 *                                  Amount of reserved memory in bytes to hold onto before trying
 *                                  to release memory back to the OS. When more than the release
 *                                  threshold bytes of memory are held by the memory pool, the
 *                                  allocator will try to release memory back to the OS on the
 *                                  next call to stream, event or context synchronize. (default 0)
 * - @p hipMemPoolReuseFollowEventDependencies: (value type = int)
 *                                  Allow @p hipMallocAsync to use memory asynchronously freed
 *                                  in another stream as long as a stream ordering dependency
 *                                  of the allocating stream on the free action exists.
 *                                  HIP events and null stream interactions can create the required
 *                                  stream ordered dependencies. (default enabled)
 * - @p hipMemPoolReuseAllowOpportunistic: (value type = int)
 *                                  Allow reuse of already completed frees when there is no dependency
 *                                  between the free and allocation. (default enabled)
 * - @p hipMemPoolReuseAllowInternalDependencies: (value type = int)
 *                                  Allow @p hipMallocAsync to insert new stream dependencies
 *                                  in order to establish the stream ordering required to reuse
 *                                  a piece of memory released by @p hipFreeAsync (default enabled).
 *
 * @param [in] mem_pool The memory pool to modify
 * @param [in] attr     The attribute to modify
 * @param [in] value    Pointer to the value to assign
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
 * hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAccess, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
/**
 * @brief Gets attributes of a memory pool
 *
 * Supported attributes are:
 * - @p hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
 *                                  Amount of reserved memory in bytes to hold onto before trying
 *                                  to release memory back to the OS. When more than the release
 *                                  threshold bytes of memory are held by the memory pool, the
 *                                  allocator will try to release memory back to the OS on the
 *                                  next call to stream, event or context synchronize. (default 0)
 * - @p hipMemPoolReuseFollowEventDependencies: (value type = int)
 *                                  Allow @p hipMallocAsync to use memory asynchronously freed
 *                                  in another stream as long as a stream ordering dependency
 *                                  of the allocating stream on the free action exists.
 *                                  HIP events and null stream interactions can create the required
 *                                  stream ordered dependencies. (default enabled)
 * - @p hipMemPoolReuseAllowOpportunistic: (value type = int)
 *                                  Allow reuse of already completed frees when there is no dependency
 *                                  between the free and allocation. (default enabled)
 * - @p hipMemPoolReuseAllowInternalDependencies: (value type = int)
 *                                  Allow @p hipMallocAsync to insert new stream dependencies
 *                                  in order to establish the stream ordering required to reuse
 *                                  a piece of memory released by @p hipFreeAsync (default enabled).
 *
 * @param [in] mem_pool The memory pool to get attributes of
 * @param [in] attr     The attribute to get
 * @param [in] value    Retrieved value
 *
 * @returns  #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync,
 * hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
/**
 * @brief Controls visibility of the specified pool between devices
 *
 * @param [in] mem_pool   Memory pool for acccess change
 * @param [in] desc_list  Array of access descriptors. Each descriptor instructs the access to enable for a single gpu
 * @param [in] count  Number of descriptors in the map array.
 *
 * @returns  #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
 * hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list, size_t count);
/**
 * @brief Returns the accessibility of a pool from a device
 *
 * Returns the accessibility of the pool's memory from the specified location.
 *
 * @param [out] flags    Accessibility of the memory pool from the specified location/device
 * @param [in] mem_pool   Memory pool being queried
 * @param [in] location  Location/device for memory pool access
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute,
 * hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool, hipMemLocation* location);
/**
 * @brief Creates a memory pool
 *
 * Creates a HIP memory pool and returns the handle in @p mem_pool. The @p pool_props determines
 * the properties of the pool such as the backing device and IPC capabilities.
 *
 * By default, the memory pool will be accessible from the device it is allocated on.
 *
 * @param [out] mem_pool    Contains createed memory pool
 * @param [in] pool_props   Memory pool properties
 *
 * @note Specifying hipMemHandleTypeNone creates a memory pool that will not support IPC.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolDestroy,
 * hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props);
/**
 * @brief Destroys the specified memory pool
 *
 * If any pointers obtained from this pool haven't been freed or
 * the pool has free operations that haven't completed
 * when @p hipMemPoolDestroy is invoked, the function will return immediately and the
 * resources associated with the pool will be released automatically
 * once there are no more outstanding allocations.
 *
 * Destroying the current mempool of a device sets the default mempool of
 * that device as the current mempool for that device.
 *
 * @param [in] mem_pool Memory pool for destruction
 *
 * @note A device's default memory pool cannot be destroyed.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate
 * hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool);
/**
 * @brief Allocates memory from a specified pool with stream ordered semantics.
 *
 * Inserts an allocation operation into @p stream.
 * A pointer to the allocated memory is returned immediately in @p dev_ptr.
 * The allocation must not be accessed until the the allocation operation completes.
 * The allocation comes from the specified memory pool.
 *
 * @note The specified memory pool may be from a device different than that of the specified @p stream.
 *
 * Basic stream ordering allows future work submitted into the same stream to use the allocation.
 * Stream query, stream synchronize, and HIP events can be used to guarantee that the allocation
 * operation completes before work submitted in a separate stream runs.
 *
 * @note During stream capture, this function results in the creation of an allocation node. In this case,
 * the allocation is owned by the graph instead of the memory pool. The memory pool's properties
 * are used to set the node's creation parameters.
 *
 * @param [out] dev_ptr Returned device pointer
 * @param [in] size     Number of bytes to allocate
 * @param [in] mem_pool The pool to allocate from
 * @param [in] stream   The stream establishing the stream ordering semantic
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
 *
 * @see hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate
 * hipMemPoolTrimTo, hipDeviceSetMemPool, hipMemPoolSetAttribute, hipMemPoolSetAccess, hipMemPoolGetAccess,
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream);
/**
 * @brief Exports a memory pool to the requested handle type.
 *
 * Given an IPC capable mempool, create an OS handle to share the pool with another process.
 * A recipient process can convert the shareable handle into a mempool with @p hipMemPoolImportFromShareableHandle.
 * Individual pointers can then be shared with the @p hipMemPoolExportPointer and @p hipMemPoolImportPointer APIs.
 * The implementation of what the shareable handle is and how it can be transferred is defined by the requested
 * handle type.
 *
 * @note: To create an IPC capable mempool, create a mempool with a @p hipMemAllocationHandleType other
 * than @p hipMemHandleTypeNone.
 *
 * @param [out] shared_handle Pointer to the location in which to store the requested handle
 * @param [in] mem_pool       Pool to export
 * @param [in] handle_type    The type of handle to create
 * @param [in] flags          Must be 0
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
 *
 * @see hipMemPoolImportFromShareableHandle
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolExportToShareableHandle(
    void*                      shared_handle,
    hipMemPool_t               mem_pool,
    hipMemAllocationHandleType handle_type,
    unsigned int               flags);
/**
 * @brief Imports a memory pool from a shared handle.
 *
 * Specific allocations can be imported from the imported pool with @p hipMemPoolImportPointer.
 *
 * @note Imported memory pools do not support creating new allocations.
 * As such imported memory pools may not be used in @p hipDeviceSetMemPool
 * or @p hipMallocFromPoolAsync calls.
 *
 * @param [out] mem_pool     Returned memory pool
 * @param [in] shared_handle OS handle of the pool to open
 * @param [in] handle_type   The type of handle being imported
 * @param [in] flags         Must be 0
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
 *
 * @see hipMemPoolExportToShareableHandle
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolImportFromShareableHandle(
    hipMemPool_t*              mem_pool,
    void*                      shared_handle,
    hipMemAllocationHandleType handle_type,
    unsigned int               flags);
/**
 * @brief Export data to share a memory pool allocation between processes.
 *
 * Constructs @p export_data for sharing a specific allocation from an already shared memory pool.
 * The recipient process can import the allocation with the @p hipMemPoolImportPointer api.
 * The data is not a handle and may be shared through any IPC mechanism.
 *
 * @param[out] export_data  Returned export data
 * @param[in] dev_ptr       Pointer to memory being exported
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
 *
 * @see hipMemPoolImportPointer
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* dev_ptr);
/**
 * @brief Import a memory pool allocation from another process.
 *
 * Returns in @p dev_ptr a pointer to the imported memory.
 * The imported memory must not be accessed before the allocation operation completes
 * in the exporting process. The imported memory must be freed from all importing processes before
 * being freed in the exporting process. The pointer may be freed with @p hipFree
 * or @p hipFreeAsync. If @p hipFreeAsync is used, the free must be completed
 * on the importing process before the free operation on the exporting process.
 *
 * @note The @p hipFreeAsync api may be used in the exporting process before
 * the @p hipFreeAsync operation completes in its stream as long as the
 * @p hipFreeAsync in the exporting process specifies a stream with
 * a stream dependency on the importing process's @p hipFreeAsync.
 *
 * @param [out] dev_ptr     Pointer to imported memory
 * @param [in] mem_pool     Memory pool from which to import a pointer
 * @param [in] export_data  Data specifying the memory to import
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized, #hipErrorOutOfMemory
 *
 * @see hipMemPoolExportPointer
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemPoolImportPointer(
    void**                   dev_ptr,
    hipMemPool_t             mem_pool,
    hipMemPoolPtrExportData* export_data);
// Doxygen end of ordered memory allocator
/**
 * @}
 */

/**
 *  @brief Allocate device accessible page locked host memory [Deprecated]
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size in bytes
 *  @param[in]  flags Type of host memory allocation
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @warning This API is deprecated, use hipHostMalloc() instead
 */
DEPRECATED("use hipHostMalloc instead")
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
/**
 *  @brief Get Device pointer from Host Pointer allocated through hipHostMalloc
 *
 *  @param[out] devPtr Device Pointer mapped to passed host pointer
 *  @param[in]  hstPtr Host Pointer allocated through hipHostMalloc
 *  @param[in]  flags Flags to be passed for extension
 *
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
 *
 *  @see hipSetDeviceFlags, hipHostMalloc
 */
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
/**
 *  @brief Return flags associated with host pointer
 *
 *  @param[out] flagsPtr Memory location to store flags
 *  @param[in]  hostPtr Host Pointer allocated through hipHostMalloc
 *  @return #hipSuccess, #hipErrorInvalidValue
 *
 *  @see hipHostMalloc
 */
hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
/**
 *  @brief Register host memory so it can be accessed from the current device.
 *
 *  @param[out] hostPtr Pointer to host memory to be registered.
 *  @param[in] sizeBytes Size of the host memory
 *  @param[in] flags  See below.
 *
 *  Flags:
 *  - #hipHostRegisterDefault   Memory is Mapped and Portable
 *  - #hipHostRegisterPortable  Memory is considered registered by all contexts.  HIP only supports
 * one context so this is always assumed true.
 *  - #hipHostRegisterMapped    Map the allocation into the address space for the current device.
 * The device pointer can be obtained with #hipHostGetDevicePointer.
 *
 *
 *  After registering the memory, use #hipHostGetDevicePointer to obtain the mapped device pointer.
 *  On many systems, the mapped device pointer will have a different value than the mapped host
 * pointer.  Applications must use the device pointer in device code, and the host pointer in device
 * code.
 *
 *  On some systems, registered memory is pinned.  On some systems, registered memory may not be
 * actually be pinned but uses OS or hardware facilities to all GPU access to the host memory.
 *
 *  Developers are strongly encouraged to register memory blocks which are aligned to the host
 * cache-line size. (typically 64-bytes but can be obtains from the CPUID instruction).
 *
 *  If registering non-aligned pointers, the application must take care when register pointers from
 * the same cache line on different devices.  HIP's coarse-grained synchronization model does not
 * guarantee correct results if different devices write to different parts of the same cache block -
 * typically one of the writes will "win" and overwrite data from the other registered memory
 * region.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @see hipHostUnregister, hipHostGetFlags, hipHostGetDevicePointer
 */
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
/**
 *  @brief Un-register host pointer
 *
 *  @param[in] hostPtr Host pointer previously registered with #hipHostRegister
 *  @return Error code
 *
 *  @see hipHostRegister
 */
hipError_t hipHostUnregister(void* hostPtr);
/**
 *  Allocates at least width (in bytes) * height bytes of linear memory
 *  Padding may occur to ensure alighnment requirements are met for the given row
 *  The change in width size due to padding will be returned in *pitch.
 *  Currently the alignment is set to 128 bytes
 *
 *  @param[out] ptr Pointer to the allocated device memory
 *  @param[out] pitch Pitch for allocation (in bytes)
 *  @param[in]  width Requested pitched allocation width (in bytes)
 *  @param[in]  height Requested pitched allocation height
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return Error code
 *
 *  @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
 * hipMalloc3DArray, hipHostMalloc
 */
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);
/**
 *  Allocates at least width (in bytes) * height bytes of linear memory
 *  Padding may occur to ensure alighnment requirements are met for the given row
 *  The change in width size due to padding will be returned in *pitch.
 *  Currently the alignment is set to 128 bytes
 *
 *  @param[out] dptr  Pointer to the allocated device memory
 *  @param[out] pitch  Pitch for allocation (in bytes)
 *  @param[in]  widthInBytes  Requested pitched allocation width (in bytes)
 *  @param[in]  height  Requested pitched allocation height
 *  @param[in]  elementSizeBytes  The size of element bytes, should be 4, 8 or 16
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *  The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array.
 *  Given the row and column of an array element of type T, the address is computed as:
 *  T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;
 *
 *  @return Error code
 *
 *  @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
 * hipMalloc3DArray, hipHostMalloc
 */
hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height,
   unsigned int elementSizeBytes);
/**
 *  @brief Free memory allocated by the hcc hip memory allocation API.
 *  This API performs an implicit hipDeviceSynchronize() call.
 *  If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #hipSuccess
 *  @return #hipErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated
 * with hipHostMalloc)
 *
 *  @see hipMalloc, hipMallocPitch, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
 * hipMalloc3DArray, hipHostMalloc
 */
hipError_t hipFree(void* ptr);
/**
 *  @brief Free memory allocated by the hcc hip host memory allocation API [Deprecated]
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #hipSuccess,
 *          #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated
 *  with hipMalloc)
 *
 *  @warning  This API is deprecated, use hipHostFree() instead
 */
DEPRECATED("use hipHostFree instead")
hipError_t hipFreeHost(void* ptr);
/**
 *  @brief Free memory allocated by the hcc hip host memory allocation API
 *  This API performs an implicit hipDeviceSynchronize() call.
 *  If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #hipSuccess,
 *          #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
 * hipMalloc)
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D,
 * hipMalloc3DArray, hipHostMalloc
 */
hipError_t hipHostFree(void* ptr);
/**
 *  @brief Copy data from src to dst.
 *
 *  It supports memory from host to device,
 *  device to host, device to device and host to host
 *  The src and dst must not overlap.
 *
 *  For hipMemcpy, the copy is always performed by the current device (set by hipSetDevice).
 *  For multi-gpu or peer-to-peer configurations, it is recommended to set the current device to the
 *  device where the src data is physically located. For optimal peer-to-peer copies, the copy device
 *  must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with copy
 *  agent as the current device and src/dest as the peerDevice argument.  if this is not done, the
 *  hipMemcpy will still work, but will perform the copy using a staging buffer on the host.
 *  Calling hipMemcpy with dst and src pointers that do not match the hipMemcpyKind results in
 *  undefined behavior.
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]  src Data being copy from
 *  @param[in]  sizeBytes Data size in bytes
 *  @param[in]  kind Kind of transfer
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorUnknown
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
/**
 *  @brief Memory copy on the stream.
 *  It allows single or multiple devices to do memory copy on single or multiple streams.
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]  src Data being copy from
 *  @param[in]  sizeBytes Data size in bytes
 *  @param[in]  kind Kind of transfer
 *  @param[in]  stream Valid stream
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorUnknown, #hipErrorContextIsDestroyed
 *
 *  @see hipMemcpy, hipStreamCreate, hipStreamSynchronize, hipStreamDestroy, hipSetDevice, hipLaunchKernelGGL
 *
 */
hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,
                               hipMemcpyKind kind, hipStream_t stream);
/**
 *  @brief Copy data from Host to Device
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes);
/**
 *  @brief Copy data from Device to Host
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);
/**
 *  @brief Copy data from Device to Device
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes);
/**
 *  @brief Copy data from Host to Device asynchronously
 *
 *  @param[out]  dst  Data being copy to
 *  @param[in]   src  Data being copy from
 *  @param[in]   sizeBytes  Data size in bytes
 *  @param[in]   stream  Stream identifier
 *
 *  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream);
/**
 *  @brief Copy data from Device to Host asynchronously
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *  @param[in]   stream  Stream identifier
 *
 *  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream);
/**
 *  @brief Copy data from Device to Device asynchronously
 *
 *  @param[out]  dst  Data being copy to
 *  @param[in]   src  Data being copy from
 *  @param[in]   sizeBytes  Data size in bytes
 *  @param[in]   stream  Stream identifier
 *
 *  @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream);

/**
 *  @brief Returns a global pointer from a module.
 *  Returns in *dptr and *bytes the pointer and size of the global of name name located in module hmod.
 *  If no variable of that name exists, it returns hipErrorNotFound. Both parameters dptr and bytes are optional.
 *  If one of them is NULL, it is ignored and hipSuccess is returned.
 *
 *  @param[out]  dptr  Returns global device pointer
 *  @param[out]  bytes Returns global size in bytes
 *  @param[in]   hmod  Module to retrieve global from
 *  @param[in]   name  Name of global to retrieve
 *
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotFound, #hipErrorInvalidContext
 *
 */
hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
    hipModule_t hmod, const char* name);

/**
 *  @brief Gets device pointer associated with symbol on the device.
 *
 *  @param[out]  devPtr  pointer to the device associated the symbole
 *  @param[in]   symbol  pointer to the symbole of the device
 *
 *  @return #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol);

/**
 *  @brief Gets the size of the given symbol on the device.
 *
 *  @param[in]   symbol  pointer to the device symbole
 *  @param[out]  size  pointer to the size
 *
 *  @return #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipGetSymbolSize(size_t* size, const void* symbol);

/**
 *  @brief Copies data to the given symbol on the device.
 * Symbol HIP APIs allow a kernel to define a device-side data symbol which can be accessed on
 * the host side. The symbol can be in __constant or device space.
 * Note that the symbol name needs to be encased in the HIP_SYMBOL macro.
 * This also applies to hipMemcpyFromSymbol, hipGetSymbolAddress, and hipGetSymbolSize.
 * For detail usage, see the example at
 * https://github.com/ROCm/HIP/blob/develop/docs/user_guide/hip_porting_guide.md
 *
 *  @param[out]  symbol  pointer to the device symbole
 *  @param[in]   src  pointer to the source address
 *  @param[in]   sizeBytes  size in bytes to copy
 *  @param[in]   offset  offset in bytes from start of symbole
 *  @param[in]   kind  type of memory transfer
 *
 *  @return #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipMemcpyToSymbol(const void* symbol, const void* src,
                             size_t sizeBytes, size_t offset __dparm(0),
                             hipMemcpyKind kind __dparm(hipMemcpyHostToDevice));

/**
 *  @brief Copies data to the given symbol on the device asynchronously.
 *
 *  @param[out]  symbol  pointer to the device symbole
 *  @param[in]   src  pointer to the source address
 *  @param[in]   sizeBytes  size in bytes to copy
 *  @param[in]   offset  offset in bytes from start of symbole
 *  @param[in]   kind  type of memory transfer
 *  @param[in]   stream  stream identifier
 *
 *  @return #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src,
                                  size_t sizeBytes, size_t offset,
                                  hipMemcpyKind kind, hipStream_t stream __dparm(0));

/**
 *  @brief Copies data from the given symbol on the device.
 *
 *  @param[out]  dst  Returns pointer to destinition memory address
 *  @param[in]   symbol  Pointer to the symbole address on the device
 *  @param[in]   sizeBytes  Size in bytes to copy
 *  @param[in]   offset  Offset in bytes from the start of symbole
 *  @param[in]   kind  Type of memory transfer
 *
 *  @return #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol,
                               size_t sizeBytes, size_t offset __dparm(0),
                               hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost));

/**
 *  @brief Copies data from the given symbol on the device asynchronously.
 *
 *  @param[out]  dst  Returns pointer to destinition memory address
 *  @param[in]   symbol  pointer to the symbole address on the device
 *  @param[in]   sizeBytes  size in bytes to copy
 *  @param[in]   offset  offset in bytes from the start of symbole
 *  @param[in]   kind  type of memory transfer
 *  @param[in]   stream  stream identifier
 *
 *  @return #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol,
                                    size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind,
                                    hipStream_t stream __dparm(0));
/**
 *  @brief Copy data from src to dst asynchronously.
 *
 *  @warning If host or dest are not pinned, the memory copy will be performed synchronously.  For
 * best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.
 *
 *  @warning on HCC hipMemcpyAsync does not support overlapped H2D and D2H copies.
 *  For hipMemcpy, the copy is always performed by the device associated with the specified stream.
 *
 *  For multi-gpu or peer-to-peer configurations, it is recommended to use a stream which is a
 * attached to the device where the src data is physically located. For optimal peer-to-peer copies,
 * the copy device must be able to access the src and dst pointers (by calling
 * hipDeviceEnablePeerAccess with copy agent as the current device and src/dest as the peerDevice
 * argument.  if this is not done, the hipMemcpy will still work, but will perform the copy using a
 * staging buffer on the host.
 *
 *  @param[out] dst Data being copy to
 *  @param[in]  src Data being copy from
 *  @param[in]  sizeBytes Data size in bytes
 *  @param[in]  kind  Type of memory transfer
 *  @param[in]  stream  Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorUnknown
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
 * hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyToSymbol,
 * hipMemcpyFromSymbol, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync,
 * hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync,
 * hipMemcpyFromSymbolAsync
 */
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream __dparm(0));
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * byte value value.
 *
 *  @param[out] dst  Data being filled
 *  @param[in]  value  Value to be set
 *  @param[in]  sizeBytes  Data size in bytes
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * byte value value.
 *
 *  @param[out] dest  Data ptr to be filled
 *  @param[in]  value  Value to be set
 *  @param[in]  count  Number of values to be set
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count);
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * byte value value.
 *
 * hipMemsetD8Async() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 *  @param[out] dest  Data ptr to be filled
 *  @param[in]  value  Constant value to be set
 *  @param[in]  count  Number of values to be set
 *  @param[in]  stream  Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count, hipStream_t stream __dparm(0));
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * short value value.
 *
 *  @param[out] dest  Data ptr to be filled
 *  @param[in]  value  Constant value to be set
 *  @param[in]  count  Number of values to be set
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count);
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * short value value.
 *
 * hipMemsetD16Async() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 *  @param[out] dest  Data ptr to be filled
 *  @param[in]  value  Constant value to be set
 *  @param[in]  count  Number of values to be set
 *  @param[in]  stream  Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count, hipStream_t stream __dparm(0));
/**
 *  @brief Fills the memory area pointed to by dest with the constant integer
 * value for specified number of times.
 *
 *  @param[out] dest  Data being filled
 *  @param[in]  value  Constant value to be set
 *  @param[in]  count  Number of values to be set
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count);
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant
 * byte value value.
 *
 * hipMemsetAsync() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 *  @param[out] dst Pointer to device memory
 *  @param[in]  value  Value to set for each byte of specified memory
 *  @param[in]  sizeBytes  Size in bytes to set
 *  @param[in]  stream  Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream __dparm(0));
/**
 *  @brief Fills the memory area pointed to by dev with the constant integer
 * value for specified number of times.
 *
 *  hipMemsetD32Async() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 *  @param[out] dst Pointer to device memory
 *  @param[in]  value  Value to set for each byte of specified memory
 *  @param[in]  count  Number of values to be set
 *  @param[in]  stream  Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count,
                             hipStream_t stream __dparm(0));
/**
 *  @brief Fills the memory area pointed to by dst with the constant value.
 *
 *  @param[out] dst Pointer to device memory
 *  @param[in]  pitch  Data size in bytes
 *  @param[in]  value  Constant value to be set
 *  @param[in]  width
 *  @param[in]  height
 *  @return #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height);
/**
 *  @brief Fills asynchronously the memory area pointed to by dst with the constant value.
 *
 *  @param[in]  dst Pointer to 2D device memory
 *  @param[in]  pitch  Pitch size in bytes
 *  @param[in]  value  Value to be set for each byte of specified memory
 *  @param[in]  width  Width of matrix set columns in bytes
 *  @param[in]  height  Height of matrix set rows in bytes
 *  @param[in]  stream  Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height,hipStream_t stream __dparm(0));
/**
 *  @brief Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
 *
 *  @param[in] pitchedDevPtr  Pointer to pitched device memory
 *  @param[in]  value  Value to set for each byte of specified memory
 *  @param[in]  extent  Size parameters for width field in bytes in device memory
 *  @return #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent );
/**
 *  @brief Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
 *
 *  @param[in] pitchedDevPtr  Pointer to pitched device memory
 *  @param[in]  value  Value to set for each byte of specified memory
 *  @param[in]  extent  Size parameters for width field in bytes in device memory
 *  @param[in]  stream  Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent ,hipStream_t stream __dparm(0));
/**
 * @brief Query memory info.
 *
 * On ROCM, this function gets the actual free memory left on the current device, so supports
 * the cases while running multi-workload (such as multiple processes, multiple threads, and
 * multiple GPUs).
 *
 * @warning On Windows, the free memory only accounts for memory allocated by this process and may
 * be optimistic.
 *
 * @param[out] free Returns free memory on the current device in bytes
 * @param[out] total Returns total allocatable memory on the current device in bytes
 *
 * @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 **/
hipError_t hipMemGetInfo(size_t* free, size_t* total);

/**
 * @brief Get allocated memory size via memory pointer.
 *
 * This function gets the allocated shared virtual memory size from memory pointer.
 *
 * @param[in] ptr Pointer to allocated memory
 * @param[out] size Returns the allocated memory size in bytes
 *
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 **/
hipError_t hipMemPtrGetInfo(void* ptr, size_t* size);
/**
 *  @brief Allocate an array on the device.
 *
 *  @param[out]  array  Pointer to allocated array in device memory
 *  @param[in]   desc   Requested channel format
 *  @param[in]   width  Requested array allocation width
 *  @param[in]   height Requested array allocation height
 *  @param[in]   flags  Requested properties of allocated array
 *  @return      #hipSuccess, #hipErrorOutOfMemory
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
 */
hipError_t hipMallocArray(hipArray_t* array, const hipChannelFormatDesc* desc, size_t width,
                          size_t height __dparm(0), unsigned int flags __dparm(hipArrayDefault));
/**
 *  @brief Create an array memory pointer on the device.
 *
 *  @param[out]  pHandle  Pointer to the array memory
 *  @param[in]   pAllocateArray   Requested array desciptor
 *
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 *  @see hipMallocArray, hipArrayDestroy, hipFreeArray
 */
hipError_t hipArrayCreate(hipArray_t* pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray);
 /**
 *  @brief Destroy an array memory pointer on the device.
 *
 *  @param[in]  array  Pointer to the array memory
 *
 *  @return      #hipSuccess, #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipFreeArray
 */
hipError_t hipArrayDestroy(hipArray_t array);
/**
 *  @brief Create a 3D array memory pointer on the device.
 *
 *  @param[out]  array  Pointer to the 3D array memory
 *  @param[in]   pAllocateArray   Requested array desciptor
 *
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 *  @see hipMallocArray, hipArrayDestroy, hipFreeArray
 */
hipError_t hipArray3DCreate(hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray);
/**
 *  @brief Create a 3D memory pointer on the device.
 *
 *  @param[out]  pitchedDevPtr  Pointer to the 3D memory
 *  @param[in]   extent   Requested extent
 *
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 *  @see hipMallocPitch, hipMemGetInfo, hipFree
 */
hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent);
/**
 *  @brief Frees an array on the device.
 *
 *  @param[in]  array  Pointer to array to free
 *  @return     #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
 */
hipError_t hipFreeArray(hipArray_t array);
/**
 *  @brief Allocate an array on the device.
 *
 *  @param[out]  array  Pointer to allocated array in device memory
 *  @param[in]   desc   Requested channel format
 *  @param[in]   extent Requested array allocation width, height and depth
 *  @param[in]   flags  Requested properties of allocated array
 *  @return      #hipSuccess, #hipErrorOutOfMemory
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
 */
hipError_t hipMalloc3DArray(hipArray_t* array, const struct hipChannelFormatDesc* desc,
                            struct hipExtent extent, unsigned int flags);
/**
 * @brief Gets info about the specified array
 *
 * @param[out] desc   - Returned array type
 * @param[out] extent - Returned array shape. 2D arrays will have depth of zero
 * @param[out] flags  - Returned array flags
 * @param[in]  array  - The HIP array to get info for
 *
 * @return #hipSuccess, #hipErrorInvalidValue #hipErrorInvalidHandle
 *
 * @see hipArrayGetDescriptor, hipArray3DGetDescriptor
 */
hipError_t hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent, unsigned int* flags,
                           hipArray_t array);
/**
 * @brief Gets a 1D or 2D array descriptor
 *
 * @param[out] pArrayDescriptor - Returned array descriptor
 * @param[in]  array            - Array to get descriptor of
 *
 * @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue #hipErrorInvalidHandle
 *
 * @see hipArray3DCreate, hipArray3DGetDescriptor, hipArrayCreate, hipArrayDestroy, hipMemAlloc,
 * hipMemAllocHost, hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,
 * hipMemcpy3D, hipMemcpy3DAsync, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync,
 * hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync,
 * hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoD, hipMemcpyHtoDAsync, hipMemFree,
 * hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo, hipMemHostAlloc,
 * hipMemHostGetDevicePointer, hipMemsetD8, hipMemsetD16, hipMemsetD32, hipArrayGetInfo
 */
hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor, hipArray_t array);
/**
 * @brief Gets a 3D array descriptor
 *
 * @param[out] pArrayDescriptor - Returned 3D array descriptor
 * @param[in]  array            - 3D array to get descriptor of
 *
 * @return #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue #hipErrorInvalidHandle, #hipErrorContextIsDestroyed
 *
 * @see hipArray3DCreate, hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc,
 * hipMemAllocHost, hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,
 * hipMemcpy3D, hipMemcpy3DAsync, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync,
 * hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync,
 * hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoD, hipMemcpyHtoDAsync, hipMemFree,
 * hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo, hipMemHostAlloc,
 * hipMemHostGetDevicePointer, hipMemsetD8, hipMemsetD16, hipMemsetD32, hipArrayGetInfo
 */
hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray_t array);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst    Destination memory address
 *  @param[in]   dpitch Pitch of destination memory
 *  @param[in]   src    Source memory address
 *  @param[in]   spitch Pitch of source memory
 *  @param[in]   width  Width of matrix transfer (columns in bytes)
 *  @param[in]   height Height of matrix transfer (rows)
 *  @param[in]   kind   Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind);
/**
 *  @brief Copies memory for 2D arrays.
 *  @param[in]   pCopy Parameters for the memory copy
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 *  #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
 * hipMemcpyToSymbol, hipMemcpyAsync
*/
hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy);
/**
 *  @brief Copies memory for 2D arrays.
 *  @param[in]   pCopy Parameters for the memory copy
 *  @param[in]   stream Stream to use
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
 * hipMemcpyToSymbol, hipMemcpyAsync
*/
hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream __dparm(0));
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst    Destination memory address
 *  @param[in]   dpitch Pitch of destination memory
 *  @param[in]   src    Source memory address
 *  @param[in]   spitch Pitch of source memory
 *  @param[in]   width  Width of matrix transfer (columns in bytes)
 *  @param[in]   height Height of matrix transfer (rows)
 *  @param[in]   kind   Type of transfer
 *  @param[in]   stream Stream to use
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream __dparm(0));
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst     Destination memory address
 *  @param[in]   wOffset Destination starting X offset
 *  @param[in]   hOffset Destination starting Y offset
 *  @param[in]   src     Source memory address
 *  @param[in]   spitch  Pitch of source memory
 *  @param[in]   width   Width of matrix transfer (columns in bytes)
 *  @param[in]   height  Height of matrix transfer (rows)
 *  @param[in]   kind    Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst     Destination memory address
 *  @param[in]   wOffset Destination starting X offset
 *  @param[in]   hOffset Destination starting Y offset
 *  @param[in]   src     Source memory address
 *  @param[in]   spitch  Pitch of source memory
 *  @param[in]   width   Width of matrix transfer (columns in bytes)
 *  @param[in]   height  Height of matrix transfer (rows)
 *  @param[in]   kind    Type of transfer
 *  @param[in]   stream    Accelerator view which the copy is being enqueued
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                   size_t spitch, size_t width, size_t height, hipMemcpyKind kind,
                                   hipStream_t stream __dparm(0));
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst     Destination memory address
 *  @param[in]   wOffset Destination starting X offset
 *  @param[in]   hOffset Destination starting Y offset
 *  @param[in]   src     Source memory address
 *  @param[in]   count   size in bytes to copy
 *  @param[in]   kind    Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 *  hipMemcpyAsync
 *  @warning  This API is deprecated.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst       Destination memory address
 *  @param[in]   srcArray  Source memory address
 *  @param[in]   wOffset   Source starting X offset
 *  @param[in]   hOffset   Source starting Y offset
 *  @param[in]   count     Size in bytes to copy
 *  @param[in]   kind      Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 * @warning  This API is deprecated.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst       Destination memory address
 *  @param[in]   dpitch    Pitch of destination memory
 *  @param[in]   src       Source memory address
 *  @param[in]   wOffset   Source starting X offset
 *  @param[in]   hOffset   Source starting Y offset
 *  @param[in]   width     Width of matrix transfer (columns in bytes)
 *  @param[in]   height    Height of matrix transfer (rows)
 *  @param[in]   kind      Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DFromArray( void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind);
/**
 *  @brief Copies data between host and device asynchronously.
 *
 *  @param[in]   dst       Destination memory address
 *  @param[in]   dpitch    Pitch of destination memory
 *  @param[in]   src       Source memory address
 *  @param[in]   wOffset   Source starting X offset
 *  @param[in]   hOffset   Source starting Y offset
 *  @param[in]   width     Width of matrix transfer (columns in bytes)
 *  @param[in]   height    Height of matrix transfer (rows)
 *  @param[in]   kind      Type of transfer
 *  @param[in]   stream    Accelerator view which the copy is being enqueued
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DFromArrayAsync( void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream __dparm(0));
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst       Destination memory address
 *  @param[in]   srcArray  Source array
 *  @param[in]   srcOffset Offset in bytes of source array
 *  @param[in]   count     Size of memory copy in bytes
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpyAtoH(void* dst, hipArray_t srcArray, size_t srcOffset, size_t count);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dstArray   Destination memory address
 *  @param[in]   dstOffset  Offset in bytes of destination array
 *  @param[in]   srcHost    Source host pointer
 *  @param[in]   count      Size of memory copy in bytes
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void* srcHost, size_t count);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   p   3D memory copy parameters
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p);
/**
 *  @brief Copies data between host and device asynchronously.
 *
 *  @param[in]   p        3D memory copy parameters
 *  @param[in]   stream   Stream to use
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream __dparm(0));
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   pCopy   3D memory copy parameters
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 *  #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy);
/**
 *  @brief Copies data between host and device asynchronously.
 *
 *  @param[in]   pCopy    3D memory copy parameters
 *  @param[in]   stream   Stream to use
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 *  #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream);
// doxygen end Memory
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup PeerToPeer PeerToPeer Device Memory Access
 *  @{
 *  @warning PeerToPeer support is experimental.
 *  This section describes the PeerToPeer device memory access functions of HIP runtime API.
 */
/**
 * @brief Determine if a device can access a peer's memory.
 *
 * @param [out] canAccessPeer Returns the peer access capability (0 or 1)
 * @param [in] deviceId - device from where memory may be accessed.
 * @param [in] peerDeviceId - device where memory is physically located
 *
 * Returns "1" in @p canAccessPeer if the specified @p device is capable
 * of directly accessing memory physically located on peerDevice , or "0" if not.
 *
 * Returns "0" in @p canAccessPeer if deviceId == peerDeviceId, and both are valid devices : a
 * device is not a peer of itself.
 *
 * @returns #hipSuccess,
 * @returns #hipErrorInvalidDevice if deviceId or peerDeviceId are not valid devices
 */
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
/**
 * @brief Enable direct access from current device's virtual address space to memory allocations
 * physically located on a peer device.
 *
 * Memory which already allocated on peer device will be mapped into the address space of the
 * current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
 * the address space of the current device when the memory is allocated. The peer memory remains
 * accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
 *
 *
 * @param [in] peerDeviceId  Peer device to enable direct access to from the current device
 * @param [in] flags  Reserved for future use, must be zero
 *
 * Returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
 * @returns #hipErrorPeerAccessAlreadyEnabled if peer access is already enabled for this device.
 */
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
/**
 * @brief Disable direct access from current device's virtual address space to memory allocations
 * physically located on a peer device.
 *
 * Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
 * enabled from the current device.
 *
 * @param [in] peerDeviceId  Peer device to disable direct access to
 *
 * @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
 */
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);
/**
 * @brief Get information on memory allocations.
 *
 * @param [out] pbase - BAse pointer address
 * @param [out] psize - Size of allocation
 * @param [in]  dptr- Device Pointer
 *
 * @returns #hipSuccess, #hipErrorInvalidDevicePointer
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr);
#ifndef USE_PEER_NON_UNIFIED
#define USE_PEER_NON_UNIFIED 1
#endif
#if USE_PEER_NON_UNIFIED == 1
/**
 * @brief Copies memory from one device to memory on another device.
 *
 * @param [out] dst - Destination device pointer.
 * @param [in] dstDeviceId - Destination device
 * @param [in] src - Source device pointer
 * @param [in] srcDeviceId - Source device
 * @param [in] sizeBytes - Size of memory copy in bytes
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
 */
hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId,
                         size_t sizeBytes);
/**
 * @brief Copies memory from one device to memory on another device.
 *
 * @param [out] dst - Destination device pointer.
 * @param [in] dstDeviceId - Destination device
 * @param [in] src - Source device pointer
 * @param [in] srcDevice - Source device
 * @param [in] sizeBytes - Size of memory copy in bytes
 * @param [in] stream - Stream identifier
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
 */
hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice,
                              size_t sizeBytes, hipStream_t stream __dparm(0));
#endif
// doxygen end PeerToPeer
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Context Context Management [Deprecated]
 *  @{
 *  This section describes the context management functions of HIP runtime API.
 *
 *  @warning
 *
 *  On the AMD platform, context management APIs are deprecated as there are better alternate
 *  interfaces, such as using hipSetDevice and stream APIs to achieve the required functionality.
 *
 *  On the NVIDIA platform, CUDA supports the driver API that defines "Context" and "Devices" as
 *  separate entities. Each context contains a single device, which can theoretically have multiple
 *  contexts. HIP initially added limited support for these APIs to facilitate easy porting from
 *  existing driver codes.
 *
 *  These APIs are only for equivalent driver APIs on the NVIDIA platform.
 * 
 */

/**
 * @brief Create a context and set it as current/default context
 *
 * @param [out] ctx  Context to create
 * @param [in] flags  Context creation flags
 * @param [in] device  device handle
 *
 * @return #hipSuccess
 *
 * @see hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxPushCurrent,
 * hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
/**
 * @brief Destroy a HIP context.
 *
 * @param [in] ctx Context to destroy
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipCtxCreate, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,hipCtxSetCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxDestroy(hipCtx_t ctx);
/**
 * @brief Pop the current/default context and return the popped context.
 *
 * @param [out] ctx  The current context to pop
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxSetCurrent, hipCtxGetCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPopCurrent(hipCtx_t* ctx);
/**
 * @brief Push the context to be set as current/ default context
 *
 * @param [in] ctx  The current context to push
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPushCurrent(hipCtx_t ctx);
/**
 * @brief Set the passed context as current/default
 *
 * @param [in] ctx The context to set as current
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetCurrent(hipCtx_t ctx);
/**
 * @brief Get the handle of the current/ default context
 *
 * @param [out] ctx  The context to get as current
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetCurrent(hipCtx_t* ctx);
/**
 * @brief Get the handle of the device associated with current/default context
 *
 * @param [out] device The device from the current context
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetDevice(hipDevice_t* device);
/**
 * @brief Returns the approximate HIP api version.
 *
 * @param [in]  ctx Context to check
 * @param [out] apiVersion API version to get
 *
 * @return #hipSuccess
 *
 * @warning The HIP feature set does not correspond to an exact CUDA SDK api revision.
 * This function always set *apiVersion to 4 as an approximation though HIP supports
 * some features which were introduced in later CUDA SDK revisions.
 * HIP apps code should not rely on the api revision number here and should
 * use arch feature flags to test device capabilities or conditional compilation.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion);
/**
 * @brief Get Cache configuration for a specific function
 *
 * @param [out] cacheConfig  Cache configuration
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
 * ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig);
/**
 * @brief Set L1/Shared cache partition.
 *
 * @param [in] cacheConfig  Cache configuration to set
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
 * ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig);
/**
 * @brief Set Shared memory bank configuration.
 *
 * @param [in] config  Shared memory configuration to set
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);
/**
 * @brief Get Shared memory bank configuration.
 *
 * @param [out] pConfig  Pointer of shared memory configuration
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);
/**
 * @brief Blocks until the default context has completed all preceding requested tasks.
 *
 * @return #hipSuccess
 *
 * @warning This function waits for all streams on the default context to complete execution, and
 * then returns.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSynchronize(void);
/**
 * @brief Return flags used for creating default context.
 *
 * @param [out] flags  Pointer of flags
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetFlags(unsigned int* flags);
/**
 * @brief Enables direct access to memory allocations in a peer context.
 *
 * Memory which already allocated on peer device will be mapped into the address space of the
 * current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
 * the address space of the current device when the memory is allocated. The peer memory remains
 * accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
 *
 *
 * @param [in] peerCtx  Peer context
 * @param [in] flags  flags, need to set as 0
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
 * #hipErrorPeerAccessAlreadyEnabled
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 * @warning PeerToPeer support is experimental.
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);
/**
 * @brief Disable direct access from current context's virtual address space to memory allocations
 * physically located on a peer context.Disables direct access to memory allocations in a peer
 * context and unregisters any registered allocations.
 *
 * Returns #hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
 * enabled from the current device.
 *
 * @param [in] peerCtx  Peer context to be disabled
 *
 * @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 * @warning PeerToPeer support is experimental.
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);

/**
 * @brief Get the state of the primary context.
 *
 * @param [in] dev  Device to get primary context flags for
 * @param [out] flags  Pointer to store flags
 * @param [out] active  Pointer to store context state; 0 = inactive, 1 = active
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent driver API on the
 * NVIDIA platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active);
/**
 * @brief Release the primary context on the GPU.
 *
 * @param [in] dev  Device which primary context is released
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 * @warning This function return #hipSuccess though doesn't release the primaryCtx by design on
 * HIP/HCC path.
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent driver API on the NVIDIA
 * platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);
/**
 * @brief Retain the primary context on the GPU.
 *
 * @param [out] pctx  Returned context handle of the new context
 * @param [in] dev  Device which primary context is released
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent driver API on the NVIDIA
 * platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);
/**
 * @brief Resets the primary context on the GPU.
 *
 * @param [in] dev  Device which primary context is reset
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent driver API on the NVIDIA
 * platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);
/**
 * @brief Set flags for the primary context.
 *
 * @param [in] dev  Device for which the primary context flags are set
 * @param [in] flags  New flags for the device
 *
 * @returns #hipSuccess, #hipErrorContextAlreadyInUse
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 *
 * @warning  This API is deprecated on the AMD platform, only for equivalent driver API on the NVIDIA
 * platform.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);
// doxygen end Context Management
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *
 *  @defgroup Module Module Management
 *  @{
 *  @ingroup API
 *  This section describes the module management functions of HIP runtime API.
 *
 */
/**
 * @brief Loads code object from file into a module the currrent context.
 *
 * @param [in] fname  Filename of code object to load

 * @param [out] module  Module
 *
 * @warning File/memory resources allocated in this function are released only in hipModuleUnload.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidContext, #hipErrorFileNotFound,
 * #hipErrorOutOfMemory, #hipErrorSharedObjectInitFailed, #hipErrorNotInitialized
 *
 */
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
/**
 * @brief Frees the module
 *
 * @param [in] module  Module to free
 *
 * @returns #hipSuccess, #hipErrorInvalidResourceHandle
 *
 * The module is freed, and the code objects associated with it are destroyed.
 */
hipError_t hipModuleUnload(hipModule_t module);
/**
 * @brief Function with kname will be extracted if present in module
 *
 * @param [in] module  Module to get function from
 * @param [in] kname  Pointer to the name of function
 * @param [out] function  Pointer to function handle
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidContext, #hipErrorNotInitialized,
 * #hipErrorNotFound,
 */
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
/**
 * @brief Find out attributes for a given function.
 *
 * @param [out] attr  Attributes of funtion
 * @param [in] func  Pointer to the function handle
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDeviceFunction
 */
hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func);
/**
 * @brief Find out a specific attribute for a given function.
 *
 * @param [out] value  Pointer to the value
 * @param [in]  attrib  Attributes of the given funtion
 * @param [in]  hfunc  Function to get attributes from
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDeviceFunction
 */
hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc);
/**
 * @brief returns the handle of the texture reference with the name from the module.
 *
 * @param [in] hmod  Module
 * @param [in] name  Pointer of name of texture reference
 * @param [out] texRef  Pointer of texture reference
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorNotFound, #hipErrorInvalidValue
 */
hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name);
/**
 * @brief builds module from code object which resides in host memory. Image is pointer to that
 * location.
 *
 * @param [in] image  The pointer to the location of data
 * @param [out] module  Retuned module
 *
 * @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
 */
hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
/**
 * @brief builds module from code object which resides in host memory. Image is pointer to that
 * location. Options are not used. hipModuleLoadData is called.
 *
 * @param [in] image  The pointer to the location of data
 * @param [out] module  Retuned module
 * @param [in] numOptions Number of options
 * @param [in] options Options for JIT
 * @param [in] optionValues  Option values for JIT
 *
 * @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
 */
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionValues);
/**
 * @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
 * to kernelparams or extra
 *
 * @param [in] f         Kernel to launch.
 * @param [in] gridDimX  X grid dimension specified as multiple of blockDimX.
 * @param [in] gridDimY  Y grid dimension specified as multiple of blockDimY.
 * @param [in] gridDimZ  Z grid dimension specified as multiple of blockDimZ.
 * @param [in] blockDimX X block dimensions specified in work-items
 * @param [in] blockDimY Y grid dimension specified in work-items
 * @param [in] blockDimZ Z grid dimension specified in work-items
 * @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
 * default stream is used with associated synchronization rules.
 * @param [in] kernelParams  Kernel parameters to launch
 * @param [in] extra     Pointer to kernel arguments.   These are passed directly to the kernel and
 * must be in the memory layout and alignment expected by the kernel.
 * All passed arguments must be naturally aligned according to their type. The memory address of each
 * argument should be a multiple of its size in bytes. Please refer to hip_porting_driver_api.md
 * for sample usage.
 *
 * Please note, HIP does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32. So gridDim.x * blockDim.x, gridDim.y * blockDim.y
 * and gridDim.z * blockDim.z are always less than 2^32.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue
 */
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra);
/**
 * @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
 * to kernelParams, where thread blocks can cooperate and synchronize as they execute
 *
 * @param [in] f              Kernel to launch.
 * @param [in] gridDimX       X grid dimension specified as multiple of blockDimX.
 * @param [in] gridDimY       Y grid dimension specified as multiple of blockDimY.
 * @param [in] gridDimZ       Z grid dimension specified as multiple of blockDimZ.
 * @param [in] blockDimX      X block dimension specified in work-items.
 * @param [in] blockDimY      Y block dimension specified in work-items.
 * @param [in] blockDimZ      Z block dimension specified in work-items.
 * @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream         Stream where the kernel should be dispatched. May be 0,
 * in which case the default stream is used with associated synchronization rules.
 * @param [in] kernelParams   A list of kernel arguments.
 *
 * Please note, HIP does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidHandle, #hipErrorInvalidImage, #hipErrorInvalidValue,
 * #hipErrorInvalidConfiguration, #hipErrorLaunchFailure, #hipErrorLaunchOutOfResources,
 * #hipErrorLaunchTimeOut, #hipErrorCooperativeLaunchTooLarge, #hipErrorSharedObjectInitFailed
 */
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX,
                                            unsigned int gridDimY, unsigned int gridDimZ,
                                            unsigned int blockDimX, unsigned int blockDimY,
                                            unsigned int blockDimZ, unsigned int sharedMemBytes,
                                            hipStream_t stream, void** kernelParams);
/**
 * @brief Launches kernels on multiple devices where thread blocks can cooperate and
 * synchronize as they execute.
 *
 * @param [in] launchParamsList         List of launch parameters, one per device.
 * @param [in] numDevices               Size of the launchParamsList array.
 * @param [in] flags                    Flags to control launch behavior.
 *
 * @returns #hipSuccess, #hipErrorDeinitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidHandle, #hipErrorInvalidImage, #hipErrorInvalidValue,
 * #hipErrorInvalidConfiguration, #hipErrorInvalidResourceHandle, #hipErrorLaunchFailure,
 * #hipErrorLaunchOutOfResources, #hipErrorLaunchTimeOut, #hipErrorCooperativeLaunchTooLarge,
 * #hipErrorSharedObjectInitFailed
 */
hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                       unsigned int numDevices,
                                                       unsigned int flags);
/**
 * @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
 * to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute
 *
 * @param [in] f         Kernel to launch.
 * @param [in] gridDim   Grid dimensions specified as multiple of blockDim.
 * @param [in] blockDimX  Block dimensions specified in work-items
 * @param [in] kernelParams A list of kernel arguments
 * @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
 * default stream is used with associated synchronization rules.
 *
 * Please note, HIP does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue, #hipErrorCooperativeLaunchTooLarge
 */
hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream);
/**
 * @brief Launches kernels on multiple devices where thread blocks can cooperate and
 * synchronize as they execute.
 *
 * @param [in] launchParamsList         List of launch parameters, one per device.
 * @param [in] numDevices               Size of the launchParamsList array.
 * @param [in] flags                    Flags to control launch behavior.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
 *  #hipErrorCooperativeLaunchTooLarge
 */
hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                 int  numDevices, unsigned int  flags);
/**
 * @brief Launches kernels on multiple devices and guarantees all specified kernels are dispatched
 * on respective streams before enqueuing any other work on the specified streams from any other threads
 *
 *
 * @param [in] launchParamsList          List of launch parameters, one per device.
 * @param [in] numDevices               Size of the launchParamsList array.
 * @param [in] flags                    Flags to control launch behavior.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue
 */
hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                              int  numDevices, unsigned int  flags);
// doxygen end Module
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Occupancy Occupancy
 *  @{
 *  This section describes the occupancy functions of HIP runtime API.
 *
 */
/**
 * @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
 *
 * @param [out] gridSize           minimum grid size for maximum potential occupancy
 * @param [out] blockSize          block size for maximum potential occupancy
 * @param [in]  f                  kernel function for which occupancy is calulated
 * @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
 * @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
 *
 * Please note, HIP does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
//TODO - Match CUoccupancyB2DSize
hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit);
/**
 * @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
 *
 * @param [out] gridSize           minimum grid size for maximum potential occupancy
 * @param [out] blockSize          block size for maximum potential occupancy
 * @param [in]  f                  kernel function for which occupancy is calulated
 * @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
 * @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
 * @param [in]  flags            Extra flags for occupancy calculation (only default supported)
 *
 * Please note, HIP does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
//TODO - Match CUoccupancyB2DSize
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit, unsigned int  flags);
/**
 * @brief Returns occupancy for a device function.
 *
 * @param [out] numBlocks        Returned occupancy
 * @param [in]  f                Kernel function (hipFunction) for which occupancy is calulated
 * @param [in]  blockSize        Block size the kernel is intended to be launched with
 * @param [in]  dynSharedMemPerBlk Dynamic shared memory usage (in bytes) intended for each block
 * @returns  #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
   int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk);
/**
 * @brief Returns occupancy for a device function.
 *
 * @param [out] numBlocks        Returned occupancy
 * @param [in]  f                Kernel function(hipFunction_t) for which occupancy is calulated
 * @param [in]  blockSize        Block size the kernel is intended to be launched with
 * @param [in]  dynSharedMemPerBlk Dynamic shared memory usage (in bytes) intended for each block
 * @param [in]  flags            Extra flags for occupancy calculation (only default supported)
 * @returns  #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
   int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags);
/**
 * @brief Returns occupancy for a device function.
 *
 * @param [out] numBlocks        Returned occupancy
 * @param [in]  f                Kernel function for which occupancy is calulated
 * @param [in]  blockSize        Block size the kernel is intended to be launched with
 * @param [in]  dynSharedMemPerBlk Dynamic shared memory usage (in bytes) intended for each block
 * @returns  #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
 */
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
   int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk);
/**
 * @brief Returns occupancy for a device function.
 *
 * @param [out] numBlocks        Returned occupancy
 * @param [in]  f                Kernel function for which occupancy is calulated
 * @param [in]  blockSize        Block size the kernel is intended to be launched with
 * @param [in]  dynSharedMemPerBlk Dynamic shared memory usage (in bytes) intended for each block
 * @param [in]  flags            Extra flags for occupancy calculation (currently ignored)
 * @returns  #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
 */
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
   int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags __dparm(hipOccupancyDefault));
/**
 * @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
 *
 * @param [out] gridSize           minimum grid size for maximum potential occupancy
 * @param [out] blockSize          block size for maximum potential occupancy
 * @param [in]  f                  kernel function for which occupancy is calulated
 * @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
 * @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
 *
 * Please note, HIP does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             const void* f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit);
// doxygen end Occupancy
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Profiler Profiler Control[Deprecated]
 *  @{
 *  This section describes the profiler control functions of HIP runtime API.
 *
 *  @warning The cudaProfilerInitialize API format for "configFile" is not supported.
 *
 */
// TODO - expand descriptions:
/**
 * @brief Start recording of profiling information
 * When using this API, start the profiler with profiling disabled.  (--startdisabled)
 * @returns  #hipErrorNotSupported
 * @warning : hipProfilerStart API is deprecated, use roctracer/rocTX instead.
 */
DEPRECATED("use roctracer/rocTX instead")
hipError_t hipProfilerStart();
/**
 * @brief Stop recording of profiling information.
 * When using this API, start the profiler with profiling disabled.  (--startdisabled)
 * @returns  #hipErrorNotSupported
 * @warning  hipProfilerStart API is deprecated, use roctracer/rocTX instead.
 */
DEPRECATED("use roctracer/rocTX instead")
hipError_t hipProfilerStop();
// doxygen end profiler
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Clang Launch API to support the triple-chevron syntax
 *  @{
 *  This section describes the API to support the triple-chevron syntax.
 */
/**
 * @brief Configure a kernel launch.
 *
 * @param [in] gridDim   grid dimension specified as multiple of blockDim.
 * @param [in] blockDim  block dimensions specified in work-items
 * @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 *
 * Please note, HIP does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue
 *
 */
hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dparm(0), hipStream_t stream __dparm(0));
/**
 * @brief Set a kernel argument.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue
 *
 * @param [in] arg    Pointer the argument in host memory.
 * @param [in] size   Size of the argument.
 * @param [in] offset Offset of the argument on the argument stack.
 *
 */
hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
/**
 * @brief Launch a kernel.
 *
 * @param [in] func Kernel to launch.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue
 *
 */
hipError_t hipLaunchByPtr(const void* func);
/**
 * @brief Push configuration of a kernel launch.
 *
 * @param [in] gridDim   grid dimension specified as multiple of blockDim.
 * @param [in] blockDim  block dimensions specified in work-items
 * @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 *
 * Please note, HIP does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue
 *
 */
hipError_t __hipPushCallConfiguration(dim3 gridDim,
                                      dim3 blockDim,
                                      size_t sharedMem __dparm(0),
                                      hipStream_t stream __dparm(0));
/**
 * @brief Pop configuration of a kernel launch.
 *
 * @param [out] gridDim   grid dimension specified as multiple of blockDim.
 * @param [out] blockDim  block dimensions specified in work-items
 * @param [out] sharedMem Amount of dynamic shared memory to allocate for this kernel.  The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [out] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 *
 * Please note, HIP does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * Please note, HIP does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue
 *
 */
hipError_t __hipPopCallConfiguration(dim3 *gridDim,
                                     dim3 *blockDim,
                                     size_t *sharedMem,
                                     hipStream_t *stream);
/**
 * @brief C compliant kernel launch API
 *
 * @param [in] function_address - kernel stub function pointer.
 * @param [in] numBlocks - number of blocks
 * @param [in] dimBlocks - dimension of a block
 * @param [in] args - kernel arguments
 * @param [in] sharedMemBytes - Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream - Stream where the kernel should be dispatched.  May be 0, in which case th
 *  default stream is used with associated synchronization rules.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipLaunchKernel(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes __dparm(0),
                           hipStream_t stream __dparm(0));

/**
 * @brief Enqueues a host function call in a stream.
 *
 * @param [in] stream - stream to enqueue work to.
 * @param [in] fn - function to call once operations enqueued preceeding are complete.
 * @param [in] userData - User-specified data to be passed to the function.
 * @returns #hipSuccess, #hipErrorInvalidResourceHandle, #hipErrorInvalidValue,
 * #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData);

/**
 * Copies memory for 2D arrays.
 *
 * @param pCopy           - Parameters for the memory copy
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy);
//TODO: Move this to hip_ext.h
/**
 * @brief Launches kernel from the pointer address, with arguments and shared memory on stream.
 *
 * @param [in] function_address pointer to the Kernel to launch.
 * @param [in] numBlocks number of blocks.
 * @param [in] dimBlocks dimension of a block.
 * @param [in] args pointer to kernel arguments.
 * @param [in] sharedMemBytes  Amount of dynamic shared memory to allocate for this kernel.
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream  Stream where the kernel should be dispatched.
 * May be 0, in which case the default stream is used with associated synchronization rules.
 * @param [in] startEvent  If non-null, specified event will be updated to track the start time of
 * the kernel launch. The event must be created before calling this API.
 * @param [in] stopEvent  If non-null, specified event will be updated to track the stop time of
 * the kernel launch. The event must be created before calling this API.
 * @param [in] flags  The value of hipExtAnyOrderLaunch, signifies if kernel can be
 * launched in any order.
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue.
 *
 */
hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                              void** args, size_t sharedMemBytes, hipStream_t stream,
                              hipEvent_t startEvent, hipEvent_t stopEvent, int flags);
// doxygen end Clang launch
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Texture Texture Management
 *  @{
 *  This section describes the texture management functions of HIP runtime API.
 */

/**
 * @brief Creates a texture object.
 *
 * @param [out] pTexObject  pointer to the texture object to create
 * @param [in] pResDesc  pointer to resource descriptor
 * @param [in] pTexDesc  pointer to texture descriptor
 * @param [in] pResViewDesc  pointer to resource view descriptor
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported, #hipErrorOutOfMemory
 *
 * @note 3D liner filter isn't supported on GFX90A boards, on which the API @p hipCreateTextureObject will
 * return hipErrorNotSupported.
 *
 */
hipError_t hipCreateTextureObject(
    hipTextureObject_t* pTexObject,
    const hipResourceDesc* pResDesc,
    const hipTextureDesc* pTexDesc,
    const struct hipResourceViewDesc* pResViewDesc);

/**
 * @brief Destroys a texture object.
 *
 * @param [in] textureObject  texture object to destroy
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject);

/**
 * @brief Gets the channel descriptor in an array.
 *
 * @param [in] desc  pointer to channel format descriptor
 * @param [out] array  memory array on the device
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipGetChannelDesc(
    hipChannelFormatDesc* desc,
    hipArray_const_t array);

/**
 * @brief Gets resource descriptor for the texture object.
 *
 * @param [out] pResDesc  pointer to resource descriptor
 * @param [in] textureObject  texture object
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipGetTextureObjectResourceDesc(
    hipResourceDesc* pResDesc,
    hipTextureObject_t textureObject);

/**
 * @brief Gets resource view descriptor for the texture object.
 *
 * @param [out] pResViewDesc  pointer to resource view descriptor
 * @param [in] textureObject  texture object
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipGetTextureObjectResourceViewDesc(
    struct hipResourceViewDesc* pResViewDesc,
    hipTextureObject_t textureObject);

/**
 * @brief Gets texture descriptor for the texture object.
 *
 * @param [out] pTexDesc  pointer to texture descriptor
 * @param [in] textureObject  texture object
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipGetTextureObjectTextureDesc(
    hipTextureDesc* pTexDesc,
    hipTextureObject_t textureObject);

/**
 * @brief Creates a texture object.
 *
 * @param [out] pTexObject  pointer to texture object to create
 * @param [in] pResDesc  pointer to resource descriptor
 * @param [in] pTexDesc  pointer to texture descriptor
 * @param [in] pResViewDesc  pointer to resource view descriptor
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipTexObjectCreate(
    hipTextureObject_t* pTexObject,
    const HIP_RESOURCE_DESC* pResDesc,
    const HIP_TEXTURE_DESC* pTexDesc,
    const HIP_RESOURCE_VIEW_DESC* pResViewDesc);

/**
 * @brief Destroys a texture object.
 *
 * @param [in] texObject  texture object to destroy
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipTexObjectDestroy(
    hipTextureObject_t texObject);

/**
 * @brief Gets resource descriptor of a texture object.
 *
 * @param [out] pResDesc  pointer to resource descriptor
 * @param [in] texObject  texture object
 *
 * @returns #hipSuccess, #hipErrorNotSupported, #hipErrorInvalidValue
 *
 */
hipError_t hipTexObjectGetResourceDesc(
    HIP_RESOURCE_DESC* pResDesc,
    hipTextureObject_t texObject);

/**
 * @brief Gets resource view descriptor of a texture object.
 *
 * @param [out] pResViewDesc  pointer to resource view descriptor
 * @param [in] texObject  texture object
 *
 * @returns #hipSuccess, #hipErrorNotSupported, #hipErrorInvalidValue
 *
 */
hipError_t hipTexObjectGetResourceViewDesc(
    HIP_RESOURCE_VIEW_DESC* pResViewDesc,
    hipTextureObject_t texObject);

/**
 * @brief Gets texture descriptor of a texture object.
 *
 * @param [out] pTexDesc  pointer to texture descriptor
 * @param [in] texObject  texture object
 *
 * @returns #hipSuccess, #hipErrorNotSupported, #hipErrorInvalidValue
 *
 */
hipError_t hipTexObjectGetTextureDesc(
    HIP_TEXTURE_DESC* pTexDesc,
    hipTextureObject_t texObject);

/**
 * @brief Allocate a mipmapped array on the device.
 *
 * @param[out] mipmappedArray  - Pointer to allocated mipmapped array in device memory
 * @param[in]  desc            - Requested channel format
 * @param[in]  extent          - Requested allocation size (width field in elements)
 * @param[in]  numLevels       - Number of mipmap levels to allocate
 * @param[in]  flags           - Flags for extensions
 *
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
 *
 * @note  This API is implemented on Windows, under development on Linux.
 *
 */
hipError_t hipMallocMipmappedArray(
    hipMipmappedArray_t *mipmappedArray,
    const struct hipChannelFormatDesc* desc,
    struct hipExtent extent,
    unsigned int numLevels,
    unsigned int flags __dparm(0));

/**
 * @brief Frees a mipmapped array on the device.
 *
 * @param[in] mipmappedArray - Pointer to mipmapped array to free
 *
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * @note  This API is implemented on Windows, under development on Linux.
 *
 */
hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray);

/**
 * @brief Gets a mipmap level of a HIP mipmapped array.
 *
 * @param[out] levelArray     - Returned mipmap level HIP array
 * @param[in]  mipmappedArray - HIP mipmapped array
 * @param[in]  level          - Mipmap level
 *
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * @note  This API is implemented on Windows, under development on Linux.
 *
 */
hipError_t hipGetMipmappedArrayLevel(
    hipArray_t *levelArray,
    hipMipmappedArray_const_t mipmappedArray,
    unsigned int level);

/**
 * @brief Create a mipmapped array.
 *
 * @param [out] pHandle  pointer to mipmapped array
 * @param [in] pMipmappedArrayDesc  mipmapped array descriptor
 * @param [in] numMipmapLevels  mipmap level
 *
 * @returns #hipSuccess, #hipErrorNotSupported, #hipErrorInvalidValue
 *
 * @note  This API is implemented on Windows, under development on Linux.
 */
hipError_t hipMipmappedArrayCreate(
    hipMipmappedArray_t* pHandle,
    HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
    unsigned int numMipmapLevels);

/**
 * @brief Destroy a mipmapped array.
 *
 * @param [out] hMipmappedArray  pointer to mipmapped array to destroy
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @note  This API is implemented on Windows, under development on Linux.
 *
 */
hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray);

/**
 * @brief Get a mipmapped array on a mipmapped level.
 *
 * @param [in] pLevelArray Pointer of array
 * @param [out] hMipMappedArray Pointer of mipmapped array on the requested mipmap level
 * @param [out] level  Mipmap level
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @note  This API is implemented on Windows, under development on Linux.
 *
 */
hipError_t hipMipmappedArrayGetLevel(
    hipArray_t* pLevelArray,
    hipMipmappedArray_t hMipMappedArray,
    unsigned int level);

/**
 *
 *  @addtogroup TextureD Texture Management [Deprecated]
 *  @{
 *  @ingroup Texture
 *  This section describes the deprecated texture management functions of HIP runtime API.
 */

/**
 * @brief  Binds a mipmapped array to a texture.
 *
 * @param [in] tex  pointer to the texture reference to bind
 * @param [in] mipmappedArray memory mipmapped array on the device
 * @param [in] desc  opointer to the channel format
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipBindTextureToMipmappedArray(
    const textureReference* tex,
    hipMipmappedArray_const_t mipmappedArray,
    const hipChannelFormatDesc* desc);

/**
 * @brief Gets the texture reference related with the symbol.
 *
 * @param [out] texref  texture reference
 * @param [in] symbol  pointer to the symbol related with the texture for the reference
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipGetTextureReference(
    const textureReference** texref,
    const void* symbol);

/**
 * @brief Gets the border color used by a texture reference.
 *
 * @param [out] pBorderColor  Returned Type and Value of RGBA color.
 * @param [in] texRef  Texture reference.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetBorderColor(float* pBorderColor, const textureReference* texRef);

/**
 * @brief Gets the array bound to a texture reference.

 *
 * @param [in] pArray  Returned array.
 * @param [in] texRef  texture reference.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetArray(hipArray_t* pArray, const textureReference* texRef);

/**
 * @brief Sets address mode for a texture reference.
 *
 * @param [in] texRef  texture reference.
 * @param [in] dim  Dimension of the texture.
 * @param [in] am  Value of the texture address mode.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetAddressMode(
    textureReference* texRef,
    int dim,
    enum hipTextureAddressMode am);
/**
 * @brief Binds an array as a texture reference.
 *
 * @param [in] tex  Pointer texture reference.
 * @param [in] array  Array to bind.
 * @param [in] flags  Flags should be set as HIP_TRSA_OVERRIDE_FORMAT, as a valid value.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetArray(
    textureReference* tex,
    hipArray_const_t array,
    unsigned int flags);
/**
 * @brief Set filter mode for a texture reference.
 *
 * @param [in] texRef  Pointer texture reference.
 * @param [in] fm  Value of texture filter mode.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetFilterMode(
    textureReference* texRef,
    enum hipTextureFilterMode fm);
/**
 * @brief Set flags for a texture reference.
 *
 * @param [in] texRef  Pointer texture reference.
 * @param [in] Flags  Value of flags.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetFlags(
    textureReference* texRef,
    unsigned int Flags);
/**
 * @brief Set format for a texture reference.
 *
 * @param [in] texRef  Pointer texture reference.
 * @param [in] fmt  Value of format.
 * @param [in] NumPackedComponents  Number of components per array.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetFormat(
    textureReference* texRef,
    hipArray_Format fmt,
    int NumPackedComponents);
/**
 * @brief Binds a memory area to a texture.
 *
 * @param [in] offset  Offset in bytes.
 * @param [in] tex  Texture to bind.
 * @param [in] devPtr  Pointer of memory on the device.
 * @param [in] desc  Pointer of channel format descriptor.
 * @param [in] size  Size of memory in bites.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipBindTexture(
    size_t* offset,
    const textureReference* tex,
    const void* devPtr,
    const hipChannelFormatDesc* desc,
    size_t size __dparm(UINT_MAX));
/**
 * @brief Binds a 2D memory area to a texture.
 *
 * @param [in] offset  Offset in bytes.
 * @param [in] tex  Texture to bind.
 * @param [in] devPtr  Pointer of 2D memory area on the device.
 * @param [in] desc  Pointer of channel format descriptor.
 * @param [in] width  Width in texel units.
 * @param [in] height  Height in texel units.
 * @param [in] pitch  Pitch in bytes.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipBindTexture2D(
    size_t* offset,
    const textureReference* tex,
    const void* devPtr,
    const hipChannelFormatDesc* desc,
    size_t width,
    size_t height,
    size_t pitch);
/**
 * @brief Binds a memory area to a texture.
 *
 * @param [in] tex  Pointer of texture reference.
 * @param [in] array  Array to bind.
 * @param [in] desc  Pointer of channel format descriptor.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipBindTextureToArray(
    const textureReference* tex,
    hipArray_const_t array,
    const hipChannelFormatDesc* desc);
/**
 * @brief Get the offset of the alignment in a texture.
 *
 * @param [in] offset  Offset in bytes.
 * @param [in] texref  Pointer of texture reference.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipGetTextureAlignmentOffset(
    size_t* offset,
    const textureReference* texref);
/**
 * @brief Unbinds a texture.
 *
 * @param [in] tex  Texture to unbind.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipUnbindTexture(const textureReference* tex);
/**
 * @brief Gets the the address for a texture reference.
 *
 * @param [out] dev_ptr  Pointer of device address.
 * @param [in] texRef  Pointer of texture reference.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetAddress(
    hipDeviceptr_t* dev_ptr,
    const textureReference* texRef);
/**
 * @brief Gets the address mode for a texture reference.
 *
 * @param [out] pam  Pointer of address mode.
 * @param [in] texRef  Pointer of texture reference.
 * @param [in] dim  Dimension.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetAddressMode(
    enum hipTextureAddressMode* pam,
    const textureReference* texRef,
    int dim);
/**
 * @brief Gets filter mode for a texture reference.
 *
 * @param [out] pfm  Pointer of filter mode.
 * @param [in] texRef  Pointer of texture reference.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetFilterMode(
    enum hipTextureFilterMode* pfm,
    const textureReference* texRef);
/**
 * @brief Gets flags for a texture reference.
 *
 * @param [out] pFlags  Pointer of flags.
 * @param [in] texRef  Pointer of texture reference.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetFlags(
    unsigned int* pFlags,
    const textureReference* texRef);
/**
 * @brief Gets texture format for a texture reference.
 *
 * @param [out] pFormat  Pointer of the format.
 * @param [out] pNumChannels  Pointer of number of channels.
 * @param [in] texRef  Pointer of texture reference.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetFormat(
    hipArray_Format* pFormat,
    int* pNumChannels,
    const textureReference* texRef);
/**
 * @brief Gets the maximum anisotropy for a texture reference.
 *
 * @param [out] pmaxAnsio  Pointer of the maximum anisotropy.
 * @param [in] texRef  Pointer of texture reference.
 *
 * @returns #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetMaxAnisotropy(
    int* pmaxAnsio,
    const textureReference* texRef);
/**
 * @brief Gets the mipmap filter mode for a texture reference.
 *
 * @param [out] pfm  Pointer of the mipmap filter mode.
 * @param [in] texRef  Pointer of texture reference.
 *
 * @returns #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetMipmapFilterMode(
    enum hipTextureFilterMode* pfm,
    const textureReference* texRef);
/**
 * @brief Gets the mipmap level bias for a texture reference.
 *
 * @param [out] pbias  Pointer of the mipmap level bias.
 * @param [in] texRef  Pointer of texture reference.
 *
 * @returns #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetMipmapLevelBias(
    float* pbias,
    const textureReference* texRef);
/**
 * @brief Gets the minimum and maximum mipmap level clamps for a texture reference.
 *
 * @param [out] pminMipmapLevelClamp  Pointer of the minimum mipmap level clamp.
 * @param [out] pmaxMipmapLevelClamp  Pointer of the maximum mipmap level clamp.
 * @param [in] texRef  Pointer of texture reference.
 *
 * @returns #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetMipmapLevelClamp(
    float* pminMipmapLevelClamp,
    float* pmaxMipmapLevelClamp,
    const textureReference* texRef);
/**
 * @brief Gets the mipmapped array bound to a texture reference.
 *
 * @param [out] pArray  Pointer of the mipmapped array.
 * @param [in] texRef  Pointer of texture reference.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetMipMappedArray(
    hipMipmappedArray_t* pArray,
    const textureReference* texRef);
/**
 * @brief Sets an bound address for a texture reference.
 *
 * @param [out] ByteOffset  Pointer of the offset in bytes.
 * @param [in] texRef  Pointer of texture reference.
 * @param [in] dptr  Pointer of device address to bind.
 * @param [in] bytes  Size in bytes.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetAddress(
    size_t* ByteOffset,
    textureReference* texRef,
    hipDeviceptr_t dptr,
    size_t bytes);
/**
 * @brief Set a bind an address as a 2D texture reference.
 *
 * @param [in] texRef  Pointer of texture reference.
 * @param [in] desc  Pointer of array descriptor.
 * @param [in] dptr  Pointer of device address to bind.
 * @param [in] Pitch  Pitch in bytes.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetAddress2D(
    textureReference* texRef,
    const HIP_ARRAY_DESCRIPTOR* desc,
    hipDeviceptr_t dptr,
    size_t Pitch);
/**
 * @brief Sets the maximum anisotropy for a texture reference.
 *
 * @param [in] texRef  Pointer of texture reference.
 * @param [out] maxAniso  Value of the maximum anisotropy.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetMaxAnisotropy(
    textureReference* texRef,
    unsigned int maxAniso);
/**
 * @brief Sets border color for a texture reference.
 *
 * @param [in] texRef  Pointer of texture reference.
 * @param [in] pBorderColor  Pointer of border color.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetBorderColor(
    textureReference* texRef,
    float* pBorderColor);
/**
 * @brief Sets mipmap filter mode for a texture reference.
 *
 * @param [in] texRef  Pointer of texture reference.
 * @param [in] fm  Value of filter mode.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetMipmapFilterMode(
    textureReference* texRef,
    enum hipTextureFilterMode fm);
/**
 * @brief Sets mipmap level bias for a texture reference.
 *
 * @param [in] texRef  Pointer of texture reference.
 * @param [in] bias  Value of mipmap bias.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetMipmapLevelBias(
    textureReference* texRef,
    float bias);
/**
 * @brief Sets mipmap level clamp for a texture reference.
 *
 * @param [in] texRef  Pointer of texture reference.
 * @param [in] minMipMapLevelClamp  Value of minimum mipmap level clamp.
 * @param [in] maxMipMapLevelClamp  Value of maximum mipmap level clamp.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetMipmapLevelClamp(
    textureReference* texRef,
    float minMipMapLevelClamp,
    float maxMipMapLevelClamp);
/**
 * @brief Binds mipmapped array to a texture reference.
 *
 * @param [in] texRef  Pointer of texture reference to bind.
 * @param [in] mipmappedArray  Pointer of mipmapped array to bind.
 * @param [in] Flags  Flags should be set as HIP_TRSA_OVERRIDE_FORMAT, as a valid value.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning This API is deprecated.
 *
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetMipmappedArray(
    textureReference* texRef,
    struct hipMipmappedArray* mipmappedArray,
    unsigned int Flags);

// doxygen end deprecated texture management
/**
 * @}
 */

// doxygen end Texture management
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Runtime Runtime Compilation
 *  @{
 *  This section describes the runtime compilation functions of HIP runtime API.
 *
 */
// This group is for HIPrtc

// doxygen end Runtime
/**
 * @}
 */

/**
 *
 *  @defgroup Callback Callback Activity APIs
 *  @{
 *  This section describes the callback/Activity of HIP runtime API.
 */
/**
 * @brief Returns HIP API name by ID.
 *
 * @param [in] id ID of HIP API
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
const char* hipApiName(uint32_t id);
/**
 * @brief Returns kernel name reference by function name.
 *
 * @param [in] f Name of function
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
const char* hipKernelNameRef(const hipFunction_t f);
/**
 * @brief Retrives kernel for a given host pointer, unless stated otherwise.
 *
 * @param [in] hostFunction Pointer of host function.
 * @param [in] stream Stream the kernel is executed on.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
const char* hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream);
/**
 * @brief Returns device ID on the stream.
 *
 * @param [in] stream Stream of device executed on.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
int hipGetStreamDeviceId(hipStream_t stream);

// doxygen end Callback
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Graph Graph Management
 *  @{
 *  This section describes the graph management types & functions of HIP runtime API.
 */

/**
 * @brief Begins graph capture on a stream.
 *
 * @param [in] stream - Stream to initiate capture.
 * @param [in] mode - Controls the interaction of this capture sequence with other API calls that
 * are not safe.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode);

/**
 * @brief Ends capture on a stream, returning the captured graph.
 *
 * @param [in] stream - Stream to end capture.
 * @param [out] pGraph - returns the graph captured.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph);

/**
 * @brief Get capture status of a stream.
 *
 * @param [in] stream - Stream under capture.
 * @param [out] pCaptureStatus - returns current status of the capture.
 * @param [out] pId - unique ID of the capture.
 *
 * @returns #hipSuccess, #hipErrorStreamCaptureImplicit
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId);

/**
 * @brief Get stream's capture state
 *
 * @param [in] stream - Stream under capture.
 * @param [out] captureStatus_out - returns current status of the capture.
 * @param [out] id_out - unique ID of the capture.
 * @param [in] graph_out - returns the graph being captured into.
 * @param [out] dependencies_out - returns pointer to an array of nodes.
 * @param [out] numDependencies_out - returns size of the array returned in dependencies_out.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out __dparm(0),
                                      hipGraph_t* graph_out __dparm(0),
                                      const hipGraphNode_t** dependencies_out __dparm(0),
                                      size_t* numDependencies_out __dparm(0));

/**
 * @brief Get stream's capture state
 *
 * @param [in] stream - Stream under capture.
 * @param [out] pCaptureStatus - returns current status of the capture.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);

/**
 * @brief Update the set of dependencies in a capturing stream
 *
 * @param [in] stream  Stream under capture.
 * @param [in] dependencies  pointer to an array of nodes to Add/Replace.
 * @param [in] numDependencies  size of the array in dependencies.
 * @param [in] flags  Flag how to update dependency set. Should be one of value in enum
 * #hipStreamUpdateCaptureDependenciesFlags
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorIllegalState
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                              size_t numDependencies,
                                              unsigned int flags __dparm(0));

/**
 * @brief Swaps the stream capture mode of a thread.
 *
 * @param [in] mode - Pointer to mode value to swap with the current mode
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode);

/**
 * @brief Creates a graph
 *
 * @param [out] pGraph - pointer to graph to create.
 * @param [in] flags - flags for graph creation, must be 0.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags);

/**
 * @brief Destroys a graph
 *
 * @param [in] graph - instance of graph to destroy.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphDestroy(hipGraph_t graph);

/**
 * @brief Adds dependency edges to a graph.
 *
 * @param [in] graph - instance of the graph to add dependencies.
 * @param [in] from - pointer to the graph nodes with dependenties to add from.
 * @param [in] to - pointer to the graph nodes to add dependenties to.
 * @param [in] numDependencies - the number of dependencies to add.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                   const hipGraphNode_t* to, size_t numDependencies);

/**
 * @brief Removes dependency edges from a graph.
 *
 * @param [in] graph - instance of the graph to remove dependencies.
 * @param [in] from - Array of nodes that provide the dependencies.
 * @param [in] to - Array of dependent nodes.
 * @param [in] numDependencies - the number of dependencies to remove.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                      const hipGraphNode_t* to, size_t numDependencies);

/**
 * @brief Returns a graph's dependency edges.
 *
 * @param [in] graph - instance of the graph to get the edges from.
 * @param [out] from - pointer to the graph nodes to return edge endpoints.
 * @param [out] to - pointer to the graph nodes to return edge endpoints.
 * @param [out] numEdges - returns number of edges.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * from and to may both be NULL, in which case this function only returns the number of edges in
 * numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher than the actual
 * number of edges, the remaining entries in from and to will be set to NULL, and the number of
 * edges actually returned will be written to numEdges
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                            size_t* numEdges);

/**
 * @brief Returns graph nodes.
 *
 * @param [in] graph - instance of graph to get the nodes.
 * @param [out] nodes - pointer to return the  graph nodes.
 * @param [out] numNodes - returns number of graph nodes.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * nodes may be NULL, in which case this function will return the number of nodes in numNodes.
 * Otherwise, numNodes entries will be filled in. If numNodes is higher than the actual number of
 * nodes, the remaining entries in nodes will be set to NULL, and the number of nodes actually
 * obtained will be returned in numNodes.
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes);

/**
 * @brief Returns graph's root nodes.
 *
 * @param [in] graph - instance of the graph to get the nodes.
 * @param [out] pRootNodes - pointer to return the graph's root nodes.
 * @param [out] pNumRootNodes - returns the number of graph's root nodes.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * pRootNodes may be NULL, in which case this function will return the number of root nodes in
 * pNumRootNodes. Otherwise, pNumRootNodes entries will be filled in. If pNumRootNodes is higher
 * than the actual number of root nodes, the remaining entries in pRootNodes will be set to NULL,
 * and the number of nodes actually obtained will be returned in pNumRootNodes.
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                size_t* pNumRootNodes);

/**
 * @brief Returns a node's dependencies.
 *
 * @param [in] node - graph node to get the dependencies from.
 * @param [out] pDependencies - pointer to to return the dependencies.
 * @param [out] pNumDependencies -  returns the number of graph node dependencies.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * pDependencies may be NULL, in which case this function will return the number of dependencies in
 * pNumDependencies. Otherwise, pNumDependencies entries will be filled in. If pNumDependencies is
 * higher than the actual number of dependencies, the remaining entries in pDependencies will be set
 * to NULL, and the number of nodes actually obtained will be returned in pNumDependencies.
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                       size_t* pNumDependencies);

/**
 * @brief Returns a node's dependent nodes.
 *
 * @param [in] node - graph node to get the Dependent nodes from.
 * @param [out] pDependentNodes - pointer to return the graph dependent nodes.
 * @param [out] pNumDependentNodes - returns the number of graph node dependent nodes.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * DependentNodes may be NULL, in which case this function will return the number of dependent nodes
 * in pNumDependentNodes. Otherwise, pNumDependentNodes entries will be filled in. If
 * pNumDependentNodes is higher than the actual number of dependent nodes, the remaining entries in
 * pDependentNodes will be set to NULL, and the number of nodes actually obtained will be returned
 * in pNumDependentNodes.
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                         size_t* pNumDependentNodes);

/**
 * @brief Returns a node's type.
 *
 * @param [in] node - instance of the graph to add dependencies.
 * @param [out] pType - pointer to the return the type
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType);

/**
 * @brief Remove a node from the graph.
 *
 * @param [in] node - graph node to remove
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphDestroyNode(hipGraphNode_t node);

/**
 * @brief Clones a graph.
 *
 * @param [out] pGraphClone - Returns newly created cloned graph.
 * @param [in] originalGraph - original graph to clone from.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph);

/**
 * @brief Finds a cloned version of a node.
 *
 * @param [out] pNode - Returns the cloned node.
 * @param [in] originalNode - original node handle.
 * @param [in] clonedGraph - Cloned graph to query.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph);

/**
 * @brief Creates an executable graph from a graph
 *
 * @param [out] pGraphExec - pointer to instantiated executable graph that is created.
 * @param [in] graph - instance of graph to instantiate.
 * @param [out] pErrorNode - pointer to error node in case error occured in graph instantiation,
 *  it could modify the correponding node.
 * @param [out] pLogBuffer - pointer to log buffer.
 * @param [out] bufferSize - the size of log buffer.
 *
 * @returns #hipSuccess, #hipErrorOutOfMemory
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);

/**
 * @brief Creates an executable graph from a graph.
 *
 * @param [out] pGraphExec - pointer to instantiated executable graph that is created.
 * @param [in] graph - instance of graph to instantiate.
 * @param [in] flags - Flags to control instantiation.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.It does not support
 * any of flag and is behaving as hipGraphInstantiate.
 */
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                        unsigned long long flags);

/**
 * @brief launches an executable graph in a stream
 *
 * @param [in] graphExec - instance of executable graph to launch.
 * @param [in] stream - instance of stream in which to launch executable graph.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);

/**
 * @brief uploads an executable graph in a stream
 *
 * @param [in] graphExec - instance of executable graph to launch.
 * @param [in] stream - instance of stream in which to launch executable graph.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream);

/**
 * @brief Destroys an executable graph
 *
 * @param [in] graphExec - instance of executable graph to destry.
 *
 * @returns #hipSuccess.
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec);

// Check whether an executable graph can be updated with a graph and perform the update if possible.
/**
 * @brief Check whether an executable graph can be updated with a graph and perform the update if  *
 * possible.
 *
 * @param [in] hGraphExec - instance of executable graph to update.
 * @param [in] hGraph - graph that contains the updated parameters.
 * @param [in] hErrorNode_out -  node which caused the permissibility check to forbid the update.
 * @param [in] updateResult_out - Whether the graph update was permitted.
 * @returns #hipSuccess, #hipErrorGraphExecUpdateFailure
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t* hErrorNode_out,
                              hipGraphExecUpdateResult* updateResult_out);

/**
 * @brief Creates a kernel execution node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to graph node to create.
 * @param [in] graph - instance of graph to add the created node.
 * @param [in] pDependencies - pointer to the dependencies on the kernel execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] pNodeParams - pointer to the parameters to the kernel execution node on the GPU.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDeviceFunction
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams);

/**
 * @brief Gets kernel node's parameters.
 *
 * @param [in] node - instance of the node to get parameters from.
 * @param [out] pNodeParams - pointer to the parameters
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams);

/**
 * @brief Sets a kernel node's parameters.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - const pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams* pNodeParams);

/**
 * @brief Sets the parameters for a kernel node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - const pointer to the kernel node parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipKernelNodeParams* pNodeParams);

/**
 * @brief Creates a memcpy node and adds it to a graph.
 *
 * @param [out] phGraphNode - pointer to graph node to create.
 * @param [in] hGraph - instance of graph to add the created node.
 * @param [in] dependencies - const pointer to the dependencies on the memcpy execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] copyParams - const pointer to the parameters for the memory copy.
 * @param [in] ctx - cotext related to current device.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                    const hipGraphNode_t* dependencies,
                                    size_t numDependencies,
                                    const HIP_MEMCPY3D* copyParams, hipCtx_t ctx);
/**
 * @brief Creates a memcpy node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to graph node to create.
 * @param [in] graph - instance of graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] pCopyParams - const pointer to the parameters for the memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams);
/**
 * @brief Gets a memcpy node's parameters.
 *
 * @param [in] node - instance of the node to get parameters from.
 * @param [out] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams);

/**
 * @brief Sets a memcpy node's parameters.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - const pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams);

/**
 * @brief Sets a node attribute.
 *
 * @param [in] hNode - instance of the node to set parameters to.
 * @param [in] attr - the attribute node is set to.
 * @param [in] value - const pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          const hipKernelNodeAttrValue* value);
/**
 * @brief Gets a node attribute.
 *
 * @param [in] hNode - instance of the node to set parameters to.
 * @param [in] attr - the attribute node is set to.
 * @param [in] value - const pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          hipKernelNodeAttrValue* value);
/**
 * @brief Sets the parameters for a memcpy node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - const pointer to the kernel node parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           hipMemcpy3DParms* pNodeParams);

/**
 * @brief Creates a 1D memcpy node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to graph node to create.
 * @param [in] graph - instance of graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] src - pointer to memory address to the source.
 * @param [in] count - the size of the memory to copy.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind);

/**
 * @brief Sets a memcpy node's parameters to perform a 1-dimensional copy.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] src - pointer to memory address to the source.
 * @param [in] count - the size of the memory to copy.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src,
                                         size_t count, hipMemcpyKind kind);

/**
 * @brief Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional
 * copy.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] src - pointer to memory address to the source.
 * @param [in] count - the size of the memory to copy.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                             void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind);

/**
 * @brief Creates a memcpy node to copy from a symbol on the device and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to graph node to create.
 * @param [in] graph - instance of graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] symbol - Device symbol address.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies, void* dst, const void* symbol,
                                           size_t count, size_t offset, hipMemcpyKind kind);

/**
 * @brief Sets a memcpy node's parameters to copy from a symbol on the device.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] symbol - Device symbol address.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol,
                                                 size_t count, size_t offset, hipMemcpyKind kind);

/**
 * @brief Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the
 * * device.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] symbol - Device symbol address.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind);

/**
 * @brief Creates a memcpy node to copy to a symbol on the device and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to graph node to create.
 * @param [in] graph - instance of graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] symbol - Device symbol address.
 * @param [in] src - pointer to memory address of the src.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, const void* symbol,
                                         const void* src, size_t count, size_t offset,
                                         hipMemcpyKind kind);

/**
 * @brief Sets a memcpy node's parameters to copy to a symbol on the device.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] symbol - Device symbol address.
 * @param [in] src - pointer to memory address of the src.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol,
                                               const void* src, size_t count, size_t offset,
                                               hipMemcpyKind kind);


/**
 * @brief Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the
 * device.
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] symbol - Device symbol address.
 * @param [in] src - pointer to memory address of the src.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                   const void* symbol, const void* src,
                                                   size_t count, size_t offset, hipMemcpyKind kind);

/**
 * @brief Creates a memset node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create.
 * @param [in] graph - instance of the graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] pMemsetParams - const pointer to the parameters for the memory set.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams);

/**
 * @brief Gets a memset node's parameters.
 *
 * @param [in] node - instane of the node to get parameters from.
 * @param [out] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams);

/**
 * @brief Sets a memset node's parameters.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams);

/**
 * @brief Sets the parameters for a memset node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipMemsetParams* pNodeParams);

/**
 * @brief Creates a host execution node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create.
 * @param [in] graph - instance of the graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] pNodeParams -pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipHostNodeParams* pNodeParams);

/**
 * @brief Returns a host node's parameters.
 *
 * @param [in] node - instane of the node to get parameters from.
 * @param [out] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams);

/**
 * @brief Sets a host node's parameters.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams);

/**
 * @brief Sets the parameters for a host node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                         const hipHostNodeParams* pNodeParams);

/**
 * @brief Creates a child graph node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create.
 * @param [in] graph - instance of the graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] childGraph - the graph to clone into this node
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipGraph_t childGraph);

/**
 * @brief Gets a handle to the embedded graph of a child graph node.
 *
 * @param [in] node - instane of the node to get child graph.
 * @param [out] pGraph - pointer to get the graph.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph);

/**
 * @brief Updates node parameters in the child graph node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - node from the graph which was used to instantiate graphExec.
 * @param [in] childGraph - child graph with updated parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                               hipGraph_t childGraph);

/**
 * @brief Creates an empty node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
 * @param [in] graph - instane of the graph the node is add to.
 * @param [in] pDependencies - const pointer to the node dependenties.
 * @param [in] numDependencies - the number of dependencies.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t* pDependencies, size_t numDependencies);


/**
 * @brief Creates an event record node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
 * @param [in] graph - instane of the graph the node to be added.
 * @param [in] pDependencies - const pointer to the node dependenties.
 * @param [in] numDependencies - the number of dependencies.
 * @param [in] event - Event for the node.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies, size_t numDependencies,
                                      hipEvent_t event);

/**
 * @brief Returns the event associated with an event record node.
 *
 * @param [in] node -  instane of the node to get event from.
 * @param [out] event_out - Pointer to return the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);

/**
 * @brief Sets an event record node's event.
 *
 * @param [in] node - instane of the node to set event to.
 * @param [in] event - pointer to the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event);

/**
 * @brief Sets the event for an event record node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] hNode - node from the graph which was used to instantiate graphExec.
 * @param [in] event - pointer to the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               hipEvent_t event);

/**
 * @brief Creates an event wait node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
 * @param [in] graph - instane of the graph the node to be added.
 * @param [in] pDependencies - const pointer to the node dependenties.
 * @param [in] numDependencies - the number of dependencies.
 * @param [in] event - Event for the node.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    hipEvent_t event);


/**
 * @brief Returns the event associated with an event wait node.
 *
 * @param [in] node -  instane of the node to get event from.
 * @param [out] event_out - Pointer to return the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);

/**
 * @brief Sets an event wait node's event.
 *
 * @param [in] node - instane of the node to set event to.
 * @param [in] event - pointer to the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event);

/**
 * @brief Sets the event for an event record node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] hNode - node from the graph which was used to instantiate graphExec.
 * @param [in] event - pointer to the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                             hipEvent_t event);

/**
 * @brief Creates a memory allocation node and adds it to a graph
 *
 * @param [out] pGraphNode      - Pointer to the graph node to create and add to the graph
 * @param [in] graph            - Instane of the graph the node to be added
 * @param [in] pDependencies    - Const pointer to the node dependenties
 * @param [in] numDependencies  - The number of dependencies
 * @param [in] pNodeParams      - Node parameters for memory allocation
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
    const hipGraphNode_t* pDependencies, size_t numDependencies, hipMemAllocNodeParams* pNodeParams);

/**
 * @brief Returns parameters for memory allocation node
 *
 * @param [in] node         - Memory allocation node for a query
 * @param [out] pNodeParams - Parameters for the specified memory allocation node
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams* pNodeParams);

/**
 * @brief Creates a memory free node and adds it to a graph
 *
 * @param [out] pGraphNode      - Pointer to the graph node to create and add to the graph
 * @param [in] graph            - Instane of the graph the node to be added
 * @param [in] pDependencies    - Const pointer to the node dependenties
 * @param [in] numDependencies  - The number of dependencies
 * @param [in] dev_ptr          - Pointer to the memory to be freed
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
    const hipGraphNode_t* pDependencies, size_t numDependencies, void* dev_ptr);

/**
 * @brief Returns parameters for memory free node
 *
 * @param [in] node     - Memory free node for a query
 * @param [out] dev_ptr - Device pointer for the specified memory free node
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr);

/**
 * @brief Get the mem attribute for graphs.
 *
 * @param [in] device - device the attr is get for.
 * @param [in] attr - attr to get.
 * @param [out] value - value for specific attr.
 * @returns #hipSuccess, #hipErrorInvalidDevice
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value);

/**
 * @brief Set the mem attribute for graphs.
 *
 * @param [in] device - device the attr is set for.
 * @param [in] attr - attr to set.
 * @param [in] value - value for specific attr.
 * @returns #hipSuccess, #hipErrorInvalidDevice
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value);

/**
 * @brief Free unused memory on specific device used for graph back to OS.
 *
 * @param [in] device - device the memory is used for graphs
 * @returns #hipSuccess, #hipErrorInvalidDevice
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipDeviceGraphMemTrim(int device);

/**
 * @brief Create an instance of userObject to manage lifetime of a resource.
 *
 * @param [out] object_out - pointer to instace of userobj.
 * @param [in] ptr - pointer to pass to destroy function.
 * @param [in] destroy - destroy callback to remove resource.
 * @param [in] initialRefcount - reference to resource.
 * @param [in] flags - flags passed to API.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy,
                               unsigned int initialRefcount, unsigned int flags);

/**
 * @brief Release number of references to resource.
 *
 * @param [in] object - pointer to instace of userobj.
 * @param [in] count - reference to resource to be retained.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count __dparm(1));

/**
 * @brief Retain number of references to resource.
 *
 * @param [in] object - pointer to instace of userobj.
 * @param [in] count - reference to resource to be retained.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count __dparm(1));

/**
 * @brief Retain user object for graphs.
 *
 * @param [in] graph - pointer to graph to retain the user object for.
 * @param [in] object - pointer to instace of userobj.
 * @param [in] count - reference to resource to be retained.
 * @param [in] flags - flags passed to API.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object,
                                    unsigned int count __dparm(1), unsigned int flags __dparm(0));

/**
 * @brief Release user object from graphs.
 *
 * @param [in] graph - pointer to graph to retain the user object for.
 * @param [in] object - pointer to instace of userobj.
 * @param [in] count - reference to resource to be retained.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object,
                                     unsigned int count __dparm(1));

/**
 * @brief Write a DOT file describing graph structure.
 *
 * @param [in] graph - graph object for which DOT file has to be generated.
 * @param [in] path - path to write the DOT file.
 * @param [in] flags - Flags from hipGraphDebugDotFlags to get additional node information.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorOperatingSystem
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags);

/**
 * @brief Copies attributes from source node to destination node.
 *
 * Copies attributes from source node to destination node.
 * Both node must have the same context.
 *
 * @param [out] hDst - Destination node.
 * @param [in] hSrc - Source node.
 * For list of attributes see ::hipKernelNodeAttrID.
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst);

/**
 * @brief Enables or disables the specified node in the given graphExec
 *
 * Sets hNode to be either enabled or disabled. Disabled nodes are functionally equivalent
 * to empty nodes until they are reenabled. Existing node parameters are not affected by
 * disabling/enabling the node.
 *
 * The node is identified by the corresponding hNode in the non-executable graph, from which the
 * executable graph was instantiated.
 *
 * hNode must not have been removed from the original graph.
 *
 * @note Currently only kernel, memset and memcpy nodes are supported.
 *
 * @param [in] hGraphExec - The executable graph in which to set the specified node.
 * @param [in] hNode      - Node from the graph from which graphExec was instantiated.
 * @param [in] isEnabled  - Node is enabled if != 0, otherwise the node is disabled.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue,
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int isEnabled);
/**
 * @brief Query whether a node in the given graphExec is enabled
 *
 * Sets isEnabled to 1 if hNode is enabled, or 0 if it is disabled.
 *
 * The node is identified by the corresponding node in the non-executable graph, from which the
 * executable graph was instantiated.
 *
 * hNode must not have been removed from the original graph.
 *
 * @note Currently only kernel, memset and memcpy nodes are supported.
 *
 * @param [in]  hGraphExec - The executable graph in which to set the specified node.
 * @param [in]  hNode      - Node from the graph from which graphExec was instantiated.
 * @param [out] isEnabled  - Location to return the enabled status of the node.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int* isEnabled);

/**
 * @brief Creates a external semaphor wait node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create.
 * @param [in] graph - instance of the graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] nodeParams -pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddExternalSemaphoresWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipExternalSemaphoreWaitNodeParams* nodeParams);

/**
 * @brief Creates a external semaphor signal node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create.
 * @param [in] graph - instance of the graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] nodeParams -pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddExternalSemaphoresSignalNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipExternalSemaphoreSignalNodeParams* nodeParams);
/**
 * @brief Updates node parameters in the external semaphore signal node.
 *
 * @param [in]  hNode      - Node from the graph from which graphExec was instantiated.
 * @param [in]  nodeParams  - Pointer to the params to be set.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(hipGraphNode_t hNode,
                                                         const hipExternalSemaphoreSignalNodeParams* nodeParams);
/**
 * @brief Updates node parameters in the external semaphore wait node.
 *
 * @param [in]  hNode      - Node from the graph from which graphExec was instantiated.
 * @param [in]  nodeParams  - Pointer to the params to be set.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(hipGraphNode_t hNode,
                                                       const hipExternalSemaphoreWaitNodeParams* nodeParams);
/**
 * @brief Returns external semaphore signal node params.
 *
 * @param [in]   hNode       - Node from the graph from which graphExec was instantiated.
 * @param [out]  params_out  - Pointer to params.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(hipGraphNode_t hNode,
                                                         hipExternalSemaphoreSignalNodeParams* params_out);
/**
 * @brief Returns external semaphore wait node params.
 *
 * @param [in]   hNode       - Node from the graph from which graphExec was instantiated.
 * @param [out]  params_out  - Pointer to params.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(hipGraphNode_t hNode,
                                                       hipExternalSemaphoreWaitNodeParams* params_out);
/**
 * @brief Updates node parameters in the external semaphore signal node in the given graphExec.
 *
 * @param [in]  hGraphExec - The executable graph in which to set the specified node.
 * @param [in]  hNode      - Node from the graph from which graphExec was instantiated.
 * @param [in]  nodeParams  - Pointer to the params to be set.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                                             const hipExternalSemaphoreSignalNodeParams* nodeParams);
/**
 * @brief Updates node parameters in the external semaphore wait node in the given graphExec.
 *
 * @param [in]  hGraphExec - The executable graph in which to set the specified node.
 * @param [in]  hNode      - Node from the graph from which graphExec was instantiated.
 * @param [in]  nodeParams  - Pointer to the params to be set.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                                           const hipExternalSemaphoreWaitNodeParams* nodeParams);

/**
 * @brief Creates a memset node and adds it to a graph.
 *
 * @param [out] phGraphNode - pointer to graph node to create.
 * @param [in] hGraph - instance of graph to add the created node to.
 * @param [in] dependencies - const pointer to the dependencies on the memset execution node.
 * @param [in] numDependencies - number of the dependencies.
 * @param [in] memsetParams - const pointer to the parameters for the memory set.
 * @param [in] ctx - cotext related to current device.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                 const hipGraphNode_t* dependencies, size_t numDependencies,
                                 const HIP_MEMSET_NODE_PARAMS* memsetParams, hipCtx_t ctx);

// doxygen end graph API
/**
 * @}
 */


/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Virtual Virtual Memory Management
 *  @{
 *  This section describes the virtual memory management functions of HIP runtime API.
 *
 *  @note  Please note, the virtual memory management functions of HIP runtime API are implemented
 *  on Linux, under development on Windows.
 */

/**
 * @brief Frees an address range reservation made via hipMemAddressReserve
 *
 * @param [in] devPtr - starting address of the range.
 * @param [in] size - size of the range.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemAddressFree(void* devPtr, size_t size);

/**
 * @brief Reserves an address range
 *
 * @param [out] ptr - starting address of the reserved range.
 * @param [in] size - size of the reservation.
 * @param [in] alignment - alignment of the address.
 * @param [in] addr - requested starting address of the range.
 * @param [in] flags - currently unused, must be zero.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr, unsigned long long flags);

/**
 * @brief Creates a memory allocation described by the properties and size
 *
 * @param [out] handle - value of the returned handle.
 * @param [in] size - size of the allocation.
 * @param [in] prop - properties of the allocation.
 * @param [in] flags - currently unused, must be zero.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size, const hipMemAllocationProp* prop, unsigned long long flags);

/**
 * @brief Exports an allocation to a requested shareable handle type.
 *
 * @param [out] shareableHandle - value of the returned handle.
 * @param [in] handle - handle to share.
 * @param [in] handleType - type of the shareable handle.
 * @param [in] flags - currently unused, must be zero.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemExportToShareableHandle(void* shareableHandle, hipMemGenericAllocationHandle_t handle, hipMemAllocationHandleType handleType, unsigned long long flags);

/**
 * @brief Get the access flags set for the given location and ptr.
 *
 * @param [out] flags - flags for this location.
 * @param [in] location - target location.
 * @param [in] ptr - address to check the access flags.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr);

/**
 * @brief Calculates either the minimal or recommended granularity.
 *
 * @param [out] granularity - returned granularity.
 * @param [in] prop - location properties.
 * @param [in] option - determines which granularity to return.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 *
 */
hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop, hipMemAllocationGranularity_flags option);

/**
 * @brief Retrieve the property structure of the given handle.
 *
 * @param [out] prop - properties of the given handle.
 * @param [in] handle - handle to perform the query on.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux under development on Windows.
 */
hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle);

/**
 * @brief Imports an allocation from a requested shareable handle type.
 *
 * @param [out] handle - returned value.
 * @param [in] osHandle - shareable handle representing the memory allocation.
 * @param [in] shHandleType - handle type.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle, hipMemAllocationHandleType shHandleType);

/**
 * @brief Maps an allocation handle to a reserved virtual address range.
 *
 * @param [in] ptr - address where the memory will be mapped.
 * @param [in] size - size of the mapping.
 * @param [in] offset - offset into the memory, currently must be zero.
 * @param [in] handle - memory allocation to be mapped.
 * @param [in] flags - currently unused, must be zero.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle, unsigned long long flags);

/**
 * @brief Maps or unmaps subregions of sparse HIP arrays and sparse HIP mipmapped arrays.
 *
 * @param [in] mapInfoList - list of hipArrayMapInfo.
 * @param [in] count - number of hipArrayMapInfo in mapInfoList.
 * @param [in] stream - stream identifier for the stream to use for map or unmap operations.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemMapArrayAsync(hipArrayMapInfo* mapInfoList, unsigned int  count, hipStream_t stream);

/**
 * @brief Release a memory handle representing a memory allocation which was previously allocated through hipMemCreate.
 *
 * @param [in] handle - handle of the memory allocation.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);

/**
 * @brief Returns the allocation handle of the backing memory allocation given the address.
 *
 * @param [out] handle - handle representing addr.
 * @param [in] addr - address to look up.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr);

/**
 * @brief Set the access flags for each location specified in desc for the given virtual address range.
 *
 * @param [in] ptr - starting address of the virtual address range.
 * @param [in] size - size of the range.
 * @param [in] desc - array of hipMemAccessDesc.
 * @param [in] count - number of hipMemAccessDesc in desc.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count);

/**
 * @brief Unmap memory allocation of a given address range.
 *
 * @param [in] ptr - starting address of the range to unmap.
 * @param [in] size - size of the virtual address range.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
hipError_t hipMemUnmap(void* ptr, size_t size);

// doxygen end virtual memory management API
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 * @defgroup GL OpenGL Interop
 * @{
 * This section describes the OpenGL and graphics interoperability functions of HIP runtime API.
 */

/**
 * @brief Maps a graphics resource for access.
 *
 * @param [in] count - Number of resources to map.
 * @param [in] resources - Pointer of resources to map.
 * @param [in] stream - Stream for synchronization.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorUnknown, #hipErrorInvalidResourceHandle
 *
 */
hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources,
                                   hipStream_t stream  __dparm(0) );
/**
 * @brief Get an array through which to access a subresource of a mapped graphics resource.
 *
 * @param [out] array - Pointer of array through which a subresource of resource may be accessed.
 * @param [in] resource - Mapped resource to access.
 * @param [in] arrayIndex - Array index for the subresource to access.
 * @param [in] mipLevel - Mipmap level for the subresource to access.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @note  In this API, the value of arrayIndex higher than zero is currently not supported.
 *
 */
hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, hipGraphicsResource_t resource,
                                                unsigned int arrayIndex, unsigned int mipLevel);
/**
 * @brief Gets device accessible address of a graphics resource.
 *
 * @param [out] devPtr - Pointer of device through which graphic resource may be accessed.
 * @param [out] size - Size of the buffer accessible from devPtr.
 * @param [in] resource - Mapped resource to access.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                               hipGraphicsResource_t resource);
/**
 * @brief Unmaps graphics resources.
 *
 * @param [in] count - Number of resources to unmap.
 * @param [in] resources - Pointer of resources to unmap.
 * @param [in] stream - Stream for synchronization.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorUnknown, #hipErrorContextIsDestroyed
 *
 */
hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources,
                                     hipStream_t stream __dparm(0));
/**
 * @brief Unregisters a graphics resource.
 *
 * @param [in] resource - Graphics resources to unregister.
 *
 * @returns #hipSuccess
 *
 */
hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource);
// doxygen end GL Interop
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 * @defgroup Surface Surface Object
 * @{
 *
 *  This section describes surface object functions of HIP runtime API.
 *
 *  @note  APIs in this section are under development.
 *
 */

/**
 * @brief Create a surface object.
 *
 * @param [out] pSurfObject  Pointer of surface object to be created.
 * @param [in] pResDesc  Pointer of suface object descriptor.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject, const hipResourceDesc* pResDesc);
/**
 * @brief Destroy a surface object.
 *
 * @param [in] surfaceObject  Surface object to be destroyed.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject);
// end of surface
/**
* @}
*/
#ifdef __cplusplus
} /* extern "c" */
#endif
#ifdef __cplusplus
#if defined(__clang__) && defined(__HIP__)
template <typename T>
static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
    T f, size_t dynSharedMemPerBlk = 0, int blockSizeLimit = 0) {
    return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, reinterpret_cast<const void*>(f),dynSharedMemPerBlk,blockSizeLimit);
}
template <typename T>
static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
    T f, size_t dynSharedMemPerBlk = 0, int blockSizeLimit = 0, unsigned int  flags = 0 ) {
    return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, reinterpret_cast<const void*>(f),dynSharedMemPerBlk,blockSizeLimit);
}
#endif // defined(__clang__) && defined(__HIP__)

/**
 * @brief Gets the address of a symbol.
 * @ingroup Memory
 * @param [out] devPtr - Returns device pointer associated with symbol.
 * @param [in] symbol - Device symbol.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
template <typename T>
hipError_t hipGetSymbolAddress(void** devPtr, const T &symbol) {
  return ::hipGetSymbolAddress(devPtr, (const void *)&symbol);
}
/**
 * @ingroup Memory
 * @brief Gets the size of a symbol.
 *
 * @param [out] size - Returns the size of a symbol.
 * @param [in] symbol - Device symbol address.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
template <typename T>
hipError_t hipGetSymbolSize(size_t* size, const T &symbol) {
  return ::hipGetSymbolSize(size, (const void *)&symbol);
}

/**
 * @ingroup Memory
 * @brief Copies data to the given symbol on the device.
 *
 * @returns #hipSuccess, #hipErrorInvalidMemcpyDirection, #hipErrorInvalidValue
 *
 * @see hipMemcpyToSymbol
 */
template <typename T>
hipError_t hipMemcpyToSymbol(const T& symbol, const void* src, size_t sizeBytes,
                             size_t offset __dparm(0),
                             hipMemcpyKind kind __dparm(hipMemcpyHostToDevice)) {
  return ::hipMemcpyToSymbol((const void*)&symbol, src, sizeBytes, offset, kind);
}
/**
 * @ingroup Memory
 * @brief Copies data to the given symbol on the device asynchronously on the stream.
 *
 * @returns #hipSuccess, #hipErrorInvalidMemcpyDirection, #hipErrorInvalidValue
 *
 * @see hipMemcpyToSymbolAsync
 */
template <typename T>
hipError_t hipMemcpyToSymbolAsync(const T& symbol, const void* src, size_t sizeBytes, size_t offset,
                                  hipMemcpyKind kind, hipStream_t stream __dparm(0)) {
  return ::hipMemcpyToSymbolAsync((const void*)&symbol, src, sizeBytes, offset, kind, stream);
}
/**
 * @brief Copies data from the given symbol on the device.
 * @ingroup Memory
 * @returns #hipSuccess, #hipErrorInvalidMemcpyDirection, #hipErrorInvalidValue
 *
 * @see hipMemcpyFromSymbol
 */
template <typename T>
hipError_t hipMemcpyFromSymbol(void* dst, const T &symbol,
                               size_t sizeBytes, size_t offset __dparm(0),
                               hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost)) {
  return ::hipMemcpyFromSymbol(dst, (const void*)&symbol, sizeBytes, offset, kind);
}
/**
 * @brief Copies data from the given symbol on the device asynchronously on the stream.
 * @ingroup Memory
 * @returns #hipSuccess, #hipErrorInvalidMemcpyDirection, #hipErrorInvalidValue
 *
 * @see hipMemcpyFromSymbolAsync
 */
template <typename T>
hipError_t hipMemcpyFromSymbolAsync(void* dst, const T& symbol, size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream __dparm(0)) {
  return ::hipMemcpyFromSymbolAsync(dst, (const void*)&symbol, sizeBytes, offset, kind, stream);
}

/**
 * @brief Returns occupancy for a kernel function.
 * @ingroup Occupancy
 * @param [out] numBlocks - Pointer of occupancy in number of blocks.
 * @param [in] f - The kernel function to launch on the device.
 * @param [in] blockSize - The block size as kernel launched.
 * @param [in] dynSharedMemPerBlk - Dynamic shared memory in bytes per block.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
template <class T>
inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, T f, int blockSize, size_t dynSharedMemPerBlk) {
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks, reinterpret_cast<const void*>(f), blockSize, dynSharedMemPerBlk);
}
/**
 * @brief Returns occupancy for a device function with the specified flags.
 *
 * @ingroup Occupancy
 * @param [out] numBlocks - Pointer of occupancy in number of blocks.
 * @param [in] f - The kernel function to launch on the device.
 * @param [in] blockSize - The block size as kernel launched.
 * @param [in] dynSharedMemPerBlk - Dynamic shared memory in bytes per block.
 * @param [in] flags - Flag to handle the behavior for the occupancy calculator.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */
template <class T>
inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, T f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
    return hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, reinterpret_cast<const void*>(f), blockSize, dynSharedMemPerBlk, flags);
}
/**
 * @brief Returns grid and block size that achieves maximum potential occupancy for a device function
 *
 * @ingroup Occupancy
 * Returns in \p *min_grid_size and \p *block_size a suggested grid /
 * block size pair that achieves the best potential occupancy
 * (i.e. the maximum number of active warps on the current device with the smallest number
 * of blocks for a particular function).
 *
 * @param [out] min_grid_size minimum grid size needed to achieve the best potential occupancy
 * @param [out] block_size    block size required for the best potential occupancy
 * @param [in]  func          device function symbol
 * @param [in]  block_size_to_dynamic_smem_size - a unary function/functor that takes block size,
 * and returns the size, in bytes, of dynamic shared memory needed for a block
 * @param [in]  block_size_limit the maximum block size \p func is designed to work with. 0 means no limit.
 * @param [in]  flags         reserved
 *
 * @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue,
 * #hipErrorUnknown
 */
template<typename UnaryFunction, class T>
static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
    int*          min_grid_size,
    int*          block_size,
    T             func,
    UnaryFunction block_size_to_dynamic_smem_size,
    int           block_size_limit = 0,
    unsigned int  flags = 0) {
  if (min_grid_size == nullptr || block_size == nullptr ||
      reinterpret_cast<const void*>(func) == nullptr) {
    return hipErrorInvalidValue;
  }

  int dev;
  hipError_t status;
  if ((status = hipGetDevice(&dev)) != hipSuccess) {
    return status;
  }

  int max_threads_per_cu;
  if ((status = hipDeviceGetAttribute(&max_threads_per_cu,
      hipDeviceAttributeMaxThreadsPerMultiProcessor, dev)) != hipSuccess) {
    return status;
  }

  int warp_size;
  if ((status = hipDeviceGetAttribute(&warp_size,
      hipDeviceAttributeWarpSize, dev)) != hipSuccess) {
    return status;
  }

  int max_cu_count;
  if ((status = hipDeviceGetAttribute(&max_cu_count,
      hipDeviceAttributeMultiprocessorCount, dev)) != hipSuccess) {
    return status;
  }

  struct hipFuncAttributes attr;
  if ((status = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(func))) != hipSuccess) {
    return status;
  }

  // Initial limits for the execution
  const int func_max_threads_per_block = attr.maxThreadsPerBlock;
  if (block_size_limit == 0) {
    block_size_limit = func_max_threads_per_block;
  }

  if (func_max_threads_per_block < block_size_limit) {
    block_size_limit = func_max_threads_per_block;
  }

  const int block_size_limit_aligned =
    ((block_size_limit + (warp_size - 1)) / warp_size) * warp_size;

  // For maximum search
  int max_threads = 0;
  int max_block_size{};
  int max_num_blocks{};
  for (int block_size_check_aligned = block_size_limit_aligned;
       block_size_check_aligned > 0;
       block_size_check_aligned -= warp_size) {
    // Make sure the logic uses the requested limit and not aligned
    int block_size_check = (block_size_limit < block_size_check_aligned) ?
        block_size_limit : block_size_check_aligned;

    size_t dyn_smem_size = block_size_to_dynamic_smem_size(block_size_check);
    int optimal_blocks;
    if ((status = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &optimal_blocks, func, block_size_check, dyn_smem_size, flags)) != hipSuccess) {
      return status;
    }

    int total_threads = block_size_check * optimal_blocks;
    if (total_threads > max_threads) {
      max_block_size = block_size_check;
      max_num_blocks = optimal_blocks;
      max_threads = total_threads;
    }

    // Break if the logic reached possible maximum
    if (max_threads_per_cu == max_threads) {
      break;
    }
  }

  // Grid size is the number of blocks per CU * CU count
  *min_grid_size = max_num_blocks * max_cu_count;
  *block_size = max_block_size;

  return status;
}

/**
 * @brief Returns grid and block size that achieves maximum potential occupancy for a device function
 *
 * @ingroup Occupancy
 * Returns in \p *min_grid_size and \p *block_size a suggested grid /
 * block size pair that achieves the best potential occupancy
 * (i.e. the maximum number of active warps on the current device with the smallest number
 * of blocks for a particular function).
 *
 * @param [out] min_grid_size minimum grid size needed to achieve the best potential occupancy
 * @param [out] block_size    block size required for the best potential occupancy
 * @param [in]  func          device function symbol
 * @param [in]  block_size_to_dynamic_smem_size - a unary function/functor that takes block size,
 * and returns the size, in bytes, of dynamic shared memory needed for a block
 * @param [in]  block_size_limit the maximum block size \p func is designed to work with. 0 means no limit.
 *
 * @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue,
 * #hipErrorUnknown
 */
template<typename UnaryFunction, class T>
static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSizeVariableSMem(
    int*          min_grid_size,
    int*          block_size,
    T             func,
    UnaryFunction block_size_to_dynamic_smem_size,
    int           block_size_limit = 0)
{
    return hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(min_grid_size, block_size, func,
      block_size_to_dynamic_smem_size, block_size_limit);
}
/**
 * @brief Returns grid and block size that achieves maximum potential occupancy for a device function
 *
 * @ingroup Occupancy
 *
 * Returns in \p *min_grid_size and \p *block_size a suggested grid /
 * block size pair that achieves the best potential occupancy
 * (i.e. the maximum number of active warps on the current device with the smallest number
 * of blocks for a particular function).
 *
 * @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 * @see hipOccupancyMaxPotentialBlockSize
 */
template <typename F>
inline hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                                    F kernel, size_t dynSharedMemPerBlk, uint32_t blockSizeLimit) {
return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize,(hipFunction_t)kernel, dynSharedMemPerBlk, blockSizeLimit);
}
/**
 * @brief Launches a device function
 *
 * @ingroup Execution
 *
 * @param [in] f  device function symbol
 * @param [in] gridDim    grid dimentions
 * @param [in]  blockDim  block dimentions
 * @param [in]  kernelParams  kernel parameters
 * @param [in]  sharedMemBytes  shared memory in bytes
 * @param [in]  stream  stream on which kernel launched
 *
 * @return #hipSuccess, #hipErrorLaunchFailure, #hipErrorInvalidValue,
 * #hipErrorInvalidResourceHandle
 *
 */
template <class T>
inline hipError_t hipLaunchCooperativeKernel(T f, dim3 gridDim, dim3 blockDim,
                                             void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream) {
    return hipLaunchCooperativeKernel(reinterpret_cast<const void*>(f), gridDim,
                                      blockDim, kernelParams, sharedMemBytes, stream);
}
/**
 * @brief Launches device function on multiple devices where thread blocks can cooperate and
 * synchronize on execution.
 *
 * @ingroup Execution
 *
 * @param [in] launchParamsList  list of kernel launch parameters, one per device
 * @param [in] numDevices  size of launchParamsList array
 * @param [in]  flags  flag to handle launch behavior
 *
 * @return #hipSuccess, #hipErrorLaunchFailure, #hipErrorInvalidValue,
 * #hipErrorInvalidResourceHandle
 *
 */
template <class T>
inline hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                        unsigned int  numDevices, unsigned int  flags = 0) {
    return hipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
}
/**
 *
 * @ingroup Module
 *
 * @brief Launches kernels on multiple devices and guarantees all specified kernels are dispatched
 * on respective streams before enqueuing any other work on the specified streams from any other threads
 *
 *
 * @param [in] launchParamsList         List of launch parameters, one per device.
 * @param [in] numDevices               Size of the launchParamsList array.
 * @param [in] flags                    Flags to control launch behavior.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
template <class T>
inline hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                     unsigned int  numDevices, unsigned int  flags = 0) {
    return hipExtLaunchMultiKernelMultiDevice(launchParamsList, numDevices, flags);
}
/**
 * @brief Binds a memory area to a texture.
 *
 * @ingroup TextureD
 *
 * @param [in] offset  Offset in bytes.
 * @param [in] tex  Texture to bind.
 * @param [in] devPtr  Pointer of memory on the device.
 * @param [in] size  Size of memory in bites.
 *
 * @warning This API is deprecated.
 *
 */
template <class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTexture(size_t* offset, const struct texture<T, dim, readMode>& tex,
                                        const void* devPtr, size_t size = UINT_MAX) {
    return hipBindTexture(offset, &tex, devPtr, &tex.channelDesc, size);
}
/**
 * @brief Binds a memory area to a texture.
 *
 * @ingroup TextureD
 *
 * @param [in] offset  Offset in bytes.
 * @param [in] tex  Texture to bind.
 * @param [in] devPtr  Pointer of memory on the device.
 * @param [in] desc  Texture channel format.
 * @param [in] size  Size of memory in bites.
 *
 * @warning This API is deprecated.
 *
 */
template <class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t
    hipBindTexture(size_t* offset, const struct texture<T, dim, readMode>& tex, const void* devPtr,
                   const struct hipChannelFormatDesc& desc, size_t size = UINT_MAX) {
    return hipBindTexture(offset, &tex, devPtr, &desc, size);
}
/**
 * @brief Binds a 2D memory area to a texture.
 *
 * @ingroup TextureD
 *
 * @param [in] offset  Offset in bytes.
 * @param [in] tex  Texture to bind.
 * @param [in] devPtr  Pointer of 2D memory area on the device.
 * @param [in] width  Width in texel units.
 * @param [in] height  Height in texel units.
 * @param [in] pitch  Pitch in bytes.
 *
 * @warning This API is deprecated.
 *
 */
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTexture2D(
    size_t *offset,
    const struct texture<T, dim, readMode> &tex,
    const void *devPtr,
    size_t width,
    size_t height,
    size_t pitch)
{
    return hipBindTexture2D(offset, &tex, devPtr, &tex.channelDesc, width, height, pitch);
}
/**
 * @brief Binds a 2D memory area to a texture.
 *
 * @ingroup TextureD
 *
 * @param [in] offset  Offset in bytes.
 * @param [in] tex  Texture to bind.
 * @param [in] devPtr  Pointer of 2D memory area on the device.
 * @param [in] desc  Texture channel format.
 * @param [in] width  Width in texel units.
 * @param [in] height  Height in texel units.
 * @param [in] pitch  Pitch in bytes.
 *
 * @warning This API is deprecated.
 *
 */
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTexture2D(
  size_t *offset,
  const struct texture<T, dim, readMode> &tex,
  const void *devPtr,
  const struct hipChannelFormatDesc &desc,
  size_t width,
  size_t height,
  size_t pitch)
{
  return hipBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch);
}
/**
 * @brief Binds an array to a texture.
 *
 * @ingroup TextureD
 *
 * @param [in] tex  Texture to bind.
 * @param [in] array  Array of memory on the device.
 *
 * @warning This API is deprecated.
 *
 */
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTextureToArray(
    const struct texture<T, dim, readMode> &tex,
    hipArray_const_t array)
{
    struct hipChannelFormatDesc desc;
    hipError_t err = hipGetChannelDesc(&desc, array);
    return (err == hipSuccess) ? hipBindTextureToArray(&tex, array, &desc) : err;
}
/**
 * @brief Binds an array to a texture.
 *
 * @ingroup TextureD
 *
 * @param [in] tex  Texture to bind.
 * @param [in] array  Array of memory on the device.
 * @param [in] desc  Texture channel format.
 *
 * @warning This API is deprecated.
 *
 */
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTextureToArray(
    const struct texture<T, dim, readMode> &tex,
    hipArray_const_t array,
    const struct hipChannelFormatDesc &desc)
{
    return hipBindTextureToArray(&tex, array, &desc);
}
/**
 * @brief Binds a mipmapped array to a texture.
 *
 * @ingroup TextureD
 *
 * @param [in] tex  Texture to bind.
 * @param [in] mipmappedArray  Mipmapped Array of memory on the device.
 *
 * @warning This API is deprecated.
 *
 */
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTextureToMipmappedArray(
    const struct texture<T, dim, readMode> &tex,
    hipMipmappedArray_const_t mipmappedArray)
{
    struct hipChannelFormatDesc desc;
    hipArray_t levelArray;
    hipError_t err = hipGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0);
    if (err != hipSuccess) {
        return err;
    }
    err = hipGetChannelDesc(&desc, levelArray);
    return (err == hipSuccess) ? hipBindTextureToMipmappedArray(&tex, mipmappedArray, &desc) : err;
}
/**
 * @brief Binds a mipmapped array to a texture.
 *
 * @ingroup TextureD
 *
 * @param [in] tex  Texture to bind.
 * @param [in] mipmappedArray  Mipmapped Array of memory on the device.
 * @param [in] desc  Texture channel format.
 *
 * @warning This API is deprecated.
 *
 */
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTextureToMipmappedArray(
    const struct texture<T, dim, readMode> &tex,
    hipMipmappedArray_const_t mipmappedArray,
    const struct hipChannelFormatDesc &desc)
{
    return hipBindTextureToMipmappedArray(&tex, mipmappedArray, &desc);
}
/**
 * @brief Unbinds a texture.
 *
 * @ingroup TextureD
 *
 * @param [in] tex  Texture to unbind.
 *
 * @warning This API is deprecated.
 *
 */
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipUnbindTexture(
    const struct texture<T, dim, readMode> &tex)
{
    return hipUnbindTexture(&tex);
}
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 * @ingroup StreamO
 * @{
 *
 *  This section describes wrappers for stream Ordered allocation from memory pool functions of
 *  HIP runtime API.
 *
 *  @note  APIs in this section are implemented on Linux, under development on Windows.
 *
 */

/**
 * @brief C++ wrappers for allocations from a memory pool
 *
 * This is an alternate C++ calls for @p hipMallocFromPoolAsync made available through
 * function overloading.
 *
 * @see hipMallocFromPoolAsync
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
static inline hipError_t hipMallocAsync(
  void**        dev_ptr,
  size_t        size,
  hipMemPool_t  mem_pool,
  hipStream_t   stream) {
  return hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream);
}
/**
 * @brief C++ wrappers for allocations from a memory pool on the stream
 *
 * This is an alternate C++ calls for @p hipMallocFromPoolAsync made available through
 * function overloading.
 *
 * @see hipMallocFromPoolAsync
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
template<class T>
static inline hipError_t hipMallocAsync(
  T**           dev_ptr,
  size_t        size,
  hipMemPool_t  mem_pool,
  hipStream_t   stream) {
  return hipMallocFromPoolAsync(reinterpret_cast<void**>(dev_ptr), size, mem_pool, stream);
}
/**
 * @brief C++ wrappers for allocations from a memory pool
 *
 * This is an alternate C++ calls for @p hipMallocFromPoolAsync made available through
 * function overloading.
 *
 * @see hipMallocFromPoolAsync
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
template<class T>
static inline hipError_t hipMallocAsync(
  T**           dev_ptr,
  size_t        size,
  hipStream_t   stream) {
  return hipMallocAsync(reinterpret_cast<void**>(dev_ptr), size, stream);
}
/**
 * @brief C++ wrappers for allocations from a memory pool
 *
 * This is an alternate C++ calls for @p hipMallocFromPoolAsync made available through
 * function overloading.
 *
 * @see hipMallocFromPoolAsync
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
template<class T>
static inline hipError_t hipMallocFromPoolAsync(
  T**           dev_ptr,
  size_t        size,
  hipMemPool_t  mem_pool,
  hipStream_t   stream) {
  return hipMallocFromPoolAsync(reinterpret_cast<void**>(dev_ptr), size, mem_pool, stream);
}
/**
* @}
*/


#endif // __cplusplus

#ifdef __GNUC__
#pragma GCC visibility pop
#endif


#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "hip/nvidia_detail/nvidia_hip_runtime_api.h"
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif


/**
 * @brief: C++ wrapper for hipMalloc
 * @ingroup Memory
 * Perform automatic type conversion to eliminate need for excessive typecasting (ie void**)
 *
 * __HIP_DISABLE_CPP_FUNCTIONS__ macro can be defined to suppress these
 * wrappers. It is useful for applications which need to obtain decltypes of
 * HIP runtime APIs.
 *
 * @see hipMalloc
 */
#if defined(__cplusplus) && !defined(__HIP_DISABLE_CPP_FUNCTIONS__)
template <class T>
static inline hipError_t hipMalloc(T** devPtr, size_t size) {
    return hipMalloc((void**)devPtr, size);
}
/**
 * @brief: C++ wrapper for hipHostMalloc
 * @ingroup Memory
 * Provide an override to automatically typecast the pointer type from void**, and also provide a
 * default for the flags.
 *
 * __HIP_DISABLE_CPP_FUNCTIONS__ macro can be defined to suppress these
 * wrappers. It is useful for applications which need to obtain decltypes of
 * HIP runtime APIs.
 *
 * @see hipHostMalloc
 */
template <class T>
static inline hipError_t hipHostMalloc(T** ptr, size_t size,
                                       unsigned int flags = hipHostMallocDefault) {
    return hipHostMalloc((void**)ptr, size, flags);
}
/**
 * @brief: C++ wrapper for hipMallocManaged
 *
 * @ingroup MemoryM
 * Provide an override to automatically typecast the pointer type from void**, and also provide a
 * default for the flags.
 *
 * __HIP_DISABLE_CPP_FUNCTIONS__ macro can be defined to suppress these
 * wrappers. It is useful for applications which need to obtain decltypes of
 * HIP runtime APIs.
 *
 * @see hipMallocManaged
 *
 */
template <class T>
static inline hipError_t hipMallocManaged(T** devPtr, size_t size,
                                       unsigned int flags = hipMemAttachGlobal) {
    return hipMallocManaged((void**)devPtr, size, flags);
}

#endif
#endif
// doxygen end HIP API
/**
 * @}
 */
#include <hip/amd_detail/amd_hip_runtime_pt_api.h>

#if USE_PROF_API
#include <hip/amd_detail/hip_prof_str.h>
#endif
