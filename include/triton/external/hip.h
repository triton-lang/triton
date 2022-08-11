#ifndef __external_hip_h__
#define __external_hip_h__

/*
 * @brief hipError_t
 * @enum
 * @ingroup Enumerations
 */
// Developer note - when updating these, update the hipErrorName and hipErrorString functions in
// NVCC and HCC paths Also update the hipCUDAErrorTohipError function in NVCC path.

// Ignoring error-code return values from hip APIs is discouraged. On C++17,
// we can make that yield a warning

/*
 * @brief hipError_t
 * @enum
 * @ingroup Enumerations
 */
// Developer note - when updating these, update the hipErrorName and hipErrorString functions in
// NVCC and HCC paths Also update the hipCUDAErrorTohipError function in NVCC path.

#include <cstddef>

typedef enum hipError_t {
    hipSuccess = 0,  ///< Successful completion.
    hipErrorInvalidValue = 1,  ///< One or more of the parameters passed to the API call is NULL
                               ///< or not in an acceptable range.
    hipErrorOutOfMemory = 2,
    // Deprecated
    hipErrorMemoryAllocation = 2,  ///< Memory allocation error.
    hipErrorNotInitialized = 3,
    // Deprecated
    hipErrorInitializationError = 3,
    hipErrorDeinitialized = 4,
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,
    hipErrorInvalidPitchValue = 12,
    hipErrorInvalidSymbol = 13,
    hipErrorInvalidDevicePointer = 17,  ///< Invalid Device Pointer
    hipErrorInvalidMemcpyDirection = 21,  ///< Invalid memory copy direction
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,
    hipErrorNoDevice = 100,  ///< Call to hipGetDeviceCount returned 0 devices
    hipErrorInvalidDevice = 101,  ///< DeviceID must be in range 0...#compute-devices.
    hipErrorInvalidImage = 200,
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
    hipErrorUnsupportedLimit = 215,
    hipErrorContextAlreadyInUse = 216,
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,  ///< In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,
    hipErrorFileNotFound = 301,
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,
    hipErrorOperatingSystem = 304,
    hipErrorInvalidHandle = 400,
    // Deprecated
    hipErrorInvalidResourceHandle = 400,  ///< Resource handle (hipEvent_t or hipStream_t) invalid.
    hipErrorNotFound = 500,
    hipErrorNotReady = 600,  ///< Indicates that asynchronous operations enqueued earlier are not
                             ///< ready.  This is not actually an error, but is used to distinguish
                             ///< from hipSuccess (which indicates completion).  APIs that return
                             ///< this error include hipEventQuery and hipStreamQuery.
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,  ///< Out of resources error.
    hipErrorLaunchTimeOut = 702,
    hipErrorPeerAccessAlreadyEnabled =
        704,  ///< Peer access was already enabled from the current device.
    hipErrorPeerAccessNotEnabled =
        705,  ///< Peer access was never enabled from the current device.
    hipErrorSetOnActiveProcess = 708,
    hipErrorAssert = 710,  ///< Produced when the kernel calls assert.
    hipErrorHostMemoryAlreadyRegistered =
        712,  ///< Produced when trying to lock a page-locked memory.
    hipErrorHostMemoryNotRegistered =
        713,  ///< Produced when trying to unlock a non-page-locked memory.
    hipErrorLaunchFailure =
        719,  ///< An exception occurred on the device while executing a kernel.
    hipErrorCooperativeLaunchTooLarge =
        720,  ///< This error indicates that the number of blocks launched per grid for a kernel
              ///< that was launched via cooperative launch APIs exceeds the maximum number of
              ///< allowed blocks for the current device
    hipErrorNotSupported = 801,  ///< Produced when the hip API is not supported/implemented
    hipErrorUnknown = 999,  //< Unknown error.
    // HSA Runtime Error Codes start here.
    hipErrorRuntimeMemory = 1052,  ///< HSA runtime memory call returned error.  Typically not seen
                                   ///< in production systems.
    hipErrorRuntimeOther = 1053,  ///< HSA runtime call other than memory returned error.  Typically
                                  ///< not seen in production systems.
    hipErrorTbd  ///< Marker that more error codes are needed.
} hipError_t;


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

#define hipIpcMemLazyEnablePeerAccess 0

#define HIP_IPC_HANDLE_SIZE 64

typedef struct hipIpcMemHandle_st {
    char reserved[HIP_IPC_HANDLE_SIZE];
} hipIpcMemHandle_t;

typedef struct hipIpcEventHandle_st {
    char reserved[HIP_IPC_HANDLE_SIZE];
} hipIpcEventHandle_t;

typedef struct ihipModule_t* hipModule_t;

typedef struct ihipModuleSymbol_t* hipFunction_t;

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

/*
 * @brief hipDeviceAttribute_t
 * @enum
 * @ingroup Enumerations
 */
typedef enum hipDeviceAttribute_t {
    hipDeviceAttributeMaxThreadsPerBlock,       ///< Maximum number of threads per block.
    hipDeviceAttributeMaxBlockDimX,             ///< Maximum x-dimension of a block.
    hipDeviceAttributeMaxBlockDimY,             ///< Maximum y-dimension of a block.
    hipDeviceAttributeMaxBlockDimZ,             ///< Maximum z-dimension of a block.
    hipDeviceAttributeMaxGridDimX,              ///< Maximum x-dimension of a grid.
    hipDeviceAttributeMaxGridDimY,              ///< Maximum y-dimension of a grid.
    hipDeviceAttributeMaxGridDimZ,              ///< Maximum z-dimension of a grid.
    hipDeviceAttributeMaxSharedMemoryPerBlock,  ///< Maximum shared memory available per block in
                                                ///< bytes.
    hipDeviceAttributeTotalConstantMemory,      ///< Constant memory size in bytes.
    hipDeviceAttributeWarpSize,                 ///< Warp size in threads.
    hipDeviceAttributeMaxRegistersPerBlock,  ///< Maximum number of 32-bit registers available to a
                                             ///< thread block. This number is shared by all thread
                                             ///< blocks simultaneously resident on a
                                             ///< multiprocessor.
    hipDeviceAttributeClockRate,             ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeMemoryClockRate,       ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeMemoryBusWidth,        ///< Global memory bus width in bits.
    hipDeviceAttributeMultiprocessorCount,   ///< Number of multiprocessors on the device.
    hipDeviceAttributeComputeMode,           ///< Compute mode that device is currently in.
    hipDeviceAttributeL2CacheSize,  ///< Size of L2 cache in bytes. 0 if the device doesn't have L2
                                    ///< cache.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,  ///< Maximum resident threads per
                                                    ///< multiprocessor.
    hipDeviceAttributeComputeCapabilityMajor,       ///< Major compute capability version number.
    hipDeviceAttributeComputeCapabilityMinor,       ///< Minor compute capability version number.
    hipDeviceAttributeConcurrentKernels,  ///< Device can possibly execute multiple kernels
                                          ///< concurrently.
    hipDeviceAttributePciBusId,           ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,        ///< PCI Device ID.
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,  ///< Maximum Shared Memory Per
                                                         ///< Multiprocessor.
    hipDeviceAttributeIsMultiGpuBoard,                   ///< Multiple GPU devices.
    hipDeviceAttributeIntegrated,                        ///< iGPU
    hipDeviceAttributeCooperativeLaunch,                 ///< Support cooperative launch
    hipDeviceAttributeCooperativeMultiDeviceLaunch,      ///< Support cooperative launch on multiple devices
    hipDeviceAttributeMaxTexture1DWidth,    ///< Maximum number of elements in 1D images
    hipDeviceAttributeMaxTexture2DWidth,    ///< Maximum dimension width of 2D images in image elements
    hipDeviceAttributeMaxTexture2DHeight,   ///< Maximum dimension height of 2D images in image elements
    hipDeviceAttributeMaxTexture3DWidth,    ///< Maximum dimension width of 3D images in image elements
    hipDeviceAttributeMaxTexture3DHeight,   ///< Maximum dimensions height of 3D images in image elements
    hipDeviceAttributeMaxTexture3DDepth,    ///< Maximum dimensions depth of 3D images in image elements

    hipDeviceAttributeHdpMemFlushCntl,      ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl,      ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register

    hipDeviceAttributeMaxPitch,             ///< Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeTextureAlignment,     ///<Alignment requirement for textures
    hipDeviceAttributeTexturePitchAlignment, ///<Pitch alignment requirement for 2D texture references bound to pitched memory;
    hipDeviceAttributeKernelExecTimeout,    ///<Run time limit for kernels executed on the device
    hipDeviceAttributeCanMapHostMemory,     ///<Device can map host memory into device address space
    hipDeviceAttributeEccEnabled,           ///<Device has ECC support enabled

    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,        ///< Supports cooperative launch on multiple
                                                                  ///devices with unmatched functions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,     ///< Supports cooperative launch on multiple
                                                                  ///devices with unmatched grid dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,    ///< Supports cooperative launch on multiple
                                                                  ///devices with unmatched block dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,   ///< Supports cooperative launch on multiple
                                                                  ///devices with unmatched shared memories
    hipDeviceAttributeAsicRevision,         ///< Revision of the GPU in this device
    hipDeviceAttributeManagedMemory,        ///< Device supports allocating managed memory on this system
    hipDeviceAttributeDirectManagedMemAccessFromHost, ///< Host can directly access managed memory on
                                                      /// the device without migration
    hipDeviceAttributeConcurrentManagedAccess,  ///< Device can coherently access managed memory
                                                /// concurrently with the CPU
    hipDeviceAttributePageableMemoryAccess,     ///< Device supports coherently accessing pageable memory
                                                /// without calling hipHostRegister on it
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables, ///< Device accesses pageable memory via
                                                              /// the host's page tables
    hipDeviceAttributeCanUseStreamWaitValue ///< '1' if Device supports hipStreamWaitValue32() and
                                            ///< hipStreamWaitValue64() , '0' otherwise.

} hipDeviceAttribute_t;

typedef void* hipDeviceptr_t;

/*
 * @brief hipJitOption
 * @enum
 * @ingroup Enumerations
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


#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define HIP_LAUNCH_PARAM_END ((void*)0x03)

#endif