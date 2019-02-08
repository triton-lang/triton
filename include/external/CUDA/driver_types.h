/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__DRIVER_TYPES_H__)
#define __DRIVER_TYPES_H__

#include "host_defines.h"
#include "vector_types.h"

/**
 * \defgroup CUDART_TYPES Data types used by CUDA Runtime
 * \ingroup CUDART
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*  TYPE DEFINITIONS USED BY RUNTIME API                                        *
*                                                                              *
*******************************************************************************/

#if !defined(__CUDA_INTERNAL_COMPILATION__)

#if !defined(__CUDACC_RTC__)
#include <limits.h>
#include <stddef.h>
#endif /* !defined(__CUDACC_RTC__) */

#define cudaHostAllocDefault                0x00  /**< Default page-locked allocation flag */
#define cudaHostAllocPortable               0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostAllocMapped                 0x02  /**< Map allocation into device space */
#define cudaHostAllocWriteCombined          0x04  /**< Write-combined memory */

#define cudaHostRegisterDefault             0x00  /**< Default host memory registration flag */
#define cudaHostRegisterPortable            0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostRegisterMapped              0x02  /**< Map registered memory into device space */
#define cudaHostRegisterIoMemory            0x04  /**< Memory-mapped I/O space */

#define cudaPeerAccessDefault               0x00  /**< Default peer addressing enable flag */

#define cudaStreamDefault                   0x00  /**< Default stream flag */
#define cudaStreamNonBlocking               0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */

 /**
 * Legacy stream handle
 *
 * Stream handle that can be passed as a cudaStream_t to use an implicit stream
 * with legacy synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define cudaStreamLegacy                    ((cudaStream_t)0x1)

/**
 * Per-thread stream handle
 *
 * Stream handle that can be passed as a cudaStream_t to use an implicit stream
 * with per-thread synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define cudaStreamPerThread                 ((cudaStream_t)0x2)

#define cudaEventDefault                    0x00  /**< Default event flag */
#define cudaEventBlockingSync               0x01  /**< Event uses blocking synchronization */
#define cudaEventDisableTiming              0x02  /**< Event will not record timing data */
#define cudaEventInterprocess               0x04  /**< Event is suitable for interprocess use. cudaEventDisableTiming must be set */

#define cudaDeviceScheduleAuto              0x00  /**< Device flag - Automatic scheduling */
#define cudaDeviceScheduleSpin              0x01  /**< Device flag - Spin default scheduling */
#define cudaDeviceScheduleYield             0x02  /**< Device flag - Yield default scheduling */
#define cudaDeviceScheduleBlockingSync      0x04  /**< Device flag - Use blocking synchronization */
#define cudaDeviceBlockingSync              0x04  /**< Device flag - Use blocking synchronization 
                                                    *  \deprecated This flag was deprecated as of CUDA 4.0 and
                                                    *  replaced with ::cudaDeviceScheduleBlockingSync. */
#define cudaDeviceScheduleMask              0x07  /**< Device schedule flags mask */
#define cudaDeviceMapHost                   0x08  /**< Device flag - Support mapped pinned allocations */
#define cudaDeviceLmemResizeToMax           0x10  /**< Device flag - Keep local memory allocation after launch */
#define cudaDeviceMask                      0x1f  /**< Device flags mask */

#define cudaArrayDefault                    0x00  /**< Default CUDA array allocation flag */
#define cudaArrayLayered                    0x01  /**< Must be set in cudaMalloc3DArray to create a layered CUDA array */
#define cudaArraySurfaceLoadStore           0x02  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array */
#define cudaArrayCubemap                    0x04  /**< Must be set in cudaMalloc3DArray to create a cubemap CUDA array */
#define cudaArrayTextureGather              0x08  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array */

#define cudaIpcMemLazyEnablePeerAccess      0x01  /**< Automatically enable peer access between remote devices as needed */

#define cudaMemAttachGlobal                 0x01  /**< Memory can be accessed by any stream on any device*/
#define cudaMemAttachHost                   0x02  /**< Memory cannot be accessed by any stream on any device */
#define cudaMemAttachSingle                 0x04  /**< Memory can only be accessed by a single stream on the associated device */

#define cudaOccupancyDefault                0x00  /**< Default behavior */
#define cudaOccupancyDisableCachingOverride 0x01  /**< Assume global caching is enabled and cannot be automatically turned off */

#define cudaCpuDeviceId                     ((int)-1) /**< Device id that represents the CPU */
#define cudaInvalidDeviceId                 ((int)-2) /**< Device id that represents an invalid device */

/**
 * If set, each kernel launched as part of ::cudaLaunchCooperativeKernelMultiDevice only
 * waits for prior work in the stream corresponding to that GPU to complete before the
 * kernel begins execution.
 */
#define cudaCooperativeLaunchMultiDeviceNoPreSync  0x01

/**
 * If set, any subsequent work pushed in a stream that participated in a call to
 * ::cudaLaunchCooperativeKernelMultiDevice will only wait for the kernel launched on
 * the GPU corresponding to that stream to complete before it begins execution.
 */
#define cudaCooperativeLaunchMultiDeviceNoPostSync 0x02

#endif /* !__CUDA_INTERNAL_COMPILATION__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/**
 * CUDA error types
 */
enum __device_builtin__ cudaError
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cudaEventQuery() and ::cudaStreamQuery()).
     */
    cudaSuccess                           =      0,
  
    /**
     * The device function being invoked (usually via ::cudaLaunchKernel()) was not
     * previously configured via the ::cudaConfigureCall() function.
     */
    cudaErrorMissingConfiguration         =      1,
  
    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    cudaErrorMemoryAllocation             =      2,
  
    /**
     * The API call failed because the CUDA driver and runtime could not be
     * initialized.
     */
    cudaErrorInitializationError          =      3,
  
    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The device cannot be used until
     * ::cudaThreadExit() is called. All existing device memory allocations
     * are invalid and must be reconstructed if the program is to continue
     * using CUDA.
     */
    cudaErrorLaunchFailure                =      4,
  
    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorPriorLaunchFailure           =      5,
  
    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
     * for more information.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchTimeout                =      6,
  
    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::cudaErrorInvalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    cudaErrorLaunchOutOfResources         =      7,
  
    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    cudaErrorInvalidDeviceFunction        =      8,
  
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See ::cudaDeviceProp for more
     * device limitations.
     */
    cudaErrorInvalidConfiguration         =      9,
  
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    cudaErrorInvalidDevice                =     10,
  
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    cudaErrorInvalidValue                 =     11,
  
    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    cudaErrorInvalidPitchValue            =     12,
  
    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    cudaErrorInvalidSymbol                =     13,
  
    /**
     * This indicates that the buffer object could not be mapped.
     */
    cudaErrorMapBufferObjectFailed        =     14,
  
    /**
     * This indicates that the buffer object could not be unmapped.
     */
    cudaErrorUnmapBufferObjectFailed      =     15,
  
    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     */
    cudaErrorInvalidHostPointer           =     16,
  
    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     */
    cudaErrorInvalidDevicePointer         =     17,
  
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    cudaErrorInvalidTexture               =     18,
  
    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
     */
    cudaErrorInvalidTextureBinding        =     19,
  
    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
     */
    cudaErrorInvalidChannelDescriptor     =     20,
  
    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::cudaMemcpyKind.
     */
    cudaErrorInvalidMemcpyDirection       =     21,
  
    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the CUDA 3.1 release.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * ::cudaGetSymbolAddress().
     */
    cudaErrorAddressOfConstant            =     22,
  
    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureFetchFailed           =     23,
  
    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureNotBound              =     24,
  
    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorSynchronizationError         =     25,
  
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by CUDA.
     */
    cudaErrorInvalidFilterSetting         =     26,
  
    /**
     * This indicates that an attempt was made to read a non-float texture as a
     * normalized float. This is not supported by CUDA.
     */
    cudaErrorInvalidNormSetting           =     27,
  
    /**
     * Mixing of device and device emulation code was not allowed.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMixedDeviceExecution         =     28,
  
    /**
     * This indicates that a CUDA Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * CUDA driver has been unloaded.
     */
    cudaErrorCudartUnloading              =     29,
  
    /**
     * This indicates that an unknown internal error has occurred.
     */
    cudaErrorUnknown                      =     30,

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of CUDA will never return this error.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorNotYetImplemented            =     31,
  
    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMemoryValueTooLarge          =     32,
  
    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::cudaStream_t and
     * ::cudaEvent_t.
     */
    cudaErrorInvalidResourceHandle        =     33,
  
    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::cudaSuccess (which indicates completion). Calls that
     * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
     */
    cudaErrorNotReady                     =     34,
  
    /**
     * This indicates that the installed NVIDIA CUDA driver is older than the
     * CUDA runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    cudaErrorInsufficientDriver           =     35,
  
    /**
     * This indicates that the user has called ::cudaSetValidDevices(),
     * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
     * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
     * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::CUcontext active on the
     * host thread.
     */
    cudaErrorSetOnActiveProcess           =     36,
  
    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    cudaErrorInvalidSurface               =     37,
  
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    cudaErrorNoDevice                     =     38,
  
    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    cudaErrorECCUncorrectable             =     39,
  
    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    cudaErrorSharedObjectSymbolNotFound   =     40,
  
    /**
     * This indicates that initialization of a shared object failed.
     */
    cudaErrorSharedObjectInitFailed       =     41,
  
    /**
     * This indicates that the ::cudaLimit passed to the API call is not
     * supported by the active device.
     */
    cudaErrorUnsupportedLimit             =     42,
  
    /**
     * This indicates that multiple global or constant variables (across separate
     * CUDA source files in the application) share the same string name.
     */
    cudaErrorDuplicateVariableName        =     43,
  
    /**
     * This indicates that multiple textures (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateTextureName         =     44,
  
    /**
     * This indicates that multiple surfaces (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateSurfaceName         =     45,
  
    /**
     * This indicates that all CUDA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long
     * running CUDA kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active CUDA work being performed.
     */
    cudaErrorDevicesUnavailable           =     46,
  
    /**
     * This indicates that the device kernel image is invalid.
     */
    cudaErrorInvalidKernelImage           =     47,
  
    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    cudaErrorNoKernelImageForDevice       =     48,
  
    /**
     * This indicates that the current context is not compatible with this
     * the CUDA Runtime. This can only occur if you are using CUDA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version 
     * of the API, because the Runtime API call expects a primary driver 
     * context and the Driver context is not primary, or because the Driver 
     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
     * with the CUDA Driver API" for more information.
     */
    cudaErrorIncompatibleDriverContext    =     49,
      
    /**
     * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    cudaErrorPeerAccessAlreadyEnabled     =     50,
    
    /**
     * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to 
     * disable peer addressing which has not been enabled yet via 
     * ::cudaDeviceEnablePeerAccess().
     */
    cudaErrorPeerAccessNotEnabled         =     51,
    
    /**
     * This indicates that a call tried to access an exclusive-thread device that 
     * is already in use by a different thread.
     */
    cudaErrorDeviceAlreadyInUse           =     54,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    cudaErrorProfilerDisabled             =     55,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cudaProfilerStart or
     * ::cudaProfilerStop without initialization.
     */
    cudaErrorProfilerNotInitialized       =     56,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStart() when profiling is already enabled.
     */
    cudaErrorProfilerAlreadyStarted       =     57,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStop() when profiling is already disabled.
     */
     cudaErrorProfilerAlreadyStopped       =    58,

    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again until ::cudaThreadExit() is called. All existing 
     * allocations are invalid and must be reconstructed if the program is to
     * continue using CUDA. 
     */
    cudaErrorAssert                        =    59,
  
    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::cudaEnablePeerAccess().
     */
    cudaErrorTooManyPeers                 =     60,
  
    /**
     * This error indicates that the memory range passed to ::cudaHostRegister()
     * has already been registered.
     */
    cudaErrorHostMemoryAlreadyRegistered  =     61,
        
    /**
     * This error indicates that the pointer passed to ::cudaHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    cudaErrorHostMemoryNotRegistered      =     62,

    /**
     * This error indicates that an OS call failed.
     */
    cudaErrorOperatingSystem              =     63,

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    cudaErrorPeerAccessUnsupported        =     64,

    /**
     * This error indicates that a device runtime grid launch did not occur 
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches. 
     */
    cudaErrorLaunchMaxDepthExceeded       =     65,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped textures which are unsupported by the device runtime. 
     * Kernels launched via the device runtime only support textures created with 
     * the Texture Object API's.
     */
    cudaErrorLaunchFileScopedTex          =     66,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    cudaErrorLaunchFileScopedSurf         =     67,

    /**
     * This error indicates that a call to ::cudaDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device 
     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on 
     * launched grids at a greater depth successfully, the maximum nested 
     * depth at which ::cudaDeviceSynchronize will be called must be specified 
     * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime. 
     * Keep in mind that additional levels of sync depth require the runtime 
     * to reserve large amounts of device memory that cannot be used for 
     * user allocations.
     */
    cudaErrorSyncDepthExceeded            =     68,

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    cudaErrorLaunchPendingCountExceeded   =     69,
    
    /**
     * This error indicates the attempted operation is not permitted.
     */
    cudaErrorNotPermitted                 =     70,

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    cudaErrorNotSupported                 =     71,

    /**
     * Device encountered an error in the call stack during kernel execution,
     * possibly due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorHardwareStackError           =     72,

    /**
     * The device encountered an illegal instruction during kernel execution
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalInstruction           =     73,

    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorMisalignedAddress            =     74,

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidAddressSpace          =     75,

    /**
     * The device encountered an invalid program counter.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidPc                    =     76,

    /**
     * The device encountered a load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalAddress               =     77,

    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    cudaErrorInvalidPtx                   =     78,

    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    cudaErrorInvalidGraphicsContext       =     79,

    /**
     * This indicates that an uncorrectable NVLink error was detected during the
     * execution.
     */
    cudaErrorNvlinkUncorrectable          =     80,

    /**
     * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
     * library is used for PTX compilation. The runtime may fall back to compiling PTX
     * if an application does not contain a suitable binary for the current device.
     */
    cudaErrorJitCompilerNotFound          =     81,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::cudaLaunchCooperativeKernel or ::cudaLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
     */
    cudaErrorCooperativeLaunchTooLarge    =     82,

    /**
     * This indicates an internal startup failure in the CUDA runtime.
     */
    cudaErrorStartupFailure               =   0x7f,

    /**
     * Any unhandled CUDA driver error is added to this value and returned via
     * the runtime. Production releases of CUDA should not return such errors.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorApiFailureBase               =  10000
};

/**
 * Channel format kind
 */
enum __device_builtin__ cudaChannelFormatKind
{
    cudaChannelFormatKindSigned           =   0,      /**< Signed channel format */
    cudaChannelFormatKindUnsigned         =   1,      /**< Unsigned channel format */
    cudaChannelFormatKindFloat            =   2,      /**< Float channel format */
    cudaChannelFormatKindNone             =   3       /**< No channel format */
};

/**
 * CUDA Channel format descriptor
 */
struct __device_builtin__ cudaChannelFormatDesc
{
    int                        x; /**< x */
    int                        y; /**< y */
    int                        z; /**< z */
    int                        w; /**< w */
    enum cudaChannelFormatKind f; /**< Channel format kind */
};

/**
 * CUDA array
 */
typedef struct cudaArray *cudaArray_t;

/**
 * CUDA array (as source copy argument)
 */
typedef const struct cudaArray *cudaArray_const_t;

struct cudaArray;

/**
 * CUDA mipmapped array
 */
typedef struct cudaMipmappedArray *cudaMipmappedArray_t;

/**
 * CUDA mipmapped array (as source argument)
 */
typedef const struct cudaMipmappedArray *cudaMipmappedArray_const_t;

struct cudaMipmappedArray;

/**
 * CUDA memory types
 */
enum __device_builtin__ cudaMemoryType
{
    cudaMemoryTypeHost   = 1, /**< Host memory */
    cudaMemoryTypeDevice = 2  /**< Device memory */
};

/**
 * CUDA memory copy types
 */
enum __device_builtin__ cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

/**
 * CUDA Pitched memory pointer
 *
 * \sa ::make_cudaPitchedPtr
 */
struct __device_builtin__ cudaPitchedPtr
{
    void   *ptr;      /**< Pointer to allocated memory */
    size_t  pitch;    /**< Pitch of allocated memory in bytes */
    size_t  xsize;    /**< Logical width of allocation in elements */
    size_t  ysize;    /**< Logical height of allocation in elements */
};

/**
 * CUDA extent
 *
 * \sa ::make_cudaExtent
 */
struct __device_builtin__ cudaExtent
{
    size_t width;     /**< Width in elements when referring to array memory, in bytes when referring to linear memory */
    size_t height;    /**< Height in elements */
    size_t depth;     /**< Depth in elements */
};

/**
 * CUDA 3D position
 *
 * \sa ::make_cudaPos
 */
struct __device_builtin__ cudaPos
{
    size_t x;     /**< x */
    size_t y;     /**< y */
    size_t z;     /**< z */
};

/**
 * CUDA 3D memory copying parameters
 */
struct __device_builtin__ cudaMemcpy3DParms
{
    cudaArray_t            srcArray;  /**< Source memory address */
    struct cudaPos         srcPos;    /**< Source position offset */
    struct cudaPitchedPtr  srcPtr;    /**< Pitched source memory address */
  
    cudaArray_t            dstArray;  /**< Destination memory address */
    struct cudaPos         dstPos;    /**< Destination position offset */
    struct cudaPitchedPtr  dstPtr;    /**< Pitched destination memory address */
  
    struct cudaExtent      extent;    /**< Requested memory copy size */
    enum cudaMemcpyKind    kind;      /**< Type of transfer */
};

/**
 * CUDA 3D cross-device memory copying parameters
 */
struct __device_builtin__ cudaMemcpy3DPeerParms
{
    cudaArray_t            srcArray;  /**< Source memory address */
    struct cudaPos         srcPos;    /**< Source position offset */
    struct cudaPitchedPtr  srcPtr;    /**< Pitched source memory address */
    int                    srcDevice; /**< Source device */
  
    cudaArray_t            dstArray;  /**< Destination memory address */
    struct cudaPos         dstPos;    /**< Destination position offset */
    struct cudaPitchedPtr  dstPtr;    /**< Pitched destination memory address */
    int                    dstDevice; /**< Destination device */
  
    struct cudaExtent      extent;    /**< Requested memory copy size */
};

/**
 * CUDA graphics interop resource
 */
struct cudaGraphicsResource;

/**
 * CUDA graphics interop register flags
 */
enum __device_builtin__ cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone             = 0,  /**< Default */
    cudaGraphicsRegisterFlagsReadOnly         = 1,  /**< CUDA will not write to this resource */ 
    cudaGraphicsRegisterFlagsWriteDiscard     = 2,  /**< CUDA will only write to and will not read from this resource */
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,  /**< CUDA will bind this resource to a surface reference */
    cudaGraphicsRegisterFlagsTextureGather    = 8   /**< CUDA will perform texture gather operations on this resource */
};

/**
 * CUDA graphics interop map flags
 */
enum __device_builtin__ cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
    cudaGraphicsMapFlagsReadOnly     = 1,  /**< CUDA will not write to this resource */
    cudaGraphicsMapFlagsWriteDiscard = 2   /**< CUDA will only write to and will not read from this resource */
};

/**
 * CUDA graphics interop array indices for cube maps
 */
enum __device_builtin__ cudaGraphicsCubeFace 
{
    cudaGraphicsCubeFacePositiveX = 0x00, /**< Positive X face of cubemap */
    cudaGraphicsCubeFaceNegativeX = 0x01, /**< Negative X face of cubemap */
    cudaGraphicsCubeFacePositiveY = 0x02, /**< Positive Y face of cubemap */
    cudaGraphicsCubeFaceNegativeY = 0x03, /**< Negative Y face of cubemap */
    cudaGraphicsCubeFacePositiveZ = 0x04, /**< Positive Z face of cubemap */
    cudaGraphicsCubeFaceNegativeZ = 0x05  /**< Negative Z face of cubemap */
};

/**
 * CUDA resource types
 */
enum __device_builtin__ cudaResourceType
{
    cudaResourceTypeArray          = 0x00, /**< Array resource */
    cudaResourceTypeMipmappedArray = 0x01, /**< Mipmapped array resource */
    cudaResourceTypeLinear         = 0x02, /**< Linear resource */
    cudaResourceTypePitch2D        = 0x03  /**< Pitch 2D resource */
};

/**
 * CUDA texture resource view formats
 */
enum __device_builtin__ cudaResourceViewFormat
{
    cudaResViewFormatNone                      = 0x00, /**< No resource view format (use underlying resource format) */
    cudaResViewFormatUnsignedChar1             = 0x01, /**< 1 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar2             = 0x02, /**< 2 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar4             = 0x03, /**< 4 channel unsigned 8-bit integers */
    cudaResViewFormatSignedChar1               = 0x04, /**< 1 channel signed 8-bit integers */
    cudaResViewFormatSignedChar2               = 0x05, /**< 2 channel signed 8-bit integers */
    cudaResViewFormatSignedChar4               = 0x06, /**< 4 channel signed 8-bit integers */
    cudaResViewFormatUnsignedShort1            = 0x07, /**< 1 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort2            = 0x08, /**< 2 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort4            = 0x09, /**< 4 channel unsigned 16-bit integers */
    cudaResViewFormatSignedShort1              = 0x0a, /**< 1 channel signed 16-bit integers */
    cudaResViewFormatSignedShort2              = 0x0b, /**< 2 channel signed 16-bit integers */
    cudaResViewFormatSignedShort4              = 0x0c, /**< 4 channel signed 16-bit integers */
    cudaResViewFormatUnsignedInt1              = 0x0d, /**< 1 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt2              = 0x0e, /**< 2 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt4              = 0x0f, /**< 4 channel unsigned 32-bit integers */
    cudaResViewFormatSignedInt1                = 0x10, /**< 1 channel signed 32-bit integers */
    cudaResViewFormatSignedInt2                = 0x11, /**< 2 channel signed 32-bit integers */
    cudaResViewFormatSignedInt4                = 0x12, /**< 4 channel signed 32-bit integers */
    cudaResViewFormatHalf1                     = 0x13, /**< 1 channel 16-bit floating point */
    cudaResViewFormatHalf2                     = 0x14, /**< 2 channel 16-bit floating point */
    cudaResViewFormatHalf4                     = 0x15, /**< 4 channel 16-bit floating point */
    cudaResViewFormatFloat1                    = 0x16, /**< 1 channel 32-bit floating point */
    cudaResViewFormatFloat2                    = 0x17, /**< 2 channel 32-bit floating point */
    cudaResViewFormatFloat4                    = 0x18, /**< 4 channel 32-bit floating point */
    cudaResViewFormatUnsignedBlockCompressed1  = 0x19, /**< Block compressed 1 */
    cudaResViewFormatUnsignedBlockCompressed2  = 0x1a, /**< Block compressed 2 */
    cudaResViewFormatUnsignedBlockCompressed3  = 0x1b, /**< Block compressed 3 */
    cudaResViewFormatUnsignedBlockCompressed4  = 0x1c, /**< Block compressed 4 unsigned */
    cudaResViewFormatSignedBlockCompressed4    = 0x1d, /**< Block compressed 4 signed */
    cudaResViewFormatUnsignedBlockCompressed5  = 0x1e, /**< Block compressed 5 unsigned */
    cudaResViewFormatSignedBlockCompressed5    = 0x1f, /**< Block compressed 5 signed */
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20, /**< Block compressed 6 unsigned half-float */
    cudaResViewFormatSignedBlockCompressed6H   = 0x21, /**< Block compressed 6 signed half-float */
    cudaResViewFormatUnsignedBlockCompressed7  = 0x22  /**< Block compressed 7 */
};

/**
 * CUDA resource descriptor
 */
struct __device_builtin__ cudaResourceDesc {
	enum cudaResourceType resType;             /**< Resource type */
	
	union {
		struct {
			cudaArray_t array;                 /**< CUDA array */
		} array;
        struct {
            cudaMipmappedArray_t mipmap;       /**< CUDA mipmapped array */
        } mipmap;
		struct {
			void *devPtr;                      /**< Device pointer */
			struct cudaChannelFormatDesc desc; /**< Channel descriptor */
			size_t sizeInBytes;                /**< Size in bytes */
		} linear;
		struct {
			void *devPtr;                      /**< Device pointer */
			struct cudaChannelFormatDesc desc; /**< Channel descriptor */
			size_t width;                      /**< Width of the array in elements */
			size_t height;                     /**< Height of the array in elements */
			size_t pitchInBytes;               /**< Pitch between two rows in bytes */
		} pitch2D;
	} res;
};

/**
 * CUDA resource view descriptor
 */
struct __device_builtin__ cudaResourceViewDesc
{
    enum cudaResourceViewFormat format;           /**< Resource view format */
    size_t                      width;            /**< Width of the resource view */
    size_t                      height;           /**< Height of the resource view */
    size_t                      depth;            /**< Depth of the resource view */
    unsigned int                firstMipmapLevel; /**< First defined mipmap level */
    unsigned int                lastMipmapLevel;  /**< Last defined mipmap level */
    unsigned int                firstLayer;       /**< First layer index */
    unsigned int                lastLayer;        /**< Last layer index */
};

/**
 * CUDA pointer attributes
 */
struct __device_builtin__ cudaPointerAttributes
{
    /**
     * The physical location of the memory, ::cudaMemoryTypeHost or 
     * ::cudaMemoryTypeDevice.
     */
    enum cudaMemoryType memoryType;

    /** 
     * The device against which the memory was allocated or registered.
     * If the memory type is ::cudaMemoryTypeDevice then this identifies 
     * the device on which the memory referred physically resides.  If
     * the memory type is ::cudaMemoryTypeHost then this identifies the 
     * device which was current when the memory was allocated or registered
     * (and if that device is deinitialized then this allocation will vanish
     * with that device's state).
     */
    int device;

    /**
     * The address which may be dereferenced on the current device to access 
     * the memory or NULL if no such address exists.
     */
    void *devicePointer;

    /**
     * The address which may be dereferenced on the host to access the
     * memory or NULL if no such address exists.
     */
    void *hostPointer;

    /**
     * Indicates if this pointer points to managed memory
     */
    int isManaged;
};

/**
 * CUDA function attributes
 */
struct __device_builtin__ cudaFuncAttributes
{
   /**
    * The size in bytes of statically-allocated shared memory per block
    * required by this function. This does not include dynamically-allocated
    * shared memory requested by the user at runtime.
    */
   size_t sharedSizeBytes;

   /**
    * The size in bytes of user-allocated constant memory required by this
    * function.
    */
   size_t constSizeBytes;

   /**
    * The size in bytes of local memory used by each thread of this function.
    */
   size_t localSizeBytes;

   /**
    * The maximum number of threads per block, beyond which a launch of the
    * function would fail. This number depends on both the function and the
    * device on which the function is currently loaded.
    */
   int maxThreadsPerBlock;

   /**
    * The number of registers used by each thread of this function.
    */
   int numRegs;

   /**
    * The PTX virtual architecture version for which the function was
    * compiled. This value is the major PTX version * 10 + the minor PTX
    * version, so a PTX version 1.3 function would return the value 13.
    */
   int ptxVersion;

   /**
    * The binary architecture version for which the function was compiled.
    * This value is the major binary version * 10 + the minor binary version,
    * so a binary version 1.3 function would return the value 13.
    */
   int binaryVersion;

   /**
    * The attribute to indicate whether the function has been compiled with 
    * user specified option "-Xptxas --dlcm=ca" set.
    */
   int cacheModeCA;

   /**
    * The maximum size in bytes of dynamic shared memory per block for 
    * this function. Any launch must have a dynamic shared memory size
    * smaller than this value.
    */
   int maxDynamicSharedSizeBytes;

   /**
    * On devices where the L1 cache and shared memory use the same hardware resources, 
    * this sets the shared memory carveout preference, in percent of the maximum shared memory. 
    * This is only a hint, and the driver can choose a different ratio if required to execute the function.
    */
   int preferredShmemCarveout;
};

/**
 * CUDA function attributes that can be set using cudaFuncSetAttribute
 */
enum __device_builtin__ cudaFuncAttribute
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8, /**< Maximum dynamic shared memory size */
    cudaFuncAttributePreferredSharedMemoryCarveout = 9, /**< Preferred shared memory-L1 cache split ratio */
    cudaFuncAttributeMax
};

/**
 * CUDA function cache configurations
 */
enum __device_builtin__ cudaFuncCache
{
    cudaFuncCachePreferNone   = 0,    /**< Default function cache configuration, no preference */
    cudaFuncCachePreferShared = 1,    /**< Prefer larger shared memory and smaller L1 cache  */
    cudaFuncCachePreferL1     = 2,    /**< Prefer larger L1 cache and smaller shared memory */
    cudaFuncCachePreferEqual  = 3     /**< Prefer equal size L1 cache and shared memory */
};

/**
 * CUDA shared memory configuration
 */

enum __device_builtin__ cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};

/** 
 * Shared memory carveout configurations
 */
enum __device_builtin__ cudaSharedCarveout {
    cudaSharedmemCarveoutDefault      = -1,  /* * < no preference for shared memory or L1 (default) */
    cudaSharedmemCarveoutMaxShared    = 100, /* * < prefer maximum available shared memory, minimum L1 cache */
    cudaSharedmemCarveoutMaxL1        = 0    /* * < prefer maximum available L1 cache, minimum shared memory */
};

/**
 * CUDA device compute modes
 */
enum __device_builtin__ cudaComputeMode
{
    cudaComputeModeDefault          = 0,  /**< Default compute mode (Multiple threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusive        = 1,  /**< Compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device) */
    cudaComputeModeProhibited       = 2,  /**< Compute-prohibited mode (No threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusiveProcess = 3   /**< Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device) */
};

/**
 * CUDA Limits
 */
enum __device_builtin__ cudaLimit
{
    cudaLimitStackSize                    = 0x00, /**< GPU thread stack size */
    cudaLimitPrintfFifoSize               = 0x01, /**< GPU printf/fprintf FIFO size */
    cudaLimitMallocHeapSize               = 0x02, /**< GPU malloc heap size */
    cudaLimitDevRuntimeSyncDepth          = 0x03, /**< GPU device runtime synchronize depth */
    cudaLimitDevRuntimePendingLaunchCount = 0x04  /**< GPU device runtime pending launch count */
};

/**
 * CUDA Memory Advise values
 */
enum __device_builtin__ cudaMemoryAdvise
{
    cudaMemAdviseSetReadMostly          = 1, /**< Data will mostly be read and only occassionally be written to */
    cudaMemAdviseUnsetReadMostly        = 2, /**< Undo the effect of ::cudaMemAdviseSetReadMostly */
    cudaMemAdviseSetPreferredLocation   = 3, /**< Set the preferred location for the data as the specified device */
    cudaMemAdviseUnsetPreferredLocation = 4, /**< Clear the preferred location for the data */
    cudaMemAdviseSetAccessedBy          = 5, /**< Data will be accessed by the specified device, so prevent page faults as much as possible */
    cudaMemAdviseUnsetAccessedBy        = 6  /**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device */
};

/**
 * CUDA range attributes
 */
enum __device_builtin__ cudaMemRangeAttribute
{
    cudaMemRangeAttributeReadMostly           = 1, /**< Whether the range will mostly be read and only occassionally be written to */
    cudaMemRangeAttributePreferredLocation    = 2, /**< The preferred location of the range */
    cudaMemRangeAttributeAccessedBy           = 3, /**< Memory range has ::cudaMemAdviseSetAccessedBy set for specified device */
    cudaMemRangeAttributeLastPrefetchLocation = 4  /**< The last location to which the range was prefetched */
};

/**
 * CUDA Profiler Output modes
 */
enum __device_builtin__ cudaOutputMode
{
    cudaKeyValuePair    = 0x00, /**< Output mode Key-Value pair format. */
    cudaCSV             = 0x01  /**< Output mode Comma separated values format. */
};

/**
 * CUDA device attributes
 */
enum __device_builtin__ cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock             = 1,  /**< Maximum number of threads per block */
    cudaDevAttrMaxBlockDimX                   = 2,  /**< Maximum block dimension X */
    cudaDevAttrMaxBlockDimY                   = 3,  /**< Maximum block dimension Y */
    cudaDevAttrMaxBlockDimZ                   = 4,  /**< Maximum block dimension Z */
    cudaDevAttrMaxGridDimX                    = 5,  /**< Maximum grid dimension X */
    cudaDevAttrMaxGridDimY                    = 6,  /**< Maximum grid dimension Y */
    cudaDevAttrMaxGridDimZ                    = 7,  /**< Maximum grid dimension Z */
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,  /**< Maximum shared memory available per block in bytes */
    cudaDevAttrTotalConstantMemory            = 9,  /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    cudaDevAttrWarpSize                       = 10, /**< Warp size in threads */
    cudaDevAttrMaxPitch                       = 11, /**< Maximum pitch in bytes allowed by memory copies */
    cudaDevAttrMaxRegistersPerBlock           = 12, /**< Maximum number of 32-bit registers available per block */
    cudaDevAttrClockRate                      = 13, /**< Peak clock frequency in kilohertz */
    cudaDevAttrTextureAlignment               = 14, /**< Alignment requirement for textures */
    cudaDevAttrGpuOverlap                     = 15, /**< Device can possibly copy memory and execute a kernel concurrently */
    cudaDevAttrMultiProcessorCount            = 16, /**< Number of multiprocessors on device */
    cudaDevAttrKernelExecTimeout              = 17, /**< Specifies whether there is a run time limit on kernels */
    cudaDevAttrIntegrated                     = 18, /**< Device is integrated with host memory */
    cudaDevAttrCanMapHostMemory               = 19, /**< Device can map host memory into CUDA address space */
    cudaDevAttrComputeMode                    = 20, /**< Compute mode (See ::cudaComputeMode for details) */
    cudaDevAttrMaxTexture1DWidth              = 21, /**< Maximum 1D texture width */
    cudaDevAttrMaxTexture2DWidth              = 22, /**< Maximum 2D texture width */
    cudaDevAttrMaxTexture2DHeight             = 23, /**< Maximum 2D texture height */
    cudaDevAttrMaxTexture3DWidth              = 24, /**< Maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeight             = 25, /**< Maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepth              = 26, /**< Maximum 3D texture depth */
    cudaDevAttrMaxTexture2DLayeredWidth       = 27, /**< Maximum 2D layered texture width */
    cudaDevAttrMaxTexture2DLayeredHeight      = 28, /**< Maximum 2D layered texture height */
    cudaDevAttrMaxTexture2DLayeredLayers      = 29, /**< Maximum layers in a 2D layered texture */
    cudaDevAttrSurfaceAlignment               = 30, /**< Alignment requirement for surfaces */
    cudaDevAttrConcurrentKernels              = 31, /**< Device can possibly execute multiple kernels concurrently */
    cudaDevAttrEccEnabled                     = 32, /**< Device has ECC support enabled */
    cudaDevAttrPciBusId                       = 33, /**< PCI bus ID of the device */
    cudaDevAttrPciDeviceId                    = 34, /**< PCI device ID of the device */
    cudaDevAttrTccDriver                      = 35, /**< Device is using TCC driver model */
    cudaDevAttrMemoryClockRate                = 36, /**< Peak memory clock frequency in kilohertz */
    cudaDevAttrGlobalMemoryBusWidth           = 37, /**< Global memory bus width in bits */
    cudaDevAttrL2CacheSize                    = 38, /**< Size of L2 cache in bytes */
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39, /**< Maximum resident threads per multiprocessor */
    cudaDevAttrAsyncEngineCount               = 40, /**< Number of asynchronous engines */
    cudaDevAttrUnifiedAddressing              = 41, /**< Device shares a unified address space with the host */    
    cudaDevAttrMaxTexture1DLayeredWidth       = 42, /**< Maximum 1D layered texture width */
    cudaDevAttrMaxTexture1DLayeredLayers      = 43, /**< Maximum layers in a 1D layered texture */
    cudaDevAttrMaxTexture2DGatherWidth        = 45, /**< Maximum 2D texture width if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture2DGatherHeight       = 46, /**< Maximum 2D texture height if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture3DWidthAlt           = 47, /**< Alternate maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeightAlt          = 48, /**< Alternate maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepthAlt           = 49, /**< Alternate maximum 3D texture depth */
    cudaDevAttrPciDomainId                    = 50, /**< PCI domain ID of the device */
    cudaDevAttrTexturePitchAlignment          = 51, /**< Pitch alignment requirement for textures */
    cudaDevAttrMaxTextureCubemapWidth         = 52, /**< Maximum cubemap texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53, /**< Maximum cubemap layered texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54, /**< Maximum layers in a cubemap layered texture */
    cudaDevAttrMaxSurface1DWidth              = 55, /**< Maximum 1D surface width */
    cudaDevAttrMaxSurface2DWidth              = 56, /**< Maximum 2D surface width */
    cudaDevAttrMaxSurface2DHeight             = 57, /**< Maximum 2D surface height */
    cudaDevAttrMaxSurface3DWidth              = 58, /**< Maximum 3D surface width */
    cudaDevAttrMaxSurface3DHeight             = 59, /**< Maximum 3D surface height */
    cudaDevAttrMaxSurface3DDepth              = 60, /**< Maximum 3D surface depth */
    cudaDevAttrMaxSurface1DLayeredWidth       = 61, /**< Maximum 1D layered surface width */
    cudaDevAttrMaxSurface1DLayeredLayers      = 62, /**< Maximum layers in a 1D layered surface */
    cudaDevAttrMaxSurface2DLayeredWidth       = 63, /**< Maximum 2D layered surface width */
    cudaDevAttrMaxSurface2DLayeredHeight      = 64, /**< Maximum 2D layered surface height */
    cudaDevAttrMaxSurface2DLayeredLayers      = 65, /**< Maximum layers in a 2D layered surface */
    cudaDevAttrMaxSurfaceCubemapWidth         = 66, /**< Maximum cubemap surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67, /**< Maximum cubemap layered surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68, /**< Maximum layers in a cubemap layered surface */
    cudaDevAttrMaxTexture1DLinearWidth        = 69, /**< Maximum 1D linear texture width */
    cudaDevAttrMaxTexture2DLinearWidth        = 70, /**< Maximum 2D linear texture width */
    cudaDevAttrMaxTexture2DLinearHeight       = 71, /**< Maximum 2D linear texture height */
    cudaDevAttrMaxTexture2DLinearPitch        = 72, /**< Maximum 2D linear texture pitch in bytes */
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73, /**< Maximum mipmapped 2D texture width */
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74, /**< Maximum mipmapped 2D texture height */
    cudaDevAttrComputeCapabilityMajor         = 75, /**< Major compute capability version number */ 
    cudaDevAttrComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77, /**< Maximum mipmapped 1D texture width */
    cudaDevAttrStreamPrioritiesSupported      = 78, /**< Device supports stream priorities */
    cudaDevAttrGlobalL1CacheSupported         = 79, /**< Device supports caching globals in L1 */
    cudaDevAttrLocalL1CacheSupported          = 80, /**< Device supports caching locals in L1 */
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81, /**< Maximum shared memory available per multiprocessor in bytes */
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82, /**< Maximum number of 32-bit registers available per multiprocessor */
    cudaDevAttrManagedMemory                  = 83, /**< Device can allocate managed memory on this system */
    cudaDevAttrIsMultiGpuBoard                = 84, /**< Device is on a multi-GPU board */
    cudaDevAttrMultiGpuBoardGroupID           = 85, /**< Unique identifier for a group of devices on the same multi-GPU board */
    cudaDevAttrHostNativeAtomicSupported      = 86, /**< Link between the device and the host supports native atomic operations */
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87, /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    cudaDevAttrPageableMemoryAccess           = 88, /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    cudaDevAttrConcurrentManagedAccess        = 89, /**< Device can coherently access managed memory concurrently with the CPU */
    cudaDevAttrComputePreemptionSupported     = 90, /**< Device supports Compute Preemption */
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
    cudaDevAttrReserved92                     = 92,
    cudaDevAttrReserved93                     = 93,
    cudaDevAttrReserved94                     = 94,
    cudaDevAttrCooperativeLaunch              = 95, /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel*/
    cudaDevAttrCooperativeMultiDeviceLaunch   = 96, /**< Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice */
    cudaDevAttrMaxSharedMemoryPerBlockOptin   = 97  /**< The maximum optin shared memory per block. This value may vary by chip. See ::cudaFuncSetAttribute */
};

/**
 * CUDA device P2P attributes
 */

enum __device_builtin__ cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank              = 1, /**< A relative value indicating the performance of the link between two devices */
    cudaDevP2PAttrAccessSupported              = 2, /**< Peer access is enabled */
    cudaDevP2PAttrNativeAtomicSupported        = 3  /**< Native atomic operation over the link supported */
};
/**
 * CUDA device properties
 */
struct __device_builtin__ cudaDeviceProp
{
    char   name[256];                  /**< ASCII string identifying device */
    size_t totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t sharedMemPerBlock;          /**< Shared memory available per block in bytes */
    int    regsPerBlock;               /**< 32-bit registers available per block */
    int    warpSize;                   /**< Warp size in threads */
    size_t memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int    maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int    maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int    maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    int    clockRate;                  /**< Clock frequency in kilohertz */
    size_t totalConstMem;              /**< Constant memory available on device in bytes */
    int    major;                      /**< Major compute capability */
    int    minor;                      /**< Minor compute capability */
    size_t textureAlignment;           /**< Alignment requirement for textures */
    size_t texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
    int    deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int    multiProcessorCount;        /**< Number of multiprocessors on device */
    int    kernelExecTimeoutEnabled;   /**< Specified whether there is a run time limit on kernels */
    int    integrated;                 /**< Device is integrated as opposed to discrete */
    int    canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int    computeMode;                /**< Compute mode (See ::cudaComputeMode) */
    int    maxTexture1D;               /**< Maximum 1D texture size */
    int    maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
    int    maxTexture1DLinear;         /**< Maximum size for 1D textures bound to linear memory */
    int    maxTexture2D[2];            /**< Maximum 2D texture dimensions */
    int    maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
    int    maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int    maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int    maxTexture3D[3];            /**< Maximum 3D texture dimensions */
    int    maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
    int    maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
    int    maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
    int    maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
    int    maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
    int    maxSurface1D;               /**< Maximum 1D surface size */
    int    maxSurface2D[2];            /**< Maximum 2D surface dimensions */
    int    maxSurface3D[3];            /**< Maximum 3D surface dimensions */
    int    maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
    int    maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
    int    maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
    int    maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
    size_t surfaceAlignment;           /**< Alignment requirements for surfaces */
    int    concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
    int    ECCEnabled;                 /**< Device has ECC support enabled */
    int    pciBusID;                   /**< PCI bus ID of the device */
    int    pciDeviceID;                /**< PCI device ID of the device */
    int    pciDomainID;                /**< PCI domain ID of the device */
    int    tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int    asyncEngineCount;           /**< Number of asynchronous engines */
    int    unifiedAddressing;          /**< Device shares a unified address space with the host */
    int    memoryClockRate;            /**< Peak memory clock frequency in kilohertz */
    int    memoryBusWidth;             /**< Global memory bus width in bits */
    int    l2CacheSize;                /**< Size of L2 cache in bytes */
    int    maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
    int    streamPrioritiesSupported;  /**< Device supports stream priorities */
    int    globalL1CacheSupported;     /**< Device supports caching globals in L1 */
    int    localL1CacheSupported;      /**< Device supports caching locals in L1 */
    size_t sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int    regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
    int    managedMemory;              /**< Device supports allocating managed memory on this system */
    int    isMultiGpuBoard;            /**< Device is on a multi-GPU board */
    int    multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
    int    hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
    int    singleToDoublePrecisionPerfRatio; /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    int    pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    int    concurrentManagedAccess;    /**< Device can coherently access managed memory concurrently with the CPU */
    int    computePreemptionSupported; /**< Device supports Compute Preemption */
    int    canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
    int    cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
    int    cooperativeMultiDeviceLaunch; /**< Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice */
    size_t sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
};

#define cudaDevicePropDontCare                             \
        {                                                  \
          {'\0'},    /* char   name[256];               */ \
          0,         /* size_t totalGlobalMem;          */ \
          0,         /* size_t sharedMemPerBlock;       */ \
          0,         /* int    regsPerBlock;            */ \
          0,         /* int    warpSize;                */ \
          0,         /* size_t memPitch;                */ \
          0,         /* int    maxThreadsPerBlock;      */ \
          {0, 0, 0}, /* int    maxThreadsDim[3];        */ \
          {0, 0, 0}, /* int    maxGridSize[3];          */ \
          0,         /* int    clockRate;               */ \
          0,         /* size_t totalConstMem;           */ \
          -1,        /* int    major;                   */ \
          -1,        /* int    minor;                   */ \
          0,         /* size_t textureAlignment;        */ \
          0,         /* size_t texturePitchAlignment    */ \
          -1,        /* int    deviceOverlap;           */ \
          0,         /* int    multiProcessorCount;     */ \
          0,         /* int    kernelExecTimeoutEnabled */ \
          0,         /* int    integrated               */ \
          0,         /* int    canMapHostMemory         */ \
          0,         /* int    computeMode              */ \
          0,         /* int    maxTexture1D             */ \
          0,         /* int    maxTexture1DMipmap       */ \
          0,         /* int    maxTexture1DLinear       */ \
          {0, 0},    /* int    maxTexture2D[2]          */ \
          {0, 0},    /* int    maxTexture2DMipmap[2]    */ \
          {0, 0, 0}, /* int    maxTexture2DLinear[3]    */ \
          {0, 0},    /* int    maxTexture2DGather[2]    */ \
          {0, 0, 0}, /* int    maxTexture3D[3]          */ \
          {0, 0, 0}, /* int    maxTexture3DAlt[3]       */ \
          0,         /* int    maxTextureCubemap        */ \
          {0, 0},    /* int    maxTexture1DLayered[2]   */ \
          {0, 0, 0}, /* int    maxTexture2DLayered[3]   */ \
          {0, 0},    /* int    maxTextureCubemapLayered[2] */ \
          0,         /* int    maxSurface1D             */ \
          {0, 0},    /* int    maxSurface2D[2]          */ \
          {0, 0, 0}, /* int    maxSurface3D[3]          */ \
          {0, 0},    /* int    maxSurface1DLayered[2]   */ \
          {0, 0, 0}, /* int    maxSurface2DLayered[3]   */ \
          0,         /* int    maxSurfaceCubemap        */ \
          {0, 0},    /* int    maxSurfaceCubemapLayered[2] */ \
          0,         /* size_t surfaceAlignment         */ \
          0,         /* int    concurrentKernels        */ \
          0,         /* int    ECCEnabled               */ \
          0,         /* int    pciBusID                 */ \
          0,         /* int    pciDeviceID              */ \
          0,         /* int    pciDomainID              */ \
          0,         /* int    tccDriver                */ \
          0,         /* int    asyncEngineCount         */ \
          0,         /* int    unifiedAddressing        */ \
          0,         /* int    memoryClockRate          */ \
          0,         /* int    memoryBusWidth           */ \
          0,         /* int    l2CacheSize              */ \
          0,         /* int    maxThreadsPerMultiProcessor */ \
          0,         /* int    streamPrioritiesSupported */ \
          0,         /* int    globalL1CacheSupported   */ \
          0,         /* int    localL1CacheSupported    */ \
          0,         /* size_t sharedMemPerMultiprocessor; */ \
          0,         /* int    regsPerMultiprocessor;   */ \
          0,         /* int    managedMemory            */ \
          0,         /* int    isMultiGpuBoard          */ \
          0,         /* int    multiGpuBoardGroupID     */ \
          0,         /* int    hostNativeAtomicSupported */ \
          0,         /* int    singleToDoublePrecisionPerfRatio */ \
          0,         /* int    pageableMemoryAccess     */ \
          0,         /* int    concurrentManagedAccess  */ \
          0,         /* int    computePreemptionSupported */ \
          0,         /* int    canUseHostPointerForRegisteredMem */ \
          0,         /* int    cooperativeLaunch */ \
          0,         /* int    cooperativeMultiDeviceLaunch */ \
          0,         /* size_t sharedMemPerBlockOptin */ \
        } /**< Empty device properties */

/**
 * CUDA IPC Handle Size
 */
#define CUDA_IPC_HANDLE_SIZE 64

/**
 * CUDA IPC event handle
 */
typedef __device_builtin__ struct __device_builtin__ cudaIpcEventHandle_st
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcEventHandle_t;

/**
 * CUDA IPC memory handle
 */
typedef __device_builtin__ struct __device_builtin__ cudaIpcMemHandle_st 
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcMemHandle_t;

/*******************************************************************************
*                                                                              *
*  SHORTHAND TYPE DEFINITION USED BY RUNTIME API                               *
*                                                                              *
*******************************************************************************/

/**
 * CUDA Error types
 */
typedef __device_builtin__ enum cudaError cudaError_t;

/**
 * CUDA stream
 */
typedef __device_builtin__ struct CUstream_st *cudaStream_t;

/**
 * CUDA event types
 */
typedef __device_builtin__ struct CUevent_st *cudaEvent_t;

/**
 * CUDA graphics resource types
 */
typedef __device_builtin__ struct cudaGraphicsResource *cudaGraphicsResource_t;

/**
 * CUDA UUID types
 */
typedef __device_builtin__ struct CUuuid_st cudaUUID_t;

/**
 * CUDA output file modes
 */
typedef __device_builtin__ enum cudaOutputMode cudaOutputMode_t;

/**
 * CUDA cooperative group scope
 */
enum __device_builtin__ cudaCGScope {
    cudaCGScopeInvalid   = 0, /**< Invalid cooperative group scope */
    cudaCGScopeGrid      = 1, /**< Scope represented by a grid_group */
    cudaCGScopeMultiGrid = 2  /**< Scope represented by a multi_grid_group */
};

/**
 * CUDA launch parameters
 */
struct __device_builtin__ cudaLaunchParams
{
    void *func;          /**< Device function symbol */
    dim3 gridDim;        /**< Grid dimentions */
    dim3 blockDim;       /**< Block dimentions */
    void **args;         /**< Arguments */
    size_t sharedMem;    /**< Shared memory */
    cudaStream_t stream; /**< Stream identifier */
};

/** @} */
/** @} */ /* END CUDART_TYPES */

#endif /* !__DRIVER_TYPES_H__ */
