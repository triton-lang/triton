/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_HIP_HIP_EXT_H
#define HIP_INCLUDE_HIP_HIP_EXT_H
#include "hip/hip_runtime.h"
#if defined(__cplusplus)
#include <tuple>
#include <type_traits>
#endif
/** @addtogroup Module Module Management
 *  @{
 */

/**
 * @brief Launches kernel with parameters and shared memory on stream with arguments passed
 * to kernel params or extra arguments.
 *
 * @param [in] f Kernel to launch.
 * @param [in] globalWorkSizeX  X grid dimension specified in work-items.
 * @param [in] globalWorkSizeY  Y grid dimension specified in work-items.
 * @param [in] globalWorkSizeZ  Z grid dimension specified in work-items.
 * @param [in] localWorkSizeX  X block dimension specified in work-items.
 * @param [in] localWorkSizeY  Y block dimension specified in work-items.
 * @param [in] localWorkSizeZ  Z block dimension specified in work-items.
 * @param [in] sharedMemBytes  Amount of dynamic shared memory to allocate for this kernel.
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] hStream  Stream where the kernel should be dispatched.
 * May be 0, in which case the default stream is used with associated synchronization rules.
 * @param [in] kernelParams  pointer to kernel parameters.
 * @param [in] extra  Pointer to kernel arguments. These are passed directly to the kernel and
 * must be in the memory layout and alignment expected by the kernel.
 * All passed arguments must be naturally aligned according to their type. The memory address of each
 * argument should be a multiple of its size in bytes. Please refer to hip_porting_driver_api.md
 * for sample usage.
 * @param [in] startEvent  If non-null, specified event will be updated to track the start time of
 * the kernel launch. The event must be created before calling this API.
 * @param [in] stopEvent  If non-null, specified event will be updated to track the stop time of
 * the kernel launch. The event must be created before calling this API.
 * @param [in] flags  The value of hipExtAnyOrderLaunch, signifies if kernel can be
 * launched in any order.
 * @returns #hipSuccess, #hipInvalidDeviceId, #hipErrorNotInitialized, #hipErrorInvalidValue.
 *
 * HIP/ROCm actually updates the start event when the associated kernel completes.
 * Currently, timing between startEvent and stopEvent does not include the time it takes to perform
 * a system scope release/cache flush - only the time it takes to issues writes to cache.
 *
 */
HIP_PUBLIC_API
 extern "C" hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent __dparm(NULL),
                                    hipEvent_t stopEvent __dparm(NULL),
                                    uint32_t flags __dparm(0));
/**
 * @brief This HIP API is deprecated, please use hipExtModuleLaunchKernel() instead.
 *
 */
HIP_PUBLIC_API
extern "C" hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent __dparm(NULL),
                                    hipEvent_t stopEvent __dparm(NULL))
                                    __attribute__((deprecated("use hipExtModuleLaunchKernel instead")));

#if defined(__cplusplus)

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
 * @returns #hipSuccess, #hipInvalidDeviceId, #hipErrorNotInitialized, #hipErrorInvalidValue.
 *
 */
extern "C" hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks,
                                         dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                         hipStream_t stream, hipEvent_t startEvent,
                                         hipEvent_t stopEvent, int flags);

/**
 * @brief Launches kernel with dimention parameters and shared memory on stream with templated kernel and arguments.
 *
 * @param [in] kernel  Kernel to launch.
 * @param [in] numBlocks  const number of blocks.
 * @param [in] dimBlocks  const dimension of a block.
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
 * @param [in] args  templated kernel arguments.
 *
 */
template <typename... Args, typename F = void (*)(Args...)>
inline void hipExtLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                                  std::uint32_t sharedMemBytes, hipStream_t stream,
                                  hipEvent_t startEvent, hipEvent_t stopEvent, std::uint32_t flags,
                                  Args... args) {
    constexpr size_t count = sizeof...(Args);
    auto tup_ = std::tuple<Args...>{args...};
    auto tup = validateArgsCountType(kernel, tup_);
    void* _Args[count];
    pArgs<0>(tup, _Args);

    auto k = reinterpret_cast<void*>(kernel);
    hipExtLaunchKernel(k, numBlocks, dimBlocks, _Args, sharedMemBytes, stream, startEvent,
                       stopEvent, (int)flags);
}

#endif // defined(__cplusplus)

// doxygen end AMD-specific features
/**
 * @}
 */
#endif  // #iidef HIP_INCLUDE_HIP_HIP_EXT_H
