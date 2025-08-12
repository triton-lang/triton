#ifndef PROTON_DRIVER_GPU_HIP_H_
#define PROTON_DRIVER_GPU_HIP_H_

#include "Device.h"
#include "hip/hip_runtime_api.h"

namespace proton {

namespace hip {

template <bool CheckSuccess> hipError_t ctxGetDevice(hipDevice_t *device);

template <bool CheckSuccess> hipError_t deviceSynchronize();

template <bool CheckSuccess>
hipError_t deviceGetAttribute(int *value, hipDeviceAttribute_t attribute,
                              int deviceId);

template <bool CheckSuccess> hipError_t getDeviceCount(int *count);

template <bool CheckSuccess>
hipError_t getDeviceProperties(hipDeviceProp_t *prop, int deviceId);

Device getDevice(uint64_t index);

template <bool CheckSuccess>
hipError_t ctxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

template <bool CheckSuccess>
hipError_t streamCreateWithPriority(hipStream_t *pStream, unsigned int flags,
                                    int priority);

template <bool CheckSuccess> hipError_t streamSynchronize(hipStream_t stream);

template <bool CheckSuccess>
hipError_t memcpyDToHAsync(void *dst, hipDeviceptr_t src, size_t count,
                           hipStream_t stream);

const std::string getHipArchName(uint64_t index);

const char *getKernelNameRef(const hipFunction_t f);
const char *getKernelNameRefByPtr(const void *hostFunction, hipStream_t stream);

template <bool CheckSuccess>
hipError_t memAllocHost(void **pp, size_t bytesize);
template <bool CheckSuccess> hipError_t memFreeHost(void *p);

} // namespace hip

} // namespace proton

#endif // PROTON_DRIVER_GPU_HIP_H_
