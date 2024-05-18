#ifndef PROTON_DRIVER_GPU_HIP_H_
#define PROTON_DRIVER_GPU_HIP_H_

#include "Driver/Device.h"
// clang-format off
// Order matters here--hip_deprecated.h depends on hip_runtime_api.h.
#include <hip/hip_runtime_api.h>
#include <hip/hip_deprecated.h>
// clang-format off

namespace proton {

namespace hip {

template <bool CheckSuccess> hipError_t deviceSynchronize();

template <bool CheckSuccess>
hipError_t getDeviceProperties(hipDeviceProp_tR0000 *properties, int ordinal);

Device getDevice(uint64_t index);

} // namespace hip

} // namespace proton

#endif // PROTON_DRIVER_GPU_HIP_H_
