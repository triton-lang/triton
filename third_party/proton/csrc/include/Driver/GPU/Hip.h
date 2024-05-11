#ifndef PROTON_DRIVER_GPU_HIP_H_
#define PROTON_DRIVER_GPU_HIP_H_

#include <hip/hip_runtime_api.h>

namespace proton {

namespace hip {

template <bool CheckSuccess> hipError_t deviceSynchronize();

} // namespace hip

} // namespace proton

#endif // PROTON_DRIVER_GPU_HIP_H_
