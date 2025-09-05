#ifndef PROTON_DRIVER_GPU_NVTX_API_H
#define PROTON_DRIVER_GPU_NVTX_API_H

#include <string>

namespace proton {

namespace nvtx {

void enable();

void disable();

std::string getMessageFromRangePushA(const void *params);

std::string getMessageFromRangePushW(const void *params);

} // namespace nvtx

} // namespace proton

#endif // PROTON_DRIVER_GPU_NVTX_API_H
