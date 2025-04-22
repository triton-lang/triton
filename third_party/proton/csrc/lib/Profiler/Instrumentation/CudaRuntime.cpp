#include "Instrumentation/CudaRuntime.h"

namespace proton {

void CudaRuntime::allocateHostBuffer(uint8_t **buffer) {}

void CudaRuntime::freeHostBuffer(uint8_t *buffer) {}

void *CudaRuntime::getPriorityStream() {}

void CudaRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, const uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {}

} // namespace proton