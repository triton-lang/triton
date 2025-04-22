#include "Instrumentation/HipRuntime.h"

namespace proton {

void HipRuntime::allocateHostBuffer(uint8_t **buffer) {}

void HipRuntime::freeHostBuffer(uint8_t *buffer) {}

void *HipRuntime::getPriorityStream() {}

void HipRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, const uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {}
} // namespace proton