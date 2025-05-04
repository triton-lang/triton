#include "Profiler/Instrumentation/HipRuntime.h"

namespace proton {

void HipRuntime::allocateHostBuffer(uint8_t **buffer, size_t size) {}

void HipRuntime::freeHostBuffer(uint8_t *buffer) {}

uint64_t HipRuntime::getDevice() { return 0; }

void *HipRuntime::getPriorityStream() { return nullptr; }

void HipRuntime::synchronizeStream(void *stream) {}

void HipRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {}
} // namespace proton
