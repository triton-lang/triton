#include "Profiler/Instrumentation/HipRuntime.h"
#include "Driver/GPU/HipApi.h"

namespace proton {

void HipRuntime::allocateHostBuffer(uint8_t **buffer, size_t size) {
  (void)hip::memAllocHost<true>(reinterpret_cast<void **>(buffer), size);
}

void HipRuntime::freeHostBuffer(uint8_t *buffer) {
  (void)hip::memFreeHost<true>(buffer);
}

uint64_t HipRuntime::getDevice() {
  hipDevice_t device;
  (void)hip::ctxGetDevice<true>(&device);
  return static_cast<uint64_t>(device);
}

void *HipRuntime::getPriorityStream() {
  hipStream_t stream;
  int lowestPriority, highestPriority;
  (void)hip::ctxGetStreamPriorityRange<true>(&lowestPriority, &highestPriority);
  (void)hip::streamCreateWithPriority<true>(&stream, hipStreamNonBlocking,
                                            highestPriority);
  return reinterpret_cast<void *>(stream);
}

void HipRuntime::synchronizeStream(void *stream) {
  (void)hip::streamSynchronize<true>(reinterpret_cast<hipStream_t>(stream));
}

void HipRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {
  int64_t chunkSize = std::min(hostBufferSize, deviceBufferSize);
  int64_t sizeLeftOnDevice = deviceBufferSize;
  while (chunkSize > 0) {
    (void)hip::memcpyDToHAsync<true>(
        reinterpret_cast<void *>(hostBuffer),
        reinterpret_cast<hipDeviceptr_t>(deviceBuffer), chunkSize,
        reinterpret_cast<hipStream_t>(stream));
    (void)hip::streamSynchronize<true>(reinterpret_cast<hipStream_t>(stream));
    callback(hostBuffer, chunkSize);
    sizeLeftOnDevice -= chunkSize;
    chunkSize =
        std::min(static_cast<int64_t>(hostBufferSize), sizeLeftOnDevice);
  }
}
} // namespace proton
