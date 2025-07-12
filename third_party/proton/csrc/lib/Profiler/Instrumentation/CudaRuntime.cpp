#include "Profiler/Instrumentation/CudaRuntime.h"

#include "Driver/GPU/CudaApi.h"
#include <stdexcept>
namespace proton {

void CudaRuntime::allocateHostBuffer(uint8_t **buffer, size_t size) {
  cuda::memAllocHost<true>(reinterpret_cast<void **>(buffer), size);
}

void CudaRuntime::freeHostBuffer(uint8_t *buffer) {
  cuda::memFreeHost<true>(buffer);
}

uint64_t CudaRuntime::getDevice() {
  CUdevice device;
  cuda::ctxGetDevice<true>(&device);
  return static_cast<uint64_t>(device);
}

void *CudaRuntime::getPriorityStream() {
  CUstream stream;
  // TODO: Change priority
  int lowestPriority, highestPriority;
  cuda::ctxGetStreamPriorityRange<true>(&lowestPriority, &highestPriority);
  cuda::streamCreateWithPriority<true>(&stream, CU_STREAM_NON_BLOCKING,
                                       highestPriority);
  return reinterpret_cast<void *>(stream);
}

void CudaRuntime::synchronizeStream(void *stream) {
  cuda::streamSynchronize<true>(reinterpret_cast<CUstream>(stream));
}

void CudaRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {
  int64_t chunkSize = std::min(hostBufferSize, deviceBufferSize);
  int64_t sizeLeftOnDevice = deviceBufferSize;
  while (chunkSize > 0) {
    cuda::memcpyDToHAsync<true>(reinterpret_cast<void *>(hostBuffer),
                                reinterpret_cast<CUdeviceptr>(deviceBuffer),
                                chunkSize, reinterpret_cast<CUstream>(stream));
    // We should not use synchronization here in general if we want to copy
    // buffer while the kernel is running. But for the sake of simplicity, we
    // only copy the buffer after the kernel is finished for now.
    cuda::streamSynchronize<true>(reinterpret_cast<CUstream>(stream));
    callback(hostBuffer, chunkSize);
    sizeLeftOnDevice -= chunkSize;
    chunkSize =
        std::min(static_cast<int64_t>(hostBufferSize), sizeLeftOnDevice);
  }
}

} // namespace proton
