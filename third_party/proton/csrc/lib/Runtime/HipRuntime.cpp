#include "Runtime/HipRuntime.h"

#include "Driver/GPU/HipApi.h"
#include <cstdint>
#include <stdexcept>

namespace proton {

void HipRuntime::launchKernel(void *kernel, unsigned int gridDimX,
                              unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY,
                              unsigned int blockDimZ,
                              unsigned int sharedMemBytes, void *stream,
                              void **kernelParams, void **extra) {
  auto status = hip::launchKernel<true>(
      reinterpret_cast<hipFunction_t>(kernel), gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ, sharedMemBytes,
      reinterpret_cast<hipStream_t>(stream), kernelParams, extra);
  (void)status;
}

void HipRuntime::memset(void *devicePtr, uint32_t value, size_t size,
                        void *stream) {
  auto status = hip::memsetD32Async<true>(
      reinterpret_cast<hipDeviceptr_t>(devicePtr), value,
      size / sizeof(uint32_t), reinterpret_cast<hipStream_t>(stream));
  (void)status;
}

void HipRuntime::allocateHostBuffer(uint8_t **buffer, size_t size,
                                    bool mapped) {
  if (mapped) {
    (void)hip::memHostAlloc<true>(reinterpret_cast<void **>(buffer), size,
                                  hipHostAllocMapped);
  } else {
    (void)hip::memAllocHost<true>(reinterpret_cast<void **>(buffer), size);
  }
}

void HipRuntime::getHostDevicePointer(uint8_t *hostPtr, uint8_t **devicePtr) {
  hipDeviceptr_t devicePtrV;
  (void)hip::memHostGetDevicePointer<true>(&devicePtrV, hostPtr, 0);
  *devicePtr = reinterpret_cast<uint8_t *>(devicePtrV);
}

void HipRuntime::freeHostBuffer(uint8_t *buffer) {
  (void)hip::memFreeHost<true>(buffer);
}

void HipRuntime::allocateDeviceBuffer(uint8_t **buffer, size_t size) {
  hipDeviceptr_t devicePtr;
  (void)hip::memAlloc<true>(reinterpret_cast<void **>(&devicePtr), size);
  *buffer = reinterpret_cast<uint8_t *>(devicePtr);
}

void HipRuntime::freeDeviceBuffer(uint8_t *buffer) {
  hipDeviceptr_t devicePtr = reinterpret_cast<hipDeviceptr_t>(buffer);
  (void)hip::memFree<true>(devicePtr);
}

void HipRuntime::copyDeviceToHostAsync(void *dst, const void *src, size_t size,
                                       void *stream) {
  (void)hip::memcpyDToHAsync<true>(
      dst, reinterpret_cast<hipDeviceptr_t>(const_cast<void *>(src)), size,
      reinterpret_cast<hipStream_t>(stream));
}

void *HipRuntime::getDevice() {
  hipDevice_t device;
  (void)hip::ctxGetDevice<true>(&device);
  return reinterpret_cast<void *>(static_cast<uintptr_t>(device));
}

void *HipRuntime::getPriorityStream() {
  hipStream_t stream;
  int lowestPriority, highestPriority;
  (void)hip::ctxGetStreamPriorityRange<true>(&lowestPriority, &highestPriority);
  (void)hip::streamCreateWithPriority<true>(&stream, hipStreamNonBlocking,
                                            highestPriority);
  return reinterpret_cast<void *>(stream);
}

void *HipRuntime::createEvent() {
  hipEvent_t event;
  (void)hip::eventCreate<true>(&event);
  return reinterpret_cast<void *>(event);
}

void HipRuntime::destroyEvent(void *event) {
  (void)hip::eventDestroy<true>(reinterpret_cast<hipEvent_t>(event));
}

void HipRuntime::recordEvent(void *event, void *stream) {
  (void)hip::eventRecord<true>(reinterpret_cast<hipEvent_t>(event),
                               reinterpret_cast<hipStream_t>(stream));
}

void HipRuntime::waitEvent(void *stream, void *event) {
  (void)hip::streamWaitEvent<true>(reinterpret_cast<hipStream_t>(stream),
                                   reinterpret_cast<hipEvent_t>(event), 0);
}

bool HipRuntime::queryEvent(void *event) {
  auto status = hip::eventQuery<false>(reinterpret_cast<hipEvent_t>(event));
  if (status == hipSuccess) {
    return true;
  }
  if (status == hipErrorNotReady) {
    return false;
  }
  throw std::runtime_error("Failed to query HIP event");
}

void HipRuntime::synchronizeStream(void *stream) {
  (void)hip::streamSynchronize<true>(reinterpret_cast<hipStream_t>(stream));
}

void HipRuntime::synchronizeDevice() { (void)hip::deviceSynchronize<true>(); }

void HipRuntime::destroyStream(void *stream) {
  (void)hip::streamDestroy<true>(reinterpret_cast<hipStream_t>(stream));
}

} // namespace proton
