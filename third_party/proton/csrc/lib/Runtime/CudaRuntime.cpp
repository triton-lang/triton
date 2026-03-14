#include "Runtime/CudaRuntime.h"

#include "Driver/GPU/CudaApi.h"
#include <cstdint>
#include <stdexcept>

namespace proton {

void CudaRuntime::launchKernel(void *kernel, unsigned int gridDimX,
                               unsigned int gridDimY, unsigned int gridDimZ,
                               unsigned int blockDimX, unsigned int blockDimY,
                               unsigned int blockDimZ,
                               unsigned int sharedMemBytes, void *stream,
                               void **kernelParams, void **extra) {
  cuda::launchKernel<true>(reinterpret_cast<CUfunction>(kernel), gridDimX,
                           gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                           sharedMemBytes, reinterpret_cast<CUstream>(stream),
                           kernelParams, extra);
}

void CudaRuntime::memset(void *devicePtr, uint32_t value, size_t size,
                         void *stream) {
  cuda::memsetD32Async<true>(reinterpret_cast<CUdeviceptr>(devicePtr), value,
                             size / sizeof(uint32_t),
                             reinterpret_cast<CUstream>(stream));
}

void CudaRuntime::allocateHostBuffer(uint8_t **buffer, size_t size,
                                     bool mapped) {
  if (mapped) {
    cuda::memHostAlloc<true>(reinterpret_cast<void **>(buffer), size,
                             CU_MEMHOSTALLOC_DEVICEMAP |
                                 CU_MEMHOSTALLOC_PORTABLE |
                                 CU_MEMHOSTALLOC_DEVICEMAP);
  } else {
    cuda::memAllocHost<true>(reinterpret_cast<void **>(buffer), size);
  }
}

void CudaRuntime::getHostDevicePointer(uint8_t *hostPtr, uint8_t **devicePtr) {
  CUdeviceptr devicePtrV;
  cuda::memHostGetDevicePointer<true>(&devicePtrV, hostPtr, 0);
  *devicePtr = reinterpret_cast<uint8_t *>(devicePtrV);
}

void CudaRuntime::freeHostBuffer(uint8_t *buffer) {
  cuda::memFreeHost<true>(buffer);
}

void CudaRuntime::allocateDeviceBuffer(uint8_t **buffer, size_t size) {
  CUdeviceptr devicePtr;
  cuda::memAlloc<true>(&devicePtr, size);
  *buffer = reinterpret_cast<uint8_t *>(devicePtr);
}

void CudaRuntime::freeDeviceBuffer(uint8_t *buffer) {
  CUdeviceptr devicePtr = reinterpret_cast<CUdeviceptr>(buffer);
  cuda::memFree<true>(devicePtr);
}

void CudaRuntime::copyDeviceToHostAsync(void *dst, const void *src, size_t size,
                                        void *stream) {
  cuda::memcpyDToHAsync<true>(dst, reinterpret_cast<CUdeviceptr>(src), size,
                              reinterpret_cast<CUstream>(stream));
}

void *CudaRuntime::getDevice() {
  CUdevice device;
  cuda::ctxGetDevice<true>(&device);
  return reinterpret_cast<void *>(static_cast<uintptr_t>(device));
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

void *CudaRuntime::createEvent() {
  CUevent event;
  cuda::eventCreate<true>(&event, CU_EVENT_DEFAULT);
  return reinterpret_cast<void *>(event);
}

void CudaRuntime::destroyEvent(void *event) {
  cuda::eventDestroy<true>(reinterpret_cast<CUevent>(event));
}

void CudaRuntime::recordEvent(void *event, void *stream) {
  cuda::eventRecord<true>(reinterpret_cast<CUevent>(event),
                          reinterpret_cast<CUstream>(stream));
}

void CudaRuntime::waitEvent(void *stream, void *event) {
  cuda::streamWaitEvent<true>(reinterpret_cast<CUstream>(stream),
                              reinterpret_cast<CUevent>(event), 0);
}

bool CudaRuntime::queryEvent(void *event) {
  auto status = cuda::eventQuery<false>(reinterpret_cast<CUevent>(event));
  if (status == CUDA_SUCCESS) {
    return true;
  }
  if (status == CUDA_ERROR_NOT_READY) {
    return false;
  }
  throw std::runtime_error("Failed to query CUDA event");
}

void CudaRuntime::synchronizeStream(void *stream) {
  cuda::streamSynchronize<true>(reinterpret_cast<CUstream>(stream));
}

void CudaRuntime::destroyStream(void *stream) {
  cuda::streamDestroy<true>(reinterpret_cast<CUstream>(stream));
}

void CudaRuntime::synchronizeDevice() {
  CUcontext cuContext = nullptr;
  cuda::ctxGetCurrent<false>(&cuContext);
  if (cuContext) {
    cuda::ctxSynchronize<true>();
  }
}

} // namespace proton
