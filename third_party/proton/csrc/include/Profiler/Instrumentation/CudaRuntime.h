#ifndef PROTON_PROFILER_INSTRUMENTATION_CUDA_RUNTIME_H_
#define PROTON_PROFILER_INSTRUMENTATION_CUDA_RUNTIME_H_

#include "Runtime.h"

namespace proton {

class CudaRuntime : public Runtime {
public:
  CudaRuntime() : Runtime(DeviceType::CUDA) {}
  ~CudaRuntime() = default;

  void allocateHostBuffer(uint8_t **buffer, size_t size) override;
  void freeHostBuffer(uint8_t *buffer) override;
  uint64_t getDevice() override;
  void *getPriorityStream() override;
  void synchronizeStream(void *stream) override;
  void
  processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
                    uint8_t *deviceBuffer, size_t deviceBufferSize,
                    void *stream,
                    std::function<void(uint8_t *, size_t)> callback) override;
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_CUDA_RUNTIME_H
