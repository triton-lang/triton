#ifndef PROTON_RUNTIME_HIP_RUNTIME_H_
#define PROTON_RUNTIME_HIP_RUNTIME_H_

#include "Runtime.h"
#include "Utility/Singleton.h"

namespace proton {

class HipRuntime : public Singleton<HipRuntime>, public Runtime {
public:
  HipRuntime() : Runtime(DeviceType::HIP) {}
  ~HipRuntime() = default;

  void launchKernel(void *kernel, unsigned int gridDimX, unsigned int gridDimY,
                    unsigned int gridDimZ, unsigned int blockDimX,
                    unsigned int blockDimY, unsigned int blockDimZ,
                    unsigned int sharedMemBytes, void *stream,
                    void **kernelParams, void **extra) override;
  void memset(void *devicePtr, uint32_t value, size_t size,
              void *stream) override;
  void allocateHostBuffer(uint8_t **buffer, size_t size, bool mapped) override;
  void getHostDevicePointer(uint8_t *hostPtr, uint8_t **devicePtr) override;
  void freeHostBuffer(uint8_t *buffer) override;
  void allocateDeviceBuffer(uint8_t **buffer, size_t size) override;
  void freeDeviceBuffer(uint8_t *buffer) override;
  void copyDeviceToHostAsync(void *dst, const void *src, size_t size,
                             void *stream) override;
  void *getDevice() override;
  void *getPriorityStream() override;
  void synchronizeStream(void *stream) override;
  void synchronizeDevice() override;
  void destroyStream(void *stream) override;
  void
  processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
                    uint8_t *deviceBuffer, size_t deviceBufferSize,
                    void *stream,
                    std::function<void(uint8_t *, size_t)> callback) override;
};

} // namespace proton

#endif // PROTON_RUNTIME_HIP_RUNTIME_H_
