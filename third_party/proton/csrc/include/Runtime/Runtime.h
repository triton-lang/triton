#ifndef PROTON_RUNTIME_RUNTIME_H_
#define PROTON_RUNTIME_RUNTIME_H_

#include <cstddef>
#include <cstdlib>
#include <functional>

#include "Device.h"

namespace proton {

/// Abstract base class for different runtime implementations
class Runtime {
public:
  Runtime(DeviceType deviceType) : deviceType(deviceType) {}
  virtual ~Runtime() = default;

  virtual void launchKernel(void *kernel, unsigned int gridDimX,
                            unsigned int gridDimY, unsigned int gridDimZ,
                            unsigned int blockDimX, unsigned int blockDimY,
                            unsigned int blockDimZ, unsigned int sharedMemBytes,
                            void *stream, void **kernelParams,
                            void **extra) = 0;

  virtual void memset(void *devicePtr, uint32_t value, size_t size,
                      void *stream) = 0;

  virtual void allocateHostBuffer(uint8_t **buffer, size_t size,
                                  bool mapped = false) = 0;

  virtual void getHostDevicePointer(uint8_t *hostPtr, uint8_t **devicePtr) = 0;

  virtual void freeHostBuffer(uint8_t *buffer) = 0;

  virtual void allocateDeviceBuffer(uint8_t **buffer, size_t size) = 0;

  virtual void freeDeviceBuffer(uint8_t *buffer) = 0;

  virtual void copyDeviceToHostAsync(void *dst, const void *src, size_t size,
                                     void *stream) = 0;

  virtual void *getDevice() = 0;

  virtual void *getPriorityStream() = 0;

  virtual void destroyStream(void *stream) = 0;

  virtual void synchronizeStream(void *stream) = 0;

  virtual void synchronizeDevice() = 0;

  virtual void
  processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
                    uint8_t *deviceBuffer, size_t deviceBufferSize,
                    void *stream,
                    std::function<void(uint8_t *, size_t)> callback) = 0;

  DeviceType getDeviceType() const { return deviceType; }

protected:
  DeviceType deviceType;
};

} // namespace proton

#endif // PROTON_RUNTIME_RUNTIME_H_
