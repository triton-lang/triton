#ifndef PROTON_RUNTIME_RUNTIME_H_
#define PROTON_RUNTIME_RUNTIME_H_

#include <cstddef>
#include <cstdlib>

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

  /// Allocates pinned host memory for staging profiler payloads. When
  /// `mapped=true`, the returned host buffer must also be mappable into the
  /// device address space so kernels can access it directly.
  virtual void allocateHostBuffer(uint8_t **buffer, size_t size,
                                  bool mapped = false) = 0;

  /// Returns the device-visible pointer for a previously allocated mapped host
  /// buffer.
  virtual void getHostDevicePointer(uint8_t *hostPtr, uint8_t **devicePtr) = 0;

  virtual void freeHostBuffer(uint8_t *buffer) = 0;

  virtual void allocateDeviceBuffer(uint8_t **buffer, size_t size) = 0;

  virtual void freeDeviceBuffer(uint8_t *buffer) = 0;

  /// Enqueues an asynchronous device-to-host copy on `stream`.
  virtual void copyDeviceToHostAsync(void *dst, const void *src, size_t size,
                                     void *stream) = 0;

  virtual void *getDevice() = 0;

  /// Returns a non-blocking high-priority stream suitable for background copy
  /// or drain work that should not serialize the main compute stream.
  virtual void *getPriorityStream() = 0;

  virtual void destroyStream(void *stream) = 0;

  /// Creates an event used to signal compute-step completion or copy
  /// completion without synchronizing the host.
  virtual void *createEvent() = 0;

  virtual void destroyEvent(void *event) = 0;

  /// Records `event` on `stream` after all previously enqueued work in that
  /// stream completes.
  virtual void recordEvent(void *event, void *stream) = 0;

  /// Makes `stream` wait until `event` has completed, without blocking the
  /// host thread.
  virtual void waitEvent(void *stream, void *event) = 0;

  /// Returns whether `event` has completed, without blocking.
  virtual bool queryEvent(void *event) = 0;

  virtual void synchronizeStream(void *stream) = 0;

  virtual void synchronizeDevice() = 0;

  DeviceType getDeviceType() const { return deviceType; }

protected:
  DeviceType deviceType;
};

} // namespace proton

#endif // PROTON_RUNTIME_RUNTIME_H_
