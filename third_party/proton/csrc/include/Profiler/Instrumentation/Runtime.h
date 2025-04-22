#ifndef PROTON_PROFILER_INSTRUMENTATION_RUNTIME_H
#define PROTON_PROFILER_INSTRUMENTATION_RUNTIME_H

#include <cstddef>
#include <cstdlib>
#include <functional>

#include "Driver/Device.h"

namespace proton {
class Runtime {
public:
  Runtime(DeviceType deviceType) : deviceType(deviceType) {}
  virtual ~Runtime() = default;

  virtual void allocateHostBuffer(uint8_t **buffer, size_t size) = 0;

  virtual void freeHostBuffer(uint8_t *buffer) = 0;

  virtual uint64_t getDevice() = 0;

  virtual void *getPriorityStream() = 0;

  virtual void
  processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
                    uint8_t *deviceBuffer, size_t deviceBufferSize,
                    void *stream,
                    std::function<void(uint8_t *, size_t)> callback) = 0;

protected:
  DeviceType deviceType;
};
} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_RUNTIME_H
