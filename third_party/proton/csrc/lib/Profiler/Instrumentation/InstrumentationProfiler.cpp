#include "Profiler/Instrumentation/InstrumentationProfiler.h"

#include "Driver/GPU/CudaApi.h"
#include "Profiler/Instrumentation/CudaRuntime.h"
#include "Profiler/Instrumentation/HipRuntime.h"
#include "Utility/String.h"
#include <memory>
#include <stdexcept>

namespace proton {

constexpr size_t HOST_BUFFER_SIZE = 64 * 1024 * 1024;

InstrumentationProfiler::~InstrumentationProfiler() {
  if (hostBuffer != nullptr) {
    runtime->freeHostBuffer(hostBuffer);
  }
}

void InstrumentationProfiler::doStart() {
  // Start the instrumentation profiler.
}

void InstrumentationProfiler::doFlush() {
  // Flush the instrumentation profiler.
}

void InstrumentationProfiler::doStop() {
  // Stop the instrumentation profiler.
}

InstrumentationProfiler *
InstrumentationProfiler::setMode(const std::vector<std::string> &mode) {
  if (mode.empty()) {
    throw std::runtime_error("Mode cannot be empty");
  }
  if (toLower(mode[0]) == toLower(DeviceTraits<DeviceType::CUDA>::name)) {
    runtime = std::make_unique<CudaRuntime>();
  } else if (toLower(mode[0]) == toLower(DeviceTraits<DeviceType::HIP>::name)) {
    runtime = std::make_unique<HipRuntime>();
  } else {
    throw std::runtime_error("Unknown device type: " + mode[0]);
  }
  return this;
}

void InstrumentationProfiler::initScopeIds(
    uint64_t functionId,
    const std::vector<std::pair<size_t, std::string>> &scopeIdPairs) {
  // Initialize the scope IDs.
  functionScopeIds[functionId] = scopeIdPairs;
}

void InstrumentationProfiler::enterInstrumentedOp(uint64_t functionId,
                                                  uint8_t *buffer,
                                                  size_t size) {
  // If the buffer is null, we cannot process it.
  if (!buffer)
    return;
  // Enter an instrumented operation.
  if (hostBuffer == nullptr) {
    runtime->allocateHostBuffer(&hostBuffer, HOST_BUFFER_SIZE);
  }
}

void InstrumentationProfiler::exitInstrumentedOp(uint64_t functionId,
                                                 uint8_t *buffer, size_t size) {
  // If the buffer is null, we cannot process it.
  if (!buffer)
    return;
  // Exit an instrumented operation.
  uint64_t device = runtime->getDevice();
  void *&stream = deviceStreams[reinterpret_cast<void *>(device)];
  if (!stream) {
    stream = runtime->getPriorityStream();
  }
  runtime->processHostBuffer(hostBuffer, HOST_BUFFER_SIZE, buffer, size,
                             deviceStreams[reinterpret_cast<void *>(device)],
                             [this](uint8_t *data, size_t size) {
                               // Process the data in the host buffer.
                               // This is where you would implement your
                               // callback logic.
                             });
}

} // namespace proton
