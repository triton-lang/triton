#include "Profiler/Instrumentation/InstrumentationProfiler.h"

#include <memory>
#include "Driver/GPU/CudaApi.h"
namespace proton {

constexpr size_t HOST_BUFFER_SIZE = 64 * 1024 * 1024;
namespace {

void allocateHostBuffer(uint8_t **buffer, DeviceType deviceType) {
  if (deviceType == DeviceType::CUDA) {
    cuda::memAllocHost<true>(reinterpret_cast<void **>(buffer), HOST_BUFFER_SIZE);
  } else {
    throw std::runtime_error("Unsupported device type for host buffer allocation");
  }
}

void freeHostBuffer(uint8_t *buffer, DeviceType deviceType) {
  if (deviceType == DeviceType::CUDA) {
    cuda::memFreeHost<true>(buffer);
  } else {
    throw std::runtime_error("Unsupported device type for host buffer deallocation");
  }
}

void *getPriorityStream(std::map<void *, void *> &deviceStreams,
                        DeviceType deviceType) {
  if (deviceType == DeviceType::CUDA) {
    CUdevice device;
    cuda::cuCtxGetDevice<true>(&device);
    if (deviceStreams.find(reinterpret_cast<void *>(device)) !=
        deviceStreams.end()) {
      return deviceStreams[reinterpret_cast<void *>(device)];
    }
    CUstream stream;
    cuda::streamCreateWithPriority<true>(&stream, 0, 0);
    deviceStreams[reinterpret_cast<void *>(device)] =
        reinterpret_cast<void *>(stream);
    return reinterpret_cast<void *>(stream);
  } else {
    throw std::runtime_error("Unsupported device type for stream creation");
  }
}

void processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
                      const uint8_t *deviceBuffer, size_t deviceBufferSize,
                      DeviceType deviceType, void *stream,
                      std::function<void(uint8_t *, size_t)> callback) {
  int64_t chunkSize = std::min(hostBufferSize, deviceBufferSize);
  int64_t sizeLeftOnDevice = deviceBufferSize;
  while (chunkSize > 0) {
    if (deviceType == DeviceType::CUDA) {
      cuda::memcpyAsync<true>(reinterpret_cast<void *>(hostBuffer),
                              reinterpret_cast<const void *>(deviceBuffer),
                              chunkSize, reinterpret_cast<CUstream>(stream));
      cuda::streamSynchronize<true>(reinterpret_cast<CUstream>(stream));
    } else {
      throw std::runtime_error("Unsupported device type for memory copy");
    }
    callback(hostBuffer, chunkSize);
    sizeLeftOnDevice -= chunkSize;
    chunkSize = std::min(static_cast<int64_t>(hostBufferSize), sizeLeftOnDevice);
  }
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
  if (mode[0] == DeviceTraits<DeviceType::CUDA>::name) {
    deviceType = DeviceType::CUDA;
  } else if (mode[0] == DeviceTraits<DeviceType::HIP>::name) {
    deviceType = DeviceType::HIP;
  } else {
    throw std::runtime_error("Unknown device type: " + mode[0]);
  }
}

void InstrumentationProfiler::initScopeIds(
    uint64_t functionId,
    const std::vector<std::pair<size_t, std::string>> &scopeIdPairs) {
  // Initialize the scope IDs.
  functionScopeIds[functionId] = scopeIdPairs;
}

void InstrumentationProfiler::enterInstrumentedOp(uint64_t functionId,
                                                  const uint8_t *buffer,
                                                  size_t size) {
  // Enter an instrumented operation.
  if (hostBuffer == nullptr) {
    allocateHostBuffer(&hostBuffer, deviceType);
  }
}

void InstrumentationProfiler::exitInstrumentedOp(uint64_t functionId,
                                                 const uint8_t *buffer,
                                                 size_t size) {
  // Exit an instrumented operation.
  auto stream = getPriorityStream(deviceStreams, deviceType);
  if (stream == nullptr) {
    throw std::runtime_error("Failed to get priority stream");
  }
  processHostBuffer(hostBuffer, HOST_BUFFER_SIZE, buffer, size, deviceType,
                    stream, [this](uint8_t *data, size_t size) {
                      // Process the data in the host buffer.
                      // This is where you would implement your callback logic.
                    });
}

} // namespace proton
