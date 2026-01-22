#include "Data/Metric.h"

#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace proton {

std::map<size_t, MetricBuffer::MetricDescriptor>
    MetricBuffer::metricDescriptors;
std::map<std::string, size_t> MetricBuffer::metricNameToId;
std::shared_mutex MetricBuffer::metricDescriptorMutex;

std::atomic<size_t> MetricBuffer::metricId{0};

MetricBuffer::~MetricBuffer() {
  for (auto &[device, buffer] : deviceBuffers) {
    runtime->freeHostBuffer(buffer.hostPtr);
    runtime->freeHostBuffer(reinterpret_cast<uint8_t *>(buffer.hostOffset));
    if (!mappedHostBuffer) {
      runtime->freeDeviceBuffer(buffer.devicePtr);
      runtime->freeDeviceBuffer(buffer.deviceOffsetPtr);
    }
    if (buffer.priorityStream) {
      runtime->destroyStream(buffer.priorityStream);
    }
  }
}

void MetricBuffer::receive(
    const std::map<std::string, MetricValueType> &scalarMetrics,
    const std::map<std::string, TensorMetric> &tensorMetrics,
    void *tensorMetricKernel, void *scalarMetricKernel, void *stream) {
  queueMetrics(tensorMetrics, tensorMetricKernel, stream);
  queueMetrics(scalarMetrics, scalarMetricKernel, stream);
}

MetricBuffer::MetricDescriptor
MetricBuffer::getOrCreateMetricDescriptor(const std::string &name,
                                          size_t typeIndex) {
  {
    std::shared_lock<std::shared_mutex> lock(metricDescriptorMutex);
    auto nameIt = metricNameToId.find(name);
    if (nameIt != metricNameToId.end()) {
      auto &descriptor = metricDescriptors.at(nameIt->second);
      if (descriptor.typeIndex != typeIndex) {
        throw std::runtime_error(
            "[PROTON] MetricBuffer: type mismatch for metric " + name +
            ": current=" + getTypeNameForIndex(descriptor.typeIndex) +
            ", new=" + getTypeNameForIndex(typeIndex));
      }
      return descriptor;
    }
  }

  std::unique_lock<std::shared_mutex> lock(metricDescriptorMutex);
  // Check again in case another thread inserted while we were upgrading the
  // lock
  auto nameIt = metricNameToId.find(name);
  if (nameIt != metricNameToId.end()) {
    auto &descriptor = metricDescriptors.at(nameIt->second);
    if (descriptor.typeIndex != typeIndex) {
      throw std::runtime_error(
          "[PROTON] MetricBuffer: type mismatch for metric " + name +
          ": current=" + getTypeNameForIndex(descriptor.typeIndex) +
          ", new=" + getTypeNameForIndex(typeIndex));
    }
    return descriptor;
  }

  auto newMetricId = metricId.fetch_add(1);
  MetricDescriptor descriptor{newMetricId, typeIndex, name};
  metricDescriptors.emplace(newMetricId, descriptor);
  metricNameToId.emplace(name, newMetricId);
  return descriptor;
}

std::map<std::string, MetricValueType>
collectTensorMetrics(Runtime *runtime,
                     const std::map<std::string, TensorMetric> &tensorMetrics,
                     void *stream) {
  std::map<std::string, MetricValueType> tensorMetricsHost;
  for (auto &[name, tensorMetric] : tensorMetrics) {
    uint64_t metricBits = 0;
    runtime->copyDeviceToHostAsync(&metricBits, tensorMetric.ptr,
                                   sizeof(uint64_t), stream);
    runtime->synchronizeStream(stream);
    if (tensorMetric.index == variant_index_v<double, MetricValueType>) {
      double value = 0.0;
      std::memcpy(&value, &metricBits, sizeof(value));
      tensorMetricsHost[name] = value;
    } else if (tensorMetric.index ==
               variant_index_v<int64_t, MetricValueType>) {
      int64_t value = 0;
      std::memcpy(&value, &metricBits, sizeof(value));
      tensorMetricsHost[name] = value;
    }
  }
  return tensorMetricsHost;
}

void MetricBuffer::queue(size_t metricId, TensorMetric tensorMetric,
                         void *kernel, void *stream) {
  auto &buffer = getOrCreateBuffer();
  uint64_t size = capacity / sizeof(uint64_t);
  void *globalScratchPtr = nullptr;
  void *profileScratchPtr = nullptr;
  void *kernelParams[] = {reinterpret_cast<void *>(&buffer.devicePtr),
                          reinterpret_cast<void *>(&buffer.deviceOffsetPtr),
                          reinterpret_cast<void *>(&size),
                          reinterpret_cast<void *>(&metricId),
                          reinterpret_cast<void *>(&tensorMetric.ptr),
                          reinterpret_cast<void *>(&globalScratchPtr),
                          reinterpret_cast<void *>(&profileScratchPtr)};
  runtime->launchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, stream, kernelParams,
                        nullptr);
}

void MetricBuffer::queue(size_t metricId, MetricValueType scalarMetric,
                         void *kernel, void *stream) {
  auto &buffer = getOrCreateBuffer();
  uint64_t size = capacity / sizeof(uint64_t);
  uint64_t metricBits = std::visit(
      [](auto &&value) -> uint64_t {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, std::string>) {
          throw std::runtime_error(
              "[PROTON] String metrics are not supported in MetricBuffer");
        } else {
          static_assert(sizeof(T) == sizeof(uint64_t),
                        "MetricValueType alternative must be 8 bytes");
          uint64_t bits = 0;
          std::memcpy(&bits, &value, sizeof(bits));
          return bits;
        }
      },
      scalarMetric);
  void *globalScratchPtr = nullptr;
  void *profileScratchPtr = nullptr;
  void *kernelParams[] = {reinterpret_cast<void *>(&buffer.devicePtr),
                          reinterpret_cast<void *>(&buffer.deviceOffsetPtr),
                          reinterpret_cast<void *>(&size),
                          reinterpret_cast<void *>(&metricId),
                          reinterpret_cast<void *>(&metricBits),
                          reinterpret_cast<void *>(&globalScratchPtr),
                          reinterpret_cast<void *>(&profileScratchPtr)};
  runtime->launchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, stream, kernelParams,
                        nullptr);
}

void MetricBuffer::synchronize(DeviceBuffer &buffer) {
  runtime->synchronizeDevice();
  if (mappedHostBuffer) {
    // Buffer lives in mapped host memory; avoid treating mapped pointers as
    // device allocations (e.g. cuMemcpyDtoH / cuMemset) which can error.
    return;
  }
  runtime->copyDeviceToHostAsync(buffer.hostPtr, buffer.devicePtr, capacity,
                                 buffer.priorityStream);
  runtime->copyDeviceToHostAsync(buffer.hostOffset, buffer.deviceOffsetPtr,
                                 sizeof(uint64_t), buffer.priorityStream);
  runtime->memset(buffer.deviceOffsetPtr, 0, sizeof(uint64_t),
                  buffer.priorityStream);
  runtime->synchronizeStream(buffer.priorityStream); // Ensure memset is done
}

MetricBuffer::DeviceBuffer &MetricBuffer::getOrCreateBuffer() {
  std::lock_guard<std::mutex> lock(bufferMutex);
  auto device = runtime->getDevice();
  if (deviceBuffers.find(device) == deviceBuffers.end()) {
    deviceBuffers[device] = DeviceBuffer{};
    auto &buffer = deviceBuffers.at(device);
    if (mappedHostBuffer) {
      runtime->allocateHostBuffer(&buffer.hostPtr, capacity, /*mapped=*/true);
      runtime->getHostDevicePointer(buffer.hostPtr, &buffer.devicePtr);
      runtime->allocateHostBuffer(
          reinterpret_cast<uint8_t **>(&buffer.hostOffset), sizeof(uint64_t),
          /*mapped=*/true);
      runtime->getHostDevicePointer(
          reinterpret_cast<uint8_t *>(buffer.hostOffset),
          &buffer.deviceOffsetPtr);
      *buffer.hostOffset = 0;
    } else {
      runtime->allocateDeviceBuffer(&buffer.devicePtr, capacity);
      runtime->allocateDeviceBuffer(&buffer.deviceOffsetPtr, sizeof(uint64_t));
      runtime->allocateHostBuffer(&buffer.hostPtr, capacity, /*mapped=*/false);
      runtime->allocateHostBuffer(
          reinterpret_cast<uint8_t **>(&buffer.hostOffset), sizeof(uint64_t),
          /*mapped=*/false);
      buffer.priorityStream = runtime->getPriorityStream();
      runtime->memset(buffer.deviceOffsetPtr, 0, sizeof(uint64_t),
                      buffer.priorityStream);
      runtime->synchronizeStream(buffer.priorityStream);
    }
  }
  return deviceBuffers.at(device);
}

} // namespace proton
