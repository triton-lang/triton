#include "Profiler/Graph.h"

#include "Data/Data.h"
#include "Runtime/Runtime.h"

#include <cstring>
#include <stdexcept>

namespace proton {

namespace {
constexpr size_t kMetricWordsPerNode = 2;

constexpr size_t bytesForNodes(size_t numNodes) {
  return numNodes * kMetricWordsPerNode * sizeof(uint64_t);
}

void emitMetricRecords(MetricBuffer &metricBuffer, uint64_t *hostBasePtr,
                       const PendingGraphQueue &queue) {
  const size_t phase = queue.phase;
  const auto &pendingGraphs = queue.pendingGraphs;
  const size_t capacityWords = metricBuffer.getCapacity() / sizeof(uint64_t);

  const size_t startWordOffset = queue.startBufferOffset / sizeof(uint64_t);
  size_t wordOffset = startWordOffset;
  auto readWord = [&](size_t offset) -> uint64_t {
    return hostBasePtr[offset % capacityWords];
  };

  for (const auto &pendingGraph : pendingGraphs) {
    const size_t graphMetricNodes = pendingGraph.numNodes;
    for (size_t i = 0; i < graphMetricNodes; ++i) {
      const uint64_t metricId = readWord(wordOffset);
      const uint64_t metricValue = readWord(wordOffset + 1);
      wordOffset = (wordOffset + kMetricWordsPerNode) % capacityWords;

      auto metricDesc = metricBuffer.getMetricDescriptor(metricId);
      const auto &metricName = metricDesc.name;
      const auto metricTypeIndex = metricDesc.typeIndex;

      for (auto &[data, entryIds] : pendingGraph.dataToEntryIds) {
        // XXX(Keren): It assumes all data have the same number of entries,
        // which may not be true in very hacky cases
        const auto entryId = entryIds[i];
        switch (metricTypeIndex) {
        case variant_index_v<uint64_t, MetricValueType>: {
          uint64_t typedValue{};
          std::memcpy(&typedValue, &metricValue, sizeof(typedValue));
          data->addEntryMetrics(phase, entryId,
                                {{metricName, MetricValueType{typedValue}}});
          break;
        }
        case variant_index_v<int64_t, MetricValueType>: {
          int64_t typedValue{};
          std::memcpy(&typedValue, &metricValue, sizeof(typedValue));
          data->addEntryMetrics(phase, entryId,
                                {{metricName, MetricValueType{typedValue}}});
          break;
        }
        case variant_index_v<double, MetricValueType>: {
          double typedValue{};
          std::memcpy(&typedValue, &metricValue, sizeof(typedValue));
          data->addEntryMetrics(phase, entryId,
                                {{metricName, MetricValueType{typedValue}}});
          break;
        }
        default:
          break;
        }
      }
    }
  }
}
} // namespace

/// -------------------------------- PendingGraphPool
/// -------------------------------

void PendingGraphPool::push(
    size_t phase, const std::map<Data *, std::vector<size_t>> &dataToEntryIds,
    size_t numNodes) {
  std::lock_guard<std::mutex> lock(mutex);
  auto *device = runtime->getDevice();
  if (deviceRemainingCapacity.find(device) == deviceRemainingCapacity.end()) {
    deviceRemainingCapacity[device] = metricBuffer->getCapacity();
  }
  auto [it, inserted] = pool.try_emplace(
      phase, PendingGraphQueue(deviceBufferOffset[device], phase, device));
  (void)inserted;
  it->second.push(numNodes, dataToEntryIds);
  deviceBufferOffset[device] =
      (deviceBufferOffset[device] + bytesForNodes(numNodes)) %
      metricBuffer->getCapacity();
  deviceRemainingCapacity[device] -= bytesForNodes(numNodes);
}

void PendingGraphPool::peek(size_t phase) {
  std::lock_guard<std::mutex> lock(mutex);
  auto queueIt = pool.find(phase);
  if (queueIt == pool.end()) {
    return;
  }
  auto &queue = queueIt->second;
  auto *device = queue.device;
  metricBuffer->peek(static_cast<Device *>(device), [&](uint8_t *hostPtr,
                                                        uint64_t *hostOffset) {
    (void)hostOffset;
    emitMetricRecords(*metricBuffer, reinterpret_cast<uint64_t *>(hostPtr),
                      queue);
  });
  pool.erase(queueIt);
  deviceRemainingCapacity[device] += bytesForNodes(queue.numNodes);
}

bool PendingGraphPool::flushIfNeeded(size_t numNodes) {
  auto *device = runtime->getDevice();
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (deviceRemainingCapacity.find(device) == deviceRemainingCapacity.end()) {
      deviceRemainingCapacity[device] = metricBuffer->getCapacity();
    }
    size_t requiredBytes = bytesForNodes(numNodes);
    size_t &remainingCapacity = deviceRemainingCapacity[device];
    if (remainingCapacity >= requiredBytes) {
      return false;
    }
  }
  flushAll();
  return true;
}

bool PendingGraphPool::flushAll() {
  std::lock_guard<std::mutex> lock(mutex);
  if (pool.empty()) {
    return false;
  }
  metricBuffer->flush(
      [&](uint8_t *hostPtr, uint64_t *hostOffset) {
        (void)hostOffset;
        for (auto &[phase, queue] : pool) {
          (void)phase;
          emitMetricRecords(*metricBuffer,
                            reinterpret_cast<uint64_t *>(hostPtr), queue);
        }
      },
      true);
  for (auto &[device, capacity] : deviceRemainingCapacity) {
    capacity = metricBuffer->getCapacity();
  }
  pool.clear();
  return true;
}

} // namespace proton
