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
  size_t wordOffset = queue.startBufferOffset / sizeof(uint64_t);
  auto readWord = [&](size_t offset) -> uint64_t {
    return hostBasePtr[offset % capacityWords];
  };

  for (const auto &pendingGraph : pendingGraphs) {
    for (size_t i = 0; i < pendingGraph.numNodes; ++i) {
      const uint64_t metricId = readWord(wordOffset);
      const uint64_t metricValue = readWord(wordOffset + 1);
      wordOffset = (wordOffset + kMetricWordsPerNode) % capacityWords;

      auto metricDesc = metricBuffer.getMetricDescriptor(metricId);
      const auto &metricName = metricDesc.name;
      const auto metricTypeIndex = metricDesc.typeIndex;

      for (auto &[data, entryIds] : pendingGraph.dataToEntryIds) {
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

void PendingGraphPool::push(
    size_t phase, const std::map<Data *, std::vector<size_t>> &dataToEntryIds,
    size_t numNodes) {
  const size_t requiredBytes = bytesForNodes(numNodes);
  std::shared_ptr<Slot> slot;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto [poolIt, inserted] = pool.try_emplace(phase);
    if (inserted)
      poolIt->second = std::make_shared<Slot>();
    slot = poolIt->second;
  }
  void *device = nullptr;
  {
    std::lock_guard<std::mutex> slotLock(slot->mutex);

    if (slot->queue.has_value()) {
      device = slot->queue->device;
    } else {
      device = runtime->getDevice();
      const auto startBufferOffset =
          deviceBufferOffset.try_emplace(device, 0).first->second;
      slot->queue = PendingGraphQueue(startBufferOffset, phase, device);
    }
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto &remainingCapacity =
        deviceRemainingCapacity.try_emplace(device, metricBuffer->getCapacity())
            .first->second;
    slot->queue->push(numNodes, dataToEntryIds);
    auto &bufferOffset = deviceBufferOffset[device];
    bufferOffset = (bufferOffset + requiredBytes) % metricBuffer->getCapacity();
    remainingCapacity -= requiredBytes;
  }
}

void PendingGraphPool::peek(size_t phase) {
  std::shared_ptr<Slot> slotPtr;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto slotIt = pool.find(phase);
    if (slotIt == pool.end())
      return;
    slotPtr = slotIt->second;
  }
  void *device = nullptr;
  auto numNodes = size_t{0};
  {
    std::lock_guard<std::mutex> slotLock(slotPtr->mutex);
    if (slotPtr->queue == std::nullopt)
      return;
    auto &queue = slotPtr->queue.value();
    device = queue.device;
    numNodes = queue.numNodes;
    metricBuffer->peek(static_cast<Device *>(device), [&](uint8_t *hostPtr) {
      emitMetricRecords(*metricBuffer, reinterpret_cast<uint64_t *>(hostPtr),
                        queue);
    });
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    pool.erase(phase);
    deviceRemainingCapacity[device] += bytesForNodes(numNodes);
  }
}

bool PendingGraphPool::flushIfNeeded(size_t numNodes) {
  auto *device = runtime->getDevice();
  const size_t requiredBytes = bytesForNodes(numNodes);
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it =
        deviceRemainingCapacity.try_emplace(device, metricBuffer->getCapacity())
            .first;
    if (it->second >= requiredBytes)
      return false;
  }
  flushAll();
  return true;
}

bool PendingGraphPool::flushAll() {
  auto poolCopy = decltype(pool){};
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (pool.empty())
      return false;
    poolCopy.swap(pool);
  }
  metricBuffer->flush(
      [&](uint8_t *hostPtr) {
        for (auto &[phase, slot] : poolCopy) {
          std::lock_guard<std::mutex> lock(slot->mutex);
          if (slot->queue == std::nullopt)
            continue;
          emitMetricRecords(*metricBuffer,
                            reinterpret_cast<uint64_t *>(hostPtr),
                            *slot->queue);
        }
      },
      true);
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &[phase, slot] : poolCopy) {
      std::lock_guard<std::mutex> slotLock(slot->mutex);
      if (slot->queue == std::nullopt)
        continue;
      auto *device = slot->queue->device;
      deviceRemainingCapacity[device] += bytesForNodes(slot->queue->numNodes);
    }
  }
  return true;
}

} // namespace proton
