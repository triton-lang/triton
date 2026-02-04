#include "Profiler/Graph.h"

#include "Data/Data.h"
#include "Runtime/Runtime.h"

#include <cstring>
#include <stdexcept>

namespace proton {

namespace {
constexpr size_t bytesForWords(size_t numWords) {
  return numWords * sizeof(uint64_t);
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
      wordOffset = (wordOffset + 1) % capacityWords;

      auto metricDesc = metricBuffer.getMetricDescriptor(metricId);
      const auto &metricName = metricDesc.name;
      const auto metricTypeIndex = metricDesc.typeIndex;

      MetricValueType metricValueVariant{};
      switch (metricTypeIndex) {
      case variant_index_v<uint64_t, MetricValueType>: {
        const uint64_t bits = readWord(wordOffset);
        uint64_t typedValue{};
        std::memcpy(&typedValue, &bits, sizeof(typedValue));
        metricValueVariant = typedValue;
        break;
      }
      case variant_index_v<int64_t, MetricValueType>: {
        const uint64_t bits = readWord(wordOffset);
        int64_t typedValue{};
        std::memcpy(&typedValue, &bits, sizeof(typedValue));
        metricValueVariant = typedValue;
        break;
      }
      case variant_index_v<double, MetricValueType>: {
        const uint64_t bits = readWord(wordOffset);
        double typedValue{};
        std::memcpy(&typedValue, &bits, sizeof(typedValue));
        metricValueVariant = typedValue;
        break;
      }
      case variant_index_v<std::vector<uint64_t>, MetricValueType>: {
        std::vector<uint64_t> values(metricDesc.size);
        for (size_t j = 0; j < metricDesc.size; ++j) {
          values[j] = readWord(wordOffset + j);
        }
        metricValueVariant = std::move(values);
        break;
      }
      case variant_index_v<std::vector<int64_t>, MetricValueType>: {
        std::vector<int64_t> values(metricDesc.size);
        for (size_t j = 0; j < metricDesc.size; ++j) {
          const uint64_t bits = readWord(wordOffset + j);
          std::memcpy(&values[j], &bits, sizeof(bits));
        }
        metricValueVariant = std::move(values);
        break;
      }
      case variant_index_v<std::vector<double>, MetricValueType>: {
        std::vector<double> values(metricDesc.size);
        for (size_t j = 0; j < metricDesc.size; ++j) {
          const uint64_t bits = readWord(wordOffset + j);
          std::memcpy(&values[j], &bits, sizeof(bits));
        }
        metricValueVariant = std::move(values);
        break;
      }
      default:
        throw std::runtime_error("[PROTON] Unsupported metric type index: " +
                                 std::to_string(metricTypeIndex));
        break;
      }

      wordOffset = (wordOffset + metricDesc.size) % capacityWords;

      for (auto &[data, entryIds] : pendingGraph.dataToEntryIds) {
        const auto entryId = entryIds[i];
        data->addMetrics(phase, entryId, {{metricName, metricValueVariant}});
      }
    }
  }
}
} // namespace

void PendingGraphPool::push(
    size_t phase, const std::map<Data *, std::vector<size_t>> &dataToEntryIds,
    size_t numNodes, size_t numWords) {
  const size_t requiredBytes = bytesForWords(numWords);
  void *device = runtime->getDevice();
  std::shared_ptr<Slot> slot;
  auto startBufferOffset = 0;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto &devicePool = pool[device];
    auto [poolIt, inserted] = devicePool.try_emplace(phase);
    if (inserted)
      poolIt->second = std::make_shared<Slot>();
    startBufferOffset = deviceBufferOffset.try_emplace(device, 0).first->second;
    slot = poolIt->second;
  }
  {
    std::lock_guard<std::mutex> slotLock(slot->mutex);
    if (slot->queue == std::nullopt) {
      slot->queue = PendingGraphQueue(startBufferOffset, phase, device);
    }
    slot->queue->push(numNodes, numWords, dataToEntryIds);
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto &remainingCapacity =
        deviceRemainingCapacity.try_emplace(device, metricBuffer->getCapacity())
            .first->second;
    auto &bufferOffset = deviceBufferOffset[device];
    bufferOffset = (bufferOffset + requiredBytes) % metricBuffer->getCapacity();
    remainingCapacity -= requiredBytes;
  }
}

void PendingGraphPool::peek(size_t phase) {
  std::vector<std::pair<void *, std::shared_ptr<Slot>>> slots;
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &[device, devicePool] : pool) {
      auto slotIt = devicePool.find(phase);
      if (slotIt != devicePool.end()) {
        slots.emplace_back(device, slotIt->second);
      }
    }
    for (auto &[device, slotPtr] : slots) {
      pool[device].erase(phase);
    }
  }
  std::vector<std::pair<void *, size_t>> deviceNumWords;
  for (auto &[device, slotPtr] : slots) {
    std::lock_guard<std::mutex> slotLock(slotPtr->mutex);
    if (!slotPtr->queue.has_value())
      continue;
    auto &queue = slotPtr->queue.value();
    auto numWords = queue.numWords;
    metricBuffer->peek(static_cast<Device *>(device), [&](uint8_t *hostPtr) {
      emitMetricRecords(*metricBuffer, reinterpret_cast<uint64_t *>(hostPtr),
                        queue);
    });
    deviceNumWords.emplace_back(device, numWords);
    slotPtr->queue.reset();
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &[device, numWords] : deviceNumWords) {
      deviceRemainingCapacity[device] += bytesForWords(numWords);
    }
  }
}

bool PendingGraphPool::flushIfNeeded(size_t numWords) {
  auto *device = runtime->getDevice();
  const size_t requiredBytes = bytesForWords(numWords);
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
  std::vector<std::pair<void *, size_t>> deviceNumWords;
  metricBuffer->flush(
      [&](void *device, uint8_t *hostPtr) {
        auto deviceIt = poolCopy.find(device);
        if (deviceIt == poolCopy.end())
          return;
        for (auto &[_, slot] : deviceIt->second) {
          std::lock_guard<std::mutex> lock(slot->mutex);
          if (!slot->queue.has_value())
            continue;
          deviceNumWords.emplace_back(device, slot->queue->numWords);
          emitMetricRecords(*metricBuffer,
                            reinterpret_cast<uint64_t *>(hostPtr),
                            *slot->queue);
          slot->queue.reset();
        }
      },
      true);
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &[device, numWords] : deviceNumWords) {
      deviceRemainingCapacity[device] += bytesForWords(numWords);
    }
  }
  return true;
}

} // namespace proton
