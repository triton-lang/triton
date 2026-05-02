#include "Profiler/Graph.h"

#include "Data/Data.h"
#include "Runtime/Runtime.h"

#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace proton {

namespace {
constexpr size_t bytesForWords(size_t numWords) {
  return numWords * sizeof(uint64_t);
}

void emitMetricRecords(MetricBuffer &metricBuffer, uint64_t *hostBasePtr,
                       const PendingGraphQueue &queue) {
  const size_t capacityWords = metricBuffer.getCapacity() / sizeof(uint64_t);
  const uint64_t startWordOffset = queue.startBufferOffset / sizeof(uint64_t);
  const uint64_t endWordOffset = startWordOffset + queue.numWords;
  uint64_t wordOffset = startWordOffset;
  auto readWord = [&](size_t offset) -> uint64_t {
    return hostBasePtr[offset % capacityWords];
  };

  auto ordinalToEntryQueue = queue.ordinalToEntryQueues;
  // Metric records are scanned in GPU append order, which can differ from
  // graph-node creation order. The ordinal in each record identifies the CPU
  // target queued for that metric-copy kernel.
  while (wordOffset < endWordOffset) {
    const uint64_t metricOrdinal = readWord(wordOffset);
    wordOffset += 1;

    const uint64_t metricId = readWord(wordOffset);
    wordOffset += 1;

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

    wordOffset += metricDesc.size;

    auto ordinalEntryIt = ordinalToEntryQueue.find(metricOrdinal);
    if (ordinalEntryIt == ordinalToEntryQueue.end() ||
        ordinalEntryIt->second.empty()) {
      continue;
    }
    for (auto &[data, dataEntry] : ordinalEntryIt->second.front()) {
      if (dataEntry.id != Scope::DummyScopeId) {
        dataEntry.upsertLinkedFlexibleMetric(metricName, metricValueVariant,
                                             dataEntry.id);
      } else {
        dataEntry.upsertFlexibleMetric(metricName, metricValueVariant);
      }
    }
    ordinalEntryIt->second.pop_front();
  }

  size_t remainingRecords = 0;
  for (const auto &[_, entries] : ordinalToEntryQueue) {
    remainingRecords += entries.size();
  }
  if (remainingRecords != 0) {
    throw std::runtime_error(
        "[PROTON] Missing CUDA graph metric records during flush");
  }
}
} // namespace

void PendingGraphPool::push(
    size_t phase,
    const std::map<uint64_t, std::map<Data *, DataEntry>> &ordinalToEntries,
    size_t numWords) {
  const size_t requiredBytes = bytesForWords(numWords);
  void *device = runtime->getDevice();
  std::shared_ptr<Slot> slot;
  size_t startBufferOffset = 0;
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
      slot->queue = PendingGraphQueue(startBufferOffset);
    }
    slot->queue->push(numWords, ordinalToEntries);
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto &remainingCapacity =
        deviceRemainingCapacity.try_emplace(device, metricBuffer->getCapacity())
            .first->second;
    auto &bufferOffset = deviceBufferOffset[device];
    bufferOffset += requiredBytes;
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
    for (auto &[device, _] : slots) {
      pool[device].erase(phase);
    }
  }

  std::vector<std::pair<void *, size_t>> deviceNumWords;
  for (auto &[device, slot] : slots) {
    std::lock_guard<std::mutex> slotLock(slot->mutex);
    if (!slot->queue.has_value())
      continue;
    auto &queue = *slot->queue;
    metricBuffer->peek(static_cast<Device *>(device), [&](uint8_t *hostPtr) {
      emitMetricRecords(*metricBuffer, reinterpret_cast<uint64_t *>(hostPtr),
                        queue);
    });
    deviceNumWords.emplace_back(device, queue.numWords);
    slot->queue.reset();
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
        auto combinedQueue = std::optional<PendingGraphQueue>(std::nullopt);
        for (auto &[_, slot] : deviceIt->second) {
          std::lock_guard<std::mutex> lock(slot->mutex);
          if (!slot->queue.has_value())
            continue;
          auto &queue = *slot->queue;
          if (!combinedQueue.has_value()) {
            combinedQueue.emplace(queue.startBufferOffset);
          }
          combinedQueue->append(queue);
          deviceNumWords.emplace_back(device, queue.numWords);
          slot->queue.reset();
        }
        if (combinedQueue.has_value()) {
          emitMetricRecords(*metricBuffer,
                            reinterpret_cast<uint64_t *>(hostPtr),
                            *combinedQueue);
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
