#include "Profiler/Graph.h"

#include "Data/Data.h"
#include "Runtime/Runtime.h"

#include <cstring>
#include <deque>
#include <optional>
#include <stdexcept>

namespace proton {

namespace {
constexpr size_t bytesForWords(size_t numWords) {
  return numWords * sizeof(uint64_t);
}

void emitMetricRecords(MetricBuffer &metricBuffer, uint64_t *hostBasePtr,
                       const PendingGraphQueue &queue, uint64_t endWordOffset) {
  const auto &pendingGraphs = queue.pendingGraphs;
  const size_t capacityWords = metricBuffer.getCapacity() / sizeof(uint64_t);
  const uint64_t startWordOffset = queue.startBufferOffset / sizeof(uint64_t);
  if (endWordOffset < startWordOffset) {
    endWordOffset = startWordOffset + queue.numWords;
  }
  uint64_t wordOffset = startWordOffset;
  auto readWord = [&](size_t offset) -> uint64_t {
    return hostBasePtr[offset % capacityWords];
  };

  std::map<uint64_t, std::deque<std::map<Data *, DataEntry>>>
      ordinalToEntryQueue;
  for (const auto &pendingGraph : pendingGraphs) {
    for (const auto &[ordinal, entries] : pendingGraph.ordinalToEntries) {
      ordinalToEntryQueue[ordinal].push_back(entries);
    }
  }

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
    size_t numNodes, size_t numWords) {
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
      slot->queue = PendingGraphQueue(startBufferOffset, phase, device);
    }
    slot->queue->push(numNodes, numWords, ordinalToEntries);
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
  (void)phase;
  // Graph metric-copy kernels append records through device-side atomics, and
  // the shared metric buffer order is the actual GPU execution order. A no-sync
  // phase peek can observe only a prefix of that stream. Defer graph flexible
  // metric decoding to synchronized flushAll(), where all records in the buffer
  // can be scanned and matched back to queued graph launches by metric ordinal.
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
            combinedQueue.emplace(queue.startBufferOffset, queue.phase,
                                  queue.device);
          } else if (queue.startBufferOffset <
                     combinedQueue->startBufferOffset) {
            combinedQueue->startBufferOffset = queue.startBufferOffset;
          }
          combinedQueue->pendingGraphs.insert(
              combinedQueue->pendingGraphs.end(), queue.pendingGraphs.begin(),
              queue.pendingGraphs.end());
          combinedQueue->numNodes += queue.numNodes;
          combinedQueue->numWords += queue.numWords;
          deviceNumWords.emplace_back(device, queue.numWords);
          slot->queue.reset();
        }
        if (combinedQueue.has_value()) {
          emitMetricRecords(
              *metricBuffer, reinterpret_cast<uint64_t *>(hostPtr),
              *combinedQueue,
              metricBuffer->getWrittenWords(static_cast<Device *>(device)));
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
