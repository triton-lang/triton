#include "Profiler/Graph.h"

#include "Data/Data.h"
#include "Runtime/Runtime.h"

#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace proton {

namespace {
constexpr size_t bytesForWords(size_t numWords) {
  return numWords * sizeof(uint64_t);
}

void emitMetricRecords(MetricBuffer &metricBuffer, uint64_t *hostBasePtr,
                       PendingGraphQueue &queue) {
  const size_t capacityWords = metricBuffer.getCapacity() / sizeof(uint64_t);
  const uint64_t scanStartWordOffset =
      queue.startBufferOffset / sizeof(uint64_t);
  const uint64_t endWordOffset = scanStartWordOffset + queue.numWords;
  auto readWord = [&](size_t offset) -> uint64_t {
    return hostBasePtr[offset % capacityWords];
  };

  for (uint64_t wordOffset = scanStartWordOffset; wordOffset < endWordOffset;) {
    const uint64_t seqId = readWord(wordOffset);
    wordOffset += 1;

    auto seqIdIt = queue.seqIdToState.find(seqId);
    auto &metricNodeState = seqIdIt->second;

    auto metricDesc =
        metricBuffer.getMetricDescriptor(metricNodeState.metricId);
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

    for (auto &[data, dataEntry] : metricNodeState.dataToEntry) {
      if (dataEntry.id != Scope::DummyScopeId) {
        dataEntry.upsertLinkedFlexibleMetric(metricName, metricValueVariant,
                                             dataEntry.id);
      } else {
        dataEntry.upsertFlexibleMetric(metricName, metricValueVariant);
      }
    }
    ordinalEntryIt->second.pop_front();
    ++matchedRecords;
  }

  size_t remainingRecords = 0;
  std::vector<uint64_t> firstRemainingOrdinals;
  for (const auto &[_, entries] : ordinalToEntryQueue) {
    remainingRecords += entries.size();
    if (!entries.empty() && firstRemainingOrdinals.size() < 8)
      firstRemainingOrdinals.push_back(_);
  }
  if (remainingRecords != 0) {
    auto join = [](const std::vector<uint64_t> &values) {
      std::ostringstream os;
      for (size_t i = 0; i < values.size(); ++i) {
        if (i)
          os << ",";
        os << values[i];
      }
      return os.str();
    };
    std::ostringstream os;
    os << "[PROTON] Missing CUDA graph metric records during flush"
       << " remaining=" << remainingRecords << " records_read=" << recordsRead
       << " matched=" << matchedRecords
       << " unknown_ordinals=" << unknownOrdinalRecords
       << " queue_start_word=" << startWordOffset
       << " queue_num_words=" << queue.numWords
       << " scan_start_word=" << scanStartWordOffset
       << " scan_num_words=" << scanNumWords
       << " words_written=" << wordsWritten << " first_seen=["
       << join(firstSeenOrdinals) << "]"
       << " first_unknown=[" << join(firstUnknownOrdinals) << "]"
       << " first_remaining=[" << join(firstRemainingOrdinals) << "]";
    throw std::runtime_error(os.str());
  }
}
} // namespace

void PendingGraphPool::push(size_t phase, size_t numWords,
                            PendingGraphQueue::SeqIdToStateMap &&seqIdToState) {
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
    slot->queue->push(numWords, std::move(seqIdToState));
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
    metricBuffer->peek(static_cast<Device *>(device),
                       [&](uint8_t *hostPtr, uint64_t wordsWritten) {
                         emitMetricRecords(
                             *metricBuffer,
                             reinterpret_cast<uint64_t *>(hostPtr), queue,
                             wordsWritten);
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
      [&](void *device, uint8_t *hostPtr, uint64_t wordsWritten) {
        auto deviceIt = poolCopy.find(device);
        if (deviceIt == poolCopy.end())
          return;
        for (auto &[_, slot] : deviceIt->second) {
          std::lock_guard<std::mutex> lock(slot->mutex);
          if (!slot->queue.has_value())
            continue;
          auto &queue = *slot->queue;
          deviceNumWords.emplace_back(device, queue.numWords);
          emitMetricRecords(*metricBuffer,
                            reinterpret_cast<uint64_t *>(hostPtr), queue);
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
