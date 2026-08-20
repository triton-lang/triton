#include "Profiler/Graph.h"

#include "Data/Data.h"
#include "Runtime/Runtime.h"
#include "Utility/Errors.h"

#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace proton {

void GraphState::recordNode(uint64_t nodeId, const std::string &name,
                            std::optional<MetricNodeState> metricNodeState,
                            const std::set<Data *> &dataSet,
                            bool isApiExternOp) {
  auto &nodeState = nodeIdToState[nodeId];
  nodeState.nodeId = nodeId;
  if (name.empty())
    nodeState.status.setMissingName();

  const bool isMetricKernelNode = metricNodeState.has_value();
  if (isMetricKernelNode) {
    nodeState.status.setMetricNode();
    metricNodeIdToState.insert_or_assign(nodeId, std::move(*metricNodeState));
    const auto &storedMetricNodeState = metricNodeIdToState.at(nodeId);
    metricSeqIdToNodeId.insert_or_assign(storedMetricNodeState.seqId, nodeId);
    numMetricWords += storedMetricNodeState.numWords;
  }

  for (auto *data : dataSet) {
    auto currentContexts = data->getContexts();
    std::vector<Context> contexts;
    contexts.emplace_back(captureTag);
    for (const auto &context : currentContexts)
      contexts.push_back(context);

    if (isMetricKernelNode) {
      auto flexibleMetricContexts = data->getContexts(false);
      std::vector<Context> flexibleMetricEntryContexts;
      flexibleMetricEntryContexts.emplace_back(captureTag);
      for (const auto &context : flexibleMetricContexts)
        flexibleMetricEntryContexts.push_back(context);
      if (!isApiExternOp)
        flexibleMetricEntryContexts.emplace_back(name);
      contexts.emplace_back(metricTag);
      flexibleMetricEntryContexts.emplace_back(metricTag);

      // For metric nodes, timing is attributed to a frame under the metadata
      // state, while flexible metrics are attributed to the current GPU op.
      auto staticEntry =
          data->addOp(Data::kVirtualPhase, Data::kRootEntryId, contexts);
      nodeState.dataToEntryId.insert_or_assign(data, staticEntry.id);
      dataToEntryIdToNodeStates[data][staticEntry.id].insert(&nodeState);
      auto flexibleMetricEntry = data->addOp(
          Data::kVirtualPhase, Data::kRootEntryId, flexibleMetricEntryContexts);
      metricNodeIdToState.at(nodeId).dataToEntryId.insert_or_assign(
          data, flexibleMetricEntry.id);
    } else {
      contexts.emplace_back(name);
      auto staticEntry =
          data->addOp(Data::kVirtualPhase, Data::kRootEntryId, contexts);
      nodeState.dataToEntryId.insert_or_assign(data, staticEntry.id);
      dataToEntryIdToNodeStates[data][staticEntry.id].insert(&nodeState);
    }
  }
}

void GraphState::buildLaunchEntries(const DataToEntryMap &dataToEntry,
                                    DataToEntryMap &dataToGraphEntry) const {
  for (const auto &[data, entry] : dataToEntry) {
    if (dataToEntryIdToNodeStates.find(data) == dataToEntryIdToNodeStates.end())
      // This data object was not enabled during graph capture.
      continue;
    dataToGraphEntry.insert({data, entry});
  }
}

void GraphState::queueMetrics(PendingGraphPool *pendingGraphPool,
                              const DataToEntryMap *dataToGraphEntry,
                              bool flushIfNeeded) const {
  if (metricSeqIdToNodeId.empty())
    return;

  PendingGraphQueue::SeqIdToStateMap seqIdToState;
  size_t phase = Data::kNoCompletePhase;
  for (const auto &[seqId, nodeId] : metricSeqIdToNodeId) {
    const auto &metricNodeState = metricNodeIdToState.at(nodeId);
    if (dataToGraphEntry) {
      for (const auto &[data, graphEntry] : *dataToGraphEntry) {
        phase = graphEntry.phase;
        auto &pendingMetricNode =
            seqIdToState
                .emplace(seqId,
                         PendingGraphQueue::MetricNodeState{
                             metricNodeState.metricId, {}})
                .first->second;
        auto entryId = Scope::DummyScopeId;
        if (auto entryIdIt = metricNodeState.dataToEntryId.find(data);
            entryIdIt != metricNodeState.dataToEntryId.end()) {
          entryId = entryIdIt->second;
        }
        // DummyScopeId makes emitMetricRecords attach the flexible metric to
        // the graph launch entry instead of a linked captured entry.
        pendingMetricNode.dataToEntry.emplace(
            data, DataEntry(entryId, phase, graphEntry.metricSet.get()));
      }
    }
  }

  // Metric nodes write to the buffer even when no Data object is active, so
  // retain the complete graph word count to keep buffer offsets aligned.
  if (flushIfNeeded)
    pendingGraphPool->flushIfNeeded(numMetricWords);
  pendingGraphPool->push(phase, numMetricWords, std::move(seqIdToState));
}

namespace {
constexpr size_t bytesForWords(size_t numWords) {
  return numWords * sizeof(uint64_t);
}

void emitMetricRecords(MetricBuffer &metricBuffer, uint64_t *hostBasePtr,
                       PendingGraphQueue &queue) {
  if (queue.seqIdToState.empty()) // Profiler was deactivated while graph launch
                                  // was in progress
    return;

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
