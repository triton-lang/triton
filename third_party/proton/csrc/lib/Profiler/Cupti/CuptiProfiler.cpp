#include "Profiler/Cupti/CuptiProfiler.h"
#include "Context/Context.h"
#include "Context/Shadow.h"
#include "Data/Metric.h"
#include "Data/TreeData.h"
#include "Device.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"
#include "Driver/GPU/NvtxApi.h"
#include "Profiler/Cupti/CuptiPCSampling.h"
#include "Profiler/Graph.h"
#include "Runtime/CudaRuntime.h"
#include "Utility/Env.h"
#include "Utility/Map.h"
#include "Utility/String.h"
#include "Utility/Vector.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <unistd.h>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace proton {

template <>
thread_local GPUProfiler<CuptiProfiler>::ThreadState
    GPUProfiler<CuptiProfiler>::threadState(CuptiProfiler::instance());

namespace {

std::unique_ptr<Metric>
convertKernelActivityToMetric(CUpti_Activity *activity) {
  std::unique_ptr<Metric> metric;
  auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
  if (kernel->start < kernel->end) {
    metric =
        std::make_unique<KernelMetric>(static_cast<uint64_t>(kernel->start),
                                       static_cast<uint64_t>(kernel->end), 1,
                                       static_cast<uint64_t>(kernel->deviceId),
                                       static_cast<uint64_t>(DeviceType::CUDA),
                                       static_cast<uint64_t>(kernel->streamId));
  } // else: not a valid kernel activity
  return metric;
}

bool cuptiDebugStateEnabled() {
  static const bool enabled = getBoolEnv("PROTON_CUPTI_DEBUG_STATE", false);
  return enabled;
}

bool reuseCuptiActivityBuffersEnabled() {
  static const bool enabled =
      getBoolEnv("PROTON_CUPTI_REUSE_ACTIVITY_BUFFERS", false);
  return enabled;
}

bool useKernelActivityOnlyEnabled() {
  static const bool enabled =
      getBoolEnv("PROTON_CUPTI_USE_KERNEL_ACTIVITY", false);
  return enabled;
}

bool disableForcedFlushEnabled() {
  static const bool enabled =
      getBoolEnv("PROTON_CUPTI_DISABLE_FORCED_FLUSH", false);
  return enabled;
}

bool disableGraphCallbacksEnabled() {
  static const bool enabled =
      getBoolEnv("PROTON_CUPTI_DISABLE_GRAPH_CALLBACKS", false);
  return enabled;
}

bool disableNvtxCallbacksEnabled() {
  static const bool enabled =
      getBoolEnv("PROTON_CUPTI_DISABLE_NVTX_CALLBACKS", false);
  return enabled;
}

struct GraphStateSummary {
  size_t graphs{0};
  size_t graphNodes{0};
  size_t dataNodeStateMaps{0};
  size_t metricNodes{0};
  size_t metricWords{0};
};

struct ThreadStateSummary {
  size_t scopeStack{0};
  size_t threadDataEntries{0};
  bool isApiExternOp{false};
  bool isStreamCapturing{false};
  size_t metricKernelWordsQueued{0};
};

struct BufferLifecycleSummary {
  size_t allocated{0};
  size_t completed{0};
  size_t outstanding{0};
  size_t outstandingBytes{0};
  size_t maxOutstanding{0};
  size_t maxOutstandingBytes{0};
  size_t pooledBuffers{0};
  size_t pooledBytes{0};
  size_t poolHits{0};
  size_t poolMisses{0};
};

std::atomic<size_t> cuptiBuffersAllocated{0};
std::atomic<size_t> cuptiBuffersCompleted{0};
std::atomic<size_t> cuptiBuffersOutstanding{0};
std::atomic<size_t> cuptiBufferBytesOutstanding{0};
std::atomic<size_t> cuptiBuffersMaxOutstanding{0};
std::atomic<size_t> cuptiBufferBytesMaxOutstanding{0};
std::atomic<size_t> cuptiPooledBuffers{0};
std::atomic<size_t> cuptiPooledBufferBytes{0};
std::atomic<size_t> cuptiPooledBufferHits{0};
std::atomic<size_t> cuptiPooledBufferMisses{0};
std::mutex cuptiBufferPoolMutex;
std::vector<std::pair<uint8_t *, size_t>> cuptiBufferPool;

BufferLifecycleSummary getBufferLifecycleSummary() {
  return BufferLifecycleSummary{
      cuptiBuffersAllocated.load(std::memory_order_relaxed),
      cuptiBuffersCompleted.load(std::memory_order_relaxed),
      cuptiBuffersOutstanding.load(std::memory_order_relaxed),
      cuptiBufferBytesOutstanding.load(std::memory_order_relaxed),
      cuptiBuffersMaxOutstanding.load(std::memory_order_relaxed),
      cuptiBufferBytesMaxOutstanding.load(std::memory_order_relaxed),
      cuptiPooledBuffers.load(std::memory_order_relaxed),
      cuptiPooledBufferBytes.load(std::memory_order_relaxed),
      cuptiPooledBufferHits.load(std::memory_order_relaxed),
      cuptiPooledBufferMisses.load(std::memory_order_relaxed)};
}

uint8_t *tryAcquirePooledCuptiBuffer(size_t size) {
  std::lock_guard<std::mutex> lock(cuptiBufferPoolMutex);
  for (auto it = cuptiBufferPool.begin(); it != cuptiBufferPool.end(); ++it) {
    if (it->second != size) {
      continue;
    }
    auto *buffer = it->first;
    cuptiBufferPool.erase(it);
    cuptiPooledBuffers.fetch_sub(1, std::memory_order_relaxed);
    cuptiPooledBufferBytes.fetch_sub(size, std::memory_order_relaxed);
    cuptiPooledBufferHits.fetch_add(1, std::memory_order_relaxed);
    return buffer;
  }
  cuptiPooledBufferMisses.fetch_add(1, std::memory_order_relaxed);
  return nullptr;
}

void releasePooledCuptiBuffer(uint8_t *buffer, size_t size) {
  std::lock_guard<std::mutex> lock(cuptiBufferPoolMutex);
  cuptiBufferPool.emplace_back(buffer, size);
  cuptiPooledBuffers.fetch_add(1, std::memory_order_relaxed);
  cuptiPooledBufferBytes.fetch_add(size, std::memory_order_relaxed);
}

void freeAllPooledCuptiBuffers() {
  std::vector<std::pair<uint8_t *, size_t>> buffers;
  {
    std::lock_guard<std::mutex> lock(cuptiBufferPoolMutex);
    buffers.swap(cuptiBufferPool);
  }
  for (const auto &[buffer, _] : buffers) {
    std::free(buffer);
  }
  cuptiPooledBuffers.store(0, std::memory_order_relaxed);
  cuptiPooledBufferBytes.store(0, std::memory_order_relaxed);
}

GraphStateSummary summarizeGraphStates(
    const ThreadSafeMap<uint32_t, GraphState> &graphStates) {
  GraphStateSummary summary;
  summary.graphs = graphStates.size();
  graphStates.withReadAll([&](const auto &states) {
    for (const auto &[_, state] : states) {
      summary.graphNodes += state.nodeIdToState.size();
      summary.dataNodeStateMaps += state.dataToEntryIdToNodeStates.size();
      summary.metricNodes += state.metricNodeIdToNumWords.size();
      summary.metricWords += state.numMetricWords;
    }
  });
  return summary;
}

void logCuptiState(const char *tag, uint64_t sequenceId,
                   size_t corrIdToExternIdSize, size_t externIdToStateSize,
                   const ThreadSafeMap<uint32_t, GraphState> &graphStates,
                   const ThreadStateSummary &threadStateSummary,
                   const std::set<Data *> &dataSet,
                   const std::map<Data *, size_t> *dataFlushedPhases) {
  auto graphSummary = summarizeGraphStates(graphStates);
  auto bufferSummary = getBufferLifecycleSummary();
  auto shadowSummary = ShadowContextSource::debugStats();
  std::cerr << "[PROTON][CUPTI_DEBUG] pid=" << getpid() << " " << tag
            << " seq=" << sequenceId
            << " corrIdToExternId=" << corrIdToExternIdSize
            << " externIdToState=" << externIdToStateSize
            << " graphStates=" << graphSummary.graphs
            << " graphNodes=" << graphSummary.graphNodes
            << " graphDataMaps=" << graphSummary.dataNodeStateMaps
            << " graphMetricNodes=" << graphSummary.metricNodes
            << " graphMetricWords=" << graphSummary.metricWords
            << " cuptiBuffersAllocated=" << bufferSummary.allocated
            << " cuptiBuffersCompleted=" << bufferSummary.completed
            << " cuptiBuffersOutstanding=" << bufferSummary.outstanding
            << " cuptiBufferBytesOutstanding=" << bufferSummary.outstandingBytes
            << " cuptiBuffersMaxOutstanding=" << bufferSummary.maxOutstanding
            << " cuptiBufferBytesMaxOutstanding="
            << bufferSummary.maxOutstandingBytes
            << " cuptiPooledBuffers=" << bufferSummary.pooledBuffers
            << " cuptiPooledBufferBytes=" << bufferSummary.pooledBytes
            << " cuptiPooledBufferHits=" << bufferSummary.poolHits
            << " cuptiPooledBufferMisses=" << bufferSummary.poolMisses
            << " shadowTlsThreads=" << shadowSummary.threads
            << " shadowTlsInitializedKeys=" << shadowSummary.initializedKeys
            << " shadowTlsInitializedTrue=" << shadowSummary.initializedTrue
            << " shadowTlsStackKeys=" << shadowSummary.stackKeys
            << " shadowTlsTotalContexts=" << shadowSummary.totalContexts;
  if (dataFlushedPhases != nullptr) {
    std::cerr << " dataFlushedPhases=" << dataFlushedPhases->size();
  }
  std::cerr << " dataSets=" << dataSet.size()
            << " scopeStack=" << threadStateSummary.scopeStack
            << " threadDataEntries=" << threadStateSummary.threadDataEntries
            << " isApiExternOp=" << threadStateSummary.isApiExternOp
            << " isStreamCapturing=" << threadStateSummary.isStreamCapturing
            << " metricKernelWordsQueued="
            << threadStateSummary.metricKernelWordsQueued << std::endl;

  for (auto *data : dataSet) {
    const auto phaseInfo = data->getPhaseInfo();
    std::cerr << "[PROTON][CUPTI_DEBUG] pid=" << getpid()
              << " data path=" << data->getPath()
              << " currentPhase=" << phaseInfo.current
              << " completeUpTo=" << phaseInfo.completeUpTo;
    if (auto *treeData = dynamic_cast<TreeData *>(data)) {
      auto stats = treeData->debugStats();
      std::cerr << " activePhases=" << stats.activePhases
                << " treePhases=" << stats.treePhases
                << " retainedTreeNodes=" << stats.retainedTreeNodes
                << " currentTreeNodes=" << stats.currentTreeNodes
                << " virtualTreeNodes=" << stats.virtualTreeNodes
                << " scopeIdToContextId=" << stats.scopeIdToContextId;
    }
    std::cerr << std::endl;
  }
}

uint32_t processActivityKernel(
    CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
    CuptiProfiler::ExternIdToStateMap &externIdToState,
    std::map<uint64_t, std::reference_wrapper<CuptiProfiler::ExternIdState>>
        &externIdToStateCache,
    std::map<Data *, std::pair<size_t, size_t>> &dataPhases,
    CUpti_Activity *activity) {
  // Support CUDA >= 11.0
  auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
  auto correlationId = kernel->correlationId;
  size_t externId = 0;
  if (!/*not valid*/ corrIdToExternId.withRead(
          correlationId, [&externId](size_t value) { externId = value; })) {
    corrIdToExternId.erase(correlationId);
    return correlationId;
  }
  if (kernel->graphId == 0) { // XXX: This is a misnomer confirmed by NVIDIA,
                              // actually it refers to graphExecId
    // Non-graph kernels
    bool isMissingName = false;
    DataToEntryMap dataToEntry;
    externIdToState.withRead(externId,
                             [&](const CuptiProfiler::ExternIdState &state) {
                               isMissingName = state.isMissingName;
                               dataToEntry = state.dataToEntry;
                             });
    if (!isMissingName) {
      for (auto &[data, entry] : dataToEntry) {
        if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
          entry.upsertMetric(std::move(kernelMetric));
          detail::updateDataPhases(dataPhases, data, entry.phase);
        }
      }
    } else {
      for (auto &[data, entry] : dataToEntry) {
        if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
          auto childEntry =
              data->addOp(entry.phase, entry.id, {Context(kernel->name)});
          childEntry.upsertMetric(std::move(kernelMetric));
          detail::updateDataPhases(dataPhases, data, entry.phase);
        }
      }
    }
    externIdToState.erase(externId);
    corrIdToExternId.erase(correlationId);
  } else {
    // Graph kernels
    // A single graph launch can trigger multiple kernels.
    // Our solution is to construct the following maps:
    // --- Application threads ---
    // If graph creation has been captured:
    // - parentId, nodeId -> launch context + capture context
    // Otherwise:
    // - parentId -> launch context
    // --- CUPTI thread ---
    // - corrId -> numNodes
    auto iter = externIdToStateCache.find(externId);
    CuptiProfiler::ExternIdState *state = nullptr;
    if (iter != externIdToStateCache.end()) {
      state = &iter->second.get();
    } else {
      // Cache miss, fetch from the main map
      auto ref = externIdToState.find(externId);
      // Update the cache
      externIdToStateCache.emplace(externId, ref.value());
      state = &ref.value().get();
    }
    auto &externState = *state;
    // We have a graph creation captured
    auto &graphNodeIdToState = externState.graphNodeIdToState;
    auto *nodeState = graphNodeIdToState.find(kernel->graphNodeId);
    if (nodeState && !nodeState->isMetricNode()) {
      const bool isMissingName = nodeState->isMissingName();
      if (!isMissingName) {
        nodeState->forEachEntry(
            [activity, &dataPhases](Data *data, DataEntry &entry) {
              if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
                entry.upsertLinkedMetric(std::move(kernelMetric), entry.id);
                detail::updateDataPhases(dataPhases, data, entry.phase);
              }
            });
      } else {
        nodeState->forEachEntry([kernel, activity,
                                 &dataPhases](Data *data, DataEntry &entry) {
          if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
            auto childEntry = data->addOp(Data::kVirtualPhase, entry.id,
                                          {Context(kernel->name)});
            entry.upsertLinkedMetric(std::move(kernelMetric), childEntry.id);
            detail::updateDataPhases(dataPhases, data, entry.phase);
          }
        });
      }
    } else if (!nodeState) {
      // This can happen when graph creation is not captured, or the node is
      // skipped during capture. In both cases we don't have per-node info, so
      // we just attach the kernel metric to the graph launch entry without
      // creating a child entry for the node.
      for (auto &[data, entry] : externState.dataToEntry) {
        if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
          auto childEntry =
              data->addOp(entry.phase, entry.id, {Context(kernel->name)});
          childEntry.upsertMetric(std::move(kernelMetric));
          detail::updateDataPhases(dataPhases, data, childEntry.phase);
        }
      }
    }
    // Decrease the expected kernel count
    if (externState.numNodes > 0) {
      externState.numNodes--;
    }
    // If all kernels have been processed, clean up
    if (externState.numNodes == 0) {
      externIdToState.erase(externId);
      corrIdToExternId.erase(correlationId);
    }
  }
  return correlationId;
}

uint32_t processActivity(
    CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
    CuptiProfiler::ExternIdToStateMap &externIdToState,
    std::map<uint64_t, std::reference_wrapper<CuptiProfiler::ExternIdState>>
        &externIdToStateCache,
    std::map<Data *, std::pair<size_t, size_t>> &dataPhases,
    CUpti_Activity *activity) {
  auto correlationId = 0;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    correlationId =
        processActivityKernel(corrIdToExternId, externIdToState,
                              externIdToStateCache, dataPhases, activity);
    break;
  }
  default:
    break;
  }
  return correlationId;
}

void buildGraphNodeEntries(
    const DataToEntryMap &dataToEntry, const GraphState &graphState,
    CuptiProfiler::ExternIdState::GraphNodeStateTable &graphNodeIdToState) {
  for (const auto &[data, launchEntry] : dataToEntry) {
    auto nodeStateIt = graphState.dataToEntryIdToNodeStates.find(data);
    if (nodeStateIt == graphState.dataToEntryIdToNodeStates.end())
      // This is a new data which was not enabled during graph capture
      continue;
    auto baseEntry = data->addOp(launchEntry.phase, launchEntry.id,
                                 {Context{GraphState::captureTag}});
    for (const auto &[targetEntryId, nodeStates] : nodeStateIt->second) {
      for (const auto *nodeState : nodeStates) {
        auto &graphNodeState = graphNodeIdToState.emplace(nodeState->nodeId);
        graphNodeState.status = nodeState->status;
        graphNodeState.setEntry(data, DataEntry(targetEntryId, baseEntry.phase,
                                                baseEntry.metricSet.get()));
      }
    }
  }
}

void queueGraphMetrics(
    const DataToEntryMap &dataToEntry, PendingGraphPool *pendingGraphPool,
    const CUpti_CallbackData *callbackData, const GraphState &graphState,
    CuptiProfiler::ExternIdState::GraphNodeStateTable &graphNodeIdToState) {
  if (graphState.metricNodeIdToNumWords.empty()) {
    return;
  }
  std::map<Data *, std::vector<DataEntry>> metricNodeEntries;
  size_t phase = Data::kNoCompletePhase;
  for (const auto [data, launchEntry] : dataToEntry) {
    phase = launchEntry.phase;
    for (const auto &metricNode : graphState.metricNodeIdToNumWords) {
      auto nodeId = metricNode.first;
      auto *nodeState = graphNodeIdToState.find(nodeId);
      if (!nodeState) // The node has been skipped during graph capture
        continue;
      if (nodeState->dataToEntry.count(data)) {
        metricNodeEntries[data].emplace_back(nodeState->dataToEntry.at(data));
      } else {
        // Indicate that we'll call upsertFlexibleMetric instead of
        // upsertLinkedFlexibleMetric in queueGraphMetrics, so that the kernel
        // metric can be attached to the graph launch entry when node entry is
        // not found.
        metricNodeEntries[data].emplace_back(
            DataEntry(Scope::DummyScopeId, phase, launchEntry.metricSet.get()));
      }
    }
  }

  const auto numMetricNodes = graphState.metricNodeIdToNumWords.size();
  const auto numMetricWords = graphState.numMetricWords;
  if (callbackData->context != nullptr)
    pendingGraphPool->flushIfNeeded(numMetricWords);
  pendingGraphPool->push(phase, metricNodeEntries, numMetricNodes,
                         numMetricWords);
}

constexpr std::array<CUpti_CallbackId, 11> kGraphCallbacks = {
    CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch,
    CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture,
    CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture};

#define PROTON_KERNEL_CALLBACK_LIST(X)                                         \
  X(CUPTI_DRIVER_TRACE_CBID_cuLaunch)                                          \
  X(CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid)                                      \
  X(CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync)                                 \
  X(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)                                    \
  X(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz)                               \
  X(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx)                                  \
  X(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz)                             \
  X(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel)                         \
  X(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz)                    \
  X(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice)

#define PROTON_KERNEL_CB_AS_ID(cbId) cbId,
constexpr std::array<CUpti_CallbackId, 10> kKernelCallbacks = {
    PROTON_KERNEL_CALLBACK_LIST(PROTON_KERNEL_CB_AS_ID)};
#undef PROTON_KERNEL_CB_AS_ID

constexpr std::array<CUpti_CallbackId, 5> kGraphResourceCallbacks = {
    CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED,
    CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED,
    CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING,
    CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING,
    CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING,
};

constexpr std::array<CUpti_CallbackId, 4> kResourceCallbacks = {
    CUPTI_CBID_RESOURCE_MODULE_LOADED,
    CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING,
    CUPTI_CBID_RESOURCE_CONTEXT_CREATED,
    CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING,
};

constexpr std::array<CUpti_CallbackId, 2> kNvtxCallbacks = {
    CUPTI_CBID_NVTX_nvtxRangePushA,
    CUPTI_CBID_NVTX_nvtxRangePop,
};

void setLaunchCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
  for (auto cbId : kKernelCallbacks) {
    cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                CUPTI_CB_DOMAIN_DRIVER_API, cbId);
  }
}

void setGraphCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
  for (auto cbId : kGraphCallbacks) {
    cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                CUPTI_CB_DOMAIN_DRIVER_API, cbId);
  }
  for (auto cbId : kGraphResourceCallbacks) {
    cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                CUPTI_CB_DOMAIN_RESOURCE, cbId);
  }
}

void setResourceCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
  for (auto cbId : kResourceCallbacks) {
    cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                CUPTI_CB_DOMAIN_RESOURCE, cbId);
  }
}

void setNvtxCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
  for (auto cbId : kNvtxCallbacks) {
    cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                CUPTI_CB_DOMAIN_NVTX, cbId);
  }
}

bool isKernel(CUpti_CallbackId cbId) {
  switch (cbId) {
#define PROTON_KERNEL_CB_AS_CASE(cbId)                                         \
  case cbId:                                                                   \
    return true;
    PROTON_KERNEL_CALLBACK_LIST(PROTON_KERNEL_CB_AS_CASE)
#undef PROTON_KERNEL_CB_AS_CASE
  default:
    return false;
  }
}

bool isGraphLaunch(CUpti_CallbackId cbId) {
  return cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz;
}

bool isLaunch(CUpti_CallbackId cbId) {
  return isKernel(cbId) || isGraphLaunch(cbId);
}

#undef PROTON_KERNEL_CALLBACK_LIST

} // namespace

struct CuptiProfiler::CuptiProfilerPimpl
    : public GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface {
  CuptiProfilerPimpl(CuptiProfiler &profiler)
      : GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface(profiler) {
    auto runtime = &CudaRuntime::instance();
    profiler.metricBuffer =
        std::make_unique<MetricBuffer>(1024 * 1024 * 64, runtime,
                                       /*mapped=*/true);
    profiler.pendingGraphPool =
        std::make_unique<PendingGraphPool>(profiler.metricBuffer.get());
  }
  virtual ~CuptiProfilerPimpl() = default;

  void doStart() override;
  void doFlush() override;
  void doStop() override;

  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords);
  static void completeBuffer(CUcontext context, uint32_t streamId,
                             uint8_t *buffer, size_t size, size_t validSize);
  static void callbackFn(void *userData, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbId, const void *cbData);

  static constexpr size_t AlignSize = 8;
  static constexpr size_t AttributeSize = sizeof(size_t);
  static constexpr const char *CaptureTag = "<captured_at>";

  CUpti_SubscriberHandle subscriber{};
  CuptiPCSampling pcSampling;

  ThreadSafeMap<uint32_t, GraphState> graphStates;

private:
  void handleGraphResourceCallbacks(CuptiProfiler &profiler,
                                    CUpti_CallbackId cbId,
                                    CUpti_GraphData *graphData);
  void handleResourceCallbacks(CuptiProfiler &profiler, CUpti_CallbackId cbId,
                               const void *cbData);
  void handleNvtxCallbacks(CUpti_CallbackId cbId, const void *cbData);

  bool handleStreamCaptureCallbacks(CUpti_CallbackId cbId);
  void handleApiEnterLaunchCallbacks(CuptiProfiler &profiler,
                                     CUpti_CallbackId cbId,
                                     const CUpti_CallbackData *callbackData);
  void handleApiExitLaunchCallbacks(CuptiProfiler &profiler,
                                    CUpti_CallbackId cbId,
                                    const CUpti_CallbackData *callbackData);
  void handleApiCallbacks(CuptiProfiler &profiler, CUpti_CallbackId cbId,
                          const void *cbData);
};

void CuptiProfiler::CuptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                    size_t *bufferSize,
                                                    size_t *maxNumRecords) {
  const auto envBufferSize =
      getIntEnv("TRITON_PROFILE_BUFFER_SIZE", 64 * 1024 * 1024);
  *buffer = reuseCuptiActivityBuffersEnabled()
                ? tryAcquirePooledCuptiBuffer(envBufferSize)
                : nullptr;
  if (*buffer == nullptr) {
    *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, envBufferSize));
  }
  if (*buffer == nullptr) {
    throw std::runtime_error("[PROTON] aligned_alloc failed");
  }
  *bufferSize = envBufferSize;
  *maxNumRecords = 0;
  const auto outstanding =
      cuptiBuffersOutstanding.fetch_add(1, std::memory_order_relaxed) + 1;
  const auto outstandingBytes =
      cuptiBufferBytesOutstanding.fetch_add(envBufferSize,
                                            std::memory_order_relaxed) +
      envBufferSize;
  cuptiBuffersAllocated.fetch_add(1, std::memory_order_relaxed);
  atomicMax(cuptiBuffersMaxOutstanding, outstanding);
  atomicMax(cuptiBufferBytesMaxOutstanding, outstandingBytes);
}

void CuptiProfiler::CuptiProfilerPimpl::completeBuffer(CUcontext ctx,
                                                       uint32_t streamId,
                                                       uint8_t *buffer,
                                                       size_t size,
                                                       size_t validSize) {
  CuptiProfiler &profiler = threadState.profiler;
  auto *pImpl = dynamic_cast<CuptiProfilerPimpl *>(profiler.pImpl.get());
  static thread_local uint64_t completedBufferCount = 0;
  const ThreadStateSummary threadStateSummary{
      threadState.scopeStack.size(),
      threadState.dataToEntry.size(),
      threadState.isApiExternOp,
      threadState.isStreamCapturing,
      threadState.metricKernelNumWordsQueue.size()};
  uint32_t maxCorrelationId = 0;
  static thread_local std::map<Data *, size_t> dataFlushedPhases;
  std::map<Data *, std::pair<size_t, size_t>> dataPhases;
  CUptiResult status;
  CUpti_Activity *activity = nullptr;
  std::map<uint64_t, std::reference_wrapper<CuptiProfiler::ExternIdState>>
      externIdToStateCache;
  do {
    status = cupti::activityGetNextRecord<false>(buffer, validSize, &activity);
    if (status == CUPTI_SUCCESS) {
      auto correlationId =
          processActivity(profiler.correlation.corrIdToExternId,
                          profiler.correlation.externIdToState,
                          externIdToStateCache, dataPhases, activity);
      maxCorrelationId = std::max(maxCorrelationId, correlationId);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw std::runtime_error("[PROTON] cupti::activityGetNextRecord failed");
    }
  } while (true);

  if (reuseCuptiActivityBuffersEnabled()) {
    releasePooledCuptiBuffer(buffer, size);
  } else {
    std::free(buffer);
  }
  cuptiBuffersCompleted.fetch_add(1, std::memory_order_relaxed);
  cuptiBuffersOutstanding.fetch_sub(1, std::memory_order_relaxed);
  cuptiBufferBytesOutstanding.fetch_sub(size, std::memory_order_relaxed);

  profiler.correlation.complete(maxCorrelationId);
  profiler.flushDataPhases(dataFlushedPhases, dataPhases,
                           profiler.pendingGraphPool.get());
  ++completedBufferCount;
  if (cuptiDebugStateEnabled()) {
    logCuptiState("completeBuffer", completedBufferCount,
                  profiler.correlation.corrIdToExternId.size(),
                  profiler.correlation.externIdToState.size(), pImpl->graphStates,
                  threadStateSummary,
                  profiler.getDataSet(), &dataFlushedPhases);
  }
}

void CuptiProfiler::CuptiProfilerPimpl::handleGraphResourceCallbacks(
    CuptiProfiler &profiler, CUpti_CallbackId cbId,
    CUpti_GraphData *graphData) {
  uint32_t graphId = 0;
  uint32_t graphExecId = 0;
  if (graphData->graph)
    cupti::getGraphId<true>(graphData->graph, &graphId);
  if (graphData->graphExec)
    cupti::getGraphExecId<true>(graphData->graphExec, &graphExecId);
  if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED ||
      cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED) {
    if (graphData->nodeType != CU_GRAPH_NODE_TYPE_KERNEL) {
      // We only care about kernel nodes
      return;
    }
    uint64_t nodeId = 0;
    cupti::getGraphNodeId<true>(graphData->node, &nodeId);
    if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED) {
      if (!profiler.isOpInProgress()) {
        // else no op in progress; creation triggered by graph
        // clone/instantiate, we don't increase the numNodes because the
        // original graph has already accounted for it
        return;
      }
      auto &graphState = graphStates[graphId];
      auto &nodeState = graphState.nodeIdToState[nodeId];
      nodeState.nodeId = nodeId;
      const auto &name = threadState.scopeStack.back().name;
      if (name.empty() ||
          (threadState.isApiExternOp && threadState.isMetricKernelLaunching)) {
        nodeState.status.setMissingName();
      }
      if (threadState.isMetricKernelLaunching) {
        nodeState.status.setMetricNode();
        auto metricKernelNumWords =
            threadState.metricKernelNumWordsQueue.front();
        threadState.metricKernelNumWordsQueue.pop_front();
        graphState.metricNodeIdToNumWords.insert_or_assign(
            nodeId, metricKernelNumWords);
        graphState.numMetricWords += metricKernelNumWords;
      }
      for (auto *data : profiler.dataSet) {
        auto contexts = data->getContexts();
        if (!threadState.isApiExternOp || !threadState.isMetricKernelLaunching)
          contexts.push_back(name);
        auto staticEntry =
            data->addOp(Data::kVirtualPhase, Data::kRootEntryId, contexts);
        nodeState.dataToEntryId.insert_or_assign(data, staticEntry.id);
        graphState.dataToEntryIdToNodeStates[data][staticEntry.id].insert(
            &nodeState);
      }
    } else { // CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED
      // When a graph is cloned under the stream capture mode, graphId is the
      // same as the graphExecId to be created
      uint32_t originalGraphId = 0;
      uint64_t originalNodeId = 0;
      cupti::getGraphId<true>(graphData->originalGraph, &originalGraphId);
      cupti::getGraphNodeId<true>(graphData->originalNode, &originalNodeId);
      auto &originalGraphState = graphStates[originalGraphId];
      auto &graphState = graphStates[graphId];
      graphState.nodeIdToState[nodeId] =
          originalGraphState.nodeIdToState[originalNodeId];
      auto &nodeState = graphState.nodeIdToState[nodeId];
      nodeState.nodeId = nodeId;
      for (const auto &[data, entryId] : nodeState.dataToEntryId) {
        graphState.dataToEntryIdToNodeStates[data][entryId].insert(&nodeState);
      }
      auto originalMetricNodeIt =
          originalGraphState.metricNodeIdToNumWords.find(originalNodeId);
      if (originalMetricNodeIt !=
          originalGraphState.metricNodeIdToNumWords.end()) {
        const auto numMetricWords = originalMetricNodeIt->second;
        graphState.metricNodeIdToNumWords.insert_or_assign(nodeId,
                                                           numMetricWords);
        graphState.numMetricWords += numMetricWords;
      }
    }
  } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING) {
    if (graphData->nodeType != CU_GRAPH_NODE_TYPE_KERNEL) {
      // We only care about kernel nodes
      return;
    }
    auto &graphState = graphStates[graphId];
    uint64_t nodeId = 0;
    cupti::getGraphNodeId<true>(graphData->node, &nodeId);
    graphState.numMetricWords -= graphState.metricNodeIdToNumWords[nodeId];
    for (const auto &[data, entryId] :
         graphState.nodeIdToState[nodeId].dataToEntryId) {
      graphState.dataToEntryIdToNodeStates[data][entryId].erase(
          &graphState.nodeIdToState[nodeId]);
    }
    graphState.nodeIdToState.erase(nodeId);
    graphState.metricNodeIdToNumWords.erase(nodeId);
  } else if (cbId == CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING) {
    graphStates.erase(graphId);
  } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING) {
    graphStates.erase(graphExecId);
  }
}

void CuptiProfiler::CuptiProfilerPimpl::handleResourceCallbacks(
    CuptiProfiler &profiler, CUpti_CallbackId cbId, const void *cbData) {
  auto *resourceData =
      static_cast<CUpti_ResourceData *>(const_cast<void *>(cbData));
  if (cbId == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
    auto *moduleResource = static_cast<CUpti_ModuleResourceData *>(
        resourceData->resourceDescriptor);
    if (profiler.pcSamplingEnabled)
      pcSampling.loadModule(moduleResource->pCubin, moduleResource->cubinSize);
  } else if (cbId == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
    auto *moduleResource = static_cast<CUpti_ModuleResourceData *>(
        resourceData->resourceDescriptor);
    if (profiler.pcSamplingEnabled)
      pcSampling.unloadModule(moduleResource->pCubin,
                              moduleResource->cubinSize);
  } else if (cbId == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
    if (profiler.pcSamplingEnabled)
      pcSampling.initialize(resourceData->context);
  } else if (cbId == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
    if (profiler.pcSamplingEnabled)
      pcSampling.finalize(resourceData->context);
  } else {
    auto *graphData =
        static_cast<CUpti_GraphData *>(resourceData->resourceDescriptor);
    handleGraphResourceCallbacks(profiler, cbId, graphData);
  }
}

void CuptiProfiler::CuptiProfilerPimpl::handleNvtxCallbacks(
    CUpti_CallbackId cbId, const void *cbData) {
  auto *nvtxData = static_cast<const CUpti_NvtxData *>(cbData);
  if (cbId == CUPTI_CBID_NVTX_nvtxRangePushA) {
    auto message = nvtx::getMessageFromRangePushA(nvtxData->functionParams);
    threadState.enterScope(message);
  } else if (cbId == CUPTI_CBID_NVTX_nvtxRangePop) {
    threadState.exitScope();
  } // TODO: else handle other NVTX range functions
}

bool CuptiProfiler::CuptiProfilerPimpl::handleStreamCaptureCallbacks(
    CUpti_CallbackId cbId) {
  if (cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture ||
      cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_ptsz ||
      cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2 ||
      cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz) {
    threadState.isStreamCapturing = true;
    profiler.metricBuffer->reserve();
    return true;
  }
  if (cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture ||
      cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz) {
    threadState.isStreamCapturing = false;
    return true;
  }
  return false;
}

void CuptiProfiler::CuptiProfilerPimpl::handleApiEnterLaunchCallbacks(
    CuptiProfiler &profiler, CUpti_CallbackId cbId,
    const CUpti_CallbackData *callbackData) {
  size_t numNodes = 1;
  if (isGraphLaunch(cbId)) {
    threadState.enterOp(Scope(""));
  } else {
    // Symbol name is only available for kernel launch APIs.
    const auto symbolName = callbackData->context && callbackData->symbolName
                                ? std::string(callbackData->symbolName)
                                : "";
    threadState.enterOp(Scope(symbolName));
  }

  auto &dataToEntry = threadState.dataToEntry;
  if (threadState.isStreamCapturing) // Do not correlate stream captured kernels
    return;
  if (dataToEntry.empty()) // Profiler is deactivated
    return;

  const auto &scope = threadState.scopeStack.back();
  if (isGraphLaunch(cbId)) {
    auto graphExec =
        static_cast<const cuGraphLaunch_params *>(callbackData->functionParams)
            ->hGraph;
    uint32_t graphExecId = 0;
    cupti::getGraphExecId<true>(graphExec, &graphExecId);
    numNodes = std::numeric_limits<size_t>::max();
    auto findGraph = false;
    if (graphStates.contain(graphExecId)) {
      if (!graphStates[graphExecId].captureStatusChecked)
        numNodes = graphStates[graphExecId].nodeIdToState.size();
      findGraph = true;
    }
    if (!findGraph && !graphStates[graphExecId].captureStatusChecked) {
      graphStates[graphExecId].captureStatusChecked = true;
      std::cerr << "[PROTON] Cannot find graph for graphExecId: " << graphExecId
                << ", and it may cause memory leak. To avoid this problem, "
                   "please start profiling before the graph is created."
                << std::endl;
      if (cuptiDebugStateEnabled()) {
        const ThreadStateSummary threadStateSummary{
            threadState.scopeStack.size(),
            threadState.dataToEntry.size(),
            threadState.isApiExternOp,
            threadState.isStreamCapturing,
            threadState.metricKernelNumWordsQueue.size()};
        logCuptiState("missingGraphExec", callbackData->correlationId,
                      profiler.correlation.corrIdToExternId.size(),
                      profiler.correlation.externIdToState.size(), graphStates,
                      threadStateSummary,
                      profiler.getDataSet(), nullptr);
      }
    } else if (findGraph && !graphStates[graphExecId].captureStatusChecked) {
      auto &graphState = graphStates[graphExecId];

      // For each unique call path, we generate an entry per data object.
      auto &graphNodeIdToState =
          profiler.correlation.externIdToState[scope.scopeId]
              .graphNodeIdToState;
      if (!graphState.nodeIdToState.empty()) {
        auto minNodeId = graphState.nodeIdToState.begin()->first;
        auto maxNodeId = graphState.nodeIdToState.rbegin()->first;
        graphNodeIdToState.resetRange(minNodeId, maxNodeId);
      } else {
        graphNodeIdToState.clear();
      }
      static const bool timingEnabled =
          getBoolEnv("PROTON_GRAPH_LAUNCH_TIMING", false);
      using Clock = std::chrono::steady_clock;
      auto t0 = decltype(Clock::now()){};
      if (timingEnabled)
        t0 = Clock::now();

      buildGraphNodeEntries(dataToEntry, graphState, graphNodeIdToState);

      if (timingEnabled) {
        auto t1 = Clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        std::cerr << "[PROTON] Graph launch call path time: " << elapsed
                  << " us for graphExecId: " << graphExecId << std::endl;
        t0 = Clock::now();
      }

      queueGraphMetrics(dataToEntry, profiler.pendingGraphPool.get(),
                        callbackData, graphState, graphNodeIdToState);

      if (timingEnabled) {
        auto t1 = Clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        std::cerr << "[PROTON] Graph launch metric time: " << elapsed
                  << " us for graphExecId: " << graphExecId << std::endl;
      }
    }
  }

  profiler.correlation.correlate(callbackData->correlationId, scope.scopeId,
                                 numNodes, scope.name.empty(), dataToEntry);
  if (profiler.pcSamplingEnabled)
    pcSampling.start(callbackData->context);
}

void CuptiProfiler::CuptiProfilerPimpl::handleApiExitLaunchCallbacks(
    CuptiProfiler &profiler, CUpti_CallbackId cbId,
    const CUpti_CallbackData *callbackData) {
  auto &dataToEntry = threadState.dataToEntry;
  bool deactivated = dataToEntry.empty();

  if (profiler.pcSamplingEnabled) {
    // XXX: Conservatively stop every GPU kernel for now.
    pcSampling.stop(callbackData->context, dataToEntry);
  }

  threadState.exitOp();

  if (threadState
          .isStreamCapturing) // Do not correlate for stream captured kernels
    return;
  if (deactivated) // Profiler is deactivated
    return;
  profiler.correlation.submit(callbackData->correlationId);
}

void CuptiProfiler::CuptiProfilerPimpl::handleApiCallbacks(
    CuptiProfiler &profiler, CUpti_CallbackId cbId, const void *cbData) {
  // Do not track metric kernel launches for triton ops.
  // In this case, metric kernels are launched after a triton op is entered.
  // We should track metric kernel launches for scopes.
  // In this case, the metric kernel's stack has the same name as the scope's
  // stack.
  if (threadState.isMetricKernelLaunching && profiler.isOpInProgress())
    return;

  const CUpti_CallbackData *callbackData =
      static_cast<const CUpti_CallbackData *>(cbData);
  handleStreamCaptureCallbacks(cbId);
  if (isLaunch(cbId)) {
    if (callbackData->callbackSite == CUPTI_API_ENTER) {
      handleApiEnterLaunchCallbacks(profiler, cbId, callbackData);
    } else if (callbackData->callbackSite == CUPTI_API_EXIT) {
      handleApiExitLaunchCallbacks(profiler, cbId, callbackData);
    }
  }
}

void CuptiProfiler::CuptiProfilerPimpl::callbackFn(void *userData,
                                                   CUpti_CallbackDomain domain,
                                                   CUpti_CallbackId cbId,
                                                   const void *cbData) {
  CuptiProfiler &profiler = threadState.profiler;
  auto *pImpl = dynamic_cast<CuptiProfilerPimpl *>(profiler.pImpl.get());
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    pImpl->handleResourceCallbacks(profiler, cbId, cbData);
  } else if (domain == CUPTI_CB_DOMAIN_NVTX) {
    pImpl->handleNvtxCallbacks(cbId, cbData);
  } else {
    pImpl->handleApiCallbacks(profiler, cbId, cbData);
  }
}

void CuptiProfiler::CuptiProfilerPimpl::doStart() {
  cupti::subscribe<true>(&subscriber, callbackFn, nullptr);
  if (profiler.pcSamplingEnabled) {
    setResourceCallbacks(subscriber, /*enable=*/true);
    // Continuous PC sampling is not compatible with concurrent kernel profiling
    cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
  } else {
    if (useKernelActivityOnlyEnabled()) {
      cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
    } else {
      cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    }
    if (getBoolEnv("TRITON_ENABLE_HW_TRACE", false))
      cupti::activityEnableHWTrace<true>(/*enable=*/1);
  }
  cupti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  if (!disableGraphCallbacksEnabled()) {
    setGraphCallbacks(subscriber, /*enable=*/true);
  }
  setLaunchCallbacks(subscriber, /*enable=*/true);
  if (getBoolEnv("TRITON_ENABLE_NVTX", true) && !disableNvtxCallbacksEnabled()) {
    nvtx::enable();
    setNvtxCallbacks(subscriber, /*enable=*/true);
  }
}

void CuptiProfiler::CuptiProfilerPimpl::doFlush() {
  // cuptiActivityFlushAll returns the activity records associated with all
  // contexts/streams.
  // This is a blocking call but it doesn’t issue any CUDA synchronization calls
  // implicitly thus it’s not guaranteed that all activities are completed on
  // the underlying devices.
  // We do an "opportunistic" synchronization here to try to ensure that all
  // activities are completed on the current context.
  // If the current context is not set, we don't do any synchronization.
  CUcontext cuContext = nullptr;
  cuda::ctxGetCurrent<false>(&cuContext);
  if (cuContext) {
    cuda::ctxSynchronize<true>();
  }
  profiler.correlation.flush(
      /*maxRetries=*/100, /*sleepUs=*/10,
      /*flush=*/[]() {
        cupti::activityFlushAll<true>(
            /*flag=*/0);
      });
  // CUPTI_ACTIVITY_FLAG_FLUSH_FORCED is used to ensure that even incomplete
  // activities are flushed so that the next profiling session can start with
  // new activities.
  if (!disableForcedFlushEnabled()) {
    cupti::activityFlushAll<true>(/*flag=*/CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  }
  // Flush the tensor metric buffer
  profiler.pendingGraphPool->flushAll();
}

void CuptiProfiler::CuptiProfilerPimpl::doStop() {
  if (cuptiDebugStateEnabled()) {
    const ThreadStateSummary threadStateSummary{
        threadState.scopeStack.size(),
        threadState.dataToEntry.size(),
        threadState.isApiExternOp,
        threadState.isStreamCapturing,
        threadState.metricKernelNumWordsQueue.size()};
    logCuptiState("doStop.pre", /*sequenceId=*/0,
                  profiler.correlation.corrIdToExternId.size(),
                  profiler.correlation.externIdToState.size(), graphStates,
                  threadStateSummary,
                  profiler.getDataSet(), nullptr);
  }
  if (profiler.pcSamplingEnabled) {
    profiler.pcSamplingEnabled = false;
    CUcontext cuContext = nullptr;
    cuda::ctxGetCurrent<false>(&cuContext);
    if (cuContext)
      pcSampling.finalize(cuContext);
    setResourceCallbacks(subscriber, /*enable=*/false);
    cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
  } else {
    if (useKernelActivityOnlyEnabled()) {
      cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
    } else {
      cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    }
    if (getBoolEnv("TRITON_ENABLE_HW_TRACE", false))
      cupti::activityEnableHWTrace<true>(/*enable=*/0);
  }
  profiler.periodicFlushingEnabled = false;
  profiler.periodicFlushingFormat.clear();
  // We have to clear the correlation maps before unsubscribing because CUPTI
  // will reset correlation ID after unsubscribing
  profiler.correlation.clear();
  if (!disableGraphCallbacksEnabled()) {
    setGraphCallbacks(subscriber, /*enable=*/false);
  }
  setLaunchCallbacks(subscriber, /*enable=*/false);
  if (!disableNvtxCallbacksEnabled()) {
    nvtx::disable();
    setNvtxCallbacks(subscriber, /*enable=*/false);
  }
  cupti::unsubscribe<true>(subscriber);
  cupti::finalize<true>();
  freeAllPooledCuptiBuffers();
  if (cuptiDebugStateEnabled()) {
    const ThreadStateSummary threadStateSummary{
        threadState.scopeStack.size(),
        threadState.dataToEntry.size(),
        threadState.isApiExternOp,
        threadState.isStreamCapturing,
        threadState.metricKernelNumWordsQueue.size()};
    logCuptiState("doStop.post", /*sequenceId=*/0,
                  profiler.correlation.corrIdToExternId.size(),
                  profiler.correlation.externIdToState.size(), graphStates,
                  threadStateSummary,
                  profiler.getDataSet(), nullptr);
  }
}

CuptiProfiler::CuptiProfiler() {
  pImpl = std::make_unique<CuptiProfilerPimpl>(*this);
}

CuptiProfiler::~CuptiProfiler() = default;

void CuptiProfiler::doSetMode(const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions[0];
  if (proton::toLower(mode) == "pcsampling") {
    pcSamplingEnabled = true;
  } else if (proton::toLower(mode) == "periodic_flushing") {
    detail::setPeriodicFlushingMode(periodicFlushingEnabled,
                                    periodicFlushingFormat, modeAndOptions,
                                    "CuptiProfiler");
  } else if (!mode.empty()) {
    throw std::invalid_argument("[PROTON] CuptiProfiler: unsupported mode: " +
                                mode);
  }
}

} // namespace proton
