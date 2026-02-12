#include "Profiler/Cupti/CuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
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
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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
        nodeState->forEachEntry(
            [kernel, activity, &dataPhases](Data *data, DataEntry &entry) {
              if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
                auto childEntry = data->addOp(Data::kVirtualPhase, entry.id,
                                              {Context(kernel->name)});
                entry.upsertLinkedMetric(std::move(kernelMetric), childEntry.id);
                detail::updateDataPhases(dataPhases, data, entry.phase);
              }
            });
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
    CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED,
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
  *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, envBufferSize));
  if (*buffer == nullptr) {
    throw std::runtime_error("[PROTON] aligned_alloc failed");
  }
  *bufferSize = envBufferSize;
  *maxNumRecords = 0;
}

void CuptiProfiler::CuptiProfilerPimpl::completeBuffer(CUcontext ctx,
                                                       uint32_t streamId,
                                                       uint8_t *buffer,
                                                       size_t size,
                                                       size_t validSize) {
  CuptiProfiler &profiler = threadState.profiler;
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

  std::free(buffer);

  profiler.correlation.complete(maxCorrelationId);
  profiler.flushDataPhases(dataFlushedPhases, dataPhases,
                           profiler.pendingGraphPool.get());
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
    uint64_t nodeId = 0;
    cupti::getGraphNodeId<true>(graphData->node, &nodeId);
    if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED) {
      // When `cuGraphClone` or `cuGraphInstantiate` is called, CUPTI triggers
      // both CREATED and CLONED callbacks for each node. So we only increase
      // the numNodes in CREATED callback.
      if (!graphStates.contain(graphId))
        graphStates[graphId] = GraphState();
      else
        graphStates[graphId].numNodes++;
      if (profiler.isOpInProgress()) {
        auto &graphState = graphStates[graphId];
        auto &nodeState = graphState.nodeIdToState[nodeId];
        nodeState.nodeId = nodeId;
        const auto &name = threadState.scopeStack.back().name;
        if (name.empty() || (threadState.isApiExternOp &&
                             threadState.isMetricKernelLaunching)) {
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
          if (!threadState.isApiExternOp ||
              !threadState.isMetricKernelLaunching)
            contexts.push_back(name);
          auto staticEntry =
              data->addOp(Data::kVirtualPhase, Data::kRootEntryId, contexts);
          nodeState.dataToEntryId.insert_or_assign(data, staticEntry.id);
          graphState.dataToEntryIdToNodeStates[data][staticEntry.id].push_back(
              std::ref(nodeState));
        }
      } // else no op in progress; creation triggered by graph clone/instantiate
    } else { // CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED
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
        graphState.dataToEntryIdToNodeStates[data][entryId].push_back(
            std::ref(nodeState));
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
    auto &graphState = graphStates[graphId];
    graphState.numNodes--;
    uint64_t nodeId = 0;
    cupti::getGraphNodeId<true>(graphData->node, &nodeId);
    graphState.numMetricWords -= graphState.metricNodeIdToNumWords[nodeId];
    for (const auto &[data, entryId] :
         graphState.nodeIdToState[nodeId].dataToEntryId) {
      auto &nodeStates = graphState.dataToEntryIdToNodeStates[data][entryId];
      nodeStates.erase(
          std::remove_if(nodeStates.begin(), nodeStates.end(),
                         [nodeId](const GraphState::NodeStateRef &state) {
                           return state.get().nodeId == nodeId;
                         }),
          nodeStates.end());
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
  if (handleStreamCaptureCallbacks(cbId))
    return;
  if (!isLaunch(cbId))
    return;

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

  const auto &scope = threadState.scopeStack.back();
  auto &dataToEntry = threadState.dataToEntry;
  if (isGraphLaunch(cbId)) {
    auto graphExec =
        static_cast<const cuGraphLaunch_params *>(callbackData->functionParams)
            ->hGraph;
    uint32_t graphExecId = 0;
    cupti::getGraphExecId<true>(graphExec, &graphExecId);
    numNodes = std::numeric_limits<size_t>::max();
    auto findGraph = false;
    if (graphStates.contain(graphExecId)) {
      numNodes = graphStates[graphExecId].numNodes;
      findGraph = true;
    }
    if (!findGraph && !graphStates[graphExecId].captureStatusChecked) {
      graphStates[graphExecId].captureStatusChecked = true;
      std::cerr << "[PROTON] Cannot find graph for graphExecId: " << graphExecId
                << ", and t may cause memory leak. To avoid this problem, "
                   "please start profiling before the graph is created."
                << std::endl;
    } else if (findGraph) {
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

      for (const auto &[data, launchEntry] : dataToEntry) {
        auto nodeStateIt = graphState.dataToEntryIdToNodeStates.find(data);
        if (nodeStateIt == graphState.dataToEntryIdToNodeStates.end()) {
          continue;
        }
        auto baseEntry = data->addOp(launchEntry.phase, launchEntry.id,
                                     {Context{GraphState::captureTag}});
        for (const auto &[targetEntryId, nodeStateRefs] : nodeStateIt->second) {
          for (const auto &nodeStateRef : nodeStateRefs) {
            auto &graphNodeState =
                graphNodeIdToState.emplace(nodeStateRef.get().nodeId);
            graphNodeState.status = nodeStateRef.get().status;
            graphNodeState.setEntry(
                data, DataEntry(targetEntryId, baseEntry.phase,
                                baseEntry.metricSet.get()));
          }
        }
      }
      if (timingEnabled) {
        auto t1 = Clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        std::cerr << "[PROTON] Graph launch call path time: " << elapsed
                  << " us for graphExecId: " << graphExecId << std::endl;
        t0 = Clock::now();
      }

      if (!graphStates[graphExecId].metricNodeIdToNumWords.empty()) {
        auto &graphExecState = graphStates[graphExecId];
        std::map<Data *, std::vector<DataEntry>> metricNodeEntries;
        auto phase = Data::kNoCompletePhase;
        for (const auto &metricNode : graphExecState.metricNodeIdToNumWords) {
          auto nodeId = metricNode.first;
          auto *nodeState = graphNodeIdToState.find(nodeId);
          if (!nodeState) // The node has been skipped during graph capture
            continue;
          nodeState->forEachEntry([&](Data *data, const DataEntry &entry) {
            metricNodeEntries[data].push_back(entry);
            if (phase == Data::kNoCompletePhase) {
              phase = entry.phase;
            } else if (phase != entry.phase) {
              throw std::runtime_error(
                  "[PROTON] Inconsistent phases in graph metric nodes");
            }
          });
        }
        const auto numMetricNodes =
            graphExecState.metricNodeIdToNumWords.size();
        const auto numMetricWords = graphExecState.numMetricWords;
        if (callbackData->context != nullptr)
          profiler.pendingGraphPool->flushIfNeeded(numMetricWords);
        profiler.pendingGraphPool->push(phase, metricNodeEntries,
                                        numMetricNodes, numMetricWords);
      }
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
  if (!isLaunch(cbId))
    return;

  if (profiler.pcSamplingEnabled) {
    auto &dataToEntry = threadState.dataToEntry;
    // XXX: Conservatively stop every GPU kernel for now.
    pcSampling.stop(callbackData->context, dataToEntry);
  }

  threadState.exitOp();
  profiler.correlation.submit(callbackData->correlationId);
}

void CuptiProfiler::CuptiProfilerPimpl::handleApiCallbacks(
    CuptiProfiler &profiler, CUpti_CallbackId cbId, const void *cbData) {
  // Do not track metric kernel launches for triton ops.
  // In this case, metric kernels are launched after a triton op is entered.
  // We should track metric kernel launches for scopes. In this case, the metric
  // kernel's stack has the same name as the scope's stack.
  if (threadState.isMetricKernelLaunching && profiler.isOpInProgress())
    return;

  const CUpti_CallbackData *callbackData =
      static_cast<const CUpti_CallbackData *>(cbData);
  if (callbackData->callbackSite == CUPTI_API_ENTER) {
    handleApiEnterLaunchCallbacks(profiler, cbId, callbackData);
  } else if (callbackData->callbackSite == CUPTI_API_EXIT) {
    handleApiExitLaunchCallbacks(profiler, cbId, callbackData);
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
    cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (getBoolEnv("TRITON_ENABLE_HW_TRACE", false))
      cupti::activityEnableHWTrace<true>(/*enable=*/1);
  }
  cupti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  setGraphCallbacks(subscriber, /*enable=*/true);
  setLaunchCallbacks(subscriber, /*enable=*/true);
  if (getBoolEnv("TRITON_ENABLE_NVTX", true)) {
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
  cupti::activityFlushAll<true>(/*flag=*/CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  // Flush the tensor metric buffer
  profiler.pendingGraphPool->flushAll();
}

void CuptiProfiler::CuptiProfilerPimpl::doStop() {
  if (profiler.pcSamplingEnabled) {
    profiler.pcSamplingEnabled = false;
    CUcontext cuContext = nullptr;
    cuda::ctxGetCurrent<false>(&cuContext);
    if (cuContext)
      pcSampling.finalize(cuContext);
    setResourceCallbacks(subscriber, /*enable=*/false);
    cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
  } else {
    cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (getBoolEnv("TRITON_ENABLE_HW_TRACE", false))
      cupti::activityEnableHWTrace<true>(/*enable=*/0);
  }
  profiler.periodicFlushingEnabled = false;
  profiler.periodicFlushingFormat.clear();
  setGraphCallbacks(subscriber, /*enable=*/false);
  setLaunchCallbacks(subscriber, /*enable=*/false);
  nvtx::disable();
  setNvtxCallbacks(subscriber, /*enable=*/false);
  cupti::unsubscribe<true>(subscriber);
  cupti::finalize<true>();
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
