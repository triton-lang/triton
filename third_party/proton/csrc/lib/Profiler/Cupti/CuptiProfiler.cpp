#include "Profiler/Cupti/CuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"
#include "Driver/GPU/NvtxApi.h"
#include "Profiler/Cupti/CuptiPCSampling.h"
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

void updateDataPhases(std::map<Data *, std::pair<size_t, size_t>> &dataSet,
                      Data *data, size_t phase) {
  auto it = dataSet.find(data);
  if (it == dataSet.end()) {
    dataSet.emplace(data, std::make_pair(phase, phase));
  } else {
    it->second.first = std::min(it->second.first, phase); // update start phase
    it->second.second = std::max(it->second.second, phase); // update end phase
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
          updateDataPhases(dataPhases, data, entry.phase);
        }
      }
    } else {
      for (auto &[data, entry] : dataToEntry) {
        if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
          auto childEntry = data->addOp(entry.id, {Context(kernel->name)});
          childEntry.upsertMetric(std::move(kernelMetric));
          updateDataPhases(dataPhases, data, entry.phase);
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
    if (nodeState && !nodeState->isMetricNode) {
      const bool isMissingName = nodeState->isMissingName;
      if (!isMissingName) {
        nodeState->forEachEntry([activity, &dataPhases](Data *data, DataEntry &entry) {
          if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
            entry.upsertMetric(std::move(kernelMetric));
            updateDataPhases(dataPhases, data, entry.phase);
          }
        });
      } else {
        nodeState->forEachEntry([kernel, activity, &dataPhases](Data *data,
                                                   DataEntry &entry) {
          if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
            auto childEntry = data->addOp(entry.id, {Context(kernel->name)});
            childEntry.upsertMetric(std::move(kernelMetric));
            updateDataPhases(dataPhases, data, entry.phase);
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
    correlationId = processActivityKernel(corrIdToExternId, externIdToState,
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

// TODO: Move it to GPUProfiler.h once AMD side is settled
struct GraphState {
  using Callpath = std::vector<Context>;

  struct NodeState {
    // Mapping from Data object to captured callpath.
    std::map<Data *, Callpath> captureContexts;
    // A unique id for the graph node
    uint64_t nodeId{};
    // Whether the node is missing name
    bool isMissingName{};
    // Whether the node is a metric kernel node
    bool isMetricNode{};
  };

  // Capture tag to identify captured call paths
  static constexpr const char *captureTag = "<captured_at>";
  using NodeStateRef = std::reference_wrapper<NodeState>;
  // Cached per-Data callpath groups: Data -> (callpath -> [nodeStates...])
  std::map<Data *, std::map<Callpath, std::vector<NodeStateRef>>>
      dataToCallpathToNodeStates;
  // Mapping from node id to node state, has to be ordered based on node id
  // which is the order of node creation
  std::map<uint64_t, NodeState> nodeIdToState;
  // Identify whether a node is a metric kernel node.
  // NOTE: This set has to be ordered to match the node creation order.
  std::set<uint64_t> metricKernelNodeIds;
  // If the graph is launched after profiling started,
  // we need to throw an error and this error is only thrown once
  bool captureStatusChecked{};
  // A unique id for the graph and graphExec instances; they don't overlap
  uint32_t graphId{};
  // Total number of GPU kernels launched by this graph
  size_t numNodes{1};
};

// Track pending graphs per device so flushing a single device won't drain
// graphs from other devices.
class PendingGraphQueue {
public:
  explicit PendingGraphQueue(Runtime *runtime) : runtime(runtime) {}

  struct PendingGraph {
    std::map<Data *, std::vector<size_t>> dataToEntryIds;
    size_t numMetricNodes;
  };
  using PopResult = std::pair<size_t, std::vector<PendingGraph>>;

  void push(const std::map<Data *, std::vector<size_t>> &dataToEntryIds,
            size_t numNodes) {
    std::lock_guard<std::mutex> lock(mutex);
    auto device = runtime->getDevice();
    auto &queue = deviceQueues[device];
    queue.pendingGraphs.push_back(PendingGraph{dataToEntryIds, numNodes});
    queue.totalNumNodes += numNodes;
  }

  PopResult pop(size_t numNewNodes, size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex);
    if (deviceQueues.empty()) {
      return {0, {}};
    }
    auto device = runtime->getDevice();
    auto &queue = deviceQueues[device];
    if ((queue.totalNumNodes + numNewNodes) * 2 * sizeof(uint64_t) <=
        capacity) {
      return {0, {}};
    }
    return popLocked(queue);
  }

  std::vector<PopResult> popAll() {
    std::lock_guard<std::mutex> lock(mutex);
    if (deviceQueues.empty()) {
      return {{0, {}}};
    }
    std::vector<PopResult> results;
    for (auto &[device, queue] : deviceQueues) {
      results.emplace_back(popLocked(queue));
    }
    return results;
  }

private:
  struct Queue {
    size_t totalNumNodes{};
    std::vector<PendingGraph> pendingGraphs;
  };

  PopResult popLocked(Queue &queue) {
    std::vector<PendingGraph> items;
    items.swap(queue.pendingGraphs);
    size_t numNodes = queue.totalNumNodes;
    queue.totalNumNodes = 0;
    return {numNodes, items};
  }

  Runtime *runtime{};
  std::map<void *, Queue> deviceQueues;
  mutable std::mutex mutex;
};

} // namespace

struct CuptiProfiler::CuptiProfilerPimpl
    : public GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface {
  CuptiProfilerPimpl(CuptiProfiler &profiler)
      : GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface(profiler),
        pendingGraphQueue(&CudaRuntime::instance()) {
    runtime = &CudaRuntime::instance();
    metricBuffer = std::make_unique<MetricBuffer>(1024 * 1024 * 64, runtime);
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
  PendingGraphQueue pendingGraphQueue;

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

  void emitMetricRecords(
      uint64_t *recordPtr,
      std::vector<PendingGraphQueue::PendingGraph> &pendingGraphs);
};

void CuptiProfiler::CuptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                    size_t *bufferSize,
                                                    size_t *maxNumRecords) {
  const auto envBufferSize =
      getIntEnv("TRITON_CUPTI_BUFFER_SIZE", 64 * 1024 * 1024);
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
  profiler.periodicFlush(dataFlushedPhases, dataPhases);
}

void CuptiProfiler::CuptiProfilerPimpl::emitMetricRecords(
    uint64_t *recordPtr,
    std::vector<PendingGraphQueue::PendingGraph> &pendingGraphs) {
  for (auto &pendingGraph : pendingGraphs) {
    auto graphMetricNodes = pendingGraph.numMetricNodes;
    for (size_t i = 0; i < graphMetricNodes; ++i) {
      auto metricId = recordPtr[0];
      auto metricValue = recordPtr[1];
      recordPtr += 2;
      auto metricDesc = metricBuffer->getMetricDescriptor(metricId);
      auto metricName = metricDesc.name;
      auto metricTypeIndex = metricDesc.typeIndex;
      for (auto &[data, entryIds] : pendingGraph.dataToEntryIds) {
        auto entryId = entryIds[i];
        switch (metricTypeIndex) {
        case variant_index_v<uint64_t, MetricValueType>: {
          uint64_t typedValue{};
          std::memcpy(&typedValue, &metricValue, sizeof(typedValue));
          data->addEntryMetrics(entryId,
                                {{metricName, MetricValueType{typedValue}}});
          break;
        }
        case variant_index_v<int64_t, MetricValueType>: {
          int64_t typedValue{};
          std::memcpy(&typedValue, &metricValue, sizeof(typedValue));
          data->addEntryMetrics(entryId,
                                {{metricName, MetricValueType{typedValue}}});
          break;
        }
        case variant_index_v<double, MetricValueType>: {
          double typedValue{};
          std::memcpy(&typedValue, &metricValue, sizeof(typedValue));
          data->addEntryMetrics(entryId,
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
          nodeState.isMissingName = true;
        }
        if (threadState.isMetricKernelLaunching) {
          nodeState.isMetricNode = true;
          graphState.metricKernelNodeIds.insert(nodeId);
        }
        for (auto *data : profiler.dataSet) {
          auto contexts = data->getContexts();
          if (!threadState.isApiExternOp ||
              !threadState.isMetricKernelLaunching)
            contexts.push_back(name);
          nodeState.captureContexts[data] = std::move(contexts);
          graphState
              .dataToCallpathToNodeStates[data][nodeState.captureContexts[data]]
              .push_back(std::ref(nodeState));
        }
      } // else no op in progress; creation triggered by graph clone/instantiate
    } else { // CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED
      uint32_t originalGraphId = 0;
      uint64_t originalNodeId = 0;
      cupti::getGraphId<true>(graphData->originalGraph, &originalGraphId);
      cupti::getGraphNodeId<true>(graphData->originalNode, &originalNodeId);
      auto &graphState = graphStates[graphId];
      // Clone all node states.
      graphState.nodeIdToState[nodeId] =
          graphStates[originalGraphId].nodeIdToState[originalNodeId];
      auto &nodeState = graphState.nodeIdToState[nodeId];
      nodeState.nodeId = nodeId;
      for (const auto &[data, callpath] : nodeState.captureContexts) {
        graphState.dataToCallpathToNodeStates[data][callpath].push_back(
            std::ref(nodeState));
      }
      if (graphStates[originalGraphId].metricKernelNodeIds.find(
              originalNodeId) !=
          graphStates[originalGraphId].metricKernelNodeIds.end()) {
        graphState.metricKernelNodeIds.insert(nodeId);
      }
    }
  } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING) {
    auto &numNodes = graphStates[graphId].numNodes;
    numNodes--;
    uint64_t nodeId = 0;
    cupti::getGraphNodeId<true>(graphData->node, &nodeId);
    auto &graphState = graphStates[graphId];
    for (const auto &[data, callpath] :
         graphState.nodeIdToState[nodeId].captureContexts) {
      auto &nodeStates = graphState.dataToCallpathToNodeStates[data][callpath];
      nodeStates.erase(
          std::remove_if(nodeStates.begin(), nodeStates.end(),
                         [nodeId](const GraphState::NodeStateRef &state) {
                           return state.get().nodeId == nodeId;
                         }),
          nodeStates.end());
    }
    graphState.nodeIdToState.erase(nodeId);
    graphState.metricKernelNodeIds.erase(nodeId);
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
    metricBuffer->reserve();
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

      // For each unique call path, we generate a scope id per data object.
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
      for (auto &[data, callpathToNodeStates] :
           graphState.dataToCallpathToNodeStates) {
        auto *dataPtr = data;
        auto entryIt = dataToEntry.find(dataPtr);
        if (entryIt == dataToEntry.end())
          continue;
        auto baseEntry = dataPtr->addOp(entryIt->second.id,
                                        {Context{GraphState::captureTag}});
        for (const auto &[callpath, nodeStates] : callpathToNodeStates) {
          const auto nodeEntry = dataPtr->addOp(baseEntry.id, callpath);
          for (const auto &nodeStateRef : nodeStates) {
            const auto &nodeState = nodeStateRef.get();
            auto &graphNodeState = graphNodeIdToState.emplace(nodeState.nodeId);
            graphNodeState.isMissingName = nodeState.isMissingName;
            graphNodeState.isMetricNode = nodeState.isMetricNode;
            graphNodeState.setEntry(data, nodeEntry);
          }
        }
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

  if (isGraphLaunch(cbId)) {
    // Cuda context can be lazily initialized, so we need to call device get
    // here after the first kernel is launched.
    auto graphExec =
        static_cast<const cuGraphLaunch_params *>(callbackData->functionParams)
            ->hGraph;
    uint32_t graphExecId = 0;
    cupti::getGraphExecId<true>(graphExec, &graphExecId);
    auto graphRef = graphStates.find(graphExecId);
    if (graphRef.has_value() &&
        !graphRef.value().get().metricKernelNodeIds.empty()) {
      std::map<Data *, std::vector<size_t>> metricNodeEntryIds;
      auto &graphExecState = graphRef.value().get();
      auto &externIdState =
          profiler.correlation
              .externIdToState[threadState.scopeStack.back().scopeId];
      for (auto nodeId : graphExecState.metricKernelNodeIds) {
        auto *nodeState = externIdState.graphNodeIdToState.find(nodeId);
        if (!nodeState)
          continue;
        nodeState->forEachEntry([&](Data *data, const DataEntry &entry) {
          metricNodeEntryIds[data].push_back(entry.id);
        });
      }
      auto metricBufferCapacity = metricBuffer->getCapacity(); // bytes
      auto metricNodeCount = graphExecState.metricKernelNodeIds.size();
      auto drained =
          pendingGraphQueue.pop(metricNodeCount, metricBufferCapacity);
      if (drained.first != 0) { // Reached capacity
        metricBuffer->flush([&](uint8_t *data, size_t dataSize) {
          auto *recordPtr = reinterpret_cast<uint64_t *>(data);
          emitMetricRecords(recordPtr, drained.second);
        });
      }
      pendingGraphQueue.push(metricNodeEntryIds, metricNodeCount);
    }
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
  if (auto popResult = pendingGraphQueue.popAll(); !popResult.empty()) {
    auto resultIdx = 0;
    metricBuffer->flush(
        [&](uint8_t *data, size_t dataSize) {
          auto *recordPtr = reinterpret_cast<uint64_t *>(data);
          emitMetricRecords(recordPtr, popResult[resultIdx].second);
          resultIdx++;
        },
        /*flushAll=*/true);
  }
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
    auto delimiterPos = modeAndOptions[1].find('=');
    periodicFlushingEnabled = true;
    if (delimiterPos != std::string::npos) {
      const std::string key = modeAndOptions[1].substr(0, delimiterPos);
      const std::string value = modeAndOptions[1].substr(delimiterPos + 1);
      if (key != "format") {
        throw std::invalid_argument(
            "[PROTON] CuptiProfiler: unsupported option key: " + key);
      }
      if (value != "hatchet_msgpack" && value != "chrome_trace" && 
          value != "hatchet") {
        throw std::invalid_argument(
            "[PROTON] CuptiProfiler: unsupported format: " + value);
      }
      periodicFlushingFormat = value;
    } else {
      periodicFlushingFormat = "hatchet";
    }
  } else if (!mode.empty()) {
    throw std::invalid_argument("[PROTON] CuptiProfiler: unsupported mode: " +
                                mode);
  }
}

} // namespace proton
