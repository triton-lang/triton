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

template <>
thread_local std::deque<size_t>
    GPUProfiler<CuptiProfiler>::Correlation::externIdQueue{};

namespace {

std::shared_ptr<Metric>
convertKernelActivityToMetric(CUpti_Activity *activity) {
  std::shared_ptr<Metric> metric;
  auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
  if (kernel->start < kernel->end) {
    metric =
        std::make_shared<KernelMetric>(static_cast<uint64_t>(kernel->start),
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
    std::set<Data *> &dataSet, CUpti_Activity *activity) {
  // Support CUDA >= 11.0
  auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
  auto correlationId = kernel->correlationId;
  size_t parentId = 0;
  if (!/*not valid*/ corrIdToExternId.withRead(
          correlationId, [&](const size_t &value) { parentId = value; })) {
    corrIdToExternId.erase(correlationId);
    return correlationId;
  }
  if (kernel->graphId == 0) { // XXX: This is a misnomer confirmed by NVIDIA,
                              // actually it refers to graphExecId
    // Non-graph kernels
    bool isApiExternId = false;
    externIdToState.withRead(parentId,
                             [&](const CuptiProfiler::ExternIdState &state) {
                               isApiExternId = state.isApiExternId;
                             });
    // Do not share the same Metric instance across multiple Data objects.
    // Otherwise, updating one Data will mutate the Metric observed by others
    // (counts will incorrectly compound with the number of active sessions).
    for (auto *data : dataSet) {
      if (auto metric = convertKernelActivityToMetric(activity)) {
        if (isApiExternId) {
          data->addOpAndMetric(parentId, kernel->name, metric);
        } else {
          data->addMetric(parentId, metric);
        }
      }
    }
    externIdToState.erase(parentId);
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
    auto scopeId = parentId;
    bool isAPI = true;
    auto iter = externIdToStateCache.find(scopeId);
    std::optional<std::reference_wrapper<CuptiProfiler::ExternIdState>> ref;
    if (iter == externIdToStateCache.end()) {
      // Cache miss, fetch from the main map
      ref = externIdToState.find(scopeId);
      // Update the cache
      if (ref.has_value()) {
        externIdToStateCache.emplace(scopeId, ref.value());
      }
    } else {
      ref = std::ref(iter->second);
    }
    for (auto *data : dataSet) {
      if (ref.has_value()) {
        // We have a graph creation captured
        auto &graphNodeIdToScopes = ref.value().get().graphNodeIdToScopes;
        auto nodeIt = graphNodeIdToScopes.find(kernel->graphNodeId);
        if (nodeIt == graphNodeIdToScopes.end()) {
          // No captured context for this node
          continue;
        }
        auto scopeIdPtr = nodeIt->second.findScopeId(data);
        if (scopeIdPtr == nullptr) {
          // No captured context for this data
          continue;
        }
        isAPI = nodeIt->second.isApiExternId;
        scopeId = *scopeIdPtr;
      }
      if (auto metric = convertKernelActivityToMetric(activity)) {
        if (isAPI) {
          data->addOpAndMetric(scopeId, kernel->name, metric);
        } else {
          data->addMetric(scopeId, metric);
        }
      }
    }
    if (ref.has_value()) {
      // Decrease the expected kernel count
      auto &state = ref.value().get();
      if (state.numNodes > 0) {
        state.numNodes--;
      }
      // If all kernels have been processed, clean up
      if (state.numNodes == 0) {
        externIdToState.erase(parentId);
        corrIdToExternId.erase(correlationId);
      }
    }
  }
  return correlationId;
}

uint32_t processActivity(
    CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
    CuptiProfiler::ExternIdToStateMap &externIdToState,
    std::map<uint64_t, std::reference_wrapper<CuptiProfiler::ExternIdState>>
        &externIdToStateCache,
    std::set<Data *> &dataSet, CUpti_Activity *activity) {
  auto correlationId = 0;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    correlationId =
        processActivityKernel(corrIdToExternId, externIdToState,
                              externIdToStateCache, dataSet, activity);
    break;
  }
  default:
    break;
  }
  return correlationId;
}

constexpr std::array<CUpti_CallbackId, 22> kDriverApiLaunchCallbacks = {
    CUPTI_DRIVER_TRACE_CBID_cuLaunch,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice,
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

constexpr std::array<CUpti_CallbackId, 11> kRuntimeApiLaunchCallbacks = {
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000,
};

constexpr std::array<CUpti_CallbackId, 6> kGraphResourceCallbacks = {
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

void setRuntimeCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
  for (auto cbId : kRuntimeApiLaunchCallbacks) {
    cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                CUPTI_CB_DOMAIN_RUNTIME_API, cbId);
  }
}

void setDriverCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
  for (auto cbId : kDriverApiLaunchCallbacks) {
    cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,
                                CUPTI_CB_DOMAIN_DRIVER_API, cbId);
  }
}

void setGraphCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {

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

bool isDriverAPILaunch(CUpti_CallbackId cbId) {
  return std::find(kDriverApiLaunchCallbacks.begin(),
                   kDriverApiLaunchCallbacks.end(),
                   cbId) != kDriverApiLaunchCallbacks.end();
}

// TODO: Move it to GPUProfiler.h once AMD side is settled
struct GraphState {
  using Callpath = std::vector<Context>;

  struct NodeState {
    // Mapping from Data object to captured callpath.
    std::map<Data *, Callpath> captureContexts;
    // A unique id for the graph node
    uint64_t nodeId{};
  };

  // Capture tag to identify captured call paths
  static constexpr const char *captureTag = "<captured_at>";
  // Cached per-Data callpath groups: Data -> (callpath -> [nodeIds...])
  std::map<Data *, std::map<Callpath, std::vector<uint64_t>>>
      dataToCallpathToNodes;
  // Mapping from node id to node state, has to be ordered based on node id
  // which is the order of node creation
  std::map<uint64_t, NodeState> nodeIdToState;
  // Identify whether a node is a metric kernel node.
  // NOTE: This set has to be ordered to match the node creation order.
  std::set<uint64_t> metricKernelNodeIds;
  // Identify if a node is launched by an API call or triton
  std::unordered_set<uint64_t> apiNodeIds;
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
    size_t externId;
    std::map<Data *, std::vector<std::pair<bool, size_t>>> dataToScopeIds;
    size_t numMetricNodes;
  };
  using PopResult = std::pair<size_t, std::vector<PendingGraph>>;

  void push(size_t externId,
            const std::map<Data *, std::vector<std::pair<bool, size_t>>>
                &dataToScopeIds,
            size_t numNodes) {
    std::lock_guard<std::mutex> lock(mutex);
    auto device = runtime->getDevice();
    auto &queue = deviceQueues[device];
    queue.pendingGraphs.push_back(
        PendingGraph{externId, dataToScopeIds, numNodes});
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
  static constexpr size_t BufferSize = 64 * 1024 * 1024;
  static constexpr size_t AttributeSize = sizeof(size_t);
  static constexpr const char *CaptureTag = "<captured_at>";

  CUpti_SubscriberHandle subscriber{};
  CuptiPCSampling pcSampling;

  ThreadSafeMap<uint32_t, GraphState> graphStates;
  PendingGraphQueue pendingGraphQueue;

  void emitMetricRecords(
      uint64_t *recordPtr,
      std::vector<PendingGraphQueue::PendingGraph> &pendingGraphs);
};

void CuptiProfiler::CuptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                    size_t *bufferSize,
                                                    size_t *maxNumRecords) {
  *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
  if (*buffer == nullptr) {
    throw std::runtime_error("[PROTON] aligned_alloc failed");
  }
  *bufferSize = BufferSize;
  *maxNumRecords = 0;
}

void CuptiProfiler::CuptiProfilerPimpl::completeBuffer(CUcontext ctx,
                                                       uint32_t streamId,
                                                       uint8_t *buffer,
                                                       size_t size,
                                                       size_t validSize) {
  CuptiProfiler &profiler = threadState.profiler;
  auto dataSet = profiler.getDataSet();
  uint32_t maxCorrelationId = 0;
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
                          externIdToStateCache, dataSet, activity);
      maxCorrelationId = std::max(maxCorrelationId, correlationId);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw std::runtime_error("[PROTON] cupti::activityGetNextRecord failed");
    }
  } while (true);

  std::free(buffer);

  profiler.correlation.complete(maxCorrelationId);
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
      for (auto &[data, scopeIds] : pendingGraph.dataToScopeIds) {
        auto scopeId = scopeIds[i].second;
        switch (metricTypeIndex) {
        case variant_index_v<uint64_t, MetricValueType>: {
          uint64_t typedValue{};
          std::memcpy(&typedValue, &metricValue, sizeof(typedValue));
          data->addMetrics(scopeId, {{metricName, typedValue}});
          break;
        }
        case variant_index_v<int64_t, MetricValueType>: {
          int64_t typedValue{};
          std::memcpy(&typedValue, &metricValue, sizeof(typedValue));
          data->addMetrics(scopeId, {{metricName, typedValue}});
          break;
        }
        case variant_index_v<double, MetricValueType>: {
          double typedValue{};
          std::memcpy(&typedValue, &metricValue, sizeof(typedValue));
          data->addMetrics(scopeId, {{metricName, typedValue}});
          break;
        }
        default:
          break;
        }
      }
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
    auto *resourceData =
        static_cast<CUpti_ResourceData *>(const_cast<void *>(cbData));
    if (cbId == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      auto *moduleResource = static_cast<CUpti_ModuleResourceData *>(
          resourceData->resourceDescriptor);
      if (profiler.pcSamplingEnabled) {
        pImpl->pcSampling.loadModule(moduleResource->pCubin,
                                     moduleResource->cubinSize);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
      auto *moduleResource = static_cast<CUpti_ModuleResourceData *>(
          resourceData->resourceDescriptor);
      if (profiler.pcSamplingEnabled) {
        pImpl->pcSampling.unloadModule(moduleResource->pCubin,
                                       moduleResource->cubinSize);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
      if (profiler.pcSamplingEnabled) {
        pImpl->pcSampling.initialize(resourceData->context);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
      if (profiler.pcSamplingEnabled) {
        pImpl->pcSampling.finalize(resourceData->context);
      }
    } else {
      auto *graphData =
          static_cast<CUpti_GraphData *>(resourceData->resourceDescriptor);
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
        auto dataSet = profiler.getDataSet();
        if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED) {
          // When `cuGraphClone` or `cuGraphInstantiate` is called, CUPTI
          // triggers both CREATED and CLONED callbacks for each node. So we
          // only increase the numNodes in CREATED callback
          if (!pImpl->graphStates.contain(graphId))
            pImpl->graphStates[graphId] = GraphState();
          else
            pImpl->graphStates[graphId].numNodes++;
          if (profiler.isOpInProgress()) {
            auto &graphState = pImpl->graphStates[graphId];
            for (auto *data : dataSet) {
              auto contexts = data->getContexts();
              // Trick: if the scope name is empty, it means the graph is
              // created by an API kernel but not Triton op
              if (threadState.scopeStack.back().name.empty()) {
                if (!threadState
                         .isMetricKernelLaunching) // Ignore metric kernels
                  pImpl->graphStates[graphId].apiNodeIds.insert(nodeId);
              } else {
                contexts.push_back(threadState.scopeStack.back());
              }
              graphState.nodeIdToState[nodeId].captureContexts[data] =
                  std::move(contexts);
              graphState
                  .dataToCallpathToNodes[data][graphState.nodeIdToState[nodeId]
                                                   .captureContexts[data]]
                  .push_back(nodeId);
            }
            if (threadState.isMetricKernelLaunching)
              graphState.metricKernelNodeIds.insert(nodeId);
          } // else no op in progress, the creation is triggered by graph
            // clone/instantiate
        } else { // CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED
          uint32_t originalGraphId = 0;
          uint64_t originalNodeId = 0;
          cupti::getGraphId<true>(graphData->originalGraph, &originalGraphId);
          cupti::getGraphNodeId<true>(graphData->originalNode, &originalNodeId);
          auto &graphState = pImpl->graphStates[graphId];
          // Clone all node states
          graphState.nodeIdToState[nodeId] =
              pImpl->graphStates[originalGraphId].nodeIdToState[originalNodeId];
          for (const auto &[data, callpath] :
               graphState.nodeIdToState[nodeId].captureContexts) {
            graphState.dataToCallpathToNodes[data][callpath].push_back(nodeId);
          }
          if (pImpl->graphStates[originalGraphId].metricKernelNodeIds.find(
                  originalNodeId) !=
              pImpl->graphStates[originalGraphId].metricKernelNodeIds.end()) {
            graphState.metricKernelNodeIds.insert(nodeId);
          }
          if (pImpl->graphStates[originalGraphId].apiNodeIds.find(
                  originalNodeId) !=
              pImpl->graphStates[originalGraphId].apiNodeIds.end()) {
            graphState.apiNodeIds.insert(nodeId);
          }
        }
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING) {
        auto &numNodes = pImpl->graphStates[graphId].numNodes;
        numNodes--;
        uint64_t nodeId = 0;
        cupti::getGraphNodeId<true>(graphData->node, &nodeId);
        auto &graphState = pImpl->graphStates[graphId];
        for (const auto &[data, callpath] :
             graphState.nodeIdToState[nodeId].captureContexts) {
          auto &nodes = graphState.dataToCallpathToNodes[data][callpath];
          nodes.erase(std::remove(nodes.begin(), nodes.end(), nodeId),
                      nodes.end());
        }
        graphState.nodeIdToState.erase(nodeId);
        graphState.metricKernelNodeIds.erase(nodeId);
        graphState.apiNodeIds.erase(nodeId);
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING) {
        pImpl->graphStates.erase(graphId);
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING) {
        pImpl->graphStates.erase(graphExecId);
      }
    }
  } else if (domain == CUPTI_CB_DOMAIN_NVTX) {
    auto *nvtxData = static_cast<const CUpti_NvtxData *>(cbData);
    if (cbId == CUPTI_CBID_NVTX_nvtxRangePushA) {
      auto message = nvtx::getMessageFromRangePushA(nvtxData->functionParams);
      threadState.enterScope(message);
    } else if (cbId == CUPTI_CBID_NVTX_nvtxRangePop) {
      threadState.exitScope();
    } // TODO: else handle other NVTX range functions
  } else {
    // Do not track metric kernel launches for triton ops.
    // In this case, metric kernels are launched after a triton op is entered.
    // We should track metric kernel launches for scopes.
    // In this case, the metric kernel's stack has the same name as the scope's
    // stack.
    if (threadState.isMetricKernelLaunching && profiler.isOpInProgress())
      return;
    const CUpti_CallbackData *callbackData =
        static_cast<const CUpti_CallbackData *>(cbData);
    auto *pImpl = dynamic_cast<CuptiProfilerPimpl *>(profiler.pImpl.get());
    if (callbackData->callbackSite == CUPTI_API_ENTER) {
      if (cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture ||
          cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_ptsz ||
          cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2 ||
          cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz) {
        threadState.isStreamCapturing = true;
        pImpl->metricBuffer->reserve();
        return;
      } else if (cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture ||
                 cbId == CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz) {
        threadState.isStreamCapturing = false;
        return;
      }
      threadState.enterOp();
      size_t numNodes = 1;
      if (cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
          cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz) {
        auto graphExec = static_cast<const cuGraphLaunch_params *>(
                             callbackData->functionParams)
                             ->hGraph;
        uint32_t graphExecId = 0;
        cupti::getGraphExecId<true>(graphExec, &graphExecId);
        numNodes = std::numeric_limits<size_t>::max();
        auto findGraph = false;
        if (pImpl->graphStates.contain(graphExecId)) {
          numNodes = pImpl->graphStates[graphExecId].numNodes;
          findGraph = true;
        }
        if (!findGraph &&
            !pImpl->graphStates[graphExecId].captureStatusChecked) {
          pImpl->graphStates[graphExecId].captureStatusChecked = true;
          std::cerr << "[PROTON] Cannot find graph for graphExecId: "
                    << graphExecId
                    << ", and t may cause memory leak. To avoid this problem, "
                       "please start profiling before the graph is created."
                    << std::endl;
        } else if (findGraph) {
          auto externId = profiler.correlation.externIdQueue.back();
          auto &graphState = pImpl->graphStates[graphExecId];

          // For each unique call path, we generate a scope id per data object
          auto &graphNodeIdToScopes =
              profiler.correlation.externIdToState[externId]
                  .graphNodeIdToScopes;
          graphNodeIdToScopes.reserve(graphState.numNodes * 2);
          for (auto &[data, callpathToNodes] :
               graphState.dataToCallpathToNodes) {
            const auto baseScopeId =
                data->addOp(externId, GraphState::captureTag);
            for (const auto &[callpath, nodeIds] : callpathToNodes) {
              const auto nodeScopeId = data->addOp(baseScopeId, callpath);
              for (auto nodeId : nodeIds) {
                auto [nodeIt, inserted] =
                    graphNodeIdToScopes.try_emplace(nodeId);
                if (inserted) {
                  nodeIt->second.isApiExternId =
                      graphState.apiNodeIds.find(nodeId) !=
                      graphState.apiNodeIds.end();
                }
                nodeIt->second.setScopeId(data, nodeScopeId);
              }
            }
          }
        }
      }
      profiler.correlation.correlate(callbackData->correlationId, numNodes);
      if (profiler.pcSamplingEnabled && isDriverAPILaunch(cbId)) {
        pImpl->pcSampling.start(callbackData->context);
      }
    } else if (callbackData->callbackSite == CUPTI_API_EXIT) {
      auto externId = profiler.correlation.externIdQueue.back();
      if (profiler.pcSamplingEnabled && isDriverAPILaunch(cbId)) {
        // XXX: Conservatively stop every GPU kernel for now
        pImpl->pcSampling.stop(callbackData->context, externId,
                               profiler.correlation.isApiExternId(externId));
      }
      if (cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
          cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz) {
        // Cuda context can be lazily initialized, so we need to call device get
        // here after the first kernel is launched
        auto graphExec = static_cast<const cuGraphLaunch_params *>(
                             callbackData->functionParams)
                             ->hGraph;
        uint32_t graphExecId = 0;
        cupti::getGraphExecId<true>(graphExec, &graphExecId);
        auto graphRef = pImpl->graphStates.find(graphExecId);
        if (graphRef.has_value() &&
            !graphRef.value().get().metricKernelNodeIds.empty()) {
          std::map<Data *,
                   std::vector<std::pair</*isAPI=*/bool, /*scopeId=*/size_t>>>
              metricNodeScopes;
          auto &graphExec = graphRef.value().get();
          auto &externIdState = profiler.correlation.externIdToState[externId];
          for (auto nodeId : graphExec.metricKernelNodeIds) {
            auto nodeIt = externIdState.graphNodeIdToScopes.find(nodeId);
            if (nodeIt == externIdState.graphNodeIdToScopes.end()) {
              continue;
            }
            bool isApi = nodeIt->second.isApiExternId;
            nodeIt->second.forEachScopeId([&](Data *data, size_t scopeId) {
              metricNodeScopes[data].push_back({isApi, scopeId});
            });
          }
          auto metricBufferCapacity =
              pImpl->metricBuffer->getCapacity(); // bytes
          auto metricNodeCount = graphExec.metricKernelNodeIds.size();
          auto drained = pImpl->pendingGraphQueue.pop(metricNodeCount,
                                                      metricBufferCapacity);
          if (drained.first != 0) { // Reached capacity
            pImpl->metricBuffer->flush([&](uint8_t *data, size_t dataSize) {
              auto *recordPtr = reinterpret_cast<uint64_t *>(data);
              pImpl->emitMetricRecords(recordPtr, drained.second);
            });
          }
          pImpl->pendingGraphQueue.push(externId, metricNodeScopes,
                                        metricNodeCount);
        }
      }
      threadState.exitOp();
      profiler.correlation.submit(callbackData->correlationId);
    }
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
  setRuntimeCallbacks(subscriber, /*enable=*/true);
  setDriverCallbacks(subscriber, /*enable=*/true);
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
  setGraphCallbacks(subscriber, /*enable=*/false);
  setRuntimeCallbacks(subscriber, /*enable=*/false);
  setDriverCallbacks(subscriber, /*enable=*/false);
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
  } else if (!mode.empty()) {
    throw std::invalid_argument("[PROTON] CuptiProfiler: unsupported mode: " +
                                mode);
  }
}

} // namespace proton
