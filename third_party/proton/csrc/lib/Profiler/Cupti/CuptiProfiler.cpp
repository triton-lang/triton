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

uint32_t processActivityKernel(
    CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
    CuptiProfiler::ExternIdToStateMap &externIdToState,
    std::map<uint64_t, std::reference_wrapper<CuptiProfiler::ExternIdState>>
        &externIdToStateCache,
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
        }
      }
    } else {
      for (auto &[data, entry] : dataToEntry) {
        if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
          auto childEntry = data->addOp(entry.id, {Context(kernel->name)});
          childEntry.upsertMetric(std::move(kernelMetric));
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
    auto nodeIt = graphNodeIdToState.find(kernel->graphNodeId);
    if (nodeIt != graphNodeIdToState.end() && !nodeIt->second.isMetricNode) {
      const bool isMissingName = nodeIt->second.isMissingName;
      if (!isMissingName) {
        nodeIt->second.forEachEntry(
            [activity](Data *, DataEntry &entry) {
              if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
                entry.upsertMetric(std::move(kernelMetric));
              }
            });
      } else {
        nodeIt->second.forEachEntry(
            [kernel, activity](Data *data, DataEntry &entry) {
              if (auto kernelMetric = convertKernelActivityToMetric(activity)) {
                auto childEntry = data->addOp(entry.id, {Context(kernel->name)});
                childEntry.upsertMetric(std::move(kernelMetric));
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
    CUpti_Activity *activity) {
  auto correlationId = 0;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    correlationId = processActivityKernel(corrIdToExternId, externIdToState,
                                          externIdToStateCache, activity);
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
  uint32_t maxCorrelationId = 0;
  CUptiResult status;
  CUpti_Activity *activity = nullptr;
  std::map<uint64_t, std::reference_wrapper<CuptiProfiler::ExternIdState>>
      externIdToStateCache;
  do {
    status = cupti::activityGetNextRecord<false>(buffer, validSize, &activity);
    if (status == CUPTI_SUCCESS) {
      auto correlationId = processActivity(
          profiler.correlation.corrIdToExternId,
          profiler.correlation.externIdToState, externIdToStateCache, activity);
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
                  !threadState.isMetricKernelLaunching) {
                contexts.push_back(name);
              }
              nodeState.captureContexts[data] = std::move(contexts);
              graphState.dataToCallpathToNodeStates
                  [data][nodeState.captureContexts[data]]
                      .push_back(std::ref(nodeState));
            }
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
          auto &nodeState = graphState.nodeIdToState[nodeId];
          nodeState.nodeId = nodeId;
          for (const auto &[data, callpath] : nodeState.captureContexts) {
            graphState.dataToCallpathToNodeStates[data][callpath].push_back(
                std::ref(nodeState));
          }
          if (pImpl->graphStates[originalGraphId].metricKernelNodeIds.find(
                  originalNodeId) !=
              pImpl->graphStates[originalGraphId].metricKernelNodeIds.end()) {
            graphState.metricKernelNodeIds.insert(nodeId);
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
          auto &nodeStates =
              graphState.dataToCallpathToNodeStates[data][callpath];
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
      const auto symbolName =
          callbackData->symbolName ? std::string(callbackData->symbolName) : "";
      threadState.enterOp(Scope(std::move(symbolName)));
      const auto &scope = threadState.scopeStack.back();
      auto &dataToEntry = threadState.dataToEntry;
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
          auto &graphState = pImpl->graphStates[graphExecId];

          // For each unique call path, we generate a scope id per data object
          auto &graphNodeIdToState =
              profiler.correlation.externIdToState[scope.scopeId]
                  .graphNodeIdToState;
          graphNodeIdToState.reserve(graphState.numNodes);
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
                auto [nodeIt, inserted] =
                    graphNodeIdToState.try_emplace(nodeState.nodeId);
                nodeIt->second.isMissingName = nodeState.isMissingName;
                nodeIt->second.isMetricNode = nodeState.isMetricNode;
                nodeIt->second.setEntry(data, nodeEntry);
              }
            }
          }
        }
      }
      bool isMissingName = scope.name.empty();
      profiler.correlation.correlate(callbackData->correlationId, scope.scopeId,
                                     numNodes, isMissingName, dataToEntry);
      if (profiler.pcSamplingEnabled && isDriverAPILaunch(cbId)) {
        pImpl->pcSampling.start(callbackData->context);
      }
    } else if (callbackData->callbackSite == CUPTI_API_EXIT) {
      if (profiler.pcSamplingEnabled && isDriverAPILaunch(cbId)) {
        auto &scope = threadState.scopeStack.back();
        auto &dataToEntry = threadState.dataToEntry;
        // XXX: Conservatively stop every GPU kernel for now
        pImpl->pcSampling.stop(callbackData->context, dataToEntry);
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
          auto &scope = threadState.scopeStack.back();
          std::map<Data *, std::vector<size_t>> metricNodeEntryIds;
          auto &graphExec = graphRef.value().get();
          auto &externIdState =
              profiler.correlation.externIdToState[scope.scopeId];
          for (auto nodeId : graphExec.metricKernelNodeIds) {
            auto nodeIt = externIdState.graphNodeIdToState.find(nodeId);
            nodeIt->second.forEachEntry(
                [&](Data *data, const DataEntry &entry) {
                  metricNodeEntryIds[data].push_back(entry.id);
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
          pImpl->pendingGraphQueue.push(metricNodeEntryIds, metricNodeCount);
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
