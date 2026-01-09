#include "Profiler/Roctracer/RoctracerProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/HipApi.h"
#include "Driver/GPU/HsaApi.h"
#include "Driver/GPU/RoctracerApi.h"
#include "Runtime/HipRuntime.h"
#include "Utility/Env.h"

#include "hip/amd_detail/hip_runtime_prof.h"
#include "roctracer/roctracer_ext.h"
#include "roctracer/roctracer_hip.h"
#include "roctracer/roctracer_roctx.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace proton {

template <>
thread_local GPUProfiler<RoctracerProfiler>::ThreadState
    GPUProfiler<RoctracerProfiler>::threadState(RoctracerProfiler::instance());

namespace {

class DeviceInfo : public Singleton<DeviceInfo> {
public:
  DeviceInfo() = default;
  int mapDeviceId(int id) {
    // Lazy initialization of device offset by calling hip API.
    // Otherwise on nvidia platforms, the HSA call will fail because of no
    // available libraries.
    std::call_once(deviceOffsetFlag, [this]() { initDeviceOffset(); });
    return id - deviceOffset;
  }

private:
  void initDeviceOffset() {
    int dc = 0;
    auto ret = hip::getDeviceCount<true>(&dc);
    hsa::iterateAgents(
        [](hsa_agent_t agent, void *data) {
          auto &offset = *static_cast<int *>(data);
          int nodeId;
          hsa::agentGetInfo<true>(
              agent,
              static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
              &nodeId);
          int deviceType;
          hsa::agentGetInfo<true>(
              agent, static_cast<hsa_agent_info_t>(HSA_AGENT_INFO_DEVICE),
              &deviceType);
          if ((nodeId < offset) && (deviceType == HSA_DEVICE_TYPE_GPU))
            offset = nodeId;

          return HSA_STATUS_SUCCESS;
        },
        &deviceOffset);
  }

  std::once_flag deviceOffsetFlag;
  int deviceOffset = 0x7fffffff;
};

std::unique_ptr<Metric>
convertActivityToMetric(const roctracer_record_t *activity) {
  std::unique_ptr<Metric> metric;
  switch (activity->kind) {
  case kHipVdiCommandTask:
  case kHipVdiCommandKernel: {
    if (activity->begin_ns < activity->end_ns) {
      metric = std::make_unique<KernelMetric>(
          static_cast<uint64_t>(activity->begin_ns),
          static_cast<uint64_t>(activity->end_ns), 1,
          static_cast<uint64_t>(
              DeviceInfo::instance().mapDeviceId(activity->device_id)),
          static_cast<uint64_t>(DeviceType::HIP),
          static_cast<uint64_t>(activity->queue_id));
    }
    break;
  }
  default:
    break;
  }
  return metric;
}

void processActivityKernel(
    RoctracerProfiler::CorrIdToExternIdMap &corrIdToExternId,
    RoctracerProfiler::ExternIdToStateMap &externIdToState,
    ThreadSafeMap<uint64_t, bool, std::unordered_map<uint64_t, bool>>
        &corrIdToIsHipGraph,
    std::map<Data *, std::pair<size_t, size_t>> &dataPhases, size_t externId,
    const roctracer_record_t *activity) {
  if (externId == Scope::DummyScopeId)
    return;
  bool isGraph = corrIdToIsHipGraph.contain(activity->correlation_id);
  auto &state = externIdToState[externId];
  if (!isGraph) {
    for (auto [data, entry] : state.dataToEntry) {
      if (auto metric = convertActivityToMetric(activity)) {
        if (state.isMissingName) {
          auto childEntry = data->addOp(entry.phase, entry.id,
                                        {Context(activity->kernel_name)});
          childEntry.upsertMetric(std::move(metric));
        } else {
          entry.upsertMetric(std::move(metric));
        }
        detail::updateDataPhases(dataPhases, data, entry.phase);
      }
    }
  } else {
    // Graph kernels
    // A single graph launch can trigger multiple kernels.
    // Our solution is to construct the following maps:
    // --- Application threads ---
    // 1. Graph -> numNodes
    // 2. GraphExec -> Graph
    // --- Roctracer thread ---
    // 3. corrId -> numNodes
    for (auto [data, entry] : state.dataToEntry) {
      if (auto metric = convertActivityToMetric(activity)) {
        auto childEntry = data->addOp(entry.phase, entry.id,
                                      {Context(activity->kernel_name)});
        childEntry.upsertMetric(std::move(metric));
        detail::updateDataPhases(dataPhases, data, entry.phase);
      }
    }
  }
  --state.numNodes;
  if (state.numNodes == 0) {
    corrIdToExternId.erase(activity->correlation_id);
    corrIdToIsHipGraph.erase(activity->correlation_id);
    externIdToState.erase(externId);
  }
  return;
}

void processActivity(
    RoctracerProfiler::CorrIdToExternIdMap &corrIdToExternId,
    RoctracerProfiler::ExternIdToStateMap &externIdToState,
    ThreadSafeMap<uint64_t, bool, std::unordered_map<uint64_t, bool>>
        &corrIdToIsHipGraph,
    std::map<Data *, std::pair<size_t, size_t>> &dataPhases, size_t parentId,
    const roctracer_record_t *record) {
  switch (record->kind) {
  case kHipVdiCommandTask:
  case kHipVdiCommandKernel: {
    processActivityKernel(corrIdToExternId, externIdToState, corrIdToIsHipGraph,
                          dataPhases, parentId, record);
    break;
  }
  default:
    break;
  }
}

} // namespace

namespace {

std::tuple<bool, bool> matchKernelCbId(uint32_t cbId) {
  bool isRuntimeApi = false;
  bool isDriverApi = false;
  switch (cbId) {
  // TODO: switch to directly subscribe the APIs
  case HIP_API_ID_hipStreamBeginCapture:
  case HIP_API_ID_hipStreamEndCapture:
  case HIP_API_ID_hipExtLaunchKernel:
  case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
  case HIP_API_ID_hipExtModuleLaunchKernel:
  case HIP_API_ID_hipHccModuleLaunchKernel:
  case HIP_API_ID_hipLaunchCooperativeKernel:
  case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
  case HIP_API_ID_hipLaunchKernel:
  case HIP_API_ID_hipModuleLaunchKernel:
  case HIP_API_ID_hipGraphLaunch:
  case HIP_API_ID_hipModuleLaunchCooperativeKernel:
  case HIP_API_ID_hipModuleLaunchCooperativeKernelMultiDevice:
  case HIP_API_ID_hipGraphExecDestroy:
  case HIP_API_ID_hipGraphInstantiateWithFlags:
  case HIP_API_ID_hipGraphInstantiate: {
    isRuntimeApi = true;
    break;
  }
  default:
    break;
  }
  return std::make_pair(isRuntimeApi, isDriverApi);
}

} // namespace

struct RoctracerProfiler::RoctracerProfilerPimpl
    : public GPUProfiler<RoctracerProfiler>::GPUProfilerPimplInterface {
  RoctracerProfilerPimpl(RoctracerProfiler &profiler)
      : GPUProfiler<RoctracerProfiler>::GPUProfilerPimplInterface(profiler) {
    runtime = &HipRuntime::instance();
    metricBuffer = std::make_unique<MetricBuffer>(1024 * 1024 * 64, runtime);
  }
  virtual ~RoctracerProfilerPimpl() = default;

  void doStart() override;
  void doFlush() override;
  void doStop() override;

  static void apiCallback(uint32_t domain, uint32_t cid,
                          const void *callbackData, void *arg);
  static void activityCallback(const char *begin, const char *end, void *arg);

  ThreadSafeMap<uint64_t, bool, std::unordered_map<uint64_t, bool>>
      corrIdToIsHipGraph;

  ThreadSafeMap<hipGraphExec_t, hipGraph_t,
                std::unordered_map<hipGraphExec_t, hipGraph_t>>
      graphExecToGraph;

  ThreadSafeMap<hipGraph_t, uint32_t, std::unordered_map<hipGraph_t, uint32_t>>
      graphToNumInstances;

  ThreadSafeMap<hipStream_t, uint32_t,
                std::unordered_map<hipStream_t, uint32_t>>
      streamToCaptureCount;

  ThreadSafeMap<hipStream_t, bool, std::unordered_map<hipStream_t, bool>>
      streamToCapture;
};

void RoctracerProfiler::RoctracerProfilerPimpl::apiCallback(
    uint32_t domain, uint32_t cid, const void *callbackData, void *arg) {
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    auto [isRuntimeAPI, isDriverAPI] = matchKernelCbId(cid);
    if (!(isRuntimeAPI || isDriverAPI)) {
      return;
    }
    auto &profiler =
        dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
    auto *pImpl = dynamic_cast<RoctracerProfiler::RoctracerProfilerPimpl *>(
        profiler.pImpl.get());
    const hip_api_data_t *data =
        static_cast<const hip_api_data_t *>(callbackData);
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      // Valid context and outermost level of the kernel launch
      // TODO: Get kernel name from hip_api_data_t
      threadState.enterOp(Scope(""));
      auto &dataToEntry = threadState.dataToEntry;
      size_t numInstances = 1;
      if (cid == HIP_API_ID_hipGraphLaunch) {
        pImpl->corrIdToIsHipGraph[data->correlation_id] = true;
        hipGraphExec_t GraphExec = data->args.hipGraphLaunch.graphExec;
        numInstances = std::numeric_limits<size_t>::max();
        bool findGraph = false;
        if (pImpl->graphExecToGraph.contain(GraphExec)) {
          hipGraph_t Graph = pImpl->graphExecToGraph[GraphExec];
          if (pImpl->graphToNumInstances.contain(Graph)) {
            numInstances = pImpl->graphToNumInstances[Graph];
            findGraph = true;
          }
        }
        if (!findGraph)
          std::cerr
              << "[PROTON] Cannot find graph and it may cause a memory leak."
                 "To avoid this problem, please start profiling before the "
                 "graph is created."
              << std::endl;
      }
      auto &scope = threadState.scopeStack.back();
      auto isMissingName = scope.name.empty();
      profiler.correlation.correlate(data->correlation_id, scope.scopeId,
                                     numInstances, isMissingName, dataToEntry);
    } else if (data->phase == ACTIVITY_API_PHASE_EXIT) {
      switch (cid) {
      case HIP_API_ID_hipStreamBeginCapture: {
        hipStream_t Stream = data->args.hipStreamBeginCapture.stream;
        pImpl->streamToCaptureCount[Stream] = 0;
        pImpl->streamToCapture[Stream] = true;
        break;
      }
      case HIP_API_ID_hipStreamEndCapture: {
        hipGraph_t Graph = *(data->args.hipStreamEndCapture.pGraph);
        hipStream_t Stream = data->args.hipStreamEndCapture.stream;
        // How many times did we capture a kernel launch for this stream
        uint32_t StreamCaptureCount = pImpl->streamToCaptureCount[Stream];
        pImpl->graphToNumInstances[Graph] = StreamCaptureCount;
        pImpl->streamToCapture.erase(Stream);
        break;
      }
      case HIP_API_ID_hipLaunchKernel: {
        hipStream_t Stream = data->args.hipLaunchKernel.stream;
        if (pImpl->streamToCapture.contain(Stream))
          pImpl->streamToCaptureCount[Stream]++;
        break;
      }
      case HIP_API_ID_hipExtLaunchKernel: {
        hipStream_t Stream = data->args.hipExtLaunchKernel.stream;
        if (pImpl->streamToCapture.contain(Stream))
          pImpl->streamToCaptureCount[Stream]++;
        break;
      }
      case HIP_API_ID_hipLaunchCooperativeKernel: {
        hipStream_t Stream = data->args.hipLaunchCooperativeKernel.stream;
        if (pImpl->streamToCapture.contain(Stream))
          pImpl->streamToCaptureCount[Stream]++;
        break;
      }
      case HIP_API_ID_hipModuleLaunchKernel: {
        hipStream_t Stream = data->args.hipModuleLaunchKernel.stream;
        if (pImpl->streamToCapture.contain(Stream))
          pImpl->streamToCaptureCount[Stream]++;
        break;
      }
      case HIP_API_ID_hipModuleLaunchCooperativeKernel: {
        hipStream_t Stream = data->args.hipModuleLaunchCooperativeKernel.stream;
        if (pImpl->streamToCapture.contain(Stream))
          pImpl->streamToCaptureCount[Stream]++;
        break;
      }
      case HIP_API_ID_hipGraphInstantiateWithFlags: {
        hipGraph_t Graph = data->args.hipGraphInstantiateWithFlags.graph;
        hipGraphExec_t GraphExec =
            *(data->args.hipGraphInstantiateWithFlags.pGraphExec);
        pImpl->graphExecToGraph[GraphExec] = Graph;
        break;
      }
      case HIP_API_ID_hipGraphInstantiate: {
        hipGraph_t Graph = data->args.hipGraphInstantiate.graph;
        hipGraphExec_t GraphExec = *(data->args.hipGraphInstantiate.pGraphExec);
        pImpl->graphExecToGraph[GraphExec] = Graph;
        break;
      }
      }
      threadState.exitOp();
      // Track outstanding op for flush
      profiler.correlation.submit(data->correlation_id);
    }
  } else if (domain == ACTIVITY_DOMAIN_ROCTX) {
    const roctx_api_data_t *data =
        static_cast<const roctx_api_data_t *>(callbackData);
    if (cid == ROCTX_API_ID_roctxRangePushA) {
      threadState.enterScope((data->args).message);
    } else if (cid == ROCTX_API_ID_roctxRangePop) {
      threadState.exitScope();
    }
  }
}

void RoctracerProfiler::RoctracerProfilerPimpl::activityCallback(
    const char *begin, const char *end, void *arg) {
  auto &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto *pImpl = dynamic_cast<RoctracerProfiler::RoctracerProfilerPimpl *>(
      profiler.pImpl.get());
  auto &correlation = profiler.correlation;

  static thread_local std::map<Data *, size_t> dataFlushedPhases;
  const roctracer_record_t *record =
      reinterpret_cast<const roctracer_record_t *>(begin);
  const roctracer_record_t *endRecord =
      reinterpret_cast<const roctracer_record_t *>(end);
  uint64_t maxCorrelationId = 0;
  std::map<Data *, std::pair<size_t, size_t>> dataPhases;

  while (record != endRecord) {
    // Log latest completed correlation id.  Used to ensure we have flushed all
    // data on stop
    maxCorrelationId =
        std::max<uint64_t>(maxCorrelationId, record->correlation_id);
    auto externId = Scope::DummyScopeId;
    bool hasCorrelation = correlation.corrIdToExternId.withRead(
        record->correlation_id, [&](const size_t &value) { externId = value; });

    if (hasCorrelation) {
      // Track correlation ids from the same stream and erase those <
      // correlationId
      processActivity(correlation.corrIdToExternId, correlation.externIdToState,
                      pImpl->corrIdToIsHipGraph, dataPhases, externId, record);
    } else {
      correlation.corrIdToExternId.erase(record->correlation_id);
      pImpl->corrIdToIsHipGraph.erase(record->correlation_id);
    }
    roctracer::getNextRecord<true>(record, &record);
  }
  correlation.complete(maxCorrelationId);
  profiler.flushDataPhases(dataFlushedPhases, dataPhases);
}

void RoctracerProfiler::RoctracerProfilerPimpl::doStart() {
  if (getBoolEnv("TRITON_ENABLE_NVTX", true)) {
    roctracer::enableDomainCallback<true>(ACTIVITY_DOMAIN_ROCTX, apiCallback,
                                          nullptr);
  }
  roctracer::enableDomainCallback<true>(ACTIVITY_DOMAIN_HIP_API, apiCallback,
                                        nullptr);
  // Activity Records
  roctracer_properties_t properties{0};
  const auto envBufferSize =
      getIntEnv("TRITON_PROFILE_BUFFER_SIZE", 64 * 1024 * 1024);
  properties.buffer_size = envBufferSize;
  properties.buffer_callback_fun = activityCallback;
  roctracer::openPool<true>(&properties);
  roctracer::enableDomainActivity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::start();
}

void RoctracerProfiler::RoctracerProfilerPimpl::doFlush() {
  // Implement reliable flushing.
  // Wait for all dispatched ops to be reported.
  std::ignore = hip::deviceSynchronize<true>();
  // If flushing encounters an activity record still being written, flushing
  // stops. Use a subsequent flush when the record has completed being written
  // to resume the flush.
  profiler.correlation.flush(
      /*maxRetries=*/100, /*sleepUs=*/10, /*flush=*/
      []() { roctracer::flushActivity<true>(); });
}

void RoctracerProfiler::RoctracerProfilerPimpl::doStop() {
  roctracer::stop();
  roctracer::disableDomainCallback<true>(ACTIVITY_DOMAIN_HIP_API);
  roctracer::disableDomainCallback<true>(ACTIVITY_DOMAIN_ROCTX);
  roctracer::disableDomainActivity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::closePool<true>();
}

RoctracerProfiler::RoctracerProfiler() {
  pImpl = std::make_unique<RoctracerProfilerPimpl>(*this);
}

RoctracerProfiler::~RoctracerProfiler() = default;

void RoctracerProfiler::doSetMode(
    const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions[0];
  if (proton::toLower(mode) == "periodic_flushing") {
    detail::setPeriodicFlushingMode(periodicFlushingEnabled,
                                    periodicFlushingFormat, modeAndOptions,
                                    "RoctracerProfiler");
  } else if (!mode.empty()) {
    throw std::invalid_argument(
        "[PROTON] RoctracerProfiler: unsupported mode: " + mode);
  }
}

} // namespace proton
