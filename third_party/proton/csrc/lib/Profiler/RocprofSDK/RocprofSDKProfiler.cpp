#include "Profiler/RocprofSDK/RocprofSDKProfiler.h"

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/Dispatch.h"
#include "Driver/GPU/HipApi.h"
#include "Driver/GPU/RocprofApi.h"
#include "Driver/GPU/RoctxTypes.h"
#include "Profiler/GPUProfiler.h"
#include "Runtime/HipRuntime.h"
#include "Utility/Env.h"
#include "Utility/Map.h"
#include "Utility/Singleton.h"

#include "hip/hip_runtime_api.h"
#include "rocprofiler-sdk/agent.h"
#include "rocprofiler-sdk/buffer_tracing.h"
#include "rocprofiler-sdk/callback_tracing.h"
#include "rocprofiler-sdk/hip/api_args.h"
#include "rocprofiler-sdk/hip/runtime_api_id.h"
#include "rocprofiler-sdk/marker/api_id.h"
#include "rocprofiler-sdk/registration.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace proton {

template <>
thread_local GPUProfiler<RocprofSDKProfiler>::ThreadState
    GPUProfiler<RocprofSDKProfiler>::threadState(
        RocprofSDKProfiler::instance());

namespace {

constexpr size_t BufferSize = 64 * 1024 * 1024;
constexpr const char *UnknownKernelName = "<unknown>";

struct RocprofSDKProfilerPimpl;

// ---- SDK runtime state (singleton, outlives any profiler instance) ----

struct RocprofilerRuntimeState {
  std::mutex mutex;
  rocprofiler_context_id_t codeObjectContext{};
  rocprofiler_context_id_t profilingContext{};
  rocprofiler_buffer_id_t kernelBuffer{};
  rocprofiler_callback_thread_t callbackThread{};
  rocprofiler_client_finalize_t finalizeFunc = nullptr;
  rocprofiler_client_id_t *clientId{nullptr};
  bool configured{false};
  bool codeObjectStarted{false};
  bool profilingStarted{false};
  std::atomic<bool> nvtxEnabled{false};
  RocprofSDKProfiler::RocprofSDKProfilerPimpl *pimpl{nullptr};
};

RocprofilerRuntimeState &getRuntimeState() {
  static RocprofilerRuntimeState state;
  return state;
}

using RoctxTracerCallbackFn = int (*)(uint32_t domain, uint32_t operationId,
                                      void *data);
using RoctxRegisterTracerCallbackFn = void (*)(RoctxTracerCallbackFn);

// registerRoctxCallback is defined after the Pimpl class (needs access to
// the static roctxCallback member).
void registerRoctxCallback(bool enable);

// ---- Agent (GPU) ID mapping ----

class AgentIdMapper : public Singleton<AgentIdMapper> {
public:
  AgentIdMapper() = default;

  void initialize() {
    std::call_once(initializeFlag, [this]() {
      rocprofiler::queryAvailableAgents<true>(
          ROCPROFILER_AGENT_INFO_VERSION_0, &AgentIdMapper::callback,
          sizeof(rocprofiler_agent_t), this);
    });
  }

  uint32_t map(uint64_t agentHandle) const {
    auto it = agentToDevice.find(agentHandle);
    if (it != agentToDevice.end())
      return it->second;
    return 0;
  }

private:
  static rocprofiler_status_t callback(rocprofiler_agent_version_t version,
                                       const void **agents, size_t count,
                                       void *userData) {
    auto *self = static_cast<AgentIdMapper *>(userData);
    if (version != ROCPROFILER_AGENT_INFO_VERSION_0)
      return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    auto agentList =
        reinterpret_cast<const rocprofiler_agent_t *const *>(agents);
    self->agentToDevice.clear();
    for (size_t i = 0; i < count; ++i) {
      const auto *agent = agentList[i];
      if (agent->type == ROCPROFILER_AGENT_TYPE_GPU &&
          agent->runtime_visibility.hip) {
        auto ordinal = getHipOrdinal(*agent);
        if (ordinal)
          self->agentToDevice[agent->id.handle] =
              static_cast<uint32_t>(*ordinal);
      }
    }
    return ROCPROFILER_STATUS_SUCCESS;
  }

  static uint64_t getUuidValue(const rocprofiler_uuid_t &uuid) {
    uint64_t value = 0;
    static_assert(sizeof(value) <= sizeof(uuid.bytes));
    std::memcpy(&value, uuid.bytes, sizeof(value));
    return value;
  }

  static bool isDecimalOrdinal(const std::string &token) {
    return !token.empty() &&
           std::all_of(token.begin(), token.end(),
                       [](unsigned char c) { return std::isdigit(c); });
  }

  // Bridge rocprofiler-sdk's ROCR agent identity to the HIP device ordinals
  // used by Proton metrics. Visibility filters are layered: for example,
  // ROCR_VISIBLE_DEVICES can first create a reordered physical-agent list, and
  // HIP_VISIBLE_DEVICES then indexes into that ROCR-visible list. HIP reports
  // the selected device as ordinal 0, while rocprofiler records still identify
  // the underlying agent.
  static std::optional<int32_t>
  getVisibleIndex(const std::string &envName, int32_t ordinal,
                  const rocprofiler_uuid_t &uuid) {
    auto env = getStrEnv(envName);
    if (env.empty())
      return std::nullopt;

    constexpr const char *UuidPrefix = "GPU-";
    auto uuidValue = getUuidValue(uuid);
    int32_t index = 0;
    size_t tokenBegin = env.find_first_not_of(", ");
    while (tokenBegin != std::string::npos) {
      auto tokenEnd = env.find_first_of(", ", tokenBegin);
      auto token = env.substr(tokenBegin, tokenEnd - tokenBegin);
      if (isDecimalOrdinal(token)) {
        if (std::stoll(token) == ordinal)
          return index;
      } else if (token.rfind(UuidPrefix, 0) == 0 &&
                 token.size() > std::strlen(UuidPrefix)) {
        auto tokenUuid =
            std::strtoull(token.c_str() + std::strlen(UuidPrefix), nullptr, 16);
        if (tokenUuid == uuidValue)
          return index;
      }
      ++index;
      if (tokenEnd == std::string::npos)
        break;
      tokenBegin = env.find_first_not_of(", ", tokenEnd);
    }
    return -1;
  }

  static std::optional<int32_t>
  getHipOrdinal(const rocprofiler_agent_t &agent) {
    auto rocrIndex = agent.logical_node_type_id;
    auto rocrVisible = getVisibleIndex("ROCR_VISIBLE_DEVICES",
                                       agent.logical_node_type_id, agent.uuid);
    if (rocrVisible) {
      if (*rocrVisible < 0)
        return std::nullopt;
      rocrIndex = *rocrVisible;
    }

    auto hipVisible =
        getVisibleIndex("HIP_VISIBLE_DEVICES", rocrIndex, agent.uuid);
    if (!hipVisible)
      hipVisible =
          getVisibleIndex("CUDA_VISIBLE_DEVICES", rocrIndex, agent.uuid);
    if (!hipVisible)
      hipVisible = getVisibleIndex("GPU_DEVICE_ORDINAL", rocrIndex, agent.uuid);
    if (hipVisible) {
      if (*hipVisible < 0)
        return std::nullopt;
      return *hipVisible;
    }

    return rocrVisible ? std::optional<int32_t>{rocrIndex}
                       : std::optional<int32_t>{agent.logical_node_type_id};
  }

  std::once_flag initializeFlag;
  std::unordered_map<uint64_t, uint32_t> agentToDevice;
};

// ---- Metric conversion ----

std::unique_ptr<Metric> convertDispatchToMetric(
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record,
    uint64_t streamId) {
  if (record->start_timestamp >= record->end_timestamp)
    return nullptr;
  auto deviceId = static_cast<uint64_t>(
      AgentIdMapper::instance().map(record->dispatch_info.agent_id.handle));
  return std::make_unique<KernelMetric>(
      static_cast<uint64_t>(record->start_timestamp),
      static_cast<uint64_t>(record->end_timestamp), 1, deviceId,
      static_cast<uint64_t>(DeviceType::HIP), streamId);
}

// ---- Kernel name resolution at API ENTER time ----

const char *resolveKernelNameAtEnter(
    rocprofiler_tracing_operation_t op,
    const rocprofiler_callback_tracing_hip_api_data_t *payload) {
  switch (op) {
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
    return hip::getKernelNameRefByPtr(
        payload->args.hipLaunchKernel.function_address,
        payload->args.hipLaunchKernel.stream);
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
    return hip::getKernelNameRefByPtr(
        payload->args.hipExtLaunchKernel.function_address,
        payload->args.hipExtLaunchKernel.stream);
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
    return hip::getKernelNameRefByPtr(
        payload->args.hipLaunchCooperativeKernel.func,
        payload->args.hipLaunchCooperativeKernel.stream);
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
    return hip::getKernelNameRef(payload->args.hipModuleLaunchKernel.func);
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
    return hip::getKernelNameRef(payload->args.hipExtModuleLaunchKernel.func);
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
    return hip::getKernelNameRef(payload->args.hipHccModuleLaunchKernel.func);
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
    return hip::getKernelNameRef(
        payload->args.hipModuleLaunchCooperativeKernel.func);
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice: {
    const auto *params =
        payload->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList;
    if (params &&
        payload->args.hipExtLaunchMultiKernelMultiDevice.numDevices > 0)
      return hip::getKernelNameRefByPtr(params->func, params->stream);
    return nullptr;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice: {
    const auto *params =
        payload->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList;
    if (params &&
        payload->args.hipLaunchCooperativeKernelMultiDevice.numDevices > 0)
      return hip::getKernelNameRefByPtr(params->func, params->stream);
    return nullptr;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice: {
    const auto *params =
        payload->args.hipModuleLaunchCooperativeKernelMultiDevice
            .launchParamsList;
    if (params &&
        payload->args.hipModuleLaunchCooperativeKernelMultiDevice.numDevices >
            0)
      return hip::getKernelNameRef(params->function);
    return nullptr;
  }
  default:
    return nullptr;
  }
}

// ---- HIP stream extraction at API ENTER time ----

uint64_t
extractStreamId(rocprofiler_tracing_operation_t op,
                const rocprofiler_callback_tracing_hip_api_data_t *payload) {
  hipStream_t stream = nullptr;
  switch (op) {
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
    stream = payload->args.hipLaunchKernel.stream;
    break;
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
    stream = payload->args.hipExtLaunchKernel.stream;
    break;
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
    stream = payload->args.hipLaunchCooperativeKernel.stream;
    break;
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
    stream = payload->args.hipModuleLaunchKernel.stream;
    break;
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
    stream = payload->args.hipExtModuleLaunchKernel.stream;
    break;
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
    stream = payload->args.hipHccModuleLaunchKernel.stream;
    break;
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
    stream = payload->args.hipModuleLaunchCooperativeKernel.stream;
    break;
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch:
    stream = payload->args.hipGraphLaunch.stream;
    break;
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice: {
    const auto *p =
        payload->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList;
    if (p && payload->args.hipExtLaunchMultiKernelMultiDevice.numDevices > 0)
      stream = p->stream;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice: {
    const auto *p =
        payload->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList;
    if (p && payload->args.hipLaunchCooperativeKernelMultiDevice.numDevices > 0)
      stream = p->stream;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice: {
    const auto *p = payload->args.hipModuleLaunchCooperativeKernelMultiDevice
                        .launchParamsList;
    if (p &&
        payload->args.hipModuleLaunchCooperativeKernelMultiDevice.numDevices >
            0)
      stream = p->hStream;
    break;
  }
  default:
    break;
  }
  return reinterpret_cast<uint64_t>(stream);
}

// ---- Operation classification ----

bool isKernelLaunchOperation(rocprofiler_tracing_operation_t op) {
  switch (op) {
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice:
    return true;
  default:
    return false;
  }
}

// ---- Kernel dispatch processing (matches main's GPUProfiler interface) ----

void processKernelRecord(
    RocprofSDKProfiler &profiler,
    RocprofSDKProfiler::CorrIdToExternIdMap &corrIdToExternId,
    RocprofSDKProfiler::ExternIdToStateMap &externIdToState,
    ThreadSafeMap<uint64_t, bool, std::unordered_map<uint64_t, bool>>
        &corrIdToIsHipGraph,
    std::map<Data *, std::pair<size_t, size_t>> &dataPhases,
    const std::string &kernelName,
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record,
    uint64_t streamId) {
  auto externId = Scope::DummyScopeId;
  bool hasCorrelation =
      corrIdToExternId.withRead(record->correlation_id.internal,
                                [&](const size_t &value) { externId = value; });

  if (!hasCorrelation)
    return;

  if (externId == Scope::DummyScopeId)
    return;

  bool isGraph = corrIdToIsHipGraph.contain(record->correlation_id.internal);
  auto &state = externIdToState[externId];

  if (!isGraph) {
    for (auto [data, entry] : state.dataToEntry) {
      if (auto metric = convertDispatchToMetric(record, streamId)) {
        if (state.isMissingName) {
          auto childEntry =
              data->addOp(entry.phase, entry.id, {Context(kernelName)});
          childEntry.upsertMetric(std::move(metric));
        } else {
          entry.upsertMetric(std::move(metric));
        }
        detail::updateDataPhases(dataPhases, data, entry.phase);
      }
    }
  } else {
    for (auto [data, entry] : state.dataToEntry) {
      if (auto metric = convertDispatchToMetric(record, streamId)) {
        auto childEntry =
            data->addOp(entry.phase, entry.id, {Context(kernelName)});
        childEntry.upsertMetric(std::move(metric));
        detail::updateDataPhases(dataPhases, data, entry.phase);
      }
    }
  }

  --state.numNodes;
  if (state.numNodes == 0) {
    corrIdToExternId.erase(record->correlation_id.internal);
    corrIdToIsHipGraph.erase(record->correlation_id.internal);
    externIdToState.erase(externId);
  }
}

} // namespace

// ---- Pimpl ----

struct RocprofSDKProfiler::RocprofSDKProfilerPimpl
    : public GPUProfiler<RocprofSDKProfiler>::GPUProfilerPimplInterface {
  RocprofSDKProfilerPimpl(RocprofSDKProfiler &profiler)
      : GPUProfiler<RocprofSDKProfiler>::GPUProfilerPimplInterface(profiler) {
    auto runtime = &HipRuntime::instance();
    profiler.metricBuffer =
        std::make_unique<MetricBuffer>(1024 * 1024 * 64, runtime);
  }
  virtual ~RocprofSDKProfilerPimpl() = default;

  void doStart() override;
  void doFlush() override;
  void doStop() override;

  static void hipRuntimeCallback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t *userData, void *arg);
  static void markerCallback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t *userData, void *arg);
  static void roctxCallback(uint32_t operationId, void *data);
  static void codeObjectCallback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t *userData, void *arg);
  static void kernelBufferCallback(rocprofiler_context_id_t context,
                                   rocprofiler_buffer_id_t buffer,
                                   rocprofiler_record_header_t **headers,
                                   size_t numHeaders, void *userData,
                                   uint64_t dropCount);

  using KernelNameMap =
      ThreadSafeMap<uint64_t, std::string,
                    std::unordered_map<uint64_t, std::string>>;

  std::string getKernelName(uint64_t kernelId) {
    std::string name;
    if (!kernelNames.withRead(kernelId,
                              [&](const std::string &v) { name = v; }))
      return UnknownKernelName;
    // AMDGPU ELF objects append ".kd" (kernel descriptor) to symbol names.
    // Strip it so user-visible kernel names match the source.
    const std::string suffix = ".kd";
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      name.resize(name.size() - suffix.size());
    return name;
  }

  void setKernelName(uint64_t kernelId, const char *name) {
    if (name == nullptr)
      return;
    kernelNames[kernelId] = std::string(name);
  }

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

  // Fast check: non-zero when any stream is being captured. Avoids acquiring
  // a shared_mutex on every kernel launch EXIT just to find an empty map.
  std::atomic<int> activeCaptureCount{0};

  KernelNameMap kernelNames;

  // correlation_id → HIP stream pointer, captured at hipLaunchKernel ENTER.
  // Used to distinguish streams in trace output when the SDK's queue_id
  // maps multiple HIP streams to the same underlying HSA queue.
  ThreadSafeMap<uint64_t, uint64_t, std::unordered_map<uint64_t, uint64_t>>
      corrIdToStreamId;
};

// ---- HIP Runtime API callback (correlation tracking) ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipRuntimeCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API)
    return;

  auto operation =
      static_cast<rocprofiler_tracing_operation_t>(record.operation);
  bool isKernelOp = isKernelLaunchOperation(operation);
  auto &profiler = threadState.profiler;
  auto *impl = static_cast<RocprofSDKProfilerPimpl *>(profiler.pImpl.get());
  auto *payload = static_cast<rocprofiler_callback_tracing_hip_api_data_t *>(
      record.payload);

  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    if (!isKernelOp)
      return;

    const char *resolvedName = resolveKernelNameAtEnter(operation, payload);
    threadState.enterOp(
        Scope(resolvedName ? std::string(resolvedName) : std::string()));
    auto &dataToEntry = threadState.dataToEntry;
    size_t numInstances = 1;
    if (operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch) {
      impl->corrIdToIsHipGraph[record.correlation_id.internal] = true;
      numInstances = std::numeric_limits<size_t>::max();
      bool foundGraph = false;
      auto graphExec = payload->args.hipGraphLaunch.graphExec;
      if (impl->graphExecToGraph.contain(graphExec)) {
        auto graph = impl->graphExecToGraph[graphExec];
        if (impl->graphToNumInstances.contain(graph)) {
          numInstances = impl->graphToNumInstances[graph];
          foundGraph = true;
        }
      }
      if (!foundGraph) {
        std::cerr
            << "[PROTON] Cannot find graph and it may cause a memory leak."
               "To avoid this problem, please start profiling before the "
               "graph is created."
            << std::endl;
      }
    }
    auto &scope = threadState.scopeStack.back();
    auto isMissingName = scope.name.empty();
    profiler.correlation.correlate(record.correlation_id.internal,
                                   scope.scopeId, numInstances, isMissingName,
                                   dataToEntry);
    impl->corrIdToStreamId[record.correlation_id.internal] =
        extractStreamId(operation, payload);
    return;
  }

  if (record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT)
    return;

  switch (operation) {
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture: {
    auto stream = payload->args.hipStreamBeginCapture.stream;
    impl->streamToCaptureCount[stream] = 0;
    impl->streamToCapture[stream] = true;
    impl->activeCaptureCount.fetch_add(1, std::memory_order_release);
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture: {
    auto stream = payload->args.hipStreamEndCapture.stream;
    auto graph = *(payload->args.hipStreamEndCapture.pGraph);
    uint32_t captured = impl->streamToCaptureCount.contain(stream)
                            ? impl->streamToCaptureCount[stream]
                            : 0;
    impl->graphToNumInstances[graph] = captured;
    impl->streamToCapture.erase(stream);
    impl->streamToCaptureCount.erase(stream);
    impl->activeCaptureCount.fetch_sub(1, std::memory_order_release);
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags: {
    auto graph = payload->args.hipGraphInstantiateWithFlags.graph;
    auto graphExec = *(payload->args.hipGraphInstantiateWithFlags.pGraphExec);
    impl->graphExecToGraph[graphExec] = graph;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate: {
    auto graph = payload->args.hipGraphInstantiate.graph;
    auto graphExec = *(payload->args.hipGraphInstantiate.pGraphExec);
    impl->graphExecToGraph[graphExec] = graph;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecDestroy: {
    auto graphExec = payload->args.hipGraphExecDestroy.graphExec;
    impl->graphExecToGraph.erase(graphExec);
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDestroy: {
    auto graph = payload->args.hipGraphDestroy.graph;
    impl->graphToNumInstances.erase(graph);
    break;
  }
  default:
    break;
  }

  // Count kernel launches during graph capture. The atomic fast-check avoids
  // acquiring the shared_mutex on streamToCapture for every kernel launch
  // when no capture is active (the overwhelmingly common case).
  if (isKernelOp &&
      impl->activeCaptureCount.load(std::memory_order_acquire) > 0) {
    hipStream_t stream = nullptr;
    switch (operation) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
      stream = payload->args.hipLaunchKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
      stream = payload->args.hipExtLaunchKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
      stream = payload->args.hipLaunchCooperativeKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
      stream = payload->args.hipModuleLaunchKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
      stream = payload->args.hipModuleLaunchCooperativeKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
      stream = payload->args.hipExtModuleLaunchKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
      stream = payload->args.hipHccModuleLaunchKernel.stream;
      break;
    default:
      break;
    }
    if (stream && impl->streamToCapture.contain(stream))
      impl->streamToCaptureCount[stream]++;
  }

  if (isKernelOp) {
    threadState.exitOp();
    profiler.correlation.submit(record.correlation_id.internal);
  }
}

// ---- ROCTx marker callback via rocprofiler-sdk ----
//
// Prefer rocprofiler-sdk marker tracing for ROCTx events. Some PyTorch/ROCm
// environments load the legacy libroctx64 provider for torch.cuda.nvtx calls
// without making its symbols globally visible, so MARKER_CORE_API alone does
// not see those ranges. registerRoctxCallback below attaches to the loaded
// legacy provider when present.

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::markerCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API)
    return;
  if (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER)
    return;
  if (!getRuntimeState().nvtxEnabled.load(std::memory_order_relaxed))
    return;

  auto op = static_cast<rocprofiler_tracing_operation_t>(record.operation);
  if (op == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA) {
    auto *payload =
        static_cast<rocprofiler_callback_tracing_marker_api_data_t *>(
            record.payload);
    threadState.enterScope(payload->args.roctxRangePushA.message);
  } else if (op == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop) {
    threadState.exitScope();
  }
}

// Legacy libroctx64.so callback — kept as fallback for environments where
// librocprofiler-sdk-roctx.so is not loaded (e.g. bare ROCm without TheRock).
void RocprofSDKProfiler::RocprofSDKProfilerPimpl::roctxCallback(
    uint32_t operationId, void *data) {
  auto *apiData = static_cast<roctx_api_data_t *>(data);
  if (operationId == ROCTX_API_ID_roctxRangePushA) {
    threadState.enterScope(apiData->args.roctxRangePushA.message);
  } else if (operationId == ROCTX_API_ID_roctxRangePop) {
    threadState.exitScope();
  }
}

namespace {
int roctxTracerCallback(uint32_t /*domain*/, uint32_t operationId, void *data) {
  RocprofSDKProfiler::RocprofSDKProfilerPimpl::roctxCallback(operationId, data);
  return 0;
}

void registerRoctxCallback(bool enable) {
  // torch.cuda.nvtx may route through a locally loaded libroctx64.so. In that
  // case dlsym(RTLD_DEFAULT, "roctxRegisterTracerCallback") does not find the
  // callback registration entry point, but resolving it from the library handle
  // does.
  void *roctxLib = dlopen("libroctx64.so", RTLD_NOLOAD | RTLD_NOW);
  if (!roctxLib)
    return;
  auto *fn = reinterpret_cast<RoctxRegisterTracerCallbackFn>(
      dlsym(roctxLib, "roctxRegisterTracerCallback"));
  dlclose(roctxLib);
  if (!fn)
    return;
  fn(enable ? &roctxTracerCallback : nullptr);
}
} // namespace

// ---- Code object callback (kernel_id -> name mapping) ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::codeObjectCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
      record.operation !=
          ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
      record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) {
    return;
  }
  auto *impl = static_cast<RocprofSDKProfilerPimpl *>(arg);
  if (!impl)
    return;
  auto *payload = static_cast<
      rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *>(
      record.payload);
  impl->setKernelName(payload->kernel_id, payload->kernel_name);
}

// ---- Kernel dispatch buffer callback ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::kernelBufferCallback(
    rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer,
    rocprofiler_record_header_t **headers, size_t numHeaders, void *userData,
    uint64_t dropCount) {
  if (dropCount > 0) {
    std::cerr << "[PROTON] ROCProfiler-SDK dropped " << dropCount
              << " kernel dispatch records" << std::endl;
  }
  auto &profiler = threadState.profiler;
  auto *impl = static_cast<RocprofSDKProfilerPimpl *>(profiler.pImpl.get());
  auto &correlation = profiler.correlation;

  uint64_t maxCorrelationId = 0;
  std::map<Data *, std::pair<size_t, size_t>> dataPhases;

  for (size_t i = 0; i < numHeaders; ++i) {
    auto *header = headers[i];
    if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
        header->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
      continue;
    }
    auto *record =
        static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(
            header->payload);
    maxCorrelationId =
        std::max(maxCorrelationId, record->correlation_id.internal);
    auto kernelName = impl->getKernelName(record->dispatch_info.kernel_id);
    uint64_t streamId =
        static_cast<uint64_t>(record->dispatch_info.queue_id.handle);
    impl->corrIdToStreamId.withRead(
        record->correlation_id.internal,
        [&](const uint64_t &sid) { streamId = sid; });
    processKernelRecord(profiler, correlation.corrIdToExternId,
                        correlation.externIdToState, impl->corrIdToIsHipGraph,
                        dataPhases, kernelName, record, streamId);
    impl->corrIdToStreamId.erase(record->correlation_id.internal);
  }
  profiler.flushDataPhases(dataPhases, profiler.pendingGraphPool.get());
  if (maxCorrelationId > 0) {
    correlation.complete(maxCorrelationId);
  }
}

// ---- SDK tool init / fini (called by rocprofiler_force_configure) ----

namespace {

int protonToolInit(rocprofiler_client_finalize_t finiFunc, void *toolData) {
  auto *state = static_cast<RocprofilerRuntimeState *>(toolData);
  state->finalizeFunc = finiFunc;

  // Context 1: lightweight, always-active context for code object tracking.
  // Captures kernel_id -> name mappings as kernels are compiled.
  rocprofiler::createContext<true>(&state->codeObjectContext);

  const rocprofiler_tracing_operation_t codeObjectOps[] = {
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};
  rocprofiler::configureCallbackTracingService<true>(
      state->codeObjectContext, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
      codeObjectOps, 1,
      &RocprofSDKProfiler::RocprofSDKProfilerPimpl::codeObjectCallback,
      static_cast<void *>(state->pimpl));

  // Context 2: on-demand profiling context for HIP callback tracing and
  // kernel dispatch buffer tracing. Started/stopped in doStart()/doStop().
  // Registering BUFFER_TRACING_KERNEL_DISPATCH here causes
  // enable_queue_intercept() to install HSA queue hooks at force_configure
  // time, even though the context is not yet active.
  rocprofiler::createContext<true>(&state->profilingContext);

  // Subscribe only to the HIP operations Proton needs: kernel launches,
  // graph capture/instantiate/destroy. Passing nullptr/0 would subscribe to
  // all ~519 HIP runtime APIs, causing the SDK to construct correlation IDs
  // and invoke our callback for every hipMalloc, hipMemcpy, etc.
  constexpr rocprofiler_tracing_operation_t kTracedHipOps[] = {
      // Kernel launches (ENTER: correlation tracking, EXIT: capture counting)
      ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch,
      // Graph capture (EXIT only)
      ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture,
      // Graph instantiate (EXIT only)
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags,
      // Graph cleanup (EXIT only)
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecDestroy,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDestroy,
  };

  rocprofiler::configureCallbackTracingService<true>(
      state->profilingContext, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
      kTracedHipOps, std::size(kTracedHipOps),
      &RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipRuntimeCallback,
      nullptr);

  // Marker tracing: always configure MARKER_CORE_API so we intercept roctx
  // calls that go through librocprofiler-sdk-roctx.so (TheRock/torch
  // environments where the SDK's roctx interposes the global symbol).
  // This is configured unconditionally because force_configure may run before
  // torch loads the SDK's roctx library, and we can't add tracing services
  // after startContext. If the SDK's roctx isn't loaded, these callbacks
  // simply never fire. The legacy libroctx64.so callback registration in
  // doStart()/doStop() handles environments where only libroctx64.so is used.
  {
    constexpr rocprofiler_tracing_operation_t kMarkerOps[] = {
        ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
        ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
    };
    rocprofiler::configureCallbackTracingService<true>(
        state->profilingContext, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
        kMarkerOps, std::size(kMarkerOps),
        &RocprofSDKProfiler::RocprofSDKProfilerPimpl::markerCallback, nullptr);
  }

  // Flush the buffer when it reaches 87.5% capacity, leaving headroom for
  // in-flight records while the callback drains the buffer.
  size_t watermark = BufferSize - (BufferSize / 8);
  rocprofiler::createBuffer<true>(
      state->profilingContext, BufferSize, watermark,
      ROCPROFILER_BUFFER_POLICY_LOSSLESS,
      &RocprofSDKProfiler::RocprofSDKProfilerPimpl::kernelBufferCallback,
      nullptr, &state->kernelBuffer);

  rocprofiler::configureBufferTracingService<true>(
      state->profilingContext, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
      nullptr, 0, state->kernelBuffer);

  rocprofiler::createCallbackThread<true>(&state->callbackThread);
  rocprofiler::assignCallbackThread<true>(state->kernelBuffer,
                                          state->callbackThread);

  AgentIdMapper::instance().initialize();

  // Start the code object context now so the upcoming
  // invoke_register_propagation() replay of already-loaded code objects
  // triggers our callback while it's active.
  rocprofiler::startContext<false>(state->codeObjectContext);
  state->codeObjectStarted = true;

  state->configured = true;
  return 0;
}

void protonToolFini(void *toolData) {
  auto *state = static_cast<RocprofilerRuntimeState *>(toolData);
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->profilingStarted) {
      rocprofiler::stopContext<false>(state->profilingContext);
      state->profilingStarted = false;
    }
    if (state->codeObjectStarted) {
      rocprofiler::stopContext<false>(state->codeObjectContext);
      state->codeObjectStarted = false;
    }
  }
  rocprofiler::flushBuffer<false>(state->kernelBuffer);
  if (state->finalizeFunc && state->clientId) {
    state->finalizeFunc(*state->clientId);
  }
}

rocprofiler_tool_configure_result_t *
protonConfigure(uint32_t version, const char *runtimeVersion, uint32_t priority,
                rocprofiler_client_id_t *id) {
  auto &state = getRuntimeState();
  id->name = "ProtonRocprofSDK";
  state.clientId = id;
  static rocprofiler_tool_configure_result_t config{
      sizeof(rocprofiler_tool_configure_result_t), &protonToolInit,
      &protonToolFini, static_cast<void *>(&state)};
  return &config;
}

} // namespace

// ---- Profiler lifecycle ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::doStart() {
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.profilingStarted) {
    rocprofiler::startContext<true>(state.profilingContext);
    state.profilingStarted = true;
  }
  bool nvtx = getBoolEnv("TRITON_ENABLE_NVTX", true);
  state.nvtxEnabled.store(nvtx, std::memory_order_relaxed);
  if (nvtx)
    registerRoctxCallback(true);
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::doFlush() {
  auto &state = getRuntimeState();
  std::ignore = hip::deviceSynchronize<true>();
  profiler.correlation.flush(
      /*maxRetries=*/100, /*sleepUs=*/10,
      [&state]() { rocprofiler::flushBuffer<true>(state.kernelBuffer); });
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::doStop() {
  auto &state = getRuntimeState();
  state.nvtxEnabled.store(false, std::memory_order_relaxed);
  registerRoctxCallback(false);
  // Keep the profiling context running. rocprofiler-sdk does not reliably
  // re-intercept HIP runtime API calls after a stopContext→startContext
  // cycle on the same context. The correlation ID mechanism ensures that
  // kernel dispatch records without a matching active session are discarded.
}

RocprofSDKProfiler::RocprofSDKProfiler() {
  pImpl = std::make_unique<RocprofSDKProfilerPimpl>(*this);
  auto &state = getRuntimeState();
  state.pimpl = static_cast<RocprofSDKProfilerPimpl *>(pImpl.get());
  // Configure rocprofiler-sdk as soon as this singleton is constructed.
  // Deferring until doStart() is unsafe: any code that fully initializes HSA
  // beforehand (e.g. triton's HIP driver query at pytest collection time,
  // or a torch import chain) causes rocprofiler-sdk 1.2.0 to silently skip
  // kernel-dispatch buffer tracing installation on already-existing queues,
  // producing an empty dispatch buffer and no per-kernel timing data.
  // Construction of this singleton is triggered at libproton.so load time
  // via the __attribute__((constructor)) hook below, so force_configure
  // lands before any user code touches the HIP/HSA runtimes.
  if (!state.configured) {
    rocprofiler::forceConfigure<true>(&protonConfigure);
  }
}

RocprofSDKProfiler::~RocprofSDKProfiler() = default;

namespace {
// Runs during dlopen of libproton.so (i.e. `import triton.profiler._C`).
// Touches the singleton so its constructor — which calls
// rocprofiler_force_configure — runs before any Python code executes.
// Wrapped in try/catch so non-ROCm environments (where
// librocprofiler-sdk.so cannot be dlopen'd) continue to import cleanly;
// a subsequent attempt to start a "rocprofiler" session will surface the
// error through the normal lazy-dispatch path.
__attribute__((constructor)) void protonRocprofSDKLoadHook() {
  try {
    (void)RocprofSDKProfiler::instance();
  } catch (...) {
    // Intentionally swallowed: non-ROCm or rocprofiler-sdk unavailable.
  }
}
} // namespace

void RocprofSDKProfiler::doSetMode(
    const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions.empty() ? std::string() : modeAndOptions[0];
  if (proton::toLower(mode) == "periodic_flushing") {
    detail::setPeriodicFlushingMode(periodicFlushingEnabled,
                                    periodicFlushingFormat, modeAndOptions,
                                    "RocprofSDKProfiler");
  } else if (!mode.empty()) {
    throw std::invalid_argument(
        "[PROTON] RocprofSDKProfiler: unsupported mode: " + mode);
  }
}

} // namespace proton
