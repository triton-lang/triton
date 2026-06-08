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
#include "rocprofiler-sdk/external_correlation.h"
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
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace proton {

template <>
thread_local GPUProfiler<RocprofSDKProfiler>::ThreadState
    GPUProfiler<RocprofSDKProfiler>::threadState(
        RocprofSDKProfiler::instance());

namespace {

constexpr size_t BufferSize = 64 * 1024 * 1024;
constexpr const char *UnknownKernelName = "<unknown>";

struct RocprofSDKProfilerPimpl;

struct GraphDispatchCorrelation {
  size_t externId{Scope::DummyScopeId};
  uint64_t graphExecId{};
  uint64_t graphNodeId{};
};

struct ActiveGraphLaunch {
  size_t externId{Scope::DummyScopeId};
  uint64_t graphExecId{};
  uint64_t nextNodeId{};
};

thread_local std::vector<ActiveGraphLaunch> graphLaunchStack;
thread_local GraphState streamCaptureGraphState;

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
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt:
    stream = payload->args.hipGraphLaunch_spt.stream;
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
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice:
    return true;
  default:
    return false;
  }
}

bool isGraphLaunchOperation(rocprofiler_tracing_operation_t op) {
  return op == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch ||
         op == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt;
}

bool isStreamCaptureBeginOperation(rocprofiler_tracing_operation_t op) {
  return op == ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture ||
         op == ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture_spt;
}

bool isStreamCaptureEndOperation(rocprofiler_tracing_operation_t op) {
  return op == ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture ||
         op == ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture_spt;
}

// ---- Kernel dispatch processing (matches main's GPUProfiler interface) ----

void processKernelRecord(
    RocprofSDKProfiler &profiler,
    RocprofSDKProfiler::CorrIdToExternIdMap &corrIdToExternId,
    RocprofSDKProfiler::ExternIdToStateMap &externIdToState,
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

  DataToEntryMap dataToEntry;
  bool isMissingName = true;
  if (!externIdToState.withRead(
          externId, [&](const RocprofSDKProfiler::ExternIdState &state) {
            dataToEntry = state.dataToEntry;
            isMissingName = state.isMissingName;
          }))
    return;

  for (auto [data, entry] : dataToEntry) {
    if (auto metric = convertDispatchToMetric(record, streamId)) {
      if (isMissingName) {
        auto childEntry =
            data->addOp(entry.phase, entry.id, {Context(kernelName)});
        childEntry.upsertMetric(std::move(metric));
      } else {
        entry.upsertMetric(std::move(metric));
      }
      detail::updateDataPhases(dataPhases, data, entry.phase);
    }
  }

  bool complete = false;
  externIdToState.withWrite(externId,
                            [&](RocprofSDKProfiler::ExternIdState &state) {
                              if (state.numNodes > 0)
                                --state.numNodes;
                              complete = state.numNodes == 0;
                            });
  if (complete) {
    corrIdToExternId.erase(record->correlation_id.internal);
    externIdToState.erase(externId);
  }
}

bool graphStateContainsData(const GraphState &graphState, Data *data) {
  return std::any_of(graphState.nodeIdToState.begin(),
                     graphState.nodeIdToState.end(),
                     [data](const auto &nodeIt) {
                       return nodeIt.second.dataToEntryId.find(data) !=
                              nodeIt.second.dataToEntryId.end();
                     });
}

RocprofSDKProfiler::ExternIdState *
buildGraphNodeEntries(const DataToEntryMap &dataToEntry, GraphState &graphState,
                      RocprofSDKProfiler::ExternIdToStateMap &externIdToState,
                      size_t externId) {
  if (dataToEntry.empty())
    return nullptr;

  auto &externIdState = externIdToState[externId];
  for (auto &[data, entry] : dataToEntry) {
    if (!graphStateContainsData(graphState, data))
      continue;
    externIdState.dataToGraphEntry.insert({data, entry});
  }
  externIdState.nodeIdToState = &graphState.nodeIdToState;
  return &externIdState;
}

void recordGraphKernelNode(const std::set<Data *> &dataSet,
                           GraphState &graphState, uint64_t nodeId,
                           const std::string &name, bool isMissingName) {
  auto &nodeState = graphState.nodeIdToState[nodeId];
  nodeState.nodeId = nodeId;
  if (isMissingName)
    nodeState.status.setMissingName();

  for (auto *data : dataSet) {
    auto currentContexts = data->getContexts();
    std::vector<Context> contexts;
    contexts.emplace_back(GraphState::captureTag);
    for (const auto &context : currentContexts) {
      contexts.push_back(context);
    }
    contexts.emplace_back(name);
    auto staticEntry =
        data->addOp(Data::kVirtualPhase, Data::kRootEntryId, contexts);
    nodeState.dataToEntryId.insert_or_assign(data, staticEntry.id);
    graphState.dataToEntryIdToNodeStates[data][staticEntry.id].insert(
        &nodeState);
  }
}

bool processGraphKernelRecord(
    RocprofSDKProfiler::ExternIdToStateMap &externIdToState,
    std::map<Data *, std::pair<size_t, size_t>> &dataPhases,
    const std::string &kernelName,
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record,
    const GraphDispatchCorrelation &graphCorrelation, uint64_t streamId) {
  auto externId = graphCorrelation.externId;
  if (externId == Scope::DummyScopeId)
    return false;
  auto externStateRef = externIdToState.find(externId);
  if (!externStateRef)
    return false;
  auto &externState = externStateRef->get();
  auto *nodeIdToState = externState.nodeIdToState;
  bool handledByGraphState = false;
  if (nodeIdToState) {
    auto nodeStateIt = nodeIdToState->find(graphCorrelation.graphNodeId);
    if (nodeStateIt != nodeIdToState->end()) {
      handledByGraphState = true;
      const auto &nodeState = nodeStateIt->second;
      if (nodeState.status.isMissingName()) {
        throw std::runtime_error(
            "[PROTON] Kernel name is missing for a graph node.");
      }
      for (auto &[data, entry] : externState.dataToGraphEntry) {
        auto targetEntryIdIt = nodeState.dataToEntryId.find(data);
        if (targetEntryIdIt == nodeState.dataToEntryId.end())
          continue;
        if (auto metric = convertDispatchToMetric(record, streamId)) {
          entry.upsertLinkedMetric(std::move(metric), targetEntryIdIt->second);
          detail::updateDataPhases(dataPhases, data, entry.phase);
        }
      }
    }
  }
  if (!handledByGraphState) {
    // This can happen when graph creation was not captured by Proton. Since we
    // do not have per-node static metadata, attach replay kernels under the
    // graph launch entry.
    for (auto [data, entry] : externState.dataToEntry) {
      if (auto metric = convertDispatchToMetric(record, streamId)) {
        auto childEntry =
            data->addOp(entry.phase, entry.id, {Context(kernelName)});
        childEntry.upsertMetric(std::move(metric));
        detail::updateDataPhases(dataPhases, data, childEntry.phase);
      }
    }
  }
  bool complete = false;
  externIdToState.withWrite(externId,
                            [&](RocprofSDKProfiler::ExternIdState &state) {
                              if (state.numNodes > 0)
                                --state.numNodes;
                              complete = state.numNodes == 0;
                            });
  if (complete)
    externIdToState.erase(externId);
  return complete;
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
  static void
  handleRuntimeEnter(rocprofiler_callback_tracing_record_t record,
                     rocprofiler_tracing_operation_t operation,
                     rocprofiler_callback_tracing_hip_api_data_t *payload);
  static void
  handleRuntimeExit(rocprofiler_callback_tracing_record_t record,
                    rocprofiler_tracing_operation_t operation,
                    rocprofiler_callback_tracing_hip_api_data_t *payload);
  static void handleStreamCaptureBegin();
  static void
  handleCapturedKernelEnter(rocprofiler_tracing_operation_t operation);
  static void
  handleGraphLaunchEnter(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_tracing_operation_t operation,
                         rocprofiler_callback_tracing_hip_api_data_t *payload,
                         RocprofSDKProfilerPimpl *impl,
                         DataToEntryMap &dataToEntry, size_t &numInstances);
  static void handleSuccessfulRuntimeExit(
      rocprofiler_tracing_operation_t operation,
      rocprofiler_callback_tracing_hip_api_data_t *payload,
      RocprofSDKProfilerPimpl *impl);
  static void handleStreamCaptureEnd(rocprofiler_tracing_operation_t operation);
  static void handleKernelExit(rocprofiler_callback_tracing_record_t record,
                               rocprofiler_tracing_operation_t operation);
  static void markerCallback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t *userData, void *arg);
  static void hipGraphCallback(rocprofiler_callback_tracing_record_t record,
                               rocprofiler_user_data_t *userData, void *arg);
  static int externalCorrelationCallback(
      rocprofiler_thread_id_t threadId, rocprofiler_context_id_t contextId,
      rocprofiler_external_correlation_id_request_kind_t kind,
      rocprofiler_tracing_operation_t operation, uint64_t internalCorrelationId,
      rocprofiler_user_data_t *externalCorrelationId, void *userData);
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

  KernelNameMap kernelNames;

  // correlation_id → HIP stream pointer, captured at hipLaunchKernel ENTER.
  // Used to distinguish streams in trace output when the SDK's queue_id
  // maps multiple HIP streams to the same underlying HSA queue.
  ThreadSafeMap<uint64_t, uint64_t, std::unordered_map<uint64_t, uint64_t>>
      corrIdToStreamId;

  ThreadSafeMap<hipGraph_t, GraphState> graphToState;

  ThreadSafeMap<hipGraphExec_t, hipGraph_t,
                std::unordered_map<hipGraphExec_t, hipGraph_t>>
      graphExecToGraph;

  ThreadSafeMap<hipGraphExec_t, uint64_t,
                std::unordered_map<hipGraphExec_t, uint64_t>>
      graphExecToGraphExecId;

  ThreadSafeMap<uint64_t, GraphState> graphStates;
};

void tryBindGraphExecState(RocprofSDKProfiler::RocprofSDKProfilerPimpl *impl,
                           hipGraphExec_t graphExec) {
  if (!impl || !graphExec)
    return;

  uint64_t graphExecId = 0;
  if (!impl->graphExecToGraphExecId.withRead(
          graphExec, [&](const uint64_t &value) { graphExecId = value; }) ||
      graphExecId == 0) {
    return;
  }

  hipGraph_t graph = nullptr;
  if (!impl->graphExecToGraph.withRead(
          graphExec, [&](const hipGraph_t &value) { graph = value; }) ||
      graph == nullptr) {
    return;
  }

  GraphState graphState;
  if (!impl->graphToState.withRead(
          graph, [&](const GraphState &value) { graphState = value; })) {
    return;
  }
  impl->graphStates.insert(graphExecId, graphState);
}

// ---- HIP Runtime API callback (correlation tracking) ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::handleStreamCaptureBegin() {
  threadState.isStreamCapturing = true;
  streamCaptureGraphState = GraphState{};
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::handleCapturedKernelEnter(
    rocprofiler_tracing_operation_t operation) {
  if (!threadState.isStreamCapturing || isGraphLaunchOperation(operation))
    return;
  auto &profiler = threadState.profiler;
  if (profiler.isOpInProgress()) {
    auto &scope = threadState.scopeStack.back();
    const auto nodeId = streamCaptureGraphState.nodeIdToState.size();
    recordGraphKernelNode(profiler.dataSet, streamCaptureGraphState, nodeId,
                          scope.name, scope.name.empty());
  }
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::handleGraphLaunchEnter(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_tracing_operation_t operation,
    rocprofiler_callback_tracing_hip_api_data_t *payload,
    RocprofSDKProfilerPimpl *impl, DataToEntryMap &dataToEntry,
    size_t &numInstances) {
  hipGraphExec_t graphExec =
      operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch
          ? payload->args.hipGraphLaunch.graphExec
          : payload->args.hipGraphLaunch_spt.graphExec;
  uint64_t graphExecId = 0;
  impl->graphExecToGraphExecId.withRead(
      graphExec, [&](const uint64_t &value) { graphExecId = value; });
  if (graphExecId != 0 && impl->graphStates.contain(graphExecId)) {
    auto &graphState = impl->graphStates[graphExecId];
    numInstances = graphState.nodeIdToState.size();
    buildGraphNodeEntries(dataToEntry, graphState,
                          threadState.profiler.correlation.externIdToState,
                          threadState.scopeStack.back().scopeId);
  } else {
    std::cerr << "[PROTON] Cannot find graph state for graphExec "
                 "and graph replay attribution may be incomplete. "
                 "Please start profiling before the graph is captured."
              << std::endl;
  }
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::handleRuntimeEnter(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_tracing_operation_t operation,
    rocprofiler_callback_tracing_hip_api_data_t *payload) {
  if (isStreamCaptureBeginOperation(operation)) {
    handleStreamCaptureBegin();
    return;
  }

  if (!isKernelLaunchOperation(operation))
    return;

  auto &profiler = threadState.profiler;
  auto *impl = static_cast<RocprofSDKProfilerPimpl *>(profiler.pImpl.get());

  const char *resolvedName = resolveKernelNameAtEnter(operation, payload);
  std::string kernelName =
      resolvedName ? std::string(resolvedName) : std::string();
  threadState.enterOp(Scope(kernelName));
  auto &dataToEntry = threadState.dataToEntry;
  if (threadState.isStreamCapturing && !isGraphLaunchOperation(operation)) {
    handleCapturedKernelEnter(operation);
    return;
  }

  size_t numInstances = 1;
  if (isGraphLaunchOperation(operation)) {
    handleGraphLaunchEnter(record, operation, payload, impl, dataToEntry,
                           numInstances);
  }
  auto &scope = threadState.scopeStack.back();
  auto isMissingName = scope.name.empty();
  profiler.correlation.correlate(record.correlation_id.internal, scope.scopeId,
                                 numInstances, isMissingName, dataToEntry);
  impl->corrIdToStreamId[record.correlation_id.internal] =
      extractStreamId(operation, payload);
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::handleSuccessfulRuntimeExit(
    rocprofiler_tracing_operation_t operation,
    rocprofiler_callback_tracing_hip_api_data_t *payload,
    RocprofSDKProfilerPimpl *impl) {
  switch (operation) {
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture_spt: {
    auto graphPtr =
        operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture
            ? payload->args.hipStreamEndCapture.pGraph
            : payload->args.hipStreamEndCapture_spt.pGraph;
    if (graphPtr && *graphPtr) {
      impl->graphToState.insert(*graphPtr, streamCaptureGraphState);
    }
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithParams: {
    hipGraph_t graph = nullptr;
    hipGraphExec_t *graphExecPtr = nullptr;
    if (operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate) {
      graph = payload->args.hipGraphInstantiate.graph;
      graphExecPtr = payload->args.hipGraphInstantiate.pGraphExec;
    } else if (operation ==
               ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags) {
      graph = payload->args.hipGraphInstantiateWithFlags.graph;
      graphExecPtr = payload->args.hipGraphInstantiateWithFlags.pGraphExec;
    } else {
      graph = payload->args.hipGraphInstantiateWithParams.graph;
      graphExecPtr = payload->args.hipGraphInstantiateWithParams.pGraphExec;
    }
    if (graphExecPtr && *graphExecPtr) {
      impl->graphExecToGraph[*graphExecPtr] = graph;
      tryBindGraphExecState(impl, *graphExecPtr);
    }
    break;
  }
  default:
    break;
  }
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::handleStreamCaptureEnd(
    rocprofiler_tracing_operation_t operation) {
  if (isStreamCaptureEndOperation(operation)) {
    threadState.isStreamCapturing = false;
    streamCaptureGraphState = GraphState{};
  }
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::handleKernelExit(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_tracing_operation_t operation) {
  if (!isKernelLaunchOperation(operation))
    return;

  auto &profiler = threadState.profiler;
  if (threadState.isStreamCapturing && !isGraphLaunchOperation(operation)) {
    threadState.exitOp();
    return;
  }

  threadState.exitOp();
  profiler.correlation.submit(record.correlation_id.internal);
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::handleRuntimeExit(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_tracing_operation_t operation,
    rocprofiler_callback_tracing_hip_api_data_t *payload) {
  auto &profiler = threadState.profiler;
  auto *impl = static_cast<RocprofSDKProfilerPimpl *>(profiler.pImpl.get());

  if (payload && payload->retval.hipError_t_retval == hipSuccess) {
    handleSuccessfulRuntimeExit(operation, payload, impl);
  }
  handleStreamCaptureEnd(operation);
  handleKernelExit(record, operation);
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipRuntimeCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API)
    return;

  auto operation =
      static_cast<rocprofiler_tracing_operation_t>(record.operation);
  auto *payload = static_cast<rocprofiler_callback_tracing_hip_api_data_t *>(
      record.payload);

  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    handleRuntimeEnter(record, operation, payload);
    return;
  }

  if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
    handleRuntimeExit(record, operation, payload);
  }
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipGraphCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  auto *payload = static_cast<rocprofiler_callback_tracing_hip_graph_data_t *>(
      record.payload);
  auto *impl = static_cast<RocprofSDKProfilerPimpl *>(arg);
  auto graphExec = reinterpret_cast<hipGraphExec_t>(
      const_cast<void *>(payload->graph_exec_value.ptr));
  const auto graphExecId = payload->graph_exec_id.handle;

  if (record.operation ==
      ROCPROFILER_HIP_GRAPH_OPERATION_HIP_GRAPH_EXEC_CREATE) {
    impl->graphExecToGraphExecId[graphExec] = graphExecId;
    return;
  }

  if (record.operation ==
      ROCPROFILER_HIP_GRAPH_OPERATION_HIP_GRAPH_EXEC_DESTROY) {
    impl->graphExecToGraph.erase(graphExec);
    impl->graphExecToGraphExecId.erase(graphExec);
    return;
  }

  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    auto externId = threadState.scopeStack.empty()
                        ? Scope::DummyScopeId
                        : threadState.scopeStack.back().scopeId;
    if (impl->graphStates.contain(graphExecId) &&
        externId != Scope::DummyScopeId && !threadState.dataToEntry.empty()) {
      auto &graphState = impl->graphStates[graphExecId];
      auto &profiler = threadState.profiler;
      buildGraphNodeEntries(threadState.dataToEntry, graphState,
                            profiler.correlation.externIdToState, externId);
      profiler.correlation.externIdToState.withWrite(
          externId, [&](RocprofSDKProfiler::ExternIdState &state) {
            state.numNodes = graphState.nodeIdToState.size();
          });
    }
    graphLaunchStack.push_back(
        ActiveGraphLaunch{externId, graphExecId, /*nextNodeId=*/0});
  } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
    graphLaunchStack.pop_back();
  }
}

int RocprofSDKProfiler::RocprofSDKProfilerPimpl::externalCorrelationCallback(
    rocprofiler_thread_id_t threadId, rocprofiler_context_id_t contextId,
    rocprofiler_external_correlation_id_request_kind_t kind,
    rocprofiler_tracing_operation_t operation, uint64_t internalCorrelationId,
    rocprofiler_user_data_t *externalCorrelationId, void *userData) {
  externalCorrelationId->value = 0;
  if (kind != ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH)
    return 0;
  if (graphLaunchStack.empty())
    return 0;

  auto &graphLaunch = graphLaunchStack.back();
  auto externId = graphLaunch.externId;
  if (externId == Scope::DummyScopeId && !threadState.scopeStack.empty()) {
    externId = threadState.scopeStack.back().scopeId;
  }
  if (externId == Scope::DummyScopeId)
    return 0;
  if (!threadState.dataToEntry.empty()) {
    auto &profiler = threadState.profiler;
    auto isMissingName = threadState.scopeStack.empty()
                             ? true
                             : threadState.scopeStack.back().name.empty();
    profiler.correlation.externIdToState.upsert(
        externId, [&](RocprofSDKProfiler::ExternIdState &state) {
          if (!state.dataToEntry.empty())
            return;
          state.numNodes = 1;
          state.dataToEntry = threadState.dataToEntry;
          state.isMissingName = isMissingName;
        });
  }
  externalCorrelationId->ptr = new GraphDispatchCorrelation{
      externId, graphLaunch.graphExecId, graphLaunch.nextNodeId++};
  return 0;
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
    if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) {
      continue;
    }
    if (header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
      auto *record =
          static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(
              header->payload);
      maxCorrelationId =
          std::max(maxCorrelationId, record->correlation_id.internal);
      auto kernelName = impl->getKernelName(record->dispatch_info.kernel_id);
      if (record->correlation_id.external.ptr != nullptr) {
        auto *graphCorrelation = static_cast<GraphDispatchCorrelation *>(
            record->correlation_id.external.ptr);
        uint64_t streamId =
            static_cast<uint64_t>(record->dispatch_info.queue_id.handle);
        impl->corrIdToStreamId.withRead(
            record->correlation_id.internal,
            [&](const uint64_t &sid) { streamId = sid; });
        if (processGraphKernelRecord(correlation.externIdToState, dataPhases,
                                     kernelName, record, *graphCorrelation,
                                     streamId)) {
          correlation.corrIdToExternId.erase(record->correlation_id.internal);
          impl->corrIdToStreamId.erase(record->correlation_id.internal);
        }
        delete graphCorrelation;
        record->correlation_id.external.value = 0;
        continue;
      }
      uint64_t streamId =
          static_cast<uint64_t>(record->dispatch_info.queue_id.handle);
      impl->corrIdToStreamId.withRead(
          record->correlation_id.internal,
          [&](const uint64_t &sid) { streamId = sid; });
      processKernelRecord(profiler, correlation.corrIdToExternId,
                          correlation.externIdToState, dataPhases, kernelName,
                          record, streamId);
      impl->corrIdToStreamId.erase(record->correlation_id.internal);
    }
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

  // Subscribe only to the HIP operations Proton needs. Passing nullptr/0 would
  // subscribe to all ~519 HIP runtime APIs, causing the SDK to construct
  // correlation IDs and invoke our callback for every hipMalloc, hipMemcpy,
  // etc.
  constexpr rocprofiler_tracing_operation_t kTracedHipOps[] = {
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
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture_spt,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture_spt,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithParams,
  };

  rocprofiler::configureCallbackTracingService<true>(
      state->profilingContext, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
      kTracedHipOps, std::size(kTracedHipOps),
      &RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipRuntimeCallback,
      nullptr);

  rocprofiler::configureCallbackTracingService<true>(
      state->profilingContext, ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH, nullptr,
      0, &RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipGraphCallback,
      static_cast<void *>(state->pimpl));

  {
    constexpr rocprofiler_external_correlation_id_request_kind_t
        kExternalCorrelationKinds[] = {
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
        };
    rocprofiler::configureExternalCorrelationIdRequestService<true>(
        state->profilingContext, kExternalCorrelationKinds,
        std::size(kExternalCorrelationKinds),
        &RocprofSDKProfiler::RocprofSDKProfilerPimpl::
            externalCorrelationCallback,
        nullptr);
  }

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
