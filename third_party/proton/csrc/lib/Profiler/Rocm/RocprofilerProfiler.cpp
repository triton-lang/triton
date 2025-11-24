#include "Profiler/Rocm/RocprofilerProfiler.h"

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/HipApi.h"
#include "Driver/GPU/RocprofilerApi.h"
#include "Profiler/GPUProfiler.h"
#include "Utility/Env.h"
#include "Utility/Map.h"
#include "Utility/Singleton.h"

#include "hip/hip_runtime_api.h"
#include "rocprofiler-sdk/agent.h"
#include "rocprofiler-sdk/buffer_tracing.h"
#include "rocprofiler-sdk/callback_tracing.h"
#include "rocprofiler-sdk/hip/api_args.h"
#include "rocprofiler-sdk/hip/runtime_api_id.h"
#include "rocprofiler-sdk/marker/api_args.h"
#include "rocprofiler-sdk/marker/api_id.h"
#include "rocprofiler-sdk/registration.h"

#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

namespace proton {

template <>
thread_local GPUProfiler<RocprofilerProfiler>::ThreadState
    GPUProfiler<RocprofilerProfiler>::threadState(
        RocprofilerProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<RocprofilerProfiler>::Correlation::externIdQueue{};

namespace {

constexpr size_t BufferSize = 64 * 1024 * 1024;
constexpr const char *UnknownKernelName = "<unknown>";

struct RocprofilerRuntimeState {
  std::mutex mutex;
  rocprofiler_context_id_t context{};
  rocprofiler_buffer_id_t kernelBuffer{};
  rocprofiler_callback_thread_t callbackThread{};
  rocprofiler_client_finalize_t finalizeFunc = nullptr;
  rocprofiler_client_id_t *clientId{nullptr};
  bool configured{false};
  bool markerCallbacksEnabled{false};
  bool started{false};
};

RocprofilerRuntimeState &getRuntimeState() {
  static RocprofilerRuntimeState state;
  return state;
}

std::once_flag configureOnce;

void ensureRocprofilerConfigured();

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
    for (size_t i = 0; i < count; ++i) {
      const auto *agent = agentList[i];
      if (agent->type == ROCPROFILER_AGENT_TYPE_GPU) {
        self->agentToDevice[agent->id.handle] =
            static_cast<uint32_t>(agent->logical_node_type_id);
      }
    }
    return ROCPROFILER_STATUS_SUCCESS;
  }

  std::once_flag initializeFlag;
  std::unordered_map<uint64_t, uint32_t> agentToDevice;
};

std::shared_ptr<Metric> convertDispatchToMetric(
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record) {
  if (record->start_timestamp >= record->end_timestamp)
    return nullptr;
  auto deviceId = static_cast<uint64_t>(
      AgentIdMapper::instance().map(record->dispatch_info.agent_id.handle));
  return std::make_shared<KernelMetric>(
      static_cast<uint64_t>(record->start_timestamp),
      static_cast<uint64_t>(record->end_timestamp), 1, deviceId,
      static_cast<uint64_t>(DeviceType::HIP),
      static_cast<uint64_t>(record->dispatch_info.queue_id.handle));
}

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

void ensureRocprofilerConfigured() {
  std::call_once(configureOnce, []() {
    int status = 0;
    rocprofiler::isInitialized<true>(&status);
    if (status > 0) {
      throw std::runtime_error(
          "[PROTON] ROCProfiler-SDK is already configured by another tool");
    }
    auto ret = rocprofiler::forceConfigure<true>(&rocprofiler_configure);
    if (ret != ROCPROFILER_STATUS_SUCCESS) {
      throw std::runtime_error(
          "[PROTON] Failed to configure ROCProfiler-SDK runtime");
    }
  });

  if (!getRuntimeState().configured) {
    throw std::runtime_error(
        "[PROTON] ROCProfiler-SDK runtime is not initialized");
  }
}

} // namespace

struct RocprofilerProfiler::RocprofilerProfilerPimpl
    : public GPUProfiler<RocprofilerProfiler>::GPUProfilerPimplInterface {
  RocprofilerProfilerPimpl(RocprofilerProfiler &profiler)
      : GPUProfiler<RocprofilerProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~RocprofilerProfilerPimpl() = default;

  void doStart() override {
    ensureRocprofilerConfigured();
    auto &state = getRuntimeState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.started) {
      rocprofiler::startContext<true>(state.context);
      state.started = true;
    }
  }

  void doFlush() override {
    ensureRocprofilerConfigured();
    auto &state = getRuntimeState();
    std::ignore = hip::deviceSynchronize<true>();
    profiler.correlation.flush(
        /*maxRetries=*/100, /*sleepMs=*/10,
        /*flushFn=*/
        [&state]() { rocprofiler::flushBuffer<true>(state.kernelBuffer); });
  }

  void doStop() override {
    auto &state = getRuntimeState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.started) {
      rocprofiler::stopContext<true>(state.context);
      state.started = false;
    }
  }

  static void hipRuntimeCallback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t *userData, void *arg);
  static void markerCallback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t *userData, void *arg);
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
    if (kernelNames.contain(kernelId))
      return kernelNames[kernelId];
    return UnknownKernelName;
  }

  void setKernelName(uint64_t kernelId, const char *name) {
    if (name == nullptr)
      return;
    kernelNames[kernelId] = std::string(name);
  }

  ThreadSafeMap<uint64_t, bool, std::unordered_map<uint64_t, bool>>
      CorrIdToIsHipGraph;

  ThreadSafeMap<hipGraphExec_t, hipGraph_t,
                std::unordered_map<hipGraphExec_t, hipGraph_t>>
      GraphExecToGraph;

  ThreadSafeMap<hipGraph_t, uint32_t, std::unordered_map<hipGraph_t, uint32_t>>
      GraphToNumInstances;

  ThreadSafeMap<hipStream_t, uint32_t,
                std::unordered_map<hipStream_t, uint32_t>>
      StreamToCaptureCount;

  ThreadSafeMap<hipStream_t, bool, std::unordered_map<hipStream_t, bool>>
      StreamToCapture;

  KernelNameMap kernelNames;

private:
  static void processKernelRecord(
      RocprofilerProfiler &profiler, RocprofilerProfilerPimpl &impl,
      const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record);
};

namespace {} // namespace

void RocprofilerProfiler::RocprofilerProfilerPimpl::hipRuntimeCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API)
    return;

  auto operation =
      static_cast<rocprofiler_tracing_operation_t>(record.operation);
  bool isKernelOp = isKernelLaunchOperation(operation);
  auto &profiler = RocprofilerProfiler::instance();
  auto *impl = static_cast<RocprofilerProfiler::RocprofilerProfilerPimpl *>(
      profiler.pImpl.get());
  auto *payload = static_cast<rocprofiler_callback_tracing_hip_api_data_t *>(
      record.payload);

  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    if (!isKernelOp)
      return;
    threadState.enterOp();
    size_t numInstances = 1;
    if (operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch) {
      impl->CorrIdToIsHipGraph[record.correlation_id.internal] = true;
      numInstances = std::numeric_limits<size_t>::max();
      bool foundGraph = false;
      auto graphExec = payload->args.hipGraphLaunch.graphExec;
      if (impl->GraphExecToGraph.contain(graphExec)) {
        auto graph = impl->GraphExecToGraph[graphExec];
        if (impl->GraphToNumInstances.contain(graph)) {
          numInstances = impl->GraphToNumInstances[graph];
          foundGraph = true;
        }
      }
      if (!foundGraph) {
        std::cerr
            << "[PROTON] Unable to determine hipGraph kernel count. Start "
               "profiling before creating graphs to avoid leaks."
            << std::endl;
      }
    }
    profiler.correlation.correlate(record.correlation_id.internal,
                                   numInstances);
    return;
  }

  if (record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT)
    return;

  switch (operation) {
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture: {
    auto stream = payload->args.hipStreamBeginCapture.stream;
    impl->StreamToCapture[stream] = true;
    impl->StreamToCaptureCount[stream] = 0;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture: {
    auto stream = payload->args.hipStreamEndCapture.stream;
    auto graph = *(payload->args.hipStreamEndCapture.pGraph);
    uint32_t captured = impl->StreamToCaptureCount.contain(stream)
                            ? impl->StreamToCaptureCount[stream]
                            : 0;
    impl->GraphToNumInstances[graph] = captured;
    impl->StreamToCapture.erase(stream);
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel: {
    auto stream = payload->args.hipLaunchKernel.stream;
    if (impl->StreamToCapture.contain(stream))
      impl->StreamToCaptureCount[stream]++;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel: {
    auto stream = payload->args.hipExtLaunchKernel.stream;
    if (impl->StreamToCapture.contain(stream))
      impl->StreamToCaptureCount[stream]++;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel: {
    auto stream = payload->args.hipLaunchCooperativeKernel.stream;
    if (impl->StreamToCapture.contain(stream))
      impl->StreamToCaptureCount[stream]++;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel: {
    auto stream = payload->args.hipModuleLaunchKernel.stream;
    if (impl->StreamToCapture.contain(stream))
      impl->StreamToCaptureCount[stream]++;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel: {
    auto stream = payload->args.hipModuleLaunchCooperativeKernel.stream;
    if (impl->StreamToCapture.contain(stream))
      impl->StreamToCaptureCount[stream]++;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags: {
    auto graph = payload->args.hipGraphInstantiateWithFlags.graph;
    auto graphExec = *(payload->args.hipGraphInstantiateWithFlags.pGraphExec);
    impl->GraphExecToGraph[graphExec] = graph;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate: {
    auto graph = payload->args.hipGraphInstantiate.graph;
    auto graphExec = *(payload->args.hipGraphInstantiate.pGraphExec);
    impl->GraphExecToGraph[graphExec] = graph;
    break;
  }
  default:
    break;
  }

  if (isKernelOp) {
    threadState.exitOp();
    profiler.correlation.submit(record.correlation_id.internal);
  }
}

void RocprofilerProfiler::RocprofilerProfilerPimpl::markerCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API)
    return;
  auto *payload = static_cast<rocprofiler_callback_tracing_marker_api_data_t *>(
      record.payload);
  auto op = static_cast<rocprofiler_tracing_operation_t>(record.operation);
  if (op == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA &&
      record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    threadState.enterScope(payload->args.roctxRangePushA.message);
  } else if (op == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop &&
             record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
    threadState.exitScope();
  }
}

void RocprofilerProfiler::RocprofilerProfilerPimpl::codeObjectCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
      record.operation !=
          ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
      record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) {
    return;
  }
  auto *payload = static_cast<
      rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *>(
      record.payload);
  auto &profiler = RocprofilerProfiler::instance();
  auto *impl = static_cast<RocprofilerProfiler::RocprofilerProfilerPimpl *>(
      profiler.pImpl.get());
  impl->setKernelName(payload->kernel_id, payload->kernel_name);
}

void RocprofilerProfiler::RocprofilerProfilerPimpl::kernelBufferCallback(
    rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer,
    rocprofiler_record_header_t **headers, size_t numHeaders, void *userData,
    uint64_t dropCount) {
  if (dropCount > 0) {
    std::cerr << "[PROTON] ROCProfiler-SDK dropped " << dropCount
              << " kernel dispatch records" << std::endl;
  }
  auto &profiler = RocprofilerProfiler::instance();
  auto *impl = static_cast<RocprofilerProfiler::RocprofilerProfilerPimpl *>(
      profiler.pImpl.get());
  uint64_t maxCorrelationId = 0;
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
    RocprofilerProfiler::RocprofilerProfilerPimpl::processKernelRecord(
        profiler, *impl, record);
  }
  if (maxCorrelationId > 0) {
    profiler.correlation.complete(maxCorrelationId);
  }
}

void RocprofilerProfiler::RocprofilerProfilerPimpl::processKernelRecord(
    RocprofilerProfiler &profiler, RocprofilerProfilerPimpl &impl,
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record) {
  auto metric = convertDispatchToMetric(record);
  if (!metric)
    return;

  auto &correlation = profiler.correlation;
  auto hasCorrelation =
      correlation.corrIdToExternId.contain(record->correlation_id.internal);
  auto externId =
      hasCorrelation
          ? correlation.corrIdToExternId[record->correlation_id.internal].first
          : Scope::DummyScopeId;
  auto isAPI = correlation.apiExternIds.contain(externId);
  bool isGraph =
      impl.CorrIdToIsHipGraph.contain(record->correlation_id.internal);

  auto dataSet = profiler.getDataSet();
  if (externId == Scope::DummyScopeId)
    isAPI = false;

  auto kernelName = impl.getKernelName(record->dispatch_info.kernel_id);

  if (!isGraph) {
    for (auto *data : dataSet) {
      auto scopeId = externId;
      if (isAPI) {
        scopeId = data->addOp(externId, kernelName);
      }
      data->addMetric(scopeId, metric);
    }
  } else {
    for (auto *data : dataSet) {
      auto childId = data->addOp(externId, kernelName);
      data->addMetric(childId, metric);
    }
  }

  if (hasCorrelation) {
    auto &[parentId, remaining] =
        correlation.corrIdToExternId[record->correlation_id.internal];
    if (remaining > 1) {
      correlation.corrIdToExternId[record->correlation_id.internal].second =
          remaining - 1;
    } else {
      correlation.corrIdToExternId.erase(record->correlation_id.internal);
    }
  } else {
    correlation.apiExternIds.erase(externId);
  }

  if (isGraph) {
    impl.CorrIdToIsHipGraph.erase(record->correlation_id.internal);
  }
}

RocprofilerProfiler::RocprofilerProfiler() {
  pImpl = std::make_unique<RocprofilerProfilerPimpl>(*this);
  ensureRocprofilerConfigured();
}

RocprofilerProfiler::~RocprofilerProfiler() = default;

void RocprofilerProfiler::doSetMode(
    const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions.empty() ? std::string() : modeAndOptions[0];
  if (!mode.empty()) {
    throw std::invalid_argument(
        "[PROTON] RocprofilerProfiler: unsupported mode: " + mode);
  }
}

extern "C" {

rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t version, const char *runtimeVersion,
                      uint32_t priority, rocprofiler_client_id_t *id);

} // extern "C"

namespace {

int proton_tool_init(rocprofiler_client_finalize_t finiFunc, void *toolData) {
  auto *state = static_cast<RocprofilerRuntimeState *>(toolData);
  state->finalizeFunc = finiFunc;

  rocprofiler::createContext<true>(&state->context);

  bool enableMarkers = getBoolEnv("TRITON_ENABLE_NVTX", true);
  state->markerCallbacksEnabled = enableMarkers;

  rocprofiler::configureCallbackTracingService<true>(
      state->context, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, nullptr, 0,
      &RocprofilerProfiler::RocprofilerProfilerPimpl::hipRuntimeCallback,
      nullptr);

  if (enableMarkers) {
    rocprofiler::configureCallbackTracingService<true>(
        state->context, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API, nullptr,
        0, &RocprofilerProfiler::RocprofilerProfilerPimpl::markerCallback,
        nullptr);
  }

  const rocprofiler_tracing_operation_t codeObjectOps[] = {
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};
  rocprofiler::configureCallbackTracingService<true>(
      state->context, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, codeObjectOps,
      1, &RocprofilerProfiler::RocprofilerProfilerPimpl::codeObjectCallback,
      nullptr);

  size_t watermark = BufferSize - (BufferSize / 8);
  rocprofiler::createBuffer<true>(
      state->context, BufferSize, watermark, ROCPROFILER_BUFFER_POLICY_LOSSLESS,
      &RocprofilerProfiler::RocprofilerProfilerPimpl::kernelBufferCallback,
      nullptr, &state->kernelBuffer);

  rocprofiler::configureBufferTracingService<true>(
      state->context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
      state->kernelBuffer);

  rocprofiler::createCallbackThread<true>(&state->callbackThread);
  rocprofiler::assignCallbackThread<true>(state->kernelBuffer,
                                          state->callbackThread);

  int valid = 0;
  rocprofiler::contextIsValid<true>(state->context, &valid);
  if (valid == 0) {
    return -1;
  }

  AgentIdMapper::instance().initialize();

  state->configured = true;
  return 0;
}

void proton_tool_fini(void *toolData) {
  auto *state = static_cast<RocprofilerRuntimeState *>(toolData);
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->started) {
      rocprofiler::stopContext<false>(state->context);
      state->started = false;
    }
  }
  rocprofiler::flushBuffer<false>(state->kernelBuffer);
  if (state->finalizeFunc && state->clientId) {
    state->finalizeFunc(*state->clientId);
  }
}

} // namespace

extern "C" rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t version, const char *runtimeVersion,
                      uint32_t priority, rocprofiler_client_id_t *id) {
  auto &state = getRuntimeState();
  id->name = "ProtonRocprofiler";
  state.clientId = id;
  static rocprofiler_tool_configure_result_t config{
      sizeof(rocprofiler_tool_configure_result_t), &proton_tool_init,
      &proton_tool_fini, static_cast<void *>(&state)};
  (void)version;
  (void)runtimeVersion;
  (void)priority;
  return &config;
}

} // namespace proton
