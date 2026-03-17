#include "Profiler/RocprofSDK/RocprofSDKProfiler.h"

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/HipApi.h"
#include "Driver/GPU/RocprofApi.h"
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
#include "rocprofiler-sdk/registration.h"

#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
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
};

RocprofilerRuntimeState &getRuntimeState() {
  static RocprofilerRuntimeState state;
  return state;
}

// ROCTx marker interception via libroctx64.so's callback registration.
// rocprofiler-sdk's own marker tracing requires its replacement roctx library
// to be loaded, which doesn't happen with late-start (force_configure). Instead
// we use the standard libroctx64.so's built-in callback mechanism.
constexpr uint32_t kRoctxPushA = 1;
constexpr uint32_t kRoctxPop = 2;

struct RoctxApiData {
  union {
    struct {
      const char *message;
    } roctxRangePushA;
    struct {
      const char *message;
    } roctxRangePop;
  } args;
};

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

// ---- Metric conversion ----

std::unique_ptr<Metric> convertDispatchToMetric(
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record) {
  if (record->start_timestamp >= record->end_timestamp)
    return nullptr;
  auto deviceId = static_cast<uint64_t>(
      AgentIdMapper::instance().map(record->dispatch_info.agent_id.handle));
  return std::make_unique<KernelMetric>(
      static_cast<uint64_t>(record->start_timestamp),
      static_cast<uint64_t>(record->end_timestamp), 1, deviceId,
      static_cast<uint64_t>(DeviceType::HIP),
      static_cast<uint64_t>(record->dispatch_info.queue_id.handle));
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
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record) {
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
      if (auto metric = convertDispatchToMetric(record)) {
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
      if (auto metric = convertDispatchToMetric(record)) {
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
  auto &profiler = RocprofSDKProfiler::instance();
  auto *impl = static_cast<RocprofSDKProfiler::RocprofSDKProfilerPimpl *>(
      profiler.pImpl.get());
  auto *payload = static_cast<rocprofiler_callback_tracing_hip_api_data_t *>(
      record.payload);

  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    if (!isKernelOp)
      return;
    threadState.enterOp(Scope(""));
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

// ---- ROCTx marker callback via libroctx64.so ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::roctxCallback(
    uint32_t operationId, void *data) {
  auto *apiData = static_cast<RoctxApiData *>(data);
  if (operationId == kRoctxPushA) {
    threadState.enterScope(apiData->args.roctxRangePushA.message);
  } else if (operationId == kRoctxPop) {
    threadState.exitScope();
  }
}

namespace {
int roctxTracerCallback(uint32_t /*domain*/, uint32_t operationId, void *data) {
  RocprofSDKProfiler::RocprofSDKProfilerPimpl::roctxCallback(operationId, data);
  return 0;
}

void registerRoctxCallback(bool enable) {
  // libroctx64.so is typically loaded with RTLD_LOCAL (e.g. by PyTorch), so
  // dlsym(RTLD_DEFAULT, ...) won't find it. Use RTLD_NOLOAD to get a handle
  // to the already-loaded library.
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
  auto *payload = static_cast<
      rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *>(
      record.payload);
  auto &profiler = RocprofSDKProfiler::instance();
  auto *impl = static_cast<RocprofSDKProfiler::RocprofSDKProfilerPimpl *>(
      profiler.pImpl.get());
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
  auto &profiler = RocprofSDKProfiler::instance();
  auto *impl = static_cast<RocprofSDKProfiler::RocprofSDKProfilerPimpl *>(
      profiler.pImpl.get());
  auto &correlation = profiler.correlation;

  static thread_local std::map<Data *, size_t> dataFlushedPhases;
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
    processKernelRecord(profiler, correlation.corrIdToExternId,
                        correlation.externIdToState, impl->corrIdToIsHipGraph,
                        dataPhases, kernelName, record);
  }
  if (maxCorrelationId > 0) {
    correlation.complete(maxCorrelationId);
  }
  profiler.flushDataPhases(dataFlushedPhases, dataPhases,
                           profiler.pendingGraphPool.get());
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
      nullptr);

  int valid = 0;
  rocprofiler::contextIsValid<true>(state->codeObjectContext, &valid);
  if (valid == 0)
    return -1;

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

  // Marker tracing (ROCTx) is handled via direct roctxRegisterTracerCallback
  // in doStart()/doStop(), since rocprofiler-sdk's marker callback tracing
  // requires its replacement roctx library which isn't available with
  // late-start (force_configure).

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

  valid = 0;
  rocprofiler::contextIsValid<true>(state->profilingContext, &valid);
  if (valid == 0)
    return -1;

  AgentIdMapper::instance().initialize();

  state->configured = true;
  return 0;
}

void proton_tool_fini(void *toolData) {
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
      &proton_tool_fini, static_cast<void *>(&state)};
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
  if (getBoolEnv("TRITON_ENABLE_NVTX", true))
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
  registerRoctxCallback(false);
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.profilingStarted) {
    rocprofiler::stopContext<true>(state.profilingContext);
    state.profilingStarted = false;
  }
}

RocprofSDKProfiler::RocprofSDKProfiler() {
  pImpl = std::make_unique<RocprofSDKProfilerPimpl>(*this);
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.configured) {
    rocprofiler::forceConfigure<true>(&protonConfigure);
  }
  if (!state.codeObjectStarted) {
    rocprofiler::startContext<true>(state.codeObjectContext);
    state.codeObjectStarted = true;
  }
}

RocprofSDKProfiler::~RocprofSDKProfiler() = default;

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
