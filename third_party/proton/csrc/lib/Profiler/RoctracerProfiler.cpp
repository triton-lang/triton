#include "Profiler/RoctracerProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/HipApi.h"
#include "Driver/GPU/RoctracerApi.h"

#include "hip/amd_detail/hip_runtime_prof.h"
#include "roctracer/roctracer_ext.h"
#include "roctracer/roctracer_hip.h"

#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <tuple>

#include <cxxabi.h>
#include <unistd.h>

namespace proton {

template <>
thread_local GPUProfiler<RoctracerProfiler>::ProfilerState
    GPUProfiler<RoctracerProfiler>::profilerState(
        RoctracerProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<RoctracerProfiler>::Correlation::externIdQueue{};

namespace {

std::shared_ptr<Metric>
convertActivityToMetric(const roctracer_record_t *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
  case kHipVdiCommandKernel: {
    metric = std::make_shared<KernelMetric>(
        static_cast<uint64_t>(activity->begin_ns),
        static_cast<uint64_t>(activity->end_ns), 1,
        static_cast<uint64_t>(activity->device_id),
        static_cast<uint64_t>(DeviceType::HIP));
    break;
  }
  default:
    break;
  }
  return metric;
}

void addMetric(size_t scopeId, std::set<Data *> &dataSet,
               const roctracer_record_t *activity) {
  for (auto *data : dataSet) {
    data->addMetric(scopeId, convertActivityToMetric(activity));
  }
}

void processActivityKernel(size_t externId, std::set<Data *> &dataSet,
                           const roctracer_record_t *activity) {
  if (externId == Scope::DummyScopeId)
    return;
  auto correlationId = activity->correlation_id;
  addMetric(externId, dataSet, activity);
}

void processActivity(size_t externId, std::set<Data *> &dataSet,
                     const roctracer_record_t *record) {
  switch (record->kind) {
  case 0x11F1: // Task - kernel enqueued by graph launch
  case kHipVdiCommandKernel: {
    processActivityKernel(externId, dataSet, record);
    break;
  }
  default:
    break;
  }
}

} // namespace

namespace {

std::pair<bool, bool> matchKernelCbId(uint32_t cbId) {
  bool isRuntimeApi = false;
  bool isDriverApi = false;
  switch (cbId) {
  // TODO: switch to directly subscribe the APIs
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
  case HIP_API_ID_hipModuleLaunchCooperativeKernelMultiDevice: {
    isRuntimeApi = true;
    break;
  }
  default:
    break;
  }
  return std::make_pair(isRuntimeApi, isDriverApi);
}
// C++ symbol demangle
static inline const std::string cxxDemangle(const char *symbol) {
  size_t funcNameSize;
  int status;
  if (const char *name =
          abi::__cxa_demangle(symbol, NULL, &funcNameSize, &status)) {
    std::string ret(name);
    std::free(reinterpret_cast<void *>(const_cast<char *>(name)));
    return ret;
  }
  return std::string(symbol);
}

const std::string kernelName(uint32_t domain, uint32_t cid,
                             const void *callback_data) {
  std::string name;
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t *data = (const hip_api_data_t *)(callback_data);
    switch (cid) {
    case HIP_API_ID_hipExtLaunchKernel: {
      auto &params = data->args.hipExtLaunchKernel;
      name = cxxDemangle(
          hip::getKernelNameRefByPtr(params.function_address, params.stream));
    } break;
    case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice: {
      auto &params =
          data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList__val;
      name =
          cxxDemangle(hip::getKernelNameRefByPtr(params.func, params.stream));
    } break;
    case HIP_API_ID_hipExtModuleLaunchKernel: {
      auto &params = data->args.hipExtModuleLaunchKernel;
      name = cxxDemangle(hip::getKernelNameRef(params.f));
    } break;
    case HIP_API_ID_hipHccModuleLaunchKernel: {
      auto &params = data->args.hipHccModuleLaunchKernel;
      name = cxxDemangle(hip::getKernelNameRef(params.f));
    } break;
    case HIP_API_ID_hipLaunchCooperativeKernel: {
      auto &params = data->args.hipLaunchCooperativeKernel;
      name = cxxDemangle(hip::getKernelNameRefByPtr(params.f, params.stream));
    } break;
    case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice: {
      auto &params = data->args.hipLaunchCooperativeKernelMultiDevice
                         .launchParamsList__val;
      name =
          cxxDemangle(hip::getKernelNameRefByPtr(params.func, params.stream));
    } break;
    case HIP_API_ID_hipLaunchKernel: {
      auto &params = data->args.hipLaunchKernel;
      name = cxxDemangle(
          hip::getKernelNameRefByPtr(params.function_address, params.stream));
    } break;
    case HIP_API_ID_hipModuleLaunchKernel: {
      auto &params = data->args.hipModuleLaunchKernel;
      name = cxxDemangle(hip::getKernelNameRef(params.f));
    } break;
    case HIP_API_ID_hipGraphLaunch: {
      name = "graphLaunch";
    } break;
    default:
      break;
    }
  }
  return name;
}

} // namespace

struct RoctracerProfiler::RoctracerProfilerPimpl
    : public GPUProfiler<RoctracerProfiler>::GPUProfilerPimplInterface {
  RoctracerProfilerPimpl(RoctracerProfiler &profiler)
      : GPUProfiler<RoctracerProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~RoctracerProfilerPimpl() = default;

  void startOp(const Scope &scope);
  void stopOp(const Scope &scope);

  void doStart();
  void doFlush();
  void doStop();

  static void apiCallback(uint32_t domain, uint32_t cid,
                          const void *callbackData, void *arg);
  static void activityCallback(const char *begin, const char *end, void *arg);

  static constexpr size_t BufferSize = 64 * 1024 * 1024;
};

void RoctracerProfiler::RoctracerProfilerPimpl::apiCallback(
    uint32_t domain, uint32_t cid, const void *callback_data, void *arg) {
  auto [isRuntimeAPI, isDriverAPI] = matchKernelCbId(cid);
  if (!(isRuntimeAPI || isDriverAPI)) {
    return;
  }
  auto &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto &pImpl = dynamic_cast<RoctracerProfiler::RoctracerProfilerPimpl &>(
      *profiler.pImpl);
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t *data = (const hip_api_data_t *)(callback_data);
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      // Valid context and outermost level of the kernel launch
      const std::string name = kernelName(domain, cid, callback_data);
      auto scopeId = Scope::getNewScopeId();
      auto scope = Scope(scopeId, name);
      profilerState.record(scope);
      profilerState.enterOp();
      profiler.correlation.correlate(data->correlation_id);
    } else if (data->phase == ACTIVITY_API_PHASE_EXIT) {
      profilerState.exitOp();
      // Track outstanding op for flush
      profiler.correlation.submit(data->correlation_id);
    }
  }
}

void RoctracerProfiler::RoctracerProfilerPimpl::activityCallback(
    const char *begin, const char *end, void *arg) {
  auto &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto &dataSet = profiler.dataSet;
  auto &correlation = profiler.correlation;

  const roctracer_record_t *record =
      reinterpret_cast<const roctracer_record_t *>(begin);
  const roctracer_record_t *endRecord =
      reinterpret_cast<const roctracer_record_t *>(end);
  uint64_t maxCorrelationId = 0;

  while (record != endRecord) {
    // Log latest completed correlation id.  Used to ensure we have flushed all
    // data on stop
    maxCorrelationId =
        std::max<uint64_t>(maxCorrelationId, record->correlation_id);
    auto externId =
        correlation.corrIdToExternId.find(record->correlation_id)
            ? correlation.corrIdToExternId.at(record->correlation_id)
            : Scope::DummyScopeId;
    processActivity(externId, dataSet, record);
    // Track correlation ids from the same stream and erase those <
    // correlationId
    correlation.corrIdToExternId.erase(record->correlation_id);
    roctracer::getNextRecord<true>(record, &record);
  }
  correlation.complete(maxCorrelationId);
}

void RoctracerProfiler::RoctracerProfilerPimpl::startOp(const Scope &scope) {
  // Track correlation id for the scope
  profiler.correlation.pushExternId(scope.scopeId);
}

void RoctracerProfiler::RoctracerProfilerPimpl::stopOp(const Scope &scope) {
  profiler.correlation.popExternId();
}

void RoctracerProfiler::RoctracerProfilerPimpl::doStart() {
  roctracer::enableDomainCallback<true>(ACTIVITY_DOMAIN_HIP_API, apiCallback,
                                        nullptr);
  // Activity Records
  roctracer_properties_t properties{0};
  properties.buffer_size = BufferSize;
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
      /*maxRetries=*/100, /*sleepMs=*/10, /*flush=*/
      []() { roctracer::flushActivity<true>(); });
}

void RoctracerProfiler::RoctracerProfilerPimpl::doStop() {
  roctracer::stop();
  roctracer::disableDomainCallback<true>(ACTIVITY_DOMAIN_HIP_API);
  roctracer::disableDomainActivity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::closePool<true>();
}

RoctracerProfiler::RoctracerProfiler() {
  pImpl = std::make_unique<RoctracerProfilerPimpl>(*this);
}

RoctracerProfiler::~RoctracerProfiler() = default;

} // namespace proton
