#include "Profiler/RoctracerProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/HipApi.h"
#include "Driver/GPU/HsaApi.h"
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
thread_local GPUProfiler<RoctracerProfiler>::ThreadState
    GPUProfiler<RoctracerProfiler>::threadState(RoctracerProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<RoctracerProfiler>::Correlation::externIdQueue{};

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

std::shared_ptr<Metric>
convertActivityToMetric(const roctracer_record_t *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
  case kHipVdiCommandKernel: {
  if (activity->begin_ns < activity->end_ns) {
    metric = std::make_shared<KernelMetric>(
        static_cast<uint64_t>(activity->begin_ns),
        static_cast<uint64_t>(activity->end_ns), 1,
        static_cast<uint64_t>(
            DeviceInfo::instance().mapDeviceId(activity->device_id)),
        static_cast<uint64_t>(DeviceType::HIP));
  }
    break;
  }
  default:
    break;
  }
  return metric;
}

void processActivityKernel(size_t externId, std::set<Data *> &dataSet,
                           const roctracer_record_t *activity, bool isAPI, bool isGraphKernel) {
  if (externId == Scope::DummyScopeId)
    return;
  auto correlationId = activity->correlation_id;
  if(isGraphKernel)
  	printf("correlation_id=%ld\n", correlationId );
  for (auto *data : dataSet) {
    auto scopeId = externId;
    if (isAPI)
      scopeId = data->addScope(/*parentId=*/externId, activity->kernel_name);
    data->addMetric(scopeId, convertActivityToMetric(activity));
  }
}

void processActivity(size_t externId, std::set<Data *> &dataSet,
                     const roctracer_record_t *record, bool isAPI, bool isGraphKernel) {
  switch (record->kind) {
  case 0x11F1: // Task - kernel enqueued by graph launch
  case kHipVdiCommandKernel: {
    processActivityKernel(externId, dataSet, record, isAPI, isGraphKernel);
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

} // namespace

struct RoctracerProfiler::RoctracerProfilerPimpl
    : public GPUProfiler<RoctracerProfiler>::GPUProfilerPimplInterface {
  RoctracerProfilerPimpl(RoctracerProfiler &profiler)
      : GPUProfiler<RoctracerProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~RoctracerProfilerPimpl() = default;

  void doStart() override;
  void doFlush() override;
  void doStop() override;

  static void apiCallback(uint32_t domain, uint32_t cid,
                          const void *callbackData, void *arg);
  static void activityCallback(const char *begin, const char *end, void *arg);

  static constexpr size_t BufferSize = 64 * 1024 * 1024;
  ThreadSafeMap<uint64_t, bool, std::unordered_map<uint64_t, bool>>
      CorrIdToIsHipGraphMap;  
};

void RoctracerProfiler::RoctracerProfilerPimpl::apiCallback(
    uint32_t domain, uint32_t cid, const void *callbackData, void *arg) {
  auto [isRuntimeAPI, isDriverAPI] = matchKernelCbId(cid);
  if (!(isRuntimeAPI || isDriverAPI)) {
    return;
  }
  
  auto &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto *pImpl = dynamic_cast<RoctracerProfiler::RoctracerProfilerPimpl *>(profiler.pImpl.get());
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t *data = (const hip_api_data_t *)(callbackData);
    if(cid == HIP_API_ID_hipGraphLaunch){
	pImpl->CorrIdToIsHipGraphMap[data->correlation_id] = true;
    }
//    else{
//	    pImpl->CorrIdToIsHipGraphMap[data->correlation_id] = false;
//    }
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      // Valid context and outermost level of the kernel launch
      auto scopeId = Scope::getNewScopeId();
      threadState.record(scopeId);
      threadState.enterOp(scopeId);
      profiler.correlation.correlate(data->correlation_id);
    } else if (data->phase == ACTIVITY_API_PHASE_EXIT) {
      threadState.exitOp();
      // Track outstanding op for flush
      profiler.correlation.submit(data->correlation_id);
    }
  }
}

void RoctracerProfiler::RoctracerProfilerPimpl::activityCallback(
    const char *begin, const char *end, void *arg) {
  auto &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto *pImpl = dynamic_cast<RoctracerProfiler::RoctracerProfilerPimpl *>(profiler.pImpl.get());
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
    // TODO(Keren): Roctracer doesn't support cuda graph yet.
    auto externId =
        correlation.corrIdToExternId.contain(record->correlation_id)
            ? correlation.corrIdToExternId.at(record->correlation_id).first
            : Scope::DummyScopeId;
    auto isAPI = correlation.apiExternIds.contain(externId);
    bool isGraphKernel = pImpl->CorrIdToIsHipGraphMap.contain(record->correlation_id);
//    if (pImpl->CorrIdToIsHipGraphMap.contain(record->correlation_id)) {
//    	printf("correlation_id=%ld, isGraphKernel,%d\n",  record->correlation_id, pImpl->CorrIdToIsHipGraphMap.at(record->correlation_id));
//    }
    processActivity(externId, dataSet, record, isAPI, isGraphKernel);
    // Track correlation ids from the same stream and erase those <
    // correlationId
    correlation.corrIdToExternId.erase(record->correlation_id);
    correlation.apiExternIds.erase(externId);
    roctracer::getNextRecord<true>(record, &record);
  }
  correlation.complete(maxCorrelationId);
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
