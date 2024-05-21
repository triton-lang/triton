#include "Profiler/RoctracerProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/Hip.h"
#include "Driver/GPU/Roctracer.h"
#include <hip/amd_detail/hip_runtime_prof.h>

#include <roctracer/roctracer_ext.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_hsa.h>

#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>

#include <unistd.h>

namespace proton {

namespace {

// Track dispatched ops to ensure a complete flush
class Flush {
public:
  std::mutex mutex_;
  std::atomic<uint64_t> maxCorrelationId_;
  uint64_t maxCompletedCorrelationId_{0};
  void reportCorrelation(const uint64_t &cid) {
    uint64_t prev = maxCorrelationId_;
    while (prev < cid && !maxCorrelationId_.compare_exchange_weak(prev, cid)) {
    }
  }
};
Flush flushState;

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

void processActivityKernel(std::map<uint32_t, size_t> &correlation,
                           std::set<Data *> &dataSet,
                           const roctracer_record_t *activity) {
  auto correlationId = activity->correlation_id;
  // TODO: non-triton kernels
  if (correlation.find(correlationId) == correlation.end()) {
    return;
  }
  auto externalId = correlation[correlationId];
  addMetric(externalId, dataSet, activity);
  // Track correlation ids from the same stream and erase those < correlationId
  correlation.erase(correlationId);
}

thread_local std::deque<uint64_t>
    externalIdMap[RoctracerProfiler::CorrelationDomain::size];

} // namespace

// External correlation
void RoctracerProfiler::pushCorrelationID(uint64_t id, CorrelationDomain type) {
  if (!instance().externalCorrelationEnabled) {
    return;
  }
  externalIdMap[type].push_back(id);
}

void RoctracerProfiler::popCorrelationID(CorrelationDomain type) {
  if (!instance().externalCorrelationEnabled) {
    return;
  }
  externalIdMap[type].pop_back();
}

void RoctracerProfiler::startOp(const Scope &scope) {
  pushCorrelationID(scope.scopeId, Default);
}

void RoctracerProfiler::stopOp(const Scope &scope) {
  popCorrelationID(Default);
}

void RoctracerProfiler::setOpInProgress(bool value) {
  roctracerState.isRecording = value;
}

bool RoctracerProfiler::isOpInProgress() { return roctracerState.isRecording; }

void RoctracerProfiler::doStart() {
  // Inline Callbacks
  // roctracer::enableDomainCallback<true>(ACTIVITY_DOMAIN_HSA_API,
  //                                       api_callback, nullptr);
  roctracer::enableDomainCallback<true>(ACTIVITY_DOMAIN_HIP_API, apiCallback,
                                        nullptr);

  // Activity Records
  roctracer_properties_t properties;
  memset(&properties, 0, sizeof(roctracer_properties_t));
  properties.buffer_size = 0x1000;
  properties.buffer_callback_fun = activityCallback;
  roctracer::openPool<true>(&properties);
  roctracer::enableDomainActivity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::start();
}

void RoctracerProfiler::doFlush() {
  // Implement reliable flushing.  Wait for all dispatched ops to be reported
  auto ret = hip::deviceSynchronize<true>();
  roctracer::flushActivity<true>();
  std::unique_lock<std::mutex> lock(flushState.mutex_);
  // Load ending id from the running max
  auto correlationId = flushState.maxCorrelationId_.load();

  // Poll on the worker finding the final correlation id
  int timeout = 500;
  while ((flushState.maxCompletedCorrelationId_ < correlationId) && --timeout) {
    lock.unlock();
    roctracer::flushActivity<true>();
    usleep(1000);
    lock.lock();
  }
}

void RoctracerProfiler::doStop() {
  roctracer::stop();
  // roctracer::disable_domain_callback<true>(ACTIVITY_DOMAIN_HSA_API);
  roctracer::disableDomainCallback<true>(ACTIVITY_DOMAIN_HIP_API);
  roctracer::disableDomainActivity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::closePool<true>();
}

void RoctracerProfiler::activityCallback(const char *begin, const char *end,
                                         void *arg) {
  RoctracerProfiler &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto &correlation = profiler.correlation;
  auto &dataSet = profiler.dataSet;

  std::unique_lock<std::mutex> lock(flushState.mutex_);
  const roctracer_record_t *record = (const roctracer_record_t *)(begin);
  const roctracer_record_t *end_record = (const roctracer_record_t *)(end);

  while (record < end_record) {
    // Log latest completed correlation id.  Used to ensure we have flushed all
    // data on stop
    if (record->correlation_id > flushState.maxCompletedCorrelationId_) {
      flushState.maxCompletedCorrelationId_ = record->correlation_id;
    }
    processActivity(correlation, dataSet, record);
    roctracer::getNextRecord<true>(record, &record);
  }
}

void RoctracerProfiler::processActivity(std::map<uint32_t, size_t> &correlation,
                                        std::set<Data *> &dataSet,
                                        const roctracer_record_t *record) {
  switch (record->kind) {
  case kHipVdiCommandKernel: {
    processActivityKernel(correlation, dataSet, record);
    break;
  }
  default:;
  }
}

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
  case HIP_API_ID_hipLaunchByPtr:
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

void RoctracerProfiler::apiCallback(uint32_t domain, uint32_t cid,
                                    const void *callback_data, void *arg) {
  auto [isRuntimeAPI, isDriverAPI] = matchKernelCbId(cid);
  if (!(isRuntimeAPI || isDriverAPI)) {
    return;
  }
  RoctracerProfiler &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t *data = (const hip_api_data_t *)(callback_data);
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      // if (callbackData->context && roctracerState.level == 0) {
      {
        // Valid context and outermost level of the kernel launch
        const char *name =
            roctracer::getOpString(ACTIVITY_DOMAIN_HIP_API, cid, 0);
        auto scopeId = Scope::getNewScopeId();
        auto scope = Scope(scopeId, name);
        roctracerState.record(scope, profiler.getDataSet());
        roctracerState.enterOp();

        // Generate and Report external correlation
        for (int it = CorrelationDomain::begin; it < CorrelationDomain::end;
             ++it) {
          if (externalIdMap[it].size() > 0) {
            profiler.correlation[data->correlation_id] = externalIdMap[it].back();
          }
        }
      }
      roctracerState.level++;
    } else if (data->phase == ACTIVITY_API_PHASE_EXIT) {
      roctracerState.level--;
      if (roctracerState.level == 0) {
        if (roctracerState.isRecording) {
          roctracerState.exitOp();
        }
        roctracerState.reset();
      }

      // track outstanding op for flush
      flushState.reportCorrelation(data->correlation_id);
    }
  }
}
} // namespace proton
