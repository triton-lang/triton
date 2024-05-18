#include "Profiler/RoctracerProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/Hip.h"
#include "Driver/GPU/Roctracer.h"

#include <roctracer/roctracer_ext.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_hsa.h>

#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string.h>

#include <unistd.h>

namespace proton {

namespace {

// Local copy of hip op types.  These are public (and stable) in later rocm
// releases
typedef enum {
  HIP_OP_COPY_KIND_UNKNOWN_ = 0,
  HIP_OP_COPY_KIND_DEVICE_TO_HOST_ = 0x11F3,
  HIP_OP_COPY_KIND_HOST_TO_DEVICE_ = 0x11F4,
  HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_ = 0x11F5,
  HIP_OP_COPY_KIND_DEVICE_TO_HOST_2D_ = 0x1201,
  HIP_OP_COPY_KIND_HOST_TO_DEVICE_2D_ = 0x1202,
  HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_2D_ = 0x1203,
  HIP_OP_COPY_KIND_FILL_BUFFER_ = 0x1207
} hip_op_copy_kind_t_;

typedef enum {
  HIP_OP_DISPATCH_KIND_UNKNOWN_ = 0,
  HIP_OP_DISPATCH_KIND_KERNEL_ = 0x11F0,
  HIP_OP_DISPATCH_KIND_TASK_ = 0x11F1
} hip_op_dispatch_kind_t_;

typedef enum { HIP_OP_BARRIER_KIND_UNKNOWN_ = 0 } hip_op_barrier_kind_t_;
// end hip op defines

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
static Flush s_flush;

std::shared_ptr<Metric>
convertActivityToMetric(const roctracer_record_t *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
  case HIP_OP_DISPATCH_KIND_KERNEL_:
  case HIP_OP_DISPATCH_KIND_TASK_: {
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
    t_externalIds[RoctracerProfiler::CorrelationDomain::size];

} // namespace

// External correlation
void RoctracerProfiler::pushCorrelationID(uint64_t id, CorrelationDomain type) {
  if (!instance().externalCorrelationEnabled) {
    return;
  }
  t_externalIds[type].push_back(id);
}

void RoctracerProfiler::popCorrelationID(CorrelationDomain type) {
  if (!instance().externalCorrelationEnabled) {
    return;
  }
  t_externalIds[type].pop_back();
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
  // roctracer::enable_domain_callback<true>(ACTIVITY_DOMAIN_HSA_API,
  // api_callback, nullptr);
  roctracer::enable_domain_callback<true>(ACTIVITY_DOMAIN_HIP_API, api_callback,
                                          nullptr);

  // Activity Records
  roctracer_properties_t properties;
  memset(&properties, 0, sizeof(roctracer_properties_t));
  properties.buffer_size = 0x1000;
  properties.buffer_callback_fun = activity_callback;
  roctracer::open_pool<true>(&properties);
  roctracer::enable_domain_activity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::start();
}

void RoctracerProfiler::doFlush() {
  // Implement reliable flushing.  Wait for all dispatched ops to be reported
  auto ret = hip::deviceSynchronize<true>();
  roctracer::flush_activity<true>();
  std::unique_lock<std::mutex> lock(s_flush.mutex_);
  auto correlationId =
      s_flush.maxCorrelationId_.load(); // load ending id from the running max

  // Poll on the worker finding the final correlation id
  int timeout = 500;
  while ((s_flush.maxCompletedCorrelationId_ < correlationId) && --timeout) {
    lock.unlock();
    roctracer::flush_activity<true>();
    usleep(1000);
    lock.lock();
  }
}

void RoctracerProfiler::doStop() {
  roctracer::stop();
  // roctracer::disable_domain_callback<true>(ACTIVITY_DOMAIN_HSA_API);
  roctracer::disable_domain_callback<true>(ACTIVITY_DOMAIN_HIP_API);
  roctracer::disable_domain_activity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::close_pool<true>();
}

void RoctracerProfiler::activity_callback(const char *begin, const char *end,
                                          void *arg) {
  RoctracerProfiler &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto &correlation = profiler.correlation;
  auto &dataSet = profiler.dataSet;

  std::unique_lock<std::mutex> lock(s_flush.mutex_);
  const roctracer_record_t *record = (const roctracer_record_t *)(begin);
  const roctracer_record_t *end_record = (const roctracer_record_t *)(end);

  while (record < end_record) {
    // Log latest completed correlation id.  Used to ensure we have flushed all
    // data on stop
    if (record->correlation_id > s_flush.maxCompletedCorrelationId_) {
      s_flush.maxCompletedCorrelationId_ = record->correlation_id;
    }
    processActivity(correlation, dataSet, record);
    roctracer::next_record<true>(record, &record);
  }
}

void RoctracerProfiler::processActivity(std::map<uint32_t, size_t> &correlation,
                                        std::set<Data *> &dataSet,
                                        const roctracer_record_t *record) {
  switch (record->kind) {
  case HIP_OP_DISPATCH_KIND_KERNEL_:
  case HIP_OP_DISPATCH_KIND_TASK_: {
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
  // case NO_HSA_GO_FISH:
  //{
  //   isDriverApi = true;
  //   break;
  // }
  default:
    break;
  }
  return std::make_pair(isRuntimeApi, isDriverApi);
}

} // namespace

void RoctracerProfiler::api_callback(uint32_t domain, uint32_t cid,
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
            roctracer::op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0);
        auto scopeId = Scope::getNewScopeId();
        auto scope = Scope(scopeId, name);
        roctracerState.record(scope, profiler.getDataSet());
        roctracerState.enterOp();
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

      // Generate and Report external correlation
      for (int it = CorrelationDomain::begin; it < CorrelationDomain::end;
           ++it) {
        if (t_externalIds[it].size() > 0) {
          profiler.correlation[data->correlation_id] = t_externalIds[it].back();
        }
      }

      // track outstanding op for flush
      s_flush.reportCorrelation(data->correlation_id);
    }
  }
}
} // namespace proton
