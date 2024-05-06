#include "Profiler/RoctracerProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
//#include "Driver/GPU/Cuda.h"
#include "Driver/GPU/Roctracer.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string.h>

namespace proton {

namespace {

// Local copy of hip op types.  These are public (and stable) in later rocm releases
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

typedef enum {
  HIP_OP_BARRIER_KIND_UNKNOWN_ = 0
} hip_op_barrier_kind_t_;
// end hip op defines


std::shared_ptr<Metric> convertActivityToMetric(const roctracer_record_t *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
    case HIP_OP_DISPATCH_KIND_KERNEL_:
    case HIP_OP_DISPATCH_KIND_TASK_: {
      metric =
          std::make_shared<KernelMetric>(static_cast<uint64_t>(activity->begin_ns),
                                         static_cast<uint64_t>(activity->end_ns), 1);
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

void processActivityExternalCorrelation(std::map<uint32_t, size_t> &correlation,
                                        const roctracer_record_t *activity) {
  correlation[activity->correlation_id] = correlation[activity->external_id];
}

void processActivityKernel(std::map<uint32_t, size_t> &correlation,
                           std::set<Data *> &dataSet,
                           const roctracer_record_t *activity) {
  auto correlationId = activity->correlation_id;
  if (correlation.find(correlationId) == correlation.end()) {
    return;
  }
  auto externalId = correlation[correlationId];
  addMetric(externalId, dataSet, activity);
  // Track correlation ids from the same stream and erase those < correlationId
  correlation.erase(correlationId);
}

} // namespace

void RoctracerProfiler::startOp(const Scope &scope) {
  roctracer::activity_push_external_correlation_id<true>(
      scope.scopeId);
}

void RoctracerProfiler::stopOp(const Scope &scope) {
  uint64_t correlationId;
  roctracer::activity_pop_external_correlation_id<true>(
      &correlationId);
}

void RoctracerProfiler::setOpInProgress(bool value) {
  roctracerState.isRecording = value;
}

bool RoctracerProfiler::isOpInProgress() { return roctracerState.isRecording; }

void RoctracerProfiler::doStart() {
  // magic
  roctracer::set_properties<true>(ACTIVITY_DOMAIN_HIP_API, NULL);

  //roctracer::enable_domain_callback<true>(ACTIVITY_DOMAIN_HIP_API, api_callback, NULL);

  roctracer_properties_t properties;
  memset(&properties, 0, sizeof(roctracer_properties_t));
  properties.buffer_size = 0x1000;
  properties.buffer_callback_fun = activity_callback;
  roctracer::open_pool<true>(&properties);
  roctracer::enable_domain_activity<true>(ACTIVITY_DOMAIN_HSA_API);
  roctracer::enable_domain_activity<true>(ACTIVITY_DOMAIN_HIP_API);
  roctracer::enable_domain_activity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::enable_domain_activity<true>(ACTIVITY_DOMAIN_EXT_API);
  roctracer::start();
}

void RoctracerProfiler::doFlush() {
  // FIXME: stream synchronize?
  roctracer::flush_activity<true>();
}

void RoctracerProfiler::doStop() {
  roctracer::stop();
  roctracer::disable_domain_activity<true>(ACTIVITY_DOMAIN_HSA_API);
  roctracer::disable_domain_activity<true>(ACTIVITY_DOMAIN_HIP_API);
  roctracer::disable_domain_activity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::disable_domain_activity<true>(ACTIVITY_DOMAIN_EXT_API);
  doFlush();
}

#if 0
void RoctracerProfiler::allocBuffer(uint8_t **buffer, size_t *bufferSize,
                                size_t *maxNumRecords) {
  *buffer = reinterpret_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
  if (*buffer == nullptr) {
    throw std::runtime_error("aligned_alloc failed");
  }
  *bufferSize = BufferSize;
  *maxNumRecords = 0;
}
#endif

void RoctracerProfiler::activity_callback(const char* begin, const char* end, void* arg)
{
  RoctracerProfiler &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto &correlation = profiler.correlation;
  auto &dataSet = profiler.dataSet;

  const roctracer_record_t* record = (const roctracer_record_t*)(begin);
  const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

  while (record < end_record) {
    processActivity(correlation, dataSet, record);
    roctracer::next_record<true>(record, &record);
  }
}

#if 0
void RoctracerProfiler::completeBuffer(CUcontext ctx, uint32_t streamId,
                                   uint8_t *buffer, size_t size,
                                   size_t validSize) {
  RoctracerProfiler &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto &correlation = profiler.correlation;
  auto &dataSet = profiler.dataSet;

  CUptiResult status;
  CUpti_Activity *activity = nullptr;
  do {
    status = cupti::activityGetNextRecord<false>(buffer, validSize, &activity);
    if (status == CUPTI_SUCCESS) {
      processActivity(correlation, dataSet, activity);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw std::runtime_error("cupti::activityGetNextRecord failed");
    }
  } while (true);

  free(buffer);
}
#endif

void RoctracerProfiler::processActivity(std::map<uint32_t, size_t> &correlation,
                                    std::set<Data *> &dataSet,
                                    const roctracer_record_t *activity) {
  const char *name = roctracer::op_string(activity->domain, activity->op, activity->kind);
  fprintf(stderr, "%s\n", name);	// FIXME
  switch (activity->kind) {
    case 4242: { // FIXME, stupid ids not public
      processActivityExternalCorrelation(correlation, activity);
      break;
    }
    case HIP_OP_DISPATCH_KIND_KERNEL_:
    case HIP_OP_DISPATCH_KIND_TASK_: {
      processActivityKernel(correlation, dataSet, activity);
      break;
    }
    default:
      fprintf(stderr, "%d\n", activity->kind);  // FIXME
  }
}

#if 0
void RoctracerProfiler::processActivity(std::map<uint32_t, size_t> &correlation,
                                    std::set<Data *> &dataSet,
                                    CUpti_Activity *activity) {
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
    processActivityExternalCorrelation(correlation, activity);
    break;
  }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    processActivityKernel(correlation, dataSet, activity);
    break;
  }
  default:
    break;
  }
}
#endif
#if 0
namespace {

std::pair<bool, bool> matchKernelCbId(CUpti_CallbackId cbId) {
  bool isRuntimeApi = false;
  bool isDriverApi = false;
  switch (cbId) {
  // TODO: switch to directly subscribe the APIs
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000: {
    isRuntimeApi = true;
    break;
  }
  case CUPTI_DRIVER_TRACE_CBID_cuLaunch:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice: {
    isDriverApi = true;
    break;
  }
  default:
    break;
  }
  return std::make_pair(isRuntimeApi, isDriverApi);
}

} // namespace

void RoctracerProfiler::callback(void *userData, CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbId, const void *cbData) {
  auto [isRuntimeAPI, isDriverAPI] = matchKernelCbId(cbId);
  if (!(isRuntimeAPI || isDriverAPI)) {
    return;
  }
  RoctracerProfiler &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  const CUpti_CallbackData *callbackData =
      reinterpret_cast<const CUpti_CallbackData *>(cbData);
  if (callbackData->callbackSite == CUPTI_API_ENTER) {
    if (callbackData->context && cuptiState.level == 0) {
      // Valid context and outermost level of the kernel launch
      auto scopeId = Scope::getNewScopeId();
      auto scope = Scope(scopeId, callbackData->symbolName);
      cuptiState.record(scope, profiler.getDataSetSnapshot());
      cuptiState.enterOp();
    }
    cuptiState.level++;
  } else if (callbackData->callbackSite == CUPTI_API_EXIT) {
    cuptiState.level--;
    if (cuptiState.level == 0) {
      if (cuptiState.isRecording) {
        cuptiState.exitOp();
      }
      cuptiState.reset();
    }
  }
}
#endif
} // namespace proton
