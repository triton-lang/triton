#include "Profiler/CuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/Device.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace proton {

template <>
thread_local GPUProfiler<CuptiProfiler>::ProfilerState
    GPUProfiler<CuptiProfiler>::profilerState(CuptiProfiler::instance());

namespace {

std::shared_ptr<Metric> convertActivityToMetric(CUpti_Activity *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
    metric =
        std::make_shared<KernelMetric>(static_cast<uint64_t>(kernel->start),
                                       static_cast<uint64_t>(kernel->end), 1,
                                       static_cast<uint64_t>(kernel->deviceId),
                                       static_cast<uint64_t>(DeviceType::CUDA));
    break;
  }
  default:
    break;
  }
  return metric;
}

void addMetric(size_t scopeId, std::set<Data *> &dataSet,
               CUpti_Activity *activity) {
  for (auto *data : dataSet) {
    data->addMetric(scopeId, convertActivityToMetric(activity));
  }
}

uint32_t
processActivityExternalCorrelation(std::map<uint32_t, size_t> &corrIdToExternId,
                                   CUpti_Activity *activity) {
  auto *externalActivity =
      reinterpret_cast<CUpti_ActivityExternalCorrelation *>(activity);
  corrIdToExternId[externalActivity->correlationId] =
      externalActivity->externalId;
  return externalActivity->correlationId;
}

uint32_t processActivityKernel(std::map<uint32_t, size_t> &corrIdToExternId,
                               std::set<Data *> &dataSet,
                               CUpti_Activity *activity) {
  // Support CUDA >= 11.0
  auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
  auto correlationId = kernel->correlationId;
  if (corrIdToExternId.find(correlationId) == corrIdToExternId.end())
    return correlationId;
  auto externalId = corrIdToExternId[correlationId];
  addMetric(externalId, dataSet, activity);
  // Track correlation ids from the same stream and erase those < correlationId
  corrIdToExternId.erase(correlationId);
  return correlationId;
}

uint32_t processActivity(std::map<uint32_t, size_t> &corrIdToExternId,
                         std::set<Data *> &dataSet, CUpti_Activity *activity) {
  auto correlationId = 0;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
    correlationId =
        processActivityExternalCorrelation(corrIdToExternId, activity);
    break;
  }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    correlationId = processActivityKernel(corrIdToExternId, dataSet, activity);
    break;
  }
  default:
    break;
  }
  return correlationId;
}

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

struct CuptiProfiler::CuptiProfilerPimpl
    : public GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface {
  CuptiProfilerPimpl(CuptiProfiler &profiler)
      : GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~CuptiProfilerPimpl() = default;

  void startOp(const Scope &scope);
  void stopOp(const Scope &scope);

  void doStart();
  void doFlush();
  void doStop();

  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords);
  static void completeBuffer(CUcontext context, uint32_t streamId,
                             uint8_t *buffer, size_t size, size_t validSize);
  static void callbackFn(void *userData, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbId, const void *cbData);

  const static size_t AlignSize = 8;
  const static size_t BufferSize = 64 * 1024 * 1024;

  std::map<uint32_t, size_t> corrIdToExternId;
  CUpti_SubscriberHandle subscriber{};
};

void CuptiProfiler::CuptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                    size_t *bufferSize,
                                                    size_t *maxNumRecords) {
  *buffer = reinterpret_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
  if (*buffer == nullptr) {
    throw std::runtime_error("aligned_alloc failed");
  }
  *bufferSize = BufferSize;
  *maxNumRecords = 0;
}

void CuptiProfiler::CuptiProfilerPimpl::completeBuffer(CUcontext ctx,
                                                       uint32_t streamId,
                                                       uint8_t *buffer,
                                                       size_t size,
                                                       size_t validSize) {
  CuptiProfiler &profiler =
      dynamic_cast<CuptiProfiler &>(CuptiProfiler::instance());
  auto &pImpl = dynamic_cast<CuptiProfilerPimpl &>(*profiler.pImpl.get());
  auto &dataSet = profiler.dataSet;
  uint32_t maxCorrelationId = 0;
  CUptiResult status;
  CUpti_Activity *activity = nullptr;
  do {
    status = cupti::activityGetNextRecord<false>(buffer, validSize, &activity);
    if (status == CUPTI_SUCCESS) {
      auto correlationId =
          processActivity(pImpl.corrIdToExternId, dataSet, activity);
      maxCorrelationId = std::max(maxCorrelationId, correlationId);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw std::runtime_error("cupti::activityGetNextRecord failed");
    }
  } while (true);

  std::free(buffer);

  profiler.correlation.complete(maxCorrelationId);
}

void CuptiProfiler::CuptiProfilerPimpl::callbackFn(void *userData,
                                                   CUpti_CallbackDomain domain,
                                                   CUpti_CallbackId cbId,
                                                   const void *cbData) {
  auto [isRuntimeAPI, isDriverAPI] = matchKernelCbId(cbId);
  if (!(isRuntimeAPI || isDriverAPI)) {
    return;
  }
  CuptiProfiler &profiler =
      dynamic_cast<CuptiProfiler &>(CuptiProfiler::instance());
  const CUpti_CallbackData *callbackData =
      reinterpret_cast<const CUpti_CallbackData *>(cbData);
  if (callbackData->callbackSite == CUPTI_API_ENTER) {
    if (callbackData->context) {
      // Valid context and outermost level of the kernel launch
      auto scopeId = Scope::getNewScopeId();
      auto scope = Scope(scopeId, callbackData->symbolName);
      profilerState.record(scope);
    }
    profilerState.enterOp();
  } else if (callbackData->callbackSite == CUPTI_API_EXIT) {
    profilerState.exitOp();
    profiler.correlation.submit(callbackData->correlationId);
  }
}

void CuptiProfiler::CuptiProfilerPimpl::startOp(const Scope &scope) {
  cupti::activityPushExternalCorrelationId<true>(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, scope.scopeId);
}

void CuptiProfiler::CuptiProfilerPimpl::stopOp(const Scope &scope) {
  uint64_t correlationId;
  cupti::activityPopExternalCorrelationId<true>(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &correlationId);
}

void CuptiProfiler::CuptiProfilerPimpl::doStart() {
  cupti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  // Enable driver and runtime activities after external correlation so that
  // external correlation id returned is not 0
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_DRIVER);
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_RUNTIME);
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_FUNCTION);
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  // TODO: switch to directly subscribe the APIs and measure overhead
  cupti::subscribe<true>(&subscriber, callbackFn, nullptr);
  cupti::enableDomain<true>(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  cupti::enableDomain<true>(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
}

void CuptiProfiler::CuptiProfilerPimpl::doFlush() {
  // cuptiActivityFlushAll returns the activity records associated with all
  // contexts/streams.
  // This is a blocking call but it doesn’t issue any CUDA synchronization calls
  // implicitly thus it’s not guaranteed that all activities are completed on
  // the underlying devices.
  // We do an "oppurtunistic" synchronization here to try to ensure that all
  // activities are completed on the current context.
  // If the current context is not set, we don't do any synchronization.
  CUcontext cuContext = nullptr;
  cuda::ctxGetCurrent<false>(&cuContext);
  if (cuContext)
    cuda::ctxSynchronize<true>();
  profiler.correlation.flush(
      /*maxRetries=*/100, /*sleepMs=*/10,
      /*flush=*/[]() {
        cupti::activityFlushAll<true>(
            /*flag=*/0);
      });
  // CUPTI_ACTIVITY_FLAG_FLUSH_FORCED is used to ensure that even incomplete
  // activities are flushed so that the next profiling session can start with
  // new activities.
  cupti::activityFlushAll<true>(/*flag=*/CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
}

void CuptiProfiler::CuptiProfilerPimpl::doStop() {
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_DRIVER);
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_RUNTIME);
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_FUNCTION);
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cupti::enableDomain<true>(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  cupti::enableDomain<true>(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  cupti::unsubscribe<true>(subscriber);
  cupti::finalize<true>();
}

CuptiProfiler::CuptiProfiler() {
  pImpl = std::make_unique<CuptiProfilerPimpl>(*this);
}

CuptiProfiler::~CuptiProfiler() = default;

} // namespace proton
