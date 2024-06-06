#ifndef PROTON_DRIVER_GPU_CUPTI_H_
#define PROTON_DRIVER_GPU_CUPTI_H_

#include "cupti.h"

namespace proton {

namespace cupti {

template <bool CheckSuccess>
CUptiResult activityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted);

template <bool CheckSuccess>
CUptiResult subscribe(CUpti_SubscriberHandle *subscriber,
                      CUpti_CallbackFunc callback, void *userdata);

template <bool CheckSuccess>
CUptiResult enableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                         CUpti_CallbackDomain domain);

template <bool CheckSuccess>
CUptiResult activityEnableContext(CUcontext context, CUpti_ActivityKind kind);

template <bool CheckSuccess>
CUptiResult activityDisableContext(CUcontext context, CUpti_ActivityKind kind);

template <bool CheckSuccess>
CUptiResult activityEnable(CUpti_ActivityKind kind);

template <bool CheckSuccess>
CUptiResult activityDisable(CUpti_ActivityKind kind);

template <bool CheckSuccess> CUptiResult activityFlushAll(uint32_t flag);

template <bool CheckSuccess>
CUptiResult activityGetNextRecord(uint8_t *buffer, size_t validBufferSizeBytes,
                                  CUpti_Activity **record);

template <bool CheckSuccess>
CUptiResult
activityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                  uint64_t id);

template <bool CheckSuccess>
CUptiResult activityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                             uint64_t *lastId);

template <bool CheckSuccess>
CUptiResult activitySetAttribute(CUpti_ActivityAttribute attr,
                                 size_t *valueSize, void *value);

template <bool CheckSuccess>
CUptiResult unsubscribe(CUpti_SubscriberHandle subscriber);

template <bool CheckSuccess> CUptiResult finalize();

} // namespace cupti

} // namespace proton

#endif // PROTON_EXTERN_DISPATCH_H_
