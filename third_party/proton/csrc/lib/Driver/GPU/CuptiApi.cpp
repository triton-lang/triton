#include "Driver/GPU/CuptiApi.h"
#include "Driver/Device.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace cupti {

struct ExternLibCupti : public ExternLibBase {
  using RetType = CUptiResult;
  static constexpr const char *name = "libcupti.so";
  static constexpr RetType success = CUPTI_SUCCESS;
  static void *lib;
};

void *ExternLibCupti::lib = nullptr;

DEFINE_DISPATCH(ExternLibCupti, activityRegisterCallbacks,
                cuptiActivityRegisterCallbacks,
                CUpti_BuffersCallbackRequestFunc,
                CUpti_BuffersCallbackCompleteFunc)

DEFINE_DISPATCH(ExternLibCupti, subscribe, cuptiSubscribe,
                CUpti_SubscriberHandle *, CUpti_CallbackFunc, void *)

DEFINE_DISPATCH(ExternLibCupti, enableDomain, cuptiEnableDomain, uint32_t,
                CUpti_SubscriberHandle, CUpti_CallbackDomain)

DEFINE_DISPATCH(ExternLibCupti, enableCallback, cuptiEnableCallback, uint32_t,
                CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId);

DEFINE_DISPATCH(ExternLibCupti, activityEnable, cuptiActivityEnable,
                CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityDisable, cuptiActivityDisable,
                CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityEnableContext,
                cuptiActivityEnableContext, CUcontext, CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityDisableContext,
                cuptiActivityDisableContext, CUcontext, CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityFlushAll, cuptiActivityFlushAll,
                uint32_t)

DEFINE_DISPATCH(ExternLibCupti, activityGetNextRecord,
                cuptiActivityGetNextRecord, uint8_t *, size_t,
                CUpti_Activity **)

DEFINE_DISPATCH(ExternLibCupti, activityPushExternalCorrelationId,
                cuptiActivityPushExternalCorrelationId,
                CUpti_ExternalCorrelationKind, uint64_t)

DEFINE_DISPATCH(ExternLibCupti, activityPopExternalCorrelationId,
                cuptiActivityPopExternalCorrelationId,
                CUpti_ExternalCorrelationKind, uint64_t *)

DEFINE_DISPATCH(ExternLibCupti, activitySetAttribute, cuptiActivitySetAttribute,
                CUpti_ActivityAttribute, size_t *, void *)

DEFINE_DISPATCH(ExternLibCupti, unsubscribe, cuptiUnsubscribe,
                CUpti_SubscriberHandle)

DEFINE_DISPATCH(ExternLibCupti, finalize, cuptiFinalize)

DEFINE_DISPATCH(ExternLibCupti, getGraphExecId, cuptiGetGraphExecId,
                CUgraphExec, uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getGraphId, cuptiGetGraphId, CUgraph,
                uint32_t *);

} // namespace cupti

} // namespace proton
