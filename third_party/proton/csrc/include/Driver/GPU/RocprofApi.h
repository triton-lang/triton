#ifndef PROTON_DRIVER_GPU_ROCPROFILER_API_H_
#define PROTON_DRIVER_GPU_ROCPROFILER_API_H_

#include "Driver/Dispatch.h"
#include "rocprofiler-sdk/agent.h"
#include "rocprofiler-sdk/buffer_tracing.h"
#include "rocprofiler-sdk/callback_tracing.h"
#include "rocprofiler-sdk/fwd.h"
#include "rocprofiler-sdk/hip/api_args.h"
#include "rocprofiler-sdk/hip/runtime_api_id.h"
#include "rocprofiler-sdk/marker/api_args.h"
#include "rocprofiler-sdk/marker/api_id.h"
#include "rocprofiler-sdk/registration.h"
#include "rocprofiler-sdk/rocprofiler.h"

namespace proton {

namespace rocprofiler {

struct ExternLibRocprofiler : public ExternLibBase {
  using RetType = rocprofiler_status_t;
  static constexpr const char *name = "librocprofiler-sdk.so";
  static constexpr const char *symbolName = "rocprofiler_is_initialized";
  static constexpr const char *pathEnv{};
  static constexpr RetType success = ROCPROFILER_STATUS_SUCCESS;
  static inline void *lib = nullptr;
};

template <bool CheckSuccess> rocprofiler_status_t isInitialized(int *status);

template <bool CheckSuccess>
rocprofiler_status_t forceConfigure(rocprofiler_configure_func_t configureFunc);

template <bool CheckSuccess>
rocprofiler_status_t createContext(rocprofiler_context_id_t *context);

template <bool CheckSuccess>
rocprofiler_status_t destroyContext(rocprofiler_context_id_t context);

template <bool CheckSuccess>
rocprofiler_status_t contextIsValid(rocprofiler_context_id_t context,
                                    int *valid);

template <bool CheckSuccess>
rocprofiler_status_t startContext(rocprofiler_context_id_t context);

template <bool CheckSuccess>
rocprofiler_status_t stopContext(rocprofiler_context_id_t context);

template <bool CheckSuccess>
rocprofiler_status_t
createBuffer(rocprofiler_context_id_t context, size_t size, size_t watermark,
             rocprofiler_buffer_policy_t policy,
             rocprofiler_buffer_tracing_cb_t callback, void *userData,
             rocprofiler_buffer_id_t *buffer);

template <bool CheckSuccess>
rocprofiler_status_t destroyBuffer(rocprofiler_buffer_id_t buffer);

template <bool CheckSuccess>
rocprofiler_status_t flushBuffer(rocprofiler_buffer_id_t buffer);

template <bool CheckSuccess>
rocprofiler_status_t configureBufferTracingService(
    rocprofiler_context_id_t context, rocprofiler_buffer_tracing_kind_t kind,
    const rocprofiler_tracing_operation_t *operations, size_t operationCount,
    rocprofiler_buffer_id_t buffer);

template <bool CheckSuccess>
rocprofiler_status_t configureCallbackTracingService(
    rocprofiler_context_id_t context, rocprofiler_callback_tracing_kind_t kind,
    const rocprofiler_tracing_operation_t *operations, size_t operationCount,
    rocprofiler_callback_tracing_cb_t callback, void *userData);

template <bool CheckSuccess>
rocprofiler_status_t
createCallbackThread(rocprofiler_callback_thread_t *thread);

template <bool CheckSuccess>
rocprofiler_status_t assignCallbackThread(rocprofiler_buffer_id_t buffer,
                                          rocprofiler_callback_thread_t thread);

template <bool CheckSuccess>
rocprofiler_status_t
queryAvailableAgents(rocprofiler_agent_version_t version,
                     rocprofiler_query_available_agents_cb_t callback,
                     size_t agentSize, void *userData);

} // namespace rocprofiler

} // namespace proton

#endif // PROTON_DRIVER_GPU_ROCPROFILER_API_H_
