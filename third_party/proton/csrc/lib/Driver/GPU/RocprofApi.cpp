#include "Driver/GPU/RocprofApi.h"

namespace proton {
namespace rocprofiler {

DEFINE_DISPATCH(ExternLibRocprofiler, isInitialized, rocprofiler_is_initialized,
                int *)

DEFINE_DISPATCH(ExternLibRocprofiler, forceConfigure,
                rocprofiler_force_configure, rocprofiler_configure_func_t)

DEFINE_DISPATCH(ExternLibRocprofiler, createContext, rocprofiler_create_context,
                rocprofiler_context_id_t *)

DEFINE_DISPATCH(ExternLibRocprofiler, destroyContext,
                rocprofiler_destroy_context, rocprofiler_context_id_t)

DEFINE_DISPATCH(ExternLibRocprofiler, contextIsValid,
                rocprofiler_context_is_valid, rocprofiler_context_id_t, int *)

DEFINE_DISPATCH(ExternLibRocprofiler, startContext, rocprofiler_start_context,
                rocprofiler_context_id_t)

DEFINE_DISPATCH(ExternLibRocprofiler, stopContext, rocprofiler_stop_context,
                rocprofiler_context_id_t)

DEFINE_DISPATCH(ExternLibRocprofiler, createBuffer, rocprofiler_create_buffer,
                rocprofiler_context_id_t, size_t, size_t,
                rocprofiler_buffer_policy_t, rocprofiler_buffer_tracing_cb_t,
                void *, rocprofiler_buffer_id_t *)

DEFINE_DISPATCH(ExternLibRocprofiler, destroyBuffer, rocprofiler_destroy_buffer,
                rocprofiler_buffer_id_t)

DEFINE_DISPATCH(ExternLibRocprofiler, flushBuffer, rocprofiler_flush_buffer,
                rocprofiler_buffer_id_t)

DEFINE_DISPATCH(ExternLibRocprofiler, configureBufferTracingService,
                rocprofiler_configure_buffer_tracing_service,
                rocprofiler_context_id_t, rocprofiler_buffer_tracing_kind_t,
                const rocprofiler_tracing_operation_t *, size_t,
                rocprofiler_buffer_id_t)

DEFINE_DISPATCH(ExternLibRocprofiler, configureCallbackTracingService,
                rocprofiler_configure_callback_tracing_service,
                rocprofiler_context_id_t, rocprofiler_callback_tracing_kind_t,
                const rocprofiler_tracing_operation_t *, size_t,
                rocprofiler_callback_tracing_cb_t, void *)

DEFINE_DISPATCH(ExternLibRocprofiler, createCallbackThread,
                rocprofiler_create_callback_thread,
                rocprofiler_callback_thread_t *)

DEFINE_DISPATCH(ExternLibRocprofiler, assignCallbackThread,
                rocprofiler_assign_callback_thread, rocprofiler_buffer_id_t,
                rocprofiler_callback_thread_t)

DEFINE_DISPATCH(ExternLibRocprofiler, queryAvailableAgents,
                rocprofiler_query_available_agents, rocprofiler_agent_version_t,
                rocprofiler_query_available_agents_cb_t, size_t, void *)

} // namespace rocprofiler
} // namespace proton
