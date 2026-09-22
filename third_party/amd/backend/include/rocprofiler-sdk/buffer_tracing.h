// Minimal subset of rocprofiler-sdk/buffer_tracing.h
// Contains only the types and function prototypes used by Proton.

#pragma once

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

#include <stdint.h>

typedef struct rocprofiler_buffer_tracing_kernel_dispatch_record_t {
  uint64_t size;
  rocprofiler_buffer_tracing_kind_t kind;
  rocprofiler_kernel_dispatch_operation_t operation;
  rocprofiler_async_correlation_id_t correlation_id;
  rocprofiler_thread_id_t thread_id;
  rocprofiler_timestamp_t start_timestamp;
  rocprofiler_timestamp_t end_timestamp;
  rocprofiler_kernel_dispatch_info_t dispatch_info;
} rocprofiler_buffer_tracing_kernel_dispatch_record_t;

rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_buffer_tracing_service(
    rocprofiler_context_id_t context_id, rocprofiler_buffer_tracing_kind_t kind,
    const rocprofiler_tracing_operation_t *operations, size_t operations_count,
    rocprofiler_buffer_id_t buffer_id);
