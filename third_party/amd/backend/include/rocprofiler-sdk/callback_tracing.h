// Minimal subset of rocprofiler-sdk/callback_tracing.h
// Contains only the types and function prototypes used by Proton.
// The upstream header pulls in every subsystem's API arg types (RCCL,
// rocdecode, rocjpeg, …) via a single union, which drags in headers
// that are not bundled here.  We extract only the HIP and marker
// callback data types that Proton actually uses.

#pragma once

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip/api_args.h>
#include <rocprofiler-sdk/marker/api_args.h>

#include <stdint.h>

typedef void (*rocprofiler_callback_tracing_cb_t)(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *user_data, void *callback_data)
    ROCPROFILER_NONNULL(2);

typedef struct rocprofiler_callback_tracing_hip_api_data_t {
  uint64_t size;
  rocprofiler_hip_api_args_t args;
  rocprofiler_hip_api_retval_t retval;
} rocprofiler_callback_tracing_hip_api_data_t;

typedef struct rocprofiler_callback_tracing_marker_api_data_t {
  uint64_t size;
  rocprofiler_marker_api_args_t args;
  rocprofiler_marker_api_retval_t retval;
} rocprofiler_callback_tracing_marker_api_data_t;

typedef struct
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t {
  uint64_t size;
  uint64_t kernel_id;
  uint64_t code_object_id;
  const char *kernel_name;
  uint64_t kernel_object;
  uint32_t kernarg_segment_size;
  uint32_t kernarg_segment_alignment;
  uint32_t group_segment_size;
  uint32_t private_segment_size;
  uint32_t sgpr_count;
  uint32_t arch_vgpr_count;
  uint32_t accum_vgpr_count;
  int64_t kernel_code_entry_byte_offset;
  rocprofiler_address_t kernel_address;
} rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_callback_tracing_service(
    rocprofiler_context_id_t context_id,
    rocprofiler_callback_tracing_kind_t kind,
    const rocprofiler_tracing_operation_t *operations, size_t operations_count,
    rocprofiler_callback_tracing_cb_t callback, void *callback_args);
