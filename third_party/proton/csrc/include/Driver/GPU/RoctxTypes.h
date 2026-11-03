#ifndef PROTON_DRIVER_GPU_ROCTX_TYPES_H_
#define PROTON_DRIVER_GPU_ROCTX_TYPES_H_

// Callback data types for libroctx64.so's tracer callback mechanism.
// These describe the ABI of the roctxRegisterTracerCallback interface
// and are used by both the roctracer and rocprofiler-sdk profiler backends.

#include <stdint.h>

typedef uint64_t roctx_range_id_t;

enum roctx_api_id_t {
  ROCTX_API_ID_roctxMarkA = 0,
  ROCTX_API_ID_roctxRangePushA = 1,
  ROCTX_API_ID_roctxRangePop = 2,
  ROCTX_API_ID_roctxRangeStartA = 3,
  ROCTX_API_ID_roctxRangeStop = 4,
  ROCTX_API_ID_NUMBER,
};

typedef struct roctx_api_data_s {
  union {
    struct {
      const char *message;
      roctx_range_id_t id;
    };
    struct {
      const char *message;
    } roctxMarkA;
    struct {
      const char *message;
    } roctxRangePushA;
    struct {
      const char *message;
    } roctxRangePop;
    struct {
      const char *message;
      roctx_range_id_t id;
    } roctxRangeStartA;
    struct {
      const char *message;
      roctx_range_id_t id;
    } roctxRangeStop;
  } args;
} roctx_api_data_t;

#endif // PROTON_DRIVER_GPU_ROCTX_TYPES_H_
