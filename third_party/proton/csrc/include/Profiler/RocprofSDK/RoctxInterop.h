#ifndef PROTON_PROFILER_ROCPROFSDK_ROCTX_INTEROP_H_
#define PROTON_PROFILER_ROCPROFSDK_ROCTX_INTEROP_H_

// Callback data layout for libroctx64.so's tracer callback mechanism.
// This mirrors roctx_api_data_t from roctracer/roctracer_roctx.h but is
// defined independently to avoid a dependency on the deprecated roctracer
// headers. When rocprofiler-sdk's native marker tracing supports
// force_configure, this header can be removed.

#include <cstdint>

namespace proton {
namespace roctx {

constexpr uint32_t kPushA = 1;
constexpr uint32_t kPop = 2;

struct ApiData {
  union {
    struct {
      const char *message;
    } roctxRangePushA;
    struct {
      const char *message;
    } roctxRangePop;
  } args;
};

using TracerCallbackFn = int (*)(uint32_t domain, uint32_t operationId,
                                 void *data);
using RegisterTracerCallbackFn = void (*)(TracerCallbackFn);

} // namespace roctx
} // namespace proton

#endif // PROTON_PROFILER_ROCPROFSDK_ROCTX_INTEROP_H_
