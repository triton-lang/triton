#include "Profiler/Profiler.h"

namespace proton {
thread_local void *Profiler::tensorMetricKernel = nullptr;
thread_local void *Profiler::scalarMetricKernel = nullptr;
thread_local void *Profiler::metricKernelStream = nullptr;
} // namespace proton
