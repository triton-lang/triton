#include "Profiler/Profiler.h"

namespace proton {
thread_local MetricKernelLaunchState Profiler::metricKernelLaunchState{};
} // namespace proton
