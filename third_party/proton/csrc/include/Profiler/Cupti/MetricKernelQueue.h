#ifndef PROTON_PROFILER_CUPTI_METRIC_KERNEL_QUEUE_H_
#define PROTON_PROFILER_CUPTI_METRIC_KERNEL_QUEUE_H_

#include <cstddef>
#include <deque>

namespace proton::detail {

inline bool
popMetricKernelNumWordsIfQueued(bool isMetricKernelLaunching,
                                std::deque<size_t> &metricKernelNumWordsQueue,
                                size_t &metricKernelNumWords) {
  if (!isMetricKernelLaunching || metricKernelNumWordsQueue.empty())
    return false;

  metricKernelNumWords = metricKernelNumWordsQueue.front();
  metricKernelNumWordsQueue.pop_front();
  return true;
}

} // namespace proton::detail

#endif // PROTON_PROFILER_CUPTI_METRIC_KERNEL_QUEUE_H_
