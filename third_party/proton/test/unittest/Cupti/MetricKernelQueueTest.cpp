#include "Profiler/Cupti/MetricKernelQueue.h"

#include <cstddef>
#include <deque>
#include <gtest/gtest.h>

using proton::detail::popMetricKernelNumWordsIfQueued;

TEST(MetricKernelQueueTest, IgnoresExtraGraphNodeCallbackDuringReceive) {
  // tensorMetrics has one metric with one value, so doAddMetrics queues one
  // metric-copy entry: metric_id + num_values = 2 words.
  constexpr size_t tensorMetricSize = 1;
  std::deque<size_t> metricKernelNumWordsQueue{/*metric_id=*/1 +
                                               tensorMetricSize};

  bool isMetricKernelLaunching = true;
  size_t metricCopyNumWords = 0;
  size_t extraCallbackNumWords = 123;
  bool metricCopyCallbackMatched = false;
  bool extraCallbackMatched = true;

  auto onGraphNodeCreated = [&](size_t &metricKernelNumWords) {
    return popMetricKernelNumWordsIfQueued(isMetricKernelLaunching,
                                           metricKernelNumWordsQueue,
                                           metricKernelNumWords);
  };

  auto receive = [&]() {
    // queue tensor metric:
    //   runtime->launchKernel(metric_copy_kernel)
    //     CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED
    //       consumes the queued [2] entry
    metricCopyCallbackMatched = onGraphNodeCreated(metricCopyNumWords);

    // Still inside receive(), before the caller can reset
    // isMetricKernelLaunching:
    //   CUDA/CUPTI reports another graph-node-created resource callback.
    // Old code used the flag alone and would call front() on this empty queue.
    extraCallbackMatched = onGraphNodeCreated(extraCallbackNumWords);
  };

  receive();
  isMetricKernelLaunching = false;

  EXPECT_TRUE(metricCopyCallbackMatched);
  EXPECT_EQ(metricCopyNumWords, 2);
  EXPECT_FALSE(extraCallbackMatched);
  EXPECT_EQ(extraCallbackNumWords, 123);
  EXPECT_TRUE(metricKernelNumWordsQueue.empty());
}
