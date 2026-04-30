#include "Profiler/Cupti/MetricKernelQueue.h"

#include <cstddef>
#include <deque>
#include <gtest/gtest.h>

using proton::detail::popMetricKernelNumWordsIfQueued;

TEST(MetricKernelQueueTest, IgnoresExtraGraphNodeCallbackAfterQueueIsConsumed) {
  // Simulates the concrete CUDA graph callback sequence:
  //
  //   isMetricKernelLaunching = true
  //   metricKernelNumWordsQueue = [2]
  //   receive() launches one Proton metric-copy kernel
  //     CUPTI GRAPHNODE_CREATED callback consumes [2]
  //   receive() has not returned yet, so isMetricKernelLaunching is still true
  //     CUPTI reports one more GRAPHNODE_CREATED callback
  //
  // The second callback is inside the metric-launch window, but it has no
  // matching metric-copy queue entry and must not be treated as <metric>.
  const bool isMetricKernelLaunching = true;
  std::deque<size_t> metricKernelNumWordsQueue{2};
  size_t metricKernelNumWords = 0;

  EXPECT_TRUE(popMetricKernelNumWordsIfQueued(isMetricKernelLaunching,
                                              metricKernelNumWordsQueue,
                                              metricKernelNumWords));
  EXPECT_EQ(metricKernelNumWords, 2);
  EXPECT_TRUE(metricKernelNumWordsQueue.empty());

  metricKernelNumWords = 123;
  EXPECT_FALSE(popMetricKernelNumWordsIfQueued(isMetricKernelLaunching,
                                               metricKernelNumWordsQueue,
                                               metricKernelNumWords));
  EXPECT_TRUE(metricKernelNumWordsQueue.empty());
  EXPECT_EQ(metricKernelNumWords, 123);
}

TEST(MetricKernelQueueTest, DoesNotPopEmptyQueueWhileMetricKernelLaunching) {
  // Reproduces the zen bench CUDA graph failure mode: CUPTI can report a
  // graph-node-created callback while metric-buffer receive still has
  // isMetricKernelLaunching set, but after all queued metric-copy entries were
  // consumed. The old CuptiProfiler.cpp path used front()/pop_front()
  // unconditionally when the flag was true, which segfaulted on this empty
  // queue.
  std::deque<size_t> metricKernelNumWordsQueue;
  size_t metricKernelNumWords = 123;

  EXPECT_FALSE(popMetricKernelNumWordsIfQueued(
      /*isMetricKernelLaunching=*/true, metricKernelNumWordsQueue,
      metricKernelNumWords));
  EXPECT_TRUE(metricKernelNumWordsQueue.empty());
  EXPECT_EQ(metricKernelNumWords, 123);
}

TEST(MetricKernelQueueTest, DoesNotPopQueueOutsideMetricKernelLaunch) {
  std::deque<size_t> metricKernelNumWordsQueue{7};
  size_t metricKernelNumWords = 123;

  EXPECT_FALSE(popMetricKernelNumWordsIfQueued(
      /*isMetricKernelLaunching=*/false, metricKernelNumWordsQueue,
      metricKernelNumWords));
  ASSERT_EQ(metricKernelNumWordsQueue.size(), 1);
  EXPECT_EQ(metricKernelNumWordsQueue.front(), 7);
  EXPECT_EQ(metricKernelNumWords, 123);
}

TEST(MetricKernelQueueTest, PopsQueuedMetricKernelWords) {
  std::deque<size_t> metricKernelNumWordsQueue{7, 11};
  size_t metricKernelNumWords = 0;

  EXPECT_TRUE(popMetricKernelNumWordsIfQueued(
      /*isMetricKernelLaunching=*/true, metricKernelNumWordsQueue,
      metricKernelNumWords));
  EXPECT_EQ(metricKernelNumWords, 7);
  ASSERT_EQ(metricKernelNumWordsQueue.size(), 1);
  EXPECT_EQ(metricKernelNumWordsQueue.front(), 11);
}
