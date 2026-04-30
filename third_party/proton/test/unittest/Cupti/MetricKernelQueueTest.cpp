#include "Profiler/Cupti/MetricKernelQueue.h"

#include <cstddef>
#include <deque>
#include <gtest/gtest.h>

using proton::detail::popMetricKernelNumWordsIfQueued;

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
