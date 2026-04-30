#include "Profiler/Cupti/MetricKernelQueue.h"

#include <cstddef>
#include <deque>
#include <gtest/gtest.h>

using proton::detail::popMetricKernelNumWordsIfQueued;

TEST(MetricKernelQueueTest, MatchesMetricGraphNodesOnlyWithQueuedEntries) {
  // Simulates the concrete CUDA graph callback sequence:
  //
  //   isMetricKernelLaunching = true
  //   metricKernelNumWordsQueue = [2]
  //   receive() launches one Proton metric-copy kernel
  //     CUPTI GRAPHNODE_CREATED callback consumes [2]
  //   receive() has not returned yet, so isMetricKernelLaunching is still true
  //     CUPTI reports one more GRAPHNODE_CREATED callback
  //
  // The old CuptiProfiler.cpp path used front()/pop_front() whenever
  // isMetricKernelLaunching was true. That crashes on the second callback after
  // the queue is consumed. The queue entry, not just the flag, is what
  // identifies an actual <metric> node.
  {
    SCOPED_TRACE("callbacks outside the metric-launch window do not pop");
    std::deque<size_t> metricKernelNumWordsQueue{7};
    size_t metricKernelNumWords = 123;

    EXPECT_FALSE(popMetricKernelNumWordsIfQueued(
        /*isMetricKernelLaunching=*/false, metricKernelNumWordsQueue,
        metricKernelNumWords));
    ASSERT_EQ(metricKernelNumWordsQueue.size(), 1);
    EXPECT_EQ(metricKernelNumWordsQueue.front(), 7);
    EXPECT_EQ(metricKernelNumWords, 123);
  }

  {
    SCOPED_TRACE("queued callbacks are matched in FIFO order");
    std::deque<size_t> metricKernelNumWordsQueue{7, 11};
    size_t metricKernelNumWords = 0;

    EXPECT_TRUE(popMetricKernelNumWordsIfQueued(
        /*isMetricKernelLaunching=*/true, metricKernelNumWordsQueue,
        metricKernelNumWords));
    EXPECT_EQ(metricKernelNumWords, 7);
    ASSERT_EQ(metricKernelNumWordsQueue.size(), 1);
    EXPECT_EQ(metricKernelNumWordsQueue.front(), 11);
  }

  {
    SCOPED_TRACE("extra callback delivered before receive returns is ignored");
    bool isMetricKernelLaunching = true;
    std::deque<size_t> metricKernelNumWordsQueue{2};
    size_t firstCallbackNumWords = 0;
    size_t secondCallbackNumWords = 123;
    bool firstCallbackIsMetricNode = false;
    bool secondCallbackIsMetricNode = true;

    auto onGraphNodeCreated = [&](size_t &metricKernelNumWords) {
      return popMetricKernelNumWordsIfQueued(isMetricKernelLaunching,
                                             metricKernelNumWordsQueue,
                                             metricKernelNumWords);
    };

    auto receive = [&]() {
      // First callback: receive() launches the Proton metric-copy kernel, and
      // CUDA graph capture reports the matching GRAPHNODE_CREATED callback.
      firstCallbackIsMetricNode = onGraphNodeCreated(firstCallbackNumWords);

      // Second callback: still under receive(), before the caller can reset
      // isMetricKernelLaunching, CUPTI reports another GRAPHNODE_CREATED
      // callback. There is no queued metric-copy entry left for this node.
      secondCallbackIsMetricNode = onGraphNodeCreated(secondCallbackNumWords);
    };

    receive();
    isMetricKernelLaunching = false;

    EXPECT_TRUE(firstCallbackIsMetricNode);
    EXPECT_EQ(firstCallbackNumWords, 2);
    EXPECT_FALSE(secondCallbackIsMetricNode);
    EXPECT_EQ(secondCallbackNumWords, 123);
    EXPECT_TRUE(metricKernelNumWordsQueue.empty());
  }
}
