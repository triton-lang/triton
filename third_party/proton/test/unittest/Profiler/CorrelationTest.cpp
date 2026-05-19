#include "Profiler/GPUProfiler.h"

#include <gtest/gtest.h>

#include <limits>

namespace {

class DummyProfiler : public proton::GPUProfiler<DummyProfiler> {
public:
  using proton::GPUProfiler<DummyProfiler>::Correlation;
};

using Correlation = DummyProfiler::Correlation;

TEST(CorrelationTest, FlushPurgesCompletedCorrelationsOnceAllActivitiesFinish) {
  Correlation correlation;
  proton::DataToEntryMap emptyDataToEntry;

  correlation.submit(1);
  correlation.submit(2);
  correlation.correlate(1, 101, std::numeric_limits<size_t>::max(),
                        /*isMissingName=*/false, emptyDataToEntry);
  correlation.correlate(2, 102, std::numeric_limits<size_t>::max(),
                        /*isMissingName=*/false, emptyDataToEntry);
  correlation.complete(2);

  correlation.flush(/*maxRetries=*/0, /*sleepUs=*/0, []() {});

  EXPECT_EQ(correlation.corrIdToExternId.size(), 0u);
  EXPECT_EQ(correlation.externIdToState.size(), 0u);
}

TEST(CorrelationTest, FlushDoesNotPurgeWhileSubmittedActivitiesAreIncomplete) {
  Correlation correlation;
  proton::DataToEntryMap emptyDataToEntry;

  correlation.submit(1);
  correlation.submit(2);
  correlation.correlate(1, 201, std::numeric_limits<size_t>::max(),
                        /*isMissingName=*/false, emptyDataToEntry);
  correlation.correlate(2, 202, std::numeric_limits<size_t>::max(),
                        /*isMissingName=*/false, emptyDataToEntry);
  correlation.complete(1);

  correlation.flush(/*maxRetries=*/0, /*sleepUs=*/0, []() {});

  EXPECT_EQ(correlation.corrIdToExternId.size(), 2u);
  EXPECT_EQ(correlation.externIdToState.size(), 2u);
}

} // namespace
