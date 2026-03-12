#include "Context/Context.h"
#include "Data/Data.h"
#include "Data/PhaseStore.h"
#include "Profiler/Profiler.h"

#include <gtest/gtest.h>

#include <map>
#include <ostream>
#include <string>
#include <vector>

using namespace proton;

namespace {

class TestContextSource final : public ContextSource {
public:
  size_t getDepth() override { return 0; }

protected:
  std::vector<Context> getContextsImpl() override { return {}; }
};

class TestData final : public Data {
public:
  TestData() : Data("", &contextSource) { initPhaseStore(phaseStore); }

  DataEntry addOp(size_t phase, size_t entryId,
                  const std::vector<Context> &contexts) override {
    return DataEntry(entryId, phase, metricSet);
  }

  void addMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &metrics) override {}

  std::string toJsonString(size_t phase) const override { return "{}"; }

  std::vector<uint8_t> toMsgPack(size_t phase) const override { return {}; }

protected:
  void enterScope(const Scope &scope) override {}

  void exitScope(const Scope &scope) override {}

  void doDump(std::ostream &os, OutputFormat outputFormat,
              size_t phase) const override {}

  OutputFormat getDefaultOutputFormat() const override {
    return OutputFormat::Hatchet;
  }

private:
  mutable DataEntry::MetricSet metricSet{};
  TestContextSource contextSource;
  PhaseStore<int> phaseStore;
};

class TestProfiler final : public Profiler {
public:
  size_t pollCalls = 0;
  size_t flushCalls = 0;

protected:
  void doStart() override {}

  void doPoll() override { ++pollCalls; }

  void doFlush() override { ++flushCalls; }

  void doStop() override {}

  void doSetMode(const std::vector<std::string> &modeAndOptions) override {}

  void doAddMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &scalarMetrics,
      const std::map<std::string, TensorMetric> &tensorMetrics) override {}
};

} // namespace

TEST(ProfilerTest, PollDoesNotCompletePriorPhasesButFlushDoes) {
  TestProfiler profiler;
  TestData data;

  profiler.registerData(&data);

  ASSERT_EQ(data.advancePhase(), 1u);
  auto phaseInfo = data.getPhaseInfo();
  ASSERT_EQ(phaseInfo.current, 1u);
  ASSERT_EQ(phaseInfo.completeUpTo, Data::kNoCompletePhase);
  ASSERT_FALSE(phaseInfo.isComplete(0));

  profiler.poll();

  phaseInfo = data.getPhaseInfo();
  EXPECT_EQ(profiler.pollCalls, 1u);
  EXPECT_EQ(profiler.flushCalls, 0u);
  EXPECT_EQ(phaseInfo.current, 1u);
  EXPECT_EQ(phaseInfo.completeUpTo, Data::kNoCompletePhase);
  EXPECT_FALSE(phaseInfo.isComplete(0));

  profiler.flush();

  phaseInfo = data.getPhaseInfo();
  EXPECT_EQ(profiler.pollCalls, 1u);
  EXPECT_EQ(profiler.flushCalls, 1u);
  EXPECT_EQ(phaseInfo.current, 1u);
  EXPECT_EQ(phaseInfo.completeUpTo, 0u);
  EXPECT_TRUE(phaseInfo.isComplete(0));
}
