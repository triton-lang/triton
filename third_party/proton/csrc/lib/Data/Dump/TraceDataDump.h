#ifndef PROTON_DATA_DUMP_TRACE_DATA_DUMP_H_
#define PROTON_DATA_DUMP_TRACE_DATA_DUMP_H_

#include "Context/Context.h"
#include "Data/Data.h"

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace proton::trace_data_dump {

namespace details {

inline constexpr uint64_t kTraceProcessId = 0;
inline constexpr size_t kNoLaunchEventId = std::numeric_limits<size_t>::max();

uint64_t getCpuLaneId(size_t threadId);
uint64_t getGraphLaneId(size_t streamId);
uint64_t getGpuLaneId(size_t streamId);

std::string formatFlexibleMetricValue(const MetricValueType &value);
std::string buildFlexibleMetricEventName(
    const std::vector<Context> &contexts,
    const DataEntry::FlexibleMetricMap &flexibleMetrics);
std::string buildFlexibleMetricEventName(
    const Context &context,
    const DataEntry::FlexibleMetricMap &flexibleMetrics);

} // namespace details

struct CycleEvent {
  std::vector<Context> contexts;
  const CycleMetric *cycleMetric;

  CycleEvent(const CycleMetric *metric, std::vector<Context> contexts)
      : contexts(std::move(contexts)), cycleMetric(metric) {}
};

struct KernelEvent {
  const KernelMetric *kernelMetric{};
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};
  std::vector<Context> contexts;
  size_t launchEventId{details::kNoLaunchEventId};
  bool isGraphLinked{};

  KernelEvent(const KernelMetric *metric,
              const DataEntry::FlexibleMetricMap *metrics,
              std::vector<Context> contexts, size_t launchId,
              bool isGraphLinked)
      : kernelMetric(metric), flexibleMetrics(metrics),
        contexts(std::move(contexts)), launchEventId(launchId),
        isGraphLinked(isGraphLinked) {}
};

struct CpuScopeEvent {
  size_t eventId;
  std::vector<Context> contexts;
  size_t threadId;
  uint64_t startTimeNs;
  uint64_t endTimeNs;
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};

  CpuScopeEvent(size_t eventId, const DataEntry::FlexibleMetricMap *metrics,
                std::vector<Context> contexts, size_t tid, uint64_t start,
                uint64_t end)
      : eventId(eventId), contexts(std::move(contexts)), threadId(tid),
        startTimeNs(start), endTimeNs(end), flexibleMetrics(metrics) {}
};

struct GraphScopeEvent {
  Context context;
  size_t streamId{};
  uint64_t startTimeNs{};
  uint64_t endTimeNs{};
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};
};

void reconstructGraphScopeEvents(
    const std::map<size_t, std::vector<KernelEvent>> &kernelEvents,
    std::map<size_t, std::vector<GraphScopeEvent>> &graphScopeEvents);

void dumpCycleMetricTrace(std::vector<CycleEvent> &cycleEvents,
                          std::ostream &os);
void dumpKernelMetricTrace(
    uint64_t minTimeStamp,
    const std::map<size_t, std::vector<KernelEvent>> &kernelEvents,
    const std::map<size_t, std::vector<CpuScopeEvent>> &cpuScopeEvents,
    const std::map<size_t, std::vector<GraphScopeEvent>> &graphScopeEvents,
    std::ostream &os);
void dumpCpuOnlyTrace(
    uint64_t minTimeStamp,
    const std::map<size_t, std::vector<CpuScopeEvent>> &cpuScopeEvents,
    std::ostream &os);
void dumpEmptyChromeTrace(std::ostream &os);

} // namespace proton::trace_data_dump

#endif // PROTON_DATA_DUMP_TRACE_DATA_DUMP_H_
