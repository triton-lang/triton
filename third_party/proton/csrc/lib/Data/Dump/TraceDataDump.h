#ifndef PROTON_DATA_DETAILS_TRACE_DATA_DUMP_H_
#define PROTON_DATA_DETAILS_TRACE_DATA_DUMP_H_

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

std::string formatFlexibleMetricValue(const MetricValueType &value);

std::string buildFlexibleMetricEventName(
    const std::vector<Context> &contexts,
    const DataEntry::FlexibleMetricMap &flexibleMetrics);
std::string buildFlexibleMetricEventName(
    const Context &context,
    const DataEntry::FlexibleMetricMap &flexibleMetrics);

uint64_t getCpuLaneId(size_t threadId);
uint64_t getGraphLaneId(size_t streamId);
uint64_t getGpuLaneId(size_t streamId);
uint64_t getGpuLaneId(size_t deviceId, size_t streamId);

} // namespace details

struct CycleEvent {
  std::vector<Context> contexts;
  const CycleMetric *cycleMetric;

  CycleEvent(const CycleMetric *metric, std::vector<Context> contexts)
      : contexts(std::move(contexts)), cycleMetric(metric) {}
};

struct KernelEvent {
  size_t eventId{};
  const KernelMetric *kernelMetric{};
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};
  std::vector<Context> contexts;
  size_t launchEventId{};
  bool isGraphLinked{};

  KernelEvent(size_t eventId, const KernelMetric *metric,
              const DataEntry::FlexibleMetricMap *metrics,
              std::vector<Context> contexts, size_t launchId,
              bool isGraphLinked)
      : eventId(eventId), kernelMetric(metric), flexibleMetrics(metrics),
        contexts(std::move(contexts)), launchEventId(launchId),
        isGraphLinked(isGraphLinked) {}

  uint64_t getStartTimeNs() const;
  uint64_t getEndTimeNs() const;
  uint64_t getDeviceId() const;
  uint64_t getStreamId() const;
  uint64_t getIsMetricKernel() const;
  std::string getName() const;

  static bool compare(const KernelEvent &a, const KernelEvent &b);
};

struct CpuScopeEvent {
  size_t eventId;
  size_t targetEventId{details::kNoLaunchEventId};
  std::vector<Context> contexts;
  size_t threadId;
  uint64_t startTimeNs;
  uint64_t endTimeNs;
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};

  CpuScopeEvent(size_t eventId, size_t targetEventId,
                const DataEntry::FlexibleMetricMap *metrics,
                std::vector<Context> contexts, size_t tid, uint64_t start,
                uint64_t end)
      : eventId(eventId), targetEventId(targetEventId),
        contexts(std::move(contexts)), threadId(tid), startTimeNs(start),
        endTimeNs(end), flexibleMetrics(metrics) {}
};

struct GraphScopeEvent {
  Context context;
  size_t streamId{};
  uint64_t startTimeNs{};
  uint64_t endTimeNs{};
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};
};

struct TraceDump {
  uint64_t minTimeStamp = std::numeric_limits<uint64_t>::max();
  std::map</*stream_id=*/size_t, std::vector<KernelEvent>> kernelEvents;
  std::map</*thread_id=*/size_t, std::vector<CpuScopeEvent>> cpuScopeEvents;
  std::map</*stream_id=*/size_t, std::vector<GraphScopeEvent>> graphScopeEvents;
  std::vector<CycleEvent> cycleEvents;
};

void dumpCycleEventTrace(std::vector<CycleEvent> &cycleEvents,
                         std::ostream &os);

void dumpChromeTraceData(const TraceDump &traceDump, std::ostream &os);
void dumpPerfettoTraceData(const TraceDump &traceDump, std::ostream &os);

} // namespace proton::trace_data_dump

#endif // PROTON_DATA_DETAILS_TRACE_DATA_DUMP_H_
