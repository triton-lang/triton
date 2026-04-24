#include "Dump/TraceDataDump.h"
#include "Profiler/Graph.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace proton::trace_data_dump::details {

namespace {

constexpr uint64_t kCpuLaneBase = 0;
constexpr uint64_t kGraphLaneBase = 100000;
constexpr uint64_t kGpuLaneBase = 200000;

} // namespace

std::string formatFlexibleMetricValue(const MetricValueType &value) {
  return std::visit(
      [](auto &&v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, uint64_t> ||
                      std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
          return std::to_string(v);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return v;
        } else if constexpr (std::is_same_v<T, std::vector<uint64_t>> ||
                             std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<double>>) {
          std::ostringstream ss;
          ss << "[";
          for (size_t i = 0; i < v.size(); ++i) {
            if (i != 0) {
              ss << ",";
            }
            ss << v[i];
          }
          ss << "]";
          return ss.str();
        } else {
          static_assert(sizeof(T) == 0, "Unsupported MetricValueType");
        }
      },
      value);
}

std::string buildFlexibleMetricEventName(
    const std::vector<Context> &contexts,
    const DataEntry::FlexibleMetricMap &flexibleMetrics) {
  const auto &scopeName =
      contexts.empty() ? GraphState::metricTag : contexts.back().name;
  std::ostringstream ss;
  ss << scopeName << ": ";
  bool isFirst = true;
  for (const auto &[metricName, metricValue] : flexibleMetrics) {
    if (!isFirst) {
      ss << ", ";
    }
    ss << "<" << metricName << ", "
       << formatFlexibleMetricValue(metricValue.getValues()[0]) << ">";
    isFirst = false;
  }
  return ss.str();
}

std::string buildFlexibleMetricEventName(
    const Context &context,
    const DataEntry::FlexibleMetricMap &flexibleMetrics) {
  std::ostringstream ss;
  ss << context.name << ": ";
  bool isFirst = true;
  for (const auto &[metricName, metricValue] : flexibleMetrics) {
    if (!isFirst) {
      ss << ", ";
    }
    ss << "<" << metricName << ", "
       << formatFlexibleMetricValue(metricValue.getValues()[0]) << ">";
    isFirst = false;
  }
  return ss.str();
}

uint64_t getCpuLaneId(size_t threadId) { return kCpuLaneBase + threadId; }

uint64_t getGraphLaneId(size_t streamId) { return kGraphLaneBase + streamId; }

uint64_t getGpuLaneId(size_t streamId) { return kGpuLaneBase + streamId; }

uint64_t getGpuLaneId(size_t deviceId, size_t streamId) {
  return kGpuLaneBase + deviceId * 1000000 + streamId;
}

} // namespace proton::trace_data_dump::details

namespace proton::trace_data_dump {

uint64_t KernelEvent::getStartTimeNs() const {
  return std::get<uint64_t>(kernelMetric->getValue(KernelMetric::StartTime));
}

uint64_t KernelEvent::getEndTimeNs() const {
  return std::get<uint64_t>(kernelMetric->getValue(KernelMetric::EndTime));
}

uint64_t KernelEvent::getDeviceId() const {
  return std::get<uint64_t>(kernelMetric->getValue(KernelMetric::DeviceId));
}

uint64_t KernelEvent::getStreamId() const {
  return std::get<uint64_t>(kernelMetric->getValue(KernelMetric::StreamId));
}

uint64_t KernelEvent::getIsMetricKernel() const {
  return std::get<uint64_t>(
      kernelMetric->getValue(KernelMetric::IsMetricKernel));
}

std::string KernelEvent::getName() const {
  if (getIsMetricKernel()) {
    return GraphState::metricTag;
  }
  return contexts.empty() ? "" : contexts.back().name;
}

bool KernelEvent::compare(const KernelEvent &a, const KernelEvent &b) {
  if (a.getStartTimeNs() != b.getStartTimeNs()) {
    return a.getStartTimeNs() < b.getStartTimeNs();
  }
  if (a.getEndTimeNs() != b.getEndTimeNs()) {
    return a.getEndTimeNs() < b.getEndTimeNs();
  }
  if (a.getDeviceId() != b.getDeviceId()) {
    return a.getDeviceId() < b.getDeviceId();
  }
  if (a.getStreamId() != b.getStreamId()) {
    return a.getStreamId() < b.getStreamId();
  }
  if (a.getIsMetricKernel() != b.getIsMetricKernel()) {
    return a.getIsMetricKernel() < b.getIsMetricKernel();
  }
  const auto aName = a.getName();
  const auto bName = b.getName();
  if (aName != bName) {
    return aName < bName;
  }
  if (a.eventId != b.eventId) {
    return a.eventId < b.eventId;
  }
  return std::less<const KernelMetric *>{}(a.kernelMetric, b.kernelMetric);
}

} // namespace proton::trace_data_dump
