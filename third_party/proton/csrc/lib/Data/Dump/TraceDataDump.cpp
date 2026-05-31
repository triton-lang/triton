#include "Dump/TraceDataDump.h"
#include "Profiler/Graph.h"
#include "TraceDataIO/TraceWriter.h"
#include "Utility/Errors.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <sstream>
#include <type_traits>

namespace proton::trace_data_dump::details {

namespace {

constexpr uint64_t kCpuLaneBase = 0;
constexpr uint64_t kGraphLaneBase = 100000;
constexpr uint64_t kGpuLaneBase = 200000;

} // namespace

uint64_t getCpuLaneId(size_t threadId) { return kCpuLaneBase + threadId; }

uint64_t getGraphLaneId(size_t streamId) { return kGraphLaneBase + streamId; }

uint64_t getGpuLaneId(size_t streamId) { return kGpuLaneBase + streamId; }

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

} // namespace proton::trace_data_dump::details

namespace proton::trace_data_dump {

namespace {

std::vector<KernelTrace>
convertToTimelineTrace(std::vector<CycleEvent> &cycleEvents) {
  std::vector<KernelTrace> results;

  auto getInt64Value = [](const CycleMetric *metric,
                          CycleMetric::CycleMetricKind kind) {
    return std::get<uint64_t>(metric->getValue(kind));
  };

  auto getStringValue = [](const CycleMetric *metric,
                           CycleMetric::CycleMetricKind kind) {
    return std::get<std::string>(metric->getValue(kind));
  };

  auto getKernelId = [&](const CycleEvent &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::KernelId);
  };

  auto getBlockId = [&](const CycleEvent &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::BlockId);
  };

  auto getUnitId = [&](const CycleEvent &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::UnitId);
  };

  auto getStartCycle = [&](const CycleEvent &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::StartCycle);
  };

  auto getEndCycle = [&](const CycleEvent &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::EndCycle);
  };

  auto &sortedEvents = cycleEvents;
  std::sort(sortedEvents.begin(), sortedEvents.end(),
            [&](const CycleEvent &a, const CycleEvent &b) {
              auto aKernelId = getKernelId(a);
              auto bKernelId = getKernelId(b);
              if (aKernelId != bKernelId)
                return aKernelId < bKernelId;

              auto aBlockId = getBlockId(a);
              auto bBlockId = getBlockId(b);
              if (aBlockId != bBlockId)
                return aBlockId < bBlockId;

              auto aUnitId = getUnitId(a);
              auto bUnitId = getUnitId(b);
              if (aUnitId != bUnitId)
                return aUnitId < bUnitId;

              auto aStartCycle = getStartCycle(a);
              auto bStartCycle = getStartCycle(b);
              return aStartCycle < bStartCycle;
            });

  size_t eventIndex = 0;
  while (eventIndex < sortedEvents.size()) {
    auto kernelEvent = sortedEvents[eventIndex];
    auto currentKernelId = getKernelId(kernelEvent);

    auto parserResult = std::make_shared<CircularLayoutParserResult>();
    auto metadata = std::make_shared<KernelMetadata>();
    std::map<int, std::string> scopeIdToName;
    std::map<std::string, int> scopeNameToId;
    int curScopeId = 0;
    int64_t timeShiftCost =
        getInt64Value(kernelEvent.cycleMetric, CycleMetric::TimeShiftCost);

    while (eventIndex < sortedEvents.size() &&
           getKernelId(sortedEvents[eventIndex]) == currentKernelId) {
      const auto &blockEvent = sortedEvents[eventIndex];
      uint32_t currentBlockId = getBlockId(blockEvent);
      uint32_t currentProcId =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::ProcessorId);

      CircularLayoutParserResult::BlockTrace blockTrace;
      blockTrace.blockId = currentBlockId;
      blockTrace.procId = currentProcId;
      blockTrace.initTime =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::InitTime);
      blockTrace.preFinalTime =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::PreFinalTime);
      blockTrace.postFinalTime =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::PostFinalTime);
      blockTrace.traces.reserve(16);

      while (eventIndex < sortedEvents.size()) {
        const auto &currentEvent = sortedEvents[eventIndex];
        if (getKernelId(currentEvent) != currentKernelId ||
            getBlockId(currentEvent) != currentBlockId) {
          break;
        }

        const auto &uintEvent = sortedEvents[eventIndex];
        uint32_t currentUid = getUnitId(uintEvent);

        CircularLayoutParserResult::Trace unitTrace;
        unitTrace.uid = currentUid;
        unitTrace.profileEvents.reserve(256);

        while (eventIndex < sortedEvents.size()) {
          const auto &event = sortedEvents[eventIndex];
          if (getKernelId(event) != currentKernelId ||
              getBlockId(event) != currentBlockId ||
              getUnitId(event) != currentUid) {
            break;
          }

          auto scopeName = event.contexts.back().name;
          if (scopeNameToId.count(scopeName) == 0) {
            scopeIdToName[curScopeId] = scopeName;
            scopeNameToId[scopeName] = curScopeId;
            curScopeId++;
          }

          auto startEntry = std::make_shared<CycleEntry>();
          startEntry->cycle = getStartCycle(event);
          startEntry->isStart = true;
          startEntry->scopeId = scopeNameToId[scopeName];

          auto endEntry = std::make_shared<CycleEntry>();
          endEntry->cycle = getEndCycle(event);
          endEntry->isStart = false;
          endEntry->scopeId = scopeNameToId[scopeName];

          unitTrace.profileEvents.emplace_back(startEntry, endEntry);
          eventIndex++;
        }
        blockTrace.traces.push_back(std::move(unitTrace));
      }
      parserResult->blockTraces.push_back(std::move(blockTrace));
    }
    std::vector<std::string> callStack;
    if (!sortedEvents.empty()) {
      auto &contexts = kernelEvent.contexts;
      if (!contexts.empty()) {
        callStack.resize(contexts.size() - 1);
        std::transform(contexts.begin(), contexts.end() - 1, callStack.begin(),
                       [](const Context &c) { return c.name; });
      }
    }
    metadata->kernelName =
        getStringValue(kernelEvent.cycleMetric, CycleMetric::KernelName);
    metadata->scopeName = scopeIdToName;
    metadata->callStack = std::move(callStack);
    if (timeShiftCost > 0)
      timeShift(timeShiftCost, parserResult);
    results.emplace_back(parserResult, metadata);
  }
  return results;
}

} // namespace

void reconstructGraphScopeEvents(
    const std::map<size_t, std::vector<KernelEvent>> &kernelEvents,
    std::map<size_t, std::vector<GraphScopeEvent>> &graphScopeEvents) {
  struct OpenGraphScope {
    Context context;
    uint64_t startTimeNs{};
  };
  std::map</*stream_id=*/size_t, std::vector<OpenGraphScope>> openGraphScopes;
  for (const auto &[streamId, streamKernelEvents] : kernelEvents) {
    uint64_t lastEndTimeNs = 0;
    for (const auto &kernelEvent : streamKernelEvents) {
      if (!kernelEvent.isGraphLinked)
        continue;
      auto &openScopes = openGraphScopes[streamId];
      std::vector<Context> graphContexts;
      graphContexts.reserve(kernelEvent.contexts.size());
      bool seenCaptureTag = false;
      bool isMetadataKernel = false;
      for (const auto &context : kernelEvent.contexts) {
        if (context.name == GraphState::metricTag ||
            context.isMetadataState()) {
          isMetadataKernel = true;
          break;
        }
        if (context.name == GraphState::captureTag) {
          seenCaptureTag = true;
        }
        if (seenCaptureTag) {
          graphContexts.push_back(context);
        }
      }
      if (isMetadataKernel) {
        continue;
      }
      if (!seenCaptureTag) {
        throw makeLogicError("Invalid graph contexts without capture tag");
      }
      graphContexts.pop_back();
      auto startTimeNs = std::get<uint64_t>(
          kernelEvent.kernelMetric->getValue(KernelMetric::StartTime));
      auto endTimeNs = std::get<uint64_t>(
          kernelEvent.kernelMetric->getValue(KernelMetric::EndTime));
      if (openScopes.empty()) {
        for (const auto &context : graphContexts) {
          openScopes.push_back({context, startTimeNs});
        }
      } else {
        auto numCommonPrefixes = 0;
        while (numCommonPrefixes < openScopes.size() &&
               numCommonPrefixes < graphContexts.size()) {
          if (openScopes[numCommonPrefixes].context !=
              graphContexts[numCommonPrefixes]) {
            break;
          }
          numCommonPrefixes++;
        }
        for (size_t i = openScopes.size(); i > numCommonPrefixes; --i) {
          auto &tailOpenScope = openScopes[i - 1];
          graphScopeEvents[streamId].push_back({tailOpenScope.context, streamId,
                                                tailOpenScope.startTimeNs,
                                                lastEndTimeNs});
        }
        for (size_t i = openScopes.size(); i > numCommonPrefixes; --i) {
          openScopes.pop_back();
        }
        for (size_t i = numCommonPrefixes; i < graphContexts.size(); ++i) {
          const auto &context = graphContexts[i];
          openScopes.push_back({context, startTimeNs});
        }
      }
      lastEndTimeNs = std::max(lastEndTimeNs, endTimeNs);
    }

    auto &openScopes = openGraphScopes[streamId];
    for (const auto &openScope : openScopes) {
      graphScopeEvents[streamId].push_back(
          {openScope.context, streamId, openScope.startTimeNs, lastEndTimeNs});
    }
  }
}

void dumpCycleMetricTrace(std::vector<CycleEvent> &cycleEvents,
                          std::ostream &os) {
  auto timeline = convertToTimelineTrace(cycleEvents);
  auto writer = StreamChromeTraceWriter(timeline, "");
  writer.write(os);
}

} // namespace proton::trace_data_dump
