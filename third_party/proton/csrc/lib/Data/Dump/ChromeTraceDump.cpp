#include "Dump/TraceDataDump.h"
#include "Profiler/Graph.h"
#include "TraceDataIO/TraceWriter.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_map>

using json = nlohmann::json;

namespace proton::trace_data_dump {

namespace {

json buildCallStackJson(const std::vector<Context> &contexts) {
  json callStack = json::array();
  for (const auto &ctx : contexts) {
    callStack.push_back(ctx.name);
  }
  return callStack;
}

json buildFlexibleMetricsJson(
    const DataEntry::FlexibleMetricMap &flexibleMetrics) {
  json metrics = json::object();
  for (const auto &[metricName, metricValue] : flexibleMetrics) {
    metrics[metricName] =
        details::formatFlexibleMetricValue(metricValue.getValues()[0]);
  }
  return metrics;
}

void appendThreadMetadata(json &traceEvents, uint64_t tid,
                          const std::string &name, uint64_t sortIndex) {
  json nameEvent;
  nameEvent["ph"] = "M";
  nameEvent["pid"] = details::kTraceProcessId;
  nameEvent["tid"] = tid;
  nameEvent["name"] = "thread_name";
  nameEvent["args"]["name"] = name;
  traceEvents.push_back(std::move(nameEvent));

  json sortEvent;
  sortEvent["ph"] = "M";
  sortEvent["pid"] = details::kTraceProcessId;
  sortEvent["tid"] = tid;
  sortEvent["name"] = "thread_sort_index";
  sortEvent["args"]["sort_index"] = sortIndex;
  traceEvents.push_back(std::move(sortEvent));
}

void emitTraceLaneMetadata(
    json &object,
    const std::map<size_t, std::vector<CpuScopeEvent>> &cpuScopeEvents,
    const std::map<size_t, std::vector<GraphScopeEvent>> &graphScopeEvents,
    const std::map<size_t, std::vector<KernelEvent>> &kernelEvents) {
  auto &traceEvents = object["traceEvents"];

  json processNameEvent;
  processNameEvent["ph"] = "M";
  processNameEvent["pid"] = details::kTraceProcessId;
  processNameEvent["tid"] = 0;
  processNameEvent["name"] = "process_name";
  processNameEvent["args"]["name"] = "Trace";
  traceEvents.push_back(std::move(processNameEvent));

  json processSortEvent;
  processSortEvent["ph"] = "M";
  processSortEvent["pid"] = details::kTraceProcessId;
  processSortEvent["tid"] = 0;
  processSortEvent["name"] = "process_sort_index";
  processSortEvent["args"]["sort_index"] = 0;
  traceEvents.push_back(std::move(processSortEvent));

  for (const auto &[threadId, _] : cpuScopeEvents) {
    const auto tid = details::getCpuLaneId(threadId);
    appendThreadMetadata(traceEvents, tid,
                         "CPU Thread " + std::to_string(threadId), tid);
  }

  for (const auto &[streamId, _] : graphScopeEvents) {
    const auto tid = details::getGraphLaneId(streamId);
    appendThreadMetadata(traceEvents, tid,
                         "Graph: Stream " + std::to_string(streamId), tid);
  }

  for (const auto &[streamId, _] : kernelEvents) {
    const auto tid = details::getGpuLaneId(streamId);
    appendThreadMetadata(traceEvents, tid,
                         "GPU Stream " + std::to_string(streamId), tid);
  }
}

void dumpKernelEvents(
    uint64_t minTimeStamp,
    const std::map<size_t, std::vector<KernelEvent>> &kernelEvents,
    json &object) {
  for (const auto &[streamId, events] : kernelEvents) {
    for (const auto &event : events) {
      auto *kernelMetric = event.kernelMetric;
      auto *flexibleMetrics = event.flexibleMetrics;
      uint64_t startTimeNs =
          std::get<uint64_t>(kernelMetric->getValue(KernelMetric::StartTime));
      uint64_t endTimeNs =
          std::get<uint64_t>(kernelMetric->getValue(KernelMetric::EndTime));
      bool isMetricKernel = static_cast<bool>(std::get<uint64_t>(
          kernelMetric->getValue(KernelMetric::IsMetricKernel)));
      double ts = static_cast<double>(startTimeNs - minTimeStamp) / 1000;
      double dur = static_cast<double>(endTimeNs - startTimeNs) / 1000;

      const auto &contexts = event.contexts;

      json element;
      element["name"] =
          isMetricKernel ? GraphState::metricTag : contexts.back().name;
      element["cat"] = "kernel";
      element["ph"] = "X";
      element["pid"] = details::kTraceProcessId;
      element["ts"] = ts;
      element["dur"] = dur;
      element["tid"] = details::getGpuLaneId(streamId);
      element["args"]["call_stack"] = buildCallStackJson(contexts);
      if (flexibleMetrics) {
        element["args"]["metrics"] = buildFlexibleMetricsJson(*flexibleMetrics);
      }
      object["traceEvents"].push_back(element);
    }
  }
}

void dumpCpuScopeEvents(
    uint64_t minTimeStamp,
    const std::map<size_t, std::vector<CpuScopeEvent>> &cpuScopeEvents,
    json &object) {
  for (const auto &[threadId, events] : cpuScopeEvents) {
    for (const auto &event : events) {
      const auto *flexibleMetrics = event.flexibleMetrics;
      const double ts =
          static_cast<double>(event.startTimeNs - minTimeStamp) / 1000.0;
      const double dur =
          static_cast<double>(event.endTimeNs - event.startTimeNs) / 1000.0;

      json element;
      if (flexibleMetrics != nullptr && !flexibleMetrics->empty()) {
        element["name"] =
            details::buildFlexibleMetricEventName(event.contexts,
                                                  *flexibleMetrics);
        element["cat"] = "metric";
        element["args"]["metrics"] = buildFlexibleMetricsJson(*flexibleMetrics);
      } else {
        element["name"] =
            event.contexts.empty() ? "" : event.contexts.back().name;
        element["cat"] = "scope";
      }
      element["ph"] = "X";
      element["pid"] = details::kTraceProcessId;
      element["ts"] = ts;
      element["dur"] = dur;
      element["tid"] = details::getCpuLaneId(threadId);
      element["args"]["call_stack"] = buildCallStackJson(event.contexts);
      object["traceEvents"].push_back(std::move(element));
    }
  }
}

void dumpGraphScopeEvents(
    uint64_t minTimeStamp,
    const std::map<size_t, std::vector<GraphScopeEvent>> &graphScopeEvents,
    json &object) {
  for (const auto &[streamId, events] : graphScopeEvents) {
    const auto graphTid = details::getGraphLaneId(streamId);
    for (const auto &event : events) {
      json element;
      if (event.flexibleMetrics != nullptr && !event.flexibleMetrics->empty()) {
        element["name"] =
            details::buildFlexibleMetricEventName(event.context,
                                                  *event.flexibleMetrics);
        element["cat"] = "metric";
        element["args"]["metrics"] =
            buildFlexibleMetricsJson(*event.flexibleMetrics);
      } else {
        element["name"] = event.context.name;
        element["cat"] = "scope";
      }
      element["ph"] = "X";
      element["pid"] = details::kTraceProcessId;
      element["ts"] =
          static_cast<double>(event.startTimeNs - minTimeStamp) / 1000.0;
      element["dur"] =
          static_cast<double>(event.endTimeNs - event.startTimeNs) / 1000.0;
      element["tid"] = graphTid;
      object["traceEvents"].push_back(std::move(element));
    }
  }
}

void dumpCpuToGpuFlowEvents(
    uint64_t minTimeStamp,
    const std::map<size_t, std::vector<CpuScopeEvent>> &cpuScopeEvents,
    const std::map<size_t, std::vector<KernelEvent>> &kernelEvents,
    json &object) {
  std::unordered_map<size_t, const CpuScopeEvent *> launchEventIdToCpuScopeEvent;
  for (const auto &[_, events] : cpuScopeEvents) {
    for (const auto &event : events) {
      launchEventIdToCpuScopeEvent.emplace(event.eventId, &event);
    }
  }

  for (const auto &[streamId, events] : kernelEvents) {
    auto prevLaunchEventId = details::kNoLaunchEventId;
    for (const auto &event : events) {
      if (event.launchEventId == details::kNoLaunchEventId) {
        continue;
      }
      if (prevLaunchEventId == event.launchEventId && event.isGraphLinked) {
        continue;
      }
      auto launchEventIt =
          launchEventIdToCpuScopeEvent.find(event.launchEventId);
      if (launchEventIt == launchEventIdToCpuScopeEvent.end()) {
        throw std::runtime_error(
            "Cannot find CPU scope event for kernel launch event id: " +
            std::to_string(event.launchEventId));
      }

      const auto *launchEvent = launchEventIt->second;
      const auto kernelStartTimeNs = std::get<uint64_t>(
          event.kernelMetric->getValue(KernelMetric::StartTime));

      json startElement;
      startElement["name"] = "launch->kernel";
      startElement["cat"] = "flow";
      startElement["ph"] = "s";
      startElement["pid"] = details::kTraceProcessId;
      startElement["tid"] = details::getCpuLaneId(launchEvent->threadId);
      startElement["ts"] =
          static_cast<double>(launchEvent->startTimeNs - minTimeStamp) / 1000.0;
      startElement["id"] = event.launchEventId;
      startElement["bp"] = "e";
      object["traceEvents"].push_back(std::move(startElement));

      json finishElement;
      finishElement["name"] = "launch->kernel";
      finishElement["cat"] = "flow";
      finishElement["ph"] = "f";
      finishElement["pid"] = details::kTraceProcessId;
      finishElement["tid"] = details::getGpuLaneId(streamId);
      finishElement["ts"] =
          static_cast<double>(kernelStartTimeNs - minTimeStamp) / 1000.0;
      finishElement["id"] = event.launchEventId;
      finishElement["bp"] = "e";
      object["traceEvents"].push_back(std::move(finishElement));
      prevLaunchEventId = event.launchEventId;
    }
  }
}

void dumpKernelMetricTrace(
    uint64_t minTimeStamp,
    const std::map<size_t, std::vector<KernelEvent>> &kernelEvents,
    const std::map<size_t, std::vector<CpuScopeEvent>> &cpuScopeEvents,
    const std::map<size_t, std::vector<GraphScopeEvent>> &graphScopeEvents,
    std::ostream &os) {
  json object = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};

  emitTraceLaneMetadata(object, cpuScopeEvents, graphScopeEvents, kernelEvents);
  dumpCpuScopeEvents(minTimeStamp, cpuScopeEvents, object);
  dumpGraphScopeEvents(minTimeStamp, graphScopeEvents, object);
  dumpCpuToGpuFlowEvents(minTimeStamp, cpuScopeEvents, kernelEvents, object);
  dumpKernelEvents(minTimeStamp, kernelEvents, object);

  os << object.dump() << "\n";
}

void dumpCpuOnlyTrace(
    uint64_t minTimeStamp,
    const std::map<size_t, std::vector<CpuScopeEvent>> &cpuScopeEvents,
    std::ostream &os) {
  json object = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};
  dumpCpuScopeEvents(minTimeStamp, cpuScopeEvents, object);
  os << object.dump() << "\n";
}

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

              return getStartCycle(a) < getStartCycle(b);
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

        const auto &unitEvent = sortedEvents[eventIndex];
        uint32_t currentUid = getUnitId(unitEvent);

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

void dumpCycleEventTrace(std::vector<CycleEvent> &cycleEvents,
                         std::ostream &os) {
  auto timeline = convertToTimelineTrace(cycleEvents);
  auto writer = StreamChromeTraceWriter(timeline, "");
  writer.write(os);
}

void dumpChromeTraceData(const TraceDump &traceDump, std::ostream &os) {
  if (!traceDump.kernelEvents.empty()) {
    dumpKernelMetricTrace(traceDump.minTimeStamp, traceDump.kernelEvents,
                          traceDump.cpuScopeEvents, traceDump.graphScopeEvents,
                          os);
  } else {
    dumpCpuOnlyTrace(traceDump.minTimeStamp, traceDump.cpuScopeEvents, os);
  }
}

} // namespace proton::trace_data_dump
