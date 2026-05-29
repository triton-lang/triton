#include "Dump/TraceDataDump.h"
#include "Profiler/Graph.h"
#include "Utility/Errors.h"
#include "nlohmann/json.hpp"

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
  for (auto const &[streamId, events] : kernelEvents) {
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
      if (isMetricKernel) {
        element["name"] = GraphState::metricTag;
      } else {
        element["name"] = contexts.back().name;
      }
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
        element["name"] = details::buildFlexibleMetricEventName(
            event.contexts, *flexibleMetrics);
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
  for (auto &[streamId, events] : graphScopeEvents) {
    const auto graphTid = details::getGraphLaneId(streamId);
    for (const auto &event : events) {
      json element;
      if (event.flexibleMetrics != nullptr && !event.flexibleMetrics->empty()) {
        element["name"] = details::buildFlexibleMetricEventName(
            event.context, *event.flexibleMetrics);
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
  std::unordered_map<size_t, const CpuScopeEvent *>
      launchEventIdToCpuScopeEvent;
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
        throw makeOutOfRange(
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

} // namespace

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

void dumpEmptyChromeTrace(std::ostream &os) {
  os << json({{"displayTimeUnit", "us"}, {"traceEvents", json::array()}}).dump()
     << "\n";
}

} // namespace proton::trace_data_dump
