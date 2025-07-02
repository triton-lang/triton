#include "Data/TraceData.h"
#include "TraceDataIO/TraceWriter.h"
#include "Utility/Errors.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

using json = nlohmann::json;

namespace proton {

class TraceData::Trace {
public:
  struct TraceContext : public Context {
    inline static const size_t RootId = 0;
    inline static const size_t DummyId = std::numeric_limits<size_t>::max();

    TraceContext() = default;
    explicit TraceContext(size_t id, const std::string &name)
        : id(id), Context(name) {}
    TraceContext(size_t id, size_t parentId, const std::string &name)
        : id(id), parentId(parentId), Context(name) {}
    virtual ~TraceContext() = default;

    void addChild(const Context &context, size_t id) { children[context] = id; }

    bool hasChild(const Context &context) const {
      return children.find(context) != children.end();
    }

    size_t getChild(const Context &context) const {
      return children.at(context);
    }

    size_t getParent() const { return parentId; }

    size_t parentId = DummyId;
    size_t id = DummyId;
    std::map<Context, size_t> children = {};
    friend class Trace;
  };

  struct TraceEvent {
    TraceEvent() = default;
    TraceEvent(size_t id, size_t scopeId, size_t contextId)
        : id(id), scopeId(scopeId), contextId(contextId) {}
    size_t id = 0;
    size_t scopeId = Scope::DummyScopeId;
    size_t contextId = TraceContext::DummyId;
    std::map<MetricKind, std::shared_ptr<Metric>> metrics = {};
    std::map<std::string, FlexibleMetric> flexibleMetrics = {};

    const static inline size_t DummyId = std::numeric_limits<size_t>::max();
  };

  Trace() {
    traceContextMap.try_emplace(TraceContext::RootId, TraceContext::RootId,
                                "ROOT");
  }

  size_t addContext(const std::vector<Context> &contexts, size_t parentId) {
    for (const auto &context : contexts) {
      parentId = addContext(context, parentId);
    }
    return parentId;
  }

  size_t addContext(const Context &context, size_t parentId) {
    if (traceContextMap[parentId].hasChild(context)) {
      return traceContextMap[parentId].getChild(context);
    }
    auto id = nextContextId++;
    traceContextMap.try_emplace(id, id, parentId, context.name);
    traceContextMap[parentId].addChild(context, id);
    return id;
  }

  size_t addContext(const std::vector<Context> &indices) {
    auto parentId = TraceContext::RootId;
    for (auto index : indices) {
      parentId = addContext(index, parentId);
    }
    return parentId;
  }

  std::vector<Context> getContexts(size_t contextId) {
    std::vector<Context> contexts;
    auto it = traceContextMap.find(contextId);
    if (it == traceContextMap.end()) {
      throw std::runtime_error("Context not found");
    }
    std::reference_wrapper<TraceContext> context = it->second;
    contexts.push_back(context.get());
    while (context.get().parentId != TraceContext::DummyId) {
      context = traceContextMap[context.get().parentId];
      contexts.push_back(context.get());
    }
    std::reverse(contexts.begin(), contexts.end());
    return contexts;
  }

  void addEvent(size_t scopeId, size_t contextId) {
    if (scopeIdEventIdMap.count(scopeId))
      return;
    scopeIdEventIdMap[scopeId] = nextEventId;
    traceEvents.emplace_back(nextEventId, scopeId, contextId);
    nextEventId++;
  }

  bool hasEvent(size_t scopeId) {
    return scopeIdEventIdMap.find(scopeId) != scopeIdEventIdMap.end();
  }

  TraceEvent &getEvent(size_t scopeId) {
    if (!hasEvent(scopeId)) {
      throw std::runtime_error("Event not found");
    }
    return traceEvents[scopeIdEventIdMap[scopeId]];
  }

  std::vector<TraceEvent> &getEvents() { return traceEvents; }

private:
  size_t nextContextId = TraceContext::RootId + 1;
  size_t nextEventId = 0;
  std::vector<TraceEvent> traceEvents;
  // scope id -> event id
  std::unordered_map<size_t, size_t> scopeIdEventIdMap;
  // tree node id -> trace context
  std::map<size_t, TraceContext> traceContextMap;
};

void TraceData::enterScope(const Scope &scope) {
  // enterOp and addMetric maybe called from different threads
  std::unique_lock<std::shared_mutex> lock(mutex);
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  else
    contexts.push_back(scope.name);
  auto contextId = trace->addContext(contexts);
  scopeIdToContextId[scope.scopeId] = contextId;
}

void TraceData::exitScope(const Scope &scope) {}

size_t TraceData::addOp(size_t scopeId, const std::string &name) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  if (scopeIdIt == scopeIdToContextId.end()) {
    // Obtain the current context
    std::vector<Context> contexts;
    if (contextSource != nullptr)
      contexts = contextSource->getContexts();
    // If name is empty, this is a placeholder event. Add an op under the
    // current context
    if (!name.empty())
      contexts.emplace_back(name);
    scopeIdToContextId[scopeId] = trace->addContext(contexts);
  } else {
    // Add a new context under it and update the context
    scopeId = Scope::getNewScopeId();
    scopeIdToContextId[scopeId] =
        trace->addContext(Context(name), scopeIdIt->second);
  }
  if (!name.empty()) // not a placeholder event
    trace->addEvent(scopeId, scopeIdToContextId[scopeId]);
  return scopeId;
}

size_t TraceData::addOp(size_t scopeId, const std::vector<Context> &contexts) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  if (scopeIdIt == scopeIdToContextId.end()) {
    // Obtain the current context
    std::vector<Context> currentContexts;
    if (contextSource != nullptr)
      currentContexts = contextSource->getContexts();
    // Add an op under the current context
    if (!currentContexts.empty())
      std::merge(currentContexts.begin(), currentContexts.end(),
                 contexts.begin(), contexts.end(), currentContexts.begin());
    scopeIdToContextId[scopeId] = trace->addContext(currentContexts);
  } else {
    // Add a new context under it and update the context
    scopeId = Scope::getNewScopeId();
    scopeIdToContextId[scopeId] =
        trace->addContext(contexts, scopeIdIt->second);
  }
  if (!contexts.empty()) // not a placeholder event
    trace->addEvent(scopeId, scopeIdToContextId[scopeId]);
  return scopeId;
}

void TraceData::addMetric(size_t scopeId, std::shared_ptr<Metric> metric) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  if (scopeIdIt == scopeIdToContextId.end())
    return;
  if (!trace->hasEvent(scopeId))
    return;
  auto &event = trace->getEvent(scopeId);
  if (event.metrics.find(metric->getKind()) == event.metrics.end())
    event.metrics.emplace(metric->getKind(), metric);
  else
    event.metrics[metric->getKind()]->updateMetric(*metric);
}

void TraceData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  // The profile data is deactivated, ignore the metric
  if (scopeIdIt == scopeIdToContextId.end())
    return;
  auto contextId = scopeIdIt->second;
  if (!trace->hasEvent(scopeId))
    return;
  auto &event = trace->getEvent(scopeId);
  for (auto [metricName, metricValue] : metrics) {
    if (event.flexibleMetrics.find(metricName) == event.flexibleMetrics.end()) {
      event.flexibleMetrics.emplace(metricName,
                                    FlexibleMetric(metricName, metricValue));
    } else {
      event.flexibleMetrics.at(metricName).updateValue(metricValue);
    }
  }
}

void TraceData::clear() {
  std::unique_lock<std::shared_mutex> lock(mutex);
  scopeIdToContextId.clear();
}

namespace {

// Structure to pair CycleMetric with its context for processing
struct CycleMetricWithContext {
  std::shared_ptr<CycleMetric> cycleMetric;
  uint32_t contextId;

  CycleMetricWithContext(std::shared_ptr<CycleMetric> metric, uint32_t ctx)
      : cycleMetric(metric), contextId(ctx) {}
};

std::vector<KernelTrace>
convertToTimelineTrace(TraceData::Trace *trace,
                       std::vector<CycleMetricWithContext> &cycleEvents) {
  std::vector<KernelTrace> results;

  auto getInt64Value = [](const std::shared_ptr<CycleMetric> &metric,
                          CycleMetric::CycleMetricKind kind) {
    return std::get<uint64_t>(metric->getValue(kind));
  };

  auto getStringValue = [](const std::shared_ptr<CycleMetric> &metric,
                           CycleMetric::CycleMetricKind kind) {
    return std::get<std::string>(metric->getValue(kind));
  };

  auto getKernelId = [&](const CycleMetricWithContext &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::KernelId);
  };

  auto getBlockId = [&](const CycleMetricWithContext &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::BlockId);
  };

  auto getUnitId = [&](const CycleMetricWithContext &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::UnitId);
  };

  auto getStartCycle = [&](const CycleMetricWithContext &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::StartCycle);
  };

  auto getEndCycle = [&](const CycleMetricWithContext &event) {
    return getInt64Value(event.cycleMetric, CycleMetric::EndCycle);
  };

  // Pre-sort all events once
  auto &sortedEvents = cycleEvents;
  std::sort(
      sortedEvents.begin(), sortedEvents.end(),
      [&](const CycleMetricWithContext &a, const CycleMetricWithContext &b) {
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

  // Process in perfectly sorted order
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

    // Process all events for current kernel
    while (eventIndex < sortedEvents.size() &&
           getKernelId(sortedEvents[eventIndex]) == currentKernelId) {

      const auto &blockEvent = sortedEvents[eventIndex];
      uint32_t currentBlockId = getBlockId(blockEvent);
      uint32_t currentProcId =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::ProcessorId);

      CircularLayoutParserResult::BlockTrace blockTrace;
      blockTrace.blockId = currentBlockId;
      blockTrace.procId = currentProcId;
      // Conservative estimation of the number of warps in a CTA.
      blockTrace.traces.reserve(16);

      // Process all events for current block-proc
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
        // Estimation the number of events in a unit (warp).
        unitTrace.profileEvents.reserve(256);

        // Process all events for current uid
        while (eventIndex < sortedEvents.size()) {
          const auto &event = sortedEvents[eventIndex];
          if (getKernelId(event) != currentKernelId ||
              getBlockId(event) != currentBlockId ||
              getUnitId(event) != currentUid) {
            break;
          }

          auto scopeName = trace->getContexts(event.contextId).back().name;
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
    metadata->kernelName =
        getStringValue(kernelEvent.cycleMetric, CycleMetric::KernelName);
    metadata->scopeName = scopeIdToName;
    if (timeShiftCost > 0)
      timeShift(timeShiftCost, parserResult);
    results.emplace_back(parserResult, metadata);
  }
  return results;
}

void dumpCycleMetricTrace(TraceData::Trace *trace,
                          std::vector<CycleMetricWithContext> &cycleEvents,
                          std::ostream &os) {
  auto timeline = convertToTimelineTrace(trace, cycleEvents);
  auto writer = StreamChromeTraceWriter(timeline, "");
  writer.write(os);
}

void dumpKernelMetricTrace(
    TraceData::Trace *trace, uint64_t minTimeStamp,
    std::map<size_t, std::vector<TraceData::Trace::TraceEvent>>
        &streamTraceEvents,
    std::ostream &os) {
  // for each streamId in ascending order, emit one JSON line
  for (auto const &[streamId, events] : streamTraceEvents) {
    json object = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};

    for (auto const &event : events) {
      auto kernelMetrics = std::dynamic_pointer_cast<KernelMetric>(
          event.metrics.at(MetricKind::Kernel));
      uint64_t startTimeNs =
          std::get<uint64_t>(kernelMetrics->getValue(KernelMetric::StartTime));
      uint64_t endTimeNs =
          std::get<uint64_t>(kernelMetrics->getValue(KernelMetric::EndTime));
      uint64_t ts = startTimeNs - minTimeStamp;
      uint64_t dur = endTimeNs - startTimeNs;

      auto contextId = event.contextId;
      auto contexts = trace->getContexts(contextId);

      json element;
      element["name"] = contexts.back().name;
      element["cat"] = "kernel";
      element["ph"] = "X";
      element["ts"] = ts;
      element["dur"] = dur;
      element["tid"] = streamId; // thread id = stream
      json callStack = json::array();
      for (auto const &ctx : contexts) {
        callStack.push_back(ctx.name);
      }
      element["args"]["call_stack"] = std::move(callStack);

      object["traceEvents"].push_back(element);
    }

    // one JSON object per line
    os << object.dump() << "\n";
  }
}
} // namespace

void TraceData::dumpChromeTrace(std::ostream &os) const {
  auto &events = trace->getEvents();
  // stream id -> trace event
  std::map<size_t, std::vector<Trace::TraceEvent>> streamTraceEvents;
  uint64_t minTimeStamp = std::numeric_limits<uint64_t>::max();
  bool hasKernelMetrics = false, hasCycleMetrics = false;
  // Data structure for efficient cycle metrics conversion
  std::map<uint64_t, int> kernelBlockNum;
  std::vector<CycleMetricWithContext> cycleEvents;
  cycleEvents.reserve(events.size());
  for (auto &event : events) {
    if (event.metrics.count(MetricKind::Kernel)) {
      std::shared_ptr<KernelMetric> kernelMetric =
          std::dynamic_pointer_cast<KernelMetric>(
              event.metrics.at(MetricKind::Kernel));
      auto streamId =
          std::get<uint64_t>(kernelMetric->getValue(KernelMetric::StreamId));
      streamTraceEvents[streamId].push_back(event);

      uint64_t startTime =
          std::get<uint64_t>(kernelMetric->getValue(KernelMetric::StartTime));
      minTimeStamp = std::min(minTimeStamp, startTime);
      hasKernelMetrics = true;
    }
    if (event.metrics.count(MetricKind::Cycle)) {
      std::shared_ptr<CycleMetric> cycleMetric =
          std::dynamic_pointer_cast<CycleMetric>(
              event.metrics.at(MetricKind::Cycle));
      cycleEvents.emplace_back(cycleMetric, event.contextId);
      hasCycleMetrics = true;
    }

    if (hasKernelMetrics && hasCycleMetrics) {
      throw std::runtime_error("only one active metric type is supported");
    }
  }

  if (hasCycleMetrics) {
    dumpCycleMetricTrace(trace.get(), cycleEvents, os);
  }

  if (hasKernelMetrics) {
    dumpKernelMetricTrace(trace.get(), minTimeStamp, streamTraceEvents, os);
  }
}

void TraceData::doDump(std::ostream &os, OutputFormat outputFormat) const {
  if (outputFormat == OutputFormat::ChromeTrace) {
    dumpChromeTrace(os);
  } else {
    std::logic_error("Output format not supported");
  }
}

TraceData::TraceData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  trace = std::make_unique<Trace>();
}

TraceData::~TraceData() {}

} // namespace proton
