#include "Data/TraceData.h"
#include "TraceDataIO/TraceWriter.h"
#include "Utility/MsgPackWriter.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

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
    using FlexibleMetricBatch = std::map<std::string, MetricValueType>;
    using FlexibleMetricBatches = std::vector<FlexibleMetricBatch>;

    TraceEvent() = default;
    TraceEvent(size_t id, size_t contextId, size_t parentEventId)
        : id(id), contextId(contextId), parentEventId(parentEventId) {}
    size_t id = 0;
    size_t scopeId = Scope::DummyScopeId;
    size_t contextId = TraceContext::DummyId;
    size_t parentEventId = DummyId;
    // Direct and linked metrics emitted for this trace event.
    DataEntry::MetricSet metricSet{};
    FlexibleMetricBatches flexibleMetricBatches{};

    const static inline size_t DummyId = std::numeric_limits<size_t>::max();
  };

  Trace() {
    traceContextMap.try_emplace(TraceContext::RootId, TraceContext::RootId,
                                "ROOT");
  }

  size_t addContext(const Context &context, size_t parentId) {
    if (traceContextMap[parentId].hasChild(context)) {
      return traceContextMap[parentId].getChild(context);
    }
    auto id = nextTreeContextId++;
    traceContextMap.try_emplace(id, id, parentId, context.name);
    traceContextMap[parentId].addChild(context, id);
    return id;
  }

  size_t addContexts(const std::vector<Context> &contexts, size_t parentId) {
    for (const auto &context : contexts) {
      parentId = addContext(context, parentId);
    }
    return parentId;
  }

  size_t addContexts(const std::vector<Context> &indices) {
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

  size_t addEvent(size_t contextId,
                  size_t parentEventId = TraceEvent::DummyId) {
    traceEvents.try_emplace(nextEventId, nextEventId, contextId, parentEventId);
    return nextEventId++;
  }

  bool hasEvent(size_t eventId) {
    return traceEvents.find(eventId) != traceEvents.end();
  }

  bool hasEvent(size_t eventId) const {
    return traceEvents.find(eventId) != traceEvents.end();
  }

  TraceEvent &getEvent(size_t eventId) {
    auto it = traceEvents.find(eventId);
    if (it == traceEvents.end()) {
      throw std::runtime_error("Event not found");
    }
    return it->second;
  }

  const TraceEvent &getEvent(size_t eventId) const {
    auto it = traceEvents.find(eventId);
    if (it == traceEvents.end()) {
      throw std::runtime_error("Event not found");
    }
    return it->second;
  }

  void removeEvent(size_t eventId) { traceEvents.erase(eventId); }

  const std::map<size_t, TraceEvent> &getEvents() const { return traceEvents; }

private:
  size_t nextTreeContextId = TraceContext::RootId + 1;
  size_t nextEventId = 0;
  std::map<size_t, TraceEvent> traceEvents;
  // tree node id -> trace context
  std::map<size_t, TraceContext> traceContextMap;
};

namespace {
size_t findParentEventId(const TraceData::Trace &trace,
                         const std::unordered_map<size_t, size_t> &scopeIdToEventId,
                         size_t parentContextId) {
  size_t parentEventId = TraceData::Trace::TraceEvent::DummyId;
  for (const auto &[_, eventId] : scopeIdToEventId) {
    if (!trace.hasEvent(eventId)) {
      continue;
    }
    if (trace.getEvent(eventId).contextId == parentContextId &&
        (parentEventId == TraceData::Trace::TraceEvent::DummyId ||
         eventId > parentEventId)) {
      parentEventId = eventId;
    }
  }
  return parentEventId;
}
} // namespace

void TraceData::enterScope(const Scope &scope) {
  // enterOp and addMetric maybe called from different threads
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTrace = currentPhasePtrAs<Trace>();
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  else
    contexts.push_back(scope.name);
  auto parentEventId = Trace::TraceEvent::DummyId;
  if (contexts.size() > 1) {
    std::vector<Context> parentContexts(contexts.begin(), contexts.end() - 1);
    const auto parentContextId = currentTrace->addContexts(parentContexts);
    parentEventId =
        findParentEventId(*currentTrace, scopeIdToEventId, parentContextId);
  }
  auto eventId = currentTrace->addEvent(currentTrace->addContexts(contexts),
                                        parentEventId);
  currentTrace->getEvent(eventId).scopeId = scope.scopeId;
  scopeIdToEventId[scope.scopeId] = eventId;
}

void TraceData::exitScope(const Scope &scope) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  scopeIdToEventId.erase(scope.scopeId);
}

DataEntry TraceData::addOp(size_t phase, size_t eventId,
                           const std::vector<Context> &contexts) {
  auto lock = lockIfCurrentOrVirtualPhase(phase);
  auto *trace = phasePtrAs<Trace>(phase);
  auto parentEventId = Trace::TraceEvent::DummyId;
  auto contextId = Trace::TraceContext::RootId;
  if (eventId == Data::kRootEntryId) {
    if (contexts.size() > 1) {
      std::vector<Context> parentContexts(contexts.begin(), contexts.end() - 1);
      const auto parentContextId = trace->addContexts(parentContexts);
      parentEventId = findParentEventId(*trace, scopeIdToEventId,
                                        parentContextId);
    }
    contextId = trace->addContexts(contexts);
  } else {
    auto &event = trace->getEvent(eventId);
    parentEventId = eventId;
    contextId = trace->addContexts(contexts, event.contextId);
  }
  const auto newEventId = trace->addEvent(contextId, parentEventId);
  auto &newEvent = trace->getEvent(newEventId);
  return DataEntry(newEventId, phase, newEvent.metricSet);
}

void TraceData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTrace = currentPhasePtrAs<Trace>();
  auto eventId = scopeIdToEventId.at(scopeId);
  auto &event = currentTrace->getEvent(eventId);
  event.flexibleMetricBatches.push_back(metrics);
}

std::string TraceData::toJsonString(size_t phase) const {
  std::ostringstream os;
  dumpChromeTrace(os, phase);
  return os.str();
}

std::vector<uint8_t> TraceData::toMsgPack(size_t phase) const {
  std::ostringstream os;
  dumpChromeTrace(os, phase);
  MsgPackWriter writer;
  writer.packStr(os.str());
  return std::move(writer).take();
}

namespace {

json metricValueToJson(const MetricValueType &value) {
  return std::visit([](const auto &v) { return json(v); }, value);
}

json flexibleMetricBatchToJson(
    const TraceData::Trace::TraceEvent::FlexibleMetricBatch &metrics) {
  json args = json::object();
  for (const auto &[metricName, metricValue] : metrics) {
    args[metricName] = metricValueToJson(metricValue);
  }
  return args;
}

std::vector<std::string> contextsToNames(const std::vector<Context> &contexts) {
  std::vector<std::string> names;
  names.reserve(contexts.size());
  for (const auto &context : contexts) {
    names.push_back(context.name);
  }
  return names;
}

json buildCallStackJson(const std::vector<std::string> &callStack) {
  json result = json::array();
  for (const auto &frame : callStack) {
    result.push_back(frame);
  }
  return result;
}

struct CycleMetricWithContext {
  const CycleMetric *cycleMetric;
  std::vector<Context> contexts;

  CycleMetricWithContext(const CycleMetric *metric, std::vector<Context> ctx)
      : cycleMetric(metric), contexts(std::move(ctx)) {}
};

struct KernelMetricWithContext {
  const KernelMetric *kernelMetric;
  size_t sourceEventId = TraceData::Trace::TraceEvent::DummyId;
  size_t linkedTargetEventId = TraceData::Trace::TraceEvent::DummyId;
  std::vector<Context> contexts;

  KernelMetricWithContext(const KernelMetric *metric, size_t sourceEventId,
                          size_t linkedTargetEventId, std::vector<Context> ctx)
      : kernelMetric(metric), sourceEventId(sourceEventId),
        linkedTargetEventId(linkedTargetEventId), contexts(std::move(ctx)) {}
};

struct CallPathBarRecord {
  std::string name;
  std::vector<std::string> callStack;
  json args = json::object();
  double ts = 0.0;
  double dur = 0.0;
  size_t depth = 0;
};

struct TimeSpanNs {
  uint64_t start = std::numeric_limits<uint64_t>::max();
  uint64_t end = 0;

  void include(uint64_t startNs, uint64_t endNs) {
    start = std::min(start, startNs);
    end = std::max(end, endNs);
  }
};

void appendCallPathBars(
    std::vector<CallPathBarRecord> &bars, const std::vector<Context> &contexts,
    const TraceData::Trace::TraceEvent::FlexibleMetricBatches &metricBatches,
    uint64_t minTimeStamp, const TimeSpanNs &span) {
  if (contexts.size() <= 1 ||
      span.start == std::numeric_limits<uint64_t>::max() ||
      span.end <= span.start) {
    return;
  }

  auto callStack = contextsToNames(contexts);
  callStack.erase(callStack.begin());
  if (callStack.empty()) {
    return;
  }

  auto appendBar = [&](json args) {
    bars.push_back(CallPathBarRecord{
        contexts.back().name,
        callStack,
        std::move(args),
        static_cast<double>(span.start - minTimeStamp) / 1000.0,
        static_cast<double>(span.end - span.start) / 1000.0,
        callStack.size() - 1,
    });
  };

  if (metricBatches.empty()) {
    appendBar(json::object());
    return;
  }

  for (const auto &metricBatch : metricBatches) {
    appendBar(flexibleMetricBatchToJson(metricBatch));
  }
}

std::vector<size_t> getAncestorEventIds(const TraceData::Trace &trace,
                                        size_t eventId) {
  std::vector<size_t> ancestors;
  while (eventId != TraceData::Trace::TraceEvent::DummyId) {
    const auto &event = trace.getEvent(eventId);
    ancestors.push_back(eventId);
    eventId = event.parentEventId;
  }
  return ancestors;
}

std::vector<KernelTrace>
convertToTimelineTrace(std::vector<CycleMetricWithContext> &cycleEvents) {
  std::vector<KernelTrace> results;

  auto getInt64Value = [](const CycleMetric *metric,
                          CycleMetric::CycleMetricKind kind) {
    return std::get<uint64_t>(metric->getValue(kind));
  };

  auto getStringValue = [](const CycleMetric *metric,
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
      blockTrace.initTime =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::InitTime);
      blockTrace.preFinalTime =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::PreFinalTime);
      blockTrace.postFinalTime =
          getInt64Value(blockEvent.cycleMetric, CycleMetric::PostFinalTime);
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

void dumpCycleMetricTrace(std::vector<CycleMetricWithContext> &cycleEvents,
                          std::ostream &os) {
  auto timeline = convertToTimelineTrace(cycleEvents);
  auto writer = StreamChromeTraceWriter(timeline, "");
  writer.write(os);
}

void dumpKernelMetricTrace(
    uint64_t minTimeStamp,
    const std::map<size_t, std::vector<KernelMetricWithContext>>
        &streamTraceEvents,
    const std::map<size_t, std::vector<CallPathBarRecord>> &streamCallPathBars,
    std::ostream &os) {
  static const std::vector<std::string> kChromeColor = {
      "cq_build_passed",
      "cq_build_failed",
      "thread_state_iowait",
      "thread_state_running",
      "thread_state_runnable",
      "thread_state_unknown",
      "rail_response",
      "rail_idle",
      "rail_load",
      "cq_build_attempt_passed",
      "cq_build_attempt_failed"};
  // for each streamId in ascending order, emit one JSON line
  for (auto const &[streamId, events] : streamTraceEvents) {
    json object = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};
    const auto pid = "stream " + std::to_string(streamId);

    if (auto barIt = streamCallPathBars.find(streamId);
        barIt != streamCallPathBars.end()) {
      for (const auto &bar : barIt->second) {
        json element;
        element["cname"] = kChromeColor[bar.depth % kChromeColor.size()];
        element["name"] = bar.name;
        element["cat"] = "call_path";
        element["ph"] = "X";
        element["pid"] = pid;
        element["tid"] = "path " + std::to_string(bar.depth);
        element["ts"] = bar.ts;
        element["dur"] = bar.dur;
        element["args"] = bar.args;
        element["args"]["call_stack"] = buildCallStackJson(bar.callStack);
        object["traceEvents"].push_back(element);
      }
    }

    for (const auto &event : events) {
      auto *kernelMetrics = event.kernelMetric;
      uint64_t startTimeNs =
          std::get<uint64_t>(kernelMetrics->getValue(KernelMetric::StartTime));
      uint64_t endTimeNs =
          std::get<uint64_t>(kernelMetrics->getValue(KernelMetric::EndTime));
      // Convert nanoseconds to microseconds for Chrome trace format
      double ts = static_cast<double>(startTimeNs - minTimeStamp) / 1000;
      double dur = static_cast<double>(endTimeNs - startTimeNs) / 1000;

      const auto &contexts = event.contexts;

      json element;
      element["name"] = contexts.back().name;
      element["cat"] = "kernel";
      element["ph"] = "X";
      element["pid"] = pid;
      element["ts"] = ts;
      element["dur"] = dur;
      element["tid"] = "kernels";
      element["args"]["call_stack"] =
          buildCallStackJson(contextsToNames(contexts));

      object["traceEvents"].push_back(element);
    }

    // one JSON object per line
    os << object.dump() << "\n";
  }
}
} // namespace

void TraceData::dumpChromeTrace(std::ostream &os, size_t phase) const {
  std::set<size_t> virtualTargetEntryIds;
  tracePhases.withPtr(phase, [&](Trace *trace) {
    for (const auto &[_, event] : trace->getEvents()) {
      for (const auto &[targetEntryId, _] : event.metricSet.linkedMetrics) {
        virtualTargetEntryIds.insert(targetEntryId);
      }
      for (const auto &[targetEntryId, _] :
           event.metricSet.linkedFlexibleMetrics) {
        virtualTargetEntryIds.insert(targetEntryId);
      }
    }
  });

  std::map<size_t, std::vector<Context>> targetIdToVirtualContexts;
  if (!virtualTargetEntryIds.empty()) {
    tracePhases.withPtr(Data::kVirtualPhase, [&](Trace *virtualTrace) {
      for (auto targetEntryId : virtualTargetEntryIds) {
        // Linked target ids are event ids, so resolve through the event first.
        auto &targetEvent = virtualTrace->getEvent(targetEntryId);
        auto contexts = virtualTrace->getContexts(targetEvent.contextId);
        contexts.erase(contexts.begin());
        targetIdToVirtualContexts.emplace(targetEntryId, std::move(contexts));
      }
    });
  }

  tracePhases.withPtr(phase, [&](Trace *trace) {
    auto &events = trace->getEvents();
    std::map<size_t, std::vector<KernelMetricWithContext>> streamTraceEvents;
    std::map<size_t, std::vector<CallPathBarRecord>> streamCallPathBars;
    uint64_t minTimeStamp = std::numeric_limits<uint64_t>::max();
    bool hasKernelMetrics = false, hasCycleMetrics = false;
    std::vector<CycleMetricWithContext> cycleEvents;
    cycleEvents.reserve(events.size());
    Trace *virtualTrace = nullptr;
    tracePhases.withPtr(Data::kVirtualPhase,
                        [&](Trace *tracePtr) { virtualTrace = tracePtr; });

    auto processMetricMaps =
        [&](const std::map<MetricKind, std::unique_ptr<Metric>> &metrics,
            const std::vector<Context> &contexts, size_t sourceEventId,
            size_t linkedTargetEventId) {
          if (auto kernelIt = metrics.find(MetricKind::Kernel);
              kernelIt != metrics.end()) {
            auto *kernelMetric =
                static_cast<KernelMetric *>(kernelIt->second.get());
            const auto streamId = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::StreamId));
            streamTraceEvents[streamId].emplace_back(
                kernelMetric, sourceEventId, linkedTargetEventId, contexts);
            const auto startTime = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::StartTime));
            minTimeStamp = std::min(minTimeStamp, startTime);
            hasKernelMetrics = true;
          }
          if (auto cycleIt = metrics.find(MetricKind::Cycle);
              cycleIt != metrics.end()) {
            auto *cycleMetric =
                static_cast<CycleMetric *>(cycleIt->second.get());
            cycleEvents.emplace_back(cycleMetric, contexts);
            hasCycleMetrics = true;
          }
        };

    for (const auto &[_, event] : events) {
      auto baseContexts = trace->getContexts(event.contextId);
      processMetricMaps(event.metricSet.metrics, baseContexts, event.id,
                        Trace::TraceEvent::DummyId);
      for (const auto &[targetEntryId, linkedMetrics] :
           event.metricSet.linkedMetrics) {
        auto contexts = baseContexts;
        auto &virtualContexts = targetIdToVirtualContexts[targetEntryId];
        contexts.insert(contexts.end(), virtualContexts.begin(),
                        virtualContexts.end());
        processMetricMaps(linkedMetrics, contexts, event.id, targetEntryId);
      }

      if (hasKernelMetrics && hasCycleMetrics) {
        throw std::runtime_error("only one active metric type is supported");
      }
    }

    if (hasCycleMetrics) {
      dumpCycleMetricTrace(cycleEvents, os);
    }

    if (hasKernelMetrics) {
      for (const auto &[streamId, kernelEvents] : streamTraceEvents) {
        std::map<size_t, TimeSpanNs> directScopeSpans;
        std::map<std::pair<size_t, size_t>, TimeSpanNs> linkedScopeSpans;
        auto &bars = streamCallPathBars[streamId];

        for (const auto &kernelEvent : kernelEvents) {
          const auto startTimeNs = std::get<uint64_t>(
              kernelEvent.kernelMetric->getValue(KernelMetric::StartTime));
          const auto endTimeNs = std::get<uint64_t>(
              kernelEvent.kernelMetric->getValue(KernelMetric::EndTime));

          auto directAncestors =
              getAncestorEventIds(*trace, kernelEvent.sourceEventId);
          if (kernelEvent.linkedTargetEventId == Trace::TraceEvent::DummyId &&
              !directAncestors.empty()) {
            directAncestors.erase(directAncestors.begin());
          }
          for (auto ancestorEventId : directAncestors) {
            directScopeSpans[ancestorEventId].include(startTimeNs, endTimeNs);
          }

          if (kernelEvent.linkedTargetEventId != Trace::TraceEvent::DummyId &&
              virtualTrace != nullptr) {
            auto targetEventId = kernelEvent.linkedTargetEventId;
            if (virtualTrace->hasEvent(targetEventId)) {
              auto virtualAncestors =
                  getAncestorEventIds(*virtualTrace, targetEventId);
              if (!virtualAncestors.empty()) {
                virtualAncestors.erase(virtualAncestors.begin());
              }
              for (auto virtualEventId : virtualAncestors) {
                linkedScopeSpans[{kernelEvent.sourceEventId, virtualEventId}]
                    .include(startTimeNs, endTimeNs);
              }
            }
          }
        }

        for (const auto &[scopeEventId, span] : directScopeSpans) {
          const auto &scopeEvent = trace->getEvent(scopeEventId);
          appendCallPathBars(bars, trace->getContexts(scopeEvent.contextId),
                             scopeEvent.flexibleMetricBatches, minTimeStamp,
                             span);
        }

        if (virtualTrace != nullptr) {
          for (const auto &[key, span] : linkedScopeSpans) {
            const auto &[sourceEventId, virtualEventId] = key;
            const auto &sourceEvent = trace->getEvent(sourceEventId);
            auto contexts = trace->getContexts(sourceEvent.contextId);
            auto virtualContexts = virtualTrace->getContexts(
                virtualTrace->getEvent(virtualEventId).contextId);
            if (!virtualContexts.empty()) {
              virtualContexts.erase(virtualContexts.begin());
            }
            contexts.insert(contexts.end(), virtualContexts.begin(),
                            virtualContexts.end());
            TraceData::Trace::TraceEvent::FlexibleMetricBatches metricBatches;
            if (const auto linkedFlexibleIt =
                    sourceEvent.metricSet.linkedFlexibleMetrics.find(
                        virtualEventId);
                linkedFlexibleIt !=
                sourceEvent.metricSet.linkedFlexibleMetrics.end()) {
              TraceData::Trace::TraceEvent::FlexibleMetricBatch metricBatch;
              for (const auto &[metricName, metric] :
                   linkedFlexibleIt->second) {
                metricBatch.emplace(metricName, metric.getValue(0));
              }
              metricBatches.push_back(std::move(metricBatch));
            }
            appendCallPathBars(bars, contexts, metricBatches, minTimeStamp,
                               span);
          }
        }

        std::sort(
            bars.begin(), bars.end(),
            [](const CallPathBarRecord &lhs, const CallPathBarRecord &rhs) {
              return std::tie(lhs.ts, lhs.depth, lhs.name, lhs.dur) <
                     std::tie(rhs.ts, rhs.depth, rhs.name, rhs.dur);
            });

        std::set<
            std::tuple<std::string, std::vector<std::string>, double, double>>
            barsWithMetrics;
        for (const auto &bar : bars) {
          if (!bar.args.empty()) {
            barsWithMetrics.emplace(bar.name, bar.callStack, bar.ts, bar.dur);
          }
        }
        bars.erase(std::remove_if(bars.begin(), bars.end(),
                                  [&](const CallPathBarRecord &bar) {
                                    return bar.args.empty() &&
                                           barsWithMetrics.find(std::make_tuple(
                                               bar.name, bar.callStack, bar.ts,
                                               bar.dur)) !=
                                               barsWithMetrics.end();
                                  }),
                   bars.end());
      }

      dumpKernelMetricTrace(minTimeStamp, streamTraceEvents, streamCallPathBars,
                            os);
    }
  });
}

void TraceData::doDump(std::ostream &os, OutputFormat outputFormat,
                       size_t phase) const {
  if (outputFormat == OutputFormat::ChromeTrace) {
    dumpChromeTrace(os, phase);
  } else {
    throw std::logic_error("Output format not supported");
  }
}

TraceData::TraceData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  initPhaseStore(tracePhases);
}

TraceData::~TraceData() {}

} // namespace proton
