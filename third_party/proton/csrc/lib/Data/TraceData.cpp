#include "Data/TraceData.h"
#include "TraceDataIO/TraceWriter.h"
#include "Utility/MsgPackWriter.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
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
    TraceEvent(size_t id, size_t contextId) : id(id), contextId(contextId) {}
    size_t id = 0;
    size_t scopeId = Scope::DummyScopeId;
    size_t contextId = TraceContext::DummyId;
    std::map<MetricKind, std::unique_ptr<Metric>> metrics = {};
    std::map<std::string, FlexibleMetric> flexibleMetrics = {};
    std::map<size_t, std::map<MetricKind, std::unique_ptr<Metric>>>
        linkedTargetMetrics = {};
    std::map<size_t, std::map<std::string, FlexibleMetric>>
        linkedTargetFlexibleMetrics = {};

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

  size_t addEvent(size_t contextId) {
    traceEvents.emplace(nextEventId, TraceEvent(nextEventId, contextId));
    return nextEventId++;
  }

  bool hasEvent(size_t eventId) {
    return traceEvents.find(eventId) != traceEvents.end();
  }

  TraceEvent &getEvent(size_t eventId) {
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

void TraceData::enterScope(const Scope &scope) {
  // enterOp and addMetric maybe called from different threads
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTrace = currentPhasePtrAs<Trace>();
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  else
    contexts.push_back(scope.name);
  auto eventId = currentTrace->addEvent(currentTrace->addContexts(contexts));
  scopeIdToEventId[scope.scopeId] = eventId;
}

void TraceData::exitScope(const Scope &scope) {
  scopeIdToEventId.erase(scope.scopeId);
}

DataEntry TraceData::addOp(size_t phase, size_t eventId,
                           const std::vector<Context> &contexts) {
  auto lock = lockIfCurrentOrStaticPhase(phase);
  auto *trace = phasePtrAs<Trace>(phase);
  auto parentContextId = 0;
  if (eventId == Data::kRootEntryId) {
    parentContextId = Trace::TraceContext::RootId;
  } else {
    auto &event = trace->getEvent(eventId);
    parentContextId = event.contextId;
  }
  const auto contextId = trace->addContexts(contexts, parentContextId);
  const auto newEventId = trace->addEvent(contextId);
  auto &newEvent = trace->getEvent(newEventId);
  return DataEntry(newEventId, phase, newEvent.metrics,
                   newEvent.flexibleMetrics);
}

void TraceData::linkOp(
    size_t baseEntryId, const std::vector<size_t> &targetEntryIds,
    const std::function<void(const DataEntry &)> &onLinked) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  const auto phase = currentPhase.load(std::memory_order_relaxed);
  auto *trace = currentPhasePtrAs<Trace>();
  auto &baseEvent = trace->getEvent(baseEntryId);
  for (const auto targetEntryId : targetEntryIds) {
    auto &linkedMetrics = baseEvent.linkedTargetMetrics[targetEntryId];
    auto &linkedFlexibleMetrics =
        baseEvent.linkedTargetFlexibleMetrics[targetEntryId];
    if (onLinked) {
      onLinked(DataEntry(baseEntryId, phase, linkedMetrics,
                         linkedFlexibleMetrics));
    }
  }
}

void TraceData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTrace = currentPhasePtrAs<Trace>();
  auto eventId = scopeIdToEventId.at(scopeId);
  auto &event = currentTrace->getEvent(eventId);
  for (auto [metricName, metricValue] : metrics) {
    if (event.flexibleMetrics.find(metricName) == event.flexibleMetrics.end()) {
      event.flexibleMetrics.emplace(metricName,
                                    FlexibleMetric(metricName, metricValue));
    } else {
      event.flexibleMetrics.at(metricName).updateValue(metricValue);
    }
  }
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

// Structure to pair CycleMetric with its context for processing
struct CycleMetricWithContext {
  const CycleMetric *cycleMetric;
  std::vector<Context> contexts;

  CycleMetricWithContext(const CycleMetric *metric, std::vector<Context> ctx)
      : cycleMetric(metric), contexts(std::move(ctx)) {}
};

struct KernelMetricWithContext {
  const KernelMetric *kernelMetric;
  std::vector<Context> contexts;

  KernelMetricWithContext(const KernelMetric *metric, std::vector<Context> ctx)
      : kernelMetric(metric), contexts(std::move(ctx)) {}
};

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
    std::ostream &os) {
  // for each streamId in ascending order, emit one JSON line
  for (auto const &[streamId, events] : streamTraceEvents) {
    json object = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};

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
      element["ts"] = ts;
      element["dur"] = dur;
      element["tid"] = streamId; // thread id = stream
      json callStack = json::array();
      for (const auto &ctx : contexts) {
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

void TraceData::dumpChromeTrace(std::ostream &os, size_t phase) const {
  std::set<size_t> staticTargetEntryIds;
  tracePhases.withPtr(phase, [&](Trace *trace) {
    for (const auto &[_, event] : trace->getEvents()) {
      for (const auto &[targetEntryId, _] : event.linkedTargetMetrics) {
        staticTargetEntryIds.insert(targetEntryId);
      }
      for (const auto &[targetEntryId, _] : event.linkedTargetFlexibleMetrics) {
        staticTargetEntryIds.insert(targetEntryId);
      }
    }
  });

  std::map<size_t, std::vector<Context>> targetIdToStaticContexts;
  if (!staticTargetEntryIds.empty()) {
    tracePhases.withPtr(Data::kVirtualPhase, [&](Trace *staticTrace) {
      for (auto targetEntryId : staticTargetEntryIds) {
        auto contexts = staticTrace->getContexts(targetEntryId);
        if (!contexts.empty() && contexts.front().name == "ROOT") {
          contexts.erase(contexts.begin());
        }
        targetIdToStaticContexts.emplace(targetEntryId, std::move(contexts));
      }
    });
  }

  tracePhases.withPtr(phase, [&](Trace *trace) {
    auto &events = trace->getEvents();
    // stream id -> trace event
    std::map<size_t, std::vector<KernelMetricWithContext>> streamTraceEvents;
    uint64_t minTimeStamp = std::numeric_limits<uint64_t>::max();
    bool hasKernelMetrics = false, hasCycleMetrics = false;
    std::vector<CycleMetricWithContext> cycleEvents;
    cycleEvents.reserve(events.size());

    auto processMetricMaps =
        [&](const std::map<MetricKind, std::unique_ptr<Metric>> &metrics,
            const std::vector<Context> &contexts) {
          if (auto kernelIt = metrics.find(MetricKind::Kernel);
              kernelIt != metrics.end()) {
            auto *kernelMetric =
                static_cast<KernelMetric *>(kernelIt->second.get());
            const auto streamId = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::StreamId));
            streamTraceEvents[streamId].emplace_back(kernelMetric, contexts);
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
      processMetricMaps(event.metrics, baseContexts);
      for (const auto &[targetEntryId, linkedMetrics] :
           event.linkedTargetMetrics) {
        auto contexts = baseContexts;
        auto &staticContexts = targetIdToStaticContexts[targetEntryId];
        contexts.insert(contexts.end(), staticContexts.begin(),
                        staticContexts.end());
        processMetricMaps(linkedMetrics, contexts);
      }

      if (hasKernelMetrics && hasCycleMetrics) {
        throw std::runtime_error("only one active metric type is supported");
      }
    }

    if (hasCycleMetrics) {
      dumpCycleMetricTrace(cycleEvents, os);
    }

    if (hasKernelMetrics) {
      dumpKernelMetricTrace(minTimeStamp, streamTraceEvents, os);
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
