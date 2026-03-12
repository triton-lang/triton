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
    TraceEvent(size_t id, size_t contextId, std::vector<size_t> pendingEventIds)
        : id(id), contextId(contextId), pendingEventIds(pendingEventIds) {}
    size_t id = 0;
    size_t scopeId = Scope::DummyScopeId;
    size_t contextId = TraceContext::DummyId;
    std::vector<size_t> pendingEventIds;
    // Direct and linked metrics emitted for this trace event.
    DataEntry::MetricSet metricSet{};

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
                  const std::vector<size_t> &pendingEventIds) {
    traceEvents.try_emplace(nextEventId, nextEventId, contextId,
                            pendingEventIds);
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

  std::vector<size_t> &getPendingEventIds() const { return pendingEventIds; }

  const std::map<size_t, TraceEvent> &getEvents() const { return traceEvents; }

private:
  size_t nextTreeContextId = TraceContext::RootId + 1;
  size_t nextEventId = 0;
  std::map<size_t, TraceEvent> traceEvents;
  // tree node id -> trace context
  std::map<size_t, TraceContext> traceContextMap;
  std::vector<size_t> pendingEventIds;
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
  auto &pendingEventIds = currentTrace->getPendingEventIds();
  auto eventId = currentTrace->addEvent(currentTrace->addContexts(contexts),
                                        pendingEventIds);
  pendingEventIds.push_back(eventId);
  scopeIdToEventId[scope.scopeId] = eventId;
}

void TraceData::exitScope(const Scope &scope) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto iter = scopeIdToEventId.find(scope.scopeId);
  scopeIdToEventId.erase(iter); 
  auto *currentTrace = currentPhasePtrAs<Trace>();
  auto &pendingEventIds = currentTrace->getPendingEventIds();
  pendingEventIds.erase(
      std::remove(pendingEventIds.begin(), pendingEventIds.end(), iter->second),
      pendingEventIds.end());
}

DataEntry TraceData::addOp(size_t phase, size_t eventId,
                           const std::vector<Context> &contexts) {
  auto lock = lockIfCurrentOrVirtualPhase(phase);
  auto *trace = phasePtrAs<Trace>(phase);
  auto parentContextId = 0;
  auto pendingEventIds = trace->getPendingEventIds();
  if (eventId == Data::kRootEntryId) {
    parentContextId = Trace::TraceContext::RootId;
  } else {
    auto &event = trace->getEvent(eventId);
    pendingEventIds = event.pendingEventIds;
    parentContextId = event.contextId;
  }
  const auto contextId = trace->addContexts(contexts, parentContextId);
  const auto newEventId = trace->addEvent(contextId, pendingEventIds);
  auto &newEvent = trace->getEvent(newEventId);
  return DataEntry(newEventId, phase, newEvent.metricSet);
}

void TraceData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTrace = currentPhasePtrAs<Trace>();
  auto eventId = scopeIdToEventId.at(scopeId);
  auto &event = currentTrace->getEvent(eventId);
  auto &flexibleMetrics = event.metricSet.flexibleMetrics;
  for (auto [metricName, metricValue] : metrics) {
    if (flexibleMetrics.find(metricName) == flexibleMetrics.end()) {
      flexibleMetrics.emplace(metricName,
                              FlexibleMetric(metricName, metricValue));
    } else {
      flexibleMetrics.at(metricName).updateValue(metricValue);
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
  // Full call path captured for this cycle metric event.
  std::vector<Context> contexts;

  CycleMetricWithContext(const CycleMetric *metric, std::vector<Context> ctx)
      : cycleMetric(metric), contexts(std::move(ctx)) {}
};

struct KernelMetricWithContext {
  const KernelMetric *kernelMetric;
  // Full call path captured for this kernel metric event.
  std::vector<Context> contexts;

  KernelMetricWithContext(const KernelMetric *metric, std::vector<Context> ctx)
      : kernelMetric(metric), contexts(std::move(ctx)) {}
};

struct FlexibleMetricWithStartEndTime {
  const DataEntry::FlexibleMetricMap *flexibleMetrics;
  // Full call path captured for this flexible metric event.
  std::vector<Context> contexts;
  uint64_t startNs;
  uint64_t endNs;

  FlexibleMetricWithStartEndTime(const DataEntry::FlexibleMetricMap *metrics,
                                 std::vector<Context> ctx)
      : flexibleMetrics(metrics), contexts(std::move(ctx)),
        startNs(std::numeric_limits<uint64_t>::max()), endNs(0) {}
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
    const std::map<size_t, std::map<size_t, FlexibleMetricWithStartEndTime>>
        &streamFlexibleEvents,
    std::ostream &os) {
  auto metricValueToJson = [](const MetricValueType &value) -> json {
    return std::visit([](const auto &v) -> json { return v; }, value);
  };

  auto getDepth = [](const std::vector<Context> &contexts) -> size_t {
    // `contexts` includes the synthetic ROOT context.
    return contexts.size() > 1 ? contexts.size() - 2 : 0;
  };

  auto appendCallStack = [](json &args, const std::vector<Context> &contexts) {
    json callStack = json::array();
    for (const auto &ctx : contexts) {
      callStack.push_back(ctx.name);
    }
    args["call_stack"] = std::move(callStack);
  };

  // for each streamId in ascending order, emit one JSON line
  for (auto const &[streamId, events] : streamTraceEvents) {
    json object = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};
    std::vector<json> traceEvents;
    traceEvents.reserve(events.size());

    std::vector<std::reference_wrapper<const KernelMetricWithContext>>
        sortedKernelEvents;
    sortedKernelEvents.reserve(events.size());
    for (const auto &event : events) {
      sortedKernelEvents.push_back(std::cref(event));
    }
    std::sort(sortedKernelEvents.begin(), sortedKernelEvents.end(),
              [](const KernelMetricWithContext &lhs,
                 const KernelMetricWithContext &rhs) {
                const auto lhsStartNs = std::get<uint64_t>(
                    lhs.kernelMetric->getValue(KernelMetric::StartTime));
                const auto rhsStartNs = std::get<uint64_t>(
                    rhs.kernelMetric->getValue(KernelMetric::StartTime));
                if (lhsStartNs != rhsStartNs)
                  return lhsStartNs < rhsStartNs;
                const auto lhsEndNs = std::get<uint64_t>(
                    lhs.kernelMetric->getValue(KernelMetric::EndTime));
                const auto rhsEndNs = std::get<uint64_t>(
                    rhs.kernelMetric->getValue(KernelMetric::EndTime));
                return lhsEndNs < rhsEndNs;
              });

    for (const auto &eventRef : sortedKernelEvents) {
      const auto &event = eventRef.get();
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
      element["pid"] = "stream " + std::to_string(streamId);
      element["tid"] = "kernel";
      appendCallStack(element["args"], contexts);

      traceEvents.push_back(std::move(element));
    }

    if (auto flexibleIt = streamFlexibleEvents.find(streamId);
        flexibleIt != streamFlexibleEvents.end()) {
      std::vector<std::pair<size_t, const FlexibleMetricWithStartEndTime *>>
          sortedFlexibleEvents;
      sortedFlexibleEvents.reserve(flexibleIt->second.size());
      for (const auto &[eventId, event] : flexibleIt->second) {
        if (event.startNs == std::numeric_limits<uint64_t>::max() ||
            event.endNs < event.startNs) {
          continue;
        }
        sortedFlexibleEvents.emplace_back(eventId, &event);
      }

      std::sort(sortedFlexibleEvents.begin(), sortedFlexibleEvents.end(),
                [](const auto &lhs, const auto &rhs) {
                  if (lhs.second->startNs != rhs.second->startNs)
                    return lhs.second->startNs < rhs.second->startNs;
                  if (lhs.second->endNs != rhs.second->endNs)
                    return lhs.second->endNs < rhs.second->endNs;
                  return lhs.first < rhs.first;
                });

      traceEvents.reserve(traceEvents.size() + sortedFlexibleEvents.size());
      for (const auto &[_, flexibleEvent] : sortedFlexibleEvents) {
        const auto &contexts = flexibleEvent->contexts;
        if (contexts.empty()) {
          continue;
        }

        json element;
        element["name"] = contexts.back().name;
        element["cat"] = "flexible_metric";
        element["ph"] = "X";
        element["ts"] =
            static_cast<double>(flexibleEvent->startNs - minTimeStamp) / 1000;
        element["dur"] = static_cast<double>(flexibleEvent->endNs -
                                             flexibleEvent->startNs) /
                         1000;
        element["pid"] = "stream " + std::to_string(streamId);
        element["tid"] = "depth " + std::to_string(getDepth(contexts));

        for (const auto &[metricName, flexibleMetric] :
             *flexibleEvent->flexibleMetrics) {
          element["args"][metricName] =
              metricValueToJson(flexibleMetric.getValue(0));
        }

        traceEvents.push_back(std::move(element));
      }
    }

    std::sort(traceEvents.begin(), traceEvents.end(),
              [](const json &lhs, const json &rhs) {
                const auto lhsTs = lhs["ts"].get<double>();
                const auto rhsTs = rhs["ts"].get<double>();
                if (lhsTs != rhsTs)
                  return lhsTs < rhsTs;
                const auto lhsDur = lhs["dur"].get<double>();
                const auto rhsDur = rhs["dur"].get<double>();
                if (lhsDur != rhsDur)
                  return lhsDur < rhsDur;
                return lhs["name"].get<std::string>() <
                       rhs["name"].get<std::string>();
              });
    for (auto &element : traceEvents) {
      object["traceEvents"].push_back(std::move(element));
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
    // stream id -> trace event
    std::map<size_t, std::vector<KernelMetricWithContext>> streamTraceEvents; 
    std::map<size_t, std::map<size_t, FlexibleMetricWithStartEndTime>> streamFlexibleEvents;
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

      if (event.metricSet.metrics.find(MetricKind::Kernel) !=
              event.metricSet.metrics.end()) {
        auto kernelMetric = static_cast<KernelMetric *>(
            event.metricSet.metrics.at(MetricKind::Kernel).get());
        auto startNs =
            std::get<uint64_t>(kernelMetric->getValue(KernelMetric::StartTime));
        auto endNs =
            std::get<uint64_t>(kernelMetric->getValue(KernelMetric::EndTime));
        auto streamId =
            std::get<uint64_t>(kernelMetric->getValue(KernelMetric::StreamId));
        if (!event.pendingEventIds.empty()) {
          for (auto pendingEventId : event.pendingEventIds) {
            streamFlexibleEvents[streamId][pendingEventId].endNs = std::max(
                streamFlexibleEvents[streamId][pendingEventId].endNs, endNs);
            streamFlexibleEvents[streamId][pendingEventId].startNs = std::min(
                streamFlexibleEvents[streamId][pendingEventId].startNs, startNs);
          }
        }
        streamFlexibleEvents[streamId][event.id] =
            FlexibleMetricWithStartEndTime(&event.metricSet.flexibleMetrics,
                                           baseContexts);
      }
      processMetricMaps(event.metricSet.metrics, baseContexts);
      for (const auto &[targetEntryId, linkedMetrics] :
           event.metricSet.linkedMetrics) {
        auto contexts = baseContexts;
        auto &virtualContexts = targetIdToVirtualContexts[targetEntryId];
        contexts.insert(contexts.end(), virtualContexts.begin(),
                        virtualContexts.end());
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
      dumpKernelMetricTrace(minTimeStamp, streamTraceEvents, streamFlexibleEvents, os);
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
