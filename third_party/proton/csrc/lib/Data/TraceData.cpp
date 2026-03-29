#include "Data/TraceData.h"
#include "Profiler/Graph.h"
#include "TraceDataIO/TraceWriter.h"
#include "Utility/MsgPackWriter.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <optional>
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
    TraceEvent(size_t id, size_t contextId, size_t parentEventId)
        : id(id), contextId(contextId), parentEventId(parentEventId) {}
    size_t id = 0;
    size_t scopeId = Scope::DummyScopeId;
    size_t contextId = TraceContext::DummyId;
    size_t parentEventId = DummyId;
    size_t launchScopeEventId = DummyId;
    uint64_t cpuStartTimeNs = 0;
    uint64_t cpuEndTimeNs = 0;
    uint64_t threadId = 0;
    bool hasCpuStartTime = false;
    bool hasCpuEndTime = false;
    // Direct and linked metrics emitted for this trace event.
    DataEntry::MetricSet metricSet{};

    bool hasCpuTimeRange() const {
      return hasCpuStartTime && hasCpuEndTime && cpuEndTimeNs >= cpuStartTimeNs;
    }

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

thread_local std::unordered_map<const TraceData *, std::vector<size_t>>
    traceDataToActiveEventStack;

uint64_t getCurrentCpuTimestampNs() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

void TraceData::enterScope(const Scope &scope) {
  // enterOp and addMetric maybe called from different threads
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTrace = currentPhasePtrAs<Trace>();
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  else
    contexts.push_back(scope.name);
  auto &activeEventStack = traceDataToActiveEventStack[this];
  size_t parentEventId = activeEventStack.empty() ? Trace::TraceEvent::DummyId
                                                  : activeEventStack.back();
  auto eventId =
      currentTrace->addEvent(currentTrace->addContexts(contexts), parentEventId);
  auto &event = currentTrace->getEvent(eventId);
  event.scopeId = scope.scopeId;
  event.cpuStartTimeNs = getCurrentCpuTimestampNs();
  event.hasCpuStartTime = true;
  event.threadId = getCurrentThreadTraceId();
  scopeIdToEventId[scope.scopeId] = eventId;
  activeEventStack.push_back(eventId);
}

void TraceData::exitScope(const Scope &scope) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeEventIt = scopeIdToEventId.find(scope.scopeId);
  if (scopeEventIt != scopeIdToEventId.end()) {
    auto *currentTrace = currentPhasePtrAs<Trace>();
    auto &event = currentTrace->getEvent(scopeEventIt->second);
    event.cpuEndTimeNs = getCurrentCpuTimestampNs();
    event.hasCpuEndTime = true;
    auto &activeEventStack = traceDataToActiveEventStack[this];
    if (!activeEventStack.empty() &&
        activeEventStack.back() == scopeEventIt->second) {
      activeEventStack.pop_back();
      if (activeEventStack.empty()) {
        traceDataToActiveEventStack.erase(this);
      }
    }
  }
  scopeIdToEventId.erase(scope.scopeId);
}

DataEntry TraceData::addOp(size_t phase, size_t eventId,
                           const std::vector<Context> &contexts) {
  auto lock = lockIfCurrentOrVirtualPhase(phase);
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
  if (eventId == Data::kRootEntryId) {
    newEvent.cpuStartTimeNs = getCurrentCpuTimestampNs();
    newEvent.hasCpuStartTime = true;
    newEvent.threadId = getCurrentThreadTraceId();
    auto activeEventStackIt = traceDataToActiveEventStack.find(this);
    if (activeEventStackIt != traceDataToActiveEventStack.end() &&
        !activeEventStackIt->second.empty()) {
      newEvent.launchScopeEventId = activeEventStackIt->second.back();
    }
  } else {
    const auto &parentEvent = trace->getEvent(eventId);
    if (parentEvent.scopeId != Scope::DummyScopeId) {
      newEvent.launchScopeEventId = eventId;
    } else {
      newEvent.launchScopeEventId = parentEvent.launchScopeEventId;
    }
    newEvent.cpuStartTimeNs = parentEvent.cpuStartTimeNs;
    newEvent.hasCpuStartTime = parentEvent.hasCpuStartTime;
    newEvent.threadId = parentEvent.threadId;
  }
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

uint64_t TraceData::getCurrentThreadTraceId() {
  auto threadId = std::this_thread::get_id();
  auto it = threadIdToTraceId.find(threadId);
  if (it != threadIdToTraceId.end()) {
    return it->second;
  }
  auto traceThreadId = nextThreadTraceId++;
  threadIdToTraceId.emplace(threadId, traceThreadId);
  return traceThreadId;
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

constexpr const char *kCpuThreadTidPrefix = "cpu thread ";

struct OrderedTraceEvent {
  enum class Kind { Kernel, CpuMetricScope };

  Kind kind;
  const KernelMetric *kernelMetric{};
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};
  std::vector<Context> contexts;
  size_t owningEventId = TraceData::Trace::TraceEvent::DummyId;
  size_t launchScopeEventId = TraceData::Trace::TraceEvent::DummyId;
  uint64_t threadId = 0;
  size_t streamId = 0;
  uint64_t startTimeNs = 0;
  uint64_t endTimeNs = 0;

  static OrderedTraceEvent kernel(const KernelMetric *metric,
                                  const DataEntry::FlexibleMetricMap *metrics,
                                  std::vector<Context> contexts,
                                  size_t owningEventId, size_t streamId,
                                  size_t launchScopeEventId) {
    OrderedTraceEvent event;
    event.kind = Kind::Kernel;
    event.kernelMetric = metric;
    event.flexibleMetrics = metrics;
    event.contexts = std::move(contexts);
    event.owningEventId = owningEventId;
    event.launchScopeEventId = launchScopeEventId;
    event.streamId = streamId;
    event.startTimeNs =
        std::get<uint64_t>(metric->getValue(KernelMetric::StartTime));
    event.endTimeNs =
        std::get<uint64_t>(metric->getValue(KernelMetric::EndTime));
    return event;
  }

  static OrderedTraceEvent cpuMetricScope(
      const DataEntry::FlexibleMetricMap *metrics, std::vector<Context> contexts,
      uint64_t threadId, uint64_t startTimeNs, uint64_t endTimeNs) {
    OrderedTraceEvent event;
    event.kind = Kind::CpuMetricScope;
    event.flexibleMetrics = metrics;
    event.contexts = std::move(contexts);
    event.threadId = threadId;
    event.startTimeNs = startTimeNs;
    event.endTimeNs = endTimeNs;
    return event;
  }
};

std::string formatFlexibleMetricValue(const MetricValueType &value) {
  return std::visit(
      [](auto &&v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, double>) {
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

json buildCallStackJson(const std::vector<Context> &contexts) {
  json callStack = json::array();
  for (const auto &ctx : contexts) {
    callStack.push_back(ctx.name);
  }
  return callStack;
}

json buildFlexibleMetricsJson(const DataEntry::FlexibleMetricMap &flexibleMetrics) {
  json metrics = json::object();
  for (const auto &[metricName, metricValue] : flexibleMetrics) {
    metrics[metricName] = formatFlexibleMetricValue(metricValue.getValues()[0]);
  }
  return metrics;
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

size_t findNearestMetricScopeEventId(
    size_t scopeEventId,
    const std::map<size_t, TraceData::Trace::TraceEvent> &events) {
  auto currentEventId = scopeEventId;
  while (currentEventId != TraceData::Trace::TraceEvent::DummyId) {
    auto eventIt = events.find(currentEventId);
    if (eventIt == events.end()) {
      break;
    }
    const auto &event = eventIt->second;
    if (!event.metricSet.flexibleMetrics.empty() && event.hasCpuTimeRange()) {
      return currentEventId;
    }
    currentEventId = event.parentEventId;
  }
  return TraceData::Trace::TraceEvent::DummyId;
}

void appendLaunchScopeArgs(
    json &args, size_t launchScopeEventId,
    const std::map<size_t, TraceData::Trace::TraceEvent> &events,
    const std::unordered_map<size_t, std::vector<Context>> &eventIdToContexts) {
  const auto metricScopeEventId =
      findNearestMetricScopeEventId(launchScopeEventId, events);
  if (metricScopeEventId == TraceData::Trace::TraceEvent::DummyId) {
    return;
  }
  const auto &scopeEvent = events.at(metricScopeEventId);
  args["launch_scope_event_id"] = metricScopeEventId;
  args["launch_scope_thread_id"] = scopeEvent.threadId;
  args["launch_scope_call_stack"] =
      buildCallStackJson(eventIdToContexts.at(metricScopeEventId));
  args["launch_scope_metrics"] =
      buildFlexibleMetricsJson(scopeEvent.metricSet.flexibleMetrics);
}

std::optional<int64_t> computeKernelClockOffsetNs(
    const std::vector<OrderedTraceEvent> &orderedTraceEvents,
    const std::map<size_t, TraceData::Trace::TraceEvent> &events) {
  std::vector<int64_t> offsets;
  offsets.reserve(orderedTraceEvents.size());
  for (const auto &event : orderedTraceEvents) {
    if (event.kind != OrderedTraceEvent::Kind::Kernel) {
      continue;
    }
    auto traceEventIt = events.find(event.owningEventId);
    if (traceEventIt == events.end() || !traceEventIt->second.hasCpuStartTime) {
      continue;
    }
    offsets.push_back(static_cast<int64_t>(traceEventIt->second.cpuStartTimeNs) -
                      static_cast<int64_t>(event.startTimeNs));
  }
  if (offsets.empty()) {
    return std::nullopt;
  }
  auto middle = offsets.begin() + offsets.size() / 2;
  std::nth_element(offsets.begin(), middle, offsets.end());
  return *middle;
}

uint64_t getAlignedStartTimeNs(const OrderedTraceEvent &event,
                               const std::optional<int64_t> &kernelClockOffsetNs) {
  if (event.kind != OrderedTraceEvent::Kind::Kernel || !kernelClockOffsetNs) {
    return event.startTimeNs;
  }
  const auto alignedStart =
      static_cast<int64_t>(event.startTimeNs) + *kernelClockOffsetNs;
  return alignedStart < 0 ? 0 : static_cast<uint64_t>(alignedStart);
}

void dumpKernelMetricTrace(
    const std::vector<OrderedTraceEvent> &orderedTraceEvents,
    const std::map<size_t, TraceData::Trace::TraceEvent> &events,
    const std::unordered_map<size_t, std::vector<Context>> &eventIdToContexts,
    std::ostream &os) {
  json object = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};
  const bool hasCpuMetricScopes = std::any_of(
      orderedTraceEvents.begin(), orderedTraceEvents.end(),
      [](const auto &event) {
        return event.kind == OrderedTraceEvent::Kind::CpuMetricScope;
      });
  const auto kernelClockOffsetNs =
      hasCpuMetricScopes ? computeKernelClockOffsetNs(orderedTraceEvents, events)
                         : std::nullopt;
  auto minTimeStamp = std::numeric_limits<uint64_t>::max();
  for (const auto &event : orderedTraceEvents) {
    minTimeStamp =
        std::min(minTimeStamp, getAlignedStartTimeNs(event, kernelClockOffsetNs));
  }
  for (const auto &event : orderedTraceEvents) {
    if (event.kind == OrderedTraceEvent::Kind::CpuMetricScope &&
        !event.flexibleMetrics) {
      continue;
    }

    json element;
    if (event.kind == OrderedTraceEvent::Kind::Kernel) {
      if (event.flexibleMetrics) {
        element["name"] = GraphState::metricTag;
      } else {
        element["name"] = event.contexts.back().name;
      }
      element["cat"] = "kernel";
      element["tid"] = event.streamId;
      appendLaunchScopeArgs(element["args"], event.launchScopeEventId, events,
                            eventIdToContexts);
    } else {
      element["name"] = GraphState::metricTag;
      element["cat"] = "metric";
      element["tid"] =
          std::string(kCpuThreadTidPrefix) + std::to_string(event.threadId);
      element["args"]["thread_id"] = event.threadId;
    }
    const auto alignedStartTimeNs =
        getAlignedStartTimeNs(event, kernelClockOffsetNs);
    element["ph"] = "X";
    element["ts"] =
        static_cast<double>(alignedStartTimeNs - minTimeStamp) / 1000.0;
    element["dur"] =
        static_cast<double>(event.endTimeNs - event.startTimeNs) / 1000.0;
    element["args"]["call_stack"] = buildCallStackJson(event.contexts);
    if (event.flexibleMetrics) {
      element["args"]["metrics"] =
          buildFlexibleMetricsJson(*event.flexibleMetrics);
    }
    object["traceEvents"].push_back(element);
  }

  os << object.dump() << "\n";
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
    std::vector<OrderedTraceEvent> orderedTraceEvents;
    orderedTraceEvents.reserve(events.size() * 2);
    std::unordered_map<size_t, std::vector<Context>> eventIdToContexts;
    eventIdToContexts.reserve(events.size());
    for (const auto &[eventId, event] : events) {
      eventIdToContexts.emplace(eventId, trace->getContexts(event.contextId));
    }
    bool hasKernelMetrics = false, hasCycleMetrics = false,
         hasCpuMetricScopes = false;
    std::vector<CycleMetricWithContext> cycleEvents;
    cycleEvents.reserve(events.size());

    auto processMetricMaps =
        [&](size_t owningEventId, const DataEntry::MetricMap &metrics,
            const DataEntry::FlexibleMetricMap *flexibleMetrics,
            const std::vector<Context> &contexts) {
          bool emittedKernel = false;
          if (auto kernelIt = metrics.find(MetricKind::Kernel);
              kernelIt != metrics.end()) {
            auto *kernelMetric =
                static_cast<KernelMetric *>(kernelIt->second.get());
            const auto isMetricKernel = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::IsMetricKernel));
            const auto streamId = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::StreamId));
            const auto launchScopeEventId =
                events.at(owningEventId).launchScopeEventId;
            if (isMetricKernel) {
              orderedTraceEvents.push_back(OrderedTraceEvent::kernel(
                  kernelMetric, flexibleMetrics, contexts, owningEventId,
                  streamId,
                  launchScopeEventId));
            } else {
              orderedTraceEvents.push_back(OrderedTraceEvent::kernel(
                  kernelMetric, nullptr, contexts, owningEventId, streamId,
                  launchScopeEventId));
            }
            hasKernelMetrics = true;
            emittedKernel = true;
          }
          if (auto cycleIt = metrics.find(MetricKind::Cycle);
              cycleIt != metrics.end()) {
            auto *cycleMetric =
                static_cast<CycleMetric *>(cycleIt->second.get());
            cycleEvents.emplace_back(cycleMetric, contexts);
            hasCycleMetrics = true;
          }
          return emittedKernel;
        };

    for (const auto &[_, event] : events) {
      const auto &baseContexts = eventIdToContexts.at(event.id);
      processMetricMaps(event.id, event.metricSet.metrics,
                        &event.metricSet.flexibleMetrics, baseContexts);
      if (event.scopeId != Scope::DummyScopeId &&
          !event.metricSet.flexibleMetrics.empty() && event.hasCpuTimeRange()) {
        orderedTraceEvents.push_back(OrderedTraceEvent::cpuMetricScope(
            &event.metricSet.flexibleMetrics, baseContexts, event.threadId,
            event.cpuStartTimeNs, event.cpuEndTimeNs));
        hasCpuMetricScopes = true;
      }
      std::vector<size_t> sortedLinkedTargetEntryIds;
      sortedLinkedTargetEntryIds.reserve(event.metricSet.linkedMetrics.size());
      for (const auto &[targetEntryId, _] : event.metricSet.linkedMetrics) {
        sortedLinkedTargetEntryIds.push_back(targetEntryId);
      }
      std::sort(sortedLinkedTargetEntryIds.begin(),
                sortedLinkedTargetEntryIds.end());
      for (const auto targetEntryId : sortedLinkedTargetEntryIds) {
        const auto &linkedMetrics =
            event.metricSet.linkedMetrics.at(targetEntryId);
        auto contexts = baseContexts;
        auto &virtualContexts = targetIdToVirtualContexts[targetEntryId];
        contexts.insert(contexts.end(), virtualContexts.begin(),
                        virtualContexts.end());
        const DataEntry::FlexibleMetricMap *flexibleMetrics = nullptr;
        auto iter = event.metricSet.linkedFlexibleMetrics.find(targetEntryId);
        if (iter != event.metricSet.linkedFlexibleMetrics.end()) {
          flexibleMetrics = &iter->second;
        }
        processMetricMaps(event.id, linkedMetrics, flexibleMetrics, contexts);
      }
      if (hasKernelMetrics && hasCycleMetrics) {
        throw std::runtime_error("only one active metric type is supported");
      }
    }

    if (hasCycleMetrics) {
      dumpCycleMetricTrace(cycleEvents, os);
      return;
    }

    if (hasKernelMetrics || hasCpuMetricScopes) {
      dumpKernelMetricTrace(orderedTraceEvents, events, eventIdToContexts, os);
    } else {
      os << json({{"displayTimeUnit", "us"}, {"traceEvents", json::array()}})
                .dump()
         << "\n";
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
