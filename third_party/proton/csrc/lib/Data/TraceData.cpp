#include "Data/TraceData.h"
#include "Context/Context.h"
#include "Profiler/Graph.h"
#include "TraceDataIO/TraceWriter.h"
#include "Utility/Errors.h"
#include "Utility/MsgPackWriter.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace proton {

namespace {
inline constexpr size_t kMaxActiveEventStackCacheObjects = 10;

thread_local std::unordered_map<const TraceData *, std::vector<size_t>>
    traceDataToActiveEventStack;

uint64_t getCurrentCpuTimestampNs() {
  using Clock = std::chrono::system_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

} // namespace

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
    TraceContext(size_t id, size_t parentId, const Context &context)
        : Context(context), parentId(parentId), id(id) {}
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

  struct Event {
    Event() = default;
    Event(size_t id, size_t contextId, size_t parentEventId)
        : id(id), contextId(contextId), parentEventId(parentEventId) {}
    size_t id = 0;
    size_t scopeId = Scope::DummyScopeId;
    size_t contextId = TraceContext::DummyId;
    // When the current event is added to the trace,
    // what is the parent event that is active (i.e. the last one entered but
    // not yet exited) in the trace
    size_t parentEventId = DummyId;
    uint64_t cpuStartTimeNs = 0;
    uint64_t cpuEndTimeNs = 0;
    size_t threadId = 0;
    // Direct and linked metrics emitted for this trace event.
    DataEntry::MetricSet metricSet{};

    bool hasCpuTimeRange() const {
      return cpuStartTimeNs != 0 && cpuEndTimeNs != 0 &&
             cpuEndTimeNs >= cpuStartTimeNs;
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
    traceContextMap.try_emplace(id, id, parentId, context);
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
    for (const auto &index : indices) {
      parentId = addContext(index, parentId);
    }
    return parentId;
  }

  std::vector<Context> getContexts(size_t contextId, bool skipRoot = false) {
    std::vector<const TraceContext *> reversedContexts;
    auto it = traceContextMap.find(contextId);
    if (it == traceContextMap.end()) {
      throw makeOutOfRange("Context not found");
    }
    auto *context = &it->second;
    reversedContexts.push_back(context);
    while (context->parentId != TraceContext::DummyId) {
      context = &traceContextMap.at(context->parentId);
      reversedContexts.push_back(context);
    }
    std::vector<Context> contexts;
    contexts.reserve(reversedContexts.size() - (skipRoot ? 1 : 0));
    for (auto iter = reversedContexts.rbegin(); iter != reversedContexts.rend();
         ++iter) {
      if (skipRoot && iter == reversedContexts.rbegin()) {
        continue;
      }
      contexts.push_back(**iter);
    }
    return contexts;
  }

  size_t addEvent(size_t contextId, size_t parentEventId = Event::DummyId) {
    traceEvents.try_emplace(nextEventId, nextEventId, contextId, parentEventId);
    return nextEventId++;
  }

  bool hasEvent(size_t eventId) {
    return traceEvents.find(eventId) != traceEvents.end();
  }

  Event &getEvent(size_t eventId) {
    auto it = traceEvents.find(eventId);
    if (it == traceEvents.end()) {
      throw makeOutOfRange("Event not found");
    }
    return it->second;
  }

  void removeEvent(size_t eventId) { traceEvents.erase(eventId); }

  const std::map<size_t, Event> &getEvents() const { return traceEvents; }

private:
  size_t nextTreeContextId = TraceContext::RootId + 1;
  size_t nextEventId = 0;
  std::map<size_t, Event> traceEvents;
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
    contexts.emplace_back(scope.name);
  auto &activeEventStack = traceDataToActiveEventStack[this];
  size_t parentEventId = activeEventStack.empty() ? Trace::Event::DummyId
                                                  : activeEventStack.back();
  auto eventId = currentTrace->addEvent(currentTrace->addContexts(contexts),
                                        parentEventId);
  auto &event = currentTrace->getEvent(eventId);
  event.scopeId = scope.scopeId;
  event.cpuStartTimeNs = getCurrentCpuTimestampNs();
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
    auto &activeEventStack = traceDataToActiveEventStack[this];
    activeEventStack.pop_back();
    if (activeEventStack.empty() &&
        traceDataToActiveEventStack.size() > kMaxActiveEventStackCacheObjects)
      traceDataToActiveEventStack.erase(this);
  }
  scopeIdToEventId.erase(scope.scopeId);
}

DataEntry TraceData::addOp(size_t phase, size_t eventId,
                           const std::vector<Context> &contexts) {
  auto lock = lockIfCurrentOrVirtualPhase(phase);
  auto *trace = phasePtrAs<Trace>(phase);
  auto parentContextId = 0;
  size_t threadId = 0;
  if (eventId == Data::kRootEntryId) {
    parentContextId = Trace::TraceContext::RootId;
    threadId = getCurrentThreadTraceId();
  } else {
    auto &event = trace->getEvent(eventId);
    parentContextId = event.contextId;
    // Inherit thread id from the parent event
    threadId = event.threadId;
  }
  const auto contextId = trace->addContexts(contexts, parentContextId);
  size_t parentEventId = Trace::Event::DummyId;
  if (eventId == Data::kRootEntryId) {
    auto activeEventStackIt = traceDataToActiveEventStack.find(this);
    if (activeEventStackIt != traceDataToActiveEventStack.end() &&
        !activeEventStackIt->second.empty()) {
      parentEventId = activeEventStackIt->second.back();
    }
  } else {
    parentEventId = eventId;
  }
  const auto newEventId = trace->addEvent(contextId, parentEventId);
  auto &newEvent = trace->getEvent(newEventId);
  // This is an instant or a GPU event, no CPU time range
  newEvent.threadId = threadId;
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

size_t TraceData::getCurrentThreadTraceId() {
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

constexpr uint64_t kTraceProcessId = 0;
constexpr uint64_t kCpuLaneBase = 0;
constexpr uint64_t kGraphLaneBase = 100000;
constexpr uint64_t kGpuLaneBase = 200000;

// Structure to pair CycleMetric with its context for processing
struct CycleEvent {
  std::vector<Context> contexts;
  const CycleMetric *cycleMetric;

  CycleEvent(const CycleMetric *metric, std::vector<Context> contexts)
      : contexts(std::move(contexts)), cycleMetric(metric) {}
};

struct KernelEvent {
  const KernelMetric *kernelMetric{};
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};
  std::vector<Context> contexts;
  size_t launchEventId{};
  bool isGraphLinked{};

  KernelEvent(const KernelMetric *metric,
              const DataEntry::FlexibleMetricMap *metrics,
              std::vector<Context> contexts, size_t launchId,
              bool isGraphLinked)
      : kernelMetric(metric), flexibleMetrics(metrics),
        contexts(std::move(contexts)), launchEventId(launchId),
        isGraphLinked(isGraphLinked) {}
};

struct CpuScopeEvent {
  size_t eventId;
  std::vector<Context> contexts;
  size_t threadId;
  uint64_t startTimeNs;
  uint64_t endTimeNs;
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};

  CpuScopeEvent(size_t eventId, const DataEntry::FlexibleMetricMap *metrics,
                std::vector<Context> contexts, size_t tid, uint64_t start,
                uint64_t end)
      : eventId(eventId), contexts(std::move(contexts)), threadId(tid),
        startTimeNs(start), endTimeNs(end), flexibleMetrics(metrics) {}
};

struct GraphScopeEvent {
  Context context;
  size_t streamId{};
  uint64_t startTimeNs{};
  uint64_t endTimeNs{};
  const DataEntry::FlexibleMetricMap *flexibleMetrics{};
};

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

  // Pre-sort all events once
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

void dumpCycleMetricTrace(std::vector<CycleEvent> &cycleEvents,
                          std::ostream &os) {
  auto timeline = convertToTimelineTrace(cycleEvents);
  auto writer = StreamChromeTraceWriter(timeline, "");
  writer.write(os);
}

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
    metrics[metricName] = formatFlexibleMetricValue(metricValue.getValues()[0]);
  }
  return metrics;
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

void appendThreadMetadata(json &traceEvents, uint64_t tid,
                          const std::string &name, uint64_t sortIndex) {
  json nameEvent;
  nameEvent["ph"] = "M";
  nameEvent["pid"] = kTraceProcessId;
  nameEvent["tid"] = tid;
  nameEvent["name"] = "thread_name";
  nameEvent["args"]["name"] = name;
  traceEvents.push_back(std::move(nameEvent));

  json sortEvent;
  sortEvent["ph"] = "M";
  sortEvent["pid"] = kTraceProcessId;
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
  processNameEvent["pid"] = kTraceProcessId;
  processNameEvent["tid"] = 0;
  processNameEvent["name"] = "process_name";
  processNameEvent["args"]["name"] = "Trace";
  traceEvents.push_back(std::move(processNameEvent));

  json processSortEvent;
  processSortEvent["ph"] = "M";
  processSortEvent["pid"] = kTraceProcessId;
  processSortEvent["tid"] = 0;
  processSortEvent["name"] = "process_sort_index";
  processSortEvent["args"]["sort_index"] = 0;
  traceEvents.push_back(std::move(processSortEvent));

  for (const auto &[threadId, _] : cpuScopeEvents) {
    const auto tid = getCpuLaneId(threadId);
    appendThreadMetadata(traceEvents, tid,
                         "CPU Thread " + std::to_string(threadId), tid);
  }

  for (const auto &[streamId, _] : graphScopeEvents) {
    const auto tid = getGraphLaneId(streamId);
    appendThreadMetadata(traceEvents, tid,
                         "Graph: Stream " + std::to_string(streamId), tid);
  }

  for (const auto &[streamId, _] : kernelEvents) {
    const auto tid = getGpuLaneId(streamId);
    appendThreadMetadata(traceEvents, tid,
                         "GPU Stream " + std::to_string(streamId), tid);
  }
}

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
      graphContexts.pop_back(); // Remove kernel name context
      auto startTimeNs = std::get<uint64_t>(
          kernelEvent.kernelMetric->getValue(KernelMetric::StartTime));
      auto endTimeNs = std::get<uint64_t>(
          kernelEvent.kernelMetric->getValue(KernelMetric::EndTime));
      // A streaming algorithm to find start and end time of graph scopes based
      // on common context prefix
      if (openScopes.empty()) {
        // There's no open graph scope, we start a new stack of scopes
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
          // Close scopes that are not in the common prefix
          auto &tailOpenScope = openScopes[i - 1];
          graphScopeEvents[streamId].push_back({tailOpenScope.context, streamId,
                                                tailOpenScope.startTimeNs,
                                                lastEndTimeNs});
        }
        for (size_t i = openScopes.size(); i > numCommonPrefixes; --i) {
          // Remove scopes that are not in the common prefix
          openScopes.pop_back();
        }
        for (size_t i = numCommonPrefixes; i < graphContexts.size(); ++i) {
          // Open scopes that are not in the common prefix
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

void dumpKernelEvents(uint64_t minTimeStamp,
                      std::map<size_t, std::vector<KernelEvent>> &kernelEvents,
                      json &object, std::ostream &os) {
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
      // Convert nanoseconds to microseconds for Chrome trace format
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
      element["pid"] = kTraceProcessId;
      element["ts"] = ts;
      element["dur"] = dur;
      element["tid"] = getGpuLaneId(streamId);
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
    std::map<size_t, std::vector<CpuScopeEvent>> &cpuScopeEvents, json &object,
    std::ostream &os) {
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
            buildFlexibleMetricEventName(event.contexts, *flexibleMetrics);
        element["cat"] = "metric";
        element["args"]["metrics"] = buildFlexibleMetricsJson(*flexibleMetrics);
      } else {
        element["name"] =
            event.contexts.empty() ? "" : event.contexts.back().name;
        element["cat"] = "scope";
      }
      element["ph"] = "X";
      element["pid"] = kTraceProcessId;
      element["ts"] = ts;
      element["dur"] = dur;
      element["tid"] = getCpuLaneId(threadId);
      element["args"]["call_stack"] = buildCallStackJson(event.contexts);
      object["traceEvents"].push_back(std::move(element));
    }
  }
}

void dumpGraphScopeEvents(
    uint64_t minTimeStamp,
    std::map<size_t, std::vector<GraphScopeEvent>> &graphScopeEvents,
    json &object, std::ostream &os) {
  for (auto &[streamId, events] : graphScopeEvents) {
    const auto graphTid = getGraphLaneId(streamId);
    for (const auto &event : events) {
      json element;
      if (event.flexibleMetrics != nullptr && !event.flexibleMetrics->empty()) {
        element["name"] =
            buildFlexibleMetricEventName(event.context, *event.flexibleMetrics);
        element["cat"] = "metric";
        element["args"]["metrics"] =
            buildFlexibleMetricsJson(*event.flexibleMetrics);
      } else {
        element["name"] = event.context.name;
        element["cat"] = "scope";
      }
      element["ph"] = "X";
      element["pid"] = kTraceProcessId;
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
    auto prevLaunchEventId = TraceData::Trace::Event::DummyId;
    for (const auto &event : events) {
      if (event.launchEventId == TraceData::Trace::Event::DummyId) {
        // This kernel event is not linked to any CPU scope event, skip it.
        continue;
      }
      if (prevLaunchEventId == event.launchEventId && event.isGraphLinked) {
        // For back-to-back kernel launches linked to the same CPU scope event,
        // we only create flow events for the first one to avoid creating
        // duplicated flow events that overlap with each other and cause visual
        // clutter
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
      startElement["pid"] = kTraceProcessId;
      startElement["tid"] = getCpuLaneId(launchEvent->threadId);
      startElement["ts"] =
          static_cast<double>(launchEvent->startTimeNs - minTimeStamp) / 1000.0;
      startElement["id"] = event.launchEventId;
      startElement["bp"] = "e";
      object["traceEvents"].push_back(std::move(startElement));

      json finishElement;
      finishElement["name"] = "launch->kernel";
      finishElement["cat"] = "flow";
      finishElement["ph"] = "f";
      finishElement["pid"] = kTraceProcessId;
      finishElement["tid"] = getGpuLaneId(streamId);
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
    std::map<size_t, std::vector<KernelEvent>> &kernelEvents,
    std::map<size_t, std::vector<CpuScopeEvent>> &cpuScopeEvents,
    std::map<size_t, std::vector<GraphScopeEvent>> &graphScopeEvents,
    std::ostream &os) {
  json object = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};

  emitTraceLaneMetadata(object, cpuScopeEvents, graphScopeEvents, kernelEvents);
  dumpCpuScopeEvents(minTimeStamp, cpuScopeEvents, object, os);
  dumpGraphScopeEvents(minTimeStamp, graphScopeEvents, object, os);
  dumpCpuToGpuFlowEvents(minTimeStamp, cpuScopeEvents, kernelEvents, object);
  dumpKernelEvents(minTimeStamp, kernelEvents, object, os);

  os << object.dump() << "\n";
}

void dumpCpuOnlyTrace(
    uint64_t minTimeStamp,
    std::map<size_t, std::vector<CpuScopeEvent>> &cpuScopeEvents,
    std::ostream &os) {
  json object = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};
  dumpCpuScopeEvents(minTimeStamp, cpuScopeEvents, object, os);
  os << object.dump() << "\n";
}

} // namespace

void TraceData::dumpChromeTrace(std::ostream &os, size_t phase) const {
  std::set<size_t> targetEntryIds;
  // First, check whether any target entries are linked.
  // If so, resolve their contexts with a single lock on the virtual phase.
  // The virtual phase is shared by all active phases, so locking it is
  // expensive and we want to keep the lock duration as short as possible.
  tracePhases.withPtr(phase, [&](Trace *trace) {
    for (const auto &[_, event] : trace->getEvents()) {
      for (const auto &[targetEntryId, _] : event.metricSet.linkedMetrics) {
        targetEntryIds.insert(targetEntryId);
      }
      for (const auto &[targetEntryId, _] :
           event.metricSet.linkedFlexibleMetrics) {
        targetEntryIds.insert(targetEntryId);
      }
    }
  });

  std::map<size_t, std::vector<Context>> targetIdToVirtualContexts;
  if (!targetEntryIds.empty()) {
    tracePhases.withPtr(Data::kVirtualPhase, [&](Trace *virtualTrace) {
      for (auto targetEntryId : targetEntryIds) {
        // Linked target ids are event ids, so resolve through the event first.
        auto &targetEvent = virtualTrace->getEvent(targetEntryId);
        targetIdToVirtualContexts.emplace(
            targetEntryId, virtualTrace->getContexts(targetEvent.contextId,
                                                     /*skipRoot=*/true));
      }
    });
  }

  // After virtual contexts are resolved, we can proceed to process events in
  // the actual phase without worrying about locking duration.
  tracePhases.withPtr(phase, [&](Trace *trace) {
    auto &events = trace->getEvents();
    std::map</*stream_id=*/size_t, std::vector<KernelEvent>> kernelEvents;
    std::map</*thread_id=*/size_t, std::vector<CpuScopeEvent>> cpuScopeEvents;
    std::map</*stream_id=*/size_t, std::vector<GraphScopeEvent>>
        graphScopeEvents;
    std::vector<CycleEvent> cycleEvents;
    cycleEvents.reserve(events.size());
    // Initialize minTimeStamp to the maximum possible value to ensure correct
    // calculation of relative timestamps
    uint64_t minTimeStamp = std::numeric_limits<uint64_t>::max();
    std::unordered_map<size_t, std::vector<Context>> contextIdToContexts(
        events.size());
    for (const auto &[_, event] : events) {
      contextIdToContexts.try_emplace(event.contextId,
                                      trace->getContexts(event.contextId));
    }
    bool hasKernelMetrics = false, hasCycleMetrics = false;

    auto processMetricMaps =
        [&](size_t eventId, const DataEntry::MetricMap &metrics,
            const DataEntry::FlexibleMetricMap *flexibleMetrics,
            const std::vector<Context> &contexts, bool isGraphLinked) {
          if (auto kernelIt = metrics.find(MetricKind::Kernel);
              kernelIt != metrics.end()) {
            auto *kernelMetric =
                static_cast<KernelMetric *>(kernelIt->second.get());
            auto streamId = static_cast<size_t>(std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::StreamId)));
            auto startTimeNs = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::StartTime));
            auto launchEventId = events.at(eventId).parentEventId;
            kernelEvents[streamId].emplace_back(kernelMetric, flexibleMetrics,
                                                contexts, launchEventId,
                                                isGraphLinked);
            minTimeStamp = std::min(minTimeStamp, startTimeNs);
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
      if (event.hasCpuTimeRange()) { // CPU scope event
        cpuScopeEvents[event.threadId].emplace_back(
            event.id,
            event.metricSet.flexibleMetrics.empty()
                ? nullptr
                : &event.metricSet.flexibleMetrics,
            contextIdToContexts.at(event.contextId), event.threadId,
            event.cpuStartTimeNs, event.cpuEndTimeNs);
        minTimeStamp = std::min(minTimeStamp, event.cpuStartTimeNs);
      } else { // Kernel or cycle event
        const auto &baseContexts = contextIdToContexts.at(event.contextId);
        processMetricMaps(event.id, event.metricSet.metrics,
                          &event.metricSet.flexibleMetrics, baseContexts,
                          /*isGraphLinked=*/false);
        for (const auto &[targetEntryId, _] : event.metricSet.linkedMetrics) {
          const auto &linkedMetrics =
              event.metricSet.linkedMetrics.at(targetEntryId);
          // Combine the base contexts and virtual contexts to
          // form the complete contexts for the linked metrics.
          auto contexts = baseContexts;
          auto &virtualContexts = targetIdToVirtualContexts[targetEntryId];
          contexts.reserve(contexts.size() + virtualContexts.size());
          for (const auto &context : virtualContexts) {
            contexts.push_back(context);
          }
          // Not all kernels are associated with a flexible metric
          const DataEntry::FlexibleMetricMap *flexibleMetrics = nullptr;
          auto iter = event.metricSet.linkedFlexibleMetrics.find(targetEntryId);
          if (iter != event.metricSet.linkedFlexibleMetrics.end()) {
            flexibleMetrics = &iter->second;
          }
          processMetricMaps(event.id, linkedMetrics, flexibleMetrics, contexts,
                            /*isGraphLinked=*/true);
        }
        if (hasKernelMetrics && hasCycleMetrics) {
          throw makeLogicError("only one active metric type is supported");
        }
      }
    }

    if (hasCycleMetrics) {
      dumpCycleMetricTrace(cycleEvents, os);
      return;
    }

    // Keep CPU ranges stable regardless of whether kernels were recorded.
    for (auto &[threadId, events] : cpuScopeEvents) {
      std::sort(events.begin(), events.end(),
                [](const CpuScopeEvent &a, const CpuScopeEvent &b) {
                  return a.startTimeNs < b.startTimeNs;
                });
    }

    if (hasKernelMetrics) {
      // Sort all kernel events in order
      for (auto &[streamId, events] : kernelEvents) {
        std::sort(events.begin(), events.end(),
                  [](const KernelEvent &a, const KernelEvent &b) {
                    auto aStartTime = std::get<uint64_t>(
                        a.kernelMetric->getValue(KernelMetric::StartTime));
                    auto bStartTime = std::get<uint64_t>(
                        b.kernelMetric->getValue(KernelMetric::StartTime));
                    return aStartTime < bStartTime;
                  });
      }
      // Graph scopes are constructed in order
      reconstructGraphScopeEvents(kernelEvents, graphScopeEvents);
      dumpKernelMetricTrace(minTimeStamp, kernelEvents, cpuScopeEvents,
                            graphScopeEvents, os);
    } else if (!cpuScopeEvents.empty()) {
      dumpCpuOnlyTrace(minTimeStamp, cpuScopeEvents, os);
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
    throw makeInvalidArgument("Output format not supported");
  }
}

TraceData::TraceData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  initPhaseStore(tracePhases);
}

TraceData::~TraceData() {}

} // namespace proton
