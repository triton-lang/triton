#include "Data/TraceData.h"
#include "Profiler/Graph.h"
#include "TraceDataIO/TraceWriter.h"
#include "Utility/MsgPackWriter.h"
#include "Utility/ProtoWriter.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

using json = nlohmann::json;

namespace proton {

namespace {
inline constexpr size_t kMaxActiveEventStackCacheObjects = 10;
}

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
      throw std::runtime_error("Event not found");
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

thread_local std::unordered_map<const TraceData *, std::vector<size_t>>
    traceDataToActiveEventStack;

uint64_t getCurrentCpuTimestampNs() {
  using Clock = std::chrono::system_clock;
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

std::vector<uint8_t> TraceData::toPerfettoTrace(size_t phase) const {
  std::ostringstream os;
  dumpPerfettoTrace(os, phase);
  const auto bytes = os.str();
  return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

namespace {

constexpr uint64_t kTraceProcessId = 0;
constexpr uint64_t kCpuLaneBase = 0;
constexpr uint64_t kGraphLaneBase = 100000;
constexpr uint64_t kGpuLaneBase = 200000;
constexpr uint64_t kPerfettoProcessTrackUuid = 1;
constexpr uint64_t kPerfettoLaneTrackUuidBase = 1000;
constexpr uint64_t kPerfettoCycleTrackUuidBase = 1000000;
constexpr uint64_t kPerfettoFlowIdBase = 1ULL << 32;
constexpr uint32_t kPerfettoTracePacketSequenceId = 1;
constexpr uint32_t kPerfettoSeqIncrementalStateCleared = 1;
constexpr uint32_t kPerfettoSeqNeedsIncrementalState = 2;
constexpr int32_t kPerfettoCpuTrackOrderBase = 0;
constexpr int32_t kPerfettoGraphTrackOrderBase = 100000;
constexpr int32_t kPerfettoGpuTrackOrderBase = 200000;
constexpr int32_t kPerfettoCycleTrackOrderBase = 300000;
constexpr uint32_t kPerfettoChildTracksOrderingExplicit = 3;

struct PerfettoAnnotation {
  enum class Kind { String, Json, UInt64, Int64, Double, Bool };

  std::string name;
  std::string stringValue;
  uint64_t uintValue{};
  int64_t intValue{};
  double doubleValue{};
  bool boolValue{};
  Kind kind{Kind::String};
};

struct PerfettoSliceEvent {
  uint64_t trackUuid{};
  uint64_t startTimeNs{};
  uint64_t endTimeNs{};
  std::string name;
  std::string category;
  std::vector<PerfettoAnnotation> annotations;
  std::vector<uint64_t> flowIds;
  std::vector<uint64_t> terminatingFlowIds;
};

struct PerfettoSlicePoint {
  uint64_t timestampNs{};
  bool isBegin{};
  const PerfettoSliceEvent *event{};
};

struct PerfettoTrack {
  std::string name;
  int32_t siblingOrderRank{};
};

class PerfettoInternedStringTable {
public:
  uint64_t intern(const std::string &name) {
    if (auto it = nameToIid.find(name); it != nameToIid.end()) {
      return it->second;
    }

    const auto iid = nextIid++;
    nameToIid.emplace(name, iid);
    iidToName.emplace(iid, name);
    return iid;
  }

  uint64_t get(const std::string &name) const {
    auto it = nameToIid.find(name);
    if (it == nameToIid.end()) {
      throw std::logic_error("Perfetto name was not interned: " + name);
    }
    return it->second;
  }

  bool empty() const { return iidToName.empty(); }

  const std::map<uint64_t, std::string> &entries() const { return iidToName; }

private:
  uint64_t nextIid = 1;
  std::unordered_map<std::string, uint64_t> nameToIid;
  std::map<uint64_t, std::string> iidToName;
};

struct PerfettoInternedNames {
  PerfettoInternedStringTable eventCategories;
  PerfettoInternedStringTable eventNames;
  PerfettoInternedStringTable debugAnnotationNames;

  bool empty() const {
    return eventCategories.empty() && eventNames.empty() &&
           debugAnnotationNames.empty();
  }

  void intern(const PerfettoSliceEvent &event) {
    eventNames.intern(event.name);
    if (!event.category.empty()) {
      eventCategories.intern(event.category);
    }
    for (const auto &annotation : event.annotations) {
      debugAnnotationNames.intern(annotation.name);
    }
  }
};

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

uint64_t getKernelEventStartTimeNs(const KernelEvent &event) {
  return std::get<uint64_t>(
      event.kernelMetric->getValue(KernelMetric::StartTime));
}

uint64_t getKernelEventEndTimeNs(const KernelEvent &event) {
  return std::get<uint64_t>(
      event.kernelMetric->getValue(KernelMetric::EndTime));
}

uint64_t getKernelEventDeviceId(const KernelEvent &event) {
  return std::get<uint64_t>(
      event.kernelMetric->getValue(KernelMetric::DeviceId));
}

uint64_t getKernelEventStreamId(const KernelEvent &event) {
  return std::get<uint64_t>(
      event.kernelMetric->getValue(KernelMetric::StreamId));
}

uint64_t getKernelEventIsMetricKernel(const KernelEvent &event) {
  return std::get<uint64_t>(
      event.kernelMetric->getValue(KernelMetric::IsMetricKernel));
}

std::string getKernelEventName(const KernelEvent &event) {
  if (getKernelEventIsMetricKernel(event)) {
    return GraphState::metricTag;
  }
  return event.contexts.empty() ? "" : event.contexts.back().name;
}

bool compareKernelEvents(const KernelEvent &a, const KernelEvent &b) {
  if (getKernelEventStartTimeNs(a) != getKernelEventStartTimeNs(b)) {
    return getKernelEventStartTimeNs(a) < getKernelEventStartTimeNs(b);
  }
  if (getKernelEventEndTimeNs(a) != getKernelEventEndTimeNs(b)) {
    return getKernelEventEndTimeNs(a) < getKernelEventEndTimeNs(b);
  }
  if (getKernelEventIsMetricKernel(a) != getKernelEventIsMetricKernel(b)) {
    return getKernelEventIsMetricKernel(a) < getKernelEventIsMetricKernel(b);
  }
  return getKernelEventName(a) < getKernelEventName(b);
}

bool isSameKernelEvent(const KernelEvent &a, const KernelEvent &b) {
  return getKernelEventStartTimeNs(a) == getKernelEventStartTimeNs(b) &&
         getKernelEventEndTimeNs(a) == getKernelEventEndTimeNs(b) &&
         getKernelEventDeviceId(a) == getKernelEventDeviceId(b) &&
         getKernelEventStreamId(a) == getKernelEventStreamId(b) &&
         getKernelEventIsMetricKernel(a) == getKernelEventIsMetricKernel(b) &&
         getKernelEventName(a) == getKernelEventName(b);
}

void pruneDuplicateKernelEvents(std::vector<KernelEvent> &events) {
  std::vector<KernelEvent> pruned;
  pruned.reserve(events.size());
  for (auto &event : events) {
    if (!pruned.empty() && isSameKernelEvent(pruned.back(), event)) {
      continue;
    }
    pruned.push_back(std::move(event));
  }
  events = std::move(pruned);
}

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

PerfettoAnnotation makeJsonAnnotation(const std::string &name,
                                      const std::string &value);
PerfettoAnnotation makeUInt64Annotation(const std::string &name,
                                        uint64_t value);
PerfettoAnnotation makeDoubleAnnotation(const std::string &name,
                                        double value);
void appendPerfettoTrace(std::ostream &os,
                         const std::map<uint64_t, PerfettoTrack> &tracks,
                         const std::vector<PerfettoSliceEvent> &events);

uint64_t getMinInitTime(const std::vector<KernelTrace> &streamTrace) {
  uint64_t minInitTime = std::numeric_limits<uint64_t>::max();
  for (const auto &kernelTrace : streamTrace) {
    for (const auto &blockTrace : kernelTrace.first->blockTraces) {
      minInitTime = std::min(minInitTime, blockTrace.initTime);
    }
  }
  return minInitTime;
}

using BlockTraceVec =
    std::vector<const CircularLayoutParserResult::BlockTrace *>;

void populateTraceInfo(std::shared_ptr<CircularLayoutParserResult> result,
                       std::map<int, uint64_t> &blockToMinCycle,
                       std::map<int, BlockTraceVec> &procToBlockTraces) {
  for (auto &blockTrace : result->blockTraces) {
    uint64_t minCycle = std::numeric_limits<uint64_t>::max();
    for (auto &trace : blockTrace.traces) {
      for (auto &event : trace.profileEvents) {
        minCycle = std::min(minCycle, event.first->cycle);
      }
    }
    blockToMinCycle[blockTrace.blockId] = minCycle;
    procToBlockTraces[blockTrace.procId].push_back(&blockTrace);
  }
}

std::vector<int> assignLineIds(
    const std::vector<CircularLayoutParserResult::ProfileEvent> &trace) {
  std::vector<int> result(trace.size());
  std::vector<std::pair<size_t, CircularLayoutParserResult::ProfileEvent>>
      indexedEvents;
  indexedEvents.reserve(trace.size());

  for (size_t i = 0; i < trace.size(); ++i) {
    indexedEvents.push_back({i, trace[i]});
  }
  std::sort(indexedEvents.begin(), indexedEvents.end(),
            [](const auto &a, const auto &b) {
              return a.second.first->cycle < b.second.first->cycle;
            });

  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> lines;
  for (const auto &[originalIdx, event] : indexedEvents) {
    const auto startTime = event.first->cycle;
    const auto endTime = event.second->cycle;
    size_t lineIdx = 0;
    for (; lineIdx < lines.size(); ++lineIdx) {
      bool canPlace = true;
      for (const auto &[intervalStart, intervalEnd] : lines[lineIdx]) {
        if (startTime < intervalEnd && endTime > intervalStart) {
          canPlace = false;
          break;
        }
      }
      if (canPlace) {
        break;
      }
    }
    if (lineIdx == lines.size()) {
      lines.push_back({});
    }
    lines[lineIdx].push_back({startTime, endTime});
    result[originalIdx] = static_cast<int>(lineIdx);
  }
  return result;
}

void dumpCycleMetricPerfettoTrace(std::vector<CycleEvent> &cycleEvents,
                                  std::ostream &os) {
  auto timeline = convertToTimelineTrace(cycleEvents);
  std::map<uint64_t, PerfettoTrack> tracks;
  std::vector<PerfettoSliceEvent> events;
  if (timeline.empty()) {
    appendPerfettoTrace(os, tracks, events);
    return;
  }

  const auto minInitTime = getMinInitTime(timeline);
  uint64_t nextTrackUuid = kPerfettoCycleTrackUuidBase;
  std::map<std::tuple<std::string, int, int, int, int>, uint64_t> trackUuids;

  for (const auto &kernelTrace : timeline) {
    auto result = kernelTrace.first;
    auto metadata = kernelTrace.second;
    auto callStack = json::array();
    for (const auto &frame : metadata->callStack) {
      callStack.push_back(frame);
    }

    std::map<int, uint64_t> blockToMinCycle;
    std::map<int, BlockTraceVec> procToBlockTraces;
    populateTraceInfo(result, blockToMinCycle, procToBlockTraces);

    for (const auto &[procId, blockVec] : procToBlockTraces) {
      for (auto *blockTrace : blockVec) {
        const int ctaId = blockTrace->blockId;
        for (const auto &trace : blockTrace->traces) {
          const int warpId = trace.uid;
          auto lineInfo = assignLineIds(trace.profileEvents);
          for (size_t eventIdx = 0; eventIdx < trace.profileEvents.size();
               ++eventIdx) {
            const auto &event = trace.profileEvents[eventIdx];
            const int lineId = lineInfo[eventIdx];
            const auto trackKey = std::make_tuple(
                metadata->kernelName, procId, ctaId, warpId, lineId);
            auto [trackIt, inserted] =
                trackUuids.try_emplace(trackKey, nextTrackUuid);
            if (inserted) {
              std::ostringstream name;
              name << metadata->kernelName << " Core" << procId << " CTA"
                   << ctaId << " / warp " << warpId << " (line " << lineId
                   << ")";
              tracks.emplace(
                  nextTrackUuid,
                  PerfettoTrack{name.str(),
                                kPerfettoCycleTrackOrderBase +
                                    static_cast<int32_t>(trackUuids.size())});
              ++nextTrackUuid;
            }

            const int scopeId = event.first->scopeId;
            const auto nameIt = metadata->scopeName.find(scopeId);
            const auto name = nameIt == metadata->scopeName.end()
                                  ? "scope_" + std::to_string(scopeId)
                                  : nameIt->second;
            const auto cycleAdjust =
                static_cast<int64_t>(blockTrace->initTime - minInitTime) -
                static_cast<int64_t>(blockToMinCycle[ctaId]);
            const auto adjustedStart =
                static_cast<int64_t>(event.first->cycle) + cycleAdjust;
            const auto adjustedEnd =
                static_cast<int64_t>(event.second->cycle) + cycleAdjust;
            const auto startTimeNs =
                static_cast<uint64_t>(std::max<int64_t>(0, adjustedStart));
            const auto endTimeNs = static_cast<uint64_t>(
                std::max<int64_t>(static_cast<int64_t>(startTimeNs),
                                  adjustedEnd));

            PerfettoSliceEvent slice;
            slice.trackUuid = trackIt->second;
            slice.startTimeNs = startTimeNs;
            slice.endTimeNs = endTimeNs;
            slice.name = name;
            slice.category = metadata->kernelName;
            slice.annotations.push_back(
                makeUInt64Annotation("Init Time (ns)", blockTrace->initTime));
            slice.annotations.push_back(makeUInt64Annotation(
                "Post Final Time (ns)", blockTrace->postFinalTime));
            slice.annotations.push_back(makeUInt64Annotation(
                "Finalization Time (ns)",
                blockTrace->postFinalTime - blockTrace->preFinalTime));
            slice.annotations.push_back(
                makeDoubleAnnotation("Frequency (MHz)", 1000.0));
            slice.annotations.push_back(
                makeJsonAnnotation("call_stack", callStack.dump()));
            events.push_back(std::move(slice));
          }
        }
      }
    }
  }
  appendPerfettoTrace(os, tracks, events);
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

json buildCallStackJson(const Context &context) {
  json callStack = json::array();
  callStack.push_back(context.name);
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

uint64_t getGpuLaneId(size_t deviceId, size_t streamId) {
  return kGpuLaneBase + deviceId * 1000000 + streamId;
}

uint64_t getPerfettoLaneTrackUuid(uint64_t laneId) {
  return kPerfettoLaneTrackUuidBase + laneId;
}

void appendTracePacket(ProtoWriter &trace, const ProtoWriter &packet) {
  // Trace.packet = 1.
  trace.writeMessage(1, packet);
}

void appendTrackDescriptorPacket(ProtoWriter &trace, const ProtoWriter &track) {
  ProtoWriter packet;
  // TracePacket.track_descriptor = 60.
  packet.writeMessage(60, track);
  appendTracePacket(trace, packet);
}

void setTracePacketSequence(ProtoWriter &packet, uint32_t sequenceFlags) {
  // TracePacket.trusted_packet_sequence_id = 10, sequence_flags = 13.
  packet.writeUInt32(10, kPerfettoTracePacketSequenceId);
  packet.writeUInt32(13, sequenceFlags);
}

void appendProcessTrackDescriptor(ProtoWriter &trace) {
  ProtoWriter process;
  // ProcessDescriptor.pid = 1, process_name = 6.
  process.writeInt32(1, static_cast<int32_t>(kTraceProcessId));
  process.writeString(6, "Trace");

  ProtoWriter track;
  // TrackDescriptor.uuid = 1, name = 2, process = 3.
  track.writeUInt64(1, kPerfettoProcessTrackUuid);
  track.writeString(2, "Trace");
  track.writeMessage(3, process);
  // TrackDescriptor.child_ordering = 11.
  track.writeUInt32(11, kPerfettoChildTracksOrderingExplicit);
  appendTrackDescriptorPacket(trace, track);
}

void appendLaneTrackDescriptor(ProtoWriter &trace, uint64_t trackUuid,
                               const PerfettoTrack &trackInfo) {
  ProtoWriter track;
  // TrackDescriptor.uuid = 1, name = 2, parent_uuid = 5,
  // sibling_order_rank = 12.
  track.writeUInt64(1, trackUuid);
  track.writeString(2, trackInfo.name);
  track.writeUInt64(5, kPerfettoProcessTrackUuid);
  track.writeInt32(12, trackInfo.siblingOrderRank);
  appendTrackDescriptorPacket(trace, track);
}

void appendDebugAnnotation(ProtoWriter &event,
                           const PerfettoAnnotation &annotation,
                           const PerfettoInternedNames &internedNames) {
  ProtoWriter message;
  // DebugAnnotation.name_iid = 1.
  message.writeUInt64(1,
                      internedNames.debugAnnotationNames.get(annotation.name));
  switch (annotation.kind) {
  case PerfettoAnnotation::Kind::String:
    // DebugAnnotation.string_value = 6.
    message.writeString(6, annotation.stringValue);
    break;
  case PerfettoAnnotation::Kind::Json:
    // DebugAnnotation.legacy_json_value = 9.
    message.writeString(9, annotation.stringValue);
    break;
  case PerfettoAnnotation::Kind::UInt64:
    // DebugAnnotation.uint_value = 3.
    message.writeUInt64(3, annotation.uintValue);
    break;
  case PerfettoAnnotation::Kind::Int64:
    // DebugAnnotation.int_value = 4.
    message.writeInt64(4, annotation.intValue);
    break;
  case PerfettoAnnotation::Kind::Double:
    // DebugAnnotation.double_value = 5.
    message.writeDouble(5, annotation.doubleValue);
    break;
  case PerfettoAnnotation::Kind::Bool:
    // DebugAnnotation.bool_value = 2.
    message.writeBool(2, annotation.boolValue);
    break;
  }
  // TrackEvent.debug_annotations = 4.
  event.writeMessage(4, message);
}

PerfettoAnnotation makeJsonAnnotation(const std::string &name,
                                      const std::string &value) {
  PerfettoAnnotation annotation;
  annotation.name = name;
  annotation.stringValue = value;
  annotation.kind = PerfettoAnnotation::Kind::Json;
  return annotation;
}

PerfettoAnnotation makeUInt64Annotation(const std::string &name,
                                        uint64_t value) {
  PerfettoAnnotation annotation;
  annotation.name = name;
  annotation.uintValue = value;
  annotation.kind = PerfettoAnnotation::Kind::UInt64;
  return annotation;
}

PerfettoAnnotation makeDoubleAnnotation(const std::string &name,
                                        double value) {
  PerfettoAnnotation annotation;
  annotation.name = name;
  annotation.doubleValue = value;
  annotation.kind = PerfettoAnnotation::Kind::Double;
  return annotation;
}

void appendTrackEventPacket(ProtoWriter &trace, uint64_t timestampNs,
                            uint32_t type, const PerfettoSliceEvent &event,
                            const PerfettoInternedNames &internedNames) {
  ProtoWriter trackEvent;
  // TrackEvent.type = 9, track_uuid = 11.
  trackEvent.writeUInt32(9, type);
  trackEvent.writeUInt64(11, event.trackUuid);
  if (type == 1) {
    // TrackEvent.category_iids = 3, name_iid = 10.
    if (!event.category.empty()) {
      trackEvent.writeUInt64(3,
                             internedNames.eventCategories.get(event.category));
    }
    trackEvent.writeUInt64(10, internedNames.eventNames.get(event.name));
    for (const auto &annotation : event.annotations) {
      appendDebugAnnotation(trackEvent, annotation, internedNames);
    }
    for (auto flowId : event.flowIds) {
      // TrackEvent.flow_ids = 36.
      trackEvent.writeUInt64(36, flowId);
    }
    for (auto flowId : event.terminatingFlowIds) {
      // TrackEvent.terminating_flow_ids = 42.
      trackEvent.writeUInt64(42, flowId);
    }
  }

  ProtoWriter packet;
  // TracePacket.timestamp = 8, track_event = 11.
  packet.writeUInt64(8, timestampNs);
  packet.writeMessage(11, trackEvent);
  setTracePacketSequence(packet, kPerfettoSeqNeedsIncrementalState);
  appendTracePacket(trace, packet);
}

void appendInternedStringEntries(
    ProtoWriter &internedData, uint32_t fieldId,
    const PerfettoInternedStringTable &internedStrings) {
  for (const auto &[iid, name] : internedStrings.entries()) {
    ProtoWriter entry;
    // Interned string messages use iid = 1, name = 2.
    entry.writeUInt64(1, iid);
    entry.writeString(2, name);
    internedData.writeMessage(fieldId, entry);
  }
}

void appendInternedDataPacket(ProtoWriter &trace,
                              const PerfettoInternedNames &internedNames) {
  if (internedNames.empty()) {
    return;
  }

  ProtoWriter internedData;
  // InternedData.event_categories = 1, event_names = 2,
  // debug_annotation_names = 3.
  appendInternedStringEntries(internedData, 1, internedNames.eventCategories);
  appendInternedStringEntries(internedData, 2, internedNames.eventNames);
  appendInternedStringEntries(internedData, 3,
                              internedNames.debugAnnotationNames);

  ProtoWriter packet;
  // TracePacket.interned_data = 12.
  packet.writeMessage(12, internedData);
  setTracePacketSequence(packet, kPerfettoSeqIncrementalStateCleared |
                                     kPerfettoSeqNeedsIncrementalState);
  appendTracePacket(trace, packet);
}

PerfettoInternedNames
collectPerfettoInternedNames(const std::vector<PerfettoSliceEvent> &events) {
  PerfettoInternedNames internedNames;
  for (const auto &event : events) {
    internedNames.intern(event);
  }
  return internedNames;
}

void appendPerfettoTrace(std::ostream &os,
                         const std::map<uint64_t, PerfettoTrack> &tracks,
                         const std::vector<PerfettoSliceEvent> &events) {
  ProtoWriter trace;
  auto internedNames = collectPerfettoInternedNames(events);
  appendProcessTrackDescriptor(trace);
  for (const auto &[uuid, track] : tracks) {
    appendLaneTrackDescriptor(trace, uuid, track);
  }
  appendInternedDataPacket(trace, internedNames);

  std::vector<PerfettoSlicePoint> points;
  points.reserve(events.size() * 2);
  for (const auto &event : events) {
    points.push_back({event.startTimeNs, true, &event});
    points.push_back({event.endTimeNs, false, &event});
  }
  std::sort(points.begin(), points.end(),
            [](const PerfettoSlicePoint &a, const PerfettoSlicePoint &b) {
              if (a.timestampNs != b.timestampNs) {
                return a.timestampNs < b.timestampNs;
              }
              if (a.isBegin != b.isBegin) {
                return !a.isBegin;
              }
              if (a.isBegin) {
                return a.event->endTimeNs > b.event->endTimeNs;
              }
              return a.event->startTimeNs > b.event->startTimeNs;
            });

  for (const auto &point : points) {
    // TrackEvent.Type: TYPE_SLICE_BEGIN = 1, TYPE_SLICE_END = 2.
    appendTrackEventPacket(trace, point.timestampNs, point.isBegin ? 1 : 2,
                           *point.event, internedNames);
  }

  const auto &data = trace.data();
  os.write(data.data(), static_cast<std::streamsize>(data.size()));
}

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
      bool seenCaptureTag = false;
      bool isMetadataKernel = false;
      for (const auto &context : kernelEvent.contexts) {
        if (context.name == GraphState::metricTag ||
            context.name == GraphState::metadataTag) {
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
      if (!seenCaptureTag) {
        throw std::runtime_error("Invalid graph contexts without capture tag");
      }
      if (!isMetadataKernel) {
        graphContexts
            .pop_back(); // Remove kernel name context for non-metadata kernels
      }
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
        auto contexts = virtualTrace->getContexts(targetEvent.contextId);
        contexts.erase(contexts.begin());
        targetIdToVirtualContexts.emplace(targetEntryId, std::move(contexts));
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
            if (isGraphLinked) {
              // For graph-linked kernels, the parent maybe the <captured_at>
              // tag. So we need to go one level up to find the actual launch
              // event
              launchEventId = events.at(launchEventId).parentEventId;
            }
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
          contexts.insert(contexts.end(), virtualContexts.begin(),
                          virtualContexts.end());
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
          throw std::runtime_error("only one active metric type is supported");
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
        std::sort(events.begin(), events.end(), compareKernelEvents);
        pruneDuplicateKernelEvents(events);
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

void TraceData::dumpPerfettoTrace(std::ostream &os, size_t phase) const {
  std::set<size_t> targetEntryIds;
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
        auto &targetEvent = virtualTrace->getEvent(targetEntryId);
        auto contexts = virtualTrace->getContexts(targetEvent.contextId);
        contexts.erase(contexts.begin());
        targetIdToVirtualContexts.emplace(targetEntryId, std::move(contexts));
      }
    });
  }

  tracePhases.withPtr(phase, [&](Trace *trace) {
    auto &events = trace->getEvents();
    std::map</*stream_id=*/size_t, std::vector<KernelEvent>> kernelEvents;
    std::map</*thread_id=*/size_t, std::vector<CpuScopeEvent>> cpuScopeEvents;
    std::map</*stream_id=*/size_t, std::vector<GraphScopeEvent>>
        graphScopeEvents;
    std::vector<CycleEvent> cycleEvents;
    cycleEvents.reserve(events.size());

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
            if (isGraphLinked) {
              launchEventId = events.at(launchEventId).parentEventId;
            }
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
      if (event.hasCpuTimeRange()) {
        cpuScopeEvents[event.threadId].emplace_back(
            event.id,
            event.metricSet.flexibleMetrics.empty()
                ? nullptr
                : &event.metricSet.flexibleMetrics,
            contextIdToContexts.at(event.contextId), event.threadId,
            event.cpuStartTimeNs, event.cpuEndTimeNs);
        minTimeStamp = std::min(minTimeStamp, event.cpuStartTimeNs);
      } else {
        const auto &baseContexts = contextIdToContexts.at(event.contextId);
        processMetricMaps(event.id, event.metricSet.metrics,
                          &event.metricSet.flexibleMetrics, baseContexts,
                          /*isGraphLinked=*/false);
        for (const auto &[targetEntryId, _] : event.metricSet.linkedMetrics) {
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
          processMetricMaps(event.id, linkedMetrics, flexibleMetrics, contexts,
                            /*isGraphLinked=*/true);
        }
        if (hasKernelMetrics && hasCycleMetrics) {
          throw std::runtime_error("only one active metric type is supported");
        }
      }
    }

    if (hasCycleMetrics) {
      dumpCycleMetricPerfettoTrace(cycleEvents, os);
      return;
    }

    for (auto &[threadId, cpuEvents] : cpuScopeEvents) {
      std::sort(cpuEvents.begin(), cpuEvents.end(),
                [](const CpuScopeEvent &a, const CpuScopeEvent &b) {
                  return a.startTimeNs < b.startTimeNs;
                });
    }

    if (hasKernelMetrics) {
      for (auto &[streamId, streamKernelEvents] : kernelEvents) {
        std::sort(streamKernelEvents.begin(), streamKernelEvents.end(),
                  compareKernelEvents);
        pruneDuplicateKernelEvents(streamKernelEvents);
      }
      reconstructGraphScopeEvents(kernelEvents, graphScopeEvents);
    }

    std::map<uint64_t, PerfettoTrack> tracks;
    for (const auto &[threadId, _] : cpuScopeEvents) {
      const auto laneId = getCpuLaneId(threadId);
      tracks.emplace(getPerfettoLaneTrackUuid(laneId),
                     PerfettoTrack{
                         "CPU Thread " + std::to_string(threadId),
                         kPerfettoCpuTrackOrderBase +
                             static_cast<int32_t>(threadId)});
    }
    for (const auto &[streamId, _] : graphScopeEvents) {
      const auto laneId = getGraphLaneId(streamId);
      tracks.emplace(getPerfettoLaneTrackUuid(laneId),
                     PerfettoTrack{
                         "Graph: Stream " + std::to_string(streamId),
                         kPerfettoGraphTrackOrderBase +
                             static_cast<int32_t>(streamId)});
    }
    for (const auto &[streamId, _] : kernelEvents) {
      for (const auto &event : kernelEvents.at(streamId)) {
        const auto deviceId =
            static_cast<size_t>(getKernelEventDeviceId(event));
        const auto kernelStreamId =
            static_cast<size_t>(getKernelEventStreamId(event));
        const auto laneId = getGpuLaneId(deviceId, kernelStreamId);
        const auto rank =
            kPerfettoGpuTrackOrderBase +
            static_cast<int32_t>(deviceId * 1000000 + kernelStreamId);
        const auto name =
            deviceId == 0
                ? "GPU Stream " + std::to_string(kernelStreamId)
                : "GPU Device " + std::to_string(deviceId) + " Stream " +
                      std::to_string(kernelStreamId);
        tracks.emplace(getPerfettoLaneTrackUuid(laneId),
                       PerfettoTrack{name, rank});
      }
    }

    std::unordered_map<size_t, const CpuScopeEvent *>
        launchEventIdToCpuScopeEvent;
    for (const auto &[_, cpuEvents] : cpuScopeEvents) {
      for (const auto &event : cpuEvents) {
        launchEventIdToCpuScopeEvent.emplace(event.eventId, &event);
      }
    }

    std::unordered_map<size_t, std::vector<uint64_t>> sourceEventToFlowIds;
    std::map<std::pair<size_t, size_t>, std::vector<uint64_t>>
        kernelEventToFlowIds;
    for (const auto &[streamId, streamKernelEvents] : kernelEvents) {
      auto prevLaunchEventId = Trace::Event::DummyId;
      for (size_t i = 0; i < streamKernelEvents.size(); ++i) {
        const auto &event = streamKernelEvents[i];
        if (event.launchEventId == Trace::Event::DummyId) {
          continue;
        }
        if (prevLaunchEventId == event.launchEventId && event.isGraphLinked) {
          continue;
        }
        auto launchEventIt =
            launchEventIdToCpuScopeEvent.find(event.launchEventId);
        if (launchEventIt == launchEventIdToCpuScopeEvent.end()) {
          continue;
        }
        const auto flowId = kPerfettoFlowIdBase + event.launchEventId;
        sourceEventToFlowIds[event.launchEventId].push_back(flowId);
        kernelEventToFlowIds[{streamId, i}].push_back(flowId);
        prevLaunchEventId = event.launchEventId;
      }
    }

    std::vector<PerfettoSliceEvent> perfettoEvents;
    perfettoEvents.reserve(cpuScopeEvents.size() + graphScopeEvents.size() +
                           kernelEvents.size());

    auto relativeTimestamp = [minTimeStamp](uint64_t timestamp) {
      return minTimeStamp == std::numeric_limits<uint64_t>::max()
                 ? uint64_t{0}
                 : timestamp - minTimeStamp;
    };

    for (const auto &[threadId, cpuEvents] : cpuScopeEvents) {
      const auto trackUuid =
          getPerfettoLaneTrackUuid(getCpuLaneId(threadId));
      for (const auto &event : cpuEvents) {
        PerfettoSliceEvent slice;
        slice.trackUuid = trackUuid;
        slice.startTimeNs = relativeTimestamp(event.startTimeNs);
        slice.endTimeNs = relativeTimestamp(event.endTimeNs);
        slice.annotations.push_back(makeJsonAnnotation(
            "call_stack", buildCallStackJson(event.contexts).dump()));
        if (event.flexibleMetrics != nullptr &&
            !event.flexibleMetrics->empty()) {
          slice.name =
              buildFlexibleMetricEventName(event.contexts,
                                           *event.flexibleMetrics);
          slice.category = "metric";
          slice.annotations.push_back(makeJsonAnnotation(
              "metrics", buildFlexibleMetricsJson(*event.flexibleMetrics)
                             .dump()));
        } else {
          slice.name = event.contexts.empty() ? "" : event.contexts.back().name;
          slice.category = "scope";
        }
        if (auto flowIt = sourceEventToFlowIds.find(event.eventId);
            flowIt != sourceEventToFlowIds.end()) {
          slice.flowIds = flowIt->second;
        }
        perfettoEvents.push_back(std::move(slice));
      }
    }

    for (const auto &[streamId, graphEvents] : graphScopeEvents) {
      const auto trackUuid =
          getPerfettoLaneTrackUuid(getGraphLaneId(streamId));
      for (const auto &event : graphEvents) {
        PerfettoSliceEvent slice;
        slice.trackUuid = trackUuid;
        slice.startTimeNs = relativeTimestamp(event.startTimeNs);
        slice.endTimeNs = relativeTimestamp(event.endTimeNs);
        if (event.flexibleMetrics != nullptr &&
            !event.flexibleMetrics->empty()) {
          slice.name =
              buildFlexibleMetricEventName(event.context,
                                           *event.flexibleMetrics);
          slice.category = "metric";
          slice.annotations.push_back(makeJsonAnnotation(
              "metrics", buildFlexibleMetricsJson(*event.flexibleMetrics)
                             .dump()));
        } else {
          slice.name = event.context.name;
          slice.category = "scope";
        }
        perfettoEvents.push_back(std::move(slice));
      }
    }

    for (const auto &[streamId, streamKernelEvents] : kernelEvents) {
      for (size_t i = 0; i < streamKernelEvents.size(); ++i) {
        const auto &event = streamKernelEvents[i];
        const auto trackUuid = getPerfettoLaneTrackUuid(getGpuLaneId(
            static_cast<size_t>(getKernelEventDeviceId(event)),
            static_cast<size_t>(getKernelEventStreamId(event))));
        auto *flexibleMetrics = event.flexibleMetrics;
        const auto startTimeNs = getKernelEventStartTimeNs(event);
        const auto endTimeNs = getKernelEventEndTimeNs(event);

        PerfettoSliceEvent slice;
        slice.trackUuid = trackUuid;
        slice.startTimeNs = relativeTimestamp(startTimeNs);
        slice.endTimeNs = relativeTimestamp(endTimeNs);
        slice.name = getKernelEventName(event);
        slice.category = "kernel";
        slice.annotations.push_back(makeJsonAnnotation(
            "call_stack", buildCallStackJson(event.contexts).dump()));
        if (flexibleMetrics != nullptr) {
          slice.annotations.push_back(makeJsonAnnotation(
              "metrics", buildFlexibleMetricsJson(*flexibleMetrics).dump()));
        }
        if (auto flowIt = kernelEventToFlowIds.find({streamId, i});
            flowIt != kernelEventToFlowIds.end()) {
          slice.terminatingFlowIds = flowIt->second;
        }
        perfettoEvents.push_back(std::move(slice));
      }
    }

    appendPerfettoTrace(os, tracks, perfettoEvents);
  });
}

void TraceData::doDump(std::ostream &os, OutputFormat outputFormat,
                       size_t phase) const {
  if (outputFormat == OutputFormat::ChromeTrace) {
    dumpChromeTrace(os, phase);
  } else if (outputFormat == OutputFormat::PerfettoTrace) {
    dumpPerfettoTrace(os, phase);
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
