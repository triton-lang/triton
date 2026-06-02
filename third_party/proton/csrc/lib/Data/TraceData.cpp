#include "Data/TraceData.h"
#include "Context/Context.h"
#include "Dump/TraceDataDump.h"
#include "Utility/Errors.h"
#include "Utility/MsgPackWriter.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>

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
    std::map</*stream_id=*/size_t, std::vector<trace_data_dump::KernelEvent>>
        kernelEvents;
    std::map</*thread_id=*/size_t, std::vector<trace_data_dump::CpuScopeEvent>>
        cpuScopeEvents;
    std::map</*stream_id=*/size_t,
             std::vector<trace_data_dump::GraphScopeEvent>>
        graphScopeEvents;
    std::vector<trace_data_dump::CycleEvent> cycleEvents;
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
      trace_data_dump::dumpCycleMetricTrace(cycleEvents, os);
      return;
    }

    // Keep CPU ranges stable regardless of whether kernels were recorded.
    for (auto &[threadId, events] : cpuScopeEvents) {
      std::sort(events.begin(), events.end(),
                [](const trace_data_dump::CpuScopeEvent &a,
                   const trace_data_dump::CpuScopeEvent &b) {
                  return a.startTimeNs < b.startTimeNs;
                });
    }

    if (hasKernelMetrics) {
      // Sort all kernel events in order
      for (auto &[streamId, events] : kernelEvents) {
        std::sort(events.begin(), events.end(),
                  [](const trace_data_dump::KernelEvent &a,
                     const trace_data_dump::KernelEvent &b) {
                    auto aStartTime = std::get<uint64_t>(
                        a.kernelMetric->getValue(KernelMetric::StartTime));
                    auto bStartTime = std::get<uint64_t>(
                        b.kernelMetric->getValue(KernelMetric::StartTime));
                    return aStartTime < bStartTime;
                  });
      }
      // Graph scopes are constructed in order
      trace_data_dump::reconstructGraphScopeEvents(kernelEvents,
                                                   graphScopeEvents);
      trace_data_dump::dumpKernelMetricTrace(
          minTimeStamp, kernelEvents, cpuScopeEvents, graphScopeEvents, os);
    } else if (!cpuScopeEvents.empty()) {
      trace_data_dump::dumpCpuOnlyTrace(minTimeStamp, cpuScopeEvents, os);
    } else {
      trace_data_dump::dumpEmptyChromeTrace(os);
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
