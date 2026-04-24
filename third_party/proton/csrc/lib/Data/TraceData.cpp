#include "Data/TraceData.h"
#include "Dump/TraceDataDump.h"
#include "Profiler/Graph.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

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

void reconstructGraphScopeEvents(
    const std::map<size_t, std::vector<trace_data_dump::KernelEvent>>
        &kernelEvents,
    std::map<size_t, std::vector<trace_data_dump::GraphScopeEvent>>
        &graphScopeEvents) {
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
        graphContexts.pop_back();
      }

      const auto startTimeNs = kernelEvent.getStartTimeNs();
      const auto endTimeNs = kernelEvent.getEndTimeNs();
      if (openScopes.empty()) {
        for (const auto &context : graphContexts) {
          openScopes.push_back({context, startTimeNs});
        }
      } else {
        size_t numCommonPrefixes = 0;
        while (numCommonPrefixes < openScopes.size() &&
               numCommonPrefixes < graphContexts.size()) {
          if (openScopes[numCommonPrefixes].context !=
              graphContexts[numCommonPrefixes]) {
            break;
          }
          numCommonPrefixes++;
        }
        for (size_t i = openScopes.size(); i > numCommonPrefixes; --i) {
          auto &tailOpenScope = openScopes[i - 1];
          graphScopeEvents[streamId].push_back({tailOpenScope.context, streamId,
                                                tailOpenScope.startTimeNs,
                                                lastEndTimeNs, i - 1});
        }
        for (size_t i = openScopes.size(); i > numCommonPrefixes; --i) {
          openScopes.pop_back();
        }
        for (size_t i = numCommonPrefixes; i < graphContexts.size(); ++i) {
          const auto &context = graphContexts[i];
          openScopes.push_back({context, startTimeNs});
        }
      }
      lastEndTimeNs = std::max(lastEndTimeNs, endTimeNs);
    }

    auto &openScopes = openGraphScopes[streamId];
    for (size_t i = 0; i < openScopes.size(); ++i) {
      const auto &openScope = openScopes[i];
      graphScopeEvents[streamId].push_back(
          {openScope.context, streamId, openScope.startTimeNs, lastEndTimeNs,
           i});
    }
  }
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

std::vector<uint8_t> TraceData::toPerfettoTrace(size_t phase) const {
  std::ostringstream os;
  dumpPerfettoTrace(os, phase);
  const auto bytes = os.str();
  return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

Data::SerializedData TraceData::doSerialize(OutputFormat outputFormat,
                                            size_t phase) const {
  if (outputFormat == OutputFormat::ChromeTrace) {
    const auto jsonStr = toJsonString(phase);
    return {{jsonStr.begin(), jsonStr.end()}, /*binary=*/false};
  } else if (outputFormat == OutputFormat::PerfettoTrace) {
    return {toPerfettoTrace(phase), /*binary=*/true};
  }
  throw std::logic_error("Output format not supported");
}

template <typename CycleHandler, typename KernelHandler>
void TraceData::withTraceData(size_t phase, CycleHandler &&onCycleTrace,
                              KernelHandler &&onTraceData) const {
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
    trace_data_dump::TraceDump traceDump;
    std::unordered_map<size_t, size_t> launchEventIdToTargetEventId;

    std::unordered_map<size_t, std::vector<Context>> contextIdToContexts(
        events.size());
    for (const auto &[_, event] : events) {
      contextIdToContexts.try_emplace(event.contextId,
                                      trace->getContexts(event.contextId));
    }

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
              // event.
              launchEventId = events.at(launchEventId).parentEventId;
            }
            if (launchEventId != trace_data_dump::details::kNoLaunchEventId &&
                launchEventIdToTargetEventId.find(launchEventId) ==
                    launchEventIdToTargetEventId.end()) {
              launchEventIdToTargetEventId.emplace(launchEventId, eventId);
            }
            traceDump.kernelEvents[streamId].emplace_back(
                eventId, kernelMetric, flexibleMetrics, contexts, launchEventId,
                isGraphLinked);
            traceDump.minTimeStamp =
                std::min(traceDump.minTimeStamp, startTimeNs);
          }
          if (auto cycleIt = metrics.find(MetricKind::Cycle);
              cycleIt != metrics.end()) {
            auto *cycleMetric =
                static_cast<CycleMetric *>(cycleIt->second.get());
            traceDump.cycleEvents.emplace_back(cycleMetric, contexts);
          }
        };

    for (const auto &[_, event] : events) {
      if (event.hasCpuTimeRange()) {
        traceDump.cpuScopeEvents[event.threadId].emplace_back(
            event.id, trace_data_dump::details::kNoLaunchEventId,
            event.metricSet.flexibleMetrics.empty()
                ? nullptr
                : &event.metricSet.flexibleMetrics,
            contextIdToContexts.at(event.contextId), event.threadId,
            event.cpuStartTimeNs, event.cpuEndTimeNs);
        traceDump.minTimeStamp =
            std::min(traceDump.minTimeStamp, event.cpuStartTimeNs);
        continue;
      }

      const auto &baseContexts = contextIdToContexts.at(event.contextId);
      processMetricMaps(event.id, event.metricSet.metrics,
                        &event.metricSet.flexibleMetrics, baseContexts,
                        /*isGraphLinked=*/false);
      for (const auto &[targetEntryId, _] : event.metricSet.linkedMetrics) {
        const auto &linkedMetrics =
            event.metricSet.linkedMetrics.at(targetEntryId);
        // Combine the base contexts and virtual contexts to form the complete
        // contexts for the linked metrics.
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

      if (!traceDump.kernelEvents.empty() && !traceDump.cycleEvents.empty()) {
        throw std::runtime_error("only one active metric type is supported");
      }
    }

    for (auto &[_, cpuEvents] : traceDump.cpuScopeEvents) {
      for (auto &cpuEvent : cpuEvents) {
        if (auto targetIt = launchEventIdToTargetEventId.find(cpuEvent.eventId);
            targetIt != launchEventIdToTargetEventId.end()) {
          cpuEvent.targetEventId = targetIt->second;
        }
      }
    }
    for (auto &[_, streamKernelEvents] : traceDump.kernelEvents) {
      std::stable_sort(streamKernelEvents.begin(), streamKernelEvents.end(),
                       trace_data_dump::KernelEvent::compare);
      bool hasPreviousKernel = false;
      uint64_t previousEndTimeNs = 0;
      for (auto &kernelEvent : streamKernelEvents) {
        // Ensure the kernel events are non-overlapping and sorted by start
        // time. This is required by if we use the
        // instrumentation-based measurement where timing can have minor issues
        // for very short kernels.
        if (hasPreviousKernel && kernelEvent.startTimeNs <= previousEndTimeNs) {
          kernelEvent.startTimeNs = previousEndTimeNs + 1;
          kernelEvent.endTimeNs =
              std::max(kernelEvent.endTimeNs, kernelEvent.startTimeNs + 1);
        }
        previousEndTimeNs = kernelEvent.endTimeNs;
        hasPreviousKernel = true;

        if (auto targetIt =
                launchEventIdToTargetEventId.find(kernelEvent.launchEventId);
            targetIt != launchEventIdToTargetEventId.end()) {
          if (targetIt->second != kernelEvent.eventId) {
            kernelEvent.launchEventId =
                trace_data_dump::details::kNoLaunchEventId;
          }
        }
      }
    }

    if (!traceDump.cycleEvents.empty()) {
      std::forward<CycleHandler>(onCycleTrace)(traceDump);
      return;
    }

    if (!traceDump.kernelEvents.empty()) {
      // Graph scopes are constructed in order.
      reconstructGraphScopeEvents(traceDump.kernelEvents,
                                  traceDump.graphScopeEvents);
    }

    std::forward<KernelHandler>(onTraceData)(traceDump);
  });
}

void TraceData::dumpChromeTrace(std::ostream &os, size_t phase) const {
  withTraceData(
      phase,
      [&](trace_data_dump::TraceDump &traceDump) {
        trace_data_dump::dumpCycleEventTrace(traceDump.cycleEvents, os);
      },
      [&](trace_data_dump::TraceDump &traceDump) {
        trace_data_dump::dumpChromeTraceData(traceDump, os);
      });
}

void TraceData::dumpPerfettoTrace(std::ostream &os, size_t phase) const {
  withTraceData(
      phase,
      [&](trace_data_dump::TraceDump &) {
        throw std::logic_error(
            "cycle metric Perfetto traces are not supported yet");
      },
      [&](trace_data_dump::TraceDump &traceDump) {
        trace_data_dump::dumpPerfettoTraceData(traceDump, os);
      });
}

TraceData::TraceData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  initPhaseStore(tracePhases);
}

TraceData::~TraceData() {}

} // namespace proton
