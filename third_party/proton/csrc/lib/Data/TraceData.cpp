#include "Data/TraceData.h"
#include "Utility/Errors.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <iostream>
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
  if (!trace->hasEvent(scopeId)) // TODO: custom metrics not supported yet
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
  if (!trace->hasEvent(scopeId)) // TODO: custom metrics not supported yet
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

void TraceData::dumpChromeTrace(std::ostream &os) const {
  auto events = trace->getEvents();
  // stream id -> trace event
  std::map<size_t, std::vector<Trace::TraceEvent>> streamTraceEvents;
  uint64_t minTimeStamp = std::numeric_limits<uint64_t>::max();
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
    }
  }

  // 3) for each streamId in ascending order, emit one JSON line
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
