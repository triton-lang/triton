#include "Session/Session.h"
#include "Context/Python.h"
#include "Context/Shadow.h"
#include "Data/TraceData.h"
#include "Data/TreeData.h"
#include "Profiler/Cupti/CuptiProfiler.h"
#include "Profiler/Instrumentation/InstrumentationProfiler.h"
#include "Profiler/Roctracer/RoctracerProfiler.h"
#include "Utility/String.h"

namespace proton {

namespace {

Profiler *makeProfiler(const std::string &name) {
  if (proton::toLower(name) == "cupti") {
    return &CuptiProfiler::instance();
  } else if (proton::toLower(name) == "roctracer") {
    return &RoctracerProfiler::instance();
  } else if (proton::toLower(name) == "instrumentation") {
    return &InstrumentationProfiler::instance();
  }
  throw std::runtime_error("Unknown profiler: " + name);
}

std::unique_ptr<Data> makeData(const std::string &dataName,
                               const std::string &path,
                               ContextSource *contextSource) {
  if (toLower(dataName) == "tree") {
    return std::make_unique<TreeData>(path, contextSource);
  } else if (toLower(dataName) == "trace") {
    return std::make_unique<TraceData>(path, contextSource);
  }
  throw std::runtime_error("Unknown data: " + dataName);
}

std::unique_ptr<ContextSource>
makeContextSource(const std::string &contextSourceName) {
  if (toLower(contextSourceName) == "shadow") {
    return std::make_unique<ShadowContextSource>();
  } else if (toLower(contextSourceName) == "python") {
    return std::make_unique<PythonContextSource>();
  }
  throw std::runtime_error("Unknown context source: " + contextSourceName);
}

void throwIfSessionNotInitialized(
    const std::map<size_t, std::unique_ptr<Session>> &sessions,
    size_t sessionId) {
  if (!sessions.count(sessionId)) {
    throw std::runtime_error("Session has not been initialized: " +
                             std::to_string(sessionId));
  }
}

} // namespace

void Session::activate() {
  profiler->start();
  profiler->flush();
  profiler->registerData(data.get());
}

void Session::deactivate() {
  profiler->flush();
  profiler->unregisterData(data.get());
  data->clearCache();
}

void Session::finalize(const std::string &outputFormat) {
  profiler->stop();
  data->dump(outputFormat);
}

size_t Session::getContextDepth() { return contextSource->getDepth(); }

Profiler *SessionManager::validateAndSetProfilerMode(Profiler *profiler,
                                                     const std::string &mode) {
  std::vector<std::string> modeAndOptions = proton::split(mode, ":");
  for (auto &[id, session] : sessions) {
    if (session->getProfiler() == profiler &&
        session->getProfiler()->getMode() != modeAndOptions) {
      throw std::runtime_error("Cannot add a session with the same profiler "
                               "but a different mode than existing sessions");
    }
  }
  return profiler->setMode(modeAndOptions);
}

std::unique_ptr<Session> SessionManager::makeSession(
    size_t id, const std::string &path, const std::string &profilerName,
    const std::string &contextSourceName, const std::string &dataName,
    const std::string &mode) {
  auto *profiler = makeProfiler(profilerName);
  profiler = validateAndSetProfilerMode(profiler, mode);
  auto contextSource = makeContextSource(contextSourceName);
  auto data = makeData(dataName, path, contextSource.get());
  auto *session = new Session(id, path, profiler, std::move(contextSource),
                              std::move(data));
  return std::unique_ptr<Session>(session);
}

void SessionManager::activateSession(size_t sessionId) {
  std::lock_guard<std::mutex> lock(mutex);
  activateSessionImpl(sessionId);
}

void SessionManager::activateAllSessions() {
  std::lock_guard<std::mutex> lock(mutex);
  for (auto iter : sessionActive) {
    activateSessionImpl(iter.first);
  }
}

void SessionManager::deactivateSession(size_t sessionId) {
  std::lock_guard<std::mutex> lock(mutex);
  deActivateSessionImpl(sessionId);
}

void SessionManager::deactivateAllSessions() {
  std::lock_guard<std::mutex> lock(mutex);
  for (auto iter : sessionActive) {
    deActivateSessionImpl(iter.first);
  }
}

void SessionManager::activateSessionImpl(size_t sessionId) {
  throwIfSessionNotInitialized(sessions, sessionId);
  if (sessionActive[sessionId])
    return;
  sessionActive[sessionId] = true;
  sessions[sessionId]->activate();
  registerInterface<ScopeInterface>(sessionId, scopeInterfaceCounts);
  registerInterface<OpInterface>(sessionId, opInterfaceCounts);
  registerInterface<InstrumentationInterface>(sessionId,
                                              instrumentationInterfaceCounts);
  registerInterface<ContextSource>(sessionId, contextSourceCounts);
  registerInterface<MetricInterface>(sessionId, metricInterfaceCounts);
}

void SessionManager::deActivateSessionImpl(size_t sessionId) {
  throwIfSessionNotInitialized(sessions, sessionId);
  if (!sessionActive[sessionId]) {
    return;
  }
  sessionActive[sessionId] = false;
  sessions[sessionId]->deactivate();
  unregisterInterface<ScopeInterface>(sessionId, scopeInterfaceCounts);
  unregisterInterface<OpInterface>(sessionId, opInterfaceCounts);
  unregisterInterface<InstrumentationInterface>(sessionId,
                                                instrumentationInterfaceCounts);
  unregisterInterface<ContextSource>(sessionId, contextSourceCounts);
  unregisterInterface<MetricInterface>(sessionId, metricInterfaceCounts);
}

void SessionManager::removeSession(size_t sessionId) {
  if (!hasSession(sessionId)) {
    return;
  }
  auto path = sessions[sessionId]->path;
  sessionPaths.erase(path);
  sessionActive.erase(sessionId);
  sessions.erase(sessionId);
}

size_t SessionManager::addSession(const std::string &path,
                                  const std::string &profilerName,
                                  const std::string &contextSourceName,
                                  const std::string &dataName,
                                  const std::string &mode) {
  std::lock_guard<std::mutex> lock(mutex);
  if (hasSession(path)) {
    auto sessionId = getSessionId(path);
    activateSessionImpl(sessionId);
    return sessionId;
  }
  auto sessionId = nextSessionId++;
  auto newSession = makeSession(sessionId, path, profilerName,
                                contextSourceName, dataName, mode);
  sessionPaths[path] = sessionId;
  sessions[sessionId] = std::move(newSession);
  return sessionId;
}

void SessionManager::finalizeSession(size_t sessionId,
                                     const std::string &outputFormat) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!hasSession(sessionId)) {
    return;
  }
  deActivateSessionImpl(sessionId);
  sessions[sessionId]->finalize(outputFormat);
  removeSession(sessionId);
}

void SessionManager::finalizeAllSessions(const std::string &outputFormat) {
  std::lock_guard<std::mutex> lock(mutex);
  auto sessionIds = std::vector<size_t>{};
  for (auto &[sessionId, session] : sessions) {
    deActivateSessionImpl(sessionId);
    session->finalize(outputFormat);
    sessionIds.push_back(sessionId);
  }
  for (auto sessionId : sessionIds) {
    removeSession(sessionId);
  }
}

void SessionManager::enterScope(const Scope &scope) {
  std::lock_guard<std::mutex> lock(mutex);
  executeInterface(scopeInterfaceCounts, [&](auto *scopeInterface) {
    scopeInterface->enterScope(scope);
  });
}

void SessionManager::exitScope(const Scope &scope) {
  std::lock_guard<std::mutex> lock(mutex);
  executeInterface(
      scopeInterfaceCounts,
      [&](auto *scopeInterface) { scopeInterface->exitScope(scope); },
      /*isReversed=*/true);
}

void SessionManager::enterOp(const Scope &scope) {
  std::lock_guard<std::mutex> lock(mutex);
  executeInterface(opInterfaceCounts,
                   [&](auto *opInterface) { opInterface->enterOp(scope); });
}

void SessionManager::exitOp(const Scope &scope) {
  std::lock_guard<std::mutex> lock(mutex);
  executeInterface(
      opInterfaceCounts, [&](auto *opInterface) { opInterface->exitOp(scope); },
      /*isReversed=*/true);
}

void SessionManager::initFunctionMetadata(
    uint64_t functionId, const std::string &functionName,
    const std::vector<std::pair<size_t, std::string>> &scopeIdNames,
    const std::vector<std::pair<size_t, size_t>> &scopeIdParents,
    const std::string &metadataPath) {
  std::lock_guard<std::mutex> lock(mutex);
  executeInterface(instrumentationInterfaceCounts,
                   [&](auto *instrumentationInterface) {
                     instrumentationInterface->initFunctionMetadata(
                         functionId, functionName, scopeIdNames, scopeIdParents,
                         metadataPath);
                   });
}

void SessionManager::enterInstrumentedOp(uint64_t streamId, uint64_t functionId,
                                         uint8_t *buffer, size_t size) {
  std::lock_guard<std::mutex> lock(mutex);
  executeInterface(instrumentationInterfaceCounts,
                   [&](auto *instrumentationInterface) {
                     instrumentationInterface->enterInstrumentedOp(
                         streamId, functionId, buffer, size);
                   });
}

void SessionManager::exitInstrumentedOp(uint64_t streamId, uint64_t functionId,
                                        uint8_t *buffer, size_t size) {
  std::lock_guard<std::mutex> lock(mutex);
  executeInterface(
      instrumentationInterfaceCounts,
      [&](auto *instrumentationInterface) {
        instrumentationInterface->exitInstrumentedOp(streamId, functionId,
                                                     buffer, size);
      },
      /*isReversed=*/true);
}

void SessionManager::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &scalarMetrics,
    const std::map<std::string, TensorMetric> &tensorMetrics) {
  std::lock_guard<std::mutex> lock(mutex);
  executeInterface(metricInterfaceCounts, [&](auto *metricInterface) {
    metricInterface->addMetrics(scopeId, scalarMetrics, tensorMetrics);
  });
}

void SessionManager::setMetricKernels(void *tensorMetricKernel,
                                      void *scalarMetricKernel, void *stream) {
  std::lock_guard<std::mutex> lock(mutex);
  executeInterface(metricInterfaceCounts, [&](auto *metricInterface) {
    metricInterface->setMetricKernels(tensorMetricKernel, scalarMetricKernel,
                                      stream);
  });
}

void SessionManager::setState(std::optional<Context> context) {
  std::lock_guard<std::mutex> lock(mutex);
  for (auto iter : contextSourceCounts) {
    auto [contextSource, count] = iter;
    if (count > 0) {
      contextSource->setState(context);
    }
  }
}

size_t SessionManager::getContextDepth(size_t sessionId) {
  std::lock_guard<std::mutex> lock(mutex);
  throwIfSessionNotInitialized(sessions, sessionId);
  return sessions[sessionId]->getContextDepth();
}

std::string SessionManager::getData(size_t sessionId) {
  std::lock_guard<std::mutex> lock(mutex);
  throwIfSessionNotInitialized(sessions, sessionId);
  auto *profiler = sessions[sessionId]->getProfiler();
  auto dataSet = profiler->getDataSet();
  if (dataSet.find(sessions[sessionId]->data.get()) != dataSet.end()) {
    throw std::runtime_error(
        "Cannot get data while the session is active. Please deactivate the "
        "session first.");
  }
  auto *treeData = dynamic_cast<TreeData *>(sessions[sessionId]->data.get());
  if (!treeData) {
    throw std::runtime_error(
        "Only TreeData is supported for getData() for now");
  }
  return treeData->toJsonString();
}

void SessionManager::clearData(size_t sessionId) {
  std::lock_guard<std::mutex> lock(mutex);
  throwIfSessionNotInitialized(sessions, sessionId);
  auto *profiler = sessions[sessionId]->getProfiler();
  auto dataSet = profiler->getDataSet();
  if (dataSet.find(sessions[sessionId]->data.get()) != dataSet.end()) {
    throw std::runtime_error(
        "Cannot clear data while the session is active. Please deactivate the "
        "session first.");
  }
  sessions[sessionId]->data->clear();
}

} // namespace proton
