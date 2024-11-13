#include "Session/Session.h"
#include "Context/Python.h"
#include "Context/Shadow.h"
#include "Data/TreeData.h"
#include "Profiler/Cupti/CuptiProfiler.h"
#include "Profiler/Roctracer/RoctracerProfiler.h"
#include "Utility/String.h"

namespace proton {

namespace {
Profiler *getProfiler(const std::string &profilerName) {
  if (proton::toLower(profilerName) == "cupti") {
    return &CuptiProfiler::instance();
  }
  if (proton::toLower(profilerName) == "cupti_pcsampling") {
    return &CuptiProfiler::instance().enablePCSampling();
  }
  if (proton::toLower(profilerName) == "roctracer") {
    return &RoctracerProfiler::instance();
  }
  throw std::runtime_error("Unknown profiler: " + profilerName);
}

std::unique_ptr<Data> makeData(const std::string &dataName,
                               const std::string &path,
                               ContextSource *contextSource) {
  if (toLower(dataName) == "tree") {
    return std::make_unique<TreeData>(path, contextSource);
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
}

void Session::finalize(OutputFormat outputFormat) {
  profiler->stop();
  data->dump(outputFormat);
}

std::unique_ptr<Session> SessionManager::makeSession(
    size_t id, const std::string &path, const std::string &profilerName,
    const std::string &contextSourceName, const std::string &dataName) {
  auto profiler = getProfiler(profilerName);
  auto contextSource = makeContextSource(contextSourceName);
  auto data = makeData(dataName, path, contextSource.get());
  auto *session = new Session(id, path, profiler, std::move(contextSource),
                              std::move(data));
  return std::unique_ptr<Session>(session);
}

void SessionManager::activateSession(size_t sessionId) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  activateSessionImpl(sessionId);
}

void SessionManager::deactivateSession(size_t sessionId) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  deActivateSessionImpl(sessionId);
}

void SessionManager::activateSessionImpl(size_t sessionId) {
  throwIfSessionNotInitialized(sessions, sessionId);
  if (sessionActive[sessionId])
    return;
  sessionActive[sessionId] = true;
  sessions[sessionId]->activate();
  registerInterface<ScopeInterface>(sessionId, scopeInterfaceCounts);
  registerInterface<OpInterface>(sessionId, opInterfaceCounts);
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
}

void SessionManager::removeSession(size_t sessionId) {
  if (!hasSession(sessionId)) {
    return;
  }
  auto path = sessions[sessionId]->path;
  sessionPaths.erase(path);
  sessions.erase(sessionId);
}

size_t SessionManager::addSession(const std::string &path,
                                  const std::string &profilerName,
                                  const std::string &contextSourceName,
                                  const std::string &dataName) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  if (hasSession(path)) {
    auto sessionId = getSessionId(path);
    activateSessionImpl(sessionId);
    return sessionId;
  }
  auto sessionId = nextSessionId++;
  sessionPaths[path] = sessionId;
  sessions[sessionId] =
      makeSession(sessionId, path, profilerName, contextSourceName, dataName);
  return sessionId;
}

void SessionManager::finalizeSession(size_t sessionId,
                                     OutputFormat outputFormat) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  if (!hasSession(sessionId)) {
    return;
  }
  deActivateSessionImpl(sessionId);
  sessions[sessionId]->finalize(outputFormat);
  removeSession(sessionId);
}

void SessionManager::finalizeAllSessions(OutputFormat outputFormat) {
  std::unique_lock<std::shared_mutex> lock(mutex);
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
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (auto iter : scopeInterfaceCounts) {
    auto [scopeInterface, count] = iter;
    if (count > 0) {
      scopeInterface->enterScope(scope);
    }
  }
}

void SessionManager::exitScope(const Scope &scope) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (auto iter : scopeInterfaceCounts) {
    auto [scopeInterface, count] = iter;
    if (count > 0) {
      scopeInterface->exitScope(scope);
    }
  }
}

void SessionManager::enterOp(const Scope &scope) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (auto iter : opInterfaceCounts) {
    auto [opInterface, count] = iter;
    if (count > 0) {
      opInterface->enterOp(scope);
    }
  }
}

void SessionManager::exitOp(const Scope &scope) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (auto iter : opInterfaceCounts) {
    auto [opInterface, count] = iter;
    if (count > 0) {
      opInterface->exitOp(scope);
    }
  }
}

void SessionManager::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics,
    bool aggregable) {
  std::shared_lock<std::shared_mutex> lock(mutex);
  for (auto [sessionId, active] : sessionActive) {
    if (active) {
      sessions[sessionId]->data->addMetrics(scopeId, metrics, aggregable);
    }
  }
}

} // namespace proton
