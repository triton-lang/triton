#ifndef PROTON_SESSION_SESSION_H_
#define PROTON_SESSION_SESSION_H_

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Utility/Singleton.h"
#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <vector>

namespace proton {

class Profiler;
class Data;
enum class OutputFormat;

/// A session is a collection of profiler, context source, and data objects.
/// There could be multiple sessions in the system, each can correspond to a
/// different duration, or the same duration but with different configurations.
class Session {
public:
  ~Session() = default;

  void activate();

  void deactivate();

  void finalize(OutputFormat outputFormat);

private:
  Session(size_t id, const std::string &path, Profiler *profiler,
          std::unique_ptr<ContextSource> contextSource,
          std::unique_ptr<Data> data)
      : id(id), path(path), profiler(profiler),
        contextSource(std::move(contextSource)), data(std::move(data)) {}

  template <typename T> std::vector<T *> getInterfaces() {
    std::vector<T *> interfaces;
    if (auto interface = dynamic_cast<T *>(profiler)) {
      interfaces.push_back(interface);
    }
    if (auto interface = dynamic_cast<T *>(data.get())) {
      interfaces.push_back(interface);
    }
    if (auto interface = dynamic_cast<T *>(contextSource.get())) {
      interfaces.push_back(interface);
    }
    return interfaces;
  }

  const std::string path{};
  size_t id{};
  Profiler *profiler{};
  std::unique_ptr<ContextSource> contextSource{};
  std::unique_ptr<Data> data{};

  friend class SessionManager;
};

/// A session manager is responsible for managing the lifecycle of sessions.
/// There's a single and unique session manager in the system.
class SessionManager : public Singleton<SessionManager> {
public:
  SessionManager() = default;
  ~SessionManager() = default;

  size_t addSession(const std::string &path, const std::string &profilerName,
                    const std::string &contextSourceName,
                    const std::string &dataName);

  void finalizeSession(size_t sessionId, OutputFormat outputFormat);

  void finalizeAllSessions(OutputFormat outputFormat);

  void activateSession(size_t sessionId);

  void activateAllSessions();

  void deactivateSession(size_t sessionId);

  void deactivateAllSessions();

  void enterScope(const Scope &scope);

  void exitScope(const Scope &scope);

  void enterOp(const Scope &scope);

  void exitOp(const Scope &scope);

  void addMetrics(size_t scopeId,
                  const std::map<std::string, MetricValueType> &metrics,
                  bool aggregable);

private:
  std::unique_ptr<Session> makeSession(size_t id, const std::string &path,
                                       const std::string &profilerName,
                                       const std::string &contextSourceName,
                                       const std::string &dataName);

  void activateSessionImpl(size_t sessionId);

  void deActivateSessionImpl(size_t sessionId);

  size_t getSessionId(const std::string &path) { return sessionPaths[path]; }

  bool hasSession(const std::string &path) {
    return sessionPaths.find(path) != sessionPaths.end();
  }

  bool hasSession(size_t sessionId) {
    return sessions.find(sessionId) != sessions.end();
  }

  void removeSession(size_t sessionId);

  template <typename Interface, typename Counter>
  void registerInterface(size_t sessionId, Counter &interfaceCounts) {
    auto interfaces = sessions[sessionId]->getInterfaces<Interface>();
    for (auto *interface : interfaces) {
      interfaceCounts[interface] += 1;
    }
  }

  template <typename Interface, typename Counter>
  void unregisterInterface(size_t sessionId, Counter &interfaceCounts) {
    auto interfaces = sessions[sessionId]->getInterfaces<Interface>();
    for (auto *interface : interfaces) {
      interfaceCounts[interface] -= 1;
    }
  }

  mutable std::shared_mutex mutex;

  size_t nextSessionId{};
  // path -> session id
  std::map<std::string, size_t> sessionPaths;
  // session id -> active
  std::map<size_t, bool> sessionActive;
  // session id -> session
  std::map<size_t, std::unique_ptr<Session>> sessions;
  // scope -> active count
  std::map<ScopeInterface *, size_t> scopeInterfaceCounts;
  // op -> active count
  std::map<OpInterface *, size_t> opInterfaceCounts;
};

} // namespace proton

#endif // PROTON_SESSION_H_
