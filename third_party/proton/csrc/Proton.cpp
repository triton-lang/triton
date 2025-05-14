#include "Proton.h"

#include <map>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

using namespace proton;

static void initProton(pybind11::module &&m) {
  using ret = pybind11::return_value_policy;
  using namespace pybind11::literals;

  m.def(
      "start",
      [](const std::string &path, const std::string &contextSourceName,
         const std::string &dataName, const std::string &profilerName,
         const std::string &mode, const std::string &profilerPath) {
        auto sessionId = SessionManager::instance().addSession(
            path, profilerName, profilerPath, contextSourceName, dataName,
            mode);
        SessionManager::instance().activateSession(sessionId);
        return sessionId;
      },
      pybind11::arg("path"), pybind11::arg("contextSourceName"),
      pybind11::arg("dataName"), pybind11::arg("profilerName"),
      pybind11::arg("mode") = "", pybind11::arg("profilerPath") = "");

  m.def("activate", [](size_t sessionId) {
    SessionManager::instance().activateSession(sessionId);
  });

  m.def("activate_all",
        []() { SessionManager::instance().activateAllSessions(); });

  m.def("deactivate", [](size_t sessionId) {
    SessionManager::instance().deactivateSession(sessionId);
  });

  m.def("deactivate_all",
        []() { SessionManager::instance().deactivateAllSessions(); });

  m.def("finalize", [](size_t sessionId, const std::string &outputFormat) {
    SessionManager::instance().finalizeSession(sessionId, outputFormat);
  });

  m.def("finalize_all", [](const std::string &outputFormat) {
    SessionManager::instance().finalizeAllSessions(outputFormat);
  });

  m.def("record_scope", []() { return Scope::getNewScopeId(); });

  m.def("enter_scope", [](size_t scopeId, const std::string &name) {
    SessionManager::instance().enterScope(Scope(scopeId, name));
  });

  m.def("exit_scope", [](size_t scopeId, const std::string &name) {
    SessionManager::instance().exitScope(Scope(scopeId, name));
  });

  m.def("enter_op", [](size_t scopeId, const std::string &name) {
    SessionManager::instance().enterOp(Scope(scopeId, name));
  });

  m.def("exit_op", [](size_t scopeId, const std::string &name) {
    SessionManager::instance().exitOp(Scope(scopeId, name));
  });

  m.def("init_function_metadata",
        [](uint64_t functionId, const std::string &functionName,
           const std::vector<std::pair<size_t, std::string>> &scopeIdNames,
           const std::vector<std::pair<size_t, size_t>> &scopeIdParents,
           const std::string &metadataPath) {
          SessionManager::instance().initFunctionMetadata(
              functionId, functionName, scopeIdNames, scopeIdParents,
              metadataPath);
        });

  m.def("enter_instrumented_op", [](uint64_t streamId, uint64_t functionId,
                                    uint64_t buffer, size_t size) {
    SessionManager::instance().enterInstrumentedOp(
        streamId, functionId, reinterpret_cast<uint8_t *>(buffer), size);
  });

  m.def("exit_instrumented_op", [](uint64_t streamId, uint64_t functionId,
                                   uint64_t buffer, size_t size) {
    SessionManager::instance().exitInstrumentedOp(
        streamId, functionId, reinterpret_cast<uint8_t *>(buffer), size);
  });

  m.def("enter_state", [](const std::string &state) {
    SessionManager::instance().setState(state);
  });

  m.def("exit_state",
        []() { SessionManager::instance().setState(std::nullopt); });

  m.def("add_metrics",
        [](size_t scopeId,
           const std::map<std::string, MetricValueType> &metrics) {
          SessionManager::instance().addMetrics(scopeId, metrics);
        });

  m.def("get_context_depth", [](size_t sessionId) {
    return SessionManager::instance().getContextDepth(sessionId);
  });

  pybind11::bind_map<std::map<std::string, MetricValueType>>(m, "MetricMap");
}

PYBIND11_MODULE(libproton, m) {
  m.doc() = "Python bindings to the Proton API";
  initProton(std::move(m.def_submodule("proton")));
}
