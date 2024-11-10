#include "Proton.h"

#include <map>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

using namespace proton;

void initProton(pybind11::module &&m) {
  using ret = pybind11::return_value_policy;
  using namespace pybind11::literals;

  m.def("start",
        [](const std::string &path, const std::string &contextSourceName,
           const std::string &dataName, const std::string &profilerName) {
          auto sessionId = SessionManager::instance().addSession(
              path, profilerName, contextSourceName, dataName);
          SessionManager::instance().activateSession(sessionId);
          return sessionId;
        });

  m.def("activate", [](std::optional<size_t> sessionId) {
    if (sessionId.has_value()) {
      SessionManager::instance().activateSession(*sessionId);
    } else {
      SessionManager::instance().activateAllSessions();
    }
  });

  m.def("deactivate", [](std::optional<size_t> sessionId, bool flush) {
    if (sessionId.has_value()) {
      SessionManager::instance().deactivateSession(*sessionId, flush);
    } else {
      SessionManager::instance().deactivateAllSessions(flush);
    }
  });

  m.def("finalize", [](std::optional<size_t> sessionId,
                       const std::string &outputFormat) {
    auto outputFormatEnum = parseOutputFormat(outputFormat);
    if (sessionId.has_value()) {
      SessionManager::instance().finalizeSession(*sessionId, outputFormatEnum);
    } else {
      SessionManager::instance().finalizeAllSessions(outputFormatEnum);
    }
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

  m.def("add_metrics",
        [](size_t scopeId,
           const std::map<std::string, MetricValueType> &metrics) {
          SessionManager::instance().addMetrics(scopeId, metrics,
                                                /*aggregable=*/true);
        });

  m.def("set_properties",
        [](size_t scopeId,
           const std::map<std::string, MetricValueType> &metrics) {
          SessionManager::instance().addMetrics(scopeId, metrics,
                                                /*aggregable=*/false);
        });

  pybind11::bind_map<std::map<std::string, MetricValueType>>(m, "MetricMap");
}

PYBIND11_MODULE(libproton, m) {
  m.doc() = "Python bindings to the Proton API";
  initProton(std::move(m.def_submodule("proton")));
}
