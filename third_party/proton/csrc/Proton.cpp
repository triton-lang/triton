#include "Proton.h"
#include "Driver/GPU/Cuda.h"

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
        [](const std::string &path, const std::string &profilerName,
           const std::string &contextSourceName, const std::string &dataName) {
          auto sessionId = SessionManager::instance().addSession(
              path, profilerName, contextSourceName, dataName);
          SessionManager::instance().activateSession(sessionId);
          return sessionId;
        });

  m.def("activate", [](size_t sessionId) {
    SessionManager::instance().activateSession(sessionId);
  });

  m.def("deactivate", [](size_t sessionId) {
    SessionManager::instance().deactivateSession(sessionId);
  });

  m.def("finalize", [](size_t sessionId, const std::string &outputFormat) {
    auto outputFormatEnum = parseOutputFormat(outputFormat);
    SessionManager::instance().finalizeSession(sessionId, outputFormatEnum);
  });

  m.def("finalize_all", [](const std::string &outputFormat) {
    auto outputFormatEnum = parseOutputFormat(outputFormat);
    SessionManager::instance().finalizeAllSessions(outputFormatEnum);
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
          SessionManager::instance().addMetrics(scopeId, metrics);
        });

  m.def("device_info", [](int device_id) {
    std::map<std::string, int> devAttrs;
    CUdevice device;
    CUcontext context;

    cuda::init<true>(0);
    cuda::ctxGetCurrent<true>(&context);
    cuda::deviceGet<true>(&device, device_id);

#define FILL_DEVICE_ATTRIBUTE(NAME)                                            \
  cuda::deviceGetAttribute<true>(&devAttrs[#NAME], NAME, device)

    FILL_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    FILL_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    FILL_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
    FILL_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
#undef FILL_DEVICE_ATTRIBUTE
    return devAttrs;
  });

  pybind11::bind_map<std::map<std::string, MetricValueType>>(m, "MetricMap");
}

PYBIND11_MODULE(libproton, m) {
  m.doc() = "Python bindings to the Proton API";
  initProton(std::move(m.def_submodule("proton")));
}
