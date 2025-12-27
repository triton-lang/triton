#include "Proton.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <variant>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

using namespace proton;

// For simplicity, the Python interface restricts metrics to int64_t and double.
// without uint64_t. Allowing types such as uint64_t vs. int64_t would force
// users to handle subtle type differences for the same metric name, which would
// be confusing and error-prone.
using PythonMetricValueType = std::variant<int64_t, double>;
namespace {

std::map<std::string, MetricValueType> convertPythonMetrics(
    const std::map<std::string, PythonMetricValueType> &metrics) {
  std::map<std::string, MetricValueType> converted;
  for (const auto &[name, value] : metrics) {
    converted.emplace(name, std::visit(
                                [](auto &&v) -> MetricValueType {
                                  return MetricValueType(v);
                                },
                                value));
  }
  return converted;
}

} // namespace

static void initProton(pybind11::module &&m) {
  using ret = pybind11::return_value_policy;
  using namespace pybind11::literals;

  // Accept raw integer pointers from Python (e.g., Tensor.data_ptr()) instead
  // of requiring a PyCapsule, which matches how tensor metric values are passed
  // in transform_tensor_metrics.
  pybind11::class_<TensorMetric>(m, "TensorMetric")
      .def(pybind11::init<>())
      .def(pybind11::init([](uintptr_t ptr, size_t index) {
             return TensorMetric{reinterpret_cast<uint8_t *>(ptr), index};
           }),
           pybind11::arg("ptr"), pybind11::arg("index"))
      .def_property_readonly("ptr",
                             [](const TensorMetric &metric) {
                               return reinterpret_cast<uintptr_t>(metric.ptr);
                             })
      .def_property_readonly(
          "index", [](const TensorMetric &metric) { return metric.index; });

  m.attr("metric_int64_index") =
      pybind11::cast(variant_index_v<int64_t, MetricValueType>);
  m.attr("metric_double_index") =
      pybind11::cast(variant_index_v<double, MetricValueType>);

  m.def(
      "start",
      [](const std::string &path, const std::string &contextSourceName,
         const std::string &dataName, const std::string &profilerName,
         const std::string &mode) {
        auto sessionId = SessionManager::instance().addSession(
            path, profilerName, contextSourceName, dataName, mode);
        SessionManager::instance().activateSession(sessionId);
        return sessionId;
      },
      pybind11::arg("path"), pybind11::arg("contextSourceName"),
      pybind11::arg("dataName"), pybind11::arg("profilerName"),
      pybind11::arg("mode") = "");

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

  m.def(
      "add_metrics",
      [](size_t scopeId,
         const std::map<std::string, PythonMetricValueType> &metrics,
         const std::map<std::string, TensorMetric> &tensorMetrics) {
        auto convertedMetrics = convertPythonMetrics(metrics);
        SessionManager::instance().addMetrics(scopeId, convertedMetrics,
                                              tensorMetrics);
      },
      pybind11::arg("scopeId"), pybind11::arg("metrics"),
      pybind11::arg("tensorMetrics") = std::map<std::string, TensorMetric>());

  m.def("set_metric_kernels",
        [](uintptr_t tensorMetricKernel, uintptr_t scalarMetricKernel,
           uintptr_t stream) {
          SessionManager::instance().setMetricKernels(
              reinterpret_cast<void *>(tensorMetricKernel),
              reinterpret_cast<void *>(scalarMetricKernel),
              reinterpret_cast<void *>(stream));
        });

  m.def("get_context_depth", [](size_t sessionId) {
    return SessionManager::instance().getContextDepth(sessionId);
  });

  m.def(
      "get_data",
      [](size_t sessionId) {
        return SessionManager::instance().getData(sessionId);
      },
      pybind11::arg("sessionId"));

  m.def(
      "get_data_msgpack",
      [](size_t sessionId) {
        auto data = SessionManager::instance().getDataMsgPack(sessionId);
        return pybind11::bytes(reinterpret_cast<const char *>(data.data()),
                               data.size());
      },
      pybind11::arg("sessionId"));

  m.def(
      "clear_data",
      [](size_t sessionId) { SessionManager::instance().clearData(sessionId); },
      pybind11::arg("sessionId"));
}

PYBIND11_MODULE(libproton, m) {
  m.doc() = "Python bindings to the Proton API";
  initProton(std::move(m.def_submodule("proton")));
}
