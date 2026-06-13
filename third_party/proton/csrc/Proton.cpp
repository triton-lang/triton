#include "Proton.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "Backend/Backend.h"
#include "Context/Context.h"
#include "Session/Session.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

using namespace proton;

// For simplicity, the Python interface restricts *scalar* metrics to int64_t
// and double (i.e. no uint64_t) to avoid subtle signed-vs-unsigned differences
// for the same metric name. For vector-valued (FlexibleMetric) metrics, mirror
// the scalar restriction (int64_t / double).
using PythonMetricValueType =
    std::variant<int64_t, double, std::vector<int64_t>, std::vector<double>>;
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

static void initProton(nanobind::module_ &m) {
  using ret = nanobind::rv_policy;
  using namespace nanobind::literals;

  // Accept raw integer pointers from Python (e.g., Tensor.data_ptr()) instead
  // of requiring a PyCapsule, which matches how tensor metric values are passed
  // in transform_tensor_metrics.
  nanobind::class_<TensorMetric>(m, "TensorMetric")
      .def(nanobind::init<>())
      .def(nanobind::new_([](uintptr_t ptr, size_t typeIndex, uint64_t size) {
             return new TensorMetric{reinterpret_cast<uint8_t *>(ptr),
                                     typeIndex, size};
           }),
           nanobind::arg("ptr"), nanobind::arg("index"),
           nanobind::arg("size") = 1)
      .def_prop_ro("ptr",
                   [](const TensorMetric &metric) {
                     return reinterpret_cast<uintptr_t>(metric.ptr);
                   })
      .def_prop_ro("index",
                   [](const TensorMetric &metric) { return metric.typeIndex; })
      .def_prop_ro("size",
                   [](const TensorMetric &metric) { return metric.size; });

  auto metricTypeInt64Index =
      nanobind::cast(variant_index_v<int64_t, MetricValueType>);
  auto metricTypeDoubleIndex =
      nanobind::cast(variant_index_v<double, MetricValueType>);
  auto metricTypeVectorInt64Index =
      nanobind::cast(variant_index_v<std::vector<int64_t>, MetricValueType>);
  auto metricTypeVectorDoubleIndex =
      nanobind::cast(variant_index_v<std::vector<double>, MetricValueType>);

  m.attr("metadata_scope_name") = std::string(kMetadataScopeName);
  m.attr("metadata_scope_prefix") = std::string(kMetadataScopePrefix);
  m.attr("metric_type_int64_index") = metricTypeInt64Index;
  m.attr("metric_type_double_index") = metricTypeDoubleIndex;
  m.attr("metric_type_vector_int64_index") = metricTypeVectorInt64Index;
  m.attr("metric_type_vector_double_index") = metricTypeVectorDoubleIndex;

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
      nanobind::arg("path"), nanobind::arg("contextSourceName"),
      nanobind::arg("dataName"), nanobind::arg("profilerName"),
      nanobind::arg("mode") = "");

  m.def("activate", [](size_t sessionId) {
    SessionManager::instance().activateSession(sessionId);
  });

  m.def("activate_all",
        []() { SessionManager::instance().activateAllSessions(); });

  m.def("deactivate", [](size_t sessionId, bool flushing) {
    SessionManager::instance().deactivateSession(sessionId, flushing);
  });

  m.def("deactivate_all", [](bool flushing) {
    SessionManager::instance().deactivateAllSessions(flushing);
  });

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
  m.def("destroy_function_metadata", [](uint64_t functionId) {
    SessionManager::instance().destroyFunctionMetadata(functionId);
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
      nanobind::arg("scopeId"), nanobind::arg("metrics"),
      nanobind::arg("tensorMetrics") = std::map<std::string, TensorMetric>());

  m.def(
      "set_metric_kernels",
      [](uintptr_t tensorMetricKernel, uintptr_t scalarMetricKernel,
         uintptr_t stream, unsigned int tensorMetricKernelNumThreads,
         unsigned int tensorMetricKernelSharedMemBytes,
         unsigned int scalarMetricKernelNumThreads,
         unsigned int scalarMetricKernelSharedMemBytes) {
        MetricKernelLaunchState metricKernelLaunchState{
            MetricKernelLaunchConfig{
                reinterpret_cast<void *>(tensorMetricKernel),
                reinterpret_cast<void *>(stream), tensorMetricKernelNumThreads,
                tensorMetricKernelSharedMemBytes},
            MetricKernelLaunchConfig{
                reinterpret_cast<void *>(scalarMetricKernel),
                reinterpret_cast<void *>(stream), scalarMetricKernelNumThreads,
                scalarMetricKernelSharedMemBytes}};
        SessionManager::instance().setMetricKernels(metricKernelLaunchState);
      },
      nanobind::arg("tensorMetricKernel"), nanobind::arg("scalarMetricKernel"),
      nanobind::arg("stream"),
      nanobind::arg("tensorMetricKernelNumThreads") = 1,
      nanobind::arg("tensorMetricKernelSharedMemBytes") = 0,
      nanobind::arg("scalarMetricKernelNumThreads") = 1,
      nanobind::arg("scalarMetricKernelSharedMemBytes") = 0);

  m.def("get_context_depth", [](size_t sessionId) {
    return SessionManager::instance().getContextDepth(sessionId);
  });

  m.def(
      "get_data",
      [](size_t sessionId, size_t phase) {
        return SessionManager::instance().getData(sessionId, phase);
      },
      nanobind::arg("sessionId"), nanobind::arg("phase"));

  m.def(
      "get_data_msgpack",
      [](size_t sessionId, size_t phase) {
        auto data = SessionManager::instance().getDataMsgPack(sessionId, phase);
        return nanobind::bytes(reinterpret_cast<const char *>(data.data()),
                               data.size());
      },
      nanobind::arg("sessionId"), nanobind::arg("phase"));
  m.def(
      "clear_data",
      [](size_t sessionId, size_t phase, bool clearUpToPhase) {
        SessionManager::instance().clearData(sessionId, phase, clearUpToPhase);
      },
      nanobind::arg("sessionId"), nanobind::arg("phase"),
      nanobind::arg("clearUpToPhase") = false);
  m.def(
      "advance_data_phase",
      [](size_t sessionId) {
        return SessionManager::instance().advanceDataPhase(sessionId);
      },
      nanobind::arg("sessionId"));
  m.def(
      "is_data_phase_complete",
      [](size_t sessionId, size_t phase) {
        return SessionManager::instance().isDataPhaseComplete(sessionId, phase);
      },
      nanobind::arg("sessionId"), nanobind::arg("phase"));
  m.def("get_available_profilers",
        []() { return getRegisteredProfilerNames(); });
  m.def(
      "select_profiler_from_triton_backend",
      [](const std::string &tritonBackend) {
        const auto profiler = getProfilerForTritonBackend(tritonBackend);
        if (profiler.has_value()) {
          return profiler.value();
        }
        auto message =
            "No profiler registered for triton backend " + tritonBackend;
        throw nanobind::value_error(message.c_str());
      },
      nanobind::arg("tritonBackend"));
}

NB_MODULE(libproton, m) {
  m.doc() = "Python bindings to the Proton API";
  auto proton_m = m.def_submodule("proton");
  initProton(proton_m);
}
