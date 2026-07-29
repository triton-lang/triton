#include "Analysis/ScopeIdAllocation.h"
#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "Conversion/ProtonToProtonGPU/Passes.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "Dialect/ProtonGPU/Transforms/Passes.h"
#include "ir.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;
using namespace mlir::triton;

void init_triton_proton(py::module_ &m) {
  m.doc() = "Python bindings to the Proton backend";

  // Proton enums
  py::enum_<proton::MetricType>(m, "METRIC_TYPE")
      .value("CYCLE", proton::MetricType::CYCLE)
      .export_values();

  py::enum_<proton::MetricValueType>(m, "METRIC_VALUE_TYPE")
      .value("NONE", proton::MetricValueType::NONE)
      .value("BOOL", proton::MetricValueType::BOOL)
      .value("I8", proton::MetricValueType::I8)
      .value("I16", proton::MetricValueType::I16)
      .value("I32", proton::MetricValueType::I32)
      .value("U8", proton::MetricValueType::U8)
      .value("U16", proton::MetricValueType::U16)
      .value("U32", proton::MetricValueType::U32)
      .value("F16", proton::MetricValueType::F16)
      .value("BF16", proton::MetricValueType::BF16)
      .value("F32", proton::MetricValueType::F32)
      .export_values();

  py::enum_<proton::SamplingStrategy>(m, "SAMPLING_STRATEGY")
      .value("NONE", proton::SamplingStrategy::NONE)
      .value("SELECTIVE", proton::SamplingStrategy::SELECTIVE)
      .export_values();

  // ProtonGPU enums
  py::enum_<proton::gpu::Granularity>(m, "GRANULARITY")
      .value("CTA", proton::gpu::Granularity::CTA)
      .value("WARP", proton::gpu::Granularity::WARP)
      .value("WARP_2", proton::gpu::Granularity::WARP_2)
      .value("WARP_4", proton::gpu::Granularity::WARP_4)
      .value("WARP_8", proton::gpu::Granularity::WARP_8)
      .value("WARP_GROUP", proton::gpu::Granularity::WARP_GROUP)
      .value("WARP_GROUP_2", proton::gpu::Granularity::WARP_GROUP_2)
      .value("WARP_GROUP_4", proton::gpu::Granularity::WARP_GROUP_4)
      .value("WARP_GROUP_8", proton::gpu::Granularity::WARP_GROUP_8)
      .export_values();

  py::enum_<proton::gpu::BufferStrategy>(m, "BUFFER_STRATEGY")
      .value("CIRCULAR", proton::gpu::BufferStrategy::CIRCULAR)
      .value("FLUSH", proton::gpu::BufferStrategy::FLUSH)
      .export_values();

  py::enum_<proton::gpu::BufferType>(m, "BUFFER_TYPE")
      .value("SHARED", proton::gpu::BufferType::SHARED)
      .value("GLOBAL", proton::gpu::BufferType::GLOBAL)
      .export_values();

  // Load proton dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<proton::ProtonDialect>();
    registry.insert<proton::gpu::ProtonGPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("get_scope_id_names", [](mlir::ModuleOp &module) {
    return proton::ModuleScopeIdAllocation(module).getScopeIdNames();
  });

  m.def("get_scope_id_parents", [](mlir::ModuleOp &module) {
    return proton::ModuleScopeIdAllocation(module).getScopeIdParents();
  });

  m.def("get_scope_id_metrics", [](mlir::ModuleOp &module) {
    return proton::ModuleScopeIdAllocation(module).getScopeIdMetrics();
  });

  // Proton operations
  m.def("create_proton_record",
        [](TritonOpBuilder &opBuilder, bool isStart,
           const std::string &name) -> void {
          auto nameAttr = mlir::StringAttr::get(opBuilder.getContext(),
                                                llvm::StringRef(name));
          opBuilder.create<proton::RecordOp>(isStart, nameAttr, mlir::Value(),
                                             mlir::StringAttr(),
                                             proton::MetricValueType::NONE);
        });

  m.def("create_proton_metric_record",
        [](TritonOpBuilder &opBuilder, const std::string &name,
           const std::string &metricName, mlir::Value metric,
           proton::MetricValueType metricType) -> void {
          auto nameAttr = mlir::StringAttr::get(opBuilder.getContext(), name);
          auto metricNameAttr =
              mlir::StringAttr::get(opBuilder.getContext(), metricName);
          opBuilder.create<proton::RecordOp>(true, nameAttr, metric,
                                             metricNameAttr, metricType);
        });

  m.def(
      "create_proton_allocate_event",
      [](TritonOpBuilder &opBuilder, const std::string &name) -> mlir::Value {
        auto nameAttr = mlir::StringAttr::get(opBuilder.getContext(), name);
        auto op = opBuilder.create<proton::AllocateEventOp>(
            mlir::TypeRange{mlir::IntegerType::get(opBuilder.getContext(), 32)},
            nameAttr);
        return op.getEvent();
      });

  m.def("create_proton_start_event",
        [](TritonOpBuilder &opBuilder, mlir::Value event) -> void {
          opBuilder.create<proton::EventOp>(true, event);
        });

  m.def("create_proton_end_event",
        [](TritonOpBuilder &opBuilder, mlir::Value event) -> void {
          opBuilder.create<proton::EventOp>(false, event);
        });

  m.def("create_proton_mark",
        [](TritonOpBuilder &opBuilder, const std::string &name) -> void {
          auto nameAttr = mlir::StringAttr::get(opBuilder.getContext(), name);
          opBuilder.create<proton::MarkOp>(nameAttr);
        });

  m.def("add_convert_proton_to_protongpu",
        [](mlir::PassManager &pm, proton::MetricType &metricType,
           proton::SamplingStrategy samplingStrategy,
           const std::string &samplingOptions,
           proton::gpu::Granularity granularity,
           proton::gpu::BufferStrategy bufferStrategy,
           proton::gpu::BufferType bufferType, int32_t bufferSize,
           int32_t maxSharedMemSize, int64_t profileScratchSize,
           int32_t profileScratchAlignment, bool clkExt) {
          pm.addPass(proton::createConvertProtonToProtonGPUPass(
              metricType, samplingStrategy, samplingOptions, granularity,
              bufferStrategy, bufferType, bufferSize, maxSharedMemSize,
              profileScratchSize, profileScratchAlignment, clkExt));
        });

  ADD_PASS_WRAPPER_0("add_convert_proton_nvidia_gpu_to_llvm",
                     proton::gpu::createConvertProtonNvidiaGPUToLLVMPass);
  ADD_PASS_WRAPPER_1("add_convert_proton_amd_gpu_to_llvm",
                     proton::gpu::createConvertProtonAMDGPUToLLVMPass,
                     const std::string &);
  ADD_PASS_WRAPPER_0("add_allocate_proton_shared_memory",
                     proton::gpu::createAllocateProtonSharedMemoryPass);
  ADD_PASS_WRAPPER_0("add_schedule_buffer_store",
                     proton::gpu::createScheduleBufferStorePass);
  ADD_PASS_WRAPPER_0("add_sched_barriers",
                     proton::gpu::createAddSchedBarriersPass);
}
