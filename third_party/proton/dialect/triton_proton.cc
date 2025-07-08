#include "Analysis/ScopeIdAllocation.h"
#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "Conversion/ProtonToProtonGPU/Passes.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "Dialect/ProtonGPU/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace mlir::triton;

void init_triton_proton(py::module &&m) {
  m.doc() = "Python bindings to the Proton backend";

  // Proton enums
  py::enum_<proton::MetricType>(m, "METRIC_TYPE", py::module_local())
      .value("CYCLE", proton::MetricType::CYCLE)
      .export_values();

  py::enum_<proton::SamplingStrategy>(m, "SAMPLING_STRATEGY",
                                      py::module_local())
      .value("NONE", proton::SamplingStrategy::NONE)
      .value("SELECTIVE", proton::SamplingStrategy::SELECTIVE)
      .export_values();

  // ProtonGPU enums
  py::enum_<proton::gpu::Granularity>(m, "GRANULARITY", py::module_local())
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

  py::enum_<proton::gpu::BufferStrategy>(m, "BUFFER_STRATEGY",
                                         py::module_local())
      .value("CIRCULAR", proton::gpu::BufferStrategy::CIRCULAR)
      .value("FLUSH", proton::gpu::BufferStrategy::FLUSH)
      .export_values();

  py::enum_<proton::gpu::BufferType>(m, "BUFFER_TYPE", py::module_local())
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

  // Proton operations
  m.def("create_proton_record",
        [](mlir::OpBuilder &opBuilder, bool isStart,
           const std::string &name) -> void {
          auto nameAttr = mlir::StringAttr::get(opBuilder.getContext(),
                                                llvm::StringRef(name));
          auto loc = opBuilder.getUnknownLoc();
          opBuilder.create<proton::RecordOp>(loc, isStart, nameAttr);
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
  ADD_PASS_WRAPPER_0("add_allocate_proton_global_scratch_buffer",
                     proton::gpu::createAllocateProtonGlobalScratchBufferPass);
  ADD_PASS_WRAPPER_0("add_schedule_buffer_store",
                     proton::gpu::createScheduleBufferStorePass);
  ADD_PASS_WRAPPER_0("add_sched_barriers",
                     proton::gpu::createAddSchedBarriersPass);
}
