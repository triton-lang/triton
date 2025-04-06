#include "Analysis/ScopeIdAllocation.h"
#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "Conversion/ProtonToProtonGPU/Passes.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void init_triton_proton(py::module &&m) {
  m.doc() = "Python bindings to the Proton backend";

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::proton::ProtonDialect>();
    registry.insert<mlir::triton::proton::gpu::ProtonGPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("get_scope_id_pairs", [](mlir::ModuleOp &module) {
    auto moduleScopeIdAllocation =
        mlir::triton::proton::ModuleScopeIdAllocation(module);
    return moduleScopeIdAllocation.getScopeIdPairs();
  });

  m.def("create_proton_record",
        [](mlir::OpBuilder &opBuilder, bool isStart,
           const std::string &name) -> void {
          auto nameAttr = mlir::StringAttr::get(opBuilder.getContext(),
                                                llvm::StringRef(name));
          auto loc = opBuilder.getUnknownLoc();
          opBuilder.create<mlir::triton::proton::RecordOp>(loc, isStart,
                                                           nameAttr);
        });

  ADD_PASS_WRAPPER_0("add_convert_proton_to_protongpu",
                     mlir::triton::proton::createConvertProtonToProtonGPUPass);
  ADD_PASS_WRAPPER_0(
      "add_convert_proton_nvidia_gpu_to_llvm",
      mlir::triton::proton::gpu::createConvertProtonNvidiaGPUToLLVMPass);
  ADD_PASS_WRAPPER_0(
      "add_convert_proton_amd_gpu_to_llvm",
      mlir::triton::proton::gpu::createConvertProtonAMDGPUToLLVMPass);
  ADD_PASS_WRAPPER_0(
      "add_allocate_proton_shared_memory",
      mlir::triton::proton::gpu::createAllocateProtonSharedMemoryPass);
  ADD_PASS_WRAPPER_0(
      "add_allocate_proton_global_scratch_buffer",
      mlir::triton::proton::gpu::createAllocateProtonGlobalScratchBufferPass);
}
