#include "Analysis/ScopeIdAllocation.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
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
}
