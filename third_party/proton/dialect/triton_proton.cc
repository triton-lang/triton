#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <nanobind/nanobind.h>
#include "nanobind_stl.h"

namespace py = nanobind;

void init_triton_proton(py::module &&m) {
  auto passes = m.def_submodule("passes");

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::proton::ProtonDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
