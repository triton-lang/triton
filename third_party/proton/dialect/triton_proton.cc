#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;


void init_triton_proton(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;	
  auto passes = m.def_submodule("passes");

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::proton::ProtonDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });


}

