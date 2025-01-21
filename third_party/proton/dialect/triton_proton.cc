#include "Dialect/Proton/IR/Dialect.h"
#include "Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void init_triton_proton_passes(py::module &&m) {
  ADD_PASS_WRAPPER_3("add_proton_lowering",
                     mlir::triton::proton::createProtonLowering, int32_t,
                     int32_t, int32_t);
}

void init_triton_proton(py::module &&m) {
  m.doc() = "Python bindings to the Proton Triton backend";

  init_triton_proton_passes(m.def_submodule("passes"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::proton::ProtonDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("get_group_size",
        []() { return mlir::triton::proton::getGroupSize(); });

  m.def("get_entry_byte_size",
        []() { return mlir::triton::proton::getBytesPerEntry(); });
}
