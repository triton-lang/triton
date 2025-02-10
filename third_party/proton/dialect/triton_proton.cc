#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "../third_party/proton/dialect/include/ProtonGPUToLLVM/Passes.h"

namespace py = pybind11;

namespace {

void init_triton_proton_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_proton_lowering_pass",
        [](mlir::PassManager &pm) { pm.addPass(createProtonLoweringPass()); });
}
} // namespace

void init_triton_proton(py::module &&m) {
  m.doc() = "Python bindings to the Proton backend";
  auto passes = m.def_submodule("passes");
  init_triton_proton_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::proton::ProtonDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
