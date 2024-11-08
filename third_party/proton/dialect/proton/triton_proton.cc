#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/raw_ostream.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void init_triton_proton(py::module &&m) {
  auto passes = m.def_submodule("passes");
  llvm::outs() << "init_triton_proton!\n";
}
