#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Conversion/TritonCPUToLLVM/Passes.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>

namespace py = pybind11;

void init_triton_cpu_passes_ttcpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createConvertTritonCPUToLLVMPass());
  });
}

void init_triton_cpu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_cpu_passes_ttcpuir(passes.def_submodule("ttcpuir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::cpu::TritonCPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
