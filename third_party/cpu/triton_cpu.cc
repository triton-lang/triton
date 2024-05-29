#include "TritonCPUToLLVM/Passes.h"
#include "TritonToTritonCPU/Passes.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
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
  // m.def("add_to_llvmir", [](mlir::PassManager &pm) {
  //   pm.addPass(mlir::triton::createConvertTritonCPUToLLVMPass());
  // });
  m.def("add_triton_to_triton_cpu_pipeline", [](mlir::PassManager &pm) {
    mlir::triton::cpu::tritonToTritonCPUPipelineBuilder(pm);
  });
  m.def("add_triton_cpu_to_llvmir_pipeline", [](mlir::PassManager &pm) {
    mlir::triton::cpu::tritonCPUToLLVMPipelineBuilder(pm);
  });
  m.def("add_vector_to_scf", [](mlir::PassManager &pm, bool full_unroll,
                                unsigned target_rank, bool lower_tensors) {
    mlir::VectorTransferToSCFOptions opts;
    opts.setTargetRank(target_rank);
    opts.enableFullUnroll(full_unroll);
    opts.enableLowerTensors(lower_tensors);
    pm.addPass(mlir::createConvertVectorToSCFPass(opts));
  });
  m.def("add_lower_vector_multi_dim", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        mlir::triton::cpu::createLowerMultiReductionPass());
  });
  m.def("add_vector_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertVectorToLLVMPass());
  });
  m.def("add_lower_affine", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createLowerAffinePass());
  });
  m.def("add_memref_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  });
  m.def("add_math_to_libm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertMathToLibmPass());
  });
  m.def("add_func_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertFuncToLLVMPass());
  });
}

void init_triton_cpu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_cpu_passes_ttcpuir(passes.def_submodule("ttcpuir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::cpu::TritonCPUDialect,
                    mlir::vector::VectorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("find_kernel_names", [](mlir::ModuleOp &mod) {
    std::vector<std::string> res;
    mod.walk([&](mlir::FunctionOpInterface funcOp) {
      if (funcOp.getVisibility() == mlir::SymbolTable::Visibility::Public)
        res.push_back(funcOp.getName().str());
    });
    return res;
  });
}
