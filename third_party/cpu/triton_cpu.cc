#include "TritonCPUToLLVM/Passes.h"
#include "TritonCPUTransforms/Passes.h"
#include "TritonToTritonCPU/Passes.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>

namespace py = pybind11;

void init_triton_cpu_passes_ttcpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_convert_memory_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertMemoryOps());
  });
  m.def("add_convert_ptr_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertPtrOps());
  });
  m.def("add_convert_elementwise_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertElementwiseOps());
  });
  m.def("add_convert_elem_manip_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertElemManipOps());
  });
  m.def("add_convert_dot_op", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertDotOp());
  });
  m.def("add_convert_histogram_op", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertHistogramOp());
  });
  m.def("add_convert_reduction_op", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertReductionOp());
  });
  m.def("add_convert_scan_op", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertScanOp());
  });
  m.def("add_convert_cf_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertControlFlowOps());
  });
  m.def("add_convert_atomic_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertAtomicOps());
  });
  m.def("add_convert_debug_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertDebugOps());
  });
  m.def("add_optimize_masks", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createOptimizeMasks());
  });
  m.def("add_convert_dot_product", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertDotProduct());
  });
  m.def("add_convert_unsupported_ops",
        [](mlir::PassManager &pm, bool promote_bf16_to_fp32,
           bool convert_mixed_precision_matmul, bool promote_lib_math_to_fp32) {
          pm.addPass(mlir::triton::cpu::createConvertUnsupportedOps(
              promote_bf16_to_fp32, convert_mixed_precision_matmul,
              promote_lib_math_to_fp32));
        });
  m.def("add_decompose_fp_conversions",
        [](mlir::PassManager &pm, bool decomposeBf16Conversions,
           bool decomposeFp8Conversions) {
          pm.addPass(mlir::triton::cpu::createDecomposeFpConversions(
              decomposeBf16Conversions, decomposeFp8Conversions));
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
  m.def("add_func_op_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createFuncOpToLLVMPass());
  });
  m.def("add_program_id_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createGetProgramIdOpToLLVMPass());
  });
  m.def("add_memory_op_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createMemoryOpToLLVMPass());
  });
  m.def("add_atomic_ops_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createAtomicOpsToLLVMPass());
  });
  m.def("add_debug_ops_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createDebugOpsToLLVMPass());
  });
  m.def("add_vector_to_llvmir",
        [](mlir::PassManager &pm, bool reassoc_fp_reduction) {
          mlir::ConvertVectorToLLVMPassOptions opts;
          opts.reassociateFPReductions = reassoc_fp_reduction;
          // opts.force32BitVectorIndices = true;
          // opts.amx = false;
          // opts.armNeon = false;
          // opts.armSVE = false;
          // opts.x86Vector = false;
          pm.addPass(mlir::createConvertVectorToLLVMPass(opts));
        });
  m.def("add_lower_affine", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createLowerAffinePass());
  });
  m.def("add_memref_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  });
  m.def("add_math_to_libmvec", [](mlir::PassManager &pm, bool use_sleef) {
    pm.addPass(mlir::triton::cpu::createMathToLibmvecPass(use_sleef));
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
      // Kernel functions are public and have a body.
      if (!funcOp.getFunctionBody().empty() &&
          funcOp.getVisibility() == mlir::SymbolTable::Visibility::Public)
        res.push_back(funcOp.getName().str());
    });
    return res;
  });
}
