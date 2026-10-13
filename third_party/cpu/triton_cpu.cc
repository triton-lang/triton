#include "ScalarizePass/ScalarizeInterfaceImpl.h"
#include "TritonCPUToLLVM/Passes.h"
#include "TritonCPUTransforms/Passes.h"
#include "TritonToTritonCPU/Passes.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "python/src/ir.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Constants.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#if defined(__x86_64__) || defined(__i386__)
#include <asm/prctl.h>
#endif
#include <sys/syscall.h>
#include <unistd.h>

#ifdef ONEDNN_AVAILABLE
#include "oneapi/dnnl/dnnl_config.h"
#endif
bool is_onednn_available() {
#ifdef DNNL_EXPERIMENTAL_UKERNEL
  return true;
#else
  return false;
#endif
}

bool is_xsmm_available() {
#ifdef XSMM_AVAILABLE
  return true;
#else
  return false;
#endif
}

namespace py = nanobind;

void init_triton_cpu_llvm(py::module_ &m);

namespace {

template <typename OpTy>
mlir::Value createUnaryMathOp(TritonOpBuilder &builder, mlir::Value &value) {
  return builder.create<OpTy>(value);
}

void init_triton_cpu_ir(py::module_ &m) {
  m.def("create_acos", &createUnaryMathOp<mlir::math::AcosOp>);
  m.def("create_acosh", &createUnaryMathOp<mlir::math::AcoshOp>);
  m.def("create_asin", &createUnaryMathOp<mlir::math::AsinOp>);
  m.def("create_asinh", &createUnaryMathOp<mlir::math::AsinhOp>);
  m.def("create_atan", &createUnaryMathOp<mlir::math::AtanOp>);
  m.def("create_atanh", &createUnaryMathOp<mlir::math::AtanhOp>);
  m.def("create_cbrt", &createUnaryMathOp<mlir::math::CbrtOp>);
  m.def("create_cosh", &createUnaryMathOp<mlir::math::CoshOp>);
  m.def("create_expm1", &createUnaryMathOp<mlir::math::ExpM1Op>);
  m.def("create_log1p", &createUnaryMathOp<mlir::math::Log1pOp>);
  m.def("create_log10", &createUnaryMathOp<mlir::math::Log10Op>);
  m.def("create_sinh", &createUnaryMathOp<mlir::math::SinhOp>);
  m.def("create_tan", &createUnaryMathOp<mlir::math::TanOp>);
  m.def("create_tanh", &createUnaryMathOp<mlir::math::TanhOp>);
  m.def("create_trunc", &createUnaryMathOp<mlir::math::TruncOp>);
}

} // namespace

void init_triton_cpu_passes_ttcpuir(py::module_ &m) {
  using namespace mlir::triton;

  py::enum_<cpu::VecLib>(m, "VecLib")
      .value("libsleef", cpu::VecLib::Sleef)
      .value("libmvec", cpu::VecLib::Mvec);

  py::enum_<cpu::Ukernels>(m, "Ukernels")
      .value("OneDNN", cpu::Ukernels::OneDNN)
      .value("XSMM", cpu::Ukernels::XSMM);

  m.def("add_scalarize", [](mlir::PassManager &pm, bool skip_gather_scatter) {
    pm.addPass(
        mlir::triton::cpu::createScalarizeUsingForOpPass(skip_gather_scatter));
  });
  m.def("add_convert_memory_ops", [](mlir::PassManager &pm,
                                     bool use_gather_scatter,
                                     bool assume_in_bounds) {
    pm.addPass(mlir::triton::cpu::createConvertMemoryOps(use_gather_scatter,
                                                         assume_in_bounds));
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
  m.def("add_convert_reduction_op",
        [](mlir::PassManager &pm, bool use_reduction_op,
           bool use_multidim_reduction_op) {
          pm.addPass(mlir::triton::cpu::createConvertReductionOp(
              use_reduction_op, use_multidim_reduction_op));
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
  m.def("add_triton_cpu_canonicalizer", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createCanonicalize());
  });
  m.def("add_optimize_masks", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createOptimizeMasks());
  });
  m.def("add_convert_dot_product", [](mlir::PassManager &pm,
                                      bool useHorizontalSum) {
    pm.addPass(mlir::triton::cpu::createConvertDotProduct(useHorizontalSum));
  });
  m.def("add_loop_invariant_code_motion", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  });
  m.def("add_convert_dot_to_ukernels", [](mlir::PassManager &pm,
                                          cpu::Ukernels ukernels) {
    pm.addPass(mlir::triton::cpu::createConvertDotOpToUkernelOps(ukernels));
  });
  m.def("add_convert_dot_to_amx", [](mlir::PassManager &pm, bool convertInt8,
                                     bool convertFp16, bool convertBf16) {
    pm.addPass(mlir::triton::cpu::createConvertDotToAMX(
        convertInt8, convertFp16, convertBf16));
  });
  m.def("add_convert_dot_to_fma", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertDotToFMA());
  });
  m.def("add_convert_dot_generic", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertDotGeneric());
  });
  m.def("add_convert_vector_contract_to_arm_neon_bfmmla",
        [](mlir::PassManager &pm) {
          pm.addPass(
              mlir::triton::cpu::createConvertVectorContractToArmNeonBFMMLA());
        });
  m.def("add_convert_dot_to_nanokernel", [](mlir::PassManager &pm,
                                            std::string cpuFeatures) {
    pm.addPass(mlir::triton::cpu::createConvertDotToNanokernel(cpuFeatures));
  });
  m.def("add_unroll_and_reorder_elementwise_ops",
        [](mlir::PassManager &pm, std::string cpuFeatures) {
          pm.addPass(mlir::triton::cpu::createUnrollAndReorderElementwiseOps(
              cpuFeatures));
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
  m.def("add_ukernels_to_onednn_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createUkernelOpsToOneDNNLLVMPass());
  });
  m.def("add_ukernels_to_xsmm_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createUkernelOpsToXSMMLLVMPass());
  });
  m.def("add_expand_strided_metadata", [](mlir::PassManager &pm) {
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  });
  m.def("add_vector_to_llvmir",
        [](mlir::PassManager &pm, bool reassoc_fp_reduction) {
          mlir::ConvertVectorToLLVMPassOptions opts;
          opts.reassociateFPReductions = reassoc_fp_reduction;
          // opts.force32BitVectorIndices = true;
          // Preserve emitted Neon intrinsics; keep upstream armBF16 lowering
          // off.
          opts.armNeon = true;
          // opts.armSVE = false;
          opts.x86 = true;
          // opts.vectorTransformsOptions();
          // TODO: Check whether we need these parameters.
          // Somehow it helps arm.
          //
          // VectorContractLowering::Dot is default and fine, but it takes too
          // long to compile on arm. I guess it generated too many ir ops.
          // (!WA!)
          //
          // VectorContractLowering::Matmul generates error: "Do not know
          // how to split the result of this operator!" with
          // "llvm.matrix.multiply". On those ops no progress for some time, so
          // it can be replaced with `vector.matmul`.
          //
          // VectorContractLowering::OuterProduct somehow
          // works, but it might not be the most performant way. It's most
          // widely used path for this lowering in CPU case.
          opts.vectorContractLowering =
              mlir::vector::VectorContractLowering::OuterProduct;
          pm.addPass(mlir::createConvertVectorToLLVMPass(opts));
        });
  m.def("add_lower_affine", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createLowerAffinePass());
  });
  m.def("add_memref_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  });
  m.def("add_math_to_vec_lib", [](mlir::PassManager &pm, cpu::VecLib lib,
                                  std::set<std::string> cpu_features) {
    pm.addPass(mlir::triton::cpu::createMathToVecLibPass(lib, cpu_features));
  });
  m.def("add_math_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertMathToLLVMPass());
  });
  m.def("add_math_to_libm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertMathToLibmPass());
  });
  m.def("add_func_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertFuncToLLVMPass());
  });
  m.def("add_ub_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createUBToLLVMConversionPass());
  });
}

void init_triton_cpu(py::module_ &m) {
  auto ir = m.def_submodule("ir");
  init_triton_cpu_ir(ir);

  auto llvm = m.def_submodule("llvm");
  init_triton_cpu_llvm(llvm);

  auto passes = m.def_submodule("passes");
  auto ttcpuir = passes.def_submodule("ttcpuir");
  init_triton_cpu_passes_ttcpuir(ttcpuir);

  m.def("enable_amx", []() -> bool {
#if defined(__linux__) && defined(ARCH_REQ_XCOMP_PERM)
    // AMX usage requires extended XSTATE which is disabled by default. We
    // need to request access to AMX so that XSTATE was dynamically extended
    // on the first AMX usage instead of issuing SIGILL.
    // See https://www.kernel.org/doc/Documentation/x86/xstate.rst for more
    // details.
    constexpr int XFEATURE_XTILEDATA = 18;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
      return false;
    return true;
#else
    return false;
#endif // __linux__ && ARCH_REQ_XCOMP_PERM
  });

  m.def("onednn_available", is_onednn_available);

  m.def("xsmm_available", is_xsmm_available);

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::cpu::TritonCPUDialect,
                    mlir::vector::VectorDialect>();
    mlir::registerArmNeonDialectTranslation(registry);
    mlir::triton::cpu::registerTritonOpScalarizeExternalModels(registry);
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
