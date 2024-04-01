#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include <mutex>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_amd_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(createConvertTritonAMDGPUToLLVMPass());
  });
  m.def("add_decompose_unsupported_conversions", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::AMD::createDecomposeUnsupportedConversionsPass());
  });
  ADD_PASS_WRAPPER_2("add_accelerate_matmul",
                     mlir::createTritonAMDGPUAccelerateMatmulPass,
                     const std::string, int);
  ADD_PASS_WRAPPER_0("add_decompose_conversions",
                     mlir::createTritonAMDGPUDecomposeConversionsPass);
  ADD_PASS_WRAPPER_0("add_optimize_epilogue",
                     mlir::createTritonAMDGPUOptimizeEpiloguePass);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     mlir::createTritonAMDGPURemoveLayoutConversionsPass);
  ADD_PASS_WRAPPER_0("add_reorder_instructions",
                     mlir::createTritonAMDGPUReorderInstructionsPass);
  ADD_PASS_WRAPPER_0("add_stream_pipeline",
                     mlir::createTritonAMDGPUStreamPipelinePass);
}

void init_triton_amd(py::module &&m) {
  m.doc() = "Python bindings to the AMD Triton backend";

  auto passes = m.def_submodule("passes");
  init_triton_amd_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    // registry.insert<mlir::ROCDL::ROCDLDialect>();
    mlir::registerROCDLDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // calling convention
  m.attr("CALLING_CONV_AMDGPU_KERNEL") =
      py::int_((unsigned)llvm::CallingConv::AMDGPU_KERNEL);

  m.def("set_code_object_version", [](llvm::Module *module, int version) {
    // Inject the control constant into the LLVM module so that device libraries
    // linked against module can resolve their references to it.
    llvm::Type *i32Ty = llvm::Type::getInt32Ty(module->getContext());
    llvm::GlobalVariable *abi = new llvm::GlobalVariable(
        *module, i32Ty, /*isConstant=*/true,
        llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
        llvm::ConstantInt::get(i32Ty, version), "__oclc_ABI_version", nullptr,
        llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
    abi->setVisibility(llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
    abi->setAlignment(llvm::MaybeAlign(4));
    abi->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);

    // Also attach the control attribute on the LLVM module. This is also needed
    // in addition to the above for various transformations to know what code
    // object version we are targeting at.
    module->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                          version);
  });
}
