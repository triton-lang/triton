#include "Tools/Sys/GetPlatform.hpp"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "passes.h"
#include "llvm/IR/Constants.h"
#include <mutex>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_amd_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(createConvertTritonAMDGPUToLLVMPass());
  });
  m.def("add_decompose_unsupported_conversions", [](mlir::PassManager &pm) {
    pm.addPass(
        mlir::triton::gpu::createDecomposeUnsupportedAMDConversionsPass());
  });
  ADD_PASS_WRAPPER_2("add_accelerate_matmul",
                     mlir::createTritonAMDGPUAccelerateMatmulPass, int, int);
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

  using ret = py::return_value_policy;

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

  m.def(
      "get_arch_info",
      []() {
          return std::get<0>(getArchInfo());
      },
      ret::take_ownership);

  m.def(
      "get_warp_size", []() { return std::get<1>(getArchInfo()); },
      ret::take_ownership);

  // calling convention
  m.attr("CALLING_CONV_AMDGPU_KERNEL") =
      py::int_((unsigned)llvm::CallingConv::AMDGPU_KERNEL);
}
