#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton/Conversion/NVGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_nvidia_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton::gpu;
  ADD_PASS_WRAPPER_1("add_rewrite_tensor_pointer",
                     mlir::createTritonGPURewriteTensorPointerPass, int);
  ADD_PASS_WRAPPER_0("add_triton_gpu_to_llvm",
                     mlir::triton::createConvertTritonGPUToLLVMPass);
}

void init_triton_nvidia_passes_ttnvgpuir(py::module &&m) {
  ADD_PASS_WRAPPER_1("add_plan_cta", mlir::createTritonNvidiaGPUPlanCTAPass,
                     mlir::triton::nvidia_gpu::ClusterInfo *);
  ADD_PASS_WRAPPER_1("add_wsfeasibility_checking",
                     mlir::createTritonNvidiaGPUWSFeasibilityCheckingPass, int);
  ADD_PASS_WRAPPER_1("add_wsdecomposing",
                     mlir::createTritonNvidiaGPUWSDecomposingPass, int);
  ADD_PASS_WRAPPER_1("add_wsmutex", mlir::createTritonNvidiaGPUWSMutexPass,
                     int);
  ADD_PASS_WRAPPER_1("add_wsmaterialization",
                     mlir::createTritonNvidiaGPUWSMaterializationPass, int);
  ADD_PASS_WRAPPER_0("add_wsfixup_missing_attrs",
                     mlir::createTritonNvidiaGPUWSFixupMissingAttrs);
  ADD_PASS_WRAPPER_2("add_materialize_load_store",
                     mlir::createTritonNvidiaGPUMaterializeLoadStorePass, int,
                     int);
  ADD_PASS_WRAPPER_0("add_fence_insertion",
                     mlir::createTritonNvidiaGPUFenceInsertionPass);
  ADD_PASS_WRAPPER_0("add_nvgpu_to_llvm",
                     mlir::triton::createConvertNVGPUToLLVMPass);
  ADD_PASS_WRAPPER_3("add_wspipeline",
                     mlir::createTritonNvidiaGPUWSPipelinePass, int, int, int);

  m.def("is_ws_supported", [](mlir::ModuleOp &mod) -> bool {
    return mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect::getWSSupportedAttr(
        mod);
  });
}

void init_triton_nvidia(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_nvidia_passes_ttgpuir(passes.def_submodule("ttgpuir"));
  init_triton_nvidia_passes_ttnvgpuir(passes.def_submodule("ttnvgpuir"));
}
