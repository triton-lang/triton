#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Conversion/NVGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define ADD_PASS_WRAPPER_0(name, builder)                                      \
  m.def(name, [](mlir::PassManager &pm) { pm.addPass(builder()); })

#define ADD_PASS_WRAPPER_1(name, builder, ty0)                                 \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder(val0)); })

#define ADD_PASS_WRAPPER_2(name, builder, ty0, ty1)                            \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1) {                  \
    pm.addPass(builder(val0, val1));                                           \
  })

#define ADD_PASS_WRAPPER_3(name, builder, ty0, ty1, ty2)                       \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2) {        \
    pm.addPass(builder(val0, val1, val2));                                     \
  })

#define ADD_PASS_WRAPPER_4(name, builder, ty0, ty1, ty2, ty3)                  \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3) { pm.addPass(builder(val0, val1, val2, val3)); })

void init_triton_passes_common(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
}

void init_triton_passes_ttir(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_0("add_combine", createCombineOpsPass);
  ADD_PASS_WRAPPER_0("add_reorder_broadcast", createReorderBroadcastPass);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     createRewriteTensorPointerPass);
  ADD_PASS_WRAPPER_4("add_convert_to_ttgpuir",
                     createConvertTritonToTritonGPUPass, int, int, int, int);
}

void init_triton_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton::gpu;
  ADD_PASS_WRAPPER_0("add_coalesce", createCoalescePass);
  ADD_PASS_WRAPPER_0("add_optimize_thread_locality",
                     createOptimizeThreadLocalityPass);
  ADD_PASS_WRAPPER_4("add_pipeline", createPipelinePass, int, int, int, int);
  ADD_PASS_WRAPPER_0("add_prefetch", createPrefetchPass);
  ADD_PASS_WRAPPER_1("add_accelerate_matmul", createAccelerateMatmulPass, int);
  ADD_PASS_WRAPPER_0("add_reorder_instructions", createReorderInstructionsPass);
  ADD_PASS_WRAPPER_0("add_optimize_dot_operands",
                     createOptimizeDotOperandsPass);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     createRemoveLayoutConversionsPass);
  ADD_PASS_WRAPPER_0("add_decompose_conversions",
                     createDecomposeConversionsPass);
  ADD_PASS_WRAPPER_0("add_triton_gpu_to_llvm",
                     mlir::triton::createConvertTritonGPUToLLVMPass);
}

void init_triton_passes_ttnvgpuir(py::module &&m) {

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
  ADD_PASS_WRAPPER_1("add_rewrite_tensor_pointer",
                     mlir::createTritonGPURewriteTensorPointerPass, int);

  m.def("is_ws_supported", [](mlir::ModuleOp &mod) -> bool {
    return mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect::getWSSupportedAttr(
        mod);
  });
}

void init_triton_passes(py::module &&m) {
  init_triton_passes_common(m.def_submodule("common"));
  init_triton_passes_ttir(m.def_submodule("ttir"));
  init_triton_passes_ttgpuir(m.def_submodule("ttgpuir"));
  init_triton_passes_ttnvgpuir(m.def_submodule("ttnvgpuir"));
}
