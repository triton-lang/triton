#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
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

void init_triton_transforms(py::module &&m) {

  ADD_PASS_WRAPPER_0("add_sccp", mlir::createSCCPPass);
  ADD_PASS_WRAPPER_1("add_plan_cta_pass",
                     mlir::createTritonNvidiaGPUPlanCTAPass,
                     mlir::triton::nvidia_gpu::ClusterInfo *);
  ADD_PASS_WRAPPER_0("add_tritongpu_coalesce_pass",
                     mlir::createTritonGPUCoalescePass);
  ADD_PASS_WRAPPER_0("add_tritongpu_optimize_thread_locality_pass",
                     mlir::createTritonGPUOptimizeThreadLocalityPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce_pass", mlir::createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner_pass", mlir::createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer_pass", mlir::createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse_pass", mlir::createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm_pass", mlir::createLoopInvariantCodeMotionPass);
  ADD_PASS_WRAPPER_0("add_triton_combine_pass",
                     mlir::triton::createCombineOpsPass);
  ADD_PASS_WRAPPER_0("add_reorder_broadcast_pass",
                     mlir::triton::createReorderBroadcastPass);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer_pass",
                     mlir::triton::createRewriteTensorPointerPass);
  ADD_PASS_WRAPPER_1("add_tritongpu_ws_feasibility_checking_pass",
                     mlir::createTritonNvidiaGPUWSFeasibilityCheckingPass, int);
  ADD_PASS_WRAPPER_1("add_tritongpu_wsdecomposing_pass",
                     mlir::createTritonNvidiaGPUWSDecomposingPass, int);
  ADD_PASS_WRAPPER_1("add_tritongpu_wsmutex_pass",
                     mlir::createTritonNvidiaGPUWSMutexPass, int);
  ADD_PASS_WRAPPER_1("add_tritongpu_wsmaterialization_pass",
                     mlir::createTritonNvidiaGPUWSMaterializationPass, int);
  ADD_PASS_WRAPPER_0("add_tritongpu_ws_fixup_missing_attrs_pass",
                     mlir::createTritonNvidiaGPUWSFixupMissingAttrs);
  ADD_PASS_WRAPPER_4("add_convert_triton_to_tritongpu_pass",
                     mlir::triton::createConvertTritonToTritonGPUPass, int, int,
                     int, int);
  ADD_PASS_WRAPPER_4("add_tritongpu_pipeline_pass",
                     mlir::createTritonGPUPipelinePass, int, int, int, int);
  ADD_PASS_WRAPPER_2("add_tritongpu_materialize_load_store_pass",
                     mlir::createTritonNvidiaGPUMaterializeLoadStorePass, int,
                     int);
  ADD_PASS_WRAPPER_0("add_tritongpu_prefetch_pass",
                     mlir::createTritonGPUPrefetchPass);
  ADD_PASS_WRAPPER_1("add_tritongpu_accelerate_matmul_pass",
                     mlir::createTritonGPUAccelerateMatmulPass, int);
  ADD_PASS_WRAPPER_0("add_tritongpu_optimize_dot_operands_pass",
                     mlir::createTritonGPUOptimizeDotOperandsPass);
  ADD_PASS_WRAPPER_0("add_tritongpu_remove_layout_conversions_pass",
                     mlir::createTritonGPURemoveLayoutConversionsPass);
  ADD_PASS_WRAPPER_0("add_tritongpu_reorder_instructions_pass",
                     mlir::createTritonGPUReorderInstructionsPass);
  ADD_PASS_WRAPPER_1("add_tritongpu_rewrite_tensor_pointer_pass",
                     mlir::createTritonGPURewriteTensorPointerPass, int);
  ADD_PASS_WRAPPER_0("add_tritongpu_decompose_conversions_pass",
                     mlir::createTritonGPUDecomposeConversionsPass);
  ADD_PASS_WRAPPER_0("add_tritongpu_fence_insertion_pass",
                     mlir::createTritonNvidiaGPUFenceInsertionPass);
  ADD_PASS_WRAPPER_0("add_triton_gpu_to_llvm",
                     mlir::triton::createConvertTritonGPUToLLVMPass);
  ADD_PASS_WRAPPER_0("add_nv_gpu_to_llvm",
                     mlir::triton::createConvertNVGPUToLLVMPass);
  ADD_PASS_WRAPPER_3("add_tritongpu_wspipeline_pass",
                     mlir::createTritonNvidiaGPUWSPipelinePass, int, int, int);

  m.def("is_ws_supported", [](mlir::ModuleOp &mod) -> bool {
    return mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect::getWSSupportedAttr(
        mod);
  });
}
