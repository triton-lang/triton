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

void init_triton_transforms(py::module &&m) {

  m.def("add_sccp_pass",
        [](mlir::PassManager &pm) { pm.addPass(mlir::createSCCPPass()); });
  m.def("add_plan_cta_pass",
        [](mlir::PassManager &pm,
           mlir::triton::nvidia_gpu::ClusterInfo &clusterInfo) {
          pm.addPass(mlir::createTritonNvidiaGPUPlanCTAPass(&clusterInfo));
        });
  m.def("add_tritongpu_coalesce_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonGPUCoalescePass());
  });
  m.def("add_tritongpu_optimize_thread_locality_pass",
        [](mlir::PassManager &pm) {
          pm.addPass(mlir::createTritonGPUOptimizeThreadLocalityPass());
        });
  m.def("add_symbol_dce_pass",
        [](mlir::PassManager &pm) { pm.addPass(mlir::createSymbolDCEPass()); });
  m.def("add_inliner_pass",
        [](mlir::PassManager &pm) { pm.addPass(mlir::createInlinerPass()); });
  m.def("add_canonicalizer_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createCanonicalizerPass());
  });
  m.def("add_cse_pass",
        [](mlir::PassManager &pm) { pm.addPass(mlir::createCSEPass()); });
  m.def("add_licm_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  });
  m.def("add_triton_combine_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createCombineOpsPass());
  });
  m.def("add_reorder_broadcast_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createReorderBroadcastPass());
  });
  m.def("add_rewrite_tensor_pointer_pass",
        [](mlir::PassManager &pm, int capability) {
          pm.addPass(mlir::triton::createRewriteTensorPointerPass(capability));
        });
  m.def("add_tritongpu_ws_feasibility_checking_pass",
        [](mlir::PassManager &pm, int computeCapability) {
          pm.addPass(mlir::createTritonNvidiaGPUWSFeasibilityCheckingPass(
              computeCapability));
        });
  m.def("add_tritongpu_wsdecomposing_pass", [](mlir::PassManager &pm,
                                               int computeCapability) {
    pm.addPass(mlir::createTritonNvidiaGPUWSDecomposingPass(computeCapability));
  });
  m.def("add_tritongpu_wspipeline_pass", [](mlir::PassManager &pm,
                                            int numStages, int numWarps,
                                            int computeCapability) {
    pm.addPass(mlir::createTritonNvidiaGPUWSPipelinePass(numStages, numWarps,
                                                         computeCapability));
  });
  m.def("add_tritongpu_wsmutex_pass",
        [](mlir::PassManager &pm, int computeCapability) {
          pm.addPass(mlir::createTritonNvidiaGPUWSMutexPass(computeCapability));
        });
  m.def("add_tritongpu_wsmaterialization_pass",
        [](mlir::PassManager &pm, int computeCapability) {
          pm.addPass(mlir::createTritonNvidiaGPUWSMaterializationPass(
              computeCapability));
        });
  m.def("add_tritongpu_ws_fixup_missing_attrs_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonNvidiaGPUWSFixupMissingAttrs());
  });
  m.def("add_convert_triton_to_tritongpu_pass",
        [](mlir::PassManager &pm, int numWarps, int threadsPerWarp, int numCTAs,
           int computeCapability) {
          pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
              numWarps, threadsPerWarp, numCTAs, computeCapability));
        });
  m.def("add_tritongpu_pipeline_pass", [](mlir::PassManager &pm, int numStages,
                                          int numWarps, int numCTAs,
                                          int computeCapability) {
    pm.addPass(mlir::createTritonGPUPipelinePass(numStages, numWarps, numCTAs,
                                                 computeCapability));
  });
  m.def("add_tritongpu_materialize_load_store_pass",
        [](mlir::PassManager &pm, int numWarps, int computeCapability) {
          pm.addPass(mlir::createTritonNvidiaGPUMaterializeLoadStorePass(
              numWarps, computeCapability));
        });
  m.def("add_tritongpu_prefetch_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonGPUPrefetchPass());
  });
  m.def("add_tritongpu_accelerate_matmul_pass", [](mlir::PassManager &pm,
                                                   int computeCapability) {
    pm.addPass(mlir::createTritonGPUAccelerateMatmulPass(computeCapability));
  });
  m.def("add_tritongpu_optimize_dot_operands_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  });
  m.def("add_tritongpu_remove_layout_conversions_pass",
        [](mlir::PassManager &pm) {
          pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
        });
  m.def("add_tritongpu_reorder_instructions_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonGPUReorderInstructionsPass());
  });
  m.def("add_tritongpu_rewrite_tensor_pointer_pass",
        [](mlir::PassManager &pm, int capability) {
          pm.addPass(mlir::createTritonGPURewriteTensorPointerPass(capability));
        });
  m.def("add_tritongpu_decompose_conversions_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonGPUDecomposeConversionsPass());
  });
  m.def("add_tritongpu_fence_insertion_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonNvidiaGPUFenceInsertionPass());
  });
  m.def("add_triton_gpu_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass());
  });
  m.def("add_nv_gpu_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createConvertNVGPUToLLVMPass());
  });

  m.def("is_ws_supported", [](mlir::ModuleOp &mod) -> bool {
    return mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect::getWSSupportedAttr(
        mod);
  });
}
