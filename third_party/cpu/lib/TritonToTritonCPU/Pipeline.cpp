#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace triton {
namespace cpu {

void tritonToTritonCPUPipelineBuilder(OpPassManager &pm) {
  pm.addPass(mlir::triton::cpu::createConvertMemoryOps());
  pm.addPass(mlir::triton::cpu::createConvertPtrOps());
  pm.addPass(mlir::triton::cpu::createConvertElementwiseOps());
  pm.addPass(mlir::triton::cpu::createConvertDotOp());
  pm.addPass(mlir::triton::cpu::createConvertReductionOp());
  pm.addPass(mlir::triton::cpu::createConvertControlFlowOps());
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

void registerTritonToTritonCPUPipeline() {
  PassPipelineRegistration<>("triton-to-triton-cpu",
                             "Triton to TritonCPU conversion pipeline.",
                             tritonToTritonCPUPipelineBuilder);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
