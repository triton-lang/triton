#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

namespace {
struct TritonAMDGPUMembarAnalysis
    : public mlir::TritonAMDGPUMembarAnalysisBase<TritonAMDGPUMembarAnalysis> {

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Allocate shared memory and set barrier
    mlir::ModuleAllocation allocation(mod);
    mlir::ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();
  }
};
} // namespace

namespace mlir {
std::unique_ptr<Pass> createTritonAMDGPUMembarAnalysisPass() {
  return std::make_unique<TritonAMDGPUMembarAnalysis>();
}
} // namespace mlir
