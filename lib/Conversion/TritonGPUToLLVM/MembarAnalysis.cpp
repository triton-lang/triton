#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

#define GEN_PASS_DEF_TRITONGPUMEMBARANALYSIS
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

namespace {
struct TritonGPUMembarAnalysis
    : public ::impl::TritonGPUMembarAnalysisBase<TritonGPUMembarAnalysis> {

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Allocate shared memory and set barrier
    mlir::ModuleAllocation allocation(mod);
    mlir::ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();
  }
};
} // namespace

namespace mlir::triton::gpu {
std::unique_ptr<Pass> createTritonGPUMembarAnalysis() {
  return std::make_unique<TritonGPUMembarAnalysis>();
}
} // namespace mlir::triton::gpu
