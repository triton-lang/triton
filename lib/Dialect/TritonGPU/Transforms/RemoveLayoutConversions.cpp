#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/Transforms/LayoutAssignment.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUREMOVELAYOUTCONVERSIONS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPURemoveLayoutConversionsPass
    : public impl::TritonGPURemoveLayoutConversionsBase<
          TritonGPURemoveLayoutConversionsPass> {
public:
  using impl::TritonGPURemoveLayoutConversionsBase<
      TritonGPURemoveLayoutConversionsPass>::
      TritonGPURemoveLayoutConversionsBase;

  void runOnOperation() override {
    if (failed(optimizeDistributedLayouts(getOperation(), disableRematSplitting,
                                          LayoutAssignmentStrategy::Legacy)))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
