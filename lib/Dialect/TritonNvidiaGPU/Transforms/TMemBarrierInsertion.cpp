#include "triton/Analysis/MembarPlacement.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMEMBARRIERINSERTIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

struct TMemBarrierInsertionPass
    : public impl::TritonNvidiaGPUTMemBarrierInsertionPassBase<
          TMemBarrierInsertionPass> {
  using impl::TritonNvidiaGPUTMemBarrierInsertionPassBase<
      TMemBarrierInsertionPass>::TritonNvidiaGPUTMemBarrierInsertionPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!membar::usesTensorMemory(mod))
      return;
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<BufferRegionAnalysis>(
        BufferRegionAnalysis::Mode::TensorMemoryOnly);
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();
    ModuleAllocation allocation(mod);
    if (failed(membar::placeBarriers(allocation, *regions, {},
                                     membar::PlacementMode::TensorMemory)))
      signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
