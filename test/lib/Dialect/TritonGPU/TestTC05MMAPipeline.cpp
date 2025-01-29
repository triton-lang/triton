#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUTC05MMAPIPELINE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct TC05MMAPipelinePass
    : public impl::TritonGPUTC05MMAPipelineBase<TC05MMAPipelinePass> {

  using impl::TritonGPUTC05MMAPipelineBase<
      TC05MMAPipelinePass>::TritonGPUTC05MMAPipelineBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    if (loops.empty())
      return;

    pipelineTC05MMALoops(m, loops, /*numStages=*/2, disableExpander);
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
