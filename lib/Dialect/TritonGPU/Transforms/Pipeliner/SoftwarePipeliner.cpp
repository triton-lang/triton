#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create async operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPIPELINE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static void tryAndPipelineOuterLoop(scf::ForOp forOp) {
  mlir::triton::PipeliningOption options;
  bool foundSchedule = false;
  // Limit 2 stages to not require extra shared memory.
  foundSchedule = getOuterLoopSchedule(forOp, /*numStage=*/2, options);
  if (!foundSchedule)
    return;
  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);
}

static scf::ForOp pipelineLoop(scf::ForOp forOp, int numStages) {
  mlir::triton::PipeliningOption options;

  bool foundSchedule = false;
  foundSchedule = preProcessLoopAndGetSchedule(forOp, numStages, options);

  // TODO: add more pipelines strategy.
  if (!foundSchedule)
    return nullptr;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);

  if (failed(newForOp))
    return nullptr;
  mlir::triton::asyncLaunchDots(newForOp.value());
  return newForOp.value();
}

struct PipelinePass : public impl::TritonGPUPipelineBase<PipelinePass> {

  using impl::TritonGPUPipelineBase<PipelinePass>::TritonGPUPipelineBase;

  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
      return numStages;
    return mlir::cast<IntegerAttr>(
               forOp->getAttr(mlir::triton::kNumStagesAttrName))
        .getInt();
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (getNumStagesOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    if (loops.empty())
      return;

    llvm::SmallSetVector<scf::ForOp, 8> outerLoops;
    llvm::SmallVector<scf::ForOp> pipelinedLoops;
    for (scf::ForOp forOp : loops) {
      auto outerLoop = dyn_cast<scf::ForOp>(forOp->getParentOp());
      int loopNumStages = getNumStagesOrDefault(forOp);
      scf::ForOp pipelinedFor = pipelineLoop(forOp, loopNumStages);
      if (pipelinedFor != nullptr)
        pipelinedLoops.push_back(pipelinedFor);
      if (pipelinedFor != nullptr && outerLoop &&
          getNumStagesOrDefault(outerLoop) > 1)
        outerLoops.insert(outerLoop);
    }

    // There is a hard dependency between load pipelining and the TC05MMA
    // pipelining. We can pipeline the TC05MMA only after the loads are
    // pipelined and buffers are allocated.
    mlir::triton::pipelineTC05MMALoops(getOperation(), pipelinedLoops, 2);

    // schedule the waits
    mlir::triton::updateWaits(getOperation());

    // Clean up arithmetic before applying the next level of pipelining to
    // simplify the IR.
    auto arithDialect =
        getOperation().getContext()->getLoadedDialect<arith::ArithDialect>();
    RewritePatternSet patterns(getOperation().getContext());
    arithDialect->getCanonicalizationPatterns(patterns);
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      return signalPassFailure();

    // Try to pipeline the outer loop to overlap the prologue and epilogue of
    // the inner loop.
    for (scf::ForOp outerLoop : outerLoops)
      tryAndPipelineOuterLoop(outerLoop);

    // Re-collect loop ops
    loops.clear();
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (getNumStagesOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      mlir::triton::pipelineTMAStores(forOp);
    }

    for (scf::ForOp forOp : loops) {
      mlir::triton::pipelineMMAWithScaledAcc(forOp);
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
