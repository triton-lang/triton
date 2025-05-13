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

static void pipelineWgmma(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
    mlir::triton::asyncLaunchDots(forOp);
  }
}

static Operation *wrapInMaskOp(RewriterBase &rewriter, Operation *op,
                               Value pred) {
  auto mask = rewriter.create<MaskOp>(op->getLoc(), op->getResultTypes(), pred);
  rewriter.createBlock(&mask->getRegion(0));
  rewriter.setInsertionPointToStart(&mask->getRegion(0).front());
  auto newOp = rewriter.clone(*op);
  newOp->replaceAllUsesWith(mask->getResults());
  rewriter.create<MaskReturnOp>(op->getLoc(), newOp->getResults());
  return mask;
}

static void resolveMaskOp(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp);
  SmallVector<MaskOp> opsToErase;
  moduleOp->walk([&](MaskOp maskOp) {
    rewriter.setInsertionPoint(maskOp);
    assert(maskOp.getBody()->getOperations().size() == 2 &&
           "MaskOp should have exactly one operation and a return statement");
    Operation *op = &maskOp.getBody()->getOperations().front();
    auto newOp = rewriter.clone(*op);
    newOp = triton::predicateOp(rewriter, newOp, maskOp.getPred());
    maskOp->replaceAllUsesWith(newOp);
    opsToErase.push_back(maskOp);
  });
  for (MaskOp op : opsToErase) {
    rewriter.eraseOp(op);
  }
}

static void expandLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (scf::ForOp forOp : loops) {
    CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp))) {
      continue;
    }

    std::vector<std::pair<Operation *, unsigned>> finalSchedule =
        schedule.createFinalSchedule(forOp);
    triton::PipeliningOption options;
    options.supportDynamicLoops = true;
    options.peelEpilogue = false;
    options.predicateFn = wrapInMaskOp;
    options.getScheduleFn =
        [&](scf::ForOp forOp,
            std::vector<std::pair<Operation *, unsigned>> &schedule) {
          schedule = finalSchedule;
        };
    // Testing feature: allow for unresolved predicate stage ops
    // in the loop body.
    if (forOp->hasAttr("__test_keep_predicate_stage")) {
      options.emitPredicateStageFn =
          [](RewriterBase &rewriter, Value inductionVar, Value upperBound,
             Value step, uint64_t maxStage, uint64_t stage) {
            return rewriter.create<triton::gpu::PredicateStageOp>(
                inductionVar.getLoc(), inductionVar, upperBound, step, maxStage,
                stage);
          };
    }
    IRRewriter rewriter(forOp);
    FailureOr<scf::ForOp> newForOp =
        triton::pipelineForLoop(rewriter, forOp, options);
  }
}

static void removeAttributes(ModuleOp moduleOp) {
  moduleOp->walk([&](Operation *op) {
    op->removeAttr(mlir::triton::kLoopStageAttrName);
    op->removeAttr(mlir::triton::kLoopClusterAttrName);
    op->removeAttr(mlir::triton::kScheduledMaxStageAttrName);
  });
}

// This function peels the last 'numIterations' iterations from the given
// scf::ForOp
static void peelLastIterations(scf::ForOp forOp, unsigned numIterations) {
  IRRewriter rewriter(forOp);
  auto loc = forOp.getLoc();

  // Fetch loop bounds and step
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  // Compute new upper bound for the main loop: newUpperBound = upperBound -
  // (numIterations * step)
  auto numConst = rewriter.create<arith::ConstantIndexOp>(loc, numIterations);
  Value peelOffset = rewriter.create<arith::MulIOp>(loc, numConst, step);
  Value newUpperBound =
      rewriter.create<arith::SubIOp>(loc, upperBound, peelOffset);

  forOp.getUpperBoundMutable().assign(newUpperBound);

  // Create the main loop running from lowerBound to newUpperBound
  auto mainFor = rewriter.create<scf::ForOp>(loc, lowerBound, newUpperBound,
                                             step, forOp.getIterOperands());

  // Remap the original loop body into the main loop
  Block &origBlock = forOp.getBody()->front();
  Block &mainBlock = mainFor.getBody()->front();
  llvm::DenseMap<Value, Value> mapping;
  for (unsigned i = 0, e = origBlock.getNumArguments(); i < e; ++i) {
    mapping[origBlock.getArgument(i)] = mainBlock.getArgument(i);
  }
  for (Operation &op : origBlock.getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    rewriter.clone(op, mapping);
  }

  // Create the peeled loop for the last iterations: from newUpperBound to
  // original upperBound
  auto peeledFor = rewriter.create<scf::ForOp>(loc, newUpperBound, upperBound,
                                               step, forOp.getIterOperands());
  Block &peeledBlock = peeledFor.getBody()->front();
  mapping.clear();
  for (unsigned i = 0, e = origBlock.getNumArguments(); i < e; ++i) {
    mapping[origBlock.getArgument(i)] = peeledBlock.getArgument(i);
  }
  for (Operation &op : origBlock.getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    rewriter.clone(op, mapping);
  }

  // Replace the original loop with the peeled loop's results (adjust as needed
  // if loop results must be merged)
  rewriter.replaceOp(forOp, peeledFor.getResults());
}

struct PipelinePass : public impl::TritonGPUPipelineBase<PipelinePass> {

  using impl::TritonGPUPipelineBase<PipelinePass>::TritonGPUPipelineBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Transform the loop by introducing async operations to prepare it for
    // pipeline expansion.
    lowerLoops(moduleOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// SoftwarePipeliner internal IR Dump After: LowerLoops\n"
          << moduleOp << "\n\n\n";
    }

    // Apply the pipeline expansion.
    expandLoops(moduleOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// SoftwarePipeliner internal IR Dump After: "
                      "ExpandLoops\n"
                   << moduleOp << "\n\n\n";
    }

    resolveMaskOp(moduleOp);

    // Cleanup the IR from the pipeline attributes.
    removeAttributes(moduleOp);

    pipelineWgmma(moduleOp);

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

    {
      SmallVector<scf::ForOp> loops;
      getOperation()->walk([&](scf::ForOp forOp) {
        // Bail out for loops with num_stage <= 1.
        if (getNumStagesOrDefault(forOp, numStages) > 1)
          loops.push_back(forOp);
      });

      for (scf::ForOp forOp : loops) {
        mlir::triton::pipelineTMAStores(forOp);
      }
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
