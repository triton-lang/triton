#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"
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
  if (isa<arith::ConstantOp>(op)) {
    return op;
  }
  auto mask = rewriter.create<MaskOp>(op->getLoc(), op->getResultTypes(), pred);
  rewriter.createBlock(&mask->getRegion(0));
  rewriter.setInsertionPointToStart(&mask->getRegion(0).front());
  auto newOp = rewriter.clone(*op);
  newOp->replaceAllUsesWith(mask->getResults());
  rewriter.create<MaskReturnOp>(op->getLoc(), newOp->getResults());
  rewriter.eraseOp(op);
  return mask;
}

static void resolveMaskOp(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp);

  // Canonicalize the IR to simplify the arithmetic ops defining the mask
  auto arithDialect =
      moduleOp.getContext()->getLoadedDialect<arith::ArithDialect>();
  RewritePatternSet patterns(moduleOp.getContext());
  arithDialect->getCanonicalizationPatterns(patterns);
  if (applyPatternsGreedily(moduleOp, std::move(patterns)).failed())
    return llvm::report_fatal_error("Failed to canonicalize the IR");

  SmallVector<MaskOp> maskOps;
  moduleOp->walk([&](MaskOp maskOp) { maskOps.push_back(maskOp); });
  for (auto maskOp : maskOps) {
    rewriter.setInsertionPoint(maskOp);
    while (&maskOp.getBody()->front() != maskOp.getBody()->getTerminator()) {
      Operation *op = &maskOp.getBody()->front();
      // Statically dead
      if (isConstantIntValue(maskOp.getPred(), 0)) {
        if (op->getNumResults() > 0) {
          SmallVector<Value> results;
          for (auto result : op->getResults()) {
            auto poisonOp =
                rewriter.create<ub::PoisonOp>(op->getLoc(), result.getType());
            results.push_back(poisonOp);
          }
          op->replaceAllUsesWith(results);
        }
        op->erase();
      } else {
        rewriter.moveOpBefore(op, maskOp);
        op = triton::predicateOp(rewriter, op, maskOp.getPred());
      }
    }
    maskOp->replaceAllUsesWith(
        maskOp.getBody()->getTerminator()->getOperands());
    maskOp->erase();
  }
}

struct LICMScalarsPattern : public RewritePattern {
  LICMScalarsPattern(MLIRContext *ctx)
      // MatchAnyOpTypeTag tells the pattern-matcher to try this pattern on
      // every operation it encounters.
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Keep only operations that belong to the arith dialect.
    if (op->getDialect()->getNamespace() !=
        arith::ArithDialect::getDialectNamespace())
      return failure();

    for (auto resultType : op->getResultTypes()) {
      if (isa<RankedTensorType>(resultType)) {
        return failure();
      }
    }

    scf::ForOp forOp = op->getParentOfType<scf::ForOp>();
    if (!forOp) {
      return failure();
    }

    for (auto operand : op->getOperands()) {
      if (!forOp.isDefinedOutsideOfLoop(operand)) {
        return failure();
      }
    }

    rewriter.moveOpBefore(op, forOp);
    return success();
  }
};

void resolvePredicateStageOps(
    ArrayRef<Operation *> ops, // TODO: rewriter not really needed
    std::function<Value(triton::gpu::PredicateStageOp)> fn) {
  SmallVector<triton::gpu::PredicateStageOp> opsToErase;
  for (auto op : ops) {
    auto predicateStageOp = dyn_cast<triton::gpu::PredicateStageOp>(op);
    if (!predicateStageOp) {
      continue;
    }
    Value result = fn(predicateStageOp);
    if (result) {
      predicateStageOp.getResult().replaceAllUsesWith(result);
      opsToErase.push_back(predicateStageOp);
    }
  }
  for (auto op : opsToErase) {
    op->erase();
  }
}

static void peelEpilogue(RewriterBase &rewriter, scf::ForOp forOp,
                         int maxStage) {
  mlir::triton::annotateLoopForEpiloguePeeling(rewriter, forOp, 1);

  rewriter.setInsertionPoint(forOp);
  Value vTrue = rewriter.create<mlir::arith::ConstantIntOp>(
      forOp.getLoc(), 1, rewriter.getI1Type());
  Value vFalse = rewriter.create<mlir::arith::ConstantIntOp>(
      forOp.getLoc(), 0, rewriter.getI1Type());
  SmallVector<Operation *> peeledOps;
  mlir::triton::peelLoopEpilogue(forOp, triton::predicateOp, &peeledOps);
  // Resolve all the peeled predicate stage ops to false
  resolvePredicateStageOps(
      peeledOps,
      [&](triton::gpu::PredicateStageOp op) -> Value { return vFalse; });
  // Resolve the maxStage-1 predicate in the loop body to true, and the rest
  // to the normal predicate
  SmallVector<Operation *> forOpBody =
      llvm::map_to_vector(forOp.getBody()->without_terminator(),
                          [](Operation &op) -> Operation * { return &op; });
  resolvePredicateStageOps(
      forOpBody, [&](triton::gpu::PredicateStageOp op) -> Value {
        if (op.getStage() == maxStage - 1) {
          return vTrue;
        }
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);
        return mlir::triton::emitPredicateForStage(rewriter, op.getIv(),
                                                   op.getUb(), op.getStep(),
                                                   maxStage, op.getStage());
      });
}

static void expandLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (scf::ForOp forOp : loops) {
    CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp))) {
      continue;
    }

    // Testing feature: allow for unresolved predicate stage ops
    // in the loop body.
    bool keepPredicateStage = forOp->hasAttr("__test_keep_predicate_stage");
    // TODO: Enable epilogue peeling for warp specialized loops
    bool customEpiloguePeeling =
        !forOp->getParentOfType<triton::gpu::WarpSpecializeOp>() &&
        !keepPredicateStage; // do not peel if we are testing the stage
                             // predication

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

    if (keepPredicateStage || customEpiloguePeeling) {
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

    if (failed(newForOp)) {
      continue;
    }
    forOp = *newForOp;
    if (customEpiloguePeeling) {
      peelEpilogue(rewriter, forOp, schedule.getNumStages() - 1);
    }
  }

  resolveMaskOp(moduleOp);
}

static void removeAttributes(ModuleOp moduleOp) {
  moduleOp->walk([&](Operation *op) {
    op->removeAttr(mlir::triton::kLoopStageAttrName);
    op->removeAttr(mlir::triton::kLoopClusterAttrName);
    op->removeAttr(mlir::triton::kScheduledMaxStageAttrName);
  });
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
