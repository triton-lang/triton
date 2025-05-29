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
  auto mask = rewriter.create<MaskOp>(op->getLoc(), op->getResultTypes(), pred);
  rewriter.createBlock(&mask->getRegion(0));
  rewriter.setInsertionPointToStart(&mask->getRegion(0).front());
  auto newOp = rewriter.clone(*op);
  rewriter.create<MaskReturnOp>(op->getLoc(), newOp->getResults());
  op->replaceAllUsesWith(mask->getResults());
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

static Operation *processPeeledEpilogueOp(RewriterBase &rewriter, Operation *op,
                                          bool isEpilogue) {
  if (auto predOp = dyn_cast<triton::gpu::PredicateStageOp>(op)) {
    if (isEpilogue) {
      // Return false for the predicate of the peeled iteration
      return rewriter.create<mlir::arith::ConstantIntOp>(
          predOp.getLoc(), 0, predOp.getResult().getType());
    } else {
      if (predOp.getStage() == predOp.getMaxStage() - 1) {
        return rewriter.create<mlir::arith::ConstantIntOp>(
            predOp.getLoc(), 1, predOp.getResult().getType());
      } else {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);
        return triton::emitPredicateForStage(
                   rewriter, predOp.getIv(), predOp.getUb(), predOp.getStep(),
                   predOp.getMaxStage(), predOp.getStage())
            .getDefiningOp();
      }
    }
  }
  return op;
}

static bool hasMMAv5WaitsInLastStage(scf::ForOp forOp,
                                     CoarseSchedule &schedule) {
  int maxStage = schedule.getNumStages() - 1;
  bool hasMMAv5 = false;
  bool hasWaitInLastStage = false;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<triton::nvidia_gpu::WaitBarrierOp>(op) &&
        schedule[&op].first == maxStage) {
      hasWaitInLastStage = true;
    }
    if (isa<triton::nvidia_gpu::MMAv5OpInterface>(op)) {
      hasMMAv5 = true;
    }
  }
  return hasMMAv5 && hasWaitInLastStage;
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
    bool keepPredicateStage = forOp->hasAttr("__test_keep_predicate_stage");
    // TODO: Enable epilogue peeling for warp specialized loops
    // Heuristic: only peel epilogue for MMAv5 loops with waits in the last
    // stage
    bool customEpiloguePeeling =
        hasMMAv5WaitsInLastStage(forOp, schedule) &&
        !forOp->getParentOfType<triton::gpu::WarpSpecializeOp>() &&
        !keepPredicateStage; // do not peel if we are testing the stage
                             // predication

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
      mlir::triton::peelLoopEpilogue(forOp, processPeeledEpilogueOp);
    }
  }
  assert(moduleOp.getOps<triton::gpu::PredicateStageOp>().empty() &&
         "PredicateStageOp should be resolved after the pipeline expansion");
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
