#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/CSE.h"
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
  rewriter.eraseOp(op);
  return mask;
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

class InductionVarLTUBPattern : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    if (cmpOp.getPredicate() != arith::CmpIPredicate::slt &&
        cmpOp.getPredicate() != arith::CmpIPredicate::sle) {
      return failure();
    }
    scf::ForOp forOp = cmpOp->getParentOfType<scf::ForOp>();
    if (!forOp) {
      return failure();
    }
    Value inductionVar = cmpOp.getLhs();
    Value upperBound = cmpOp.getRhs();
    if (inductionVar != forOp.getInductionVar()) {
      return failure();
    }
    if (upperBound != forOp.getUpperBound()) {
      return failure();
    }
    // Induction var is always less than upper bound
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(cmpOp, 1, 1);
    return success();
  }
};

class IntLTItselfMinusXPattern : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    if (cmpOp.getPredicate() != arith::CmpIPredicate::slt) {
      return failure();
    }
    Value lhs = cmpOp.getLhs();
    Value rhs = cmpOp.getRhs();
    auto subOp = rhs.getDefiningOp<arith::SubIOp>();
    if (!subOp) {
      return failure();
    }
    Value subLhs = subOp.getLhs();
    Value subRhs = subOp.getRhs();
    if (subLhs != lhs) {
      return failure();
    }
    std::optional<int64_t> subRhsValue = getConstantIntValue(subRhs);
    if (!subRhsValue) {
      return failure();
    }
    // a < a - x is true if x < 0
    int result = (*subRhsValue < 0) ? 1 : 0;
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(cmpOp, result, 1);
    return success();
  }
};

struct HackRemoveMaskOutsideOfLoopPattern : public OpRewritePattern<MaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaskOp maskOp,
                                PatternRewriter &rewriter) const override {
    if (maskOp->getParentOfType<scf::ForOp>()) {
      return failure();
    }
    rewriter.eraseOp(maskOp);
    return success();
  }
};
static void resolveMaskOp(ModuleOp moduleOp) {
  DominanceInfo domInfo(moduleOp);
  IRRewriter rewriter(moduleOp);
  eliminateCommonSubExpressions(rewriter, domInfo, moduleOp);

  auto arithDialect =
      moduleOp.getContext()->getLoadedDialect<arith::ArithDialect>();
  RewritePatternSet patterns(moduleOp.getContext());
  arithDialect->getCanonicalizationPatterns(patterns);
  patterns.add<InductionVarLTUBPattern>(moduleOp.getContext());
  patterns.add<IntLTItselfMinusXPattern>(moduleOp.getContext());
  if (applyPatternsGreedily(moduleOp, std::move(patterns)).failed())
    return llvm::report_fatal_error("Failed to canonicalize the IR");

  SmallVector<MaskOp> opsToErase;
  moduleOp->walk([&](MaskOp maskOp) {
    rewriter.setInsertionPoint(maskOp);
    while (&maskOp.getBody()->front() != maskOp.getBody()->getTerminator()) {
      Operation *op = &maskOp.getBody()->front();
      rewriter.moveOpAfter(op, maskOp);
      triton::predicateOp(rewriter, op, maskOp.getPred());
    }
    maskOp->replaceAllUsesWith(
        maskOp.getBody()->getTerminator()->getOperands());
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

/// Clone `op` and call `callback` on the cloned op's operands as well as any
/// operands of nested ops that:
/// 1) aren't defined within the new op or
/// 2) are block arguments.
static Operation *
cloneAndUpdateOperands(RewriterBase &rewriter, Operation *op,
                       function_ref<void(OpOperand *newOperand)> callback) {
  Operation *clone = rewriter.clone(*op);
  clone->walk<WalkOrder::PreOrder>([&](Operation *nested) {
    // 'clone' itself will be visited first.
    for (OpOperand &operand : nested->getOpOperands()) {
      Operation *def = operand.get().getDefiningOp();
      if ((def && !clone->isAncestor(def)) || isa<BlockArgument>(operand.get()))
        callback(&operand);
    }
  });
  return clone;
}

// This function peels the last 'numIterations' iterations from the given
// scf::ForOp
static void peelLastIterations(scf::ForOp forOp, unsigned numIterations) {
  assert(numIterations == 1); // for now
  IRRewriter rewriter(forOp);
  auto loc = forOp.getLoc();

  // Fetch loop bounds and step
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  // Compute new upper bound for the main loop: newUpperBound = upperBound -
  // (numIterations * step)
  auto numConst = rewriter.create<arith::ConstantIntOp>(loc, numIterations,
                                                        rewriter.getI32Type());
  Value peelOffset = rewriter.create<arith::MulIOp>(loc, numConst, step);
  Value newUpperBound =
      rewriter.create<arith::SubIOp>(loc, upperBound, peelOffset);

  forOp.getUpperBoundMutable().assign(newUpperBound);
  rewriter.setInsertionPointAfter(forOp);
  // If the original lower bound is less than the original upper bound, we will
  // execute the loop body once (which we are going to peel)
  auto lastIterPred = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, lowerBound, upperBound);

  // Last iter induction variable value
  // translatedUB = UB - LB
  // IV = (step - (translatedUB % step)) + UB
  Value translatedUB =
      rewriter.create<arith::SubIOp>(loc, newUpperBound, lowerBound);
  Value _ugh = rewriter.create<arith::RemUIOp>(loc, translatedUB, step);
  Value _ugh1 = rewriter.create<arith::SubIOp>(loc, step, _ugh);
  Value lastIterInductionVar =
      rewriter.create<arith::AddIOp>(loc, _ugh1, newUpperBound);

  llvm::DenseMap<Value, Value> mapping;
  for (auto [arg, operand] :
       llvm::zip(forOp.getRegionIterArgs(), forOp.getResults())) {
    mapping[arg] = operand;
  }

  mapping[forOp.getInductionVar()] = lastIterInductionVar;

  SmallVector<Value> peeledResults = forOp.getResults();
  Operation *lastOp = nullptr;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    // TODO HACK: skip masked operations
    if (isa<MaskOp>(op)) {
      continue;
    }
    Operation *newOp =
        cloneAndUpdateOperands(rewriter, &op, [&](OpOperand *newOperand) {
          if (mapping.count(newOperand->get())) {
            newOperand->set(mapping[newOperand->get()]);
          }
        });
    newOp = predicateOp(rewriter, newOp, lastIterPred);
    assert(newOp && "Failed to create masked operation");
    lastOp = newOp;
    for (auto [result, oldResult] :
         llvm::zip(newOp->getResults(), op.getResults())) {
      mapping[oldResult] = result;

      for (OpOperand &yieldOperand :
           forOp.getBody()->getTerminator()->getOpOperands()) {
        if (yieldOperand.get() != oldResult) {
          continue;
        }
        peeledResults[yieldOperand.getOperandNumber()] = result;
      }
    }
  }

  DominanceInfo domInfo(forOp->getParentOfType<ModuleOp>());

  forOp->replaceUsesWithIf(peeledResults, [&](OpOperand &operand) {
    return domInfo.properlyDominates(lastOp, operand.getOwner());
  });
}

static bool onlyWaitsAreUnmasked(scf::ForOp forOp) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
            triton::nvidia_gpu::AsyncTMAGatherOp,
            triton::gpu::AsyncCopyGlobalToLocalOp>(op)) {
      return false;
    }
  }
  return true;
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

    SmallVector<scf::ForOp> loops;
    moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops) {
      auto funcName =
          forOp.getOperation()->getParentOfType<triton::FuncOp>().getName();
      if (funcName.starts_with("_attention_backward")) {
        peelLastIterations(forOp, 1);
      }
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
