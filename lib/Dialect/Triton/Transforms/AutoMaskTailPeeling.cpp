#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONAUTOMASKTAILPEELING
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

// Return true if `v` is the loop induction variable or a broadcast from it.
static bool isIVOrBroadcast(mlir::Value v, mlir::scf::ForOp forOp) {
  if (v == forOp.getInductionVar())
    return true;
  // In Triton MLIR, splat is used for broadcasting scalars to tensors.
  if (auto splat = v.getDefiningOp<SplatOp>())
    return splat.getSrc() == forOp.getInductionVar();
  if (auto bcast = v.getDefiningOp<BroadcastOp>())
    return bcast.getSrc() == forOp.getInductionVar();
  return false;
}

// Checks if a value is a constant tt.make_range, possibly with expand_dims.
static bool isConstantRange(mlir::Value v) {
  if (v.getDefiningOp<MakeRangeOp>())
    return true;
  if (auto expand = v.getDefiningOp<ExpandDimsOp>()) {
    if (expand.getSrc().getDefiningOp<MakeRangeOp>())
      return true;
  }
  return false;
}

// Recognizes the pattern: `splat(%iv) + make_range(...)`
static bool isIVPlusConstantRange(mlir::Value v, mlir::scf::ForOp forOp) {
  auto addOp = v.getDefiningOp<mlir::arith::AddIOp>();
  if (!addOp)
    return false;

  Value lhs = addOp.getLhs();
  Value rhs = addOp.getRhs();

  auto match = [&](Value ivSide, Value rangeSide) {
    return isIVOrBroadcast(ivSide, forOp) && isConstantRange(rangeSide);
  };

  return match(lhs, rhs) || match(rhs, lhs);
}

// Recognizes the pattern: `splat(K - (iv * C))` where K is a loop invariant.
static bool isRemainingElementsExpr(mlir::Value v, mlir::scf::ForOp forOp) {
  auto splat = v.getDefiningOp<SplatOp>();
  if (!splat)
    return false;

  auto sub = splat.getSrc().getDefiningOp<arith::SubIOp>();
  if (!sub)
    return false;

  auto mul = sub.getRhs().getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;

  // Check the multiplication: one operand must be the IV, the other a constant.
  Value ivOperand = nullptr;
  Value constOperand = nullptr;
  if (mul.getLhs() == forOp.getInductionVar()) {
    constOperand = mul.getRhs();
  } else if (mul.getRhs() == forOp.getInductionVar()) {
    constOperand = mul.getLhs();
  } else {
    return false; // IV not found.
  }

  if (!mlir::matchPattern(constOperand, m_Constant()))
    return false; // Not multiplying by a constant.

  // The LHS of the subtraction (e.g., %K) must be defined outside the loop.
  Value invariant = sub.getLhs();
  return forOp.isDefinedOutsideOfLoop(invariant);
}

// Tries to match the "vector-add" like pattern: `(iv + range) < splat(ub)`
static bool isVectorLikeMaskPattern(mlir::Value mask, mlir::scf::ForOp forOp,
                                    mlir::Value loopUB) {
  auto cmp = mask.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp)
    return false;
  if (cmp.getPredicate() != arith::CmpIPredicate::slt)
    return false;

  if (!isIVPlusConstantRange(cmp.getLhs(), forOp))
    return false;

  auto rhsSplat = cmp.getRhs().getDefiningOp<SplatOp>();
  return rhsSplat && rhsSplat.getSrc() == loopUB;
}

// Tries to match the "matmul" like pattern: `range < K - (k * C)`
static bool isMatmulLikeMaskPattern(mlir::Value mask, mlir::scf::ForOp forOp) {
  auto cmp = mask.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp) {
    if (auto bcast = mask.getDefiningOp<BroadcastOp>()) {
      cmp = bcast.getSrc().getDefiningOp<mlir::arith::CmpIOp>();
    }
  }
  if (!cmp)
    return false;

  auto match = [&](Value range, Value remaining) {
    return isConstantRange(range) && isRemainingElementsExpr(remaining, forOp);
  };

  if (cmp.getPredicate() == arith::CmpIPredicate::slt) {
    return match(cmp.getLhs(), cmp.getRhs());
  }
  if (cmp.getPredicate() == arith::CmpIPredicate::sgt) {
    return match(cmp.getRhs(), cmp.getLhs());
  }
  return false;
}

// Main entry point for pattern recognition.
static bool isLoadMaskAlwaysTrueInMainLoop(LoadOp load, mlir::scf::ForOp forOp,
                                           Value originalUB) {
  auto mask = load.getMask();
  if (!mask)
    return false;

  // Try matching the matmul pattern first.
  if (isMatmulLikeMaskPattern(mask, forOp)) {
    return true;
  }

  // Fallback to the vector-add-like pattern.
  if (isVectorLikeMaskPattern(mask, forOp, originalUB)) {
    return true;
  }

  return false;
}

// Remove the mask from the load op by rebuilding it without mask/other.
static mlir::Operation *dropLoadMask(mlir::RewriterBase &rewriter,
                                     LoadOp load) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(load);
  auto newLoad = rewriter.create<LoadOp>(
      load.getLoc(), load.getPtr(), /*mask=*/mlir::Value(),
      /*other=*/mlir::Value(), load.getBoundaryCheckAttr(),
      load.getPaddingAttr(), load.getCache(), load.getEvict(),
      load.getIsVolatile());
  return newLoad.getOperation();
}

struct AutoMaskTailPeelingPass
    : public impl::TritonAutoMaskTailPeelingBase<AutoMaskTailPeelingPass> {

  void
  peelLoopEpilogue(scf::ForOp forOp,
                   function_ref<Operation *(RewriterBase &, Operation *, bool)>
                       processPeeledOp) {
    IRRewriter rewriter(forOp);
    Location loc = forOp.getLoc();

    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    Value step = forOp.getStep();

    // Do not peel if the loop may not have more than one iteration.
    // This is a conservative check.
    if (auto stepCst = getConstantIntValue(step)) {
      if (auto lbCst = getConstantIntValue(lowerBound)) {
        if (auto ubCst = getConstantIntValue(upperBound)) {
          if (*lbCst + *stepCst >= *ubCst)
            return;
        }
      }
    }

    // Calculate the new upper bound for the main loop.
    Value diff = rewriter.create<arith::SubIOp>(loc, upperBound, lowerBound);
    Value tripCount = rewriter.create<arith::CeilDivSIOp>(loc, diff, step);
    Value alignedDiff = rewriter.create<arith::SubIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, tripCount, step), step);
    Value newUpperBound =
        rewriter.create<arith::AddIOp>(loc, lowerBound, alignedDiff);
    rewriter.setInsertionPoint(forOp);
    auto mainLoop = cast<scf::ForOp>(rewriter.clone(*forOp.getOperation()));

    // Modify the cloned loop to become the main loop.
    mainLoop.getUpperBoundMutable().assign(newUpperBound);

    // Modify the original loop to become the epilogue loop.
    forOp.getLowerBoundMutable().assign(newUpperBound);
    forOp.getInitArgsMutable().assign(mainLoop.getResults());

    // Apply the processing function to the operations in both loops.
    if (processPeeledOp) {
      // Process main loop (isEpilogue = false)
      for (auto &op : llvm::make_early_inc_range(
               mainLoop.getBody()->without_terminator())) {
        Operation *newOp = processPeeledOp(rewriter, &op, /*isEpilogue=*/false);
        if (newOp && newOp != &op) {
          op.replaceAllUsesWith(newOp);
          rewriter.eraseOp(&op);
        }
      }
      // Process epilogue loop (isEpilogue = true).
      for (auto &op :
           llvm::make_early_inc_range(forOp.getBody()->without_terminator())) {
        processPeeledOp(rewriter, &op, /*isEpilogue=*/true);
      }
    }
  }

  void runOnOperation() override {
    llvm::SmallVector<mlir::scf::ForOp> loops;
    getOperation()->walk(
        [&](mlir::scf::ForOp forOp) { loops.push_back(forOp); });

    for (mlir::scf::ForOp forOp : loops) {
      if (auto *def = forOp.getStep().getDefiningOp()) {
        if (auto stepCst = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
          if (auto stepAttr =
                  mlir::dyn_cast<mlir::IntegerAttr>(stepCst.getValue())) {
            if (stepAttr.getInt() <= 0)
              continue;
          }
        }
      }

      bool hasCandidate = false;
      Value originalUpperBound = forOp.getUpperBound();
      forOp.getBody()->walk([&](LoadOp load) {
        if (isLoadMaskAlwaysTrueInMainLoop(load, forOp, originalUpperBound))
          hasCandidate = true;
      });
      if (!hasCandidate)
        continue;

      auto process = [&](mlir::RewriterBase &rewriter, mlir::Operation *op,
                         bool isEpilogue) -> mlir::Operation * {
        if (auto load = mlir::dyn_cast<LoadOp>(op)) {
          auto containingForOp = op->getParentOfType<scf::ForOp>();
          if (!containingForOp)
            return op;

          if (isLoadMaskAlwaysTrueInMainLoop(load, containingForOp,
                                             originalUpperBound)) {
            if (!isEpilogue) {
              return dropLoadMask(rewriter, load);
            }
          }
        }
        return op;
      };

      peelLoopEpilogue(forOp, process);
    }
  }
};

} // namespace mlir::triton
