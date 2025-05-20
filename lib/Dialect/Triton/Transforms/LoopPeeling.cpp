#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/EquivalenceClasses.h"

#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"

using namespace mlir;

namespace {
// UTILS

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

std::optional<int> getPeelEpilogueIterations(scf::ForOp forOp) {
  if (!forOp->hasAttr(triton::kPeelEpilogueIterationsAttrName)) {
    return std::nullopt;
  }
  return cast<IntegerAttr>(
             forOp->getAttr(triton::kPeelEpilogueIterationsAttrName))
      .getInt();
}

} // namespace

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONLOOPPEELING
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

void annotateLoopForEpiloguePeeling(RewriterBase &rewriter, scf::ForOp forOp,
                                    int numIterations) {
  forOp->setAttr(kPeelEpilogueIterationsAttrName,
                 rewriter.getI64IntegerAttr(numIterations));
}

void peelLoopEpilogue(
    scf::ForOp forOp, int numIterations,
    function_ref<Operation *(RewriterBase &, Operation *, Value)> predicateOp,
    SmallVector<Operation *> *peeledOps) {
  assert(numIterations == 1); // TODO: for now
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
  // execute only one iteration (which we are going to peel)
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
    Operation *newOp =
        cloneAndUpdateOperands(rewriter, &op, [&](OpOperand *newOperand) {
          if (mapping.count(newOperand->get())) {
            newOperand->set(mapping[newOperand->get()]);
          }
        });
    if (peeledOps) {
      peeledOps->push_back(newOp);
    }
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
    return domInfo.properlyDominates(lastOp, operand.getOwner(),
                                     /*enclosingOpOK=*/false);
  });
}

void peelLoopEpilogue(
    scf::ForOp forOp,
    function_ref<Operation *(RewriterBase &, Operation *, Value)> predicateOp,
    SmallVector<Operation *> *peeledOps) {
  std::optional<int64_t> numIterationsOpt = getPeelEpilogueIterations(forOp);
  if (!numIterationsOpt) {
    return;
  }
  int64_t numIterations = *numIterationsOpt;
  peelLoopEpilogue(forOp, numIterations, predicateOp, peeledOps);

  // Remove the peel epilogue attribute
  forOp->removeAttr(triton::kPeelEpilogueIterationsAttrName);
}

class LoopPeelingPass : public impl::TritonLoopPeelingBase<LoopPeelingPass> {
  void runOnOperation() override {
    getOperation().walk([&](scf::ForOp forOp) {
      if (getPeelEpilogueIterations(forOp)) {
        peelLoopEpilogue(forOp, /*predicateOp=*/nullptr, /*peeledOps=*/nullptr);
      }
    });
  }
};

} // namespace triton
} // namespace mlir
