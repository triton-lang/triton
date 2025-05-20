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

Value getConstantInt(RewriterBase &rewriter, Location loc, int64_t value,
                     Type type) {
  if (isa<IntegerType>(type)) {
    return rewriter.create<arith::ConstantIntOp>(loc, value, type);
  }
  if (isa<IndexType>(type)) {
    return rewriter.create<arith::ConstantIndexOp>(loc, value);
  }
  llvm_unreachable("Unsupported type");
}

} // namespace

namespace mlir {
namespace triton {

void peelLoopEpilogue(
    scf::ForOp forOp, int numIterations,
    function_ref<Operation *(RewriterBase &, Operation *, Value)>
        processPeeledOp,
    function_ref<Operation *(RewriterBase &, Operation *)> processLoopBodyOp) {
  // TODO: generalize for arbitrary number of peeled iterations
  assert(numIterations == 1);

  SmallVector<Operation *> loopBodyOps;
  IRRewriter rewriter(forOp);
  Location loc = forOp.getLoc();
  Type stepType = forOp.getStep().getType();

  // Fetch loop bounds and step
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  // Compute new upper bound for the main loop: newUpperBound = upperBound -
  // (numIterations * step)
  auto numConst = getConstantInt(rewriter, loc, numIterations, stepType);
  Value peelOffset = rewriter.create<arith::MulIOp>(loc, numConst, step);
  Value newUpperBound =
      rewriter.create<arith::SubIOp>(loc, upperBound, peelOffset);

  forOp.getUpperBoundMutable().assign(newUpperBound);
  rewriter.setInsertionPointAfter(forOp);

  // We are going to execute the peeled iteration if the original lower bound is
  // less than the original upper bound
  auto lastIterPred = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, lowerBound, upperBound);

  // Last iter induction variable value
  // lastIV = lb + floor( (ub – lb – 1) / s ) * s
  Value range = rewriter.create<arith::SubIOp>(loc, upperBound, lowerBound);
  Value rangeM1 = rewriter.create<arith::SubIOp>(
      loc, range, getConstantInt(rewriter, loc, 1, stepType));
  Value itersM1 = rewriter.create<arith::DivSIOp>(loc, rangeM1, step);
  Value delta = rewriter.create<arith::MulIOp>(loc, itersM1, step);
  Value lastIV = rewriter.create<arith::AddIOp>(loc, delta, lowerBound);

  llvm::DenseMap<Value, Value> mapping;
  for (auto [arg, operand] :
       llvm::zip(forOp.getRegionIterArgs(), forOp.getResults())) {
    mapping[arg] = operand;
  }

  mapping[forOp.getInductionVar()] = lastIV;

  SmallVector<Value> peeledResults = forOp.getResults();
  Operation *lastOp = nullptr;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    Operation *newOp =
        cloneAndUpdateOperands(rewriter, &op, [&](OpOperand *newOperand) {
          if (mapping.count(newOperand->get())) {
            newOperand->set(mapping[newOperand->get()]);
          }
        });
    if (processPeeledOp) {
      newOp = processPeeledOp(rewriter, newOp, lastIterPred);
    }
    if (processLoopBodyOp) {
      loopBodyOps.push_back(&op);
    }
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

  for (auto op : loopBodyOps) {
    Operation *newOp = processLoopBodyOp(rewriter, op);
    if (newOp && newOp != op) {
      op->replaceAllUsesWith(newOp);
      op->erase();
    }
  }
}

} // namespace triton
} // namespace mlir
