#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Utility.h"

using namespace mlir;

namespace mlir {
namespace triton {

void peelLoopEpilogue(
    scf::ForOp forOp,
    function_ref<Operation *(RewriterBase &, Operation *, bool)>
        processPeeledOp) {
  SmallVector<Operation *> loopBodyOps;
  IRRewriter rewriter(forOp);
  Location loc = forOp.getLoc();
  Type type = forOp.getStep().getType();

  // Fetch loop bounds and step
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value newUpperBound = rewriter.create<arith::SubIOp>(loc, upperBound, step);

  rewriter.setInsertionPointAfter(forOp);
  Value lastIV = getLastInductionValue(rewriter, forOp);

  auto cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             lowerBound, upperBound);

  // Create an if op to execute the peeled iteration
  IRMapping map;
  map.map(forOp.getRegionIterArgs(), forOp.getResults());
  map.map(forOp.getInductionVar(), lastIV);
  auto ifOp = rewriter.create<scf::IfOp>(loc, forOp.getResultTypes(), cond,
                                         /*hasElse=*/true);
  ifOp.getThenRegion().front().erase();
  forOp.getBodyRegion().cloneInto(&ifOp.getThenRegion(), map);
  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  rewriter.create<scf::YieldOp>(loc, forOp.getResults());

  forOp->replaceUsesWithIf(ifOp, [&](OpOperand &operand) {
    return !ifOp->isAncestor(operand.getOwner());
  });

  forOp.getUpperBoundMutable().assign(newUpperBound);

  if (processPeeledOp) {
    for (auto &op :
         llvm::make_early_inc_range(forOp.getBody()->without_terminator())) {
      Operation *newOp = processPeeledOp(rewriter, &op, /*isEpilogue=*/false);
      if (newOp && newOp != &op) {
        op.replaceAllUsesWith(newOp);
      }
    }
    for (auto &op : llvm::make_early_inc_range(
             ifOp.getThenRegion().front().without_terminator())) {
      Operation *newOp = processPeeledOp(rewriter, &op, /*isEpilogue=*/true);
      if (newOp && newOp != &op) {
        op.replaceAllUsesWith(newOp);
      }
    }
  }
}

} // namespace triton
} // namespace mlir
