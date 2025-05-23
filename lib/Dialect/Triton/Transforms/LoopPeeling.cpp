#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

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
  forOp.getUpperBoundMutable().assign(newUpperBound);

  rewriter.setInsertionPointAfter(forOp);

  // Last iter induction variable value
  // lastIV = lb + floor( (ub – lb – 1) / s ) * s
  Value range = rewriter.create<arith::SubIOp>(loc, upperBound, lowerBound);
  Value rangeM1 = rewriter.create<arith::SubIOp>(
      loc, range, rewriter.create<arith::ConstantIntOp>(loc, 1, type));
  Value itersM1 = rewriter.create<arith::DivSIOp>(loc, rangeM1, step);
  Value delta = rewriter.create<arith::MulIOp>(loc, itersM1, step);
  Value lastIV = rewriter.create<arith::AddIOp>(loc, delta, lowerBound);

  auto cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             lastIV, upperBound);

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
