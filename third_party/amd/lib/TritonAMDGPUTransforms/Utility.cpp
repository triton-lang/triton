#include "Utility.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include <limits>

namespace deduceMin {
int deduceMinCountInBlock(Block &block,
                          const std::function<int(Operation *)> &countFunc);

// Returns the minimum found when accumulating countFunc(op) between begin and
// end (inclusive)
int deduceMinCountBetweeOps(Operation *beginOp, Operation *endOp,
                            const std::function<int(Operation *)> &countFunc) {
  assert(beginOp && endOp);
  assert(beginOp == endOp || beginOp->isBeforeInBlock(endOp));
  int count = 0;
  for (auto op = beginOp; op != endOp; op = op->getNextNode()) {
    if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
      assert(!ifOp.getThenRegion().empty() && !ifOp.getElseRegion().empty());
      auto minThen =
          deduceMinCountInBlock(ifOp.getThenRegion().front(), countFunc);
      auto minElse =
          deduceMinCountInBlock(ifOp.getElseRegion().front(), countFunc);
      count += std::min(minThen, minElse);
    } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
      if (std::optional<APInt> tripCount = forOp.getStaticTripCount()) {
        uint64_t tcVal = 0;
        if (forOp.getUnsignedCmp() && tripCount->ugt(0))
          tcVal = tripCount->getZExtValue();
        else if (!forOp.getUnsignedCmp() && tripCount->sgt(0))
          tcVal = tripCount->getSExtValue();
        if (tcVal > 0)
          count += tcVal * deduceMinCountInBlock(*forOp.getBody(), countFunc);
      }
    } else {
      count += countFunc(op);
    }
  }
  return count;
}

// Returns the minimum found when accumulating countFunc(op) for all paths
// between the block's start and end op
int deduceMinCountInBlock(Block &block,
                          const std::function<int(Operation *)> &countFunc) {
  if (block.empty())
    return 0;
  return deduceMinCountBetweeOps(&block.front(), &block.back(), countFunc);
}
} // namespace deduceMin

int deduceMinCountOnDefChain(Value defValue, Operation *consumerOp,
                             const std::function<int(Operation *)> &countFunc,
                             int pathSum, int foundMin) {
  using namespace deduceMin;
  // If the value is not defined in the same region as the consumer we need to
  // peel the parent region of consumer until we arrive at value's region
  while (consumerOp->getParentRegion() != defValue.getParentRegion()) {
    pathSum += deduceMin::deduceMinCountBetweeOps(
        &consumerOp->getBlock()->front(), consumerOp, countFunc);
    consumerOp = consumerOp->getParentOp();
  }

  // Break recursion if we arrive at the producer updating the path based on the
  // ops between producer and consumer
  if (Operation *defOp = defValue.getDefiningOp()) {
    pathSum +=
        deduceMinCountBetweeOps(defOp->getNextNode(), consumerOp, countFunc);
    foundMin = std::min(foundMin, pathSum);
    return foundMin;
  }
  // If value is a loop carried argument (BlockArgument) we need to look at
  // initial arguments of the loop and the previous iteration
  if (auto arg = mlir::dyn_cast<BlockArgument>(defValue)) {
    Block *block = arg.getOwner();
    auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

    // Failed to track, return 0 conservatively.
    if (!forOp || forOp.getBody()->empty()) {
      return 0;
    }

    Operation *firstOpInLoop = &*forOp.getBody()->begin();
    pathSum += deduceMinCountBetweeOps(firstOpInLoop, consumerOp, countFunc);

    // Break recursion early if we exceed previous min
    if (pathSum >= foundMin)
      return foundMin;

    Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
    int countLoopInit = deduceMinCountOnDefChain(incomingVal, forOp, countFunc,
                                                 pathSum, foundMin);

    Operation *yieldOp = block->getTerminator();
    Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
    int countPreviousIter = deduceMinCountOnDefChain(
        prevVal, yieldOp, countFunc, pathSum, foundMin);

    return std::min(std::min(countLoopInit, countPreviousIter), foundMin);
  }

  // Unsupported value, return 0 conservatively.
  return 0;
}

int deduceMinCountOnDefChain(Value defValue, Operation *consumerOp,
                             llvm::function_ref<int(Operation *)> countFunc) {
  return deduceMinCountOnDefChain(defValue, consumerOp, countFunc, 0,
                                  std::numeric_limits<int>::max());
}
