#include "Utility.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace deduceMin {
int deduceMinCountInBlock(Block &block,
                          const std::function<int(Operation *)> &countFunc);

std::optional<int> getIntIfConstant(Value v) {
  if (auto constantOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      return attr.getInt();
    }
  }
  return std::nullopt;
}

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
      auto upperBound = getIntIfConstant(forOp.getUpperBound());
      auto lowerBound = getIntIfConstant(forOp.getLowerBound());
      auto step = getIntIfConstant(forOp.getStep());
      if (upperBound.has_value() && lowerBound.has_value() &&
          step.has_value() && *step != 0) {
        float range = static_cast<float>(*upperBound) - *lowerBound;
        int numSteps = std::ceil(range / *step);
        count += numSteps * deduceMinCountInBlock(*forOp.getBody(), countFunc);
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
  // initial argumets of the loop and the previous iteration to find all
  // producers
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

    // Split the path and traverse the value assigned to the initial loop
    // iteration and the yield from the previous iteation recursively.
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
                             const std::function<int(Operation *)> &countFunc) {
  return deduceMinCountOnDefChain(defValue, consumerOp, countFunc, 0,
                                  std::numeric_limits<int>::max());
}
