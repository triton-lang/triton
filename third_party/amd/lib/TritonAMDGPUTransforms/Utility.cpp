#include "Utility.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace findMin {
int findMinInBlock(Block &block,
                   const std::function<int(Operation *)> &countFunc);

// Returns the minimum found when accumulating countFunc(op) for all paths
// between op1 and op2
int findMinBetweenOps(Operation *op1, Operation *op2,
                      const std::function<int(Operation *)> &countFunc) {
  assert(op1 && op2);
  int count = 0;
  for (auto op = op1; op != op2; op = op->getNextNode()) {
    if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
      assert(!ifOp.getThenRegion().empty() && !ifOp.getElseRegion().empty());
      auto minThen = findMinInBlock(ifOp.getThenRegion().front(), countFunc);
      auto minElse = findMinInBlock(ifOp.getElseRegion().front(), countFunc);
      count += std::min(minThen, minElse);
    } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
      count += findMinInBlock(*forOp.getBody(), countFunc);
    } else {
      count += countFunc(op);
    }
  }
  return count;
}

// Returns the minimum found when accumulating countFunc(op) for all paths
// between the block's start and end op
int findMinInBlock(Block &block,
                   const std::function<int(Operation *)> &countFunc) {
  if (block.empty())
    return 0;
  return findMinBetweenOps(&block.front(), &block.back(), countFunc);
}
} // namespace findMin

int findMinCountInDefChain(Value value, Operation *consumerOp,
                           const std::function<int(Operation *)> &countFunc,
                           int pathSum, int foundMin) {

  // If the value is not defined in the same region as the consumer we need to
  // peel the parent region of consumer until we arrive at value's region
  while (consumerOp->getParentRegion() != value.getParentRegion()) {
    assert(!consumerOp->getParentRegion()->empty());
    assert(!consumerOp->getParentRegion()->front().empty());
    pathSum += findMin::findMinBetweenOps(
        &consumerOp->getParentRegion()->front().front(), consumerOp, countFunc);
    consumerOp = consumerOp->getParentOp();
  }

  // Break recursion if we arrive at the producer updating the path based on the
  // ops between producer and consumer
  if (Operation *defOp = value.getDefiningOp()) {
    pathSum +=
        findMin::findMinBetweenOps(defOp->getNextNode(), consumerOp, countFunc);
    foundMin = std::min(foundMin, pathSum);
    return foundMin;
  }
  // If value is a loop carried value (BlockArgument) we need to look at initial
  // argumets of the loop and the previous iteration to find all producers
  if (auto arg = mlir::dyn_cast<BlockArgument>(value)) {
    Block *block = arg.getOwner();
    auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

    // Failed to track, return 0 conservatively.
    if (!forOp || forOp.getBody()->empty()) {
      return 0;
    }

    Operation *firstForOp = &*forOp.getBody()->begin();
    pathSum += findMin::findMinBetweenOps(firstForOp, consumerOp, countFunc);

    // Break recursion early if we exceed previous min
    if (pathSum >= foundMin)
      return foundMin;

    // Split the path and traverse the value assigned to the initial loop
    // iteration and the yield from the previous iteation recursively.
    Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
    int countLoopInit = findMinCountInDefChain(incomingVal, forOp, countFunc,
                                               pathSum, foundMin);
    Operation *yieldOp = block->getTerminator();
    Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
    int countPreviousIter =
        findMinCountInDefChain(prevVal, yieldOp, countFunc, pathSum, foundMin);
    return std::min(std::min(countLoopInit, countPreviousIter), foundMin);
  }

  // Unsupported value, return 0 conservatively.
  return 0;
}
