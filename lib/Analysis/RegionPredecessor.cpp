#include "triton/Analysis/RegionPredecessor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

RegionPredecessorAnalysis::RegionPredecessorAnalysis(Operation *op) {
  op->walk([&](Operation *op) {
    if (auto br = dyn_cast<RegionBranchOpInterface>(op)) {
      SmallVector<RegionSuccessor> successors;
      br.getSuccessorRegions(RegionBranchPoint::parent(), successors);
      BlockIter it(op->getBlock(), op->getIterator());
      for (RegionSuccessor &successor : successors) {
        if (successor.isParent())
          predecessors[op].insert(it);
        else
          predecessors[successor.getSuccessor()].insert(it);
      }
      return WalkResult::advance();
    }

    // FIXME: `ReturnLike` adds `RegionBranchTerminatorOpInterface` for some
    // reason. Check that the parent is actually a `RegionBranchOpInterface`.
    auto br = dyn_cast<RegionBranchTerminatorOpInterface>(op);
    if (br && isa<RegionBranchOpInterface>(br->getParentOp())) {
      SmallVector<Attribute> operands(br->getNumOperands());
      SmallVector<RegionSuccessor> regions;
      br.getSuccessorRegions(operands, regions);
      BlockIter it(br->getBlock(), br->getBlock()->end());
      for (RegionSuccessor &successor : regions) {
        if (successor.isParent())
          predecessors[br->getParentOp()].insert(it);
        else
          predecessors[successor.getSuccessor()].insert(it);
      }
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });
}

ArrayRef<BlockIter> RegionPredecessorAnalysis::getPredecessors(Operation *op) {
  return predecessors.at(op).getArrayRef();
}

ArrayRef<BlockIter> RegionPredecessorAnalysis::getPredecessors(Region *region) {
  return predecessors.at(region).getArrayRef();
}
