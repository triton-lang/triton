#include "triton/Dialect/Triton/IR/Utility.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;

Value tt::getPredMask(RewriterBase &rewriter, Type typeLike, Value currentMask,
                      Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
  Location loc = pred.getLoc();
  Value mask = pred;
  if (isa<RankedTensorType>(maskType)) {
    mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
  }
  if (currentMask) {
    mask = rewriter.create<arith::AndIOp>(loc, mask, currentMask);
  }
  return mask;
}

static tt::MakeTensorPtrOp getMakeTensorPtrOpImpl(Operation *op, Value v) {

  if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
    return makeTensorPtrOp;
  }

  if (auto advanceOp = dyn_cast<tt::AdvanceOp>(op)) {
    return tt::getMakeTensorPtrOp(advanceOp.getPtr());
  }

  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    auto idx = cast<OpResult>(v).getResultNumber();
    llvm::SmallVector<scf::YieldOp> yieldOps;
    op->walk([&](Operation *op) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op))
        yieldOps.push_back(yieldOp);
    });

    // benzh@ if multi yields, all yields operand should come from same arg.
    Value newValue = yieldOps[0].getOperands()[idx];
    return tt::getMakeTensorPtrOp(newValue);
  }

  llvm_unreachable("Unable to getMakeTensorPtr()");
}

tt::MakeTensorPtrOp tt::getMakeTensorPtrOp(Value v) {
  using BranchOps = llvm::SetVector<std::pair<Operation *, int>>;
  llvm::DenseMap<Block *, BranchOps> blockToCFOps;
  auto moduleOp =
      v.getParentBlock()->getParentOp()->getParentOfType<ModuleOp>();

  moduleOp.walk([&](Operation *op) {
    if (auto br = dyn_cast<cf::BranchOp>(op)) {
      Block *block = br.getDest();
      blockToCFOps[block].insert({op, -1});
    }
    if (auto condBr = dyn_cast<cf::CondBranchOp>(op)) {
      Block *blockT = condBr.getTrueDest();
      Block *blockF = condBr.getFalseDest();
      blockToCFOps[blockT].insert({condBr, 1});
      blockToCFOps[blockF].insert({condBr, 0});
    }
  });

  if (Operation *definingOp = v.getDefiningOp())
    return getMakeTensorPtrOpImpl(definingOp, v);

  // If there is no defining op, v must be a BlockArgument.
  BlockArgument arg = cast<BlockArgument>(v);
  unsigned argNum = arg.getArgNumber();
  Operation *argOwner = arg.getOwner()->getParentOp();

  if (auto forOp = dyn_cast<scf::ForOp>(argOwner))
    return tt::getMakeTensorPtrOp(
        forOp.getOperand(argNum + forOp.getNumControlOperands() - 1));
  if (auto funcOp = dyn_cast<FunctionOpInterface>(argOwner)) {
    Block *block = arg.getOwner();
    Operation *op;
    int tOrF;
    std::tie(op, tOrF) = blockToCFOps[block][0];
    if (auto br = dyn_cast<cf::BranchOp>(op))
      return tt::getMakeTensorPtrOp(br.getDestOperands()[argNum]);
    if (auto condBr = dyn_cast<cf::CondBranchOp>(op))
      return tt::getMakeTensorPtrOp(
          tOrF ? condBr.getTrueDestOperands()[argNum]
               : condBr.getFalseDestOperands()[argNum]);
    return tt::getMakeTensorPtrOp(argOwner->getOperand(argNum));
  }
  llvm_unreachable("Unable to getMakeTensorPtr()");
}
