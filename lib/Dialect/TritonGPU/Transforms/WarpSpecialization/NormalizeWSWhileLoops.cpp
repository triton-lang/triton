#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUNORMALIZEWSWHILELOOPS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// Restore the one-to-one state/result signature expected by WS after
// canonicalization removes unused scf.while results.
//   canonicalized: %pid = scf.while (%pid, %active) -> i32
//                    scf.condition(%active) %pid
//   normalized:    %result:2 = scf.while (%pid, %active) -> (i32, i1)
//                    scf.condition(%active) %pid, %active
scf::WhileOp normalizeWhile(scf::WhileOp loop) {
  auto beforeArgs = loop.getBeforeArguments();
  auto condition = loop.getConditionOp();
  if (llvm::equal(condition.getArgs(), beforeArgs))
    return loop;

  SmallVector<unsigned> resultToState;
  for (Value value : condition.getArgs())
    resultToState.push_back(cast<BlockArgument>(value).getArgNumber());

  SmallVector<Type> stateTypes;
  for (BlockArgument arg : beforeArgs)
    stateTypes.push_back(arg.getType());
  SmallVector<Location> argLocs(stateTypes.size(), loop.getLoc());
  OpBuilder builder(loop);
  auto newLoop =
      scf::WhileOp::create(builder, loop.getLoc(), stateTypes, loop.getInits());
  newLoop->setAttrs(loop->getAttrs());
  Block *newBefore =
      builder.createBlock(&newLoop.getBefore(), {}, stateTypes, argLocs);
  Block *newAfter =
      builder.createBlock(&newLoop.getAfter(), {}, stateTypes, argLocs);

  Block &oldBefore = loop.getBefore().front();
  Block &oldAfter = loop.getAfter().front();
  newBefore->getOperations().splice(newBefore->end(),
                                    oldBefore.getOperations());
  newAfter->getOperations().splice(newAfter->end(), oldAfter.getOperations());

  for (auto [oldArg, newArg] :
       llvm::zip(oldBefore.getArguments(), newBefore->getArguments()))
    oldArg.replaceAllUsesWith(newArg);
  for (auto [resultIndex, stateIndex] : llvm::enumerate(resultToState))
    oldAfter.getArgument(resultIndex)
        .replaceAllUsesWith(newAfter->getArgument(stateIndex));

  condition.getArgsMutable().assign(newBefore->getArguments());
  for (auto [resultIndex, stateIndex] : llvm::enumerate(resultToState))
    loop.getResult(resultIndex)
        .replaceAllUsesWith(newLoop.getResult(stateIndex));

  loop.erase();
  return newLoop;
}

class NormalizeWSWhileLoopsPass
    : public impl::TritonGPUNormalizeWSWhileLoopsBase<
          NormalizeWSWhileLoopsPass> {
public:
  void runOnOperation() override {
    getOperation().walk([&](scf::WhileOp whileOp) {
      bool isCLCWhile =
          llvm::any_of(whileOp.getAfter().front(), [](Operation &op) {
            return isa<nvidia_gpu::CLCTryCancelSyncOp>(op);
          });
      if (!isCLCWhile)
        return;

      bool warpSpecialize = false;
      bool disallowAccMultiBuffer = false;
      for (Operation &op : whileOp.getAfter().front()) {
        auto forOp = dyn_cast<scf::ForOp>(op);
        if (!forOp || !forOp->hasAttr(kWarpSpecializeAttrName))
          continue;
        warpSpecialize = true;
        disallowAccMultiBuffer |=
            forOp->hasAttr(kDisallowAccMultiBufferAttrName);
        forOp->removeAttr(kWarpSpecializeAttrName);
        forOp->removeAttr(kDisallowAccMultiBufferAttrName);
      }
      if (!warpSpecialize)
        return;

      whileOp->setAttr(kWarpSpecializeAttrName,
                       UnitAttr::get(whileOp.getContext()));
      if (disallowAccMultiBuffer) {
        whileOp->setAttr(kDisallowAccMultiBufferAttrName,
                         UnitAttr::get(whileOp.getContext()));
      }
    });

    SetVector<scf::WhileOp> whiles;
    getOperation().walk([&](Operation *op) {
      if (!isa<scf::ForOp, scf::WhileOp>(op) ||
          !op->hasAttr(kWarpSpecializeAttrName))
        return;
      if (auto whileOp = dyn_cast<scf::WhileOp>(op))
        whiles.insert(whileOp);
      op->walk([&](scf::WhileOp whileOp) { whiles.insert(whileOp); });
    });
    for (scf::WhileOp whileOp : whiles)
      normalizeWhile(whileOp);
  }
};

} // namespace
} // namespace mlir::triton::gpu
