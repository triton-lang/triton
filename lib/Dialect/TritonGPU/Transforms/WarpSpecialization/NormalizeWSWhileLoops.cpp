#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUNORMALIZEWSWHILELOOPS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// Normalize `scf.while` to the form expected by WS, where the values forwarded
// by `scf.condition` are the loop's before-region block arguments. Rotate the
// loop by evaluating the original before region once before the loop and again
// after each execution of the original body.
//   canonicalized:
//     %result = scf.while (%state = %initial) -> i32 {
//       %value = compute_result(%state)
//       %cond = compute_cond(%state)
//       scf.condition(%cond) %value
//     } do {
//     ^bb0(%value: i32):
//       %next_state = body(%value)
//       scf.yield %next_state
//     }
//
//   normalized:
//     %initial_value = compute_result(%initial)
//     %initial_cond = compute_cond(%initial)
//     %result:2 = scf.while (%value = %initial_value,
//                            %cond = %initial_cond) -> (i32, i1) {
//       scf.condition(%cond) %value, %cond
//     } do {
//     ^bb0(%value: i32, %cond: i1):
//       %next_state = body(%value)
//       %next_value = compute_result(%next_state)
//       %next_cond = compute_cond(%next_state)
//       scf.yield %next_value, %next_cond
//     }
SmallVector<Value> cloneBlockBody(OpBuilder &builder, Block &block,
                                  ValueRange arguments) {
  assert(block.getNumArguments() == arguments.size());
  IRMapping mapping;
  mapping.map(block.getArguments(), arguments);
  for (Operation &op : block.without_terminator())
    builder.clone(op, mapping);

  SmallVector<Value> terminatorOperands;
  for (Value operand : block.getTerminator()->getOperands())
    terminatorOperands.push_back(mapping.lookupOrDefault(operand));
  return terminatorOperands;
}

scf::WhileOp normalizeWhile(scf::WhileOp loop) {
  auto beforeArgs = loop.getBeforeArguments();
  auto condition = loop.getConditionOp();
  if (llvm::equal(condition.getArgs(), beforeArgs))
    return loop;

  auto &oldBefore = loop.getBefore().front();
  auto &oldAfter = loop.getAfter().front();
  auto oldYield = loop.getYieldOp();
  auto oldYieldLoc = oldYield.getLoc();
  SmallVector<NamedAttribute> oldYieldAttrs(oldYield->getAttrs());
  OpBuilder builder(loop);

  // Evaluate the old before region once to seed the normalized loop. The
  // condition is carried as the last state value.
  auto initial = cloneBlockBody(builder, oldBefore, loop.getInits());
  auto initialCondition = initial.front();
  SmallVector<Value> inits(initial.begin() + 1, initial.end());
  inits.push_back(initialCondition);

  SmallVector<Type> stateTypes;
  SmallVector<Location> argLocs;
  for (Value init : inits) {
    stateTypes.push_back(init.getType());
    argLocs.push_back(init.getLoc());
  }
  auto newLoop =
      scf::WhileOp::create(builder, loop.getLoc(), stateTypes, inits);
  newLoop->setAttrs(loop->getAttrs());
  auto newBefore =
      builder.createBlock(&newLoop.getBefore(), {}, stateTypes, argLocs);
  auto newAfter =
      builder.createBlock(&newLoop.getAfter(), {}, stateTypes, argLocs);

  // The normalized before region only forwards its block arguments.
  builder.setInsertionPointToEnd(newBefore);
  auto newCondition = scf::ConditionOp::create(builder, condition.getLoc(),
                                               newBefore->getArguments().back(),
                                               newBefore->getArguments());
  newCondition->setAttrs(condition->getAttrs());

  // Execute the old body, then evaluate the old before region to produce the
  // state and condition for the next iteration.
  for (auto [oldArg, newArg] : llvm::zip(
           oldAfter.getArguments(), newAfter->getArguments().drop_back())) {
    oldArg.replaceAllUsesWith(newArg);
  }
  SmallVector<Value> nextBeforeArgs(oldYield.getOperands());
  oldYield.erase();
  newAfter->getOperations().splice(newAfter->end(), oldAfter.getOperations());

  builder.setInsertionPointToEnd(newAfter);
  auto next = cloneBlockBody(builder, oldBefore, nextBeforeArgs);
  auto nextCondition = next.front();
  SmallVector<Value> nextState(next.begin() + 1, next.end());
  nextState.push_back(nextCondition);
  auto newYield = scf::YieldOp::create(builder, oldYieldLoc, nextState);
  newYield->setAttrs(oldYieldAttrs);

  for (auto [oldResult, newResult] :
       llvm::zip(loop.getResults(), newLoop.getResults())) {
    oldResult.replaceAllUsesWith(newResult);
  }

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
    // Normalize inner loops first so cloning a before region only duplicates
    // already-normalized nested loops.
    for (scf::WhileOp whileOp : llvm::reverse(whiles))
      normalizeWhile(whileOp);
  }
};

} // namespace
} // namespace mlir::triton::gpu
