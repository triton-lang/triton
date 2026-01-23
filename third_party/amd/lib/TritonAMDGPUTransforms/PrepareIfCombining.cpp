//===----------------------------------------------------------------------===//
// This pass enables combining of scf.if operations that share the same
// condition by moving intervening operations out of the way.
//
// Example:
//   %0 = ttg.local_load %smem : ...
//   %1 = scf.if %cond -> tensor<...> { ... }
//   %2 = tt.trans %0 : ...  // <-- blocking combining if ops
//   %3 = scf.if %cond -> tensor<...> { ... }
//
// After this pass, %2 is moved before %1, making %1 and %3 adjacent so that
// they can be combined by the canonicalizer.
//===----------------------------------------------------------------------===//
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUPREPAREIFCOMBINING
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

static void handleIfPair(scf::IfOp currentIf, scf::IfOp nextIf,
                         DominanceInfo &domInfo) {
  // Skip if they are not in the same block.
  if (currentIf->getBlock() != nextIf->getBlock())
    return;
  // Skip if they have different conditions.
  if (currentIf.getCondition() != nextIf.getCondition())
    return;

  // Collect ops between the two ifs and check if they are all
  // pure (speculatable, does not touch memory) ops.
  SetVector<Operation *> opsBetween;
  for (Operation *op = currentIf->getNextNode(); op != nextIf.getOperation();
       op = op->getNextNode()) {
    if (!isPure(op))
      return;
    opsBetween.insert(op);
  }
  if (opsBetween.empty())
    return;

  // Check that all operands are either in opsBetween or dominate currentIf.
  for (Operation *op : opsBetween) {
    for (Value operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || opsBetween.contains(defOp))
        continue;
      if (!domInfo.properlyDominates(defOp, currentIf))
        return;
    }
  }

  // Move all ops before the current if.
  for (Operation *op : opsBetween)
    op->moveBefore(currentIf);
}

} // namespace

struct TritonAMDGPUPrepareIfCombiningPass
    : public impl::TritonAMDGPUPrepareIfCombiningBase<
          TritonAMDGPUPrepareIfCombiningPass> {

  void runOnOperation() override {
    triton::FuncOp funcOp = getOperation();
    DominanceInfo domInfo(funcOp);
    SmallVector<scf::IfOp> ifOps;
    funcOp.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });
    for (auto [currentIf, nextIf] : llvm::zip(ifOps, llvm::drop_begin(ifOps))) {
      handleIfPair(currentIf, nextIf, domInfo);
    }
  }
};

} // namespace mlir
