#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-licm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {

namespace {

class LoopInvariantCodeMotionPass
    : public TritonLoopInvariantCodeMotionBase<LoopInvariantCodeMotionPass> {

  DenseMap<LoopLikeOpInterface, bool> isLoopMemoryEffectFreeOrOnlyRead;

  bool isMemoryEffectFreeOrOnlyRead(Operation *op) {
    std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
        getEffectsRecursively(op);
    if (!effects)
      return false;
    return llvm::all_of(*effects,
                        [&](const MemoryEffects::EffectInstance &effect) {
                          return isa<MemoryEffects::Read>(effect.getEffect());
                        });
  }

  void runOnOperation() override {
    // Walk through all loops in a function in innermost-loop-first order.
    // This way, we first LICM from the inner loop, and place the ops in the
    // outer loop, which in turn can be further LICM'ed.
    getOperation()->walk([&](LoopLikeOpInterface loopLike) {
      moveLoopInvariantCode(
          loopLike.getLoopRegions(),
          // isDefinedOutsideOfRegion
          [&](Value value, Region *region) {
            return loopLike.isDefinedOutsideOfLoop(value);
          },
          // shouldMoveOutOfRegion
          [&](Operation *op, Region *region) {
            if (!isLoopMemoryEffectFreeOrOnlyRead.contains(loopLike))
              isLoopMemoryEffectFreeOrOnlyRead[loopLike] =
                  isMemoryEffectFreeOrOnlyRead(loopLike);
            return isSpeculatable(op) &&
                   isLoopMemoryEffectFreeOrOnlyRead[loopLike] &&
                   ((!isa<LoadOp>(op) && isMemoryEffectFree(op)) ||
                    (isa<LoadOp>(op) && isMemoryEffectFreeOrOnlyRead(op)));
          },
          // moveOutOfRegion
          [&](Operation *op, Region *) {
            // Create the new mask for load op.
            if (auto loadOp = dyn_cast<LoadOp>(op)) {
              Value mask = loadOp.getMask();
              IRRewriter rewriter(loopLike);
              Location loc = loopLike->getLoc();
              Value cond;
              if (auto forOp = dyn_cast<scf::ForOp>(loopLike.getOperation())) {
                cond = rewriter.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::slt, forOp.getLowerBound(),
                    forOp.getUpperBound());
              } else if (auto whileOp =
                             dyn_cast<scf::WhileOp>(loopLike.getOperation())) {
                IRMapping mapper;
                Block *beforeBlock = whileOp.getBeforeBody();
                // Clone before block before the loop for zero-trip-check.
                for (auto [arg, init] : llvm::zip_equal(
                         beforeBlock->getArguments(), whileOp.getInits()))
                  mapper.map(arg, init);
                for (auto &op : *beforeBlock) {
                  if (isa<scf::ConditionOp>(op))
                    break;
                  // Safe to clone everything as in a single block all defs have
                  // been cloned and added to mapper in order.
                  rewriter.insert(op.clone(mapper));
                }
                scf::ConditionOp condOp = whileOp.getConditionOp();
                Value clonedCondition =
                    mapper.lookupOrDefault(condOp.getCondition());
                cond = clonedCondition;
              } else {
                loopLike->emitError("Unrecognized loop-like op");
                llvm::report_fatal_error("Fatal LICM error");
              }
              Value newMask = getPredMask(rewriter, loadOp.getPtr().getType(),
                                          loadOp.getMask(), cond);
              loadOp.getMaskMutable().assign(newMask);
            }
            loopLike.moveOutOfLoop(op);
          });
    });
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createLoopInvariantCodeMotionPass() {
  return std::make_unique<LoopInvariantCodeMotionPass>();
}

} // namespace mlir::triton
