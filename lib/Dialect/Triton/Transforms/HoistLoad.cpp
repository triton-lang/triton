#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-hoist-load"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {

namespace {

class HoistLoadPass : public TritonHoistLoadBase<HoistLoadPass> {

  DenseMap<scf::ForOp, bool> visitedForOps;

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
    getOperation()->walk([&](scf::ForOp forOp) {
      moveLoopInvariantCode(
          forOp.getLoopRegions(),
          // isDefinedOutsideOfRegion
          [&](Value value, Region *region) {
            return forOp.isDefinedOutsideOfLoop(value);
          },
          // shouldMoveOutOfRegion
          [&](Operation *op, Region *region) {
            if (!isa<LoadOp>(op))
              return false;
            if (!visitedForOps.contains(forOp))
              visitedForOps[forOp] = isMemoryEffectFreeOrOnlyRead(forOp);
            return visitedForOps[forOp];
          },
          // moveOutOfRegion
          [&](Operation *op, Region *) {
            LoadOp loadOp = cast<LoadOp>(op);
            Value mask = loadOp.getMask();
            IRRewriter rewriter(forOp);
            Location loc = forOp->getLoc();
            arith::CmpIOp cmpIOp = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, forOp.getLowerBound(),
                forOp.getUpperBound());
            Value newMask;
            if (mask) {
              Value zeroMask = rewriter.create<arith::ConstantOp>(
                  loc, rewriter.getZeroAttr(mask.getType()));
              newMask =
                  rewriter.create<arith::SelectOp>(loc, cmpIOp, mask, zeroMask);
            } else {
              auto loadType = dyn_cast<RankedTensorType>(loadOp.getType());
              if (!loadType)
                return;
              newMask = rewriter.create<SplatOp>(
                  loc,
                  RankedTensorType::get(loadType.getShape(),
                                        rewriter.getI1Type()),
                  cmpIOp);
            }
            LoadOp newLoadOp = rewriter.create<LoadOp>(
                loc, loadOp.getPtr(), newMask, loadOp.getOther(),
                loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
            rewriter.replaceAllUsesWith(loadOp, newLoadOp);
          });
    });
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createHoistLoadPass() {
  return std::make_unique<HoistLoadPass>();
}

} // namespace mlir::triton
