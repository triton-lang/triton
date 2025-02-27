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

  // This function checks if the root operation consists of ops with only read
  // side-effects or with write side-effects but are only PrintOp or
  // AssertOp.
  bool isOnlyReadPrintAssert(Operation *rootOp) {
    SmallVector<Operation *> effectingOps(1, rootOp);
    while (!effectingOps.empty()) {
      Operation *op = effectingOps.pop_back_val();

      // If the operation has recursive effects, push all of the nested
      // operations on to the stack to consider.
      bool hasRecursiveEffects =
          op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
      if (hasRecursiveEffects) {
        for (Region &region : op->getRegions()) {
          for (Block &block : region) {
            for (Operation &nestedOp : block) {
              effectingOps.push_back(&nestedOp);
            }
          }
        }
      }
      SmallVector<MemoryEffects::EffectInstance> effects;
      if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
        effectInterface.getEffects(effects);
      } else if (!hasRecursiveEffects) {
        // the operation does not have recursive memory effects or implement
        // the memory effect op interface. Its effects are unknown.
        return false;
      }
      bool allReadPrintOrAssert = llvm::all_of(
          effects, [&](const MemoryEffects::EffectInstance &effect) {
            return isa<MemoryEffects::Read>(effect.getEffect()) ||
                   (isa<MemoryEffects::Write>(effect.getEffect()) &&
                    isa<PrintOp, AssertOp>(op));
          });
      if (!allReadPrintOrAssert)
        return false;
    }
    return true;
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
            return isa<LoadOp>(op) &&
                   isOnlyReadPrintAssert(region->getParentOp());
          },
          // moveOutOfRegion
          [&](Operation *op, Region *) {
            LoadOp loadOp = cast<LoadOp>(op);
            scf::ForOp scfForOp =
                dyn_cast<scf::ForOp>(loadOp->getParentRegion()->getParentOp());
            if (!scfForOp)
              return;
            Value mask = loadOp.getMask();
            MLIRContext *ctx = loadOp->getContext();
            IRRewriter rewriter(ctx);
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(scfForOp.getOperation());
            Location loc = scfForOp->getLoc();
            arith::CmpIOp cmpIOp = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, scfForOp.getLowerBound(),
                scfForOp.getUpperBound());
            LoadOp newLoadOp;
            if (mask) {
              Value zeroMask = rewriter.create<arith::ConstantOp>(
                  loc, rewriter.getZeroAttr(mask.getType()));
              arith::SelectOp selectOp =
                  rewriter.create<arith::SelectOp>(loc, cmpIOp, mask, zeroMask);
              newLoadOp = rewriter.create<LoadOp>(
                  loc, loadOp.getPtr(), selectOp, loadOp.getOther(),
                  loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
            } else {
              auto loadType = dyn_cast<RankedTensorType>(loadOp.getType());
              if (!loadType)
                return;
              SplatOp splatOp = rewriter.create<SplatOp>(
                  loc,
                  RankedTensorType::get(loadType.getShape(),
                                        rewriter.getI1Type()),
                  cmpIOp);
              newLoadOp = rewriter.create<LoadOp>(
                  loc, loadOp.getPtr(), splatOp, loadOp.getOther(),
                  loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
            }
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
