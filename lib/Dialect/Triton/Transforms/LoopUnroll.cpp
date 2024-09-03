#include <memory>

#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {

static const char *loopUnrollFactorAttrName = "tt.unrolled_iteration";

namespace {

class LoopUnrollPass : public TritonLoopUnrollBase<LoopUnrollPass> {

  int getUnrollFactorOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(mlir::triton::loopUnrollFactorAttrName))
      return unrollFactor;
    return mlir::cast<IntegerAttr>(
               forOp->getAttr(mlir::triton::loopUnrollFactorAttrName))
        .getInt();
  }

public:
  LoopUnrollPass() = default;
  LoopUnrollPass(const LoopUnrollPass &) {}
  void runOnOperation() override {
    LDBG("Loop unroll pass");
    SmallVector<scf::ForOp, 4> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with unroll factor <= 1.
      if (getUnrollFactorOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    for (auto loop : loops) {
      auto unrollFactor = getUnrollFactorOrDefault(loop);
      loop->removeAttr(mlir::triton::loopUnrollFactorAttrName);
      LDBG("Unrolling loop by " << unrollFactor << " times\n" << loop);
      (void)loopUnrollByFactor(loop, unrollFactor);
    }
  }

  Option<uint64_t> unrollFactor{*this, "unroll-factor",
                                llvm::cl::desc("Default loop unroll factor."),
                                llvm::cl::init(1)};
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createLoopUnrollPass() {
  return std::make_unique<LoopUnrollPass>();
}

} // namespace mlir::triton
