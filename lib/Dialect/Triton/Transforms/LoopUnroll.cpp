#include <memory>

#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
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

namespace {

class LoopUnrollPass : public TritonLoopUnrollBase<LoopUnrollPass> {

  int getUnrollFactorOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise set the
    // factor to 1 to suppress the unrolling.
    if (auto factor =
            forOp->getAttrOfType<IntegerAttr>(loopUnrollFactorAttrName))
      return factor.getInt();
    return 1;
  }

  int getUnrollIdOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise set the
    // factor to 1 to suppress the unrolling.
    if (auto factor = forOp->getAttrOfType<IntegerAttr>(unrolledLoopIdAttrName))
      return factor.getInt();
    return 0;
  }

  const char *loopUnrollFactorAttrName = "tt.loop_unroll_factor";
  const char *unrolledLoopIdAttrName = "tt.unrolled_loop_id";
  const char *pipelineStagesAttrName = "tt.num_stages";

public:
  LoopUnrollPass() = default;
  LoopUnrollPass(const LoopUnrollPass &) {}

  SmallVector<scf::ForOp, 2> getUnrolledLoopsAndClearAttrs(unsigned loopId) {
    SmallVector<scf::ForOp, 2> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      if (getUnrollIdOrDefault(forOp) == loopId)
        loops.push_back(forOp);
    });

    assert(loops.size() <= 2 && "Expect at most 2 loops, one for the main loop "
                                "and one for the prolog/epilog");
    SmallVector<int, 2> loopInstructionCounts;
    for (auto loop : loops) {
      loop->removeAttr(loopUnrollFactorAttrName);
      loop->removeAttr(unrolledLoopIdAttrName);
      int count = 0;
      loop->walk([&](Operation *op) { count++; });
      loopInstructionCounts.push_back(count);
    }
    if (loops.size() == 2) {
      // check which one is the unrolled loop and which one is the prolog/epilog
      // loop. A simple heuristic is to check the number of instructions in the
      // loop. The unrolled main loop should have the most instructions.
      // sort the loops by the number of instructions. The unrolled main loop
      // should go first.
      if (loopInstructionCounts[0] < loopInstructionCounts[1])
        std::swap(loops[0], loops[1]);
    }

    return loops;
  }

  void runOnOperation() override {
    LDBG("Loop unroll pass");
    SmallVector<scf::ForOp, 4> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with unroll factor <= 1.
      if (getUnrollFactorOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    auto ctx = getOperation()->getContext();
    for (unsigned i = 0; i < loops.size(); i++) {
      auto loop = loops[i];
      auto unrollFactor = getUnrollFactorOrDefault(loop);
      loop->setAttr(unrolledLoopIdAttrName,
                    mlir::IntegerAttr::get(IntegerType::get(ctx, 32), i + 1));
      LDBG("Unrolling loop by " << unrollFactor << " times\n" << loop);
      (void)loopUnrollByFactor(loop, unrollFactor);
      auto unrolledLoops = getUnrolledLoopsAndClearAttrs(i + 1);
      // Do not pipeline the prolog/epilog loop.
      if (unrolledLoops.size() == 2) {
        auto prologEpilogLoop = unrolledLoops[1];
        prologEpilogLoop->setAttr(
            pipelineStagesAttrName,
            mlir::IntegerAttr::get(IntegerType::get(ctx, 32), 1));
      }
    }
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createLoopUnrollPass() {
  return std::make_unique<LoopUnrollPass>();
}

} // namespace mlir::triton
