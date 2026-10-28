#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPUNROLL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static Value ceilDivNonNegative(OpBuilder &builder, Location loc,
                                Value dividend, Value divisor) {
  Value cstOne =
      arith::ConstantOp::create(builder, loc,
                                builder.getIntegerAttr(dividend.getType(), 1));
  Value divisorMinusOne =
      arith::SubIOp::create(builder, loc, divisor, cstOne);
  Value sum = arith::AddIOp::create(builder, loc, dividend, divisorMinusOne);
  return arith::DivSIOp::create(builder, loc, sum, divisor);
}

// Unrolls a loop whose trip count cannot be determined statically.  Unlike
// `mlir::loopUnrollByFactor`, this computes the trip count with signed
// division so an empty loop (upper bound <= lower bound for a positive step)
// is not turned into a nonempty epilogue.
static FailureOr<UnrolledLoopInfo>
unrollDynamicLoop(scf::ForOp forOp, uint64_t unrollFactor) {
  assert(unrollFactor > 0 && "expected positive unroll factor");

  // Nothing to unroll if the body only contains the terminator.
  if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
    return UnrolledLoopInfo{forOp, std::nullopt};

  OpBuilder boundsBuilder(forOp);
  auto loc = forOp.getLoc();
  Value step = forOp.getStep();
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();

  Value diff =
      arith::SubIOp::create(boundsBuilder, loc, upperBound, lowerBound);
  Value zero = arith::ConstantOp::create(
      boundsBuilder, loc, boundsBuilder.getIntegerAttr(diff.getType(), 0));
  Value nonNegativeDiff =
      arith::MaxSIOp::create(boundsBuilder, loc, diff, zero);
  Value tripCount =
      ceilDivNonNegative(boundsBuilder, loc, nonNegativeDiff, step);
  Value unrollFactorCst = arith::ConstantOp::create(
      boundsBuilder, loc,
      boundsBuilder.getIntegerAttr(tripCount.getType(), unrollFactor));
  Value tripCountRem =
      arith::RemSIOp::create(boundsBuilder, loc, tripCount, unrollFactorCst);
  Value tripCountEvenMultiple =
      arith::SubIOp::create(boundsBuilder, loc, tripCount, tripCountRem);
  Value upperBoundUnrolled = arith::AddIOp::create(
      boundsBuilder, loc, lowerBound,
      arith::MulIOp::create(boundsBuilder, loc, tripCountEvenMultiple, step));
  Value stepUnrolled =
      arith::MulIOp::create(boundsBuilder, loc, step, unrollFactorCst);

  OpBuilder epilogueBuilder(forOp->getContext());
  epilogueBuilder.setInsertionPoint(
      forOp->getBlock(), std::next(Block::iterator(forOp)));
  auto epilogueForOp = cast<scf::ForOp>(epilogueBuilder.clone(*forOp));
  epilogueForOp.setLowerBound(upperBoundUnrolled);

  auto results = forOp.getResults();
  auto epilogueResults = epilogueForOp.getResults();
  for (auto [idx, e] : llvm::enumerate(llvm::zip(results, epilogueResults))) {
    Value result = std::get<0>(e);
    Value epilogueResult = std::get<1>(e);
    result.replaceAllUsesWith(epilogueResult);
    epilogueForOp->replaceUsesOfWith(epilogueForOp->getOperand(3 + idx), result);
  }

  forOp.setUpperBound(upperBoundUnrolled);
  forOp.setStep(stepUnrolled);

  auto iterArgs = ValueRange(forOp.getRegionIterArgs());
  auto yieldedValues = forOp.getBody()->getTerminator()->getOperands();
  generateUnrolledLoop(
      forOp.getBody(), forOp.getInductionVar(), unrollFactor,
      [&](unsigned i, Value iv, OpBuilder b) {
        auto stride = arith::MulIOp::create(
            b, loc, step,
            arith::ConstantOp::create(b, loc,
                                      b.getIntegerAttr(iv.getType(), i)));
        return arith::AddIOp::create(b, loc, iv, stride);
      },
      nullptr, iterArgs, yieldedValues);

  return UnrolledLoopInfo{forOp, epilogueForOp};
}

class LoopUnrollPass : public impl::TritonLoopUnrollBase<LoopUnrollPass> {

  int getUnrollFactorOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise set the
    // factor to 1 to suppress the unrolling.
    if (auto factor =
            forOp->getAttrOfType<IntegerAttr>(loopUnrollFactorAttrName))
      return factor.getInt();
    return 1;
  }

  const char *loopUnrollFactorAttrName = "tt.loop_unroll_factor";
  const char *pipelineStagesAttrName = "tt.num_stages";

public:
  void runOnOperation() override {
    LDBG("Loop unroll pass");
    SmallVector<scf::ForOp, 4> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with unroll factor <= 1.
      if (getUnrollFactorOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    auto ctx = getOperation()->getContext();
    for (auto loop : loops) {
      auto unrollFactor = getUnrollFactorOrDefault(loop);
      loop->removeAttr(loopUnrollFactorAttrName);
      LDBG("Unrolling loop by " << unrollFactor << " times\n" << loop);
      std::optional<APInt> staticTripCount = loop.getStaticTripCount();
      if (staticTripCount && staticTripCount->isZero()) {
        // The loop is statically empty; unrolling it would introduce a bogus
        // epilogue that executes iterations of the empty loop.
        LDBG("Skipping unroll of an empty loop");
        continue;
      }
      FailureOr<UnrolledLoopInfo> resultLoops =
          staticTripCount
              ? loopUnrollByFactor(loop, unrollFactor)
              : unrollDynamicLoop(loop, unrollFactor);
      // Do not pipeline the epilog loop.
      if (succeeded(resultLoops) && resultLoops->epilogueLoopOp) {
        (*resultLoops->epilogueLoopOp)
            ->setAttr(pipelineStagesAttrName,
                      mlir::IntegerAttr::get(IntegerType::get(ctx, 32), 1));
      }
    }
  }
};

} // namespace mlir::triton
