#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"

using namespace mlir;

namespace {

std::optional<int> getPeelEpilogueIterations(scf::ForOp forOp) {
  if (!forOp->hasAttr("__test_peel_epilogue_iterations")) {
    return std::nullopt;
  }
  return cast<IntegerAttr>(forOp->getAttr("__test_peel_epilogue_iterations"))
      .getInt();
}

struct TestLoopPeelingPass
    : public PassWrapper<TestLoopPeelingPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLoopPeelingPass);

  StringRef getArgument() const final { return "triton-test-loop-peeling"; }
  StringRef getDescription() const final {
    return "test the loop peeling pass";
  }

  void runOnOperation() override {
    IRRewriter rewriter(getOperation());
    auto wrapInIf = [&](RewriterBase &rewriter, Operation *op, Value cond) {
      Location loc = op->getLoc();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);
      auto ifOp = rewriter.create<scf::IfOp>(loc, op->getResultTypes(), cond,
                                             /*hasElse=*/true);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto thenYieldOp = rewriter.create<scf::YieldOp>(loc, op->getResults());
      rewriter.moveOpBefore(op, thenYieldOp);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      SmallVector<Value> poisonResults;
      for (auto result : op->getResults()) {
        poisonResults.push_back(rewriter.create<arith::ConstantOp>(
            loc, result.getType(), rewriter.getZeroAttr(result.getType())));
      }
      rewriter.create<scf::YieldOp>(loc, poisonResults);
      return ifOp;
    };
    getOperation().walk([&](scf::ForOp forOp) {
      if (getPeelEpilogueIterations(forOp)) {
        mlir::triton::peelLoopEpilogue(forOp, 1, /*processPeeledOp=*/wrapInIf,
                                       /*processLoopBodyOp=*/nullptr);
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestLoopPeelingPass() { PassRegistration<TestLoopPeelingPass>(); }
} // namespace test
} // namespace mlir
