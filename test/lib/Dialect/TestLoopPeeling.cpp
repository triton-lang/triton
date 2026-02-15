#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"

using namespace mlir;

namespace {

bool getPeelEpilogue(scf::ForOp forOp) {
  return forOp->hasAttr("__test_peel_epilogue");
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
    getOperation().walk([&](scf::ForOp forOp) {
      if (getPeelEpilogue(forOp)) {
        mlir::triton::peelLoopEpilogue(forOp);
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
