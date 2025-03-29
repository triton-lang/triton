#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct TestAMDFoldTrueCmpIOpPass
    : PassWrapper<TestAMDFoldTrueCmpIOpPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAMDFoldTrueCmpIOpPass)

  StringRef getArgument() const final {
    return "test-tritonamdgpu-fold-true-cmpi";
  }
  StringRef getDescription() const final {
    return "print the result of the tritonamdgpu-fold-true-cmpi pass";
  }

  void runOnOperation() override {
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<AMD::TritonIntegerRangeAnalysis>(assumptions);
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    ModuleOp mod = getOperation();
    RewritePatternSet patterns(&getContext());
    AMD::populateFoldTrueCmpIOpPatterns(patterns, solver);
    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::test {
void registerTestTritonAMDGPUFoldTrueCmpIOp() {
  PassRegistration<TestAMDFoldTrueCmpIOpPass>();
}
} // namespace mlir::test
