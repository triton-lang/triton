#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "triton/Analysis/Utility.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct TritonAMDFoldTrueCmpIOpPass
    : TritonAMDFoldTrueCmpIBase<TritonAMDFoldTrueCmpIOpPass> {

  void runOnOperation() override {
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    ModuleOp mod = getOperation();
    std::unique_ptr solver = createDataFlowSolver();
    AMD::TritonIntegerRangeAnalysis *rangeAnalysis =
        solver->load<AMD::TritonIntegerRangeAnalysis>(assumptions);
    AMD::initializeFuncOps(mod, rangeAnalysis);
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    RewritePatternSet patterns(&getContext());
    AMD::populateFoldTrueCmpIOpPatterns(patterns, solver.get());
    (void)applyPatternsGreedily(mod, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUFoldTrueCmpIPass() {
  return std::make_unique<TritonAMDFoldTrueCmpIOpPass>();
}
