#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "triton/Analysis/Utility.h"

using namespace mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDFOLDTRUECMPI
#include "TritonAMDGPUTransforms/Passes.h.inc"

struct TritonAMDFoldTrueCmpIOpPass
    : impl::TritonAMDFoldTrueCmpIBase<TritonAMDFoldTrueCmpIOpPass> {

  void runOnOperation() override {
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    ModuleOp mod = getOperation();
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AMD::TritonIntegerRangeAnalysis *rangeAnalysis =
        solver->load<AMD::TritonIntegerRangeAnalysis>(
            assumptions, &getAnalysis<DominanceInfo>());
    AMD::initializeFuncOps(mod, rangeAnalysis);
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    RewritePatternSet patterns(&getContext());
    AMD::populateFoldTrueCmpIOpPatterns(patterns, solver.get());
    (void)applyPatternsGreedily(mod, std::move(patterns));
  }
};

} // namespace mlir
