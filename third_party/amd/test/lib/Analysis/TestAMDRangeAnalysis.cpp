#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

void collectRanges(DataFlowSolver &solver, ValueRange values,
                   SmallVectorImpl<ConstantIntRanges> &ranges) {
  for (Value val : values) {
    auto *maybeInferredRange =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(val);
    if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
      return;
    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();
    ranges.push_back(inferredRange);
  }
}

struct TestAMDRangeAnalysisPass
    : PassWrapper<TestAMDRangeAnalysisPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAMDRangeAnalysisPass)

  StringRef getArgument() const final {
    return "test-tritonamdgpu-range-analysis";
  }
  StringRef getDescription() const final {
    return "print the result of the tritonamdgpu-range-analysis pass";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();

    // Collect assumptions in the function
    DenseSet<Value> assumptions;
    mod.walk([&](LLVM::AssumeOp op) {
      if (op.getCond().getDefiningOp<arith::CmpIOp>())
        assumptions.insert(op.getCond());
    });

    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<AMD::TritonIntegerRangeAnalysis>();
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    auto nonNegativePred = [&solver](Value v) -> bool {
      return succeeded(AMD::staticallyNonNegative(*solver, v));
    };
    mod->walk<WalkOrder::PreOrder>([&solver, nonNegativePred](Operation *op) {
      SmallVector<ConstantIntRanges> outputRanges;
      auto results = op->getResults();
      collectRanges(*solver, results, outputRanges);
      if (!outputRanges.empty()) {
        for (const auto &[res, outR] : llvm::zip(results, outputRanges)) {
          std::string rangeS;
          llvm::raw_string_ostream rangeSt(rangeS);
          rangeSt << outR;
          emitRemark(res.getLoc(), rangeS);
        }
      }

      if (!op->getResults().empty() &&
          llvm::all_of(op->getResults(), nonNegativePred)) {
        emitRemark(op->getLoc(), "non-neg");
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestTritonAMDGPURangeAnalysis() {
  PassRegistration<TestAMDRangeAnalysisPass>();
}
} // namespace test
} // namespace mlir
