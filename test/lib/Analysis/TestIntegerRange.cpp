#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestIntegerRangePass
    : PassWrapper<TestIntegerRangePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIntegerRangePass)

  StringRef getArgument() const final { return "test-print-integer-range"; }
  StringRef getDescription() const final {
    return "print ranges inferred by upstream IntegerRangeAnalysis";
  }

  void runOnOperation() override {
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(getOperation())))
      return signalPassFailure();

    getOperation().walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        const auto *lattice =
            solver.lookupState<dataflow::IntegerValueRangeLattice>(result);
        if (lattice && !lattice->getValue().isUninitialized()) {
          std::string range;
          llvm::raw_string_ostream stream(range);
          stream << lattice->getValue().getValue();
          emitRemark(result.getLoc(), range);
        }
      }
    });
  }
};

} // namespace

namespace mlir::test {
void registerTestIntegerRangePass() {
  PassRegistration<TestIntegerRangePass>();
}
} // namespace mlir::test
