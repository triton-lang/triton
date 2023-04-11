#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;

namespace {

struct TestAxisInfoPass
    : public PassWrapper<TestAxisInfoPass, OperationPass<triton::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAxisInfoPass);

  StringRef getArgument() const final { return "test-print-alignment"; }
  StringRef getDescription() const final {
    return "print the result of the alignment analysis pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    auto &os = llvm::errs();
    auto opName = SymbolTable::getSymbolName(operation).getValue().str();
    os << "@" << opName << "\n";

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AxisInfoAnalysis *analysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(operation)))
      return signalPassFailure();
    operation->walk([&](Operation *op) {
      if (op->getNumResults() < 1)
        return;
      for (Value result : op->getResults()) {
        result.print(os);
        os << " => ";
        analysis->getLatticeElement(result)->getValue().print(os);
        os << "\n";
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestAlignmentPass() { PassRegistration<TestAxisInfoPass>(); }
} // namespace test
} // namespace mlir
